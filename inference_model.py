import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import time
from datetime import datetime, timedelta
import catboost
import pickle
from sklearn.metrics import mean_absolute_error
from sklearn.base import BaseEstimator, TransformerMixin
import calendar
from datetime import date
import warnings
warnings.filterwarnings("ignore")

# Creating sales lag features
def create_sales_lag_feats(df, gpby_cols, target_col, lags):
    gpby = df.groupby(gpby_cols)
    for i in lags:
        df['_'.join([target_col, 'lag', str(i)])] = \
                gpby[target_col].shift(i).values + np.random.normal(scale=1, size=(len(df),)) * 0
    return df

# Creating sales rolling mean features
def create_sales_rmean_feats(df, gpby_cols, target_col, windows, min_periods=2, 
                             shift=1, win_type=None):
    gpby = df.groupby(gpby_cols)
    for w in windows:
        df['_'.join([target_col, 'rmean', str(w)])] = \
            gpby[target_col].shift(shift).rolling(window=w, 
                                                  min_periods=min_periods,
                                                  win_type=win_type).mean().values +\
            np.random.normal(scale=1, size=(len(df),)) * 0
    return df

# Creating sales rolling median features
def create_sales_rmed_feats(df, gpby_cols, target_col, windows, min_periods=2, 
                            shift=1, win_type=None):
    gpby = df.groupby(gpby_cols)
    for w in windows:
        df['_'.join([target_col, 'rmed', str(w)])] = \
            gpby[target_col].shift(shift).rolling(window=w, 
                                                  min_periods=min_periods,
                                                  win_type=win_type).median().values +\
            np.random.normal(scale=1, size=(len(df),)) * 0
    return df

# Creating sales exponentially weighted mean features
def create_sales_ewm_feats(df, gpby_cols, target_col, alpha=[0.9], shift=[1]):
    gpby = df.groupby(gpby_cols)
    for a in alpha:
        for s in shift:
            df['_'.join([target_col, 'lag', str(s), 'ewm', str(a)])] = \
                gpby[target_col].shift(s).ewm(alpha=a).mean().values
    return df



def add_months(sourcedate, months):
    month = sourcedate.month - 1 + months
    year = sourcedate.year + month // 12
    month = month % 12 + 1
    day = min(sourcedate.day, calendar.monthrange(year,month)[1])
    return date(year, month, day)


class Preprocesser(BaseEstimator, TransformerMixin):
    def __init__(self, products_info_path, store_info_path):
        self.hash_to_numbers = {}
        self.numbers_to_hash = {}
        self.drop_cols = ['date', 'sales', 'year', 'product_name', 'month']
        self.products_info_path = products_info_path
        self.store_info_path = store_info_path
        self.cat_cols = ['store', 'item'] + \
                           ['inn', 'product_short_name', 'tnved', 'tnved10', 'brand', 
                            'country', 'region_code', 'city_with_type', 'city_fias_id', 'postal_code']
        self.mean_values = {}
        self.min_date = '2021-11-01'
        
    def fit(self, data):
        data = data.copy()
        data = data.rename(columns={'dt': 'date', 'gtin': 'item', 'id_sp_': 'store', 'cnt': 'sales'})
        data = data.drop(columns='inn', axis=1)
        data = data.dropna(subset='store')
        for col in ['item', 'store', 'prid']:
            self.hash_to_numbers[col] = {a: b for a, b in zip(np.unique(data[col]), np.arange(data[col].nunique()))}
            self.numbers_to_hash[col] = {b: a for a, b in zip(np.unique(data[col]), np.arange(data[col].nunique()))}
        
        return self
    
    def get_df(self, data):
        data = data.sort_values(by='dt').reset_index(drop=True)
        data = data.rename(columns={'dt': 'date', 'gtin': 'item', 'id_sp_': 'store', 'cnt': 'sales'})
        data = data.drop(columns='inn', axis=1)
        data = data.dropna(subset='store')
        
        for col in ['item', 'store', 'prid']:
            data.loc[:, col] = data[col].apply(lambda x: self.hash_to_numbers[col][x] if x in self.hash_to_numbers[col] else np.nan)
            data.loc[:, col] = data[col].astype('int')
        
        data['date'] = data['date'].apply(lambda x: x[:-2] + '01')
        df = data.groupby(['store', 'item', 'date']).agg(sales = ('sales', 'sum'),
                                                           price = ('price', 'mean')).reset_index()
        return df
        
    def add_zero_points(self, df, pred_date=None):
        if pred_date is None:
            pred_date = df['date'].max()
            
        i = 0
        all_dates = []
        while True:
            cur_month = add_months(datetime.fromisoformat(self.min_date), i).isoformat() 
            all_dates.append(cur_month)
            if cur_month == pred_date:
                break
            i += 1
            
        x = df.groupby(['store', 'item'])['date'].unique()
        add = []
        for store_item, now_dates in tqdm(list(x.items())):
            for d in all_dates:
                if d not in now_dates:
                    add.append({'store': store_item[0],
                                'item': store_item[1],
                                'date': d,
                                'sales': 0})

        df = pd.concat([df, pd.DataFrame(add)])
        return df
    
    def fill_price(self, prices):
        L = np.ones(len(prices)) * -1
        R = np.ones(len(prices)) * -1
        for i in range(len(prices)):
            if prices[i] == prices[i]: #not is nan
                L[i] = prices[i]
            elif i > 0:
                L[i] = L[i - 1]

        for i in range(len(prices) - 1, -1, -1):
            if prices[i] == prices[i]: #not is nan
                R[i] = prices[i]
            elif i != len(prices) - 1:
                R[i] = R[i + 1]

        for i in range(len(prices)):
            if prices[i] != prices[i]:
                if L[i] == -1:
                    prices[i] = R[i]
                elif R[i] == -1:
                    prices[i] = L[i]
                else:
                    prices[i] = (L[i] + R[i]) / 2
        return prices
    
    def fix_prices(self, df):
        groups = df.sort_values(by='date').groupby(['store', 'item'])['price']

        res = []
        for group in tqdm(groups):
            res += self.fill_price(group[1].values).tolist()
        df.sort_values(by=['store','item', 'date'], axis=0, inplace=True)
        df['price'] = res
        return df
    
    def get_product_info(self, path):
        products_info = pd.read_csv(path)

        products_info['item'] = products_info['gtin'].apply(lambda x: self.hash_to_numbers['item'][x] if x in 
                                                                    self.hash_to_numbers['item'] else np.nan)
        products_info = products_info.dropna(subset='item')
        products_info = products_info.drop(columns='gtin')
        products_info = products_info.drop_duplicates(subset='item', keep='last')

        products_info['volume'] = products_info['volume'].replace('НЕ КЛАССИФИЦИРОВАНО', np.nan)
        products_info['volume'] = products_info['volume'].apply(lambda x: float(x.replace(',', '.').replace(' г', ''))
                                                                            if x == x else np.nan)
        return products_info

    def get_store_info(self, path):
        store_info = pd.read_csv(path).drop(columns='inn')
        store_info['store'] = store_info['id_sp_'].apply(lambda x: self.hash_to_numbers['store'][x] if x in 
                                                              self.hash_to_numbers['store'] else np.nan)
        store_info = store_info.dropna(subset='store')
        store_info = store_info.drop(columns='id_sp_')
        store_info = store_info.drop_duplicates(subset='store', keep='last')
        return store_info
    
    def build_features(self, df):
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df.date.dt.month
        df['year'] = df.date.dt.year
        
        df = df.merge(self.get_store_info(self.store_info_path), on='store', how='left')
        df = df.merge(self.get_product_info(self.products_info_path), on='item', how='left')
        
        df = create_sales_lag_feats(df, gpby_cols=['store','item'], target_col='sales', 
                               lags=[1, 3, 6, 12])

        df = create_sales_rmean_feats(df, gpby_cols=['store','item'], 
                                         target_col='sales', windows=[2, 3, 6, 12], 
                                         min_periods=2, win_type='triang')

        df = create_sales_rmed_feats(df, gpby_cols=['store','item'], 
                                         target_col='sales', windows=[2, 3, 6, 12], 
                                         min_periods=2, win_type=None)

        df = create_sales_ewm_feats(df, gpby_cols=['store','item'], 
                                       target_col='sales', 
                                       alpha=[0.95, 0.9, 0.8, 0.7, 0.6, 0.5], 
                                       shift=[1, 3, 6, 12])
        
        df = create_sales_rmean_feats(df, gpby_cols=['store','item'], 
                                 target_col='price', windows=[2, 3, 6, 12], 
                                 min_periods=2,)
        
        df[self.cat_cols] = df[self.cat_cols].fillna('unknown').astype('str')
        return df
    
    def transform(self, data, pred_date=None):
        df = self.get_df(data)
        df = self.add_zero_points(df, pred_date)
        self.mean_values = df.groupby(['store', 'item'])['sales'].mean()
        df = self.fix_prices(df)        
        df = self.build_features(df)
        return df
    

def full_solver(model, preprocesser, data, pred_date='2022-12-01'):
    cnt = preprocesser.get_df(data).groupby(['store', 'item'])['sales'].count()
    df_test = preprocesser.transform(data, pred_date)
    pred_mask = df_test['date'] == pred_date
    
    X_test, y_test = df_test.drop(columns=preprocesser.drop_cols), df_test['sales']           
    preds = model.predict(X_test)
    
    df_test.loc[:, 'store'] = df_test['store'].astype(int)
    df_test.loc[:, 'item'] = df_test['item'].astype(int)
    mask = df_test.apply(lambda x: cnt[(x['store'], x['item'])], axis=1) < 10
    if mask.sum():
        # print(df_test[mask].apply(lambda x: preprocesser.mean_values[(x['store'], x['item'])], axis=1))
        preds[mask] = df_test[mask].apply(lambda x: preprocesser.mean_values[(x['store'], x['item'])], axis=1)  
    
    preds = np.around(preds)
    print('mae:', mean_absolute_error(y_test[~pred_mask], preds[~pred_mask]))
    # print('smape:', smape(preds[~pred_mask], y_test[~pred_mask]))
    
    res = df_test[['store', 'item', 'date']]
    res.loc[:, 'preds'] = preds
    res.loc[:, 'store'] = res['store'].apply(lambda x: preprocesser.numbers_to_hash['store'][x])
    res.loc[:, 'item'] = res['item'].apply(lambda x: preprocesser.numbers_to_hash['item'][x])
    return res[pred_mask]


if __name__ == '__main__':
    with open('model.pickle', 'rb') as f:
        model = pickle.load(f)
    
    with open('preprocesser.pickle', 'rb') as f:
        preprocesser = pickle.load(f)
        
    data = pd.read_csv('sample_input.csv')
    res = full_solver(model, preprocesser, data, '2022-12-01')
    print(res.head())
        
