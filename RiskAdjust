import pandas as pd
import numpy as np
from ParallelCalFactor import ParallelCalFactor


class RiskAdjust():
    def __init__(self, start_date='2018-01-01', end_date='2022-12-31') -> None:
        self.daily_path = '/mnt/132.data0/BasicData/StockDB/FundDB/raw/tonglian_db/mkt_equd_adj_af.h5'
        self.float_share_path = '/mnt/132.data0/BasicData/StockDB/FundDB/raw/tonglian_db/mkt_equd_eval.h5'
        self.risk = {}
        self.risk_coef = {}
        self.adjust_factor = {}
        self.start_date = start_date
        self.end_date = end_date
        self._get_data(start_date, end_date)
    
    def _get_data(self, start_date, end_date):
        # daily data
        self.daily = pd.read_hdf(self.daily_path)
        self.daily = self.daily.rename(columns={'TRADE_DATE': 'date', 'TICKER_SYMBOL': 'securityid', 'CLOSE_PRICE_2': 'close', 'OPEN_PRICE_2': 'open', 'LOWEST_PRICE_2': 'low', 'HIGHEST_PRICE_2': 'high', 'TURNOVER_VOL': 'volume'})
        self.daily = self.daily[['date', 'securityid', 'close', 'open', 'low', 'high', 'volume']]
        self.daily['time'] = self.daily['date'].apply(lambda x: x.strftime(format='%Y-%m-%d'))
        self.daily = self.daily[self.daily['time'] >= start_date]
        self.daily = self.daily[self.daily['time'] <= end_date]
        self.daily = self.daily.drop(columns=['time'])

        self.float_share = pd.read_hdf(self.float_share_path)
        self.float_share = self.float_share[['TICKER_SYMBOL', 'TRADE_DATE', 'NEG_MARKET_VALUE']]
        self.float_share = self.float_share.rename(columns={'TICKER_SYMBOL':'securityid', 'TRADE_DATE': 'date', 'NEG_MARKET_VALUE': 'float_share'})
        self.float_share['time'] = self.float_share['date'].apply(lambda x: x.strftime(format='%Y-%m-%d'))
        self.float_share = self.float_share[self.float_share['time'] >= start_date]
        self.float_share = self.float_share[self.float_share['time'] <= end_date]
        self.float_share = self.float_share.drop(columns=['time'])

        # minute data
        # amt = pd.read_parquet(self.minute_amt_path)
        # self.amt = amt.loc[start_date: end_date, :]
        
    
    def get_risk(self, d=10):
        self._get_daily_risk_factors(d=d)
        self._get_minute_risk_factors(d=d)
        self._get_trade_data_risk_factors(d=d)
    
    def _get_daily_risk_factors(self, d=10):
        daily = self.daily.copy()
        daily['prev_close'] = daily.groupby(['securityid'])['close'].shift(1)
        daily['h-l'] = daily['high']-daily['low']
        daily['h-pc'] = daily['high'] - daily['prev_close']
        daily['l-pc'] = daily['low'] - daily['prev_close']
        daily['true_range'] = daily[['h-l', 'h-pc', 'l-pc']].max(axis=1)/daily['prev_close']
        self.risk['true_range'] = pd.pivot_table(daily, values='true_range', index='date', columns='securityid')
        self.risk_coef['true_range'] = self.risk['true_range'].rolling(d).mean() - self.risk['true_range']

        daily = daily.merge(self.float_share, on=['securityid', 'date'], how='inner')
        daily['turnover'] = daily['volume']/daily['float_share']
        self.risk['turnover'] = pd.pivot_table(daily, values='turnover', index='date', columns='securityid')
        self.risk_coef['turnover'] = self.risk['turnover'].rolling(d).mean() - self.risk['turnover']
    
    def _get_trade_data_risk_factors(self, d=10):
        params_dict = {
            # data
            'start_date': self.start_date,                 # 因子计算开始日期
            'end_date': self.end_date,                   # 因子计算结束日期
            'data_type': 'L2',                       # daily/min/L2, 数据种类
            'data_key': ['trade'],            # 对应数据种类下想取的表名/字段名(L2:market/order/trade)
            'date_col_name': ['TRADE_DATE'],            # (daily)数据表对应时间字段名称，分发数据必要字段
            'secid_col_name': ['securityid'],        # (daily/L2)数据表对应股票代码字段名称，分发数据必要字段
            'use_cols': [['securityid', 'date', 'tradp', 'tradv', 'updatetime', 'bs']],       # (daily/L2)数据表所使用的列名称，不传入该参数默认使用全部列
            'rolling_window': 1,                        # 因子计算的时序日频窗口长度
            'show_example_data': False,                 # 如果为True，则输出展示用数据
            }
        
        Pro = ParallelCalFactor()
        # 计算结果.user_factor_func为用户提供的计算因子逻辑函数.
        res = Pro.process_factor(self._calc_risk_trade_data, params=params_dict)
        # 直接输出
        for col in res.columns:
            if col in ['date', 'securityid']:
                continue
            self.risk[col] = pd.pivot_table(res, values=col, index='date', columns='securityid')
            self.risk_coef[col] = self.risk[col].rolling(d).mean() - self.risk[col]
    
    def _calc_risk_trade_data(self, df, params=None):
        # 平均单笔成交
        df = df['trade']
        df = df.copy()
        # 数据预处理
        df = df[df['bs'] != 'N']
        df = df[df['tradp'] > 0]
        df = df[df['updatetime'] >= '09:30']
        df = df[df['updatetime'] < '14:57']
        if len(df) < 1:
            return pd.DataFrame()
        all_loop_date = df['date'].astype(str).unique().tolist()
        all_loop_date.sort()
        cdate = all_loop_date[-1]
        data = pd.DataFrame([[cdate, df['securityid'].unique().tolist()[0]]], columns=['date', 'securityid'])
        
        df['amt'] = df['tradv']*df['tradp']
        data['average_volume_transaction'] = np.sqrt(df['amt'].sum()/len(df))
        df['is_big'] = df['amt'] > 200000
        vwap_big = df[df['is_big']]['amt'].sum()/df[df['is_big']]['tradv'].sum()
        vwap = df['amt'].sum()/df['tradv'].sum()
        data['vwap_big_drift'] = abs(vwap_big - vwap) / vwap
        df['is_small'] = df['amt'] < 40000
        data['active_buy_small_ratio'] = - df[df['is_small']]['amt'].sum() / df['amt'].sum()
        return data
    
    def _get_minute_risk_factors(self, d=10):
        params_dict = {
            # data
            'start_date': self.start_date,                 # 因子计算开始日期
            'end_date': self.end_date,                   # 因子计算结束日期
            'data_type': 'min',                       # daily/min/L2, 数据种类
            'data_key': ['volume'],            # 对应数据种类下想取的表名/字段名(L2:market/order/trade)
            'date_col_name': ['TRADE_DATE'],            # (daily)数据表对应时间字段名称，分发数据必要字段
            'secid_col_name': ['securityid'],        # (daily/L2)数据表对应股票代码字段名称，分发数据必要字段
            #'use_cols': [['securityid', 'date', 'tradp', 'tradv', 'updatetime', 'bs']],       # (daily/L2)数据表所使用的列名称，不传入该参数默认使用全部列
            'rolling_window': 1,                        # 因子计算的时序日频窗口长度
            'show_example_data': False,                 # 如果为True，则输出展示用数据
            }
        
        Pro = ParallelCalFactor()
        # 计算结果.user_factor_func为用户提供的计算因子逻辑函数.
        res = Pro.process_factor(self._calc_risk_min_data, params=params_dict)
        float_share = self.float_share.copy()
        float_share['date'] = float_share['date'].apply(lambda x: x.strftime(format='%Y-%m-%d'))
        float_share = pd.pivot_table(float_share, values='float_share', index='date', columns='securityid')
        for col in res.columns:
            if col in ['date', 'securityid']:
                continue
            self.risk[f'{col}_to_cap'] = pd.pivot_table(res, values=col, index='date', columns='securityid')
            self.risk_coef[f'{col}_to_cap'] = self.risk[f'{col}_to_cap'].rolling(d).mean()-self.risk[f'{col}_to_cap']

    def _calc_risk_min_data(df, params=None):
        amt = df['money'].copy()
        col_name = [col.split('.')[0] for col in amt.columns]
        amt = amt.set_axis(col_name, axis=1)
        all_loop_dates = amt.index.strftime('%F').unique().tolist()
        cdate = all_loop_dates[-1]
        data = pd.DataFrame(index=amt.columns, columns=['date'])
        data.index.name = 'securityid'
        data['date'] = cdate

        data['opening_amt'] = amt.loc[:f'{cdate} 10:00:01'].sum()
        data['closing_amt'] = amt.loc[f'{cdate} 14:30:00': ].sum()
        data['opening_closing_amt'] = data['opening_amt'] + data['closing_amt']
        return data.reset_index()

    
    def adjust_factor(self, factor, d=10, m=20):
        """
        按风险系数调整每日因子
        factor: 待调整因子
        d: 计算基准风险的日期， 风险系数 risk_coef = risk_base - risk, risk_base = risk.rolling(d).mean()
        m: 调整因子使用的日期， adj_factor = ewm(risk_coef*factor, m)
        return: adj_factors - dict
                keys: 'riskname_factorname'
                values: pd.DataFrame(index=date, columns=securityid, value=factors_adj)
        """
        for name, risk in self.risk.items():
            self.risk = risk.rolling(d).mean() - risk
            self.adj_factors[name] = (risk*factor).ewm(span=m).mean()
        return self.adj_factors
    
    def save_risk_coef(self, risk_save_path='/mnt/Data/yanglufan/RiskAdjust/risk_factors/'):
        for key, value in self.risk_coef.item():
            value.to_parquet(risk_save_path + f'delta_{key}.parquet')
  
