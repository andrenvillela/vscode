import pandas as pd
import numpy as np
import datetime as dt
import pickle
from chart import chart

class Backtest:
    
    def __init__(self):
        ''' HERE I DEFINE THE BALANCE TO RUN THE BACKTEST '''
        ''' ALSO ADD THE OHLC_IND FILE USED TO RUN STRATEGY '''
        ''' CALENDAR FROM OHLC_IND USED IN SEVERAL LOOPS '''
        self.balance = 100000
        self.ohlc_ind = pd.read_pickle('./Data/DF/OHLC_IND.pickle')
        self.calendar = pd.date_range(start=sorted(set(self.ohlc_ind.index))[0], end=sorted(set(self.ohlc_ind.index))[-1], freq='B')

    def __repr__(self):
        return 'backtest organize data from trading strategy in new dataframes'

#############################################################################################

    def _start(self, df):
        """
        DF HAVE THE RESULTS OF STRATEGY().
        """

        self.ohlc_ind = self.ohlc_ind[self.ohlc_ind.Asset == df.Asset[0]]

        lt_long = []
        df1 = pd.DataFrame()
        df2 = pd.DataFrame()

        lt_short = []
        df3 = pd.DataFrame()
        df4 = pd.DataFrame()

        '''FIRST work the trades on BUY-LONG side, entry-exit'''
        for i in range(len(df)):
            if lt_long == []:
                if df.iloc[i].L_S == 'LONG':
                    df1 = pd.concat([df1, df.iloc[i]], axis=1, sort=True)
                    lt_long.append('LONG')
                else:
                    pass
            else:
                if df.iloc[i].L_S == 'cLONG':
                    df2 = pd.concat([df2, df.iloc[i]], axis=1, sort=True)
                    lt_long.clear()
                else:
                    pass
        '''SECOND work the trades on SELL-SHORT side, entry-exit'''
        for i in range(len(df)):
            if lt_short == []:
                if df.iloc[i].L_S == 'SHORT':
                    df3 = pd.concat([df3, df.iloc[i]], axis=1, sort=True)
                    lt_short.append('SHORT')
                else:
                    pass
            else:
                if df.iloc[i].L_S == 'cSHORT':
                    df4 = pd.concat([df4, df.iloc[i]], axis=1, sort=True)
                    lt_short.clear()
                else:
                    pass

        df1 = df1.T.reset_index().rename({'index':'del1', 'Date':'Entry_Date', 'Price':'Entry_Price'}, axis=1)

        df2 = pd.concat([df2, df.iloc[-1]], axis=1, sort=True)
        df2 = df2.T.reset_index().rename({'index':'del2', 'Date':'Exit_Date', 'Price':'Exit_Price'}, axis=1)
        df2 = df2.drop(['Asset', 'L_S', 'STRATEGY'], axis=1)

        df3 = df3.T.reset_index().rename({'index':'del1', 'Date':'Entry_Date', 'Price':'Entry_Price'}, axis=1)

        df4 = pd.concat([df4, df.iloc[-1]], axis=1, sort=True)
        df4 = df4.T.reset_index().rename({'index':'del2', 'Date':'Exit_Date', 'Price':'Exit_Price'}, axis=1)
        df4 = df4.drop(['Asset', 'L_S', 'STRATEGY'], axis=1)

        '''Aggregate the BUY-LONG'''
        df5 = pd.concat([df1, df2], axis=1, sort=True)
        df5 = df5.drop(['del1', 'del2'], axis=1) 
        df5 = df5.dropna()
        df5['Change%'] = (df5.Exit_Price - df5.Entry_Price) / df5.Entry_Price

        '''Aggregate the SELL-SHORT'''
        df6 = pd.concat([df3, df4], axis=1, sort=True)
        df6 = df6.drop(['del1', 'del2'], axis=1) 
        df6 = df6.dropna()
        df6['Change%'] = (df6.Entry_Price - df6.Exit_Price) / df6.Entry_Price

        df = pd.concat([df5, df6], ignore_index=True)
        df['Entry_Date'] = pd.to_datetime(df.Entry_Date)
        df['Exit_Date'] = pd.to_datetime(df.Exit_Date)
        df['Duration'] = df.Exit_Date - df.Entry_Date

        minima = {}
        maxima = {}
        for i in range(len(df)):
            minima.update({df.index[i]: (self.ohlc_ind.Low[(self.ohlc_ind.index > df['Entry_Date'][i]) &
                            (self.ohlc_ind.index <= df['Exit_Date'][i])].min())})
            maxima.update({df.index[i]: (self.ohlc_ind.High[(self.ohlc_ind.index > df['Entry_Date'][i]) &
                            (self.ohlc_ind.index <= df['Exit_Date'][i])].max())})

        min_max = pd.concat([
            pd.DataFrame(minima, index=['Min']).T, 
            pd.DataFrame(maxima, index=['Max']).T
        ], axis=1)

        df = pd.concat([df, min_max], axis=1)

        max_drawn = {}
        for i in range(len(df)):
            if df['L_S'][i] == 'LONG':
                max_drawn.update({df.index[i]:
                    (df.Min[i] / df.Entry_Price[i])-1
                })
            elif df['L_S'][i] == 'SHORT':
                max_drawn.update({df.index[i]:
                    (df.Entry_Price[i] / df.Max[i])-1
                    })
            else:
                pass

        max_drawn = pd.DataFrame(max_drawn, index=['Max_Drawn']).T

        df = pd.concat([df, max_drawn], axis=1)

        df.set_index('Entry_Date', inplace=True)
        df.sort_index(inplace=True)

        return df

    #############################################################################################

    def _portfolio(self, df):
        ''' HERE THE BACKTEST ADD PORTFOLIO TO DATAFRAME '''

        df = df.dropna()

        db1 = pd.DataFrame()
        db2 = pd.DataFrame()

        for i in self.calendar.unique():
            df_entry = df[df.index == i]
            df_exit = df[df.Exit_Date == i]
            db1 = pd.concat([db1, df_entry, df_exit], sort=True).drop_duplicates(keep=False)
            portf = pd.DataFrame(data=list(np.ones(len(db1)) * 1 / len(db1)), index=[i for i in db1.index], columns=['PortFolio'])
            portf.index.rename('Entry_Date', inplace=True)
            new = pd.concat([db1, portf], axis=1, sort=True).reset_index().set_index(['Asset'])
            new['Date'] = i
            db2 = pd.concat([db2, new], sort=True)
            
        db2 = db2.reset_index().set_index(['Date'])

        return self._yesterday_portfolio(db2)

    #############################################################################################

    def _yesterday_portfolio(self, df):
        ''' HERE THE BACKTEST ADD YESTERDAY PORTFOLIO TO DATAFRAME '''

        import warnings
        warnings.filterwarnings("ignore", message='Passing integers to fillna is deprecated, will raise a TypeError in a future version')

        db3 = pd.DataFrame()

        for i in range(len(self.calendar)):
            lt_old = set([i for i in df[df.index == (self.calendar[i-1])].Asset])
            lt_new = set([i for i in df[df.index == self.calendar[i]].Asset])
            lt = lt_old.intersection(lt_new)

            if lt == set():
                pass
            else:
                for ii in lt:
                    yesterday_portf = [df[(df.Asset == ii) & (df.index == self.calendar[i-1])].PortFolio][0][0]
                    adjust_portf = {(self.calendar[i], ii): {'Yesterday_Portf': yesterday_portf}}
                    db3 = pd.concat([db3, pd.DataFrame(adjust_portf).T])

        db3 = db3.reset_index().rename({'level_0': 'Date', 'level_1': 'Asset'}, axis=1).set_index(['Date', 'Asset'])
        df = df.reset_index().set_index(['Date', 'Asset'])
        df = pd.concat([df, db3], axis=1).reset_index().set_index('Date').fillna(0)

        return self._closing(df)

    #############################################################################################

    def _closing(self, df):
        ''' HERE ADD CLOSING PRICE FOR INDEX DATE '''
        pd.options.mode.chained_assignment = None

        port = df.reset_index().set_index(['Date', 'Asset'])
        self.ohlc_ind = self.ohlc_ind[['Asset', 'Close']].reset_index().set_index(['Date', 'Asset'])
        df = pd.concat([port, self.ohlc_ind], axis=1).dropna(thresh=10)
        nan = df[df.Close.isnull()]
        nan.Close = df.Entry_Price
        nan = nan.Close
        db = df.dropna().Close
        close = pd.concat([nan, db], sort=True)
        df = df.drop('Close', axis=1)
        df = pd.concat([df, close], axis=1, sort=True).reset_index().set_index('Date')
        
        return df

    #############################################################################################

    def _result(self, df):
        dicio = {}
        result = None

        for i in range(len(self.calendar)):
            for ii in df[df.index == self.calendar[i]].Asset:
                portf = df[(df.index == self.calendar[i]) & (df.Asset == ii)].iloc[0].PortFolio
                ytd_portf = df[(df.index == self.calendar[i]) & (df.Asset == ii)].iloc[0].Yesterday_Portf
                td_portf = abs(portf - ytd_portf)

                entry = df[(df.index == self.calendar[i]) & (df.Asset == ii)].iloc[0].Entry_Price
                today_close = df[(df.index == self.calendar[i]) & (df.Asset == ii)].iloc[0].Close
                tomorrow_close = df[(df.index == self.calendar[i]) & (df.Asset == ii)].iloc[0].Exit_Price

                if self.calendar[i] == self.calendar[-1] or (df[(df.index == self.calendar[i]) & (df.Asset == ii)].iloc[0].Exit_Date - self.calendar[i])==(self.calendar[i+1]-self.calendar[i]):                        
                    '''THE TRADE CLOSE TODAY'''           

                    if df[(df.index == self.calendar[i-1]) & (df.Asset == ii)].empty or df[(df.index == self.calendar[i]) & (df.Asset == ii)].iloc[0].Entry_Date == self.calendar[i]:
                        ''' YESTERDAY WAS EMPTY or ENTRY DATE EQUAL DATE'''
                        
                        if df[(df.index == self.calendar[i]) & (df.Asset == ii)].iloc[0].L_S == 'LONG':
                            result = ((tomorrow_close - today_close) / today_close) * portf

                        else:
                            result = ((tomorrow_close - today_close) / today_close) * -1 * portf  
                          
                    else:
                        yesterday_close = df[(df.index == self.calendar[i-1]) & (df.Asset == ii)].iloc[0].Close
                        if df[(df.index == self.calendar[i]) & (df.Asset == ii)].iloc[0].L_S == 'LONG':
                            result = (((tomorrow_close - today_close) / today_close) * td_portf + 
                                    ((today_close - yesterday_close) / today_close) * ytd_portf)

                        else:
                            result = (((tomorrow_close - today_close) / today_close) * -1 * td_portf + 
                                    ((today_close - yesterday_close) / today_close) * -1 * ytd_portf)

                else:
                    if df[(df.index == self.calendar[i-1]) & (df.Asset == ii)].empty or df[(df.index == self.calendar[i]) & (df.Asset == ii)].iloc[0].Entry_Date == self.calendar[i]:
                        ''' YESTERDAY WAS EMPTY or ENTRY DATE EQUAL DATE'''

                        if df[(df.index == self.calendar[i]) & (df.Asset == ii)].iloc[0].L_S == 'LONG':
                            result = ((entry - today_close) / entry) * portf
      
                        else:
                            result = ((entry - today_close) / entry) * -1 * portf   

                    else:
                        yesterday_close = df[(df.index == self.calendar[i-1]) & (df.Asset == ii)].iloc[0].Close
                        if df[(df.index == self.calendar[i]) & (df.Asset == ii)].iloc[0].L_S == 'LONG':
                            result = ((today_close - yesterday_close) / today_close) * ytd_portf
                                    
                        else:
                            result = ((today_close - yesterday_close) / today_close) * -1 * ytd_portf

                dicio.update({(self.calendar[i], ii): {'Result': round(result, 4)}})            

        db = pd.DataFrame(dicio).T
        df = df.reset_index().set_index(['Date', 'Asset'])
        df = pd.concat([df, db], axis=1, sort=True)
        
        return df

#############################################################################################

    def _summary(self, df):
        ''' SUMMARY OF THE BACKTEST() REPORTING BACK THE STATS OF STRATEGY '''
        import warnings
        warnings.filterwarnings("ignore") #, message="invalid value encountered in long_scalars")

        summary = {}
        start = df.index[0].year
        end = df.index[-1].year
        db = pd.DataFrame()

        if 'Result' in df.columns:
            perc = 'Result'
        else:
            perc = 'Change%'

        for ii in range(end - (start -1)):
            df2 = df[(df.index >= f'{start}-1-1') & (df.index <= f'{start}-12-31')]
            start = start + 1

            for i in df.Asset.unique():
                df1 = df2[df2.Asset == i]
            
                win_long = round((((df1[(df1['L_S'] == 'LONG') & (df1[perc] > 0)]).L_S.count()) /
                                ((df1[(df1['L_S'] == 'LONG')]).L_S.count())),2)
                win_short = round((((df1[(df1['L_S'] == 'SHORT') & (df1[perc] > 0)]).L_S.count()) /
                                ((df1[(df1['L_S'] == 'SHORT')]).L_S.count())),2)
                win_x_lose = round((((df1[(df1[perc] > 0)]).L_S.count()) /
                                ((df1).L_S.count())),2)

                risk_return_long = round((((df1[(df1['L_S'] == 'LONG') & (df1[perc] > 0)])[perc].mean()) /
                                abs((df1[(df1['L_S'] == 'LONG') & (df1[perc] < 0)])[perc].mean())),2)        
                risk_return_short = round((((df1[(df1['L_S'] == 'SHORT') & (df1[perc] > 0)])[perc].mean()) /
                                abs((df1[(df1['L_S'] == 'SHORT') & (df1[perc] < 0)])[perc].mean())),2)          
                risk_return = round((((df1[(df1[perc] > 0)])[perc].mean()) /
                                abs((df1[(df1[perc] < 0)])[perc].mean())),2)  

                return_long = round(df1[df1['L_S'] == 'LONG'][perc].sum(),4)
                return_short = round(df1[df1['L_S'] == 'SHORT'][perc].sum(),4)
                tt_return = round(df1[perc].sum(),4)

                summary = {(i, (start-1)): {'WIN_LONG': win_long, 'WIN_SHORT': win_short, 
                        'Win_X_Lose': win_x_lose, 'RISK_RETURN_LONG': risk_return_long, 'RISK_RETURN_SHORT': risk_return_short,
                        'RISK_RETURN': risk_return, 'RETURN_LONG': return_long, 'RETURN_SHORT': return_short, 'TOTAL_RETURN': tt_return}}


                summary_db = pd.DataFrame(summary).T.fillna(0)
                db = pd.concat([db, summary_db])
            
        return db

    #############################################################################################

    def backtest(self, df, model='backtest', final='Yes'):
        from multiprocessing import Pool, cpu_count
        num_process = min(df.shape[1], cpu_count())
        
        if model == 'backtest':
            with Pool(num_process) as pool:
                seq = [df[df['Asset'] == i] for i in df.Asset.unique()]

                results_list = pool.map(self._start, seq)

                db = pd.concat(results_list, sort=True)
                
                if final.upper() == 'YES':
                    return self.backtest(db, 'portfolio')
                else:
                    print(db.tail(), db.head())


        elif model == 'portfolio':
            with Pool(num_process) as pool:
                seq = [df[df.index.year == i] for i in df.index.year.unique()]

                results_list = pool.map(self._portfolio, seq)

                db = pd.concat(results_list, sort=True)

                if final.upper() == 'YES':
                    return self.backtest(db, 'result')
                else:
                    print(db.tail(), db.head())


        elif model == 'result':
            with Pool(num_process) as pool:
                seq = [df[df.index.year == i] for i in df.index.year.unique()]
                
                results_list = pool.map(self._result, seq)

                db = pd.concat(results_list, sort=True)

                if final.upper() == 'YES':
                    print(db.head(), db.tail())
                    db = db.reset_index().set_index('Date')
                    self.backtest(db, 'summary', 'No')     
                else:
                    print(db.tail(), db.head())


        elif model == 'summary':
            with Pool(num_process) as pool:
                seq = [df[df.index.year == i] for i in df.index.year.unique()]
                
                results_list = pool.map(self._summary, seq)

                db = pd.concat(results_list, sort=True)

                print(db.head(), '\n', db.tail())
                db_st = db.groupby(level=[0,1]).sum()
                
            print('\n Per Asset_Year ... RETURN -> ', 
                round(sum([db.loc[i, :]['TOTAL_RETURN'].mean() for i in db.index.unique()]) / 
                db.reset_index().set_index('level_1').index.nunique(),4), 
                '& Win_X_Lose ->', round((db_st.groupby(level=[0,1]).mean().mean().Win_X_Lose),2), 
                '& RISK_RETURN ->', round((db_st.groupby(level=[0,1]).mean().mean().RISK_RETURN),2),
                '& SHARPE_RATIO ->', round((db['TOTAL_RETURN'].mean() / db['TOTAL_RETURN'].std()) * 
                (db.reset_index().set_index('level_1').index.nunique()),2))
    
            pd.to_pickle(db, './Data/DF/SUMMARY.pickle')

        with open(f'./Data/DF/{model.upper()}.pickle', 'wb') as f:
            pickle.dump(db, f, protocol=pickle.HIGHEST_PROTOCOL)   

        
