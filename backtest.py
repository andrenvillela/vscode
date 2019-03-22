import pandas as pd
import numpy as np
import datetime as dt
import pickle
from chart import chart

class Backtest:
    
    def __init__(self):
        ''' HERE I DEFINE THE BALANCE TO RUN THE BACKTEST '''
        self.balance = 100000

    def __repr__(self):
        return 'backtest organize data from trading strategy in new dataframes'

#############################################################################################

    def _start(self, df):
        """
        DF HAVE THE RESULTS OF STRATEGY().
        ====================
        DATABASE HAVE THE FILE USED TO RUN STRATEGY().
        """

        database = pd.read_pickle('./Data/DF/INDICADOR.pickle')
        database = database[database.Asset == df.Asset[0]]

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

        # Aggregate the BUY-LONG
        df5 = pd.concat([df1, df2], axis=1, sort=True)
        df5 = df5.drop(['del1', 'del2'], axis=1) 
        df5 = df5.dropna()
        df5['Change%'] = (df5.Exit_Price - df5.Entry_Price) / df5.Entry_Price

        # Aggregate the SELL-SHORT
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
            minima.update({df.index[i]: (database.Low[(database.index > df['Entry_Date'][i]) &
                            (database.index <= df['Exit_Date'][i])].min())})
            maxima.update({df.index[i]: (database.High[(database.index > df['Entry_Date'][i]) &
                            (database.index <= df['Exit_Date'][i])].max())})

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

    def _summary(self, df):
        ''' SUMMARY OF THE BACKTEST() REPORTING BACK THE STATS OF STRATEGY '''
        import warnings
        warnings.filterwarnings("ignore") #, message="invalid value encountered in long_scalars")

        database = pd.read_pickle('./Data/DF/INDICADOR.pickle')

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

    def _portfolio(self, backtest):
        ''' HERE THE BACKTEST ADD PORTFOLIO MANAGEMENT TO DATAFRAME '''

        backtest = backtest.dropna()

        calendar = pd.date_range(start=sorted(set(backtest.index))[0], end=sorted(set(backtest.index))[-1], freq='B')
        db1 = pd.DataFrame()
        db2 = pd.DataFrame()

        for i in calendar:
            df_entry = backtest[backtest.index == i]
            df_exit = backtest[backtest.Exit_Date == i]
            db1 = pd.concat([db1, df_entry, df_exit], sort=True).drop_duplicates(keep=False)
            portf = pd.DataFrame(data=list(np.ones(len(db1)) * 1 / len(db1)), index=[i for i in db1.index], columns=['PortFolio'])
            portf.index.rename('Entry_Date', inplace=True)
            new = pd.concat([db1, portf], axis=1, sort=True).reset_index().set_index(['Asset'])
            new['Date'] = i
            db2 = pd.concat([db2, new], sort=True)
            
        db2 = db2.reset_index().set_index(['Date'])

        return self._yesterday_portfolio(db2)

    #############################################################################################

    def _yesterday_portfolio(self, portfolio_df):
        import warnings
        warnings.filterwarnings("ignore", message='Passing integers to fillna is deprecated, will raise a TypeError in a future version')

        calendar = pd.date_range(start=sorted(set(portfolio_df.index))[0], end=sorted(set(portfolio_df.index))[-1], freq='B')
        calendar = [i.date() for i in calendar if i in portfolio_df.index.unique()]
        calendar = pd.to_datetime(calendar)

        db3 = pd.DataFrame()

        for i in range(len(calendar)):
            lt_old = set([i for i in portfolio_df[portfolio_df.index == (calendar[i-1])].Asset])
            lt_new = set([i for i in portfolio_df[portfolio_df.index == calendar[i]].Asset])
            lt = lt_old.intersection(lt_new)

            if lt == set():
                pass
            else:
                for ii in lt:
                    yesterday_portf = [portfolio_df[(portfolio_df.Asset == ii) & (portfolio_df.index == calendar[i-1])].PortFolio][0][0]
                    adjust_portf = {(calendar[i], ii): {'Yesterday_Portf': yesterday_portf}}
                    db3 = pd.concat([db3, pd.DataFrame(adjust_portf).T])

        db3 = db3.reset_index().rename({'level_0': 'Date', 'level_1': 'Asset'}, axis=1).set_index(['Date', 'Asset'])
        df = portfolio_df.reset_index().set_index(['Date', 'Asset'])
        df = pd.concat([df, db3], axis=1).reset_index().set_index('Date').fillna(0)

        return df

    #############################################################################################

    def _closing(self, portfolio):
        pd.options.mode.chained_assignment = None

        indicador = pd.read_pickle('./Data/DF/INDICADOR.pickle')

        port = portfolio.reset_index().set_index(['Date', 'Asset'])
        indicador = indicador[['Asset', 'Close']].reset_index().set_index(['Date', 'Asset'])
        df = pd.concat([port, indicador], axis=1).dropna(thresh=10)
        nan = df[df.Close.isnull()]
        nan.Close = df.Entry_Price
        nan = nan.Close
        db = df.dropna().Close
        close = pd.concat([nan, db], sort=True)
        df = df.drop('Close', axis=1)
        df = pd.concat([df, close], axis=1, sort=True).reset_index().set_index('Date')
        
        return df

    #############################################################################################


    def _backtest_final(self, df):

        df = df.dropna()

        calendar = pd.date_range(start=sorted(set(df.index))[0], end=sorted(set(df.index))[-1], freq='B')
        calendar = [i.date() for i in calendar if i in df.index.unique()]
        calendar = pd.to_datetime(calendar)

        db5 = pd.DataFrame()
        assets = []
        dates = []
        mgt = {}

        for i in range(len(calendar)):
            for ii in df[df.index == calendar[i]].Asset: ######### .unique()
                portf = df[(df.index == calendar[i]) & (df.Asset == ii)].PortFolio[0]
                if df[(df.index == calendar[i]) & (df.Asset == ii)].Close[0] == 0:
                    close = df[(df.index == calendar[i]) & (df.Asset == ii)].Entry_Price[0]
                else:
                    close = df[(df.index == calendar[i]) & (df.Asset == ii)].Close[0]
                ytd_portf = df[(df.index == calendar[i]) & (df.Asset == ii)].Yesterday_Portf[0]
                entry_price = df[(df.index == calendar[i]) & (df.Asset == ii)].Entry_Price[0]


                if ii not in assets:
                    assets.append(ii)
                    dates.append(calendar[i])
                    qtd = self.balance * portf / entry_price
                    mgt = {ii: {'Date': calendar[i], 'QTD': qtd}}
                    db5 = pd.concat([db5, pd.DataFrame(mgt).T], sort=True)

                elif ytd_portf == 0:
                    qtd = self.balance * portf / entry_price
                    mgt = {ii: {'Date': calendar[i], 'QTD': qtd}}
                    db5 = pd.concat([db5, pd.DataFrame(mgt).T], sort=True)


                else:
                    if db5[(db5.Date == calendar[i-1]) & (db5.index == ii)].empty:
                        pass
                    else:
                        if portf == ytd_portf:
                            qtd = db5[(db5.Date == calendar[i-1]) & (db5.index == ii)].QTD[0]
                            mgt = {ii: {'Date': calendar[i], 'QTD': qtd}}
                            db5 = pd.concat([db5, pd.DataFrame(mgt).T], sort=True)
                            
                        elif portf > ytd_portf:
                            qtd = (self.balance * (portf-ytd_portf) / close) + db5[(db5.Date == calendar[i-1]) & (db5.index == ii)].QTD[0]
                            mgt = {ii: {'Date': calendar[i], 'QTD': qtd}}
                            db5 = pd.concat([db5, pd.DataFrame(mgt).T], sort=True)

                        elif portf < ytd_portf:
                            qtd = (portf / ytd_portf) * db5[(db5.Date == calendar[i-1]) & (db5.index == ii)].QTD[0]
                            mgt = {ii: {'Date': calendar[i], 'QTD': qtd}}
                            db5 = pd.concat([db5, pd.DataFrame(mgt).T], sort=True)
                
                db = df[(df.index == calendar[i])][['Asset', 'Close', 'Entry_Price', 'L_S']].reset_index().set_index(['Asset', 'Date'])
                QTD = db5[db5.Date == calendar[i]].reset_index().rename({'index': 'Asset'}, axis=1).set_index(['Asset', 'Date'])#.drop_duplicates(keep='first')
                db = pd.concat([db, QTD], axis=1, sort=True)

            total = []
            for iii in range(len(db)):
                if db.iloc[iii].L_S == 'LONG':
                    total.append(((db.iloc[iii].Close - db.iloc[iii].Entry_Price) + 
                        db.iloc[iii].Entry_Price) * db.iloc[iii].QTD)
                elif db.iloc[iii].L_S == 'SHORT':
                    total.append(((db.iloc[iii].Close - db.iloc[iii].Entry_Price)*-1 + 
                        db.iloc[iii].Entry_Price) * db.iloc[iii].QTD)

            if sum(total) > 0:
                self.balance = sum(total)            
            else:
                self.balance = 100000

        db5 = db5.reset_index().rename({'index':'Asset'}, axis=1).set_index(['Date', 'Asset'])
        df = df.reset_index().set_index(['Date', 'Asset'])
        df = pd.concat([df, db5], axis=1, sort=True)

        balance = {}
        for i in range(len(df)):  
            if 'LONG' in df.iloc[i].L_S:
                balance.update({df.index[i]: ((df.iloc[i].Close - df.iloc[i].Entry_Price) + df.iloc[i].Entry_Price) * df.iloc[i].QTD})
            else:
                balance.update({df.index[i]: ((df.iloc[i].Close - df.iloc[i].Entry_Price)*-1 + df.iloc[i].Entry_Price) * df.iloc[i].QTD})
        
        balance = pd.DataFrame(balance.values(), index=balance.keys()).rename({0: 'Balance'}, axis=1)
        
        df = pd.concat([df, balance], axis=1, sort=True)
        df = df.reset_index().set_index('Date')


        return df

    #############################################################################################

    def _backtest(self, df):
        calendar = pd.date_range(start=sorted(set(df.index))[0], end=sorted(set(df.index))[-1], freq='B')
        calendar = [i.date() for i in calendar if i in df.index.unique()]
        calendar = pd.to_datetime(calendar)
        result = {}

        for i in range(len(calendar)):
            today = df[df.index == calendar[i]].Balance.sum()
            yesterday = df[df.index == calendar[i-1]].Balance.sum()
            for ii in df[df.index == calendar[i]].Asset:
                result.update({(calendar[i], ii): {'Result': round(((today/yesterday)-1)/len(df[df.index == calendar[i]]),4)}})

        result = pd.DataFrame(result.values(), result.keys())
        df = df.reset_index().set_index(['Date', 'Asset'])
        df = pd.concat([df, result], axis=1, sort=True)
        df = df.reset_index().set_index('Date')

        df = df.drop('Balance', axis=1)
        
        return df

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

                db = pd.concat(results_list)
                db = self._closing(db)

                if final.upper() == 'YES':
                    return self.backtest(db, 'backtest_final')
                else:
                    print(db.tail(), db.head())


        elif model == 'backtest_final':
            with Pool(num_process) as pool:
                seq = [df[df.index.year == i] for i in df.index.year.unique()]
                
                results_list = pool.map(self._backtest_final, seq)

                db = pd.concat(results_list)

                if final.upper() == 'YES':
                    db = self._backtest(db)
                    print(db.head(), db.tail())
                    self.backtest(db, 'summary', 'No')     
                else:
                    print(db.tail(), db.head())


        elif model == 'summary':
            with Pool(num_process) as pool:
                seq = [df[df.index.year == i] for i in df.index.year.unique()]
                
                results_list = pool.map(self._summary, seq)

                db = pd.concat(results_list)

                print(db.head(), '\n', db.tail())
                db_st = db.groupby(level=[0,1]).sum()
                
            print('\n Per Asset_Year ... RETURN -> ', 
                round((sum([db.loc[i, :]['TOTAL_RETURN'].mean() for i in db.index.unique()]) / 
                db.reset_index().set_index('level_1').index.nunique()),4), 
                '& Win_X_Lose ->', round((db_st.groupby(level=[0,1]).mean().mean().Win_X_Lose),2), 
                '& RISK_RETURN ->', round((db_st.groupby(level=[0,1]).mean().mean().RISK_RETURN),2),
                '& SHARPE_RATIO ->', round((db['TOTAL_RETURN'].mean() / db['TOTAL_RETURN'].std()) * 
                (db.reset_index().set_index('level_1').index.nunique()),2))
    
            pd.to_pickle(db, './Data/DF/SUMMARY.pickle')

        with open(f'./Data/DF/{model.upper()}.pickle', 'wb') as f:
            pickle.dump(db, f, protocol=pickle.HIGHEST_PROTOCOL)   

        
