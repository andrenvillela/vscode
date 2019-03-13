import pandas as pd
import numpy as np
import datetime as dt
import pickle


class Backtest:
    
    def __init__(self):
        pass

    def __repr__(self):
        return 'backtest organize data from trading strategy in new dataframe, option to make a summary using backtest_st and option to build a portfolio management after backtest'

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
            ''' SUMMARY OF THE BACKTEST() REPORTING ELA ANALISA O ARQUIVO BACKTEST INFORMANDO TX_ACERTO, RISK_RETORNO, RETORNO '''
            import warnings
            warnings.filterwarnings("ignore") #, message="invalid value encountered in long_scalars")

            database = pd.read_pickle('./Data/DF/INDICADOR.pickle')

            summary = {}
            start = df.index[0].year
            end = df.index[-1].year
            db = pd.DataFrame()

            if 'Total_Result' in df.columns:
                perc = 'Total_Result'
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
                    
                    db = pd.concat([db, pd.DataFrame(summary).T])
                    db = db.dropna()

                
            return db

    #############################################################################################

    def _portfolio(self, backtest):
        ''' HERE THE BACKTEST ADD PORTFOLIO MANAGEMENT TO DATAFRAME '''

        indicador = pd.read_pickle('./Data/DF/INDICADOR.pickle')

        import warnings
        warnings.filterwarnings("ignore", message='Addition/subtraction of integers and integer-arrays to Timestamp is deprecated, will be removed in a future version')
        warnings.filterwarnings("ignore", message='Passing integers to fillna is deprecated, will raise a TypeError in a future version')

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
            
            lt_new = [i for i in db1.Asset]
            close_new = indicador[(indicador.index == (i)) & (indicador.Asset.isin(lt_new))][['Asset', 'Close']]
            close_new = close_new.reset_index().set_index('Asset')

            new = pd.concat([new, close_new], axis=1, sort=True, join='inner').reset_index().set_index('Date')

            db2 = db2.append(new)
        
        return self._yesterday_portfolio(db2)

    #############################################################################################

    def _yesterday_portfolio(self, portfolio_df):
        import warnings
        warnings.filterwarnings("ignore", message='Addition/subtraction of integers and integer-arrays to Timestamp is deprecated, will be removed in a future version')
        warnings.filterwarnings("ignore", message='Passing integers to fillna is deprecated, will raise a TypeError in a future version')

        calendar = pd.date_range(start=sorted(set(portfolio_df.index))[0], end=sorted(set(portfolio_df.index))[-1], freq='B')

        db3 = pd.DataFrame()
        db4 = pd.DataFrame()

        for i in calendar:
            lt_old = set([i for i in portfolio_df[portfolio_df.index == (i-1)].Asset])
            lt_new = set([i for i in portfolio_df[portfolio_df.index == i].Asset])
            lt = lt_old.intersection(lt_new)

            if lt == set():
                pass
            else:
                for ii in lt:
                    yesterday_portf = [portfolio_df[(portfolio_df.Asset == ii) & (portfolio_df.index == (i-1))].PortFolio][0][0]
                    adjust_portf = {(i, ii): {'Yesterday_Portf': yesterday_portf}}
                    db3 = pd.concat([db3, pd.DataFrame(adjust_portf).T])

        df = portfolio_df.reset_index().set_index(['Date', 'Asset'])
        df = pd.concat([df, db3], axis=1).fillna(0).reset_index().rename({'level_0': 'Date', 'level_1': 'Asset'}, axis=1).set_index('Date')

        return self._mprice(df)

    #############################################################################################

    def _mprice(self, yesterday_portfolio_df):
        import warnings
        warnings.filterwarnings("ignore", message='Addition/subtraction of integers and integer-arrays to Timestamp is deprecated, will be removed in a future version')
        warnings.filterwarnings("ignore", message='Passing integers to fillna is deprecated, will raise a TypeError in a future version')

        df = yesterday_portfolio_df.copy()
        calendar = pd.date_range(start=sorted(set(df.index))[0], end=sorted(set(df.index))[-1], freq='B')

        db4 = pd.DataFrame()

        for i in calendar:
            for ii in df[df.index == i].Asset.unique():
                if df[(df.Asset == ii) & (df.index == i)].iloc[0].Yesterday_Portf == 0:
                    mPrice = {(i, ii): {'mPrice': df[(df.Asset == ii) & (df.index == i)].iloc[0].Entry_Price}}
                    db4 = pd.concat([db4, pd.DataFrame(mPrice).T], sort=True)

                else:
                    yesterday_portf = abs((df[(df.Asset == ii) & (df.index == i)].iloc[0].PortFolio -
                    df[(df.Asset == ii) & (df.index == i)].iloc[0].Yesterday_Portf) / 
                    df[(df.Asset == ii) & (df.index == i)].iloc[0].PortFolio)

                    today_portf = 1 - yesterday_portf if yesterday_portf < 1 else 1

                    if 'mPrice' in db4.columns:
                        db3x = db4.reset_index().rename({'level_0': 'Date', 'level_1': 'Asset'}, axis=1).set_index('Date')
                        mprice = (db3x[(db3x.Asset == ii) & (db3x.index == (i-1))].iloc[0].mPrice * today_portf +
                            df[(df.Asset == ii) & (df.index == i)].iloc[0].Close * yesterday_portf) / (today_portf + yesterday_portf)
                        mPrice = {(i, ii): {'mPrice': mprice}}
                        db4 = pd.concat([db4, pd.DataFrame(mPrice).T], sort=True)
                    else:
                        print('----------------- NO NO NO ----------------------')

        df = df.reset_index().set_index(['Date', 'Asset'])
        df = pd.concat([df, db4], axis=1).fillna(0).reset_index().rename({'level_0': 'Date', 'level_1': 'Asset'}, axis=1).set_index('Date')

        return self._result_partial(df)

    #############################################################################################

    def _result_partial(self, mprice):
        import warnings
        warnings.filterwarnings("ignore", message='Addition/subtraction of integers and integer-arrays to Timestamp is deprecated, will be removed in a future version')
        warnings.filterwarnings("ignore", message='Passing integers to fillna is deprecated, will raise a TypeError in a future version')

        df = mprice.copy()
        calendar = pd.date_range(start=sorted(set(df.index))[0], end=sorted(set(df.index))[-1], freq='B')

        db4 = pd.DataFrame()

        for i in calendar:
            for ii in df[df.index == i].Asset.unique():
                yesterday_portf = abs((df[(df.Asset == ii) & (df.index == i)].iloc[0].PortFolio -
                df[(df.Asset == ii) & (df.index == i)].iloc[0].Yesterday_Portf) / 
                df[(df.Asset == ii) & (df.index == i)].iloc[0].PortFolio)

                today_portf = 1 - yesterday_portf if yesterday_portf < 1 else 1

                if df[(df.Asset == ii) & (df.index == i)].iloc[0].L_S == 'LONG':
                    adjust_result = ((df[(df.Asset == ii) & (df.index == (i))].iloc[0].mPrice - 
                        df[(df.Asset == ii) & (df.index == i)].iloc[0].Close) / df[(df.Asset == ii) & (df.index == i)].iloc[0].Entry_Price) * today_portf * -1
                elif df[(df.Asset == ii) & (df.index == i)].iloc[0].L_S == 'SHORT':
                    adjust_result = ((df[(df.Asset == ii) & (df.index == (i))].iloc[0].mPrice - 
                        df[(df.Asset == ii) & (df.index == i)].iloc[0].Close) / df[(df.Asset == ii) & (df.index == i)].iloc[0].Entry_Price) * today_portf
                result = {(i, ii): {'Result_Partial': adjust_result}}
                db4 = pd.concat([db4, pd.DataFrame(result).T], sort=True)

        df = df.reset_index().set_index(['Date', 'Asset'])
        df = pd.concat([df, db4], axis=1).fillna(0).reset_index().rename({'level_0': 'Date', 'level_1': 'Asset'}, axis=1).set_index('Date')

        return self._result_close(df)

    #############################################################################################

    def _result_close(self, result_partial):
        import warnings
        warnings.filterwarnings("ignore", message='Addition/subtraction of integers and integer-arrays to Timestamp is deprecated, will be removed in a future version')
        warnings.filterwarnings("ignore", message='Passing integers to fillna is deprecated, will raise a TypeError in a future version')

        df = result_partial.copy()
        calendar = pd.date_range(start=sorted(set(df.index))[0], end=sorted(set(df.index))[-1], freq='B')

        db5 = pd.DataFrame()

        df['Exit_Date'] = pd.to_datetime(df['Exit_Date'])

        for i in calendar:
            for ii in df[df.index == i].Asset.unique():
                if df[(df.index == i) & (df.Asset == ii)].L_S[0] == 'LONG':
                    result = ((df[(df.index == i) & (df.Asset == ii)].Exit_Price - df[(df.index == i) & (df.Asset == ii)].mPrice) /
                        df[(df.index == i) & (df.Asset == ii)].Entry_Price) * df[(df.index == i) & (df.Asset == ii)].PortFolio
                    result_close = {(i, ii): {'Result_Close': result[0]}}
                elif df[(df.index == i) & (df.Asset == ii)].L_S[0] == 'SHORT':
                    result = ((df[(df.index == i) & (df.Asset == ii)].Exit_Price - df[(df.index == i) & (df.Asset == ii)].mPrice) /
                        df[(df.index == i) & (df.Asset == ii)].Entry_Price) * df[(df.index == i) & (df.Asset == ii)].PortFolio * -1
                    result_close = {(i, ii): {'Result_Close': result[0]}}

                db5 = pd.concat([db5, pd.DataFrame(result_close).T])


        df = df.reset_index().set_index(['Date', 'Asset'])
        df = pd.concat([df, db5], axis=1).fillna(0).reset_index().rename({'level_0': 'Date', 'level_1': 'Asset'}, axis=1).set_index('Date')

        df['Total_Result'] = df['Result_Partial'] + df['Result_Close']
        
        return df

    #############################################################################################

    def backtest(self, df, model):
        from multiprocessing import Pool, cpu_count
        num_process = min(df.shape[1], cpu_count())
        
        if model == 'backtest':
            with Pool(num_process) as pool:
                seq = [df[df['Asset'] == i] for i in df.Asset.unique()]

                results_list = pool.map(self._start, seq)

                db = pd.concat(results_list)
                print(db.head(), db.tail())

        elif model == 'portfolio':
            with Pool(num_process) as pool:
                seq = [df[df.index.year == i] for i in df.index.year.unique()]

                results_list = pool.map(self._portfolio, seq)

                db = pd.concat(results_list)

                print(db.head(), db.tail())

        elif model == 'summary':
            with Pool(num_process) as pool:
                seq = [df[df['Asset'] == i] for i in df.Asset.unique()]

                results_list = pool.map(self._summary, seq)

                db = pd.concat(results_list)

                print(db.head(), db.tail())
                db_st = db.groupby(level=[0,1]).sum()
                
            print('\n Per Asset_Year ... RETURN -> ', 
                round((sum([db.loc[i, :]['TOTAL_RETURN'].mean() for i in db.index.unique()]) / 
                db.reset_index().set_index('level_1').index.nunique()),2), 
                '& Win_X_Lose ->', round((db_st.groupby(level=[0,1]).mean().mean().Win_X_Lose),2), 
                '& RISK_RETURN ->', round((db_st.groupby(level=[0,1]).mean().mean().RISK_RETURN),2),
                '& SHARPE_RATIO ->', round((db['TOTAL_RETURN'].mean() / db['TOTAL_RETURN'].std()) * 
                (db.reset_index().set_index('level_1').index.nunique()),2))
    
        with open(f'./Data/DF/{model.upper()}.pickle', 'wb') as f:
            pickle.dump(db, f, protocol=pickle.HIGHEST_PROTOCOL)   
