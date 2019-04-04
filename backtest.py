import pandas as pd
import numpy as np
import datetime as dt
import pickle
from chart import chart
import timeit

class Backtest:
    
    def __init__(self):
        ''' HERE I DEFINE THE BALANCE TO RUN THE BACKTEST '''
        ''' ALSO ADD THE OHLC_IND FILE USED TO RUN STRATEGY '''
        ''' CALENDAR FROM OHLC_IND USED IN SEVERAL LOOPS '''
        self.name = 'STOCH'
        self.ohlc_ind = pd.read_pickle('./Data/DF/OHLC_IND_STOCKUS.pickle')
        self.calendar = pd.date_range(start=sorted(set(self.ohlc_ind.index))[0], end=sorted(set(self.ohlc_ind.index))[-1], freq='B')

    def __repr__(self):
        return 'organize and process the data from OHLC file with indicators to new dataframes'

#############################################################################################

    def _strategy(self, df):
        """
        DF HAVE THE OHLC with the INDICATORS to be used on the STRATEGY below.
        """

        lt_ = []
        strag = None

        entry = pd.DataFrame()
        database = pd.DataFrame()


        for i in df.iterrows():
            db = i[1]

            if lt_ == []:
                '''HERE ENTER THE IDEA OF TRADING ENTRY ON LONG AND SHORT SIDE'''
                if db.Close > db.SMA7:
                    entry = pd.DataFrame(data={'Type':[self.name], 'Entry_Date':[db.name], 'Entry_Price':[db.Close], 'L_S':['LONG'], 'Asset':[db.Asset]})
                    lt_.append('LONG')
                    
                
                elif db.Close < db.SMA7:
                    entry = pd.DataFrame(data={'Type':[self.name], 'Entry_Date':[db.name], 'Entry_Price':[db.Close], 'L_S':['SHORT'], 'Asset':[db.Asset]})
                    lt_.append('SHORT')


            elif lt_ == ['LONG']:
                '''HERE ENTER THE IDEA OF CLOSING THE LONG TRADING IDEA'''
                if db.Close < db.SMA7:
                    exit = pd.DataFrame(data={'Exit_Date':[db.name], 'Exit_Price':[db.Close]})
                    entry = pd.concat([entry, exit], axis=1, sort=True)
                    lt_.clear()


            elif lt_ == ['SHORT']:
                '''HERE ENTER THE IDEA OF CLOSING THE SHORT TRADING IDEA'''
                if db.Close > db.SMA7:
                    exit = pd.DataFrame(data={'Exit_Date':[db.name], 'Exit_Price':[db.Close]})
                    entry = pd.concat([entry, exit], axis=1, sort=True)
                    lt_.clear()
                                
            database = pd.concat([database, entry], sort=True)

        database = database.dropna().drop_duplicates(keep='first').set_index('Entry_Date')
        database['Duration'] = database.Exit_Date - database.index

        return database

#############################################################################################

    def _portfolio(self, df):
        ''' HERE THE BACKTEST ADD PORTFOLIO DIVISION TO DATAFRAME '''

        calendar = self.calendar[(self.calendar >= df.index.min()) & (self.calendar <= df.Exit_Date.max())]
        df = df.dropna()

        db1 = pd.DataFrame()
        db2 = pd.DataFrame()

        for i in calendar.unique():
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
        ''' HERE THE BACKTEST ADD YESTERDAY PORTFOLIO DIVISION TO DATAFRAME '''

        import warnings
        warnings.filterwarnings("ignore", message='Passing integers to fillna is deprecated, will raise a TypeError in a future version')

        calendar = self.calendar[(self.calendar >= df.index.min()) & (self.calendar <= df.index.max())]
        db3 = pd.DataFrame()

        for i in range(len(calendar)):
            lt_old = set([i for i in df[df.index == (calendar[i-1])].Asset])
            lt_new = set([i for i in df[df.index == calendar[i]].Asset])
            lt = lt_old.intersection(lt_new)

            if lt == set():
                pass
            else:
                for ii in lt:
                    yesterday_portf = [df[(df.Asset == ii) & (df.index == calendar[i-1])].PortFolio][0][0]
                    adjust_portf = {(calendar[i], ii): {'Yesterday_Portf': yesterday_portf}}
                    db3 = pd.concat([db3, pd.DataFrame(adjust_portf).T])

        db3 = db3.reset_index().rename({'level_0': 'Date', 'level_1': 'Asset'}, axis=1).set_index(['Date', 'Asset'])
        df = df.reset_index().set_index(['Date', 'Asset'])
        df = pd.concat([df, db3], axis=1).reset_index().set_index('Date').fillna(0)

        return df

    #############################################################################################

    def _result(self, df):
        ''' HERE ADD CLOSING PRICE FOR INDEX DATE '''
        min_date = min(df.index.min(), df.Entry_Date.min(), df.Exit_Date.min()) - pd.Timedelta(30, 'D')
        max_date = max(df.index.max(), df.Entry_Date.max(), df.Exit_Date.max()) + pd.Timedelta(30, 'D')

        ohlc_ind = self.ohlc_ind[(self.ohlc_ind.index >= min_date) & (self.ohlc_ind.index <= max_date)]

        df = df.reset_index().set_index(['Date', 'Asset'])
        close = ohlc_ind.reset_index().set_index(['Date', 'Asset'])['Close']
        db1 = pd.concat([df, close], axis=1, sort=True).dropna()
        db1 = db1.reset_index().set_index('Date')

        db2 = pd.DataFrame()

        for i in db1.Asset.unique():
            ytd_close = db1[db1.Asset == i].shift(1)[['Asset', 'Close']]
            db2 = pd.concat([db2, ytd_close])
        
        db2 = db2.reset_index().set_index(['Date', 'Asset']).dropna().rename({'Close':'YTD_Close'}, axis=1)
        db1 = db1.reset_index().set_index(['Date', 'Asset'])

        db1 = pd.concat([db1, db2], axis=1, sort=True).reset_index().set_index('Date').dropna()

        dicio = {}

        for i in db1.iterrows():
            db = i[1]

            date = db.name
            asset = db.Asset

            entry_date = db.Entry_Date
            exit_date = db.Exit_Date
            entry_price = db.Entry_Price
            exit_price = db.Exit_Price

            close = db.Close
            ytd_close = db.YTD_Close

            min_period = ohlc_ind[(ohlc_ind.Asset == asset) & (ohlc_ind.index >= entry_date) & (ohlc_ind.index <= exit_date)].Low.min()
            max_period = ohlc_ind[(ohlc_ind.Asset == asset) & (ohlc_ind.index >= entry_date) & (ohlc_ind.index <= exit_date)].High.max()            

            if db.L_S == 'LONG':
                max_drawn = (min_period - entry_price) / entry_price
            else:
                max_drawn = (entry_price - max_period) / entry_price

            portf = db.PortFolio
            ytd_portf = db.Yesterday_Portf
            td_portf = abs(portf - ytd_portf)
    
            if db1[db1.index >= date].index.unique()[1] == db1.index[-1]:
                break
            else:
                closing = db1[db1.index >= date].index.unique()[1]

            if exit_date == closing:
                if db.L_S == 'LONG':
                    result = (((exit_price - close) / close) * td_portf + 
                            ((close - ytd_close) / close) * ytd_portf)

                else:
                    result = (((exit_price - close) / close) * -1 * td_portf + 
                            ((close - ytd_close) / close) * -1 * ytd_portf)

            else:
                if db.L_S == 'LONG':
                    result = ((close - ytd_close) / close) * ytd_portf
                            
                else:
                    result = ((close - ytd_close) / close) * -1 * ytd_portf

            dicio.update({(date, asset): {'Result': result, 'Min': min_period, 'Max': max_period, 'Max_Drawn': max_drawn}})


        dicio = pd.DataFrame(dicio).T
        db1 = db1.reset_index().set_index(['Date', 'Asset'])

        db1 = pd.concat([db1, dicio], axis=1, sort=True).reset_index()
        db1 = db1.rename({'level_0':'Date', 'level_1':'Asset'}, axis=1).set_index('Date')
        df = db1.dropna()        

        return df

    #############################################################################################

    def _summary(self, df):
        ''' SUMMARY OF THE BACKTEST() REPORTING BACK THE STATS OF STRATEGY '''
        import warnings
        warnings.filterwarnings("ignore") #, message="invalid value encountered in long_scalars")

        summary = {}
        db = pd.DataFrame()

        if 'Result' in df.columns:
            perc = 'Result'
        else:
            perc = 'Change%'

        for ii in df.index.year.unique():
            for i in df.Asset.unique():
                df1 = df[(df.index.year.isin([ii])) & (df.Asset == i)]
            
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

                summary = {(i, ii): {'WIN_LONG': win_long, 'WIN_SHORT': win_short, 
                        'Win_X_Lose': win_x_lose, 'RISK_RETURN_LONG': risk_return_long, 'RISK_RETURN_SHORT': risk_return_short,
                        'RISK_RETURN': risk_return, 'RETURN_LONG': return_long, 'RETURN_SHORT': return_short, 'TOTAL_RETURN': tt_return}}


                summary_db = pd.DataFrame(summary).T.fillna(0)
                db = pd.concat([db, summary_db])
            
        return db

    #############################################################################################

    def _start(self, df):
        from multiprocessing import Pool, cpu_count
        num_process = min(df.shape[1], cpu_count())

        lt = [self._strategy, self._portfolio, self._result, self._summary]
        #  
        for i in lt:
            print(i)
            start_time = timeit.default_timer()

            if i == self._strategy:
                seq = [df[df['Asset'] == i] for i in df.Asset.unique()]
            elif i == self._summary:
                df = df.reset_index().set_index('Date')
                seq = [df[df.index.year == i] for i in df.index.year.unique()]
            else:
                seq = [df[df.index.year == i] for i in df.index.year.unique()]
            
            with Pool(num_process) as pool:
                results_list = pool.map(i, seq)

                df = pd.concat(results_list, sort=True)    

                i = str(i).split('_')[1].split(' ')[0].upper()

                pd.to_pickle(df, f'./Data/BACKTEST/{i}.pickle')
                print(df.head()) 
                print(f'\n Time for this operation: ', timeit.default_timer() - start_time)


        db_st = df.groupby(level=[0,1]).sum()
                
        print('\n Per Asset_Year ... RETURN -> ', 
            round(sum([df.loc[i, :]['TOTAL_RETURN'].mean() for i in df.index.unique()]) / 
            df.reset_index().set_index('level_1').index.nunique(),4), 
            '& Win_X_Lose ->', round((db_st.groupby(level=[0,1]).mean().mean().Win_X_Lose),2), 
            '& RISK_RETURN ->', round((db_st.groupby(level=[0,1]).mean().mean().RISK_RETURN),2),
            '& SHARPE_RATIO ->', round((df['TOTAL_RETURN'].mean() / df['TOTAL_RETURN'].std()) * 
            (df.reset_index().set_index('level_1').index.nunique()),2))
