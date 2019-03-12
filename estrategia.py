import pandas as pd
import numpy as np
import datetime as dt   
import pickle
from backtest import Backtest
import timeit

''' ENTRY THE BACKTEST TO EXECUTE THE PROGRAM '''
Bt = Backtest()

''' STRATEGY NAME, WILL BE USED TO SAVE FILES AND ON DATAFRAME '''
name = 'STOCH'

#################################################################################################

def strategy(df):
    """
    Here you enter the trade idea, where all setup traded will be added to a dictionary
    with all entry and exit

    The idea most be between if-elif-else nested arguments
    """
    strag_l = {}
    strag_s = {}
    db = pd.DataFrame()
    temp = None

    for i in range(len(df.index)):
        '''HERE YOU ADD THE IDEA FOR BUY-LONG''' ####################################
        if df.iloc[i].RSI_mK > df.iloc[i].RSI_mD and df.iloc[i].RSI_5K > df.iloc[i].RSI_5D and df.iloc[i].RSI_5K < 50:
            
            strag_l.update({df.index[i]: (name, df.Asset[i], 'LONG', df.Close[i])})
            temp = df.index[i]
        
        elif temp in strag_l.keys(): ############################# SETUP EXIT #######

            strag_l.update({df.index[i]: (name, df.Asset[i], 'cLONG', df.Close[i])}) 

        else:
            pass

    for i in range(len(df.index)):
        '''HERE YOU ADD THE IDEA FOR SELL-SHORT''' ####################################
        if df.iloc[i].RSI_mK < df.iloc[i].RSI_mD and df.iloc[i].RSI_5K < df.iloc[i].RSI_5D and df.iloc[i].RSI_5K > 50:

            strag_s.update({df.index[i]: (name, df.Asset[i], 'SHORT', df.Close[i])})
            temp = df.index[i]

        
        elif temp in strag_s.keys(): ############################# SETUP EXIT #######

            strag_s.update({df.index[i]: (name, df.Asset[i], 'cSHORT', df.Close[i])}) 

        else:
            pass


    strag_l = pd.DataFrame(strag_l, index=['STRATEGY', 'Asset', 'L_S', 'Price']).T.reset_index()
    strag_s = pd.DataFrame(strag_s, index=['STRATEGY', 'Asset', 'L_S', 'Price']).T.reset_index()
    db = pd.concat([strag_l, strag_s], ignore_index=True).rename({'index':'Date'}, axis=1)


    return Bt.backtest(db,df)


#################################################################################################

def strategy_mul(func, df, backtest_st='NO'):
    ''' THIS MAKE THE TESTING MULTIPROCESS, SO WILL USE ALL YOUR PROCESSOR POWER '''
    
    from multiprocessing import Pool, cpu_count
    num_process = min(df.shape[1], cpu_count())

    with Pool(num_process) as pool:
        seq = [df[df['Asset'] == i] for i in df.Asset.unique()]

        results_list = pool.map(func, seq)

        db = pd.concat(results_list)
        
    if backtest_st.upper() == 'YES':
        db_st = Bt.backtest_st(db, df).groupby(level=[0,1]).sum()
        print(db_st.groupby(level=[0,1]).sum().head())

        if 'Total_Result' in df.columns:
            perc = 'Total_Resul'
        else:
            perc = 'Change%'

        print('\n Per Asset_Year ... RETURN -> ', 
            round((sum([db.loc[i, :][perc].mean() for i in db.index.unique()]) / db.index.year.nunique()),2), 
            '& Win_X_Lose ->', round((db_st.groupby(level=[0,1]).mean().mean().Win_X_Lose),2), 
            '& RISK_RETURN ->', round((db_st.groupby(level=[0,1]).mean().mean().RISK_RETURN),2),
            '& SHARPE_RATIO ->', round((db[perc].mean() / db[perc].std()) * (db.index.nunique()**0.5),2))

        with open('./Data/DF/BACKTEST_ST.pickle', 'wb') as f:
            pickle.dump(db_st, f, protocol=pickle.HIGHEST_PROTOCOL)

        db_st.to_csv(f'./Data/DF/{name}_backtest_st.csv')


    with open('./Data/DF/BACKTEST.pickle', 'wb') as f:
        pickle.dump(db, f, protocol=pickle.HIGHEST_PROTOCOL)
            
    db.to_csv(f'./Data/DF/{name}_backtest.csv')


    print(f'Qty of Assets -> {db.Asset.nunique()} \n')
    return db.head(3)

#################################################################################################

df = pd.read_pickle('./Data/DF/INDICADOR.pickle')

''' FILTRE FOR THE DATA TO BE TESTED, YOU CAN FILTRE BY DATE AND ASSETS '''
# df = df[df.index > '2017-10-1']
# df = df[(df.Asset == 'PETR4') | (df.Asset == 'BOVA11') | (df.Asset == 'VALE3')]

''' IN CASE INDICATORS ARE REQUIRED AND NOT IN THE DATA ADD BELOW'''
''' BELOW EXAMPLE OF POSSIBLE INDICATOR '''
# df['SMA7'] = df.Close.rolling(7).mean()


start_time = timeit.default_timer()

print(strategy_mul(strategy, df, backtest_st='YES'))

print(f'\n Time for this operation: ', timeit.default_timer() - start_time)


