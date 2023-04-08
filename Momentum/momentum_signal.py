import pandas as pd
import numpy as np

def time_signal(df, window_size, rebalancing_period): 
    returns = df.pct_change(fill_method=None)
    momentum = returns.rolling(window = window_size).mean()
    # returns = returns.shift(-1)

    long_signal = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
    long_short_signal = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)

    counter = 0

    # Initialize positive_mom and negative_mom
    first_row = momentum.iloc[0]
    positive_mom = first_row > 0
    negative_mom = first_row < 0

    for date, row in momentum.iterrows():
        if counter % rebalancing_period == 0:
            positive_mom = row > 0
            negative_mom = row < 0

        long_signal.loc[date, :] = np.where(positive_mom, 1, np.nan)
        long_short_signal.loc[date, :] = np.where(negative_mom, 1, long_signal.loc[date, :])

        counter += 1
    
    return long_signal, long_short_signal

def cross_signal(df, window_size, quantile, rebalancing_period):
    returns = df.pct_change(fill_method=None)
    momentum = returns.rolling(window = window_size).mean()

    long_signal = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
    long_short_signal = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)

    counter = 0

    for date, row in momentum[window_size:].iterrows():
        n = int(len(row) * quantile)

        if sum(~row.isna()) < n:
            raise ValueError(f"Number of stocks in the universe is less than {n}.")

        if counter % rebalancing_period == 0:
            top_quantile = row.dropna().nlargest(n).index
            bottom_quantile = row.dropna().nsmallest(n).index

        # TODO: Weighted return
        long_signal.loc[date, :] = np.where(long_signal.columns.isin(top_quantile), 1, np.nan)
        long_short_signal.loc[date, :] = np.where(long_short_signal.columns.isin(bottom_quantile), 1, long_signal.loc[date, :])

        counter += 1
        
    return long_signal, long_short_signal

def dual_signal(df, window_size, quantile, rebalancing_period):
    time_long, time_long_short = time_signal(df, window_size, rebalancing_period)
    cross_long, cross_long_short = cross_signal(df, window_size, quantile, rebalancing_period)

    long_signal = time_long * cross_long
    long_short_signal = time_long_short * cross_long_short

    return long_signal, long_short_signal

def Efficiency_signal(df, window_size, rebalancing_period):
    er = (df - df.shift(window_size)) / abs(df - df.shift()).rolling(window_size).sum()

    long_signal = pd.DataFrame(np.nan, index=df.index, columns=df.columns)
    long_short_signal = pd.DataFrame(np.nan, index=df.index, columns=df.columns)

    counter = 0

    # Initialize positive_mom and negative_mom
    first_row = er.iloc[0]
    positive_mom = first_row > 0
    negative_mom = first_row < 0

    for date, row in er.iterrows():
        if counter % rebalancing_period == 0:
            positive_mom = row > 0
            negative_mom = row < 0

        long_signal.loc[date, :] = np.where(positive_mom, 1, np.nan)
        long_short_signal.loc[date, :] = np.where(negative_mom, 1, long_signal.loc[date, :])

        counter += 1

    return long_signal, long_short_signal