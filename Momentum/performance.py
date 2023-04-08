import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def performance(df, cap, weight, signal, rebalancing_period, costs, risk_free_rate  = 0.02):
    daily_return = df.pct_change(fill_method = None)
    cap_weight = cap.div(cap.sum(axis = 1), axis = 0)
    
    cost = pd.DataFrame(0, index=df.index, columns=df.columns)
    counter = 0
    for date, _ in df.iterrows():
        if counter % rebalancing_period == 0:
            cost.loc[date, :] = costs
        counter += 1
    
    if weight == 'ew': 
        portfolio_return = ((daily_return - cost) * signal).mean(axis = 1, skipna = True)
    if weight ==  'cw': 
        portfolio_return = ((daily_return - cost) * cap_weight * signal).sum(axis = 1, skipna = True)

    cumulative_return = (1 + portfolio_return).cumprod()
    cum_return = cumulative_return[-1]

    n = len(daily_return) / 252
    CAGR = cumulative_return[-1] ** (1 / n) - 1
    
    volatility = portfolio_return.std() * np.sqrt(252)
    sharpe_ratio = (CAGR - risk_free_rate) / volatility

    peak = cumulative_return.cummax()
    draw_down = (cumulative_return - peak) / peak
    mdd = draw_down.min()
    
    print(f'Cum Return : {cum_return:.3f}', f'CAGR = {CAGR:.3f}', 
          f'Sharpe Ratio : {sharpe_ratio:.3f}', f'MDD : {mdd:.3f}')

    return  round(cum_return,3), round(CAGR,3), round(sharpe_ratio,3), round(mdd,3)

def bm_performance(bm, risk_free_rate  = 0.02):
    bm_return = bm.pct_change(fill_method = None)
    bm_cumulative_return = (1 + bm_return).cumprod()
    cum_return = bm_cumulative_return[-1]

    n = len(bm_return) / 252
    CAGR = cum_return ** (1 / n) - 1

    volatility = bm_cumulative_return.std() * np.sqrt(252)
    sharpe_ratio = (CAGR - risk_free_rate) / volatility

    bm_peak = bm_cumulative_return.cummax()
    bm_draw_down = ((bm_cumulative_return - bm_peak) / bm_peak)
    mdd = bm_draw_down.min()
    
    print(f'Cum Return : {cum_return:.3f}', f'CAGR = {CAGR:.3f}', 
          f'Sharpe Ratio : {sharpe_ratio:.3f}', f'MDD : {mdd:.3f}')
    
    return  round(cum_return,3), round(CAGR,3), round(sharpe_ratio,3), round(mdd,3)

def plot_performance(df, cap, weight, bm, signal, strategy_name):
    daily_return = df.pct_change(fill_method = None)
    cap_weight = cap.div(cap.sum(axis = 1), axis = 0)
    
    if weight == 'ew': 
        portfolio_return = (daily_return * signal).mean(axis = 1, skipna = True)
    if weight ==  'cw': 
        portfolio_return = (daily_return * cap_weight * signal).sum(axis = 1, skipna = True)
    
    cumulative_return = (1 + portfolio_return).cumprod() * 100

    bm_return = bm.pct_change(fill_method = None)
    bm_cumulative_return = (1 + bm_return).cumprod() * 100

    peak = cumulative_return.cummax()
    draw_down = ((cumulative_return - peak) / peak) * 100

    bm_peak = bm_cumulative_return.cummax()
    bm_draw_down = ((bm_cumulative_return - bm_peak) / bm_peak) * 100

    plt.subplot(2,1,1)
    plt.title(f'{strategy_name}_Cumulative_Return')
    plt.plot(cumulative_return, label = strategy_name)
    plt.plot(bm_cumulative_return, ':', label = 'bench_mark', color = 'gray')
    plt.ylabel('(%)', loc = 'top')
    plt.grid()
    plt.legend()

    plt.subplot(2,1,2)
    plt.title('Maximum_Draw_Down')
    plt.plot(draw_down, label = 'MDD')
    plt.plot(bm_draw_down, ':', label = 'bench_mark', color = 'gray')
    plt.ylabel('(%)', loc = 'top')
    plt.grid()
    plt.legend()

    plt.tight_layout()

    return plt.show()