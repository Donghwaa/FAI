import pandas as pd

def cap_weight(cap):
    cap_weight = cap.div(cap.sum(axis = 1), axis = 0)
    return cap_weight

def cost(df, rebalancing_period):
    cost = pd.DataFrame(0, index=df.index, columns=df.columns)
    counter = 0
    for date, _ in df.iterrows():
        if counter % rebalancing_period == 0:
            cost.loc[date, :] = 0.0005
        counter += 1
    return cost