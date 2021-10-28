import pandas as pd
import numpy as np
import gym
from finta import TA
import quantstats as qs

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C

from env.Parent_trade_env import TradingEnv
from env.Forex_1day import ForexEnv_1day, Actions, Positions
import matplotlib.pyplot as plt

data_path = "data/USD_CHF/USDCHF1440.csv"

def get_data_frame():
    USDCHF1440_df = pd.read_csv(data_path, sep='\t', header=None)
    USDCHF1440_df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    # USDCHF1440_df['SMA'] = TA.SMA(USDCHF1440_df, 10)
    # USDCHF1440_df['RSI'] = TA.RSI(USDCHF1440_df)
    # USDCHF1440_df['OBV'] = TA.OBV(USDCHF1440_df)
    # MACD_df = TA.MACD(USDCHF1440_df)
    # USDCHF1440_df['MACD'] = MACD_df["MACD"]
    # USDCHF1440_df['SIGNAL'] = MACD_df["SIGNAL"]
    # USDCHF1440_df.fillna(0, inplace=True)
    USDCHF1440_df['Date'] = pd.to_datetime(USDCHF1440_df['Date'])
    USDCHF1440_df.set_index("Date", inplace=True)
    yield USDCHF1440_df


# def my_process_data(env):
#     # Adding custom signals
#     start = env.frame_bound[0] - env.window_size
#     end = env.frame_bound[1]
#     prices = env.df.loc[:, 'Low'].to_numpy()[start:end]
#     signal_features = env.df.loc[:, ['SMA', 'RSI', 'OBV', 'MACD', 'SIGNAL']].to_numpy()[start:end]
#     return prices, signal_features
#

# class MyForexEnv(ForexEnv):
    # _process_data = my_process_data


print("Training")
df_int = get_data_frame().__next__()
#Leave out last 1000 days for testing
df_train = df_int[:-1000]
env2 = ForexEnv_1day(df=df_train, window_size=12, frame_bound=(12, len(df_train)))
env_maker = lambda: env2
env = DummyVecEnv([env_maker])
model = A2C('MlpLstmPolicy', env, verbose=1) #MlpLstmPolicy is available only in stable_baselines not in 3
model.learn(total_timesteps=6000)

#  Evaluation
print("Evaluating")
# df_int=get_data_frame().__next__()
# Evaluating for 60 days
df_test = df_int[-1000:]

env = ForexEnv_1day(df=df_test, window_size=12, frame_bound=(12, len(df_test)))

observation = env.reset()
while True:
    observation = observation[np.newaxis, ...] #increases dimension
    action, _states = model.predict(observation)
    # action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    # print("observation:",observation)
    # print("reward:",reward)
    # print("done:",done)
    # print("info:", info)
    # env.render()
    if done:
        print("info:", info)
        break
# plt.figure(figsize=(15, 10))
# plt.cla()
# env.render_all()
# plt.show()
# Calling quantstats to analyse the result
qs.extend_pandas()

# Start of index is window size +1
net_worth = pd.Series(env.history['total_profit'], index=df_test.index[13:len(df_test)])
returns = net_worth.pct_change().iloc[1:]

# Generate reports with quatstats
qs.reports.full(returns)

qs.reports.html(returns, output='one_day.html')