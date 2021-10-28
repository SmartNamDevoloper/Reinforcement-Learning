import datetime
import threading
from stable_baselines import A2C
from env.Forex_1day import ForexEnv_1day, Actions, Positions
import numpy as np
from ib_insync import *
import pandas as pd
import time

sell_count = 0
buy_count = 0
trade_price = 0
latest_price = 0
reference_date = None
trade_date = None
ib = None
order_id = 1
first_time = True


# util.startLoop()  # uncomment this line when in a notebook
#  Step1 Download IB Gateway Client (trading window app)
# Check the box for Enable ActiveX and socket clients check the port is matching uncheck readonly


def onBarUpdate(bars, hasNewBar):
    global reference_date
    global sell_count
    global buy_count
    global latest_price
    global trade_price
    global trade_date
    global first_time
    window_for_trend_calc = 10

    # convert to pandas dataframe:
    print("*" * 50)
    print("Got a new Bar of data ..........")
    df = util.df(bars)
    print(df.head())
    print(df.tail())

    df = df.drop(["average", "barCount"], axis=1)
    # df.date = df.date.apply(lambda x: x.split(" ")[0])
    latest_price = df.close.values[-1]
    print("Latest Price: ", latest_price)
    latest_date = str(df.tail(1).date).split(" ")[3:5]
    latest_date = latest_date[0] + " " + latest_date[-1].split("\n")[0]
    latest_date = pd.to_datetime(latest_date)
    print("Latest records date and time is:", latest_date)
    print("Time lapsed:  ", datetime.datetime.now() - latest_date)
    print("Reference date", reference_date)
    print("Time lapsed between reference and latest:  ", latest_date - reference_date)
    # Making profit in long position if profit is >.0002 or holding for the past 1 hour
    if buy_count:
        print("Buy Count is > 0 ")
        profit = round(latest_price / trade_price, 5)
        print("Profit is: ", profit)
        if profit > 1.0002:
            print("Booking profit in Long position")
            # if trade_date - latest_date > pd.Timedelta("P0DT01H0M0S"):
            order = do_trade(Actions.Sell, latest_price)
            trade = ib.placeOrder(contract, order)
            sell_count -= 1
            buy_count -= 1
            print("Trade", trade)
    # Making profit in short position if profit is >.0002 or holding for the past 1 hour
    elif sell_count:
        print("Sell Count is > 0 ")
        profit = round(trade_price / latest_price, 5)
        print("Profit is: ", profit)
    if profit > 1.0002:
        print("Booking profit in Short position")
        # if trade_date - latest_date > pd.Timedelta("P0DT01H0M0S"):
        order = do_trade(Actions.Buy, latest_price)
        trade = ib.placeOrder(contract, order)
        sell_count -= 1
        buy_count -= 1
        print("Trade", trade)
    # if latest_date - reference_date < pd.Timedelta("P0DT0H15M0S") or df.shape[0] < 2 * window_for_trend_calc:
    if latest_date - reference_date > pd.Timedelta("P0DT1H0M0S") or first_time:
        # Prediction with market data
        print("Predicting")
        # first_time = False
        reference_date = latest_date
        df_pred = df
        # df_pred=df[-window_for_trend_calc:]
        print("Predicting with last 15 minutes data")
        print(df_pred.info())
        print(df_pred.shape)
        df_pred.set_index("date", inplace=True)
        df_pred.columns = ["Open", "High", "Low", "Close", "Volume"]
        latest_price = df.Close.values[-1]
        print("latest price updtaed: ", latest_price)
        env = ForexEnv_1day(df=df_pred, window_size=window_for_trend_calc,
                            frame_bound=(window_for_trend_calc, len(df_pred)), trend_window=window_for_trend_calc)

        # load the model
        model = A2C.load('./results_hour_new/model_group_by_hour20000steps_10win')

        observation = env.reset()  # this has to be after loading model
        print("Observation", observation)
        print("_" * 100)
        print("Position Before Prediction ", env._position)
        print("_" * 100)
        observation = observation[np.newaxis, ...]  # increases dimension
        print("Observation", observation)
        action, _states = model.predict(observation)
        print("_" * 100)
        print("Position After Prediction ", env._position)
        print("_" * 100)
        print("Position History", env._position_history)
        print("*" * 50)
        print("Action predicted is:  ", action)
        #  TBD the position is short in th bot but live trade may not have taken position
        # if ((action == Actions.Buy.value and env._position == Positions.Short) or
        #         (action == Actions.Sell.value and env._position == Positions.Long)):

        trade_date = latest_date
        # order = do_trade(action, df_pred)  # If doing trade according to action
        order = do_trade(env._position, latest_price)  # If doing trade according to position
        print("Order details: ", order)
        trade = ib.placeOrder(contract, order)

        print("Trade", trade)

    else:
        print("Inserted less than 1 hour ago or still accumulating data")


def do_trade(action, price):
    print("Create order for TRADING")
    print("_" * 100)
    global buy_count
    global sell_count
    global trade_price
    global ib
    global order_id
    global first_time
    order = None
    # if action == Actions.Sell.value and sell_count == 0:  # allowing only one trade to happen at a trading session
    print("Action: ", action, "and Sell Count: ", sell_count)
    # This ensures we dont have to wait till the next 1 hour data arrives

    if (action == Positions.Short or action == Actions.Sell) and sell_count == 0:
        print("Creating a sell order")
        sell_count += 1
        # TBD the trade price is assumed as the latest price retrieved
        # Has to be changed to market price later
        trade_price = price
        print("Trade Price: ", trade_price)
        # Create order object
        order = MarketOrder('SELL', 100000)
        # order.orderId = order.get
        # order.action = 'SELL'
        # order.totalQuantity = 100000
        # order.orderType = 'MARKET'
        print("*" * 50)
        print("Placing a Sell order")
        print("*" * 50)

    # elif action == Actions.Buy.value and buy_count == 0:  # allowing only one trade to happen at a trading session
    elif (action == Positions.Long or action == Actions.Buy) and buy_count == 0:
        print("Creating a buy order")
        buy_count += 1
        trade_price = price
        print("Trade Price: ", trade_price)
        # # Create order object
        order = Order()
        order.action = 'BUY'
        order.totalQuantity = 100000
        order.orderType = 'MKT'
        print("*" * 50)
        print("Placing a buy order")
        print("*" * 50)
    if first_time:
        print("Changing first time to False")
        first_time = False
    else:
        print("Changing first time to True")
        first_time = True
    return order


# def has_live_threads(threads):
#     return True in [t.isAlive() for t in threads]


# def main():
#     global reference_date
#     global sell_count
#     global buy_count
#     global trade_price
#     global latest_price
#     global ib
#     print("Main")

ib = IB()
print("*" * 50)
print("Connecting to IB..............")
print("*" * 50)
ib.connect('127.0.0.1', 7497, clientId=1)
print("Succesfully Connected to IB")
contract = Forex('USDCHF')
print("*" * 50)
print("Getting data from IB..........")
print("*" * 50)
bars = ib.reqHistoricalData(
    contract, endDateTime='', durationStr='2 D',
    barSizeSetting='1 hour', whatToShow='MIDPOINT', useRTH=True, keepUpToDate=True)
df = util.df(bars)
print("*" * 50)
print("Got data from Interactive broker")
print("*" * 50)
print(df.head())
print(df.tail())
reference_date = str(df.tail(1).date).split(" ")[3:5]
reference_date = reference_date[0] + " " + reference_date[-1].split("\n")[0]
reference_date = pd.to_datetime(reference_date)
print("Latest records date and time is:", reference_date)

if sell_count:
    print("Closing a Short session")
    order = Order()
    order.action = 'BUY'
    order.totalQuantity = 100000
    order.orderType = 'MARKET'
    print("*" * 50)
    print("Placing a buy order to close session")
    print("*" * 50)
    trade = ib.placeOrder(contract, order)
    print("Trade", trade)
    profit = trade_price - latest_price
    print('Interrupted')
elif buy_count:
    print("Closing a Short session")
    order = Order()
    order.action = 'BUY'
    order.totalQuantity = 100000
    order.orderType = 'MARKET'
    print("*" * 50)
    print("Placing a buy order to close session")
    print("*" * 50)
    trade = ib.placeOrder(contract, order)
    print("Trade", trade)
    profit = latest_price - trade_price
    print('Interrupted')

bars.updateEvent += onBarUpdate
ib.sleep(900)

# if __name__ == "__main__":
#     main()
