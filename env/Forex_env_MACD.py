import numpy as np

from env.Parent_trade_env import TradingEnv, Actions, Positions, Trend


class ForexEnv_MACD(TradingEnv):

    def __init__(self, df, window_size, frame_bound, unit_side='left'):
        assert len(frame_bound) == 2
        assert unit_side.lower() in ['left', 'right']

        self.frame_bound = frame_bound
        self.unit_side = unit_side.lower()
        super().__init__(df, window_size)

        self.trade_fee = 0.0003  # unit

    def _process_data(self):
        prices = self.df.loc[:, 'Close'].to_numpy()

        prices[self.frame_bound[0] - self.window_size]  # validate index (TODO: Improve validation)
        # getting range of prices in df to look at
        prices = prices[self.frame_bound[0] - self.window_size:self.frame_bound[1]]
        diff = np.insert(np.diff(prices), 0, 0)
        # print(prices.shape)
        # check if it is uptrend or downtrend in the chosen window of 60 minutes
        prices_shift_wind = prices[60:]
        # print("Shifted price", prices_shift_wind[:15])
        for i in range(60):  # filling with 0 for first three entries
            prices_shift_wind = np.insert(prices_shift_wind, -1, 0)
        # print("Zeros appended", prices_shift_wind[:15])
        # print("Prices List", prices[:15])
        trend = prices - prices_shift_wind  # Calculate difference of prices between lag of 3 days
        # print("Calculated trend", trend[:15])
        signal_features = np.column_stack((prices, diff, trend))

        return prices, signal_features

    def _calculate_reward(self, action):
        step_reward = 0  # pip

        trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
                (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        if trade:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]
            price_diff = current_price - last_trade_price

            if self._position == Positions.Short and self.trend < 0:
                step_reward += -price_diff * 10000
            elif self._position == Positions.Long and self.trend >= 0:
                step_reward += price_diff * 10000
            else:
                step_reward -= abs(price_diff * 10000)  # punish for taking wrong decision

        return step_reward

    def _update_profit(self, action):
        trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
                (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True  # validating that the action is completing a pending pair trade

        if trade or self._done:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]

            if self.unit_side == 'left':
                if self._position == Positions.Short:
                    quantity = self._total_profit * (last_trade_price - self.trade_fee)
                    self._total_profit = quantity / current_price

            elif self.unit_side == 'right':
                if self._position == Positions.Long:
                    quantity = self._total_profit / last_trade_price
                    self._total_profit = quantity * (current_price - self.trade_fee)

    def max_possible_profit(self):
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.

        while current_tick <= self._end_tick:
            position = None
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] < self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Short
            else:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] >= self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Long

            current_price = self.prices[current_tick - 1]
            last_trade_price = self.prices[last_trade_tick]

            if self.unit_side == 'left':
                if position == Positions.Short:
                    quantity = profit * (last_trade_price - self.trade_fee)
                    profit = quantity / current_price

            elif self.unit_side == 'right':
                if position == Positions.Long:
                    quantity = profit / last_trade_price
                    profit = quantity * (current_price - self.trade_fee)

            last_trade_tick = current_tick - 1

        return profit
