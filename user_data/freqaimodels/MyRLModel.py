from freqtrade.freqai.prediction_models.ReinforcementLearner import ReinforcementLearner
from freqtrade.freqai.RL.Base3ActionRLEnv import Actions, Base3ActionRLEnv, Positions


class MyRLModel(ReinforcementLearner):
    """
        User created RL prediction model.

        Save this file to `freqtrade/user_data/freqaimodels`

        then use it with:

        freqtrade trade --freqaimodel MyCoolRLModel --config config.json --strategy SomeCoolStrat

        Here the users can override any of the functions
        available in the `IFreqaiModel` inheritance tree. Most importantly for RL, this
        is where the user overrides `MyRLEnv` (see below), to define custom
        `calculate_reward()` function, or to override any other parts of the environment.

        This class also allows users to override any other part of the IFreqaiModel tree.
        For example, the user can override `def fit()` or `def train()` or `def predict()`
        to take fine-tuned control over these processes.

        Another common override may be `def data_cleaning_predict()` where the user can
        take fine-tuned control over the data handling pipeline.
        """
    class MyRLEnv(Base3ActionRLEnv):
        """
        User can override any function in BaseRLEnv and gym.Env. Here the user
        sets a custom reward based on profit and trade duration.
        """
        
        import numpy as np

        def calculate_reward(self, action: int) -> float:
            """
            An updated version of the reward function, incorporating several optimizations.
            :param action: int = The action made by the agent for the current candle.
            :return: float = the reward to give to the agent for current step (used for optimization of weights in NN)
            """
            # Penalize if the action is not valid
            if not self._is_valid(action):
                return -2

            # Define constants
            max_trade_duration = self.rl_config.get('max_trade_duration_candles', 16)
            factor = 1.
            risk_factor = 1.

            # Calculate current PnL and risk
            pnl = self.get_unrealized_profit()
            # drawdown = self.get_max_drawdown()
            # volatility = self.get_volatility()

            # Increase reward for entering trades
            if action == Actions.Buy.value and self._position == Positions.Neutral:
                # Buy low by checking if the current close price is lower than the open price
                if self.prices.iloc[self._current_tick].close < self.prices.iloc[self._current_tick].open:
                    factor = 1.2
                    return 10 * factor
                else:
                    factor = 1.
                    return 7 * factor

            # Penalty for not entering trades uptrend
            if action == Actions.Neutral.value and self._position == Positions.Neutral:
                if self.prices.iloc[self._current_tick].close > self.prices.iloc[self._current_tick].open:
                    return -1

            # Penalize for holding a losing position for too long
            if self._position == Positions.Long and pnl < 0:
                return pnl

            # Increase reward for closing winning trades
            if action == Actions.Sell.value and self._position == Positions.Long:
                # Sell high by checking if the current close price is higher than the open price
                if pnl > self.profit_aim * self.rr:
                    factor = 2.
                if self.prices.iloc[self._current_tick].close > self.prices.iloc[self._current_tick].open:
                    return float(pnl * factor * risk_factor)
                else:
                    return float(pnl * factor * risk_factor / 2)

            # Adjust factor based on trade duration
            if self._last_trade_tick is not None:
                trade_duration = self._current_tick - self._last_trade_tick
                if trade_duration <= max_trade_duration:
                    factor *= 1.2
                else:
                    factor *= 0.8

            # Adjust factor based on market conditions
            # if volatility > 0.1:
            #     risk_factor *= 0.5
            # elif drawdown > 0.1:
            #     risk_factor *= 0.8
            # else:
            #     risk_factor *= 1.

            # Penalty for trading long time
            if self._last_trade_tick is not None and self._current_tick - self._last_trade_tick > max_trade_duration:
                return -1

            return 0.

        
        
        # # # # def calculate_reward(self, action: int) -> float:
        # # # #     """
        # # # #     An example reward function. This is the one function that users will likely
        # # # #     wish to inject their own creativity into.
        # # # #     :param action: int = The action made by the agent for the current candle.
        # # # #     :return: float = the reward to give to the agent for current step (used for optimization of weights in NN)
        # # # #     """
        # # # #     # Penalize if the action is not valid
        # # # #     if not self._is_valid(action):
        # # # #         self.tensorboard_log("is_valid")
        # # # #         return -2

        # # # #     pnl = self.get_unrealized_profit()
        # # # #     factor = 100.

        # # # #     # Increase reward for entering trades
        # # # #     if action == Actions.Buy.value and self._position == Positions.Neutral:
        # # # #         # Buy low by checking if the current close price is lower than the open price
        # # # #         if self.prices.iloc[self._current_tick].close < self.prices.iloc[self._current_tick].open:
        # # # #             return 10
        # # # #         else:
        # # # #             return 7

        # # # #     # Penalty for not entering trades 
        # # # #     if action == Actions.Neutral.value and self._position == Positions.Neutral:
        # # # #         if self.prices.iloc[self._current_tick].close < self.prices.iloc[self._current_tick].open:
        # # # #             return -1

        # # # #     max_trade_duration = self.rl_config.get('max_trade_duration_candles', 300)
        # # # #     if self._last_trade_tick == None:
        # # # #         return 0.
            
        # # # #     trade_duration = self._current_tick - self._last_trade_tick

        # # # #     # Penalize for holding a losing position for too long
        # # # #     if self._position == Positions.Long and pnl < 0:
        # # # #         return pnl * trade_duration

        # # # #     # Increase reward for closing winning trades
        # # # #     if action == Actions.Sell.value and self._position == Positions.Long:
        # # # #         # Sell high by checking if the current close price is higher than the open price
        # # # #         if pnl > self.profit_aim * self.rr:
        # # # #             factor *= self.rl_config['model_reward_parameters'].get('win_reward_factor', 2)
        # # # #         if self.prices.iloc[self._current_tick].close > self.prices.iloc[self._current_tick].open:
        # # # #             return float(pnl * factor)
        # # # #         else:
        # # # #             return float(pnl * factor / 2)

        # # # #     # Adjust factor based on trade duration
        # # # #     if trade_duration <= max_trade_duration:
        # # # #         factor *= 1.2
        # # # #     elif trade_duration > max_trade_duration:
        # # # #         factor *= 0.8

        # # # #     return 0.

        
        # def calculate_reward(self, action: int) -> float:
        #     """
        #     An example reward function. This is the one function that users will likely
        #     wish to inject their own creativity into.
        #     :param action: int = The action made by the agent for the current candle.
        #     :return: float = the reward to give to the agent for current step (used for optimization of weights in NN)
        #     """
        #     # Penalize if the action is not valid
        #     if not self._is_valid(action):
        #         self.tensorboard_log("is_valid")
        #         return -2

        #     pnl = self.get_unrealized_profit()
        #     factor = 100.

        #     # Increase reward for entering trades
        #     if action == Actions.Buy.value and self._position == Positions.Neutral:
        #         return 50

        #     # Decrease penalty for not entering trades
        #     if action == Actions.Neutral.value and self._position == Positions.Neutral:
        #         return -0.5

        #     max_trade_duration = self.rl_config.get('max_trade_duration_candles', 300)
        #     trade_duration = self._current_tick - self._last_trade_tick

        #     # Penalize for holding a losing position for too long
        #     if self._position == Positions.Long and pnl < 0:
        #         return pnl * factor / trade_duration

        #     # Increase reward for closing winning trades
        #     if action == Actions.Sell.value and self._position == Positions.Long:
        #         if pnl > self.profit_aim * self.rr:
        #             factor *= self.rl_config['model_reward_parameters'].get('win_reward_factor', 2)
        #         return float(pnl * factor)

        #     # Adjust factor based on trade duration
        #     if trade_duration <= max_trade_duration:
        #         factor *= 1.2
        #     elif trade_duration > max_trade_duration:
        #         factor *= 0.8

        #     return 0.

        # def calculate_reward(self, action: int) -> float:
        #     """
        #     An example reward function. This is the one function that users will likely
        #     wish to inject their own creativity into.
        #     :param action: int = The action made by the agent for the current candle.
        #     :return: float = the reward to give to the agent for current step (used for optimization of weights in NN)
        #     """
        #     first, penalize if the action is not valid
        #     if not self._is_valid(action):
        #         self.tensorboard_log("is_valid")
        #         return -2

        #     pnl = self.get_unrealized_profit()
        #     factor = 100.

        #     reward agent for entering trades
        #     if (action == Actions.Buy.value
        #             and self._position == Positions.Neutral):
        #         return 25
        #     discourage agent from not entering trades
        #     if action == Actions.Neutral.value and self._position == Positions.Neutral:
        #         return -1

        #     max_trade_duration = self.rl_config.get('max_trade_duration_candles', 300)
        #     trade_duration = self._current_tick - self._last_trade_tick  # type: ignore

        #     if trade_duration <= max_trade_duration:
        #         factor *= 1.5
        #     elif trade_duration > max_trade_duration:
        #         factor *= 0.5

        #     discourage sitting in negative position and hold in positive position
        #     if (self._position == Positions.Long and
        #             action == Actions.Neutral.value):
        #         return pnl * trade_duration / max_trade_duration


        #     close long
        #     if action == Actions.Sell.value and self._position == Positions.Long:
        #         if pnl > self.profit_aim * self.rr:
        #             factor *= self.rl_config['model_reward_parameters'].get('win_reward_factor', 2)                    
        #         return float(pnl * factor)

        #     return 0.


        # # # def calculate_reward(self, action: int) -> float:
        # # #     # first, penalize if the action is not valid
        # # #     if not self._is_valid(action):
        # # #         self.tensorboard_log("is_valid")
        # # #         return -2

        # # #     # calculate profit/loss resulting from the trade
        # # #     initial_position = self._position
        # # #     initial_pnl = self.get_unrealized_profit()
        # # #     self.execute_trade_exit(action)
        # # #     final_pnl = self.get_unrealized_profit()
        # # #     pnl = final_pnl - initial_pnl
        # # #     trade_size = abs(self._position.value - initial_position.value)

        # # #     # calculate the reward
        # # #     if action == Actions.Buy.value and self._position == Positions.Neutral:
        # # #         # encourage agent to enter trades
        # # #         reward = 50 * trade_size
        # # #     elif action == Actions.Sell.value and self._position == Positions.Long:
        # # #         # give higher reward for profitable trades
        # # #         if pnl > 0:
        # # #             reward = pnl * 100 * trade_size
        # # #         else:
        # # #             reward = pnl * 50 * trade_size
        # # #     elif action == Actions.Neutral.value and self._position == Positions.Neutral:
        # # #         # discourage agent from not entering trades
        # # #         reward = -5
        # # #     else:
        # # #         reward = 0

        # # #     # adjust reward based on risk
        # # #     if action == Actions.Buy.value and self._position == Positions.Neutral:
        # # #         # consider the risk of the trade (e.g. using stop-loss)
        # # #         if self.get_stop_loss_hit():
        # # #             reward *= 0.5

        # # #     # adjust reward based on trend-following strategies
        # # #     if action == Actions.Buy.value and self._position == Positions.Neutral:
        # # #         # use technical indicators to identify trends
        # # #         if self.is_trend_up():
        # # #             reward *= 1.5
        # # #         elif self.is_trend_down():
        # # #             reward *= 0.5

        # # #     return reward
