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

        def calculate_reward(self, action: int) -> float:
            """
            An example reward function. This is the one function that users will likely
            wish to inject their own creativity into.
            :param action: int = The action made by the agent for the current candle.
            :return:
            float = the reward to give to the agent for current step (used for optimization
                of weights in NN)
            """
            # first, penalize if the action is not valid
            if not self._is_valid(action):
                self.tensorboard_log("is_valid")
                return -2

            pnl = self.get_unrealized_profit()
            factor = 100.

            # reward agent for entering trades
            if (action == Actions.Buy.value
                    and self._position == Positions.Neutral):
                return 25
            # discourage agent from not entering trades
            if action == Actions.Neutral.value and self._position == Positions.Neutral:
                return -1

            max_trade_duration = self.rl_config.get('max_trade_duration_candles', 300)
            trade_duration = self._current_tick - self._last_trade_tick  # type: ignore

            if trade_duration <= max_trade_duration:
                factor *= 1.5
            elif trade_duration > max_trade_duration:
                factor *= 0.5

            # discourage sitting in position
            if (self._position == Positions.Long and
                    action == Actions.Neutral.value):
                return -1 * trade_duration / max_trade_duration

            # close long
            if action == Actions.Sell.value and self._position == Positions.Long:
                if pnl > self.profit_aim * self.rr:
                    factor *= self.rl_config['model_reward_parameters'].get('win_reward_factor', 2)
                return float(pnl * factor)

            return 0.
