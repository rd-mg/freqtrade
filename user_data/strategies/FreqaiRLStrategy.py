import logging
from functools import reduce
import math
import numpy as np

import talib.abstract as ta
from pandas import DataFrame
import pandas_ta
from technical import qtpylib
from technical.indicators import laguerre


from freqtrade.strategy import CategoricalParameter, IStrategy, informative


logger = logging.getLogger(__name__)


class FreqaiRLStrategy(IStrategy):
    """
    Example strategy showing how the user connects their own
    IFreqaiModel to the strategy. Namely, the user uses:
    self.freqai.start(dataframe, metadata)

    to make predictions on their data. feature_engineering_*() automatically
    generate the variety of features indicated by the user in the
    canonical freqtrade configuration file under config['freqai'].
    """
    
    plot_config = {
        'main_plot': {
            'bb_upper_band': {'color': 'blue'},
            'bb_middle_band': {'color': 'brown'},
            'bb_lower_band': {'color': 'blue'},
            'ema': {'color': 'white'},
            'ema-9': {'color': 'red'},
        },
        'subplots': {
            "laguerre": {
                'laguerre': {'color': 'black'},
            },
            "macd": {
                'macd': {'color': 'green'},
            },
            "stochrsi_fastk": {
                'STOCHRSIk_9_9_3_3': {'color': 'blue'},
            },
            "ema_slope": {
                'ema_slope': {'color': 'red'},
            },
            "&-action": {
                '& -action': {"color": "blue"}
            },
            "do_predict": {
                'do_predict': {"color": "brown"}
                },
        },
    }


    process_only_new_candles = True

    #timeframe
    timeframe = '15m'

    # ROI table:
    minimal_roi = {
    "0": 0.11900000000000001,
    "102": 0.049,
    "249": 0.018,
    "599": 0
    }
    # Stoploss:
    stoploss = -0.336
    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.319
    trailing_stop_positive_offset = 0.401
    trailing_only_offset_is_reached = True

    use_exit_signal = True
    # this is the maximum period fed to talib (timeframe independent)
    startup_candle_count: int = 300
    can_short = False
    
    @informative('1h')
    @informative('4h')
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.ta.stochrsi(length=9, rsi_length=9, k=3, d=3, append=True)
        # macd = ta.MACD(dataframe, fastperiod=5, slowperiod=9, signalperiod=9)
        # # dataframe['macd'] = macd['macd']
        # # dataframe['macdsignal'] = macd['macdsignal']
        # dataframe['macdhist'] = macd['macdhist']
        # dataframe["ema"] = ta.EMA(dataframe, timeperiod=2)
        # dataframe["ema-signal"] = ta.EMA(dataframe, timeperiod=9)
        dataframe["laguerre"] = laguerre(dataframe, gamma=0.25, smooth=1)
        return dataframe

    def feature_engineering_expand_all(self, dataframe, period, **kwargs):
        """
        *Only functional with FreqAI enabled strategies*
        This function will automatically expand the defined features on the config defined
        `indicator_periods_candles`, `include_timeframes`, `include_shifted_candles`, and
        `include_corr_pairs`. In other words, a single feature defined in this function
        will automatically expand to a total of
        `indicator_periods_candles` * `include_timeframes` * `include_shifted_candles` *
        `include_corr_pairs` numbers of features added to the model.

        All features must be prepended with `%` to be recognized by FreqAI internals.

        More details on how these config defined parameters accelerate feature engineering
        in the documentation at:

        https://www.freqtrade.io/en/latest/freqai-parameter-table/#feature-parameters

        https://www.freqtrade.io/en/latest/freqai-feature-engineering/#defining-the-features

        :param df: strategy dataframe which will receive the features
        :param period: period of the indicator - usage example:
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)
        """
        slow = period+4
        macd = ta.MACD(dataframe, fastperiod=period, slowperiod=slow, signalperiod=period)
        # dataframe[f"%-macd{period}"] = macd['macd']
        # dataframe[f"%-macd_macdsignal{period}"] = macd['macdsignal']
        dataframe[f"%-macdhist-{period}"] = macd['macdhist']
        dataframe[f"%-rsi-{period}"] = ta.RSI(dataframe, timeperiod=period)
        # dataframe["%-mfi-period{period}"] = ta.MFI(dataframe, timeperiod=period)
        dataframe[f"%-cci-{period}"] = ta.CCI(dataframe, timeperiod=period)
        dataframe[f"%-adx-{period}"] = ta.ADX(dataframe, timeperiod=period)
        # dataframe["%-sma-{period}"] = ta.SMA(dataframe, timeperiod=period)
        dataframe[f"%-ema-{period}"] = ta.EMA(dataframe, timeperiod=period)
        # StochRSI = ta.STOCHRSI(dataframe, timeperiod=period, fastk_period=3, fastd_period=3)
        # dataframe[f"%-stochrsi-k_{period}"] = StochRSI.fastk
        # dataframe[f"%-stochrsi-d_{period}"] = StochRSI.fastd


        # bollinger = qtpylib.bollinger_bands(
        #     qtpylib.typical_price(dataframe), window=period, stds=2.2
        # )
        # dataframe["bb_lowerband-period"] = bollinger["lower"]
        # dataframe["bb_middleband-period"] = bollinger["mid"]
        # dataframe["bb_upperband-period"] = bollinger["upper"]

        # dataframe["%-bb_width-period"] = (
        #     dataframe["bb_upperband-period"]
        #     - dataframe["bb_lowerband-period"]
        # ) / dataframe["bb_middleband-period"]
        # dataframe["%-close-bb_lower-period"] = (
        #     dataframe["close"] / dataframe["bb_lowerband-period"]
        # )

        # dataframe["%-roc-period"] = ta.ROC(dataframe, timeperiod=period)

        # dataframe["%-relative_volume-period"] = (
        #     dataframe["volume"] / dataframe["volume"].rolling(period).mean()
        # )
        
        ### SuperTrend indicators ###
        t=period
        dataframe[f"atr_{t}"] = ta.ATR(
                dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=t)
        dataframe[f"base_{t}"] = (dataframe['high'] + dataframe['low']) / 2
        dataframe[f"base_ema_{t}"] = ta.EMA(dataframe[f"base_{t}"], timeperiod=t)

        # Calculate the SuperTrend indicator
        dataframe[f"%-up{t}"] = dataframe[f"base_ema_{t}"] + (2 * dataframe[f"atr_{t}"])
        dataframe[f"%-down{t}"] = dataframe[f"base_ema_{t}"] - (2 * dataframe[f"atr_{t}"])
        # dataframe[f"%--supertrend_{t}"] = np.where(
        #         dataframe['close'] > dataframe[f"base_sma_{t}"], up, down)

        # # Buy and sell signals
        # dataframe[f"%--long_signal_{t}"] = np.where(
        #         dataframe[f"%--supertrend_{t}"] > dataframe['close'], 1, 0)
        # dataframe[f"%--short_signal_{t}"] = np.where(
        #         dataframe[f"%--supertrend_{t}"] < dataframe['close'], -1, 0)

        ### HMA INDICATOR ###
        # Calculate the square root of the period
        sqrt_period = math.sqrt(t)

        # Calculate the weighted moving average using the square root of the period
        dataframe[f"%-wma_sqrt_period_{t}"] = dataframe['close'].rolling(
                window=t, min_periods=t).apply(lambda x: (x * sqrt_period).mean(), raw=True)

        # Calculate the second weighted moving average using half the period of the first one
        half_period = t // 2
        dataframe[f"%-wma_half_period_{t}"] = dataframe[f"%-wma_sqrt_period_{t}"].rolling(
                window=int(half_period), min_periods=int(half_period)).mean()

        # # Calculate the Hull Moving Average
        # dataframe[f"%--HMA_{t}"] = dataframe[f"-wma_sqrt_period_{t}"] - \
        #     dataframe[f"-wma_half_period_{t}"]

        return dataframe

    def feature_engineering_expand_basic(self, dataframe, **kwargs):
        """
        *Only functional with FreqAI enabled strategies*
        This function will automatically expand the defined features on the config defined
        `include_timeframes`, `include_shifted_candles`, and `include_corr_pairs`.
        In other words, a single feature defined in this function
        will automatically expand to a total of
        `include_timeframes` * `include_shifted_candles` * `include_corr_pairs`
        numbers of features added to the model.

        Features defined here will *not* be automatically duplicated on user defined
        `indicator_periods_candles`

        All features must be prepended with `%` to be recognized by FreqAI internals.

        More details on how these config defined parameters accelerate feature engineering
        in the documentation at:

        https://www.freqtrade.io/en/latest/freqai-parameter-table/#feature-parameters

        https://www.freqtrade.io/en/latest/freqai-feature-engineering/#defining-the-features

        :param df: strategy dataframe which will receive the features
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-ema-200"] = ta.EMA(dataframe, timeperiod=200)
        """
        # dataframe["%-ema-200"] = ta.EMA(dataframe, timeperiod=200)
        # dataframe["%-pct-change"] = dataframe["close"].pct_change()
        # dataframe["%-raw_volume"] = dataframe["volume"]
        # dataframe["%-raw_price"] = dataframe["close"]
        return dataframe

    def feature_engineering_standard(self, dataframe, **kwargs):
        # The following features are necessary for RL models
        dataframe[f"%-raw_close"] = dataframe["close"]
        dataframe[f"%-raw_open"] = dataframe["open"]
        dataframe[f"%-raw_high"] = dataframe["high"]
        dataframe[f"%-raw_low"] = dataframe["low"]
        dataframe["%-raw_volume"] = dataframe["volume"]

        
        # dataframe["%-day_of_week"] = (dataframe["date"].dt.dayofweek + 1) / 7
        # dataframe["%-hour_of_day"] = (dataframe["date"].dt.hour + 1) / 25
        
        # dataframe[f"%-ema-period"] = ta.EMA(dataframe)

        # # ### SuperTrend indicators ###
        # # dataframe[f"atr"] = ta.ATR(
        # #         dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=9)
        # # dataframe[f"base"] = (dataframe['high'] + dataframe['low']) / 2
        # # dataframe[f"base_ema"] = ta.EMA(
        # #         dataframe[f"base"], timeperiod=9)

        # # # Calculate the SuperTrend indicator
        # # up = dataframe[f"base_ema"] + (2 * dataframe[f"atr"])
        # # down = dataframe[f"base_ema"] - (2 * dataframe[f"atr"])
        # # dataframe[f"%--supertrend"] = np.where(
        # #         dataframe['close'] > dataframe[f"base_ema"], up, down)

        # # # Buy and sell signals
        # # dataframe[f"%--long_signal"] = np.where(
        # #         dataframe[f"%--supertrend"] > dataframe['close'], 1, 0)
        # # dataframe[f"%--short_signal"] = np.where(
        # #         dataframe[f"%--supertrend"] < dataframe['close'], -1, 0)

        # # ### HMA INDICATOR ###
        # # # Calculate the square root of the period
        # # sqrt_period = math.sqrt(9)

        # # # Calculate the weighted moving average using the square root of the period
        # # dataframe[f"-wma_sqrt_period"] = dataframe['close'].rolling(
        # #         window=9, min_periods=9).apply(lambda x: (x * sqrt_period).mean(), raw=True)

        # # # Calculate the second weighted moving average using half the period of the first one
        # # half_period = 5
        # # dataframe[f"-wma_half_period"] = dataframe[f"-wma_sqrt_period"].rolling(
        # #         window=int(half_period), min_periods=int(half_period)).mean()

        # # # Calculate the Hull Moving Average
        # # dataframe[f"%--HMA"] = dataframe[f"-wma_sqrt_period"] - \
        # #     dataframe[f"-wma_half_period"]
        
        return dataframe

    def set_freqai_targets(self, dataframe, **kwargs):
        """
        *Only functional with FreqAI enabled strategies*
        Required function to set the targets for the model.
        All targets must be prepended with `&` to be recognized by the FreqAI internals.

        More details about feature engineering available:

        https://www.freqtrade.io/en/latest/freqai-feature-engineering

        :param df: strategy dataframe which will receive the targets
        usage example: dataframe["&-target"] = dataframe["close"].shift(-1) / dataframe["close"]
        """
        # For RL, there are no direct targets to set. This is filler (neutral)
        # until the agent sends an action.
        dataframe["&-action"] = 0

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # All indicators must be populated by feature_engineering_*() functions

        # the model will return all labels created by user in `feature_engineering_*`
        # (& appended targets), an indication of whether or not the prediction should be accepted,
        # the target mean/std values for each of the labels created by user in
        # `set_freqai_targets()` for each training period.

        dataframe = self.freqai.start(dataframe, metadata, self)
        
        dataframe.ta.stochrsi(length=9, rsi_length=9, k=3, d=3, append=True)
        
        dataframe["laguerre"] = laguerre(dataframe, gamma=0.25, smooth=1)

        # for val in self.std_dev_multiplier_buy.range:
        #     dataframe[f'target_roi_{val}'] = (
        #         dataframe["&-s_close_mean"] + dataframe["&-s_close_std"] * val
        #         )
        # for val in self.std_dev_multiplier_sell.range:
        #     dataframe[f'sell_roi_{val}'] = (
        #         dataframe["&-s_close_mean"] - dataframe["&-s_close_std"] * val
        #         )
        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        enter_long_conditions = [df["do_predict"] == 1,
                                df["&-action"] == 1,
                                # (df['STOCHRSIk_9_9_3_3_1h'] > df['STOCHRSIk_9_9_3_3_1h'].shift(1)),
                                # (df['laguerre_1h'] > df['laguerre_1h'].shift(1)),
                                # (df['STOCHRSIk_9_9_3_3_4h'] > df['STOCHRSIk_9_9_3_3_4h'].shift(1)),
                                (df['laguerre_4h'] > df['laguerre_4h'].shift(1)) 
                                ]

        if enter_long_conditions:
            df.loc[reduce(lambda x, y: x & y, enter_long_conditions), "enter_long"] = 1
            
        # enter_short_conditions = [df["do_predict"] == 1, df["&-action"] == 3]

        # if enter_short_conditions:
        #     df.loc[
        #         reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]
        #     ] = (1, "short")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        exit_long_conditions = [df["do_predict"] == 1,
                                df["&-action"] == 2]
        if exit_long_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_long_conditions), "exit_long"] = 1

        # exit_short_conditions = [df["do_predict"] == 1, df["&-action"] == 4]
        # if exit_short_conditions:
        #     df.loc[reduce(lambda x, y: x & y, exit_short_conditions), "exit_short"] = 1


        return df

    # def get_ticker_indicator(self):
    #     return int(self.config["timeframe"][:-1])

    # def confirm_trade_entry(
    #     self,
    #     pair: str,
    #     order_type: str,
    #     amount: float,
    #     rate: float,
    #     time_in_force: str,
    #     current_time,
    #     entry_tag,
    #     side: str,
    #     **kwargs,
    # ) -> bool:

    #     df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
    #     last_candle = df.iloc[-1].squeeze()

    #     if side == "long":
    #         if rate > (last_candle["close"] * (1 + 0.0025)):
    #             return False
    #     else:
    #         if rate < (last_candle["close"] * (1 - 0.0025)):
    #             return False

    #     return True
