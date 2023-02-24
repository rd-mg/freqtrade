import logging

import numpy as np
import pandas as pd
import pandas_ta as pta
import talib.abstract as ta
import talib


import math
from pandas import DataFrame
from technical import qtpylib
from technical.indicators import laguerre

from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal
from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy, merge_informative_pair, informative


logger = logging.getLogger(__name__)


class FreqaiHybridStrategySpot_simplified(IStrategy):
    """
    Example of a hybrid FreqAI strat, designed to illustrate how a user may employ
    FreqAI to bolster a typical Freqtrade strategy.

    Launching this strategy would be:

    freqtrade trade --strategy FreqaiExampleHyridStrategy --strategy-path freqtrade/templates
    --freqaimodel CatboostClassifier --config config_examples/config_freqai.example.json

    or the user simply adds this to their config:

    "freqai": {
        "enabled": true,
        "purge_old_models": true,
        "train_period_days": 15,
        "identifier": "uniqe-id",
        "feature_parameters": {
            "include_timeframes": [
                "3m",
                "15m",
                "1h"
            ],
            "include_corr_pairlist": [
                "BTC/BUSD",
                "ETH/BUSD"
            ],
            "label_period_candles": 20,
            "include_shifted_candles": 2,
            "DI_threshold": 0.9,
            "weight_factor": 0.9,
            "principal_component_analysis": false,
            "use_SVM_to_remove_outliers": true,
            "indicator_periods_candles": [10, 20]
        },
        "data_split_parameters": {
            "test_size": 0,
            "random_state": 1
        },
        "model_training_parameters": {
            "n_estimators": 800
        }
    },

    """

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "emergency_exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": True,
        "stoploss_on_exchange_interval": 60,
        "stoploss_on_exchange_limit_ratio": 0.99
    }

    # ROI table:
    minimal_roi = {
        "0": 0.326,
        "120": 0.045,
        "235": 0.029,
        "268": 0
    }

    # Stoploss:
    stoploss = -0.02

    #timeframe
    timeframe = '15m'

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.021
    trailing_stop_positive_offset = 0.025
    trailing_only_offset_is_reached = False

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
            }
        }
    }

    process_only_new_candles = True
    use_exit_signal = True
    startup_candle_count: int = 150

    # Hyperoptable parameters

    # buy_stochrsi_fastk_bbup_min = DecimalParameter(
    #     low=5, high=25, default=10, decimals=0, space='buy')
    # buy_stochrsi_fastk_bbup_max = DecimalParameter(
    #     low=50, high=80, default=70, decimals=0, space='buy')
    # buy_stochrsi_fastk_bbmid_min = DecimalParameter(
    #     low=5, high=25, default=10, decimals=0, space='buy')
    # buy_stochrsi_fastk_bbmid_max = DecimalParameter(
    #     low=50, high=80, default=70, decimals=0, space='buy')
    # buy_stochrsi_fastk_bbdown_min = DecimalParameter(
    #     low=5, high=25, default=10, decimals=0, space='buy')
    # buy_stochrsi_fastk_bbdown_max = DecimalParameter(
    #     low=50, high=80, default=70, decimals=0, space='buy')

    # buy_bb_middle_band_min_slope = DecimalParameter(
    #     low=-25, high=5, default=-20, decimals=0, space='buy', optimize=False, load=True)
    # buy_bb_middle_band_max_slope = DecimalParameter(
    #     low=5, high=25, default= 20, decimals=0, space='buy', optimize=False, load=True)

    # buy_laguerre_bbup_min = DecimalParameter(
    #     low=0.0, high=0.50, default=0.10, decimals=2, space='buy')
    # buy_laguerre_bbup_max = DecimalParameter(
    #     low=0.50, high=1, default=0.70, decimals=2, space='buy')
    # buy_laguerre_min = DecimalParameter(
    #     low=0.0, high=0.25, default=0.10, decimals=2, space='buy')
    # buy_laguerre_max = DecimalParameter(
    #     low=0.50, high=0.9, default=0.70, decimals=2, space='buy')
    # buy_laguerre_bbdown_min = DecimalParameter(
    #     low=0.0, high=0.50, default=0.10, decimals=2, space='buy')
    # buy_laguerre_bbdown_max = DecimalParameter(
    #     low=0.50, high=1, default=0.70, decimals=2, space='buy')

    # buy_srsi = IntParameter(
    #     low=50, high=80, default=70, space='buy')
    # buy_ema_slope = DecimalParameter(
    #     low=5, high=50, default= 20, decimals=0, space='buy')
    # buy_srsi_btc = IntParameter(
    #     low=30, high=80, default=60, space='buy')
    # buy_fisher_entry_max = DecimalParameter(
    #     low=25, high=50, default=50, decimals=0, space='buy')

    # buy_bbup_entry = DecimalParameter(
    #     low=0.5, high=0.8, default=0.6, decimals=1, space='buy')
    # buy_bbmid_entry = DecimalParameter(
    #     low=0.1, high=0.4, default=0.2, decimals=1, space='buy')
    # buy_bbdown_entry = DecimalParameter(
    #     low=0.1, high=0.4, default=0.2, decimals=1, space='buy')

    # sell_srsi = IntParameter(
    #     low=10, high=40, default=17, space='sell')
    # sell_ema_slope = DecimalParameter(
    #     low=-50, high=-5, default=-20, decimals=0, space='sell')
    # sell_fisher_exit_max = DecimalParameter(
    #     low=75, high=100, default=0, decimals=0, space='sell')

    # @informative('1h')
    # def populate_indicators_btc_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    #     # dataframe.ta.stochrsi(length=9, rsi_length=9, k=3, d=3, append=True)
    #     # macd = ta.MACD(dataframe, fastperiod=5, slowperiod=9, signalperiod=9)
    #     # # dataframe['macd'] = macd['macd']
    #     # # dataframe['macdsignal'] = macd['macdsignal']
    #     # dataframe['macdhist'] = macd['macdhist']
    #     # dataframe["ema"] = ta.EMA(dataframe, timeperiod=2)
    #     # dataframe["ema-signal"] = ta.EMA(dataframe, timeperiod=9)
    #     dataframe["laguerre"] = laguerre(dataframe, gamma=0.25, smooth=1)
    #     return dataframe

    # FreqAI required function, user can add or remove indicators, but general structure
    # must stay the same.
    def populate_any_indicators(
        self, pair, df, tf, informative=None, set_generalized_indicators=False
    ):
        """
        User feeds these indicators to FreqAI to train a classifier to decide
        if the market will go up or down.

        :param pair: pair to be used as informative
        :param df: strategy dataframe which will receive merges from informatives
        :param tf: timeframe of the dataframe which will modify the feature names
        :param informative: the dataframe associated with the informative pair
        """

        coin = pair.split('/')[0]

        if informative is None:
            informative = self.dp.get_pair_dataframe(pair, tf)

        # dataframe = informative.copy()
        # dataframe.ta.stochrsi(length=9, rsi_length=9, k=3, d=3, append=True)
        # informative[f"%-{coin}-stochrsik"] = dataframe[f'STOCHRSIk_9_9_3_3']
        # informative[f"%-{coin}-stochrsid"] = dataframe[f'STOCHRSId_9_9_3_3']

        # dataframe.ta.stoch(k=5, smooth_k=5, append=True)
        # dataframe['fastk_t'] = 0.1 * (dataframe['STOCHk_5_3_5'] - 50)
        # wma = pd.DataFrame()
        # wma['close'] = dataframe['fastk_t']
        # dataframe['fastk_t_avg'] = wma.ta.wma(length=9, append=True)
        # informative[f"%-{coin}-fisher_stoch"] = (np.exp(2 * dataframe['fastk_t_avg']) - 1) / (np.exp(2 * dataframe['fastk_t_avg']) + 1)

        # bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(informative), window=20, stds=2)
        # informative[f"%-{coin}-bb_lowerband"] = bollinger['lower']
        # # informative[f"%-{coin}-bb_middleband"] = bollinger['mid']
        # informative[f"%-{coin}-bb_upperband"] = bollinger['upper']
        # informative[f"%-{coin}-bb_percent"] = (
        #         (informative["close"] - bollinger['lower']) /
        #         (bollinger['upper'] - bollinger['lower'])
        #     )

        # first loop is automatically duplicating indicators for time periods
        for t in self.freqai_info["feature_parameters"]["indicator_periods_candles"]:

            t = int(t)

            # dataframe = informative.copy()
            # dataframe.ta.fisher(length=t,append=True)
            # informative[f"%-{coin}-fisher_{t}"] = dataframe[f'FISHERT_{t}_1']

            # informative[f"%-{coin}-tema_{t}"] = ta.TEMA(informative, timeperiod=t)
            # informative[f"%-{coin}-tema_slope_{t}"] = ta.LINEARREG_ANGLE(
            #     # ta.TEMA(informative, timeperiod=t), 2)
            # macd = ta.MACD(informative, fastperiod=5, slowperiod=9, signalperiod=t)
            # informative[f"%-{coin}macd{t}"] = macd['macd']
            # informative[f"%-{coin}macd_macdsignal{t}"] = macd['macdsignal']
            # informative[f"%-{coin}_macdhist{t}"] = macd['macdhist']
            # StochRSI = ta.STOCHRSI(informative, timeperiod=t, fastk_period=3, fastd_period=3)
            # informative[f"%-{coin}stochrsi-k_{t}"] = StochRSI.fastk
            # informative[f"%-{coin}stochrsi-d_{t}"] = StochRSI.fastd

            # dataframe = informative.copy()
            # dataframe.ta.stochrsi(length=t, rsi_length=t, k=3, d=3, append=True)
            # informative[f"%-{coin}-stochrsik{t}"] = dataframe[f'STOCHRSIk_{t}_{t}_3_3']
            # informative[f"%-{coin}-stochrsid{t}"] = dataframe[f'STOCHRSId_{t}_{t}_3_3']

            # # print(informative.tail())
            # WHITE PAPER
            # informative[f"%-{coin}mom-period_{t}"] = ta.MOM(informative, timeperiod=t)
            # informative[f"%-{coin}tsf-period_{t}"] = ta.TSF(informative, timeperiod=t)
            # informative[f"%-{coin}mfi_{t}"] = ta.MFI(informative, timeperiod=t)
            # informative[f"%-{coin}sma-period_{t}"] = ta.SMA(informative, timeperiod=t)
            informative[f"%-{coin}ema-period_{t}"] = ta.EMA(informative, timeperiod=t)
            # informative[f"%-{coin}-ema_slope_{t}"] = ta.LINEARREG_ANGLE(informative[f"%-{coin}ema-period_{t}"], 2)
            # informative[f"%-{coin}adx-period_{t}"] = ta.ADX(informative, timeperiod=t)
            # informative[f"%-{coin}rsi-period_{t}"] = ta.RSI(informative, timeperiod=t)

            # informative[f"%-{coin}laguerre_{t}"] = laguerre(informative, gamma=0.25, smooth=1)

            # # MAMA = ta.MAMA(informative, timeperiod=t)
            # # informative[f"%-{coin}-mama_{t}"] = MAMA['mama']
            # # informative[f"%-{coin}-mama_slope_{t}"] = ta.LINEARREG_ANGLE((MAMA['mama']), 2)
            # bollinger = qtpylib.bollinger_bands(
            #     qtpylib.typical_price(informative), window=t, stds=2)
            # informative[f"%-{coin}-bb_lowerband_{t}"] = bollinger['lower']
            # informative[f"%-{coin}-bb_middleband_{t}"] = bollinger['mid']
            # informative[f"%-{coin}-bb_upperband_{t}"] = bollinger['upper']
            # informative[f"%-{coin}-bb_percent_{t}"] = (
            #         (informative["close"] - bollinger['lower']) /
            #         (bollinger['upper'] - bollinger['lower'])
            # )

            # ############## EXTENSIONS #################

            ### SuperTrend indicators ###
            informative[f"{coin}atr_{t}"] = ta.ATR(
                informative['high'], informative['low'], informative['close'], timeperiod=t)
            informative[f"{coin}base_{t}"] = (informative['high'] + informative['low']) / 2
            informative[f"{coin}base_sma_{t}"] = ta.SMA(
                informative[f"{coin}base_{t}"], timeperiod=t)

            # Calculate the SuperTrend indicator
            up = informative[f"{coin}base_sma_{t}"] + (2 * informative[f"{coin}atr_{t}"])
            down = informative[f"{coin}base_sma_{t}"] - (2 * informative[f"{coin}atr_{t}"])
            informative[f"%-{coin}-supertrend_{t}"] = np.where(
                informative['close'] > informative[f"{coin}base_sma_{t}"], up, down)

            # Buy and sell signals
            informative[f"%-{coin}-long_signal_{t}"] = np.where(
                informative[f"%-{coin}-supertrend_{t}"] > informative['close'], 1, 0)
            informative[f"%-{coin}-short_signal_{t}"] = np.where(
                informative[f"%-{coin}-supertrend_{t}"] < informative['close'], -1, 0)

            ### HMA INDICATOR ###
            # Calculate the square root of the period
            sqrt_period = math.sqrt(t)

            # Calculate the weighted moving average using the square root of the period
            informative[f"{coin}-wma_sqrt_period_{t}"] = informative['close'].rolling(
                window=t, min_periods=t).apply(lambda x: (x * sqrt_period).mean(), raw=True)

            # Calculate the second weighted moving average using half the period of the first one
            half_period = t // 2
            informative[f"{coin}-wma_half_period_{t}"] = informative[f"{coin}-wma_sqrt_period_{t}"].rolling(
                window=int(half_period), min_periods=int(half_period)).mean()

            # Calculate the Hull Moving Average
            informative[f"%-{coin}-HMA_{t}"] = informative[f"{coin}-wma_sqrt_period_{t}"] - \
                informative[f"{coin}-wma_half_period_{t}"]

            # informative[f"%-{coin}atr_{t}"] = ta.ATR(informative['high'], informative['low'], informative['close'], timeperiod=t)
            # informative[f"%-{coin}obv_{t}"] = ta.OBV(informative, timeperiod=t)
            # informative[f"%-{coin}ad_{t}"] = ta.AD(informative, timeperiod=t)
            # informative[f"%-{coin}engulfing_{t}"] = ta.CDLENGULFING(informative, timeperiod=t)
            # Stoch = ta.STOCH(informative, timeperiod=t, fastk_period=3, slowkd_period=t-2)
            # informative[f"%-{coin}stoch-k_{t}"] = Stoch.slowk
            # informative[f"%-{coin}stoch-d_{t}"] = Stoch.slowd
            # informative[f"%-{coin}stdev_{t}"] = ta.STDDEV(informative, timeperiod=t)
            # informative[f"%-{coin}floor_{t}"] = ta.FLOOR(informative, timeperiod=t)

        # The following raw price values are necessary for RL models
        # informative[f"%-{pair}raw_close"] = informative["close"]
        # informative[f"%-{coin}pct-change"] = informative["close"].pct_change()
        # informative[f"%-{pair}raw_open"] = informative["open"]
        # informative[f"%-{pair}raw_high"] = informative["high"]
        # informative[f"%-{coin}mecha_sup"] = (informative['high'] - informative[["open",'close']].max(axis=1))
        # informative[f"%-{coin}mecha_inf"] = (informative[["open",'close']].min(axis=1) - informative['low'])

        # informative[f"%-{pair}raw_low"] = informative["low"]
        # informative[f"%-{pair}raw_volume"] = informative["volume"]
        # informative[f"%-{coin}laguerre"] = laguerre(informative, gamma=0.25, smooth=1)

        # FreqAI needs the following lines in order to detect features and automatically
        # expand upon them.
        indicators = [col for col in informative if col.startswith("%")]
        # This loop duplicates and shifts all indicators to add a sense of recency to data
        for n in range(self.freqai_info["feature_parameters"]["include_shifted_candles"] + 1):
            if n == 0:
                continue
            informative_shift = informative[indicators].shift(n)
            informative_shift = informative_shift.add_suffix("_shift-" + str(n))
            informative = pd.concat((informative, informative_shift), axis=1)

        df = merge_informative_pair(df, informative, self.config["timeframe"], tf, ffill=True)
        skip_columns = [
            (s + "_" + tf) for s in ["date", "open", "high", "low", "close", "volume"]
        ]
        # df = df.drop(columns=skip_columns)
        # informative.to_csv('user_data/informative.csv', sep='\t')

        # User can set the "target" here (in present case it is the
        # "up" or "down")
        if set_generalized_indicators:
            # User "looks into the future" here to figure out if the future
            # will be "up" or "down". This same column name is available to
            # the user
            # df['&-s-up_or_down'] = np.where((df['close'].shift(1) < df['close'])&(df["close"].shift(-2) > df["close"]), 'up', 'down')

            # df["ema-5"] = ta.EMA(df, timeperiod=5)
            # Example for multiple options for classifiers
            # df['&s-up_or_down'] = np.where(df["close"].shift(-100) > df["close"], 'up', 'down')
            # df['&s-up_or_down'] = np.where(df["close"].shift(-100) == df["close"], 'same', df['&s-up_or_down'])

            df['&-s-up_or_down'] = np.where((df["close"].shift(-3) > df["close"]), 'up', 'same')
            df['&-s-up_or_down'] = np.where((df["close"].shift(-3) <=
                                            df["close"]), 'down', df['&-s-up_or_down'])
            # df['&-s-up_or_down'] = np.where((df["close"].shift(-3) > df["close"]), 'up', 'down')

        return df

    # flake8: noqa: C901
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # User creates their own custom strat here. Present example is a supertrend
        # based strategy.

        dataframe = self.freqai.start(dataframe, metadata, self)

        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
        dataframe["bb_upper_band"] = bollinger['upper']
        dataframe["bb_middle_band"] = bollinger['mid']
        # dataframe['bb_middle_band_slope'] = ta.LINEARREG_ANGLE(dataframe['bb_middle_band'],4)
        dataframe["bb_lower_band"] = bollinger['lower']
        # dataframe["bb_percent"] = (dataframe["close"] - bollinger['lower']) / (bollinger['upper'] - bollinger['lower'])

        # tke = TKE(dataframe)
        # dataframe["tke"] = tke[0]
        # dataframe['tke_slope'] = ta.LINEARREG_ANGLE(dataframe['tke'], 2)

        # StochRSI = ta.STOCHRSI(dataframe, timeperiod= 9, fastk_period= 3, fastd_period= 3, fastd_matype=0)
        dataframe.ta.stochrsi(length=9, rsi_length=9, k=3, d=3, append=True)
        # print(dataframe.tail())
        # dataframe["stochrsi_fastk"] =
        # dataframe['stochrsi_fastk_slope'] = ta.LINEARREG_ANGLE(dataframe['stochrsi_fastk'], 2)

        dataframe["laguerre"] = laguerre(dataframe, gamma=0.25, smooth=1)
        # dataframe['laguerre_slope'] = ta.LINEARREG_ANGLE(dataframe['laguerre'], 2)

        macd = ta.MACD(dataframe, fastperiod=3, slowperiod=9, signalperiod=9)
        dataframe['macd'] = macd['macd']
        # dataframe['macdsignal'] = macd['macdsignal']
        # dataframe['macdhist'] = macd['macdhist']
        # dataframe['macd_slope'] = ta.LINEARREG_ANGLE(dataframe['macdhist'], 2)
        # print(dataframe.tail())

        # dataframe.ta.fisher(append=True)

        dataframe["ema"] = ta.EMA(dataframe, timeperiod=5)
        # dataframe["ema_slope"] = ta.LINEARREG_ANGLE(dataframe['ema'], 2)
        # dataframe["ema-9"] = ta.EMA(dataframe, timeperiod=9)
        # TEMA - Triple Exponential Moving Average
        # dataframe["tema"] = ta.TEMA(dataframe, timeperiod= 5)
        # dataframe["tema_slope"] = ta.LINEARREG_ANGLE(dataframe['tema'], timeperiod=t), 2)

        # MAMA = ta.MAMA(dataframe)
        # dataframe["mama"] = MAMA['mama']
        # dataframe[f"%-{coin}-mama_slope_{t}"] = ta.LINEARREG_ANGLE((MAMA['mama']), 2)

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[
            (
                (df['do_predict'] == 1) &
                (df['&-s-up_or_down'] == 'up')
                # &
                # (df['macd'] > df['macd'].shift(1))
                # &
                # (df['STOCHRSIk_9_9_3_3'] > df['STOCHRSIk_9_9_3_3'].shift(1))
                # &
                # (df['STOCHRSIk_9_9_3_3'] < self.buy_srsi.value)
                # &
                # (df['ema'] > df['ema'].shift(1))
                # # &
                # # (df['ema_slope'] > self.buy_ema_slope.value)
                # &
                # (df['laguerre'] > df['laguerre'].shift(1))
                # &
                # (df['laguerre_1h'] >= df['laguerre_1h'].shift(1))
                # &
                # (df['laguerre'] < self.buy_laguerre_max.value)
                # &
                # (df['laguerre'] > self.buy_laguerre_min.value)
                # &
                # (df['ema_2h'] > df['ema_2h'].shift(1))
                # &
                # # (
                # (df['ema_1h'] > df['ema_1h'].shift(1))
                # #   |
                # #   ((df['STOCHRSIk_9_9_3_3_1h'] > df['STOCHRSIk_9_9_3_3_1h'].shift(1))
                # #    &
                # #    (df['STOCHRSIk_9_9_3_3_1h'] < 25)
                # #   )
                # #   )

                # ((
                # (df['STOCHRSIk_9_9_3_3_1h'] > df['STOCHRSIk_9_9_3_3_1h'].shift(1)) &
                # (df['STOCHRSIk_9_9_3_3_1h'] < self.buy_srsi_btc.value))
                # |
                #     (df['macdhist_1h'] > df['macdhist_1h'].shift(1)))

                # (df['laguerre'] > df['laguerre'].shift(1)) &
                # (
                #     (
                #         # bb up
                #         (df['bb_middle_band_slope'] >= self.buy_bb_middle_band_max_slope.value) &
                #         # (df['laguerre'] > self.buy_laguerre_bbup_min.value) &
                #         # (df['laguerre'] < self.buy_laguerre_bbup_max.value) &
                #         # (df['STOCHRSIk_9_9_5_5'] > self.buy_stochrsi_fastk_bbup_min.value) &
                #         # (df['STOCHRSIk_9_9_5_5'] < self.buy_stochrsi_fastk_bbup_max.value) &
                #         (df['bb_percent'] <= self.buy_bbup_entry.value)
                #     ) |
                #     (
                #         # bb mid
                #         (df['bb_middle_band_slope'] <= self.buy_bb_middle_band_max_slope.value) &
                #         (df['bb_middle_band_slope'] >= self.buy_bb_middle_band_min_slope.value) &
                #         # (df['laguerre'] > self.buy_laguerre_bbmid_min.value) &
                #         # (df['laguerre'] < self.buy_laguerre_bbmid_max.value) &
                #         # (df['STOCHRSIk_9_9_5_5'] > self.buy_stochrsi_fastk_bbmid_min.value) &
                #         # (df['STOCHRSIk_9_9_5_5'] < self.buy_stochrsi_fastk_bbmid_max.value) &
                #         (df['bb_percent'] <= self.buy_bbmid_entry.value)
                #     ) |
                #     (
                #         # bb down
                #         (df['bb_middle_band_slope'] <= self.buy_bb_middle_band_min_slope.value) &
                #         # (df['laguerre'] > self.buy_laguerre_bbdown_min.value) &
                #         # (df['laguerre'] < self.buy_laguerre_bbdown_max.value) &
                #         # (df['STOCHRSIk_9_9_5_5'] > self.buy_stochrsi_fastk_bbdown_min.value) &
                #         # (df['STOCHRSIk_9_9_5_5'] < self.buy_stochrsi_fastk_bbdown_max.value) &
                #         (df['bb_percent'] <= self.buy_bbdown_entry.value)
                #     )
                # )
            ),
            'enter_long'] = 1

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[
            (
                (df['do_predict'] == 1) &
                (df['&-s-up_or_down'] == 'down')
                # # # &
                # # # (df['STOCHRSIk_9_9_3_3'] <= df['STOCHRSIk_9_9_3_3'].shift(1))
                # (df['STOCHRSIk_9_9_3_3'] > self.sell_srsi.value)
                # &
                # (df['ema'] < df['ema'].shift(1))
                # &
                # (df['ema_slope'] < self.sell_ema_slope.value)
                # &
                # (df['ema-5'] < df['ema-9'])
                # (df['macd_slope'] > self.buy_macd_slope_entry.value)&
                # (df['macdhist'] < df['macdhist'].shift(1)) &
                # (df['stochrsi_fastk_slope'] > self.buy_stochrsi_fastk_slope_entry.value)&
                # (df['laguerre'] > 0.15) &
                # (df['laguerre'] < 0.8) &
                # |
                # ((df['laguerre'] == 1)
                # &
                # (df['laguerre'] <= df['laguerre'].shift(1))
                # &
                # (df['laguerre'] > 0.25)
                # )
            ),
            'exit_long'] = 1

        return df
