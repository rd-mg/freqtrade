import logging

import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib
import pandas_ta as pta


from freqtrade.strategy import DecimalParameter,IntParameter, IStrategy, merge_informative_pair

from technical.indicators import RMI, laguerre, madrid_sqz, TKE, vwmacd, MADR



logger = logging.getLogger(__name__)

class FreqaiHybridStrategySpot_slope(IStrategy):
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
                "BTC/USDT",
                "ETH/USDT"
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

    Thanks to @smarmau and @johanvulgt for developing and sharing the strategy.
    """

    minimal_roi = {
        "60": 0.01,
        "30": 0.02,
        "0": 0.04
    }

    plot_config = {
        'main_plot': {
            'bb_upper_band': {'color': 'blue'},
            'bb_lower_band': {'color': 'blue'},
            'tema': {'color': 'white'},
        },
        'subplots': {
            "LRSI": {
                'laguerre': {'color': 'green'},
            },
            "stochrsi_fastk": {
                'STOCHRSIk_14_14_3_3': {'color': 'blue'},
            },
            "macdhist": {
                'macdhist': {'color': 'red'},
            },
            "&s-up_or_down": {
                '&-s-up_or_down': {'color': 'black'},
            }
        }
    }

    process_only_new_candles = True
    stoploss = -0.015
    use_exit_signal = True
    startup_candle_count: int = 300

    # Hyperoptable parameters
    # buy_laguerre_min = DecimalParameter(
    #     low=0.10, high=0.49, default=0.20, decimals=2, space='buy', optimize=True, load=True)
    # buy_laguerre_max = DecimalParameter(
    #     low=0.50, high=0.80, default=0.50, decimals=2, space='buy', optimize=True, load=True)
    buy_stochrsi_fastk_min = DecimalParameter(
        low=5, high=24, default=15, decimals=0, space='buy', optimize=True, load=True)
    buy_stochrsi_fastk_max = DecimalParameter(
        low=25, high=80, default=60, decimals=0, space='buy', optimize=True, load=True)
    # buy_bb_middle_band_min_slope = DecimalParameter(
    #     low=-45, high=30, default=-25, decimals=0, space='buy', optimize=True, load=True)
    buy_bb_lower_band_scope_entry = DecimalParameter(
        low=0.20, high=0.50, default=0.25, decimals=2, space='buy', optimize=True, load=True)
    # buy_macd_slope_entry = DecimalParameter(
    #     low=-10, high=10, default=0, decimals=0, space='buy', optimize=True, load=True)
    # buy_laguerre_slope_entry = DecimalParameter(
    #     low=0, high=90, default=0, decimals=0, space='buy', optimize=True, load=True)
    # buy_tke_slope_entry = DecimalParameter(
    #     low=0, high=90, default=0, decimals=0, space='buy', optimize=True, load=True)
    # buy_stochrsi_fastk_slope_entry = DecimalParameter(
    #     low=0, high=45, default=1.0, decimals=0, space='buy', optimize=True, load=True)
    # buy_fisher_entry = DecimalParameter(
    #     low=-2.0, high=0.0, default=-1.0, decimals=1, space='buy', optimize=True, load=True)




    # sell_tke = DecimalParameter(low=20, high=80, default=40, decimals=0, space='sell', optimize=True, load=True)
    # sell_fisher_exit = DecimalParameter(
    #     low=0.0, high=2.0, default=1.0, decimals=1, space='sell', optimize=True, load=True)
    # sell_stochrsi_fastk_min = DecimalParameter(
    #     low=40, high=70, default=50.0, decimals=0, space='sell', optimize=True, load=True)
    # sell_stochrsi_fastk_max = DecimalParameter(
    #     low=70, high=100, default=80.0, decimals=0, space='sell', optimize=True, load=True)
    # # sell_macd_slope_exit = DecimalParameter(
    # #     low=-10.0, high=10.0, default=-1.0, decimals=1, space='sell', optimize=True, load=True)
    # sell_stochrsi_fastk_slope_exit = DecimalParameter(
    #     low=0.0, high=45.0, default=-1.0, decimals=1, space='sell', optimize=True, load=True)


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

        # first loop is automatically duplicating indicators for time periods
        for t in self.freqai_info["feature_parameters"]["indicator_periods_candles"]:

            t = int(t)
            # informative[f"%-{coin}rsi-period_{t}"] = ta.RSI(informative, timeperiod=t)
            macd = ta.MACD(informative, timeperiod=t)
            informative[f"%-{coin}macd_macdhist{t}"] = macd['macdhist']
            informative[f"%-{coin}macd_slope_{t}"] = ta.LINEARREG_ANGLE(macd["macdhist"], 2)

            # informative[f"%-{coin}adx-period_{t}"] = ta.ADX(informative, timeperiod=t)
            # informative[f"%-{coin}sma-period_{t}"] = ta.SMA(informative, timeperiod=t)
            # informative[f"%-{coin}ema-period_{t}"] = ta.EMA(informative, timeperiod=t)
            # informative[f"%-{coin}roc-period_{t}"] = ta.ROC(informative, timeperiod=t)
            informative[f"%-{coin}-stddev_volume_{t}"] = ta.STDDEV(informative,timeperiod=t)
            
            # MADRID_SQZ = madrid_sqz(informative)
            # informative[f"%-{coin}-madrid_sqz_{t}"] = MADRID_SQZ[1]
            
            informative[f"%-{coin}-laguerre_{t}"] = laguerre(informative, gamma=0.25, smooth=1)

            # tke = TKE(informative)
            # informative[f"%-{coin}-TKE_{t}"] = tke[0]
            # informative[f"%-{coin}-TKE_slope_{t}"] = ta.LINEARREG_ANGLE(tke[0], 2)
            
            fisher = informative.copy()
            fisher.ta.fisher(length=t,append=True)
            informative[f"%-{coin}-fisher_{t}"] = fisher[f'FISHERT_{t}_1']

            # madr = MADR(informative)
            # informative[f"%-{coin}-madr_stdcenter{t}"] = madr['stdcenter']
            # informative[f"%-{coin}-madr_plusdev{t}"] = madr['plusdev']
            # informative[f"%-{coin}-madr_minusdev{t}"] = madr['minusdev']
            # informative[f"%-{coin}-madr_rate{t}"] = madr['rate']
            # informative[f"%-{coin}-madr_slope_rate{t}"] = (madr['rate'] - madr['rate'].shift(1))
                        
            bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(informative), window=20, stds=2)
            informative[f"%-{coin}-bb_lowerband_{t}"] = bollinger['lower']
            informative[f"%-{coin}-bb_middleband_{t}"] = bollinger['mid']
            informative[f"%-{coin}-bb_upperband_{t}"] = bollinger['upper']
            informative[f"%-{coin}-bb_percent"] = (
                (informative["close"] - bollinger['lower']) / (bollinger['upper'] - bollinger['lower'])
            )
            # informative[f"%-{coin}-bb_width"]=(
            #     (bollinger["upper"] - bollinger["lower"]) / bollinger["mid"]
            # )
            # # informative[f"%-{coin}-bb_slope_middleband"]=(
            #     (bollinger["mid"]- bollinger["mid"].shift(1))
            # )

            # # TEMA - Triple Exponential Moving Average
            informative[f"%-{coin}-tema_{t}"] = ta.TEMA(informative, timeperiod=t)
            informative[f"%-{coin}-tema_slope_{t}"] = ta.LINEARREG_ANGLE(ta.TEMA(informative, timeperiod=t), 2)   
            
            MAMA = ta.MAMA(informative, timeperiod=t)
            informative[f"%-{coin}-mama_{t}"] = MAMA['mama'] 
            informative[f"%-{coin}-mama_slope_{t}"] = ta.LINEARREG_ANGLE((MAMA['mama']), 2)

        informative[f"%-{coin}pct-change"] = informative["close"].pct_change()
        informative[f"%-{coin}raw_volume"] = informative["volume"]
        informative[f"%-{coin}raw_price"] = informative["close"]
        informative.ta.stochrsi(append=True)
        informative[f"%-{coin}fastk"] = informative['STOCHRSIk_14_14_3_3']
        informative[f"%-{coin}fastk_slope"] = ta.LINEARREG_ANGLE(
            informative['STOCHRSIk_14_14_3_3'], 2)



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
        df = df.drop(columns=skip_columns)

        # User can set the "target" here (in present case it is the
        # "up" or "down")
        if set_generalized_indicators:
            # User "looks into the future" here to figure out if the future
            # will be "up" or "down". This same column name is available to
            # the user
            df['&-s-up_or_down'] = np.where(df["close"].shift(-3) > df["close"], 'up', 'down')
            
        return df

    # flake8: noqa: C901
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # User creates their own custom strat here. Present example is a supertrend
        # based strategy.

        dataframe = self.freqai.start(dataframe, metadata, self)
        
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe["bb_upper_band"] = bollinger['upper']
        dataframe["bb_middle_band"] = bollinger['mid']
        # dataframe['bb_middle_band_slope'] = ta.LINEARREG_ANGLE(dataframe['bb_middle_band'],2)
        dataframe["bb_lower_band"] = bollinger['lower']
        dataframe["bb_percent"] = (dataframe["close"] - bollinger['lower']) / (bollinger['upper'] - bollinger['lower'])

        # tke = TKE(dataframe)
        # dataframe["tke"] = tke[0]
        # dataframe['tke_slope'] = ta.LINEARREG_ANGLE(dataframe['tke'], 2)
        
        # StochRSI = ta.STOCHRSI(dataframe, timeperiod= 14, fastk_period= 3, fastd_period= 3)
        # dataframe["stochrsi_fastk"] = StochRSI["fastk"]
        # dataframe['stochrsi_fastk_slope'] = ta.LINEARREG_ANGLE(dataframe['stochrsi_fastk'], 2)


        # _name = "STOCHRSI"
        # https://github.com/twopirllc/pandas-ta/blob/084dbe1c4b76082f383fa3029270ea9ac35e4dc7/pandas_ta/momentum/stochrsi.py#L8
        # _props = f"_{length}_{rsi_length}_{k}_{d}"
        # stochrsi_k.name = f"{_name}k{_props}"  STOCHRSIk_14_14_3_3
        # stochrsi_d.name = f"{_name}d{_props}"
        # def stochrsi(close, length=None, rsi_length=None, k=None, d=None, mamode=None, offset=None, **kwargs):
        dataframe.ta.stochrsi(append=True)
        

        dataframe["laguerre"] = laguerre(dataframe, gamma=0.25, smooth=1)
        # dataframe['laguerre_slope'] = ta.LINEARREG_ANGLE(dataframe['laguerre'], 2)

        macd = ta.MACD(dataframe)
        dataframe["macdhist"] = macd['macdhist']
        # dataframe['macd_slope'] = ta.LINEARREG_ANGLE(dataframe['macdhist'], 2)
        
        # column name 
        # def fisher(high, low, length=None, signal=None, offset=None, **kwargs):
        # _props = f"_{length}_{signal}"  
        # fisher.name = f"FISHERT{_props}"

        dataframe.ta.fisher(append=True)
        
        # TEMA - Triple Exponential Moving Average
        dataframe["tema"] = ta.TEMA(dataframe, timeperiod= 9)
        # dataframe["tema_slope"] = ta.LINEARREG_ANGLE(dataframe['tema'], timeperiod=t), 2)

        # MAMA = ta.MAMA(dataframe, timeperiod=t)
        # dataframe[f"%-{coin}-mama_{t}"] = MAMA['mama']
        # dataframe[f"%-{coin}-mama_slope_{t}"] = ta.LINEARREG_ANGLE((MAMA['mama']), 2)
            
        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        

        df.loc[
            (   
                (df['do_predict'] == 1) &
                (df['&-s-up_or_down'] == 'up')
                &
                # (df['FISHERT_9_1'] < self.buy_fisher_entry.value)&
                (df['STOCHRSIk_14_14_3_3'] > self.buy_stochrsi_fastk_min.value) &
                (df['STOCHRSIk_14_14_3_3'] < self.buy_stochrsi_fastk_max.value) &
                (df['STOCHRSIk_14_14_3_3'] > df['STOCHRSIk_14_14_3_3'].shift(1)) &
                # (df['macd_slope'] > self.buy_macd_slope_entry.value)&
                (df['macdhist'] > df['macdhist'].shift(1))&
                # (df['stochrsi_fastk_slope'] > self.buy_stochrsi_fastk_slope_entry.value)&
                (df['laguerre'] > 0.1)&
                (df['laguerre'] < 0.8) &
                (df['laguerre'] > df['laguerre'].shift(1))&
                (df['tema'] > df['tema'].shift(1)) &
                (
                    (    
                        # bb mid up trend
                        (df['tema'] > df['bb_middle_band'])&
                        (df['bb_middle_band'] > df['bb_middle_band'].shift(1))&
                        (df['tema'] < df['bb_upper_band'])
                    ) |
                    (
                        # bb mid down trend
                        (df['tema'] <= df['bb_middle_band']) &
                        (df['bb_middle_band'] < df['bb_middle_band'].shift(1)) & 
                        (df['bb_percent'] <= self.buy_bb_lower_band_scope_entry.value)
                    )
                )
            ),
            'enter_long'] = 1
        
        return df
    

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[
            (
                (df['do_predict'] == 1) &
                (df['&-s-up_or_down'] == 'down')
                &
                # (df['FISHERT_9_1'] < self.buy_fisher_entry.value)&
                # (df['stochrsi_fastk'] > self.buy_stochrsi_fastk_min.value) &
                # (df['stochrsi_fastk'] < self.buy_stochrsi_fastk_max.value) &
                (df['STOCHRSIk_14_14_3_3'] < df['STOCHRSIk_14_14_3_3'].shift(1)) &
                # (df['macd_slope'] > self.buy_macd_slope_entry.value)&
                (df['macdhist'] < df['macdhist'].shift(1)) &
                # (df['stochrsi_fastk_slope'] > self.buy_stochrsi_fastk_slope_entry.value)&
                # (df['laguerre'] > 0.15) &
                # (df['laguerre'] < 0.8) &
                (df['laguerre'] < df['laguerre'].shift(1)) &
                (df['tema'] < df['tema'].shift(1)) 
                # (
                #     (
                #         # bb mid up trend
                #         (df['tema'] > df['bb_middle_band']) &
                #         (df['bb_middle_band'] > df['bb_middle_band'].shift(1))

                #         ) |
                #     (
                #         # bb mid down trend
                #         (df['tema'] <= df['bb_middle_band']) &
                #         (df['bb_middle_band'] < df['bb_middle_band'].shift(1)) &
                #         (df['tema'] <= self.buy_bb_lower_band_scope_entry.value)
                #         )
                #     )
                ),
            'exit_long'] = 1

        return df
