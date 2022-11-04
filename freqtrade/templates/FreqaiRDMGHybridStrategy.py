import logging
import math
import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib
from technical.indicators import RMI, laguerre, madrid_sqz, TKE, vwmacd, MADR

from freqtrade.strategy import IntParameter, IStrategy, merge_informative_pair, DecimalParameter


logger = logging.getLogger(__name__)


class FreqaiRDMGHybridStrategy(IStrategy):
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
            'tema': {},
        },
        'subplots': {
            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},
            },
            "RSI": {
                'rsi': {'color': 'red'},
            },
            "Up_or_down": {
                '&s-up_or_down': {'color': 'green'},
            }
        }
    }

    process_only_new_candles = True
    stoploss = -0.05
    use_exit_signal = True
    startup_candle_count: int = 300
    # can_short = False

    # Hyperoptable parameters
    buy_rsi = IntParameter(low=1, high=50, default=25, space='buy', optimize=True, load=True)
    sell_rsi = IntParameter(low=50, high=100, default=70, space='sell', optimize=True, load=True)
    buy_laguerre = DecimalParameter(low=0.1, high=0.8, default=0.2, space='buy', optimize=True, load=True)
    sell_laguerre = DecimalParameter(low=0.8, high=1, default=0.5, space='sell', optimize=True, load=True)
    
    # Compute the slope between 2 values
    def slope(y1,y2):
        x1 = 1.0
        x2 = 2.0
        x = x2 - x1
        y = y2 - y1

        angle = math.atan2(y, x) * (180.0 / math.pi)
        return(angle)

    # FreqAI required function, user can add or remove indicators, but general structure
    # must stay the same.
    def populate_any_indicators(
        self, pair, df, tf, informative=None, set_generalized_indicators=True
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
            # informative[f"%-{coin}-rsi-period_{t}"] = ta.RSI(informative, timeperiod=t)

            StochRSI = ta.STOCHRSI(informative, timeperiod=t)
            informative[f"%-{coin}-stochrsi_fastk_{t}"] = StochRSI["fastk"]
            informative[f"%-{coin}-stochrsi_fastd_{t}"] = StochRSI["fastd"]
            informative[f"%-{coin}-stochrsi_fastk-fastd_{t}"] = StochRSI["fastk"] - StochRSI["fastd"]

            # # MADRID_SQZ = madrid_sqz(informative)
            # # informative[f"%-{coin}-madrid_sqz_{t}"] = MADRID_SQZ[1]
            # informative[f"%-{coin}-laguerre_{t}"] = laguerre(informative, gamma=0.2, smooth=1)
            
            # tke = TKE(informative)
            # informative[f"%-{coin}-TKE_{t}"] = tke[0]
            
            # VWMACD = vwmacd(informative)
            # informative[f"%-{coin}-vwmacd_signal{t}"] = VWMACD['signal']
            # informative[f"%-{coin}-vwmacd_{t}"] = VWMACD['vwmacd']
            
            # madr = MADR(informative)
            # informative[f"%-{coin}-madr_stdcenter{t}"] = madr['stdcenter']
            # informative[f"%-{coin}-madr_plusdev{t}"] = madr['plusdev']
            # informative[f"%-{coin}-madr_minusdev{t}"] = madr['minusdev']
            # informative[f"%-{coin}-madr_rate{t}"] = madr['rate']
            
            # bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(informative), window=20, stds=2)
            # informative[f"%-{coin}-bb_lowerband"] = bollinger['lower']
            # informative[f"%-{coin}-bb_middleband"] = bollinger['mid']
            # informative[f"%-{coin}-bb_upperband"] = bollinger['upper']
            # informative[f"%-{coin}-bb_percent"] = (
            #     (informative["close"] - bollinger['lower']) / (bollinger['upper'] - bollinger['lower'])
            # )
            # informative[f"%-{coin}-bb_width"]=(
            #     (bollinger["upper"] - bollinger["lower"]) / bollinger["mid"]
            # )
            # informative[f"%-{coin}-bb_slope_middleband"]=(
            #     slope(bollinger["mid"],bollinger["mid"].shift(4))
            # )

            # # TEMA - Triple Exponential Moving Average
            # informative[f"%-{coin}-tema"]=ta.TEMA(informative, timeperiod=t)

        
        # informative[f"%-{coin}pct-change"] = informative["close"].pct_change()
        # informative[f"%-{coin}raw_volume"] = informative["volume"]
        # informative[f"%-{coin}raw_price"] = informative["close"]

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

        # Add generalized indicators here (because in live, it will call this
        # function to populate indicators during training). Notice how we ensure not to
        # add them multiple times
        if set_generalized_indicators:
            # df["%-day_of_week"] = (df["date"].dt.dayofweek + 1) / 7
            # df["%-hour_of_day"] = (df["date"].dt.hour + 1) / 25

            # # user adds targets here by prepending them with &- (see convention below)
            # df["&-s_close"] = (
            #     df["close"]
            #     .shift(-self.freqai_info["feature_parameters"]["label_period_candles"])
            #     .rolling(self.freqai_info["feature_parameters"]["label_period_candles"])
            #     .mean()
            #     / df["close"]
            #     - 1
            # )

            # Classifiers are typically set up with strings as targets:
            df['&-s-up_or_down'] = np.where(df["close"].shift(-8) > df["close"], 'up', 'down')

            # If user wishes to use multiple targets, they can add more by
            # appending more columns with '&'. User should keep in mind that multi targets
            # requires a multioutput prediction model such as
            # templates/CatboostPredictionMultiModel.py,

            # df["&-s_range"] = (
            #     df["close"]
            #     .shift(-self.freqai_info["feature_parameters"]["label_period_candles"])
            #     .rolling(self.freqai_info["feature_parameters"]["label_period_candles"])
            #     .max()
            #     -
            #     df["close"]
            #     .shift(-self.freqai_info["feature_parameters"]["label_period_candles"])
            #     .rolling(self.freqai_info["feature_parameters"]["label_period_candles"])
            #     .min()
            # )
           
        # df.info(verbose=True)
        #df.to_csv('/freqtrade/user_data/any_indicators.csv', sep=',', encoding='utf-8')

        return df

    # flake8: noqa: C901
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # User creates their own custom strat here. Present example is a supertrend
        # based strategy.

        dataframe = self.freqai.start(dataframe, metadata, self)

        # TA indicators to combine with the Freqai targets
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe["bb_percent"] = (
            (dataframe["close"] - dataframe["bb_lowerband"]) /
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"])
        )
        dataframe["bb_width"] = (
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"]) / dataframe["bb_middleband"]
        )
        
        dataframe['bb_buy_entry_level'] = dataframe['bb_lowerband'] + ((dataframe['bb_middleband'] - dataframe['bb_lowerband']) * 0.3)

        # TEMA - Triple Exponential Moving Average
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)
        
        #Technical
        dataframe['laguerre'] = laguerre(dataframe, gamma=0.2, smooth=1)
        dataframe["rsi"] = ta.RSI(dataframe)
        StochRSI = ta.STOCHRSI(dataframe)
        dataframe["stochrsi_fastk"] = StochRSI["fastk"]
        dataframe["stochrsi_fastd"] = StochRSI["fastd"]
        
        madr = MADR(dataframe)
        dataframe["stdcenter"] = madr['stdcenter']
        dataframe["plusdev"] = madr['plusdev']
        dataframe["minusdev"] = madr['minusdev']
        dataframe["rate"] = madr['rate']
                
        dataframe.to_csv('/freqtrade/user_data/populate_indicators.csv', sep=',', encoding='utf-8')

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[
            (
                #1st Condition,  inside interval 
                (      
                (df['rate'] > df['rate'].shift()) &
                ((df['rate'] > df['minusdev']) | (df['rate'].shift() > df['minusdev']))&
                (df['tema'] < df['bb_buy_entry_level']) &  # Guard: tema 30% from de BB lowerband
                (df['do_predict'] == 1) #&  # Make sure Freqai is confident in the prediction
                )|(
                #2nd Condition
                # Signal: RMI inside 
                (df['rsi'] >= self.buy_rsi.value) & 
                (df['rsi'] < self.sell_rsi.value) &
                (df['tema'] <  df['bb_buy_entry_level']) &  # Guard: tema 30% from de BB lowerband
                (df['tema'] > df['tema'].shift(1)) &  # Guard: tema is raising
                (df['volume'] > 0) &  # Make sure Volume is not 0
                (df['laguerre'] >= self.buy_laguerre.value) & 
                (df['laguerre'] < 0.8) & 
                (df['laguerre'] > df['laguerre'].shift(1)) &
                (df['do_predict'] == 1) #&  # Make sure Freqai is confident in the prediction
                # Only enter trade if Freqai thinks the trend is in this direction
                # (df['&-s-up_or_down'] == 'up')
                )
            ),
            'enter_long'] = 1

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[
            (
                #1st Condition,
                (df['rate'] >= df['plusdev'])
                #2nd Condition
                |(
                # Signal: RSI crosses above 70
                # (qtpylib.crossed_above(df['rsi'], self.sell_rsi.value)) &
                (df['stochrsi_fastk'] < df['stochrsi_fastk'].shift(1)) &
                (df['tema'] < df['tema'].shift(1)) &  # Guard: tema is falling
                (df['volume'] > 0) &  # Make sure Volume is not 0
                (df['laguerre'] < df['laguerre'].shift(1))
                )
            ),

            'exit_long'] = 1

        return df



