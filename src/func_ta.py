import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=Warning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
from ta import add_all_ta_features
from ta.utils import dropna
from ta.momentum import RSIIndicator, ROCIndicator, WilliamsRIndicator, \
                        AwesomeOscillatorIndicator, KAMAIndicator, PercentagePriceOscillator, \
                        StochRSIIndicator, StochasticOscillator, TSIIndicator, \
                        UltimateOscillator
from ta.trend import CCIIndicator, MACD, ADXIndicator, AroonIndicator, DPOIndicator, EMAIndicator, \
                     IchimokuIndicator, KSTIndicator, MassIndex, PSARIndicator, SMAIndicator, \
                     STCIndicator, TRIXIndicator, VortexIndicator, WMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands, DonchianChannel, KeltnerChannel, UlcerIndex

def calcTA(data, window, to_dropna=True, 
           basic=False, momentum=False, trend=False, volatility=False):
    '''
    Calculate all interested TA indicators on OHLC data.
    '''
    data['return'] = np.log(data['c'] / data['c'].shift(1))
    data['vol'] = data['return'].rolling(window).std()

    if basic:
        ## Basic
        data['mom'] = np.sign(data['return'].rolling(window).mean())
        data['sma'] = data['c'].rolling(window).mean()
        data['min'] = data['c'].rolling(window).min()
        data['max'] = data['c'].rolling(window).max()

    if momentum:
        ## Momentum
        data['rsi14'] = RSIIndicator(data['c']).rsi()
        data['roc12'] = ROCIndicator(data['c']).roc()
        data['wr14'] = WilliamsRIndicator(data['h'], data['l'], data['c']).williams_r()
        data['aoi5_34'] = AwesomeOscillatorIndicator(data['h'], data['l']).awesome_oscillator()
        data['ppo26_12_9'] = PercentagePriceOscillator(data['c']).ppo()
        data['stochoscill14_3'] = StochasticOscillator(data['h'], data['l'], data['c']).stoch()
        data['tsi25_13'] = TSIIndicator(data['c']).tsi()
        data['ultimateosc'] = UltimateOscillator(data['h'], data['l'], data['c']).ultimate_oscillator()
        ## Using own formula
        data['kama10_2_30'] = calcKAMA(data['c'])
        ## Significant NAs
        # data['stochrsi14_3_3'] = StochRSIIndicator(data['c']).stochrsi()    

    if trend:
        ## Trend
        data['macd'] = MACD(data['c']).macd()
        data['macd_diff'] = MACD(data['c']).macd_diff()
        data['macd_signal'] = MACD(data['c']).macd_signal()
        data['adx'] = ADXIndicator(data['h'], data['l'], data['c'], fillna=True).adx()
        data['adx_neg'] = ADXIndicator(data['h'], data['l'], data['c'], fillna=True).adx_neg()
        data['adx_pos'] = ADXIndicator(data['h'], data['l'], data['c'], fillna=True).adx_pos()
        data['aroon_down'] = AroonIndicator(data['c']).aroon_down()
        data['aroon_indicator'] = AroonIndicator(data['c']).aroon_indicator()
        data['aroon_up'] = AroonIndicator(data['c']).aroon_up()
        data['dpo'] = DPOIndicator(data['c']).dpo()
        data['ema_indicator'] = EMAIndicator(data['c']).ema_indicator()
        data['ichimoku_a'] = IchimokuIndicator(data['h'], data['l']).ichimoku_a()
        data['ichimoku_b'] = IchimokuIndicator(data['h'], data['l']).ichimoku_b()
        data['ichimoku_base_line'] = IchimokuIndicator(data['h'], data['l']).ichimoku_base_line()
        data['ichimoku_conversion_line'] = IchimokuIndicator(data['h'], data['l']).ichimoku_conversion_line()
        data['kst'] = KSTIndicator(data['c']).kst()
        data['kst_diff'] = KSTIndicator(data['c']).kst_diff()
        data['kst_sig'] = KSTIndicator(data['c']).kst_sig()
        data['mass_index'] = MassIndex(data['h'], data['l']).mass_index()
        data['psar'] = PSARIndicator(data['h'], data['l'], data['c']).psar()
        data['psar_down_indicator'] = PSARIndicator(data['h'], data['l'], data['c']).psar_down_indicator()
        data['psar_up_indicator'] = PSARIndicator(data['h'], data['l'], data['c']).psar_up_indicator()
        data['sma_indicator'] = SMAIndicator(data['c'], window).sma_indicator()
        data['trix'] = TRIXIndicator(data['c']).trix()
        data['vortex_indicator_diff'] = VortexIndicator(data['h'], data['l'], data['c']).vortex_indicator_diff()
        data['vortex_indicator_neg'] = VortexIndicator(data['h'], data['l'], data['c']).vortex_indicator_neg()
        data['vortex_indicator_pos'] = VortexIndicator(data['h'], data['l'], data['c']).vortex_indicator_pos()
        data['wma'] = WMAIndicator(data['c'], window).wma()
        ## Significant NAs
        # data['cci'] = CCIIndicator(data['h'], data['l'], data['c']).cci()
        # data['psar_down'] = PSARIndicator(data['h'], data['l'], data['c']).psar_down()
        # data['psar_up'] = PSARIndicator(data['h'], data['l'], data['c']).psar_up()
        # data['stc'] = STCIndicator(data['c']).stc()

    if volatility:
        ## Volatility
        data['bb_hband'] = BollingerBands(data['c']).bollinger_hband()
        data['bb_high'] = BollingerBands(data['c']).bollinger_hband_indicator()
        data['bb_lband'] = BollingerBands(data['c']).bollinger_lband()
        data['bb_low'] = BollingerBands(data['c']).bollinger_lband_indicator()
        data['bb_mavg'] = BollingerBands(data['c']).bollinger_mavg()
        data['bb_wband'] = BollingerBands(data['c']).bollinger_wband()
        data['dc_hband'] = DonchianChannel(data['h'], data['l'], data['c']).donchian_channel_hband()
        data['dc_lband'] = DonchianChannel(data['h'], data['l'], data['c']).donchian_channel_lband()
        data['dc_mband'] = DonchianChannel(data['h'], data['l'], data['c']).donchian_channel_mband()
        data['dc_pband'] = DonchianChannel(data['h'], data['l'], data['c']).donchian_channel_pband()
        data['dc_wband'] = DonchianChannel(data['h'], data['l'], data['c']).donchian_channel_wband()
        data['kc_hband'] = KeltnerChannel(data['h'], data['l'], data['c']).keltner_channel_hband()
        data['kc_hband_ind'] = KeltnerChannel(data['h'], data['l'], data['c']).keltner_channel_hband_indicator()
        data['kc_lband'] = KeltnerChannel(data['h'], data['l'], data['c']).keltner_channel_lband()
        data['kc_lband_ind'] = KeltnerChannel(data['h'], data['l'], data['c']).keltner_channel_lband_indicator() 
        data['kc_mband'] = KeltnerChannel(data['h'], data['l'], data['c']).keltner_channel_mband() 
        data['kc_pband'] = KeltnerChannel(data['h'], data['l'], data['c']).keltner_channel_pband() 
        data['kc_wband'] = KeltnerChannel(data['h'], data['l'], data['c']).keltner_channel_wband() 
        data['ulcer14'] = UlcerIndex(data['c']).ulcer_index()
        data['atr'] = AverageTrueRange(data['h'], data['l'], data['c']).average_true_range()
        ## HAS INFINITY VALUES
        # data['bb_pband'] = BollingerBands(data['c']).bollinger_pband()  

    if to_dropna:
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(inplace=True)

    return data


def resampleOHLC(raw, bar_length, volume=False):
    '''
    Resampling OHLC data.  
    Raw data to have column names as: o, h, l , c, volume(if required). 
    '''
    
    cols = ['o', 'h', 'l', 'c']
    if volume:
        cols += ['volume']
        data = raw[cols]
        resample_rules = {'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last', 'volume': 'last'}
        data = data.resample(bar_length).agg(resample_rules)

    else:
        data = raw[cols]
        resample_rules = {'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last'}
        data = data.resample(bar_length).agg(resample_rules)

    data.fillna(method='ffill', inplace=True)
    data = data.iloc[:-1]

    return data    


def calcKAMA(series, window=10, fast=2, slow=30, period=1):
    ''' 
    Calculate the Kaufman’s Adaptive Moving Average (KAMA)
    https://corporatefinanceinstitute.com/resources/knowledge/trading-investing/kaufmans-adaptive-moving-average-kama/
    
    Inputs
    ======
    series: pd.Series to calculate the KAMA indicator
    window: the number of periods for the Efficiency Ratio (ER)
    fast: the number of periods for the fastest EMA constant
    slow: the number of periods for the slowest EMA constant
    period: the simple moving average value use for the first kama value
    
    Return
    ======
    pd.Series of kama value maintaining the index of the input series
    '''
    
    idx = series.index

    series = series.to_frame()
    series.columns = ['c']
    series['ema_fast'] = series['c'].ewm(fast, adjust=False).mean()
    series['ema_slow'] = series['c'].ewm(slow, adjust=False).mean()
    series['change'] = abs(series['c'] - series['c'].shift(window))
    series['volatility'] = (series['c'] - series['c'].shift()).abs().rolling(window).sum()
    series['er'] = series['change']/series['volatility']
#     series['er'] = series.apply(lambda x: x['change']/x['volatility'] if x['volatility'] != 0 else 0, axis=1)
    
    sc_fast = 2/(fast+1)
    sc_slow = 2/(slow+1)

    series['sc'] = (series['er'] * (sc_fast - sc_slow) + sc_slow) ** 2
    series['sma'] = series['c'].rolling(period).mean()
    series['sma_prior'] = series['sma'].shift(1)
    series.dropna(inplace=True)

    prior_kama = series.sma_prior[0]
    kama = [prior_kama]

    for index, row in series[1:].iterrows():
        current_kama = prior_kama + row['sc'] * (row['c'] - prior_kama)
        kama.append(current_kama)
        prior_kama = current_kama

    series['kama'] = kama    
    output = pd.concat([pd.DataFrame([], index=idx), series['kama']], axis=1)
    output = output.pad()
    
    return output['kama']    
