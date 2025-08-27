import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from .base_strategy import TechnicalStrategy

logger = logging.getLogger(__name__)

class RSIStrategy(TechnicalStrategy):
    """RSI-based trading strategy"""
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'rsi_period': 14,
            'oversold_threshold': 30,
            'overbought_threshold': 70,
            'exit_oversold': 50,
            'exit_overbought': 50
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("RSI", default_params)
    
    def get_minimum_data_length(self) -> int:
        return self.get_parameter('rsi_period', 14) + 10
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI indicator"""
        df = data.copy()
        df['rsi'] = self.calculate_rsi(df['close'], self.get_parameter('rsi_period', 14))
        return df
    
    def generate_signal(self, data: pd.DataFrame, current_price: float, 
                       portfolio: Dict) -> Dict:
        """Generate RSI-based trading signal"""
        try:
            if not self.validate_data(data):
                return {'action': 'hold', 'confidence': 0, 'reasoning': 'Invalid data'}
            
            df = self.calculate_indicators(data)
            current_rsi = df['rsi'].iloc[-1]
            prev_rsi = df['rsi'].iloc[-2]
            
            oversold = self.get_parameter('oversold_threshold', 30)
            overbought = self.get_parameter('overbought_threshold', 70)
            exit_oversold = self.get_parameter('exit_oversold', 50)
            exit_overbought = self.get_parameter('exit_overbought', 50)
            
            signal = {'action': 'hold', 'confidence': 0.5, 'reasoning': ''}
            
            # Buy signal: RSI crosses above oversold level
            if prev_rsi <= oversold and current_rsi > oversold:
                confidence = min(0.9, (oversold - current_rsi + 20) / 20)
                signal = {
                    'action': 'buy',
                    'confidence': confidence,
                    'stop_loss': current_price * 0.95,
                    'take_profit': current_price * 1.10,
                    'reasoning': f'RSI oversold reversal: {current_rsi:.2f}'
                }
            
            # Sell signal: RSI crosses below overbought level
            elif prev_rsi >= overbought and current_rsi < overbought:
                confidence = min(0.9, (current_rsi - overbought + 20) / 20)
                signal = {
                    'action': 'sell',
                    'confidence': confidence,
                    'stop_loss': current_price * 1.05,
                    'take_profit': current_price * 0.90,
                    'reasoning': f'RSI overbought reversal: {current_rsi:.2f}'
                }
            
            # Exit long position
            elif current_rsi > exit_overbought:
                signal = {
                    'action': 'sell',
                    'confidence': 0.7,
                    'reasoning': f'RSI exit signal: {current_rsi:.2f}'
                }
            
            # Exit short position
            elif current_rsi < exit_oversold:
                signal = {
                    'action': 'buy',
                    'confidence': 0.7,
                    'reasoning': f'RSI cover signal: {current_rsi:.2f}'
                }
            
            self.record_signal(signal, current_price)
            return signal
            
        except Exception as e:
            logger.error(f"Error generating RSI signal: {e}")
            return {'action': 'hold', 'confidence': 0, 'reasoning': f'Error: {e}'}


class MACDStrategy(TechnicalStrategy):
    """MACD-based trading strategy"""
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9,
            'histogram_threshold': 0.001
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("MACD", default_params)
    
    def get_minimum_data_length(self) -> int:
        return self.get_parameter('slow_period', 26) + 10
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD indicators"""
        df = data.copy()
        fast = self.get_parameter('fast_period', 12)
        slow = self.get_parameter('slow_period', 26)
        signal_period = self.get_parameter('signal_period', 9)
        
        df['macd'], df['macd_signal'], df['macd_histogram'] = self.calculate_macd(
            df['close'], fast, slow, signal_period
        )
        return df
    
    def generate_signal(self, data: pd.DataFrame, current_price: float, 
                       portfolio: Dict) -> Dict:
        """Generate MACD-based trading signal"""
        try:
            if not self.validate_data(data):
                return {'action': 'hold', 'confidence': 0, 'reasoning': 'Invalid data'}
            
            df = self.calculate_indicators(data)
            
            current_macd = df['macd'].iloc[-1]
            current_signal = df['macd_signal'].iloc[-1]
            current_histogram = df['macd_histogram'].iloc[-1]
            prev_histogram = df['macd_histogram'].iloc[-2]
            
            histogram_threshold = self.get_parameter('histogram_threshold', 0.001)
            
            signal = {'action': 'hold', 'confidence': 0.5, 'reasoning': ''}
            
            # Buy signal: MACD line crosses above signal line
            if prev_histogram <= 0 and current_histogram > 0:
                confidence = min(0.9, abs(current_histogram) * 1000)
                signal = {
                    'action': 'buy',
                    'confidence': confidence,
                    'stop_loss': current_price * 0.95,
                    'take_profit': current_price * 1.10,
                    'reasoning': f'MACD bullish crossover: {current_histogram:.4f}'
                }
            
            # Sell signal: MACD line crosses below signal line
            elif prev_histogram >= 0 and current_histogram < 0:
                confidence = min(0.9, abs(current_histogram) * 1000)
                signal = {
                    'action': 'sell',
                    'confidence': confidence,
                    'stop_loss': current_price * 1.05,
                    'take_profit': current_price * 0.90,
                    'reasoning': f'MACD bearish crossover: {current_histogram:.4f}'
                }
            
            # Strong bullish momentum
            elif current_histogram > histogram_threshold and current_macd > current_signal:
                signal = {
                    'action': 'buy',
                    'confidence': 0.6,
                    'stop_loss': current_price * 0.97,
                    'take_profit': current_price * 1.06,
                    'reasoning': f'MACD strong bullish momentum'
                }
            
            # Strong bearish momentum
            elif current_histogram < -histogram_threshold and current_macd < current_signal:
                signal = {
                    'action': 'sell',
                    'confidence': 0.6,
                    'stop_loss': current_price * 1.03,
                    'take_profit': current_price * 0.94,
                    'reasoning': f'MACD strong bearish momentum'
                }
            
            self.record_signal(signal, current_price)
            return signal
            
        except Exception as e:
            logger.error(f"Error generating MACD signal: {e}")
            return {'action': 'hold', 'confidence': 0, 'reasoning': f'Error: {e}'}


class BollingerBandsStrategy(TechnicalStrategy):
    """Bollinger Bands trading strategy"""
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'period': 20,
            'std_dev': 2,
            'squeeze_threshold': 0.02
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("BollingerBands", default_params)
    
    def get_minimum_data_length(self) -> int:
        return self.get_parameter('period', 20) + 10
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        df = data.copy()
        period = self.get_parameter('period', 20)
        std_dev = self.get_parameter('std_dev', 2)
        
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self.calculate_bollinger_bands(
            df['close'], period, std_dev
        )
        
        # Calculate band width for squeeze detection
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        return df
    
    def generate_signal(self, data: pd.DataFrame, current_price: float, 
                       portfolio: Dict) -> Dict:
        """Generate Bollinger Bands trading signal"""
        try:
            if not self.validate_data(data):
                return {'action': 'hold', 'confidence': 0, 'reasoning': 'Invalid data'}
            
            df = self.calculate_indicators(data)
            
            current_close = df['close'].iloc[-1]
            bb_upper = df['bb_upper'].iloc[-1]
            bb_lower = df['bb_lower'].iloc[-1]
            bb_middle = df['bb_middle'].iloc[-1]
            bb_width = df['bb_width'].iloc[-1]
            
            squeeze_threshold = self.get_parameter('squeeze_threshold', 0.02)
            
            signal = {'action': 'hold', 'confidence': 0.5, 'reasoning': ''}
            
            # Price touches lower band - potential buy
            if current_close <= bb_lower:
                distance_from_band = abs(current_close - bb_lower) / bb_lower
                confidence = max(0.6, 0.9 - distance_from_band * 10)
                signal = {
                    'action': 'buy',
                    'confidence': confidence,
                    'stop_loss': current_price * 0.95,
                    'take_profit': bb_middle,
                    'reasoning': f'Price at lower Bollinger Band: {current_close:.2f}'
                }
            
            # Price touches upper band - potential sell
            elif current_close >= bb_upper:
                distance_from_band = abs(current_close - bb_upper) / bb_upper
                confidence = max(0.6, 0.9 - distance_from_band * 10)
                signal = {
                    'action': 'sell',
                    'confidence': confidence,
                    'stop_loss': current_price * 1.05,
                    'take_profit': bb_middle,
                    'reasoning': f'Price at upper Bollinger Band: {current_close:.2f}'
                }
            
            # Bollinger squeeze breakout
            elif bb_width < squeeze_threshold:
                # Determine breakout direction
                if current_close > bb_middle:
                    signal = {
                        'action': 'buy',
                        'confidence': 0.7,
                        'stop_loss': bb_lower,
                        'take_profit': current_price * 1.08,
                        'reasoning': f'Bollinger squeeze bullish breakout'
                    }
                else:
                    signal = {
                        'action': 'sell',
                        'confidence': 0.7,
                        'stop_loss': bb_upper,
                        'take_profit': current_price * 0.92,
                        'reasoning': f'Bollinger squeeze bearish breakout'
                    }
            
            self.record_signal(signal, current_price)
            return signal
            
        except Exception as e:
            logger.error(f"Error generating Bollinger Bands signal: {e}")
            return {'action': 'hold', 'confidence': 0, 'reasoning': f'Error: {e}'}


class MovingAverageCrossoverStrategy(TechnicalStrategy):
    """Moving Average Crossover Strategy"""
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'fast_period': 10,
            'slow_period': 30,
            'ma_type': 'sma'  # 'sma' or 'ema'
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("MovingAverageCrossover", default_params)
    
    def get_minimum_data_length(self) -> int:
        return self.get_parameter('slow_period', 30) + 10
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate moving averages"""
        df = data.copy()
        fast_period = self.get_parameter('fast_period', 10)
        slow_period = self.get_parameter('slow_period', 30)
        ma_type = self.get_parameter('ma_type', 'sma')
        
        if ma_type == 'ema':
            df['ma_fast'] = self.calculate_ema(df['close'], fast_period)
            df['ma_slow'] = self.calculate_ema(df['close'], slow_period)
        else:
            df['ma_fast'] = self.calculate_sma(df['close'], fast_period)
            df['ma_slow'] = self.calculate_sma(df['close'], slow_period)
        
        return df
    
    def generate_signal(self, data: pd.DataFrame, current_price: float, 
                       portfolio: Dict) -> Dict:
        """Generate moving average crossover signal"""
        try:
            if not self.validate_data(data):
                return {'action': 'hold', 'confidence': 0, 'reasoning': 'Invalid data'}
            
            df = self.calculate_indicators(data)
            
            current_fast = df['ma_fast'].iloc[-1]
            current_slow = df['ma_slow'].iloc[-1]
            prev_fast = df['ma_fast'].iloc[-2]
            prev_slow = df['ma_slow'].iloc[-2]
            
            signal = {'action': 'hold', 'confidence': 0.5, 'reasoning': ''}
            
            # Golden cross: fast MA crosses above slow MA
            if prev_fast <= prev_slow and current_fast > current_slow:
                confidence = min(0.9, (current_fast - current_slow) / current_slow * 20)
                signal = {
                    'action': 'buy',
                    'confidence': confidence,
                    'stop_loss': current_slow,
                    'take_profit': current_price * 1.10,
                    'reasoning': f'Golden cross: Fast MA {current_fast:.2f} > Slow MA {current_slow:.2f}'
                }
            
            # Death cross: fast MA crosses below slow MA
            elif prev_fast >= prev_slow and current_fast < current_slow:
                confidence = min(0.9, (current_slow - current_fast) / current_slow * 20)
                signal = {
                    'action': 'sell',
                    'confidence': confidence,
                    'stop_loss': current_slow,
                    'take_profit': current_price * 0.90,
                    'reasoning': f'Death cross: Fast MA {current_fast:.2f} < Slow MA {current_slow:.2f}'
                }
            
            # Trend continuation signals
            elif current_fast > current_slow and current_price > current_fast:
                signal = {
                    'action': 'buy',
                    'confidence': 0.6,
                    'stop_loss': current_fast,
                    'take_profit': current_price * 1.05,
                    'reasoning': f'Bullish trend continuation'
                }
            
            elif current_fast < current_slow and current_price < current_fast:
                signal = {
                    'action': 'sell',
                    'confidence': 0.6,
                    'stop_loss': current_fast,
                    'take_profit': current_price * 0.95,
                    'reasoning': f'Bearish trend continuation'
                }
            
            self.record_signal(signal, current_price)
            return signal
            
        except Exception as e:
            logger.error(f"Error generating MA crossover signal: {e}")
            return {'action': 'hold', 'confidence': 0, 'reasoning': f'Error: {e}'}


class StochasticStrategy(TechnicalStrategy):
    """Stochastic Oscillator Strategy"""
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'k_period': 14,
            'd_period': 3,
            'oversold_threshold': 20,
            'overbought_threshold': 80
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("Stochastic", default_params)
    
    def get_minimum_data_length(self) -> int:
        return self.get_parameter('k_period', 14) + 10
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Stochastic Oscillator"""
        df = data.copy()
        k_period = self.get_parameter('k_period', 14)
        d_period = self.get_parameter('d_period', 3)
        
        df['stoch_k'], df['stoch_d'] = self.calculate_stochastic(
            df['high'], df['low'], df['close'], k_period, d_period
        )
        
        return df
    
    def generate_signal(self, data: pd.DataFrame, current_price: float, 
                       portfolio: Dict) -> Dict:
        """Generate Stochastic-based trading signal"""
        try:
            if not self.validate_data(data):
                return {'action': 'hold', 'confidence': 0, 'reasoning': 'Invalid data'}
            
            df = self.calculate_indicators(data)
            
            current_k = df['stoch_k'].iloc[-1]
            current_d = df['stoch_d'].iloc[-1]
            prev_k = df['stoch_k'].iloc[-2]
            prev_d = df['stoch_d'].iloc[-2]
            
            oversold = self.get_parameter('oversold_threshold', 20)
            overbought = self.get_parameter('overbought_threshold', 80)
            
            signal = {'action': 'hold', 'confidence': 0.5, 'reasoning': ''}
            
            # Buy signal: %K crosses above %D in oversold region
            if (current_k < oversold or current_d < oversold) and prev_k <= prev_d and current_k > current_d:
                confidence = min(0.9, (oversold - min(current_k, current_d)) / oversold + 0.3)
                signal = {
                    'action': 'buy',
                    'confidence': confidence,
                    'stop_loss': current_price * 0.95,
                    'take_profit': current_price * 1.10,
                    'reasoning': f'Stochastic oversold reversal: %K {current_k:.1f}, %D {current_d:.1f}'
                }
            
            # Sell signal: %K crosses below %D in overbought region
            elif (current_k > overbought or current_d > overbought) and prev_k >= prev_d and current_k < current_d:
                confidence = min(0.9, (max(current_k, current_d) - overbought) / (100 - overbought) + 0.3)
                signal = {
                    'action': 'sell',
                    'confidence': confidence,
                    'stop_loss': current_price * 1.05,
                    'take_profit': current_price * 0.90,
                    'reasoning': f'Stochastic overbought reversal: %K {current_k:.1f}, %D {current_d:.1f}'
                }
            
            # Momentum signals
            elif current_k > current_d and current_k > 50 and prev_k <= prev_d:
                signal = {
                    'action': 'buy',
                    'confidence': 0.6,
                    'stop_loss': current_price * 0.97,
                    'take_profit': current_price * 1.06,
                    'reasoning': f'Stochastic bullish crossover'
                }
            
            elif current_k < current_d and current_k < 50 and prev_k >= prev_d:
                signal = {
                    'action': 'sell',
                    'confidence': 0.6,
                    'stop_loss': current_price * 1.03,
                    'take_profit': current_price * 0.94,
                    'reasoning': f'Stochastic bearish crossover'
                }
            
            self.record_signal(signal, current_price)
            return signal
            
        except Exception as e:
            logger.error(f"Error generating Stochastic signal: {e}")
            return {'action': 'hold', 'confidence': 0, 'reasoning': f'Error: {e}'}