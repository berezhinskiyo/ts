import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from .base_strategy import TechnicalStrategy

logger = logging.getLogger(__name__)

class MomentumStrategy(TechnicalStrategy):
    """Price momentum strategy"""
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'lookback_period': 10,
            'momentum_threshold': 0.02,
            'volume_confirmation': True
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("Momentum", default_params)
    
    def get_minimum_data_length(self) -> int:
        return self.get_parameter('lookback_period', 10) + 5
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators"""
        df = data.copy()
        lookback = self.get_parameter('lookback_period', 10)
        
        # Price momentum
        df['price_momentum'] = df['close'].pct_change(lookback)
        
        # Volume momentum
        df['volume_momentum'] = df['volume'].pct_change(lookback)
        
        # Rate of change
        df['roc'] = (df['close'] - df['close'].shift(lookback)) / df['close'].shift(lookback) * 100
        
        # Momentum oscillator
        df['momentum_osc'] = df['close'] / df['close'].shift(lookback) * 100
        
        return df
    
    def generate_signal(self, data: pd.DataFrame, current_price: float, 
                       portfolio: Dict) -> Dict:
        """Generate momentum-based trading signal"""
        try:
            if not self.validate_data(data):
                return {'action': 'hold', 'confidence': 0, 'reasoning': 'Invalid data'}
            
            df = self.calculate_indicators(data)
            
            current_momentum = df['price_momentum'].iloc[-1]
            volume_momentum = df['volume_momentum'].iloc[-1]
            roc = df['roc'].iloc[-1]
            
            momentum_threshold = self.get_parameter('momentum_threshold', 0.02)
            volume_confirmation = self.get_parameter('volume_confirmation', True)
            
            signal = {'action': 'hold', 'confidence': 0.5, 'reasoning': ''}
            
            # Strong positive momentum
            if current_momentum > momentum_threshold:
                confidence = min(0.9, current_momentum * 10)
                
                # Volume confirmation
                if volume_confirmation and volume_momentum > 0:
                    confidence *= 1.2
                elif volume_confirmation and volume_momentum <= 0:
                    confidence *= 0.8
                
                signal = {
                    'action': 'buy',
                    'confidence': min(confidence, 0.9),
                    'stop_loss': current_price * (1 - momentum_threshold),
                    'take_profit': current_price * (1 + momentum_threshold * 2),
                    'reasoning': f'Strong positive momentum: {current_momentum:.3f}, ROC: {roc:.2f}'
                }
            
            # Strong negative momentum
            elif current_momentum < -momentum_threshold:
                confidence = min(0.9, abs(current_momentum) * 10)
                
                # Volume confirmation
                if volume_confirmation and volume_momentum > 0:
                    confidence *= 1.2
                elif volume_confirmation and volume_momentum <= 0:
                    confidence *= 0.8
                
                signal = {
                    'action': 'sell',
                    'confidence': min(confidence, 0.9),
                    'stop_loss': current_price * (1 + momentum_threshold),
                    'take_profit': current_price * (1 - momentum_threshold * 2),
                    'reasoning': f'Strong negative momentum: {current_momentum:.3f}, ROC: {roc:.2f}'
                }
            
            self.record_signal(signal, current_price)
            return signal
            
        except Exception as e:
            logger.error(f"Error generating momentum signal: {e}")
            return {'action': 'hold', 'confidence': 0, 'reasoning': f'Error: {e}'}


class MeanReversionStrategy(TechnicalStrategy):
    """Mean reversion strategy"""
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'lookback_period': 20,
            'deviation_threshold': 2.0,
            'mean_type': 'sma'  # 'sma' or 'ema'
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("MeanReversion", default_params)
    
    def get_minimum_data_length(self) -> int:
        return self.get_parameter('lookback_period', 20) + 10
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate mean reversion indicators"""
        df = data.copy()
        lookback = self.get_parameter('lookback_period', 20)
        mean_type = self.get_parameter('mean_type', 'sma')
        
        # Calculate mean
        if mean_type == 'ema':
            df['mean'] = self.calculate_ema(df['close'], lookback)
        else:
            df['mean'] = self.calculate_sma(df['close'], lookback)
        
        # Calculate standard deviation
        df['std'] = df['close'].rolling(window=lookback).std()
        
        # Z-score (how many standard deviations from mean)
        df['z_score'] = (df['close'] - df['mean']) / df['std']
        
        # Deviation from mean
        df['deviation'] = (df['close'] - df['mean']) / df['mean']
        
        return df
    
    def generate_signal(self, data: pd.DataFrame, current_price: float, 
                       portfolio: Dict) -> Dict:
        """Generate mean reversion signal"""
        try:
            if not self.validate_data(data):
                return {'action': 'hold', 'confidence': 0, 'reasoning': 'Invalid data'}
            
            df = self.calculate_indicators(data)
            
            current_z_score = df['z_score'].iloc[-1]
            current_mean = df['mean'].iloc[-1]
            current_deviation = df['deviation'].iloc[-1]
            
            deviation_threshold = self.get_parameter('deviation_threshold', 2.0)
            
            signal = {'action': 'hold', 'confidence': 0.5, 'reasoning': ''}
            
            # Price significantly below mean - buy signal
            if current_z_score < -deviation_threshold:
                confidence = min(0.9, abs(current_z_score) / 4)
                signal = {
                    'action': 'buy',
                    'confidence': confidence,
                    'stop_loss': current_price * 0.95,
                    'take_profit': current_mean,
                    'reasoning': f'Price below mean: Z-score {current_z_score:.2f}, deviation {current_deviation:.3f}'
                }
            
            # Price significantly above mean - sell signal
            elif current_z_score > deviation_threshold:
                confidence = min(0.9, current_z_score / 4)
                signal = {
                    'action': 'sell',
                    'confidence': confidence,
                    'stop_loss': current_price * 1.05,
                    'take_profit': current_mean,
                    'reasoning': f'Price above mean: Z-score {current_z_score:.2f}, deviation {current_deviation:.3f}'
                }
            
            # Moderate reversion signals
            elif current_z_score < -1.5:
                signal = {
                    'action': 'buy',
                    'confidence': 0.6,
                    'stop_loss': current_price * 0.97,
                    'take_profit': current_price * 1.03,
                    'reasoning': f'Moderate oversold: Z-score {current_z_score:.2f}'
                }
            
            elif current_z_score > 1.5:
                signal = {
                    'action': 'sell',
                    'confidence': 0.6,
                    'stop_loss': current_price * 1.03,
                    'take_profit': current_price * 0.97,
                    'reasoning': f'Moderate overbought: Z-score {current_z_score:.2f}'
                }
            
            self.record_signal(signal, current_price)
            return signal
            
        except Exception as e:
            logger.error(f"Error generating mean reversion signal: {e}")
            return {'action': 'hold', 'confidence': 0, 'reasoning': f'Error: {e}'}


class BreakoutStrategy(TechnicalStrategy):
    """Support/Resistance breakout strategy"""
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'lookback_period': 20,
            'min_touches': 2,
            'breakout_threshold': 0.01,
            'volume_multiplier': 1.5
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("Breakout", default_params)
    
    def get_minimum_data_length(self) -> int:
        return self.get_parameter('lookback_period', 20) + 10
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate breakout indicators"""
        df = data.copy()
        lookback = self.get_parameter('lookback_period', 20)
        
        # Support and resistance levels
        df['resistance'] = df['high'].rolling(window=lookback).max()
        df['support'] = df['low'].rolling(window=lookback).min()
        
        # Average volume
        df['avg_volume'] = df['volume'].rolling(window=lookback).mean()
        
        # Price range
        df['range_high'] = df['high'].rolling(window=lookback).max()
        df['range_low'] = df['low'].rolling(window=lookback).min()
        df['range_size'] = (df['range_high'] - df['range_low']) / df['close']
        
        return df
    
    def generate_signal(self, data: pd.DataFrame, current_price: float, 
                       portfolio: Dict) -> Dict:
        """Generate breakout signal"""
        try:
            if not self.validate_data(data):
                return {'action': 'hold', 'confidence': 0, 'reasoning': 'Invalid data'}
            
            df = self.calculate_indicators(data)
            
            current_high = df['high'].iloc[-1]
            current_low = df['low'].iloc[-1]
            current_volume = df['volume'].iloc[-1]
            resistance = df['resistance'].iloc[-2]  # Previous resistance
            support = df['support'].iloc[-2]  # Previous support
            avg_volume = df['avg_volume'].iloc[-1]
            
            breakout_threshold = self.get_parameter('breakout_threshold', 0.01)
            volume_multiplier = self.get_parameter('volume_multiplier', 1.5)
            
            signal = {'action': 'hold', 'confidence': 0.5, 'reasoning': ''}
            
            # Resistance breakout
            if current_high > resistance * (1 + breakout_threshold):
                confidence = 0.7
                
                # Volume confirmation
                if current_volume > avg_volume * volume_multiplier:
                    confidence = 0.9
                
                signal = {
                    'action': 'buy',
                    'confidence': confidence,
                    'stop_loss': resistance,
                    'take_profit': current_price * (1 + (current_price - resistance) / resistance),
                    'reasoning': f'Resistance breakout: {current_high:.2f} > {resistance:.2f}'
                }
            
            # Support breakdown
            elif current_low < support * (1 - breakout_threshold):
                confidence = 0.7
                
                # Volume confirmation
                if current_volume > avg_volume * volume_multiplier:
                    confidence = 0.9
                
                signal = {
                    'action': 'sell',
                    'confidence': confidence,
                    'stop_loss': support,
                    'take_profit': current_price * (1 - (support - current_price) / support),
                    'reasoning': f'Support breakdown: {current_low:.2f} < {support:.2f}'
                }
            
            self.record_signal(signal, current_price)
            return signal
            
        except Exception as e:
            logger.error(f"Error generating breakout signal: {e}")
            return {'action': 'hold', 'confidence': 0, 'reasoning': f'Error: {e}'}


class VolumeProfileStrategy(TechnicalStrategy):
    """Volume profile strategy"""
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'lookback_period': 50,
            'volume_threshold': 1.5,
            'price_bins': 20
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("VolumeProfile", default_params)
    
    def get_minimum_data_length(self) -> int:
        return self.get_parameter('lookback_period', 50)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume profile indicators"""
        df = data.copy()
        lookback = self.get_parameter('lookback_period', 50)
        price_bins = self.get_parameter('price_bins', 20)
        
        # Volume indicators
        df = self.calculate_volume_indicators(df)
        
        # Volume-weighted average price
        df['vwap'] = (df['close'] * df['volume']).rolling(window=lookback).sum() / \
                     df['volume'].rolling(window=lookback).sum()
        
        return df
    
    def generate_signal(self, data: pd.DataFrame, current_price: float, 
                       portfolio: Dict) -> Dict:
        """Generate volume profile signal"""
        try:
            if not self.validate_data(data):
                return {'action': 'hold', 'confidence': 0, 'reasoning': 'Invalid data'}
            
            df = self.calculate_indicators(data)
            
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume_sma'].iloc[-1]
            vwap = df['vwap'].iloc[-1]
            obv = df['obv'].iloc[-1]
            prev_obv = df['obv'].iloc[-2]
            
            volume_threshold = self.get_parameter('volume_threshold', 1.5)
            
            signal = {'action': 'hold', 'confidence': 0.5, 'reasoning': ''}
            
            # High volume above VWAP
            if (current_volume > avg_volume * volume_threshold and 
                current_price > vwap and obv > prev_obv):
                confidence = min(0.9, current_volume / avg_volume / volume_threshold)
                signal = {
                    'action': 'buy',
                    'confidence': confidence,
                    'stop_loss': vwap,
                    'take_profit': current_price * 1.08,
                    'reasoning': f'High volume breakout above VWAP: {current_volume:.0f} vs {avg_volume:.0f}'
                }
            
            # High volume below VWAP
            elif (current_volume > avg_volume * volume_threshold and 
                  current_price < vwap and obv < prev_obv):
                confidence = min(0.9, current_volume / avg_volume / volume_threshold)
                signal = {
                    'action': 'sell',
                    'confidence': confidence,
                    'stop_loss': vwap,
                    'take_profit': current_price * 0.92,
                    'reasoning': f'High volume breakdown below VWAP: {current_volume:.0f} vs {avg_volume:.0f}'
                }
            
            # VWAP reversion
            elif abs(current_price - vwap) / vwap > 0.02:
                if current_price > vwap:
                    signal = {
                        'action': 'sell',
                        'confidence': 0.6,
                        'stop_loss': current_price * 1.02,
                        'take_profit': vwap,
                        'reasoning': f'Price above VWAP reversion: {current_price:.2f} vs {vwap:.2f}'
                    }
                else:
                    signal = {
                        'action': 'buy',
                        'confidence': 0.6,
                        'stop_loss': current_price * 0.98,
                        'take_profit': vwap,
                        'reasoning': f'Price below VWAP reversion: {current_price:.2f} vs {vwap:.2f}'
                    }
            
            self.record_signal(signal, current_price)
            return signal
            
        except Exception as e:
            logger.error(f"Error generating volume profile signal: {e}")
            return {'action': 'hold', 'confidence': 0, 'reasoning': f'Error: {e}'}