from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, name: str, parameters: Dict = None):
        self.name = name
        self.parameters = parameters or {}
        self.signals_history = []
        self.performance_metrics = {}
        
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame, current_price: float, 
                       portfolio: Dict) -> Dict:
        """
        Generate trading signal based on data
        
        Returns:
            Dict with keys: action ('buy'/'sell'/'hold'), confidence (0-1), 
                          stop_loss, take_profit, reasoning
        """
        pass
    
    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the strategy"""
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate if data is sufficient for strategy"""
        try:
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                logger.error(f"Missing required columns in data for {self.name}")
                return False
            
            if len(data) < self.get_minimum_data_length():
                logger.error(f"Insufficient data length for {self.name}")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Error validating data for {self.name}: {e}")
            return False
    
    def get_minimum_data_length(self) -> int:
        """Get minimum required data length for strategy"""
        return 50  # Default minimum
    
    def record_signal(self, signal: Dict, price: float):
        """Record generated signal for analysis"""
        signal_record = {
            'timestamp': datetime.now(),
            'signal': signal,
            'price': price,
            'strategy': self.name
        }
        self.signals_history.append(signal_record)
        
        # Keep only last 1000 signals
        if len(self.signals_history) > 1000:
            self.signals_history = self.signals_history[-1000:]
    
    def get_parameter(self, key: str, default=None):
        """Get strategy parameter with default value"""
        return self.parameters.get(key, default)
    
    def update_parameters(self, new_parameters: Dict):
        """Update strategy parameters"""
        self.parameters.update(new_parameters)
    
    def calculate_position_score(self, signal: Dict, market_conditions: Dict) -> float:
        """Calculate position scoring based on signal and market conditions"""
        base_score = signal.get('confidence', 0.5)
        
        # Adjust for market volatility
        volatility = market_conditions.get('volatility', 0.02)
        if volatility > 0.05:  # High volatility
            base_score *= 0.8
        elif volatility < 0.01:  # Low volatility
            base_score *= 1.2
            
        # Adjust for market trend
        trend = market_conditions.get('trend', 'neutral')
        if trend == 'bullish' and signal.get('action') == 'buy':
            base_score *= 1.1
        elif trend == 'bearish' and signal.get('action') == 'sell':
            base_score *= 1.1
        elif (trend == 'bullish' and signal.get('action') == 'sell') or \
             (trend == 'bearish' and signal.get('action') == 'buy'):
            base_score *= 0.9
            
        return min(max(base_score, 0.0), 1.0)
    
    def __str__(self):
        return f"{self.name} Strategy"
    
    def __repr__(self):
        return f"<{self.name}Strategy(parameters={self.parameters})>"


class TechnicalStrategy(BaseStrategy):
    """Base class for technical analysis strategies"""
    
    def __init__(self, name: str, parameters: Dict = None):
        super().__init__(name, parameters)
        
    def calculate_sma(self, data: pd.Series, window: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return data.rolling(window=window).mean()
    
    def calculate_ema(self, data: pd.Series, window: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return data.ewm(span=window).mean()
    
    def calculate_rsi(self, data: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, data: pd.Series, fast: int = 12, slow: int = 26, 
                      signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = self.calculate_ema(data, fast)
        ema_slow = self.calculate_ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def calculate_bollinger_bands(self, data: pd.Series, window: int = 20, 
                                 num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = self.calculate_sma(data, window)
        std = data.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, sma, lower_band
    
    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                           k_window: int = 14, d_window: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        return k_percent, d_percent
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                     window: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=window).mean()
        return atr
    
    def detect_support_resistance(self, data: pd.DataFrame, window: int = 20) -> Dict:
        """Detect support and resistance levels"""
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Find local maxima and minima
            local_maxima = high.rolling(window=window, center=True).max() == high
            local_minima = low.rolling(window=window, center=True).min() == low
            
            resistance_levels = high[local_maxima].tolist()
            support_levels = low[local_minima].tolist()
            
            # Get recent levels
            current_price = close.iloc[-1]
            nearby_resistance = [r for r in resistance_levels if r > current_price and r < current_price * 1.1]
            nearby_support = [s for s in support_levels if s < current_price and s > current_price * 0.9]
            
            return {
                'resistance': nearby_resistance,
                'support': nearby_support,
                'all_resistance': resistance_levels,
                'all_support': support_levels
            }
        except Exception as e:
            logger.error(f"Error detecting support/resistance: {e}")
            return {'resistance': [], 'support': [], 'all_resistance': [], 'all_support': []}
    
    def calculate_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based indicators"""
        try:
            df = data.copy()
            
            # Volume Moving Average
            df['volume_sma'] = self.calculate_sma(df['volume'], 20)
            
            # On-Balance Volume
            df['obv'] = (df['volume'] * np.where(df['close'] > df['close'].shift(), 1, -1)).cumsum()
            
            # Volume Price Trend
            df['vpt'] = (df['volume'] * df['close'].pct_change()).cumsum()
            
            return df
        except Exception as e:
            logger.error(f"Error calculating volume indicators: {e}")
            return data