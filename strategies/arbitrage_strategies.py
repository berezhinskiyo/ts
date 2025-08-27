import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class PairsTradingStrategy(BaseStrategy):
    """Statistical arbitrage pairs trading strategy"""
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'lookback_period': 60,
            'entry_threshold': 2.0,
            'exit_threshold': 0.5,
            'stop_loss_threshold': 3.0,
            'correlation_threshold': 0.7,
            'pairs': []  # List of (ticker1, ticker2) tuples
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("PairsTrading", default_params)
        self.price_data = {}  # Store price data for multiple instruments
        
    def get_minimum_data_length(self) -> int:
        return self.get_parameter('lookback_period', 60) + 10
    
    def add_price_data(self, ticker: str, data: pd.DataFrame):
        """Add price data for a specific ticker"""
        self.price_data[ticker] = data
    
    def calculate_spread(self, ticker1: str, ticker2: str) -> pd.Series:
        """Calculate spread between two instruments"""
        try:
            if ticker1 not in self.price_data or ticker2 not in self.price_data:
                return pd.Series()
            
            data1 = self.price_data[ticker1]['close']
            data2 = self.price_data[ticker2]['close']
            
            # Align data by common dates
            common_dates = data1.index.intersection(data2.index)
            if len(common_dates) < self.get_minimum_data_length():
                return pd.Series()
            
            aligned_data1 = data1.loc[common_dates]
            aligned_data2 = data2.loc[common_dates]
            
            # Calculate hedge ratio using linear regression
            X = aligned_data2.values.reshape(-1, 1)
            y = aligned_data1.values
            
            # Simple linear regression
            X_mean = np.mean(X)
            y_mean = np.mean(y)
            numerator = np.sum((X.flatten() - X_mean) * (y - y_mean))
            denominator = np.sum((X.flatten() - X_mean) ** 2)
            
            if denominator == 0:
                return pd.Series()
            
            hedge_ratio = numerator / denominator
            
            # Calculate spread
            spread = aligned_data1 - hedge_ratio * aligned_data2
            
            return spread
            
        except Exception as e:
            logger.error(f"Error calculating spread for {ticker1}/{ticker2}: {e}")
            return pd.Series()
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators for pairs trading"""
        # This method is not used directly for pairs trading
        return data
    
    def generate_signal(self, data: pd.DataFrame, current_price: float, 
                       portfolio: Dict) -> Dict:
        """Generate pairs trading signal"""
        try:
            pairs = self.get_parameter('pairs', [])
            if not pairs:
                return {'action': 'hold', 'confidence': 0, 'reasoning': 'No pairs configured'}
            
            lookback = self.get_parameter('lookback_period', 60)
            entry_threshold = self.get_parameter('entry_threshold', 2.0)
            exit_threshold = self.get_parameter('exit_threshold', 0.5)
            
            best_signal = {'action': 'hold', 'confidence': 0, 'reasoning': 'No significant spreads'}
            
            for ticker1, ticker2 in pairs:
                spread = self.calculate_spread(ticker1, ticker2)
                
                if len(spread) < lookback:
                    continue
                
                # Calculate spread statistics
                recent_spread = spread.iloc[-lookback:]
                mean_spread = recent_spread.mean()
                std_spread = recent_spread.std()
                
                if std_spread == 0:
                    continue
                
                current_spread = spread.iloc[-1]
                z_score = (current_spread - mean_spread) / std_spread
                
                # Generate signals based on z-score
                if abs(z_score) > entry_threshold:
                    confidence = min(0.9, abs(z_score) / 4)
                    
                    if z_score > entry_threshold:
                        # Spread is too high - sell ticker1, buy ticker2
                        signal = {
                            'action': 'sell',
                            'confidence': confidence,
                            'pair': (ticker1, ticker2),
                            'z_score': z_score,
                            'reasoning': f'Pairs trade: Sell {ticker1}, Buy {ticker2} (Z-score: {z_score:.2f})'
                        }
                    else:
                        # Spread is too low - buy ticker1, sell ticker2
                        signal = {
                            'action': 'buy',
                            'confidence': confidence,
                            'pair': (ticker1, ticker2),
                            'z_score': z_score,
                            'reasoning': f'Pairs trade: Buy {ticker1}, Sell {ticker2} (Z-score: {z_score:.2f})'
                        }
                    
                    if signal['confidence'] > best_signal['confidence']:
                        best_signal = signal
                
                # Exit signals
                elif abs(z_score) < exit_threshold:
                    signal = {
                        'action': 'close',
                        'confidence': 0.8,
                        'pair': (ticker1, ticker2),
                        'z_score': z_score,
                        'reasoning': f'Pairs trade exit: {ticker1}/{ticker2} (Z-score: {z_score:.2f})'
                    }
                    
                    if signal['confidence'] > best_signal['confidence']:
                        best_signal = signal
            
            self.record_signal(best_signal, current_price)
            return best_signal
            
        except Exception as e:
            logger.error(f"Error generating pairs trading signal: {e}")
            return {'action': 'hold', 'confidence': 0, 'reasoning': f'Error: {e}'}


class CalendarSpreadStrategy(BaseStrategy):
    """Calendar spread arbitrage strategy"""
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'near_month_expiry': 30,  # Days to near month expiry
            'far_month_expiry': 90,   # Days to far month expiry
            'spread_threshold': 0.05,
            'volatility_threshold': 0.3
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("CalendarSpread", default_params)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate calendar spread indicators"""
        df = data.copy()
        
        # Calculate implied volatility (simplified)
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        
        return df
    
    def generate_signal(self, data: pd.DataFrame, current_price: float, 
                       portfolio: Dict) -> Dict:
        """Generate calendar spread signal"""
        try:
            if not self.validate_data(data):
                return {'action': 'hold', 'confidence': 0, 'reasoning': 'Invalid data'}
            
            df = self.calculate_indicators(data)
            current_volatility = df['volatility'].iloc[-1]
            avg_volatility = df['volatility'].rolling(window=60).mean().iloc[-1]
            
            spread_threshold = self.get_parameter('spread_threshold', 0.05)
            volatility_threshold = self.get_parameter('volatility_threshold', 0.3)
            
            signal = {'action': 'hold', 'confidence': 0.5, 'reasoning': ''}
            
            # Low volatility environment - sell calendar spreads
            if current_volatility < volatility_threshold and current_volatility < avg_volatility * 0.8:
                signal = {
                    'action': 'sell_calendar',
                    'confidence': 0.7,
                    'reasoning': f'Low volatility calendar spread: {current_volatility:.3f} vs {avg_volatility:.3f}'
                }
            
            # High volatility environment - buy calendar spreads
            elif current_volatility > volatility_threshold and current_volatility > avg_volatility * 1.2:
                signal = {
                    'action': 'buy_calendar',
                    'confidence': 0.7,
                    'reasoning': f'High volatility calendar spread: {current_volatility:.3f} vs {avg_volatility:.3f}'
                }
            
            self.record_signal(signal, current_price)
            return signal
            
        except Exception as e:
            logger.error(f"Error generating calendar spread signal: {e}")
            return {'action': 'hold', 'confidence': 0, 'reasoning': f'Error: {e}'}


class VolatilityArbitrageStrategy(BaseStrategy):
    """Volatility arbitrage strategy"""
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'volatility_window': 20,
            'vol_threshold_high': 0.4,
            'vol_threshold_low': 0.1,
            'vol_percentile_high': 80,
            'vol_percentile_low': 20
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("VolatilityArbitrage", default_params)
    
    def get_minimum_data_length(self) -> int:
        return self.get_parameter('volatility_window', 20) * 3
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility indicators"""
        df = data.copy()
        vol_window = self.get_parameter('volatility_window', 20)
        
        # Historical volatility
        df['returns'] = df['close'].pct_change()
        df['hist_vol'] = df['returns'].rolling(window=vol_window).std() * np.sqrt(252)
        
        # Parkinson volatility (high-low estimator)
        df['parkinson_vol'] = np.sqrt(
            (1 / (4 * np.log(2))) * 
            np.log(df['high'] / df['low']) ** 2
        ).rolling(window=vol_window).mean() * np.sqrt(252)
        
        # Volume-adjusted volatility
        df['volume_adj_vol'] = df['hist_vol'] * np.sqrt(df['volume'] / df['volume'].rolling(window=vol_window).mean())
        
        # Volatility percentiles
        df['vol_percentile'] = df['hist_vol'].rolling(window=vol_window*3).rank(pct=True) * 100
        
        return df
    
    def generate_signal(self, data: pd.DataFrame, current_price: float, 
                       portfolio: Dict) -> Dict:
        """Generate volatility arbitrage signal"""
        try:
            if not self.validate_data(data):
                return {'action': 'hold', 'confidence': 0, 'reasoning': 'Invalid data'}
            
            df = self.calculate_indicators(data)
            
            current_vol = df['hist_vol'].iloc[-1]
            vol_percentile = df['vol_percentile'].iloc[-1]
            parkinson_vol = df['parkinson_vol'].iloc[-1]
            
            vol_threshold_high = self.get_parameter('vol_threshold_high', 0.4)
            vol_threshold_low = self.get_parameter('vol_threshold_low', 0.1)
            vol_percentile_high = self.get_parameter('vol_percentile_high', 80)
            vol_percentile_low = self.get_parameter('vol_percentile_low', 20)
            
            signal = {'action': 'hold', 'confidence': 0.5, 'reasoning': ''}
            
            # High volatility - sell volatility
            if (current_vol > vol_threshold_high or vol_percentile > vol_percentile_high):
                confidence = min(0.9, current_vol / vol_threshold_high * 0.7)
                signal = {
                    'action': 'sell_volatility',
                    'confidence': confidence,
                    'stop_loss': current_price * 1.10,
                    'take_profit': current_price * 0.95,
                    'reasoning': f'High volatility: {current_vol:.3f} ({vol_percentile:.1f}th percentile)'
                }
            
            # Low volatility - buy volatility
            elif (current_vol < vol_threshold_low or vol_percentile < vol_percentile_low):
                confidence = min(0.9, (vol_threshold_low - current_vol) / vol_threshold_low * 3)
                signal = {
                    'action': 'buy_volatility',
                    'confidence': confidence,
                    'stop_loss': current_price * 0.90,
                    'take_profit': current_price * 1.05,
                    'reasoning': f'Low volatility: {current_vol:.3f} ({vol_percentile:.1f}th percentile)'
                }
            
            # Volatility discrepancy between estimators
            elif abs(current_vol - parkinson_vol) / current_vol > 0.2:
                if current_vol > parkinson_vol:
                    signal = {
                        'action': 'sell_volatility',
                        'confidence': 0.6,
                        'reasoning': f'Vol discrepancy: Hist {current_vol:.3f} > Parkinson {parkinson_vol:.3f}'
                    }
                else:
                    signal = {
                        'action': 'buy_volatility',
                        'confidence': 0.6,
                        'reasoning': f'Vol discrepancy: Hist {current_vol:.3f} < Parkinson {parkinson_vol:.3f}'
                    }
            
            self.record_signal(signal, current_price)
            return signal
            
        except Exception as e:
            logger.error(f"Error generating volatility arbitrage signal: {e}")
            return {'action': 'hold', 'confidence': 0, 'reasoning': f'Error: {e}'}


class ETFArbitrageStrategy(BaseStrategy):
    """ETF-underlying arbitrage strategy"""
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'premium_threshold': 0.005,  # 0.5% premium/discount threshold
            'volume_threshold': 1000000,  # Minimum volume
            'liquidity_ratio': 0.1
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("ETFArbitrage", default_params)
        self.nav_data = None  # Net Asset Value data
    
    def set_nav_data(self, nav_data: pd.DataFrame):
        """Set Net Asset Value data for ETF"""
        self.nav_data = nav_data
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate ETF arbitrage indicators"""
        df = data.copy()
        
        if self.nav_data is not None:
            # Align ETF price with NAV
            common_dates = df.index.intersection(self.nav_data.index)
            if len(common_dates) > 0:
                df = df.loc[common_dates]
                nav_aligned = self.nav_data.loc[common_dates]
                
                # Calculate premium/discount
                df['nav'] = nav_aligned['nav']
                df['premium'] = (df['close'] - df['nav']) / df['nav']
                
                # Rolling statistics
                df['premium_mean'] = df['premium'].rolling(window=20).mean()
                df['premium_std'] = df['premium'].rolling(window=20).std()
                df['z_score'] = (df['premium'] - df['premium_mean']) / df['premium_std']
        
        return df
    
    def generate_signal(self, data: pd.DataFrame, current_price: float, 
                       portfolio: Dict) -> Dict:
        """Generate ETF arbitrage signal"""
        try:
            if not self.validate_data(data) or self.nav_data is None:
                return {'action': 'hold', 'confidence': 0, 'reasoning': 'Invalid data or missing NAV'}
            
            df = self.calculate_indicators(data)
            
            if 'premium' not in df.columns:
                return {'action': 'hold', 'confidence': 0, 'reasoning': 'Cannot calculate premium'}
            
            current_premium = df['premium'].iloc[-1]
            current_volume = df['volume'].iloc[-1]
            z_score = df['z_score'].iloc[-1] if 'z_score' in df.columns else 0
            
            premium_threshold = self.get_parameter('premium_threshold', 0.005)
            volume_threshold = self.get_parameter('volume_threshold', 1000000)
            
            signal = {'action': 'hold', 'confidence': 0.5, 'reasoning': ''}
            
            # Check sufficient volume
            if current_volume < volume_threshold:
                return {'action': 'hold', 'confidence': 0, 'reasoning': 'Insufficient volume'}
            
            # ETF trading at premium - sell ETF, buy underlying
            if current_premium > premium_threshold:
                confidence = min(0.9, current_premium / premium_threshold * 0.7)
                signal = {
                    'action': 'sell_etf_buy_underlying',
                    'confidence': confidence,
                    'premium': current_premium,
                    'reasoning': f'ETF premium: {current_premium:.4f} ({current_premium*100:.2f}%)'
                }
            
            # ETF trading at discount - buy ETF, sell underlying
            elif current_premium < -premium_threshold:
                confidence = min(0.9, abs(current_premium) / premium_threshold * 0.7)
                signal = {
                    'action': 'buy_etf_sell_underlying',
                    'confidence': confidence,
                    'premium': current_premium,
                    'reasoning': f'ETF discount: {current_premium:.4f} ({current_premium*100:.2f}%)'
                }
            
            # Mean reversion based on z-score
            elif abs(z_score) > 2:
                if z_score > 2:
                    signal = {
                        'action': 'sell_etf_buy_underlying',
                        'confidence': 0.6,
                        'reasoning': f'Premium mean reversion: Z-score {z_score:.2f}'
                    }
                else:
                    signal = {
                        'action': 'buy_etf_sell_underlying',
                        'confidence': 0.6,
                        'reasoning': f'Discount mean reversion: Z-score {z_score:.2f}'
                    }
            
            self.record_signal(signal, current_price)
            return signal
            
        except Exception as e:
            logger.error(f"Error generating ETF arbitrage signal: {e}")
            return {'action': 'hold', 'confidence': 0, 'reasoning': f'Error: {e}'}