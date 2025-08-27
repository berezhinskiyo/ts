import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from core.api_client import TBankAPIClient
from tinkoff.invest.schemas import CandleInterval
import asyncio

logger = logging.getLogger(__name__)

class DataProvider:
    """Data provider for backtesting and live trading"""
    
    def __init__(self, use_tbank: bool = True):
        self.use_tbank = use_tbank
        self.tbank_client = None
        self.cache = {}
        
    async def __aenter__(self):
        if self.use_tbank:
            self.tbank_client = TBankAPIClient()
            await self.tbank_client.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.tbank_client:
            await self.tbank_client.__aexit__(exc_type, exc_val, exc_tb)
    
    def get_russian_stocks(self) -> List[str]:
        """Get list of popular Russian stocks"""
        return [
            'SBER',   # Сбербанк
            'GAZP',   # Газпром
            'LKOH',   # ЛУКОЙЛ
            'YNDX',   # Яндекс
            'ROSN',   # Роснефть
            'NVTK',   # НОВАТЭК
            'TATN',   # Татнефть
            'MTSS',   # МТС
            'MGNT',   # Магнит
            'RTKM',   # Ростелеком
            'OZON',   # OZON
            'FIVE',   # X5 Retail Group
            'TCSG',   # TCS Group
            'MAIL',   # VK
            'PHOR',   # ФосАгро
            'NLMK',   # НЛМК
            'CHMF',   # Северсталь
            'SNGS',   # Сургутнефтегаз
            'VTBR',   # ВТБ
            'AFLT'    # Аэрофлот
        ]
    
    async def get_historical_data_tbank(self, figi: str, start_date: datetime, 
                                      end_date: datetime, interval: str = 'day') -> pd.DataFrame:
        """Get historical data from T-Bank API"""
        try:
            if not self.tbank_client:
                raise ValueError("T-Bank client not initialized")
            
            # Map interval
            interval_map = {
                'day': CandleInterval.CANDLE_INTERVAL_DAY,
                'hour': CandleInterval.CANDLE_INTERVAL_HOUR,
                'minute': CandleInterval.CANDLE_INTERVAL_1_MIN
            }
            
            candle_interval = interval_map.get(interval, CandleInterval.CANDLE_INTERVAL_DAY)
            
            candles = await self.tbank_client.get_historical_candles(
                figi, start_date, end_date, candle_interval
            )
            
            if not candles:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(candles)
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting T-Bank data for {figi}: {e}")
            return pd.DataFrame()
    
    def get_historical_data_yahoo(self, symbol: str, start_date: str, 
                                 end_date: str, interval: str = '1d') -> pd.DataFrame:
        """Get historical data from Yahoo Finance"""
        try:
            # For Russian stocks, add .ME suffix
            if symbol in self.get_russian_stocks():
                yahoo_symbol = f"{symbol}.ME"
            else:
                yahoo_symbol = symbol
            
            ticker = yf.Ticker(yahoo_symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Standardize column names
            df.columns = [col.lower() for col in df.columns]
            
            # Ensure we have required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.error(f"Missing columns for {symbol}: {missing_cols}")
                return pd.DataFrame()
            
            return df[required_cols]
            
        except Exception as e:
            logger.error(f"Error getting Yahoo data for {symbol}: {e}")
            return pd.DataFrame()
    
    def generate_synthetic_data(self, symbol: str, start_date: str, end_date: str,
                              initial_price: float = 100, volatility: float = 0.02) -> pd.DataFrame:
        """Generate synthetic price data for testing"""
        try:
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            n_days = len(dates)
            
            # Generate random returns
            np.random.seed(42)  # For reproducibility
            returns = np.random.normal(0.0005, volatility, n_days)  # Small positive drift
            
            # Calculate prices
            prices = [initial_price]
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            # Generate OHLC data
            data = []
            for i, (date, price) in enumerate(zip(dates, prices)):
                # Generate intraday volatility
                daily_vol = volatility * np.random.uniform(0.5, 1.5)
                
                # Open price (previous close with gap)
                if i == 0:
                    open_price = price
                else:
                    gap = np.random.normal(0, volatility * 0.5)
                    open_price = prices[i-1] * (1 + gap)
                
                # High and low
                high_mult = 1 + abs(np.random.normal(0, daily_vol))
                low_mult = 1 - abs(np.random.normal(0, daily_vol))
                
                high = max(open_price, price) * high_mult
                low = min(open_price, price) * low_mult
                
                # Volume (random with some correlation to price movement)
                base_volume = 1000000
                vol_multiplier = 1 + abs(returns[i]) * 10  # Higher volume on big moves
                volume = int(base_volume * vol_multiplier * np.random.uniform(0.5, 2.0))
                
                data.append({
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': price,
                    'volume': volume
                })
            
            df = pd.DataFrame(data, index=dates)
            return df
            
        except Exception as e:
            logger.error(f"Error generating synthetic data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def get_multiple_symbols_data(self, symbols: List[str], start_date: str, 
                                      end_date: str, use_synthetic: bool = False) -> Dict[str, pd.DataFrame]:
        """Get historical data for multiple symbols"""
        data = {}
        
        for symbol in symbols:
            try:
                if use_synthetic:
                    # Generate synthetic data
                    initial_prices = {
                        'SBER': 250, 'GAZP': 150, 'LKOH': 5500, 'YNDX': 2500,
                        'ROSN': 450, 'NVTK': 1200, 'TATN': 650, 'MTSS': 300,
                        'MGNT': 5000, 'RTKM': 60, 'OZON': 1500, 'FIVE': 2000
                    }
                    initial_price = initial_prices.get(symbol, 100)
                    
                    df = self.generate_synthetic_data(symbol, start_date, end_date, initial_price)
                else:
                    # Try Yahoo Finance first
                    df = self.get_historical_data_yahoo(symbol, start_date, end_date)
                    
                    # If Yahoo fails and we have T-Bank client, try T-Bank
                    if df.empty and self.tbank_client:
                        # Note: would need FIGI mapping for real implementation
                        logger.info(f"Yahoo Finance failed for {symbol}, skipping T-Bank for now")
                
                if not df.empty:
                    # Add some basic validation
                    df = self._validate_and_clean_data(df, symbol)
                    if not df.empty:
                        data[symbol] = df
                        logger.info(f"Successfully loaded data for {symbol}: {len(df)} rows")
                    else:
                        logger.warning(f"Data validation failed for {symbol}")
                else:
                    logger.warning(f"No data available for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {e}")
        
        logger.info(f"Loaded data for {len(data)} symbols out of {len(symbols)} requested")
        return data
    
    def _validate_and_clean_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Validate and clean price data"""
        try:
            if df.empty:
                return df
            
            # Remove rows with NaN values
            df = df.dropna()
            
            # Check for negative prices
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in df.columns:
                    df = df[df[col] > 0]
            
            # Check for logical price relationships
            if all(col in df.columns for col in price_cols):
                # High should be >= max(open, close)
                valid_high = (df['high'] >= df[['open', 'close']].max(axis=1)) | \
                           (abs(df['high'] - df[['open', 'close']].max(axis=1)) < 0.01)
                
                # Low should be <= min(open, close)
                valid_low = (df['low'] <= df[['open', 'close']].min(axis=1)) | \
                          (abs(df['low'] - df[['open', 'close']].min(axis=1)) < 0.01)
                
                df = df[valid_high & valid_low]
            
            # Remove extreme outliers (price changes > 50%)
            if 'close' in df.columns and len(df) > 1:
                returns = df['close'].pct_change()
                df = df[abs(returns) < 0.5]
            
            # Ensure minimum number of data points
            if len(df) < 10:
                logger.warning(f"Insufficient data points for {symbol}: {len(df)}")
                return pd.DataFrame()
            
            return df
            
        except Exception as e:
            logger.error(f"Error validating data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_market_data_summary(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """Get summary statistics for market data"""
        summary = {}
        
        for symbol, df in data.items():
            if df.empty:
                continue
            
            try:
                returns = df['close'].pct_change().dropna()
                
                summary[symbol] = {
                    'start_date': df.index[0].strftime('%Y-%m-%d'),
                    'end_date': df.index[-1].strftime('%Y-%m-%d'),
                    'num_days': len(df),
                    'start_price': df['close'].iloc[0],
                    'end_price': df['close'].iloc[-1],
                    'total_return': (df['close'].iloc[-1] / df['close'].iloc[0] - 1),
                    'volatility': returns.std() * np.sqrt(252),
                    'avg_volume': df['volume'].mean(),
                    'max_price': df['high'].max(),
                    'min_price': df['low'].min()
                }
            except Exception as e:
                logger.error(f"Error calculating summary for {symbol}: {e}")
                summary[symbol] = {'error': str(e)}
        
        return summary