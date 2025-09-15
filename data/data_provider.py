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
    
    # REMOVED: generate_synthetic_data - using only real data from T-Bank API
    
    async def get_multiple_symbols_data(self, symbols: List[str], start_date: str, 
                                      end_date: str) -> Dict[str, pd.DataFrame]:
        """Get historical data for multiple symbols - ONLY REAL DATA"""
        data = {}
        
        # FIGI mapping for Russian stocks
        figi_mapping = {
            'SBER': 'BBG004730N88',   # Сбербанк
            'GAZP': 'BBG004730ZJ9',   # Газпром
            'LKOH': 'BBG004730JJ5',   # ЛУКОЙЛ
            'YNDX': 'BBG006L8G4H1',   # Яндекс
            'ROSN': 'BBG0047315Y7',   # Роснефть
            'NVTK': 'BBG00475KKY8',   # НОВАТЭК
            'TATN': 'BBG004RVFFC0',   # Татнефть
            'MTSS': 'BBG004RVFFC0',   # МТС
            'MGNT': 'BBG004S681W1',   # Магнит
            'RTKM': 'BBG004S681W1',   # Ростелеком
            'OZON': 'BBG00Y91R9T3',   # OZON
            'FIVE': 'BBG00Y91R9T3',   # X5 Retail Group
            'TCSG': 'BBG00Y91R9T3',   # TCS Group
            'MAIL': 'BBG00Y91R9T3',   # VK
            'PHOR': 'BBG00Y91R9T3',   # ФосАгро
            'NLMK': 'BBG00Y91R9T3',   # НЛМК
            'CHMF': 'BBG00Y91R9T3',   # Северсталь
            'SNGS': 'BBG00Y91R9T3',   # Сургутнефтегаз
            'VTBR': 'BBG00Y91R9T3',   # ВТБ
            'AFLT': 'BBG00Y91R9T3'    # Аэрофлот
        }
        
        for symbol in symbols:
            try:
                df = None
                
                # Try T-Bank API first (preferred)
                if self.tbank_client and symbol in figi_mapping:
                    figi = figi_mapping[symbol]
                    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                    
                    df = await self.get_historical_data_tbank(figi, start_dt, end_dt, 'day')
                    logger.info(f"Loaded T-Bank data for {symbol}: {len(df)} rows")
                
                # Fallback to Yahoo Finance if T-Bank fails
                if df is None or df.empty:
                    df = self.get_historical_data_yahoo(symbol, start_date, end_date)
                    if not df.empty:
                        logger.info(f"Loaded Yahoo Finance data for {symbol}: {len(df)} rows")
                
                if not df.empty:
                    # Add some basic validation
                    df = self._validate_and_clean_data(df, symbol)
                    if not df.empty:
                        data[symbol] = df
                        logger.info(f"Successfully loaded data for {symbol}: {len(df)} rows")
                    else:
                        logger.warning(f"Data validation failed for {symbol}")
                else:
                    logger.warning(f"No real data available for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error loading real data for {symbol}: {e}")
        
        logger.info(f"Loaded real data for {len(data)} symbols out of {len(symbols)} requested")
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