import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Configuration
    TBANK_TOKEN = os.getenv('TBANK_TOKEN')
    TBANK_SANDBOX_TOKEN = os.getenv('TBANK_SANDBOX_TOKEN')
    TBANK_APP_NAME = os.getenv('TBANK_APP_NAME', 'AutoTrader')
    
    # Trading Configuration
    MAX_RISK_PER_TRADE = float(os.getenv('MAX_RISK_PER_TRADE', 0.02))
    INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', 100000))
    USE_SANDBOX = os.getenv('USE_SANDBOX', 'True').lower() == 'true'
    
    # Risk Management
    MAX_PORTFOLIO_RISK = 0.20  # 20% of portfolio
    STOP_LOSS_PERCENTAGE = 0.05  # 5% stop loss
    TAKE_PROFIT_PERCENTAGE = 0.15  # 15% take profit
    
    # Monitoring
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
    
    # Database
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///trading_data.db')
    
    # Backtesting
    BACKTEST_START_DATE = '2023-01-01'
    BACKTEST_END_DATE = '2023-12-31'