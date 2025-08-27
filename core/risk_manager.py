import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from config import Config

logger = logging.getLogger(__name__)

class RiskManager:
    """Risk management system for trading operations"""
    
    def __init__(self, initial_capital: float = None):
        self.initial_capital = initial_capital or Config.INITIAL_CAPITAL
        self.max_risk_per_trade = Config.MAX_RISK_PER_TRADE
        self.max_portfolio_risk = Config.MAX_PORTFOLIO_RISK
        self.stop_loss_pct = Config.STOP_LOSS_PERCENTAGE
        self.take_profit_pct = Config.TAKE_PROFIT_PERCENTAGE
        
        # Risk tracking
        self.daily_pnl = []
        self.trades_history = []
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.peak_value = self.initial_capital
        
    def calculate_position_size(self, account_value: float, entry_price: float, 
                              stop_loss_price: float) -> int:
        """Calculate position size based on risk management rules"""
        try:
            # Calculate risk per share
            risk_per_share = abs(entry_price - stop_loss_price)
            
            if risk_per_share == 0:
                return 0
            
            # Calculate maximum risk amount
            max_risk_amount = account_value * self.max_risk_per_trade
            
            # Calculate position size
            position_size = int(max_risk_amount / risk_per_share)
            
            # Ensure position doesn't exceed portfolio risk limits
            max_position_value = account_value * self.max_portfolio_risk
            max_shares_by_value = int(max_position_value / entry_price)
            
            return min(position_size, max_shares_by_value)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
    
    def validate_trade(self, signal: Dict, current_portfolio: Dict, 
                      current_price: float) -> Tuple[bool, str]:
        """Validate if trade should be executed based on risk rules"""
        try:
            # Check if we have sufficient capital
            account_value = current_portfolio.get('total_amount', 0)
            if account_value < self.initial_capital * 0.5:  # 50% drawdown limit
                return False, "Account value below 50% of initial capital"
            
            # Check maximum drawdown
            if self.current_drawdown > 0.25:  # 25% max drawdown
                return False, "Maximum drawdown exceeded"
            
            # Check daily loss limit
            today_pnl = self.get_daily_pnl()
            if today_pnl < -account_value * 0.05:  # 5% daily loss limit
                return False, "Daily loss limit exceeded"
            
            # Check position concentration
            if self.check_position_concentration(signal['figi'], current_portfolio):
                return False, "Position concentration limit exceeded"
            
            # Check correlation with existing positions
            if self.check_correlation_risk(signal['figi'], current_portfolio):
                return False, "High correlation with existing positions"
            
            return True, "Trade validated"
            
        except Exception as e:
            logger.error(f"Error validating trade: {e}")
            return False, f"Validation error: {e}"
    
    def check_position_concentration(self, figi: str, portfolio: Dict) -> bool:
        """Check if adding position would exceed concentration limits"""
        try:
            positions = portfolio.get('positions', [])
            total_value = portfolio.get('total_amount', 0)
            
            # Check if single position would exceed 20% of portfolio
            current_position_value = 0
            for position in positions:
                if position['figi'] == figi:
                    current_position_value = position['quantity'] * position['current_price']
                    break
            
            max_position_value = total_value * 0.20  # 20% concentration limit
            return current_position_value > max_position_value
            
        except Exception as e:
            logger.error(f"Error checking position concentration: {e}")
            return True  # Conservative approach
    
    def check_correlation_risk(self, figi: str, portfolio: Dict) -> bool:
        """Check correlation risk with existing positions"""
        # Simplified correlation check - in real implementation,
        # you would calculate actual correlations between instruments
        try:
            positions = portfolio.get('positions', [])
            
            # For now, limit to 10 different positions
            if len(positions) >= 10:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking correlation risk: {e}")
            return True
    
    def update_portfolio_metrics(self, current_value: float):
        """Update portfolio performance metrics"""
        try:
            # Update peak value
            if current_value > self.peak_value:
                self.peak_value = current_value
                self.current_drawdown = 0.0
            else:
                self.current_drawdown = (self.peak_value - current_value) / self.peak_value
                
            # Update maximum drawdown
            if self.current_drawdown > self.max_drawdown:
                self.max_drawdown = self.current_drawdown
                
            # Add to daily PnL tracking
            pnl = current_value - self.initial_capital
            self.daily_pnl.append({
                'date': datetime.now().date(),
                'value': current_value,
                'pnl': pnl,
                'drawdown': self.current_drawdown
            })
            
            # Keep only last 30 days
            if len(self.daily_pnl) > 30:
                self.daily_pnl = self.daily_pnl[-30:]
                
        except Exception as e:
            logger.error(f"Error updating portfolio metrics: {e}")
    
    def get_daily_pnl(self) -> float:
        """Get today's PnL"""
        try:
            today = datetime.now().date()
            today_records = [record for record in self.daily_pnl 
                           if record['date'] == today]
            
            if today_records:
                return today_records[-1]['pnl']
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting daily PnL: {e}")
            return 0.0
    
    def calculate_stop_loss(self, entry_price: float, direction: str) -> float:
        """Calculate stop loss price"""
        if direction.lower() == 'buy':
            return entry_price * (1 - self.stop_loss_pct)
        else:
            return entry_price * (1 + self.stop_loss_pct)
    
    def calculate_take_profit(self, entry_price: float, direction: str) -> float:
        """Calculate take profit price"""
        if direction.lower() == 'buy':
            return entry_price * (1 + self.take_profit_pct)
        else:
            return entry_price * (1 - self.take_profit_pct)
    
    def record_trade(self, trade_info: Dict):
        """Record completed trade"""
        try:
            trade_record = {
                'timestamp': datetime.now(),
                'figi': trade_info.get('figi'),
                'direction': trade_info.get('direction'),
                'quantity': trade_info.get('quantity'),
                'entry_price': trade_info.get('entry_price'),
                'exit_price': trade_info.get('exit_price'),
                'pnl': trade_info.get('pnl'),
                'strategy': trade_info.get('strategy')
            }
            
            self.trades_history.append(trade_record)
            
            # Keep only last 1000 trades
            if len(self.trades_history) > 1000:
                self.trades_history = self.trades_history[-1000:]
                
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
    
    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics"""
        try:
            if not self.daily_pnl:
                return {}
            
            df = pd.DataFrame(self.daily_pnl)
            
            # Calculate returns
            df['returns'] = df['value'].pct_change()
            
            # Performance metrics
            total_return = (df['value'].iloc[-1] - self.initial_capital) / self.initial_capital
            
            # Annualized return (assuming 252 trading days)
            days = len(df)
            annualized_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
            
            # Sharpe ratio (simplified)
            mean_return = df['returns'].mean()
            std_return = df['returns'].std()
            sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
            
            # Win rate from trades
            if self.trades_history:
                winning_trades = sum(1 for trade in self.trades_history if trade.get('pnl', 0) > 0)
                win_rate = winning_trades / len(self.trades_history)
            else:
                win_rate = 0
            
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'max_drawdown': self.max_drawdown,
                'current_drawdown': self.current_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'win_rate': win_rate,
                'total_trades': len(self.trades_history),
                'current_value': df['value'].iloc[-1] if not df.empty else self.initial_capital
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def should_stop_trading(self) -> Tuple[bool, str]:
        """Check if trading should be stopped due to risk limits"""
        try:
            # Check maximum drawdown
            if self.current_drawdown > 0.25:
                return True, "Maximum drawdown exceeded (25%)"
            
            # Check daily loss limit
            daily_pnl = self.get_daily_pnl()
            if self.daily_pnl:
                current_value = self.daily_pnl[-1]['value']
                if daily_pnl < -current_value * 0.05:
                    return True, "Daily loss limit exceeded (5%)"
            
            # Check consecutive losing days
            if len(self.daily_pnl) >= 5:
                last_5_days = self.daily_pnl[-5:]
                if all(day['pnl'] < 0 for day in last_5_days):
                    return True, "5 consecutive losing days"
            
            return False, "Continue trading"
            
        except Exception as e:
            logger.error(f"Error checking stop trading conditions: {e}")
            return True, f"Error in risk check: {e}"