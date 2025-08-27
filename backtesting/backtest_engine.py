import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

from core.risk_manager import RiskManager
from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """Trade record for backtesting"""
    entry_time: datetime
    exit_time: Optional[datetime]
    symbol: str
    direction: str  # 'buy' or 'sell'
    entry_price: float
    exit_price: Optional[float]
    quantity: int
    stop_loss: Optional[float]
    take_profit: Optional[float]
    pnl: Optional[float]
    fees: float
    strategy: str
    is_open: bool = True

@dataclass
class BacktestResult:
    """Backtest results container"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    consecutive_wins: int
    consecutive_losses: int
    monthly_returns: pd.Series
    daily_returns: pd.Series
    equity_curve: pd.Series
    trades: List[Trade]
    performance_metrics: Dict[str, float]

class BacktestEngine:
    """Backtesting engine for trading strategies"""
    
    def __init__(self, initial_capital: float = 100000, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission  # Commission rate (0.1% default)
        self.risk_manager = RiskManager(initial_capital)
        
        # Backtesting state
        self.current_capital = initial_capital
        self.positions = {}  # symbol -> Trade
        self.closed_trades = []
        self.equity_history = []
        self.daily_pnl = []
        
    def run_backtest(self, strategy: BaseStrategy, data: Dict[str, pd.DataFrame], 
                    start_date: str, end_date: str) -> BacktestResult:
        """Run backtest for a strategy"""
        try:
            logger.info(f"Starting backtest for {strategy.name} from {start_date} to {end_date}")
            
            # Reset state
            self._reset_state()
            
            # Filter data by date range
            filtered_data = self._filter_data_by_date(data, start_date, end_date)
            
            if not filtered_data:
                raise ValueError("No data available for specified date range")
            
            # Get all trading dates
            all_dates = set()
            for symbol_data in filtered_data.values():
                all_dates.update(symbol_data.index)
            trading_dates = sorted(all_dates)
            
            # Run simulation day by day
            for current_date in trading_dates:
                self._process_trading_day(strategy, filtered_data, current_date)
            
            # Close all open positions at the end
            self._close_all_positions(trading_dates[-1], filtered_data)
            
            # Calculate results
            result = self._calculate_results(strategy, trading_dates[0], trading_dates[-1])
            
            logger.info(f"Backtest completed for {strategy.name}")
            logger.info(f"Total return: {result.total_return:.2%}")
            logger.info(f"Sharpe ratio: {result.sharpe_ratio:.3f}")
            logger.info(f"Max drawdown: {result.max_drawdown:.2%}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            raise
    
    def _reset_state(self):
        """Reset backtesting state"""
        self.current_capital = self.initial_capital
        self.positions = {}
        self.closed_trades = []
        self.equity_history = []
        self.daily_pnl = []
        self.risk_manager = RiskManager(self.initial_capital)
    
    def _filter_data_by_date(self, data: Dict[str, pd.DataFrame], 
                           start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Filter data by date range"""
        filtered_data = {}
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        for symbol, df in data.items():
            mask = (df.index >= start_dt) & (df.index <= end_dt)
            filtered_df = df.loc[mask]
            if not filtered_df.empty:
                filtered_data[symbol] = filtered_df
        
        return filtered_data
    
    def _process_trading_day(self, strategy: BaseStrategy, data: Dict[str, pd.DataFrame], 
                           current_date: datetime):
        """Process a single trading day"""
        try:
            # Update portfolio metrics
            current_value = self._calculate_portfolio_value(data, current_date)
            self.risk_manager.update_portfolio_metrics(current_value)
            
            # Check if trading should be stopped
            should_stop, reason = self.risk_manager.should_stop_trading()
            if should_stop:
                logger.warning(f"Trading stopped on {current_date}: {reason}")
                return
            
            # Check existing positions for stop loss/take profit
            self._check_exit_conditions(data, current_date)
            
            # Generate new signals for each symbol
            for symbol, symbol_data in data.items():
                if current_date not in symbol_data.index:
                    continue
                
                current_price = symbol_data.loc[current_date, 'close']
                
                # Get historical data up to current date
                historical_data = symbol_data.loc[symbol_data.index <= current_date]
                
                if len(historical_data) < strategy.get_minimum_data_length():
                    continue
                
                # Generate signal
                portfolio_info = self._get_portfolio_info()
                signal = strategy.generate_signal(historical_data, current_price, portfolio_info)
                
                # Process signal
                self._process_signal(signal, symbol, current_price, current_date, strategy)
            
            # Record daily equity
            self.equity_history.append({
                'date': current_date,
                'equity': current_value,
                'cash': self.current_capital
            })
            
        except Exception as e:
            logger.error(f"Error processing trading day {current_date}: {e}")
    
    def _check_exit_conditions(self, data: Dict[str, pd.DataFrame], current_date: datetime):
        """Check stop loss and take profit conditions"""
        positions_to_close = []
        
        for symbol, trade in self.positions.items():
            if symbol not in data or current_date not in data[symbol].index:
                continue
            
            current_price = data[symbol].loc[current_date, 'close']
            
            # Check stop loss
            if trade.stop_loss:
                if ((trade.direction == 'buy' and current_price <= trade.stop_loss) or
                    (trade.direction == 'sell' and current_price >= trade.stop_loss)):
                    positions_to_close.append((symbol, current_price, 'stop_loss'))
                    continue
            
            # Check take profit
            if trade.take_profit:
                if ((trade.direction == 'buy' and current_price >= trade.take_profit) or
                    (trade.direction == 'sell' and current_price <= trade.take_profit)):
                    positions_to_close.append((symbol, current_price, 'take_profit'))
        
        # Close positions
        for symbol, price, reason in positions_to_close:
            self._close_position(symbol, price, current_date, reason)
    
    def _process_signal(self, signal: Dict, symbol: str, current_price: float, 
                       current_date: datetime, strategy: BaseStrategy):
        """Process trading signal"""
        try:
            action = signal.get('action', 'hold')
            confidence = signal.get('confidence', 0)
            
            if action == 'hold' or confidence < 0.5:
                return
            
            # Check if we already have a position in this symbol
            if symbol in self.positions:
                existing_trade = self.positions[symbol]
                
                # Check if signal suggests closing position
                if ((existing_trade.direction == 'buy' and action == 'sell') or
                    (existing_trade.direction == 'sell' and action == 'buy')):
                    self._close_position(symbol, current_price, current_date, 'signal_reversal')
                return
            
            # Validate trade with risk manager
            portfolio_info = self._get_portfolio_info()
            is_valid, validation_reason = self.risk_manager.validate_trade(
                signal, portfolio_info, current_price
            )
            
            if not is_valid:
                logger.debug(f"Trade rejected for {symbol}: {validation_reason}")
                return
            
            # Calculate position size
            stop_loss_price = signal.get('stop_loss', 
                self.risk_manager.calculate_stop_loss(current_price, action))
            
            position_size = self.risk_manager.calculate_position_size(
                self.current_capital, current_price, stop_loss_price
            )
            
            if position_size <= 0:
                return
            
            # Open position
            self._open_position(
                symbol, action, current_price, position_size, current_date,
                signal.get('stop_loss'), signal.get('take_profit'), strategy.name
            )
            
        except Exception as e:
            logger.error(f"Error processing signal for {symbol}: {e}")
    
    def _open_position(self, symbol: str, direction: str, price: float, quantity: int,
                      entry_time: datetime, stop_loss: Optional[float], 
                      take_profit: Optional[float], strategy_name: str):
        """Open a new position"""
        try:
            trade_value = price * quantity
            fees = trade_value * self.commission
            
            # Check if we have enough capital
            if trade_value + fees > self.current_capital:
                logger.debug(f"Insufficient capital for {symbol} trade")
                return
            
            # Update capital
            self.current_capital -= trade_value + fees
            
            # Create trade record
            trade = Trade(
                entry_time=entry_time,
                exit_time=None,
                symbol=symbol,
                direction=direction,
                entry_price=price,
                exit_price=None,
                quantity=quantity,
                stop_loss=stop_loss,
                take_profit=take_profit,
                pnl=None,
                fees=fees,
                strategy=strategy_name,
                is_open=True
            )
            
            self.positions[symbol] = trade
            
            logger.debug(f"Opened {direction} position in {symbol}: {quantity} @ {price}")
            
        except Exception as e:
            logger.error(f"Error opening position for {symbol}: {e}")
    
    def _close_position(self, symbol: str, price: float, exit_time: datetime, reason: str):
        """Close an existing position"""
        try:
            if symbol not in self.positions:
                return
            
            trade = self.positions[symbol]
            trade_value = price * trade.quantity
            fees = trade_value * self.commission
            
            # Calculate P&L
            if trade.direction == 'buy':
                pnl = (price - trade.entry_price) * trade.quantity - trade.fees - fees
            else:  # sell
                pnl = (trade.entry_price - price) * trade.quantity - trade.fees - fees
            
            # Update capital
            self.current_capital += trade_value - fees
            
            # Update trade record
            trade.exit_time = exit_time
            trade.exit_price = price
            trade.pnl = pnl
            trade.fees += fees
            trade.is_open = False
            
            # Move to closed trades
            self.closed_trades.append(trade)
            del self.positions[symbol]
            
            # Record trade with risk manager
            self.risk_manager.record_trade({
                'figi': symbol,
                'direction': trade.direction,
                'quantity': trade.quantity,
                'entry_price': trade.entry_price,
                'exit_price': price,
                'pnl': pnl,
                'strategy': trade.strategy
            })
            
            logger.debug(f"Closed {trade.direction} position in {symbol}: P&L {pnl:.2f} ({reason})")
            
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
    
    def _close_all_positions(self, final_date: datetime, data: Dict[str, pd.DataFrame]):
        """Close all open positions at the end of backtest"""
        symbols_to_close = list(self.positions.keys())
        
        for symbol in symbols_to_close:
            if symbol in data and final_date in data[symbol].index:
                final_price = data[symbol].loc[final_date, 'close']
                self._close_position(symbol, final_price, final_date, 'backtest_end')
    
    def _calculate_portfolio_value(self, data: Dict[str, pd.DataFrame], 
                                 current_date: datetime) -> float:
        """Calculate total portfolio value"""
        total_value = self.current_capital
        
        for symbol, trade in self.positions.items():
            if symbol in data and current_date in data[symbol].index:
                current_price = data[symbol].loc[current_date, 'close']
                position_value = current_price * trade.quantity
                total_value += position_value
        
        return total_value
    
    def _get_portfolio_info(self) -> Dict:
        """Get current portfolio information"""
        return {
            'total_amount': self.current_capital,
            'positions': [
                {
                    'figi': symbol,
                    'quantity': trade.quantity,
                    'current_price': trade.entry_price,  # Approximation
                }
                for symbol, trade in self.positions.items()
            ]
        }
    
    def _calculate_results(self, strategy: BaseStrategy, start_date: datetime, 
                         end_date: datetime) -> BacktestResult:
        """Calculate backtest results"""
        try:
            if not self.equity_history:
                raise ValueError("No equity history available")
            
            # Create equity curve
            equity_df = pd.DataFrame(self.equity_history)
            equity_df.set_index('date', inplace=True)
            equity_curve = equity_df['equity']
            
            # Calculate returns
            daily_returns = equity_curve.pct_change().dropna()
            
            # Performance metrics
            total_days = (end_date - start_date).days
            total_return = (equity_curve.iloc[-1] - self.initial_capital) / self.initial_capital
            annualized_return = (1 + total_return) ** (365 / total_days) - 1
            
            volatility = daily_returns.std() * np.sqrt(252)
            sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 0 else 0  # Assuming 2% risk-free rate
            
            # Drawdown
            rolling_max = equity_curve.expanding().max()
            drawdown = (equity_curve - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Trade statistics
            winning_trades = [t for t in self.closed_trades if t.pnl > 0]
            losing_trades = [t for t in self.closed_trades if t.pnl <= 0]
            
            win_rate = len(winning_trades) / len(self.closed_trades) if self.closed_trades else 0
            
            avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
            
            profit_factor = abs(avg_win / avg_loss) if avg_loss < 0 else 0
            
            largest_win = max([t.pnl for t in winning_trades]) if winning_trades else 0
            largest_loss = min([t.pnl for t in losing_trades]) if losing_trades else 0
            
            # Trade duration
            completed_trades = [t for t in self.closed_trades if t.exit_time]
            avg_trade_duration = np.mean([
                (t.exit_time - t.entry_time).total_seconds() / 86400  # days
                for t in completed_trades
            ]) if completed_trades else 0
            
            # Consecutive wins/losses
            consecutive_wins, consecutive_losses = self._calculate_consecutive_trades()
            
            # Monthly returns
            monthly_returns = equity_curve.resample('M').last().pct_change().dropna()
            
            return BacktestResult(
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                total_trades=len(self.closed_trades),
                avg_trade_duration=avg_trade_duration,
                avg_win=avg_win,
                avg_loss=avg_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                consecutive_wins=consecutive_wins,
                consecutive_losses=consecutive_losses,
                monthly_returns=monthly_returns,
                daily_returns=daily_returns,
                equity_curve=equity_curve,
                trades=self.closed_trades,
                performance_metrics=self.risk_manager.get_performance_metrics()
            )
            
        except Exception as e:
            logger.error(f"Error calculating results: {e}")
            raise
    
    def _calculate_consecutive_trades(self) -> Tuple[int, int]:
        """Calculate maximum consecutive wins and losses"""
        if not self.closed_trades:
            return 0, 0
        
        current_wins = 0
        current_losses = 0
        max_wins = 0
        max_losses = 0
        
        for trade in self.closed_trades:
            if trade.pnl > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
        
        return max_wins, max_losses
    
    def plot_results(self, result: BacktestResult, strategy_name: str, save_path: str = None):
        """Plot backtest results"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Backtest Results: {strategy_name}', fontsize=16)
            
            # Equity curve
            axes[0, 0].plot(result.equity_curve.index, result.equity_curve.values)
            axes[0, 0].set_title('Equity Curve')
            axes[0, 0].set_ylabel('Portfolio Value')
            axes[0, 0].grid(True)
            
            # Drawdown
            rolling_max = result.equity_curve.expanding().max()
            drawdown = (result.equity_curve - rolling_max) / rolling_max
            axes[0, 1].fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
            axes[0, 1].set_title(f'Drawdown (Max: {result.max_drawdown:.2%})')
            axes[0, 1].set_ylabel('Drawdown %')
            axes[0, 1].grid(True)
            
            # Monthly returns
            axes[1, 0].bar(range(len(result.monthly_returns)), result.monthly_returns.values)
            axes[1, 0].set_title('Monthly Returns')
            axes[1, 0].set_ylabel('Return %')
            axes[1, 0].grid(True)
            
            # Return distribution
            axes[1, 1].hist(result.daily_returns.values, bins=50, alpha=0.7)
            axes[1, 1].set_title('Daily Returns Distribution')
            axes[1, 1].set_xlabel('Return %')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting results: {e}")
    
    def generate_report(self, result: BacktestResult, strategy_name: str) -> str:
        """Generate text report of backtest results"""
        report = f"""
BACKTEST REPORT: {strategy_name}
{'='*50}

PERFORMANCE METRICS:
- Total Return: {result.total_return:.2%}
- Annualized Return: {result.annualized_return:.2%}
- Volatility: {result.volatility:.2%}
- Sharpe Ratio: {result.sharpe_ratio:.3f}
- Maximum Drawdown: {result.max_drawdown:.2%}

TRADE STATISTICS:
- Total Trades: {result.total_trades}
- Win Rate: {result.win_rate:.2%}
- Profit Factor: {result.profit_factor:.3f}
- Average Trade Duration: {result.avg_trade_duration:.1f} days
- Average Win: {result.avg_win:.2f}
- Average Loss: {result.avg_loss:.2f}
- Largest Win: {result.largest_win:.2f}
- Largest Loss: {result.largest_loss:.2f}
- Consecutive Wins: {result.consecutive_wins}
- Consecutive Losses: {result.consecutive_losses}

RISK METRICS:
- Initial Capital: {self.initial_capital:,.2f}
- Final Capital: {result.equity_curve.iloc[-1]:,.2f}
- Maximum Portfolio Value: {result.equity_curve.max():,.2f}
- Minimum Portfolio Value: {result.equity_curve.min():,.2f}
"""
        
        return report