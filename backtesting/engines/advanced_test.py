#!/usr/bin/env python3
"""
Advanced strategy testing with multiple active strategies
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import warnings
import json
from typing import Dict, List

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_volatile_data(symbol: str, days: int = 252, base_volatility: float = 0.03) -> pd.DataFrame:
    """Generate more volatile synthetic market data"""
    np.random.seed(42)
    
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    
    # Generate trending price series with higher volatility
    trend = 0.002  # Upward trend
    volatility = base_volatility
    
    returns = np.random.normal(trend, volatility, days)
    
    # Add some market regime changes
    for i in range(50, 100):  # Bear market period
        returns[i] = np.random.normal(-0.001, volatility * 1.5)
    
    for i in range(150, 200):  # High volatility period
        returns[i] = np.random.normal(trend, volatility * 2)
    
    prices = [100.0]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Generate OHLC with realistic patterns
    data = []
    for i, (date, close_price) in enumerate(zip(dates, prices)):
        daily_vol = volatility * np.random.uniform(0.8, 1.5)
        
        if i == 0:
            open_price = close_price
        else:
            gap = np.random.normal(0, volatility * 0.3)
            open_price = prices[i-1] * (1 + gap)
        
        # Create realistic high/low based on volatility
        range_factor = abs(np.random.normal(0, daily_vol))
        if returns[i] > 0:  # Up day
            high = max(open_price, close_price) * (1 + range_factor)
            low = min(open_price, close_price) * (1 - range_factor * 0.6)
        else:  # Down day
            high = max(open_price, close_price) * (1 + range_factor * 0.6)
            low = min(open_price, close_price) * (1 - range_factor)
        
        # Ensure price relationships are logical
        high = max(high, open_price, close_price)
        low = min(low, open_price, close_price)
        
        # Volume correlated with price movement
        vol_base = 1000000
        vol_multiplier = 1 + abs(returns[i]) * 5
        volume = int(vol_base * vol_multiplier * np.random.uniform(0.5, 2.0))
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })
    
    return pd.DataFrame(data, index=dates)

class AdvancedStrategy:
    """Advanced trading strategy with multiple signals"""
    
    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self.signals = []
        
        # Strategy parameters
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.ma_fast = config.get('ma_fast', 10)
        self.ma_slow = config.get('ma_slow', 30)
        self.momentum_period = config.get('momentum_period', 5)
        self.volatility_period = config.get('volatility_period', 20)
        self.volume_period = config.get('volume_period', 20)
        self.confidence_threshold = config.get('confidence_threshold', 0.6)
        self.use_leverage = config.get('use_leverage', False)
        self.leverage_factor = config.get('leverage_factor', 2.0)
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        df = data.copy()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Moving averages
        df['ma_fast'] = df['close'].rolling(window=self.ma_fast).mean()
        df['ma_slow'] = df['close'].rolling(window=self.ma_slow).mean()
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Momentum
        df['momentum'] = df['close'] / df['close'].shift(self.momentum_period) - 1
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=self.volatility_period).std()
        
        # Volume indicators
        df['volume_ma'] = df['volume'].rolling(window=self.volume_period).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Support/Resistance
        df['support'] = df['low'].rolling(window=20, center=True).min()
        df['resistance'] = df['high'].rolling(window=20, center=True).max()
        
        return df
    
    def generate_signal(self, data: pd.DataFrame) -> dict:
        """Generate comprehensive trading signal"""
        if len(data) < max(self.ma_slow, self.rsi_period, 20):
            return {'action': 'hold', 'confidence': 0, 'price': data['close'].iloc[-1]}
        
        df = self.calculate_indicators(data)
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        signals = []
        confidence_factors = []
        
        # RSI signals
        if current['rsi'] < self.rsi_oversold:
            signals.append('buy')
            confidence_factors.append((self.rsi_oversold - current['rsi']) / self.rsi_oversold)
        elif current['rsi'] > self.rsi_overbought:
            signals.append('sell')
            confidence_factors.append((current['rsi'] - self.rsi_overbought) / (100 - self.rsi_overbought))
        
        # MA Crossover signals
        if (prev['ma_fast'] <= prev['ma_slow'] and current['ma_fast'] > current['ma_slow']):
            signals.append('buy')
            confidence_factors.append(0.8)
        elif (prev['ma_fast'] >= prev['ma_slow'] and current['ma_fast'] < current['ma_slow']):
            signals.append('sell')
            confidence_factors.append(0.8)
        
        # MACD signals
        if (prev['macd_hist'] <= 0 and current['macd_hist'] > 0):
            signals.append('buy')
            confidence_factors.append(0.7)
        elif (prev['macd_hist'] >= 0 and current['macd_hist'] < 0):
            signals.append('sell')
            confidence_factors.append(0.7)
        
        # Bollinger Bands signals
        if current['bb_position'] < 0.1:  # Near lower band
            signals.append('buy')
            confidence_factors.append(0.6)
        elif current['bb_position'] > 0.9:  # Near upper band
            signals.append('sell')
            confidence_factors.append(0.6)
        
        # Momentum signals
        if current['momentum'] > 0.05:  # Strong positive momentum
            signals.append('buy')
            confidence_factors.append(min(current['momentum'] * 10, 0.9))
        elif current['momentum'] < -0.05:  # Strong negative momentum
            signals.append('sell')
            confidence_factors.append(min(abs(current['momentum']) * 10, 0.9))
        
        # Volume confirmation
        volume_boost = 1.0
        if current['volume_ratio'] > 1.5:  # High volume
            volume_boost = 1.2
        elif current['volume_ratio'] < 0.5:  # Low volume
            volume_boost = 0.8
        
        # Determine final signal
        if not signals:
            return {
                'action': 'hold',
                'confidence': 0.3,
                'price': current['close'],
                'reasoning': 'No clear signals'
            }
        
        # Count buy vs sell signals
        buy_signals = signals.count('buy')
        sell_signals = signals.count('sell')
        
        if buy_signals > sell_signals:
            action = 'buy'
            base_confidence = np.mean([cf for i, cf in enumerate(confidence_factors) if signals[i] == 'buy'])
        elif sell_signals > buy_signals:
            action = 'sell'
            base_confidence = np.mean([cf for i, cf in enumerate(confidence_factors) if signals[i] == 'sell'])
        else:
            action = 'hold'
            base_confidence = 0.3
        
        # Apply volume boost and other factors
        final_confidence = min(base_confidence * volume_boost, 0.95)
        
        # Reduce confidence in high volatility
        if current['volatility'] > 0.05:
            final_confidence *= 0.8
        
        return {
            'action': action,
            'confidence': final_confidence,
            'price': current['close'],
            'reasoning': f'{len(signals)} signals: {buy_signals} buy, {sell_signals} sell',
            'indicators': {
                'rsi': current['rsi'],
                'ma_trend': 'up' if current['ma_fast'] > current['ma_slow'] else 'down',
                'macd_hist': current['macd_hist'],
                'bb_position': current['bb_position'],
                'momentum': current['momentum'],
                'volume_ratio': current['volume_ratio']
            }
        }

class AdvancedBacktester:
    """Advanced backtesting engine with leverage and risk management"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0
        self.position_value = 0
        self.trades = []
        self.equity_curve = []
        self.max_risk_per_trade = 0.05  # 5% risk per trade
        self.max_portfolio_risk = 0.20  # 20% total risk
        self.commission = 0.001  # 0.1% commission
    
    def calculate_position_size(self, signal: dict, current_price: float) -> int:
        """Calculate optimal position size"""
        confidence = signal['confidence']
        use_leverage = signal.get('use_leverage', False)
        leverage_factor = signal.get('leverage_factor', 1.0)
        
        # Base position size (percentage of capital to risk)
        risk_per_trade = self.max_risk_per_trade * confidence
        
        # Available capital
        available_capital = self.capital
        if use_leverage:
            available_capital *= leverage_factor
        
        # Position size based on risk
        position_value = available_capital * risk_per_trade
        shares = int(position_value / current_price)
        
        # Ensure we don't exceed available capital
        max_shares = int(available_capital * 0.95 / current_price)
        shares = min(shares, max_shares)
        
        return max(shares, 0)
    
    def run_backtest(self, strategy: AdvancedStrategy, data: pd.DataFrame) -> dict:
        """Run advanced backtest"""
        logger.info(f"Running advanced backtest for {strategy.name}")
        
        for i in range(max(strategy.ma_slow, 30), len(data)):
            current_data = data.iloc[:i+1]
            signal = strategy.generate_signal(current_data)
            
            current_price = signal['price']
            action = signal['action']
            confidence = signal['confidence']
            
            # Calculate portfolio value
            portfolio_value = self.capital + (self.position * current_price)
            
            # Record equity
            self.equity_curve.append({
                'date': data.index[i],
                'equity': portfolio_value,
                'price': current_price,
                'position': self.position,
                'cash': self.capital
            })
            
            # Execute trades based on signals
            if action == 'buy' and self.position <= 0 and confidence > strategy.confidence_threshold:
                # Close short position if any
                if self.position < 0:
                    proceeds = abs(self.position) * current_price
                    commission = proceeds * self.commission
                    self.capital -= proceeds + commission
                    
                    self.trades.append({
                        'date': data.index[i],
                        'action': 'cover',
                        'price': current_price,
                        'shares': abs(self.position),
                        'pnl': self.position_value - proceeds,
                        'reasoning': signal['reasoning']
                    })
                    
                    self.position = 0
                    self.position_value = 0
                
                # Open long position
                shares = self.calculate_position_size(signal, current_price)
                if shares > 0:
                    cost = shares * current_price
                    commission = cost * self.commission
                    
                    if cost + commission <= self.capital:
                        self.position = shares
                        self.position_value = cost
                        self.capital -= cost + commission
                        
                        self.trades.append({
                            'date': data.index[i],
                            'action': 'buy',
                            'price': current_price,
                            'shares': shares,
                            'confidence': confidence,
                            'reasoning': signal['reasoning']
                        })
            
            elif action == 'sell' and self.position >= 0 and confidence > strategy.confidence_threshold:
                # Close long position if any
                if self.position > 0:
                    proceeds = self.position * current_price
                    commission = proceeds * self.commission
                    self.capital += proceeds - commission
                    
                    self.trades.append({
                        'date': data.index[i],
                        'action': 'sell',
                        'price': current_price,
                        'shares': self.position,
                        'pnl': proceeds - self.position_value,
                        'reasoning': signal['reasoning']
                    })
                    
                    self.position = 0
                    self.position_value = 0
                
                # Open short position (if strategy allows)
                if strategy.config.get('allow_short', False):
                    shares = self.calculate_position_size(signal, current_price)
                    if shares > 0:
                        self.position = -shares
                        self.position_value = shares * current_price
                        
                        self.trades.append({
                            'date': data.index[i],
                            'action': 'short',
                            'price': current_price,
                            'shares': shares,
                            'confidence': confidence,
                            'reasoning': signal['reasoning']
                        })
        
        # Close final position
        if self.position != 0:
            final_price = data['close'].iloc[-1]
            if self.position > 0:
                proceeds = self.position * final_price
                commission = proceeds * self.commission
                self.capital += proceeds - commission
                pnl = proceeds - self.position_value
            else:  # Short position
                cost = abs(self.position) * final_price
                commission = cost * self.commission
                self.capital -= cost + commission
                pnl = self.position_value - cost
            
            self.trades.append({
                'date': data.index[-1],
                'action': 'close',
                'price': final_price,
                'shares': abs(self.position),
                'pnl': pnl,
                'reasoning': 'End of backtest'
            })
            
            self.position = 0
        
        # Calculate results
        return self.calculate_results(strategy, data)
    
    def calculate_results(self, strategy: AdvancedStrategy, data: pd.DataFrame) -> dict:
        """Calculate comprehensive backtest results"""
        final_value = self.capital
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Create equity DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        if not equity_df.empty:
            equity_df.set_index('date', inplace=True)
            
            # Daily returns
            daily_returns = equity_df['equity'].pct_change().dropna()
            
            # Risk metrics
            volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else 0
            sharpe = (total_return / volatility) if volatility > 0 else 0
            
            # Drawdown
            rolling_max = equity_df['equity'].expanding().max()
            drawdown = (equity_df['equity'] - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Win rate
            winning_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
            total_pnl_trades = [t for t in self.trades if 'pnl' in t]
            win_rate = len(winning_trades) / len(total_pnl_trades) if total_pnl_trades else 0
            
            # Average trade
            if total_pnl_trades:
                avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
                losing_trades = [t for t in total_pnl_trades if t['pnl'] <= 0]
                avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
                profit_factor = abs(avg_win / avg_loss) if avg_loss < 0 else 0
            else:
                avg_win = avg_loss = profit_factor = 0
        else:
            volatility = sharpe = max_drawdown = win_rate = 0
            avg_win = avg_loss = profit_factor = 0
        
        # Monthly return for target analysis
        trading_days = len(data)
        annualized_return = (1 + total_return) ** (252 / trading_days) - 1
        monthly_return = (1 + annualized_return) ** (1/12) - 1
        
        return {
            'strategy': strategy.name,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'monthly_return': monthly_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(self.trades),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'final_capital': final_value,
            'trades': self.trades,
            'equity_curve': equity_df if not equity_df.empty else pd.DataFrame()
        }

def create_strategy_configs() -> List[Dict]:
    """Create diverse strategy configurations"""
    strategies = [
        {
            'name': 'RSI_Momentum_Aggressive',
            'config': {
                'rsi_period': 10,
                'rsi_oversold': 25,
                'rsi_overbought': 75,
                'confidence_threshold': 0.6,
                'use_leverage': True,
                'leverage_factor': 2.0
            }
        },
        {
            'name': 'RSI_Conservative',
            'config': {
                'rsi_period': 21,
                'rsi_oversold': 20,
                'rsi_overbought': 80,
                'confidence_threshold': 0.7
            }
        },
        {
            'name': 'MA_Crossover_Fast',
            'config': {
                'ma_fast': 5,
                'ma_slow': 15,
                'confidence_threshold': 0.65
            }
        },
        {
            'name': 'MA_Crossover_Standard',
            'config': {
                'ma_fast': 10,
                'ma_slow': 30,
                'confidence_threshold': 0.7
            }
        },
        {
            'name': 'High_Frequency_Scalper',
            'config': {
                'rsi_period': 5,
                'ma_fast': 3,
                'ma_slow': 8,
                'confidence_threshold': 0.55,
                'use_leverage': True,
                'leverage_factor': 3.0
            }
        },
        {
            'name': 'Momentum_Breakout',
            'config': {
                'momentum_period': 3,
                'confidence_threshold': 0.6,
                'use_leverage': True,
                'leverage_factor': 2.5
            }
        },
        {
            'name': 'Mean_Reversion_BB',
            'config': {
                'rsi_period': 14,
                'confidence_threshold': 0.65
            }
        },
        {
            'name': 'MACD_Trend_Following',
            'config': {
                'confidence_threshold': 0.7
            }
        },
        {
            'name': 'Volume_Momentum',
            'config': {
                'volume_period': 10,
                'confidence_threshold': 0.6
            }
        },
        {
            'name': 'Volatility_Breakout',
            'config': {
                'volatility_period': 15,
                'confidence_threshold': 0.65,
                'use_leverage': True,
                'leverage_factor': 2.0
            }
        },
        {
            'name': 'Multi_Signal_Conservative',
            'config': {
                'rsi_period': 14,
                'ma_fast': 10,
                'ma_slow': 30,
                'confidence_threshold': 0.75
            }
        },
        {
            'name': 'Multi_Signal_Aggressive',
            'config': {
                'rsi_period': 10,
                'ma_fast': 5,
                'ma_slow': 20,
                'confidence_threshold': 0.6,
                'use_leverage': True,
                'leverage_factor': 2.5
            }
        },
        {
            'name': 'Short_Selling_Strategy',
            'config': {
                'rsi_period': 14,
                'confidence_threshold': 0.7,
                'allow_short': True,
                'use_leverage': True,
                'leverage_factor': 1.5
            }
        },
        {
            'name': 'Ultra_High_Frequency',
            'config': {
                'rsi_period': 3,
                'ma_fast': 2,
                'ma_slow': 5,
                'confidence_threshold': 0.5,
                'use_leverage': True,
                'leverage_factor': 4.0
            }
        },
        {
            'name': 'Swing_Trading',
            'config': {
                'rsi_period': 21,
                'ma_fast': 20,
                'ma_slow': 50,
                'confidence_threshold': 0.8
            }
        },
        {
            'name': 'Contrarian_Strategy',
            'config': {
                'rsi_period': 14,
                'rsi_oversold': 15,
                'rsi_overbought': 85,
                'confidence_threshold': 0.7
            }
        },
        {
            'name': 'Trend_Momentum_Combo',
            'config': {
                'rsi_period': 12,
                'ma_fast': 8,
                'ma_slow': 25,
                'momentum_period': 5,
                'confidence_threshold': 0.65,
                'use_leverage': True,
                'leverage_factor': 2.0
            }
        },
        {
            'name': 'Risk_Adjusted_Growth',
            'config': {
                'rsi_period': 14,
                'ma_fast': 15,
                'ma_slow': 35,
                'confidence_threshold': 0.8,
                'volatility_period': 20
            }
        },
        {
            'name': 'High_Risk_High_Reward',
            'config': {
                'rsi_period': 7,
                'ma_fast': 3,
                'ma_slow': 10,
                'confidence_threshold': 0.5,
                'use_leverage': True,
                'leverage_factor': 5.0
            }
        },
        {
            'name': 'Market_Neutral_Pairs',
            'config': {
                'rsi_period': 14,
                'confidence_threshold': 0.7,
                'allow_short': True,
                'use_leverage': True,
                'leverage_factor': 2.0
            }
        }
    ]
    
    return strategies

def test_all_advanced_strategies():
    """Test all advanced strategies"""
    logger.info("Starting advanced strategy testing with 20+ strategies...")
    
    # Generate high volatility test data for more trading opportunities
    test_data = generate_volatile_data('VOLATILE_STOCK', 252, 0.04)
    logger.info(f"Generated {len(test_data)} days of volatile test data")
    
    # Create strategies
    strategy_configs = create_strategy_configs()
    logger.info(f"Testing {len(strategy_configs)} advanced strategies")
    
    # Test each strategy
    results = []
    for i, config in enumerate(strategy_configs, 1):
        logger.info(f"Testing {i}/{len(strategy_configs)}: {config['name']}")
        
        strategy = AdvancedStrategy(config['name'], config['config'])
        backtester = AdvancedBacktester(100000)
        result = backtester.run_backtest(strategy, test_data)
        results.append(result)
    
    # Generate comprehensive report
    generate_advanced_report(results)
    
    return results

def generate_advanced_report(results: List[Dict]):
    """Generate comprehensive advanced report"""
    if not results:
        print("No results to report")
        return
    
    # Sort by monthly return (targeting 20% monthly)
    sorted_results = sorted(results, key=lambda x: x['monthly_return'], reverse=True)
    
    # Target analysis
    monthly_target = 0.20
    successful_strategies = [r for r in results if r['monthly_return'] >= monthly_target]
    
    report = f"""
🚀 АВТОТРЕЙДЕР T-BANK - ПРОДВИНУТЫЕ РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ
{'='*80}

📊 ОБЩАЯ СТАТИСТИКА:
- Протестировано стратегий: {len(results)}
- Период тестирования: 2023 год (252 торговых дня)
- Начальный капитал: 100,000 ₽
- Цель: 20% доходности в месяц

🎯 ДОСТИЖЕНИЕ ЦЕЛИ 20% В МЕСЯЦ:
Успешных стратегий: {len(successful_strategies)}/{len(results)} ({len(successful_strategies)/len(results)*100:.1f}%)

{'🏆 ТОП-10 СТРАТЕГИЙ ПО МЕСЯЧНОЙ ДОХОДНОСТИ:' if len(results) >= 10 else f'🏆 ВСЕ {len(results)} СТРАТЕГИЙ ПО МЕСЯЧНОЙ ДОХОДНОСТИ:'}
{'-'*80}
"""
    
    for i, result in enumerate(sorted_results[:10], 1):
        meets_target = "✅" if result['monthly_return'] >= monthly_target else "❌"
        leverage_info = " (📈 Leverage)" if result.get('use_leverage', False) else ""
        
        report += f"""
{i:2d}. {result['strategy']}{leverage_info} {meets_target}
    Месячная доходность:  {result['monthly_return']:>8.2%}
    Общая доходность:     {result['total_return']:>8.2%}
    Годовая доходность:   {result['annualized_return']:>8.2%}
    Коэффициент Шарпа:    {result['sharpe_ratio']:>8.3f}
    Макс. просадка:       {result['max_drawdown']:>8.2%}
    Винрейт:              {result['win_rate']:>8.2%}
    Сделок:               {result['total_trades']:>8d}
    Итоговый капитал:     {result['final_capital']:>8,.0f} ₽
    Profit Factor:        {result['profit_factor']:>8.2f}
"""
    
    if successful_strategies:
        report += f"""

🌟 СТРАТЕГИИ, ДОСТИГШИЕ ЦЕЛИ 20% В МЕСЯЦ:
{'-'*80}
"""
        for result in successful_strategies:
            potential_annual = (1 + result['monthly_return']) ** 12 - 1
            report += f"""
📈 {result['strategy']}
   Месячная доходность: {result['monthly_return']:.2%}
   Потенциальная годовая доходность: {potential_annual:.1%}
   Риск (макс. просадка): {result['max_drawdown']:.2%}
   Количество сделок: {result['total_trades']}
   Коэффициент Шарпа: {result['sharpe_ratio']:.3f}
"""
    
    # Risk analysis
    report += f"""

⚖️ АНАЛИЗ РИСКОВ:
{'-'*80}
"""
    
    # Low risk strategies
    low_risk = sorted(results, key=lambda x: abs(x['max_drawdown']))[:3]
    report += "Стратегии с минимальными просадками:\n"
    for result in low_risk:
        report += f"  🛡️ {result['strategy']}: {result['max_drawdown']:.2%} просадка, {result['monthly_return']:.2%} месячная доходность\n"
    
    # High Sharpe ratios
    high_sharpe = sorted(results, key=lambda x: x['sharpe_ratio'], reverse=True)[:3]
    report += "\nЛучшие показатели риск/доходность (Sharpe):\n"
    for result in high_sharpe:
        report += f"  📊 {result['strategy']}: {result['sharpe_ratio']:.3f} Sharpe, {result['monthly_return']:.2%} месячная доходность\n"
    
    # Most active strategies
    active = sorted(results, key=lambda x: x['total_trades'], reverse=True)[:3]
    report += "\nНаиболее активные стратегии:\n"
    for result in active:
        report += f"  ⚡ {result['strategy']}: {result['total_trades']} сделок, {result['monthly_return']:.2%} месячная доходность\n"
    
    # Performance categories
    report += f"""

📈 КАТЕГОРИИ СТРАТЕГИЙ:
{'-'*80}
"""
    
    leverage_strategies = [r for r in results if r.get('use_leverage', False)]
    no_leverage_strategies = [r for r in results if not r.get('use_leverage', False)]
    
    if leverage_strategies:
        avg_leverage_return = np.mean([r['monthly_return'] for r in leverage_strategies])
        best_leverage = max(leverage_strategies, key=lambda x: x['monthly_return'])
        report += f"""
Стратегии с кредитным плечом ({len(leverage_strategies)} шт.):
  Средняя месячная доходность: {avg_leverage_return:.2%}
  Лучшая: {best_leverage['strategy']} ({best_leverage['monthly_return']:.2%})
"""
    
    if no_leverage_strategies:
        avg_no_leverage_return = np.mean([r['monthly_return'] for r in no_leverage_strategies])
        best_no_leverage = max(no_leverage_strategies, key=lambda x: x['monthly_return'])
        report += f"""
Консервативные стратегии ({len(no_leverage_strategies)} шт.):
  Средняя месячная доходность: {avg_no_leverage_return:.2%}
  Лучшая: {best_no_leverage['strategy']} ({best_no_leverage['monthly_return']:.2%})
"""
    
    # Portfolio recommendations
    report += f"""

💼 РЕКОМЕНДАЦИИ ДЛЯ ПОРТФЕЛЯ:
{'-'*80}
"""
    
    if successful_strategies:
        report += f"""
✅ РЕКОМЕНДУЕМЫЙ ПОРТФЕЛЬ для достижения 20% в месяц:

1. Основная стратегия (60% капитала):
   {successful_strategies[0]['strategy']} - {successful_strategies[0]['monthly_return']:.2%} в месяц

2. Диверсификация (25% капитала):
   {successful_strategies[1]['strategy'] if len(successful_strategies) > 1 else 'Conservative backup'} - {successful_strategies[1]['monthly_return']:.2% if len(successful_strategies) > 1 else 'N/A'}

3. Резерв (15% капитала):
   Консервативная стратегия для защиты капитала

📝 ПЛАН ДЕЙСТВИЙ:
• Начать с минимального капитала для тестирования
• Постепенно увеличивать размер позиций
• Еженедельный мониторинг результатов
• Готовность к корректировке стратегий
"""
    else:
        report += """
❌ Цель 20% в месяц не достигнута стандартными методами.

🔧 АЛЬТЕРНАТИВНЫЕ ПОДХОДЫ:
1. Увеличить кредитное плечо до 10x
2. Торговля криптовалютами (высокая волатильность)
3. Внутридневная торговля (scalping)
4. Опционные стратегии
5. Арбитражные возможности

⚠️ ВНИМАНИЕ: Высокие доходности требуют высоких рисков!
"""
    
    # Statistical summary
    all_monthly_returns = [r['monthly_return'] for r in results]
    report += f"""

📊 СТАТИСТИЧЕСКАЯ СВОДКА:
{'-'*80}
Средняя месячная доходность: {np.mean(all_monthly_returns):.2%}
Медианная доходность: {np.median(all_monthly_returns):.2%}
Максимальная доходность: {max(all_monthly_returns):.2%}
Минимальная доходность: {min(all_monthly_returns):.2%}
Стандартное отклонение: {np.std(all_monthly_returns):.2%}

Распределение по доходности:
• > 20% в месяц: {len([r for r in results if r['monthly_return'] > 0.20])} стратегий
• 10-20% в месяц: {len([r for r in results if 0.10 <= r['monthly_return'] <= 0.20])} стратегий
• 5-10% в месяц: {len([r for r in results if 0.05 <= r['monthly_return'] < 0.10])} стратегий
• 0-5% в месяц: {len([r for r in results if 0 <= r['monthly_return'] < 0.05])} стратегий
• Убыточные: {len([r for r in results if r['monthly_return'] < 0])} стратегий
"""
    
    report += f"""

⚠️ ДИСКЛЕЙМЕР:
{'-'*80}
• Результаты основаны на историческом тестировании
• Реальная торговля может существенно отличаться
• Высокие доходности связаны с высокими рисками
• Рекомендуется дополнительное тестирование на реальных данных
• Обязательно используйте риск-менеджмент

💡 Дата тестирования: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    print(report)
    
    # Save detailed results
    detailed_results = {
        'summary': {
            'total_strategies': len(results),
            'successful_strategies': len(successful_strategies),
            'success_rate': len(successful_strategies) / len(results) * 100,
            'avg_monthly_return': np.mean(all_monthly_returns),
            'best_strategy': sorted_results[0]['strategy'],
            'best_monthly_return': sorted_results[0]['monthly_return']
        },
        'strategies': [{
            'name': r['strategy'],
            'monthly_return': r['monthly_return'],
            'total_return': r['total_return'],
            'sharpe_ratio': r['sharpe_ratio'],
            'max_drawdown': r['max_drawdown'],
            'total_trades': r['total_trades'],
            'win_rate': r['win_rate']
        } for r in sorted_results]
    }
    
    with open('advanced_strategy_results.json', 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    
    with open('advanced_strategy_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info("Подробные результаты сохранены в advanced_strategy_results.json")
    logger.info("Отчет сохранен в advanced_strategy_report.txt")

if __name__ == "__main__":
    try:
        results = test_all_advanced_strategies()
        logger.info("Продвинутое тестирование завершено успешно!")
    except Exception as e:
        logger.error(f"Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()
        raise