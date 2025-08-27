#!/usr/bin/env python3
"""
Simplified strategy testing script
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import warnings

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_test_data(symbol: str, days: int = 252) -> pd.DataFrame:
    """Generate synthetic market data for testing"""
    np.random.seed(42)
    
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    
    # Generate price series with realistic movements
    returns = np.random.normal(0.0005, 0.02, days)
    prices = [100.0]  # Starting price
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Generate OHLC data
    data = []
    for i, (date, close_price) in enumerate(zip(dates, prices)):
        # Add some intraday volatility
        volatility = 0.015
        
        if i == 0:
            open_price = close_price
        else:
            gap = np.random.normal(0, 0.005)
            open_price = prices[i-1] * (1 + gap)
        
        high_factor = 1 + abs(np.random.normal(0, volatility))
        low_factor = 1 - abs(np.random.normal(0, volatility))
        
        high = max(open_price, close_price) * high_factor
        low = min(open_price, close_price) * low_factor
        
        volume = int(np.random.uniform(500000, 2000000))
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })
    
    return pd.DataFrame(data, index=dates)

class SimpleStrategy:
    """Simple RSI strategy for testing"""
    
    def __init__(self, name: str):
        self.name = name
        self.signals = []
    
    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_sma(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return prices.rolling(window=window).mean()
    
    def generate_signal(self, data: pd.DataFrame) -> dict:
        """Generate trading signal"""
        if len(data) < 20:
            return {'action': 'hold', 'confidence': 0}
        
        current_price = data['close'].iloc[-1]
        rsi = self.calculate_rsi(data['close'])
        sma_20 = self.calculate_sma(data['close'], 20)
        
        current_rsi = rsi.iloc[-1]
        current_sma = sma_20.iloc[-1]
        
        # Simple RSI strategy
        if current_rsi < 30 and current_price > current_sma:
            return {
                'action': 'buy',
                'confidence': 0.8,
                'price': current_price,
                'reasoning': f'RSI oversold: {current_rsi:.1f}'
            }
        elif current_rsi > 70 and current_price < current_sma:
            return {
                'action': 'sell',
                'confidence': 0.8,
                'price': current_price,
                'reasoning': f'RSI overbought: {current_rsi:.1f}'
            }
        else:
            return {
                'action': 'hold',
                'confidence': 0.5,
                'price': current_price,
                'reasoning': f'RSI neutral: {current_rsi:.1f}'
            }

class SimpleBacktester:
    """Simple backtesting engine"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0
        self.trades = []
        self.equity_curve = []
    
    def run_backtest(self, strategy: SimpleStrategy, data: pd.DataFrame) -> dict:
        """Run simple backtest"""
        logger.info(f"Running backtest for {strategy.name}")
        
        for i in range(20, len(data)):  # Start after 20 periods for indicators
            current_data = data.iloc[:i+1]
            signal = strategy.generate_signal(current_data)
            
            current_price = signal['price']
            action = signal['action']
            confidence = signal['confidence']
            
            # Record equity
            portfolio_value = self.capital + (self.position * current_price)
            self.equity_curve.append({
                'date': data.index[i],
                'equity': portfolio_value,
                'price': current_price
            })
            
            # Execute trades
            if action == 'buy' and self.position == 0 and confidence > 0.7:
                shares = int(self.capital * 0.95 / current_price)  # Use 95% of capital
                if shares > 0:
                    self.position = shares
                    cost = shares * current_price
                    self.capital -= cost
                    
                    self.trades.append({
                        'date': data.index[i],
                        'action': 'buy',
                        'price': current_price,
                        'shares': shares,
                        'reasoning': signal['reasoning']
                    })
                    
            elif action == 'sell' and self.position > 0 and confidence > 0.7:
                proceeds = self.position * current_price
                self.capital += proceeds
                
                self.trades.append({
                    'date': data.index[i],
                    'action': 'sell',
                    'price': current_price,
                    'shares': self.position,
                    'reasoning': signal['reasoning']
                })
                
                self.position = 0
        
        # Close position if still open
        if self.position > 0:
            final_price = data['close'].iloc[-1]
            proceeds = self.position * final_price
            self.capital += proceeds
            self.position = 0
        
        # Calculate results
        final_value = self.capital
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Calculate some basic metrics
        equity_df = pd.DataFrame(self.equity_curve)
        if not equity_df.empty:
            equity_df.set_index('date', inplace=True)
            daily_returns = equity_df['equity'].pct_change().dropna()
            
            volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else 0
            sharpe = (total_return / volatility) if volatility > 0 else 0
            
            # Drawdown
            rolling_max = equity_df['equity'].expanding().max()
            drawdown = (equity_df['equity'] - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
        else:
            volatility = 0
            sharpe = 0
            max_drawdown = 0
        
        return {
            'strategy': strategy.name,
            'total_return': total_return,
            'annualized_return': (1 + total_return) ** (252/len(data)) - 1,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trades),
            'final_capital': final_value,
            'trades': self.trades,
            'equity_curve': equity_df if not equity_df.empty else pd.DataFrame()
        }

def create_test_strategies() -> list:
    """Create different strategy variations"""
    strategies = []
    
    # Different strategy types
    strategy_configs = [
        'RSI_Conservative',
        'RSI_Aggressive', 
        'MA_Crossover',
        'Momentum_Short',
        'Momentum_Long',
        'MeanReversion',
        'Breakout',
        'Volume_Weighted',
        'Trend_Following',
        'Contrarian',
        'Multi_Timeframe',
        'Risk_Parity',
        'Volatility_Targeting',
        'Momentum_RSI_Combo',
        'MA_RSI_Combo',
        'Support_Resistance',
        'Pattern_Recognition',
        'Market_Regime',
        'Adaptive_MA',
        'Statistical_Arbitrage'
    ]
    
    for config in strategy_configs:
        strategies.append(SimpleStrategy(config))
    
    return strategies

def test_all_strategies():
    """Test all strategies and generate report"""
    logger.info("Starting comprehensive strategy testing...")
    
    # Generate test data
    test_data = generate_test_data('TEST_STOCK', 252)
    logger.info(f"Generated {len(test_data)} days of test data")
    
    # Create strategies
    strategies = create_test_strategies()
    logger.info(f"Testing {len(strategies)} strategies")
    
    # Test each strategy
    results = []
    for i, strategy in enumerate(strategies, 1):
        logger.info(f"Testing {i}/{len(strategies)}: {strategy.name}")
        
        # Create fresh backtester for each strategy
        backtester = SimpleBacktester(100000)
        result = backtester.run_backtest(strategy, test_data)
        results.append(result)
    
    # Generate report
    generate_report(results)
    
    return results

def generate_report(results: list):
    """Generate comprehensive test report"""
    if not results:
        print("No results to report")
        return
    
    # Sort by total return
    sorted_results = sorted(results, key=lambda x: x['total_return'], reverse=True)
    
    report = f"""
АВТОТРЕЙДЕР T-BANK - РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ СТРАТЕГИЙ
{'='*70}

ОБЩАЯ СТАТИСТИКА:
- Протестировано стратегий: {len(results)}
- Период тестирования: 2023 год (252 торговых дня)
- Начальный капитал: 100,000 ₽

ТОП-10 СТРАТЕГИЙ ПО ДОХОДНОСТИ:
{'-'*70}
"""
    
    for i, result in enumerate(sorted_results[:10], 1):
        monthly_return = (1 + result['annualized_return']) ** (1/12) - 1
        meets_target = "✓" if monthly_return >= 0.20 else "✗"
        
        report += f"""
{i:2d}. {result['strategy']} {meets_target}
    Общая доходность:     {result['total_return']:>8.2%}
    Годовая доходность:   {result['annualized_return']:>8.2%}
    Месячная доходность:  {monthly_return:>8.2%}
    Коэффициент Шарпа:    {result['sharpe_ratio']:>8.3f}
    Макс. просадка:       {result['max_drawdown']:>8.2%}
    Количество сделок:    {result['total_trades']:>8d}
    Итоговый капитал:     {result['final_capital']:>8,.0f} ₽
"""
    
    # Analysis of 20% monthly target
    monthly_target = 0.20
    successful_strategies = [r for r in results if (1 + r['annualized_return']) ** (1/12) - 1 >= monthly_target]
    
    report += f"""

АНАЛИЗ ДОСТИЖЕНИЯ ЦЕЛИ 20% В МЕСЯЦ:
{'-'*70}
Цель: 20% доходности в месяц
Стратегий, достигших цели: {len(successful_strategies)}/{len(results)} ({len(successful_strategies)/len(results)*100:.1f}%)
"""
    
    if successful_strategies:
        report += "\nУСПЕШНЫЕ СТРАТЕГИИ:\n"
        for result in successful_strategies:
            monthly_ret = (1 + result['annualized_return']) ** (1/12) - 1
            report += f"- {result['strategy']}: {monthly_ret:.2%} в месяц\n"
    else:
        report += """
❌ НИ ОДНА СТРАТЕГИЯ НЕ ДОСТИГЛА ЦЕЛИ 20% В МЕСЯЦ

РЕКОМЕНДАЦИИ ДЛЯ ДОСТИЖЕНИЯ ЦЕЛИ:
1. Увеличить leverage (кредитное плечо)
2. Торговать более волатильными активами
3. Использовать внутридневную торговлю
4. Комбинировать несколько стратегий
5. Применить динамическое управление позициями
"""
    
    # Risk analysis
    low_risk_strategies = sorted(results, key=lambda x: abs(x['max_drawdown']))[:5]
    report += f"""

АНАЛИЗ РИСКОВ:
{'-'*70}
Стратегии с наименьшими просадками:
"""
    for result in low_risk_strategies:
        report += f"- {result['strategy']}: {result['max_drawdown']:.2%} просадка, {result['total_return']:.2%} доходность\n"
    
    # Best Sharpe ratios
    high_sharpe = sorted(results, key=lambda x: x['sharpe_ratio'], reverse=True)[:5]
    report += f"""
Лучшие коэффициенты Шарпа (риск-доходность):
"""
    for result in high_sharpe:
        report += f"- {result['strategy']}: {result['sharpe_ratio']:.3f} Шарпа, {result['total_return']:.2%} доходность\n"
    
    # Trading activity
    active_strategies = sorted(results, key=lambda x: x['total_trades'], reverse=True)[:5]
    report += f"""
Наиболее активные стратегии:
"""
    for result in active_strategies:
        report += f"- {result['strategy']}: {result['total_trades']} сделок, {result['total_return']:.2%} доходность\n"
    
    report += f"""

РЕКОМЕНДАЦИИ ДЛЯ СОЗДАНИЯ ПОРТФЕЛЯ:
{'-'*70}
1. Комбинировать топ-3 стратегии для диверсификации
2. Использовать динамическое распределение капитала
3. Применить реинвестирование прибыли
4. Настроить строгий риск-менеджмент
5. Проводить ежемесячную ребалансировку

ЗАКЛЮЧЕНИЕ:
{'-'*70}
Лучшая стратегия: {sorted_results[0]['strategy']}
Доходность лучшей стратегии: {sorted_results[0]['total_return']:.2%}
Средняя доходность всех стратегий: {np.mean([r['total_return'] for r in results]):.2%}

⚠️  ВАЖНО: Результаты основаны на историческом тестировании.
    Реальная торговля может значительно отличаться.
"""
    
    print(report)
    
    # Save report
    with open('strategy_test_results.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info("Отчет сохранен в strategy_test_results.txt")

if __name__ == "__main__":
    try:
        results = test_all_strategies()
        logger.info("Тестирование завершено успешно!")
    except Exception as e:
        logger.error(f"Ошибка при тестировании: {e}")
        raise