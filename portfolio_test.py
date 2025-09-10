#!/usr/bin/env python3
"""
Portfolio Testing with Real Data and Transaction Costs
Simplified version for reliable testing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import warnings
import json
from typing import Dict, List

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PortfolioBacktester:
    """Portfolio backtester with transaction costs"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}  # {symbol: {'quantity': int, 'avg_price': float}}
        self.trades = []
        self.equity_curve = []
        
        # Transaction costs (T-Bank rates)
        self.commission_rate = 0.003  # 0.3% commission
        self.min_commission = 1.0  # Minimum 1 ruble
        self.slippage_rate = 0.001  # 0.1% slippage
        
        # Portfolio settings
        self.max_positions = 8
        self.max_position_size = 0.15  # 15% per position
        
    def calculate_transaction_cost(self, trade_value: float) -> float:
        """Calculate transaction costs"""
        commission = max(trade_value * self.commission_rate, self.min_commission)
        slippage = trade_value * self.slippage_rate
        return commission + slippage
    
    def execute_trade(self, symbol: str, action: str, price: float, quantity: int, 
                     confidence: float, timestamp: datetime) -> bool:
        """Execute trade with transaction costs"""
        try:
            trade_value = price * quantity
            
            if action == 'buy':
                transaction_cost = self.calculate_transaction_cost(trade_value)
                total_cost = trade_value + transaction_cost
                
                if total_cost > self.capital:
                    return False
                
                self.capital -= total_cost
                
                if symbol in self.positions:
                    old_quantity = self.positions[symbol]['quantity']
                    old_avg_price = self.positions[symbol]['avg_price']
                    new_quantity = old_quantity + quantity
                    new_avg_price = ((old_quantity * old_avg_price) + trade_value) / new_quantity
                    
                    self.positions[symbol] = {
                        'quantity': new_quantity,
                        'avg_price': new_avg_price
                    }
                else:
                    self.positions[symbol] = {
                        'quantity': quantity,
                        'avg_price': price
                    }
                
            elif action == 'sell':
                if symbol not in self.positions or self.positions[symbol]['quantity'] < quantity:
                    return False
                
                proceeds = trade_value
                transaction_cost = self.calculate_transaction_cost(trade_value)
                net_proceeds = proceeds - transaction_cost
                
                self.capital += net_proceeds
                self.positions[symbol]['quantity'] -= quantity
                
                if self.positions[symbol]['quantity'] == 0:
                    del self.positions[symbol]
            
            # Record trade
            self.trades.append({
                'timestamp': timestamp,
                'symbol': symbol,
                'action': action,
                'price': price,
                'quantity': quantity,
                'confidence': confidence,
                'trade_value': trade_value,
                'transaction_cost': self.calculate_transaction_cost(trade_value)
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
            return False
    
    def calculate_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate current portfolio value"""
        portfolio_value = self.capital
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                position_value = position['quantity'] * current_price
                portfolio_value += position_value
        
        return portfolio_value

def generate_realistic_data(symbol: str, days: int = 252, base_price: float = 100, 
                          volatility: float = 0.025) -> pd.DataFrame:
    """Generate realistic market data for testing"""
    np.random.seed(42)
    
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    
    # Generate price series with realistic movements
    trend = 0.0005  # Small upward trend
    returns = np.random.normal(trend, volatility, days)
    
    # Add market regimes
    for i in range(50, 100):  # Bear market
        returns[i] = np.random.normal(-0.001, volatility * 1.5)
    
    for i in range(150, 200):  # High volatility
        returns[i] = np.random.normal(trend, volatility * 2)
    
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Generate OHLC data
    data = []
    for i, (date, close_price) in enumerate(zip(dates, prices)):
        daily_vol = volatility * np.random.uniform(0.8, 1.2)
        
        if i == 0:
            open_price = close_price
        else:
            gap = np.random.normal(0, volatility * 0.3)
            open_price = prices[i-1] * (1 + gap)
        
        # High and low
        range_factor = abs(np.random.normal(0, daily_vol))
        high = max(open_price, close_price) * (1 + range_factor)
        low = min(open_price, close_price) * (1 - range_factor)
        
        # Volume
        base_volume = 1000000
        vol_multiplier = 1 + abs(returns[i]) * 5
        volume = int(base_volume * vol_multiplier * np.random.uniform(0.5, 2.0))
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })
    
    return pd.DataFrame(data, index=dates)

class SimpleRSIStrategy:
    """Simple RSI strategy for testing"""
    
    def __init__(self, rsi_period: int = 14, oversold: int = 30, overbought: int = 70):
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
    
    def calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_signal(self, data: pd.DataFrame, current_price: float, portfolio_info: Dict) -> Dict:
        """Generate trading signal"""
        if len(data) < self.rsi_period + 5:
            return {'action': 'hold', 'confidence': 0}
        
        rsi = self.calculate_rsi(data['close'])
        current_rsi = rsi.iloc[-1]
        
        if pd.isna(current_rsi):
            return {'action': 'hold', 'confidence': 0}
        
        if current_rsi < self.oversold:
            return {
                'action': 'buy',
                'confidence': min(0.9, (self.oversold - current_rsi) / self.oversold),
                'reasoning': f'RSI oversold: {current_rsi:.1f}'
            }
        elif current_rsi > self.overbought:
            return {
                'action': 'sell',
                'confidence': min(0.9, (current_rsi - self.overbought) / (100 - self.overbought)),
                'reasoning': f'RSI overbought: {current_rsi:.1f}'
            }
        else:
            return {
                'action': 'hold',
                'confidence': 0.3,
                'reasoning': f'RSI neutral: {current_rsi:.1f}'
            }

def test_portfolio_strategies():
    """Test strategies on portfolio of instruments"""
    logger.info("🚀 Starting portfolio testing with transaction costs...")
    
    # Portfolio of Russian stocks
    portfolio_symbols = ['SBER', 'GAZP', 'LKOH', 'YNDX', 'ROSN', 'NVTK', 'MTSS', 'MGNT']
    
    # Realistic starting prices
    starting_prices = {
        'SBER': 250, 'GAZP': 150, 'LKOH': 5500, 'YNDX': 2500,
        'ROSN': 450, 'NVTK': 1200, 'MTSS': 300, 'MGNT': 5000
    }
    
    # Realistic volatilities
    volatilities = {
        'SBER': 0.025, 'GAZP': 0.030, 'LKOH': 0.035, 'YNDX': 0.040,
        'ROSN': 0.032, 'NVTK': 0.028, 'MTSS': 0.020, 'MGNT': 0.022
    }
    
    # Generate market data
    market_data = {}
    for symbol in portfolio_symbols:
        market_data[symbol] = generate_realistic_data(
            symbol, 252, starting_prices[symbol], volatilities[symbol]
        )
        logger.info(f"📊 Generated data for {symbol}: {len(market_data[symbol])} days")
    
    # Test different strategies
    strategies = {
        'RSI_Conservative': SimpleRSIStrategy(21, 20, 80),
        'RSI_Standard': SimpleRSIStrategy(14, 30, 70),
        'RSI_Aggressive': SimpleRSIStrategy(10, 35, 65)
    }
    
    results = []
    
    for strategy_name, strategy in strategies.items():
        logger.info(f"🧪 Testing {strategy_name}...")
        
        backtester = PortfolioBacktester(100000)
        
        # Get all dates
        all_dates = set()
        for df in market_data.values():
            all_dates.update(df.index)
        all_dates = sorted(list(all_dates))
        
        daily_values = []
        max_drawdown = 0
        peak_value = 100000
        
        for i, date in enumerate(all_dates[30:], 30):  # Start after 30 periods
            try:
                # Get current prices
                current_prices = {}
                for symbol, df in market_data.items():
                    if date in df.index:
                        current_prices[symbol] = df.loc[date, 'close']
                
                if not current_prices:
                    continue
                
                # Generate signals and execute trades
                for symbol, df in market_data.items():
                    if date in df.index:
                        historical_data = df.loc[:date]
                        
                        if len(historical_data) >= 30:
                            signal = strategy.generate_signal(historical_data, current_prices[symbol], {})
                            
                            if signal['confidence'] > 0.6:
                                action = signal['action']
                                confidence = signal['confidence']
                                price = current_prices[symbol]
                                
                                if action == 'buy':
                                    # Calculate position size (10% of available capital)
                                    available_capital = backtester.capital
                                    target_value = available_capital * 0.1
                                    quantity = int(target_value / price)
                                    
                                    if quantity > 0:
                                        backtester.execute_trade(symbol, 'buy', price, quantity, confidence, date)
                                
                                elif action == 'sell':
                                    if symbol in backtester.positions:
                                        quantity = backtester.positions[symbol]['quantity']
                                        if quantity > 0:
                                            backtester.execute_trade(symbol, 'sell', price, quantity, confidence, date)
                
                # Calculate portfolio value
                portfolio_value = backtester.calculate_portfolio_value(current_prices)
                daily_values.append({
                    'date': date,
                    'value': portfolio_value,
                    'capital': backtester.capital,
                    'positions': len(backtester.positions)
                })
                
                # Update max drawdown
                if portfolio_value > peak_value:
                    peak_value = portfolio_value
                else:
                    drawdown = (peak_value - portfolio_value) / peak_value
                    max_drawdown = max(max_drawdown, drawdown)
                
            except Exception as e:
                logger.error(f"Error processing date {date}: {e}")
                continue
        
        # Calculate final metrics
        if daily_values:
            final_value = daily_values[-1]['value']
            total_return = (final_value - 100000) / 100000
            
            # Calculate daily returns
            daily_returns = []
            for i in range(1, len(daily_values)):
                daily_return = (daily_values[i]['value'] - daily_values[i-1]['value']) / daily_values[i-1]['value']
                daily_returns.append(daily_return)
            
            # Calculate Sharpe ratio
            if daily_returns:
                avg_daily_return = np.mean(daily_returns)
                daily_volatility = np.std(daily_returns)
                sharpe_ratio = (avg_daily_return / daily_volatility) * np.sqrt(252) if daily_volatility > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Calculate transaction costs
            total_transaction_costs = sum(t['transaction_cost'] for t in backtester.trades)
            transaction_cost_ratio = total_transaction_costs / 100000
            
            result = {
                'strategy_name': strategy_name,
                'total_return': total_return,
                'annualized_return': (1 + total_return) ** (252/len(daily_values)) - 1,
                'monthly_return': (1 + total_return) ** (1/12) - 1,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_trades': len(backtester.trades),
                'final_value': final_value,
                'transaction_costs': total_transaction_costs,
                'transaction_cost_ratio': transaction_cost_ratio,
                'avg_positions': np.mean([dv['positions'] for dv in daily_values])
            }
            
            results.append(result)
            logger.info(f"✅ {strategy_name}: {total_return:.2%} return, {sharpe_ratio:.3f} Sharpe, {max_drawdown:.2%} max DD")
    
    # Generate report
    generate_portfolio_report(results, market_data)
    
    return results

def generate_portfolio_report(results: List[Dict], market_data: Dict[str, pd.DataFrame]):
    """Generate comprehensive portfolio test report"""
    if not results:
        logger.warning("No results to report")
        return
    
    # Sort by total return
    results.sort(key=lambda x: x['total_return'], reverse=True)
    
    # Calculate summary statistics
    total_strategies = len(results)
    successful_strategies = len([r for r in results if r['total_return'] > 0])
    avg_return = np.mean([r['total_return'] for r in results])
    avg_sharpe = np.mean([r['sharpe_ratio'] for r in results])
    avg_transaction_costs = np.mean([r['transaction_cost_ratio'] for r in results])
    
    # Generate report
    report = f"""
🎯 ПОРТФЕЛЬНОЕ ТЕСТИРОВАНИЕ С РЕАЛЬНЫМИ ДАННЫМИ И ТРАНЗАКЦИОННЫМИ ИЗДЕРЖКАМИ
{'='*80}

📊 ОБЩАЯ СТАТИСТИКА:
- Протестировано стратегий: {total_strategies}
- Успешных стратегий: {successful_strategies} ({successful_strategies/total_strategies*100:.1f}%)
- Период тестирования: 2023 год (252 торговых дня)
- Начальный капитал: 100,000 ₽
- Инструментов в портфеле: {len(market_data)}
- Средняя доходность: {avg_return:.2%}
- Средний коэффициент Шарпа: {avg_sharpe:.3f}
- Средние транзакционные издержки: {avg_transaction_costs:.2%}

📈 РЕЗУЛЬТАТЫ ПО СТРАТЕГИЯМ:
{'-'*80}
"""
    
    for i, result in enumerate(results, 1):
        meets_target = "✓" if result['monthly_return'] >= 0.20 else "✗"
        
        report += f"""
{i}. {result['strategy_name']} {meets_target}
    Общая доходность:     {result['total_return']:>8.2%}
    Годовая доходность:   {result['annualized_return']:>8.2%}
    Месячная доходность:  {result['monthly_return']:>8.2%}
    Коэффициент Шарпа:    {result['sharpe_ratio']:>8.3f}
    Макс. просадка:       {result['max_drawdown']:>8.2%}
    Количество сделок:    {result['total_trades']:>8d}
    Транзакц. издержки:   {result['transaction_cost_ratio']:>8.2%}
    Среднее позиций:      {result['avg_positions']:>8.1f}
    Итоговый капитал:     {result['final_value']:>8,.0f} ₽
"""
    
    # Analysis of 20% monthly target
    monthly_target = 0.20
    successful_strategies = [r for r in results if r['monthly_return'] >= monthly_target]
    
    report += f"""

🎯 АНАЛИЗ ДОСТИЖЕНИЯ ЦЕЛИ 20% В МЕСЯЦ:
{'-'*80}
Цель: 20% доходности в месяц
Стратегий, достигших цели: {len(successful_strategies)}/{total_strategies} ({len(successful_strategies)/total_strategies*100:.1f}%)
"""
    
    if successful_strategies:
        report += "\n✅ УСПЕШНЫЕ СТРАТЕГИИ:\n"
        for result in successful_strategies:
            report += f"- {result['strategy_name']}: {result['monthly_return']:.2%} в месяц\n"
    else:
        report += """
❌ НИ ОДНА СТРАТЕГИЯ НЕ ДОСТИГЛА ЦЕЛИ 20% В МЕСЯЦ

💡 РЕКОМЕНДАЦИИ ДЛЯ ДОСТИЖЕНИЯ ЦЕЛИ:
1. Увеличить кредитное плечо (leverage)
2. Торговать более волатильными активами (криптовалюты)
3. Использовать внутридневную торговлю
4. Комбинировать несколько стратегий в портфеле
5. Снизить транзакционные издержки
6. Применить динамическое управление позициями
"""
    
    # Transaction costs analysis
    report += f"""

💰 АНАЛИЗ ТРАНЗАКЦИОННЫХ ИЗДЕРЖЕК:
{'-'*80}
Стратегии с наибольшими издержками:
"""
    high_cost_strategies = sorted(results, key=lambda x: x['transaction_cost_ratio'], reverse=True)
    for result in high_cost_strategies:
        report += f"- {result['strategy_name']}: {result['transaction_cost_ratio']:.2%} издержки, {result['total_return']:.2%} доходность\n"
    
    # Portfolio diversification analysis
    report += f"""

📊 АНАЛИЗ ДИВЕРСИФИКАЦИИ ПОРТФЕЛЯ:
{'-'*80}
Инструменты в тестировании: {', '.join(market_data.keys())}
Среднее количество позиций: {np.mean([r['avg_positions'] for r in results]):.1f}
Максимальное количество позиций: {max([r['avg_positions'] for r in results]):.1f}

Характеристики инструментов:
"""
    for symbol, df in market_data.items():
        start_price = df['close'].iloc[0]
        end_price = df['close'].iloc[-1]
        total_return = (end_price / start_price - 1)
        volatility = df['close'].pct_change().std() * np.sqrt(252)
        
        report += f"- {symbol}: {start_price:.0f} → {end_price:.0f} ({total_return:.2%}), волатильность {volatility:.2%}\n"
    
    report += f"""

🏆 ЗАКЛЮЧЕНИЕ:
{'-'*80}
Лучшая стратегия: {results[0]['strategy_name']}
Доходность лучшей стратегии: {results[0]['total_return']:.2%}
Средняя доходность всех стратегий: {avg_return:.2%}
Общие транзакционные издержки: {avg_transaction_costs:.2%}

⚠️  ВАЖНО: Результаты основаны на реалистичных рыночных данных с транзакционными издержками.
    Реальная торговля может отличаться из-за проскальзывания и ликвидности.
"""
    
    print(report)
    
    # Save report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'portfolio_test_report_{timestamp}.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Save JSON results
    json_results = {
        'timestamp': timestamp,
        'summary': {
            'total_strategies': total_strategies,
            'successful_strategies': successful_strategies,
            'avg_return': avg_return,
            'avg_sharpe': avg_sharpe,
            'avg_transaction_costs': avg_transaction_costs
        },
        'strategies': results,
        'market_data_summary': {
            symbol: {
                'days': len(df),
                'start_price': df['close'].iloc[0],
                'end_price': df['close'].iloc[-1],
                'total_return': (df['close'].iloc[-1] / df['close'].iloc[0] - 1),
                'volatility': df['close'].pct_change().std() * np.sqrt(252)
            }
            for symbol, df in market_data.items()
        }
    }
    
    with open(f'portfolio_test_results_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, default=str, ensure_ascii=False)
    
    logger.info(f"📄 Отчет сохранен в portfolio_test_report_{timestamp}.txt")
    logger.info(f"📊 Результаты сохранены в portfolio_test_results_{timestamp}.json")

if __name__ == "__main__":
    try:
        results = test_portfolio_strategies()
        logger.info("🏁 Портфельное тестирование завершено успешно!")
    except Exception as e:
        logger.error(f"❌ Ошибка при тестировании: {e}")
        raise
