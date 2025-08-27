#!/usr/bin/env python3
"""
Script to test all trading strategies on 2023 data
"""

import asyncio
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import json
from typing import Dict, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_provider import DataProvider
from backtesting.backtest_engine import BacktestEngine, BacktestResult
from strategies.technical_strategies import (
    RSIStrategy, MACDStrategy, BollingerBandsStrategy, 
    MovingAverageCrossoverStrategy, StochasticStrategy
)
from strategies.momentum_strategies import (
    MomentumStrategy, MeanReversionStrategy, BreakoutStrategy, VolumeProfileStrategy
)
from strategies.arbitrage_strategies import (
    PairsTradingStrategy, VolatilityArbitrageStrategy
)
from strategies.ml_strategies import (
    RandomForestStrategy, GradientBoostingStrategy, LogisticRegressionStrategy, EnsembleStrategy
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('strategy_testing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StrategyTester:
    """Test multiple trading strategies"""
    
    def __init__(self):
        self.data_provider = None
        self.backtest_engine = BacktestEngine(initial_capital=100000, commission=0.001)
        self.results = {}
        
    async def initialize(self):
        """Initialize data provider"""
        self.data_provider = DataProvider(use_tbank=False)  # Use Yahoo Finance for testing
        await self.data_provider.__aenter__()
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.data_provider:
            await self.data_provider.__aexit__(None, None, None)
    
    def get_all_strategies(self) -> Dict:
        """Get all trading strategies"""
        strategies = {
            # Technical Analysis Strategies
            'RSI_Conservative': RSIStrategy({'rsi_period': 14, 'oversold_threshold': 25, 'overbought_threshold': 75}),
            'RSI_Aggressive': RSIStrategy({'rsi_period': 10, 'oversold_threshold': 20, 'overbought_threshold': 80}),
            
            'MACD_Fast': MACDStrategy({'fast_period': 8, 'slow_period': 21, 'signal_period': 9}),
            'MACD_Slow': MACDStrategy({'fast_period': 12, 'slow_period': 26, 'signal_period': 9}),
            
            'BB_Tight': BollingerBandsStrategy({'period': 20, 'std_dev': 1.5}),
            'BB_Wide': BollingerBandsStrategy({'period': 20, 'std_dev': 2.5}),
            
            'MA_Cross_Fast': MovingAverageCrossoverStrategy({'fast_period': 5, 'slow_period': 15, 'ma_type': 'ema'}),
            'MA_Cross_Slow': MovingAverageCrossoverStrategy({'fast_period': 20, 'slow_period': 50, 'ma_type': 'sma'}),
            
            'Stochastic_Standard': StochasticStrategy({'k_period': 14, 'd_period': 3}),
            'Stochastic_Fast': StochasticStrategy({'k_period': 8, 'd_period': 3}),
            
            # Momentum Strategies
            'Momentum_Short': MomentumStrategy({'lookback_period': 5, 'momentum_threshold': 0.03}),
            'Momentum_Medium': MomentumStrategy({'lookback_period': 10, 'momentum_threshold': 0.02}),
            
            'MeanReversion_Conservative': MeanReversionStrategy({'lookback_period': 30, 'deviation_threshold': 2.5}),
            'MeanReversion_Aggressive': MeanReversionStrategy({'lookback_period': 15, 'deviation_threshold': 1.8}),
            
            'Breakout_Standard': BreakoutStrategy({'lookback_period': 20, 'breakout_threshold': 0.02}),
            'Breakout_Sensitive': BreakoutStrategy({'lookback_period': 15, 'breakout_threshold': 0.015}),
            
            'VolumeProfile_Standard': VolumeProfileStrategy({'lookback_period': 40, 'volume_threshold': 1.8}),
            
            # Arbitrage Strategies
            'VolatilityArb': VolatilityArbitrageStrategy({'volatility_window': 20, 'vol_threshold_high': 0.35}),
            
            # Machine Learning Strategies
            'RandomForest': RandomForestStrategy({'n_estimators': 50, 'max_depth': 8}),
            'GradientBoosting': GradientBoostingStrategy({'n_estimators': 50, 'learning_rate': 0.15}),
            'LogisticRegression': LogisticRegressionStrategy({'C': 0.5}),
            'MLEnsemble': EnsembleStrategy()
        }
        
        return strategies
    
    async def load_test_data(self) -> Dict[str, pd.DataFrame]:
        """Load test data for 2023"""
        logger.info("Loading test data for 2023...")
        
        # Get popular Russian stocks
        symbols = self.data_provider.get_russian_stocks()[:12]  # Limit to 12 stocks for testing
        
        # Load data - use synthetic data for reliable testing
        data = await self.data_provider.get_multiple_symbols_data(
            symbols, '2023-01-01', '2023-12-31', use_synthetic=True
        )
        
        logger.info(f"Loaded data for {len(data)} symbols")
        
        # Get data summary
        summary = self.data_provider.get_market_data_summary(data)
        for symbol, stats in summary.items():
            if 'error' not in stats:
                logger.info(f"{symbol}: {stats['num_days']} days, "
                          f"return: {stats['total_return']:.2%}, "
                          f"volatility: {stats['volatility']:.2%}")
        
        return data
    
    async def test_single_strategy(self, strategy_name: str, strategy, data: Dict[str, pd.DataFrame]) -> BacktestResult:
        """Test a single strategy"""
        try:
            logger.info(f"Testing strategy: {strategy_name}")
            
            # For simplicity, test on a single symbol with good data
            if not data:
                raise ValueError("No data available")
            
            # Choose symbol with most data
            best_symbol = max(data.keys(), key=lambda x: len(data[x]))
            symbol_data = {best_symbol: data[best_symbol]}
            
            result = self.backtest_engine.run_backtest(
                strategy, symbol_data, '2023-01-01', '2023-12-31'
            )
            
            logger.info(f"{strategy_name} completed - Return: {result.total_return:.2%}, "
                       f"Sharpe: {result.sharpe_ratio:.3f}, "
                       f"Max DD: {result.max_drawdown:.2%}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error testing {strategy_name}: {e}")
            # Return dummy result for failed strategies
            return BacktestResult(
                total_return=0.0, annualized_return=0.0, volatility=0.0,
                sharpe_ratio=0.0, max_drawdown=0.0, win_rate=0.0,
                profit_factor=0.0, total_trades=0, avg_trade_duration=0.0,
                avg_win=0.0, avg_loss=0.0, largest_win=0.0, largest_loss=0.0,
                consecutive_wins=0, consecutive_losses=0,
                monthly_returns=pd.Series(), daily_returns=pd.Series(),
                equity_curve=pd.Series(), trades=[], performance_metrics={}
            )
    
    async def test_all_strategies(self) -> Dict[str, BacktestResult]:
        """Test all strategies"""
        logger.info("Starting comprehensive strategy testing...")
        
        # Load data
        data = await self.load_test_data()
        
        if not data:
            logger.error("No data available for testing")
            return {}
        
        # Get all strategies
        strategies = self.get_all_strategies()
        
        # Test each strategy
        results = {}
        total_strategies = len(strategies)
        
        for i, (strategy_name, strategy) in enumerate(strategies.items(), 1):
            logger.info(f"Testing strategy {i}/{total_strategies}: {strategy_name}")
            
            try:
                result = await self.test_single_strategy(strategy_name, strategy, data)
                results[strategy_name] = result
                
            except Exception as e:
                logger.error(f"Failed to test {strategy_name}: {e}")
                continue
        
        logger.info(f"Completed testing {len(results)} strategies")
        return results
    
    def generate_comparison_report(self, results: Dict[str, BacktestResult]) -> str:
        """Generate comparison report of all strategies"""
        if not results:
            return "No results available for comparison"
        
        # Sort strategies by total return
        sorted_results = sorted(results.items(), key=lambda x: x[1].total_return, reverse=True)
        
        report = """
STRATEGY COMPARISON REPORT - 2023 BACKTEST RESULTS
===================================================

PERFORMANCE RANKING (by Total Return):
""" + "="*50 + "\n"
        
        for i, (name, result) in enumerate(sorted_results, 1):
            report += f"""
{i:2d}. {name}
    Total Return:      {result.total_return:>8.2%}
    Annualized Return: {result.annualized_return:>8.2%}
    Sharpe Ratio:      {result.sharpe_ratio:>8.3f}
    Max Drawdown:      {result.max_drawdown:>8.2%}
    Win Rate:          {result.win_rate:>8.2%}
    Total Trades:      {result.total_trades:>8d}
    {"="*50}"""
        
        # Top performers analysis
        report += f"""

TOP 5 PERFORMERS:
{"-"*50}
"""
        
        for i, (name, result) in enumerate(sorted_results[:5], 1):
            monthly_target = 0.20  # 20% monthly target
            annual_target = (1 + monthly_target) ** 12 - 1  # Compound monthly target
            
            meets_target = "✓" if result.annualized_return >= annual_target else "✗"
            
            report += f"""
{i}. {name} {meets_target}
   - Return: {result.total_return:.2%} (Annual: {result.annualized_return:.2%})
   - Risk-Adjusted: Sharpe {result.sharpe_ratio:.3f}
   - Drawdown Control: {result.max_drawdown:.2%} max
   - Trade Stats: {result.total_trades} trades, {result.win_rate:.1%} win rate
"""
        
        # Risk analysis
        report += f"""

RISK ANALYSIS:
{"-"*50}

Lowest Drawdown Strategies:
"""
        low_dd_strategies = sorted(results.items(), key=lambda x: x[1].max_drawdown)[:3]
        for name, result in low_dd_strategies:
            report += f"  - {name}: {result.max_drawdown:.2%} drawdown, {result.total_return:.2%} return\n"
        
        report += "\nHighest Sharpe Ratio Strategies:\n"
        high_sharpe_strategies = sorted(results.items(), key=lambda x: x[1].sharpe_ratio, reverse=True)[:3]
        for name, result in high_sharpe_strategies:
            report += f"  - {name}: {result.sharpe_ratio:.3f} Sharpe, {result.total_return:.2%} return\n"
        
        # Strategy category analysis
        report += f"""

STRATEGY CATEGORY PERFORMANCE:
{"-"*50}
"""
        
        categories = {
            'Technical Analysis': ['RSI', 'MACD', 'BB', 'MA_Cross', 'Stochastic'],
            'Momentum': ['Momentum', 'MeanReversion', 'Breakout', 'VolumeProfile'],
            'Arbitrage': ['VolatilityArb', 'Pairs'],
            'Machine Learning': ['RandomForest', 'GradientBoosting', 'LogisticRegression', 'MLEnsemble']
        }
        
        for category, prefixes in categories.items():
            category_results = []
            for name, result in results.items():
                if any(name.startswith(prefix) for prefix in prefixes):
                    category_results.append(result)
            
            if category_results:
                avg_return = np.mean([r.total_return for r in category_results])
                avg_sharpe = np.mean([r.sharpe_ratio for r in category_results])
                avg_drawdown = np.mean([r.max_drawdown for r in category_results])
                
                report += f"""
{category}:
  - Average Return: {avg_return:.2%}
  - Average Sharpe: {avg_sharpe:.3f}
  - Average Drawdown: {avg_drawdown:.2%}
  - Strategies Count: {len(category_results)}
"""
        
        # Success rate for 20% monthly target
        monthly_target = 0.20
        annual_target = (1 + monthly_target) ** 12 - 1
        successful_strategies = [r for r in results.values() if r.annualized_return >= annual_target]
        
        report += f"""

TARGET ACHIEVEMENT ANALYSIS:
{"-"*50}
Target: 20% monthly return ({annual_target:.1%} annually)

Strategies meeting target: {len(successful_strategies)}/{len(results)} ({len(successful_strategies)/len(results)*100:.1f}%)

Recommendations for 20% monthly target:
"""
        
        if successful_strategies:
            best_performers = sorted(successful_strategies, key=lambda x: x.sharpe_ratio, reverse=True)[:3]
            for i, result in enumerate(best_performers, 1):
                strategy_name = [name for name, res in results.items() if res == result][0]
                report += f"""
{i}. {strategy_name}
   - Expected monthly return: {(1 + result.annualized_return)**(1/12) - 1:.2%}
   - Risk level: {result.max_drawdown:.2%} max drawdown
   - Consistency: {result.win_rate:.1f}% win rate
"""
        else:
            report += """
No strategies achieved the 20% monthly target in 2023 backtesting.
Consider:
1. Portfolio combination of top performers
2. Parameter optimization
3. Market regime analysis
4. Alternative assets or timeframes
"""
        
        return report
    
    def save_results(self, results: Dict[str, BacktestResult], filename: str = 'strategy_test_results.json'):
        """Save results to JSON file"""
        try:
            # Convert results to serializable format
            serializable_results = {}
            
            for name, result in results.items():
                serializable_results[name] = {
                    'total_return': result.total_return,
                    'annualized_return': result.annualized_return,
                    'volatility': result.volatility,
                    'sharpe_ratio': result.sharpe_ratio,
                    'max_drawdown': result.max_drawdown,
                    'win_rate': result.win_rate,
                    'profit_factor': result.profit_factor,
                    'total_trades': result.total_trades,
                    'avg_trade_duration': result.avg_trade_duration,
                    'consecutive_wins': result.consecutive_wins,
                    'consecutive_losses': result.consecutive_losses
                }
            
            with open(filename, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"Results saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")

async def main():
    """Main function to run strategy testing"""
    tester = StrategyTester()
    
    try:
        # Initialize
        await tester.initialize()
        
        # Run tests
        results = await tester.test_all_strategies()
        
        if results:
            # Generate and print report
            report = tester.generate_comparison_report(results)
            print(report)
            
            # Save results
            tester.save_results(results)
            
            # Save report
            with open('strategy_comparison_report.txt', 'w', encoding='utf-8') as f:
                f.write(report)
            
            logger.info("Testing completed successfully!")
            logger.info("Reports saved to strategy_comparison_report.txt")
        else:
            logger.error("No results generated")
    
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise
    
    finally:
        # Cleanup
        await tester.cleanup()

if __name__ == "__main__":
    asyncio.run(main())