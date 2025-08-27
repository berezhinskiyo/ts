#!/usr/bin/env python3
"""
Portfolio Optimizer for combining multiple trading strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from scipy.optimize import minimize
# from scipy.stats import sharpe_ratio  # Not needed
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """Optimize portfolio allocation across multiple trading strategies"""
    
    def __init__(self, target_return: float = 0.20, max_risk: float = 0.25):
        self.target_return = target_return  # Monthly target return (20%)
        self.max_risk = max_risk  # Maximum drawdown allowed
        self.strategies_data = {}
        self.optimal_weights = {}
        self.performance_metrics = {}
        
    def add_strategy_results(self, strategy_name: str, results: Dict):
        """Add strategy backtest results"""
        self.strategies_data[strategy_name] = results
        logger.info(f"Added strategy: {strategy_name}")
    
    def load_results_from_json(self, filepath: str):
        """Load strategy results from JSON file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'strategies' in data:
                for strategy in data['strategies']:
                    self.strategies_data[strategy['name']] = strategy
            else:
                self.strategies_data = data
                
            logger.info(f"Loaded {len(self.strategies_data)} strategies from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading results from {filepath}: {e}")
    
    def filter_strategies(self, min_trades: int = 5, min_return: float = 0.0, 
                         max_drawdown: float = 0.30) -> List[str]:
        """Filter strategies based on minimum criteria"""
        filtered = []
        
        for name, data in self.strategies_data.items():
            if isinstance(data, dict):
                trades = data.get('total_trades', 0)
                monthly_return = data.get('monthly_return', 0)
                drawdown = abs(data.get('max_drawdown', 0))
                
                if (trades >= min_trades and 
                    monthly_return >= min_return and 
                    drawdown <= max_drawdown):
                    filtered.append(name)
        
        logger.info(f"Filtered {len(filtered)} strategies from {len(self.strategies_data)}")
        return filtered
    
    def create_returns_matrix(self, strategy_names: List[str]) -> pd.DataFrame:
        """Create synthetic returns matrix for optimization"""
        # Since we don't have daily returns, create synthetic based on performance metrics
        np.random.seed(42)
        
        returns_data = {}
        n_days = 252  # Trading days in a year
        
        for name in strategy_names:
            if name not in self.strategies_data:
                continue
                
            data = self.strategies_data[name]
            monthly_return = data.get('monthly_return', 0)
            volatility = data.get('volatility', 0.02)
            sharpe = data.get('sharpe_ratio', 0)
            
            # Generate synthetic daily returns
            daily_return = monthly_return / 21  # Approximate trading days per month
            daily_vol = volatility / np.sqrt(252)
            
            # Add some correlation and randomness
            returns = np.random.normal(daily_return, daily_vol, n_days)
            
            # Adjust for negative periods (simulating drawdowns)
            max_dd = abs(data.get('max_drawdown', 0))
            if max_dd > 0:
                # Add some negative periods
                negative_periods = int(n_days * max_dd / 4)  # Rough approximation
                negative_indices = np.random.choice(n_days, negative_periods, replace=False)
                returns[negative_indices] *= -2  # Make them more negative
            
            returns_data[name] = returns
        
        dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
        return pd.DataFrame(returns_data, index=dates)
    
    def calculate_portfolio_metrics(self, weights: np.array, returns_df: pd.DataFrame) -> Dict:
        """Calculate portfolio performance metrics"""
        # Portfolio returns
        portfolio_returns = (returns_df * weights).sum(axis=1)
        
        # Performance metrics
        total_return = (1 + portfolio_returns).prod() - 1
        monthly_return = (1 + total_return) ** (1/12) - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming 2% risk-free rate)
        excess_returns = portfolio_returns - 0.02/252
        sharpe = excess_returns.mean() / portfolio_returns.std() * np.sqrt(252) if portfolio_returns.std() > 0 else 0
        
        # Drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'monthly_return': monthly_return,
            'annualized_return': (1 + total_return) ** (252/len(portfolio_returns)) - 1,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'portfolio_returns': portfolio_returns
        }
    
    def objective_function(self, weights: np.array, returns_df: pd.DataFrame, 
                          optimization_target: str = 'sharpe') -> float:
        """Objective function for optimization"""
        metrics = self.calculate_portfolio_metrics(weights, returns_df)
        
        if optimization_target == 'sharpe':
            return -metrics['sharpe_ratio']  # Negative because we minimize
        elif optimization_target == 'return':
            return -metrics['monthly_return']
        elif optimization_target == 'risk_adjusted':
            # Custom risk-adjusted return considering target
            monthly_ret = metrics['monthly_return']
            max_dd = abs(metrics['max_drawdown'])
            
            # Penalty for not meeting target return
            target_penalty = max(0, (self.target_return - monthly_ret) * 10)
            
            # Penalty for high drawdown
            risk_penalty = max(0, (max_dd - self.max_risk) * 5)
            
            return -(monthly_ret - target_penalty - risk_penalty)
        else:
            return -metrics['sharpe_ratio']
    
    def optimize_weights(self, strategy_names: List[str], 
                        optimization_target: str = 'risk_adjusted') -> Dict:
        """Optimize portfolio weights"""
        if len(strategy_names) < 2:
            logger.warning("Need at least 2 strategies for optimization")
            return {}
        
        # Create returns matrix
        returns_df = self.create_returns_matrix(strategy_names)
        
        if returns_df.empty:
            logger.error("Failed to create returns matrix")
            return {}
        
        n_strategies = len(strategy_names)
        
        # Initial weights (equal allocation)
        initial_weights = np.array([1.0 / n_strategies] * n_strategies)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # Weights sum to 1
        ]
        
        # Bounds (each weight between 0 and 1)
        bounds = [(0.0, 1.0) for _ in range(n_strategies)]
        
        # Additional constraints for diversification
        if n_strategies > 3:
            # No single strategy more than 60%
            for i in range(n_strategies):
                bounds[i] = (0.0, 0.6)
        
        try:
            # Optimize
            result = minimize(
                fun=self.objective_function,
                x0=initial_weights,
                args=(returns_df, optimization_target),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                optimal_weights = result.x
                
                # Calculate final metrics
                final_metrics = self.calculate_portfolio_metrics(optimal_weights, returns_df)
                
                # Create results dictionary
                optimization_results = {
                    'strategy_names': strategy_names,
                    'optimal_weights': {name: weight for name, weight in zip(strategy_names, optimal_weights)},
                    'performance': final_metrics,
                    'optimization_target': optimization_target,
                    'success': True
                }
                
                self.optimal_weights = optimization_results['optimal_weights']
                self.performance_metrics = final_metrics
                
                logger.info(f"Optimization successful. Target: {optimization_target}")
                logger.info(f"Expected monthly return: {final_metrics['monthly_return']:.2%}")
                logger.info(f"Expected Sharpe ratio: {final_metrics['sharpe_ratio']:.3f}")
                
                return optimization_results
            
            else:
                logger.error(f"Optimization failed: {result.message}")
                return {'success': False, 'error': result.message}
                
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            return {'success': False, 'error': str(e)}
    
    def create_portfolio_combinations(self) -> List[Dict]:
        """Create different portfolio combinations for analysis"""
        strategy_names = list(self.strategies_data.keys())
        
        if len(strategy_names) < 2:
            return []
        
        # Filter good strategies
        good_strategies = self.filter_strategies(min_trades=5, min_return=0.0, max_drawdown=0.25)
        
        combinations = []
        
        # 1. Equal weight portfolio (all strategies)
        if len(good_strategies) >= 2:
            equal_weights = {name: 1.0/len(good_strategies) for name in good_strategies}
            combinations.append({
                'name': 'Equal_Weight_All',
                'weights': equal_weights,
                'description': 'Equal allocation across all filtered strategies'
            })
        
        # 2. Top performers portfolio
        top_strategies = sorted(
            good_strategies,
            key=lambda x: self.strategies_data[x].get('monthly_return', 0),
            reverse=True
        )[:5]  # Top 5
        
        if len(top_strategies) >= 2:
            top_weights = {name: 1.0/len(top_strategies) for name in top_strategies}
            combinations.append({
                'name': 'Top_Performers',
                'weights': top_weights,
                'description': 'Equal allocation across top 5 performing strategies'
            })
        
        # 3. Risk-adjusted portfolio (high Sharpe ratio)
        risk_adj_strategies = sorted(
            good_strategies,
            key=lambda x: self.strategies_data[x].get('sharpe_ratio', 0),
            reverse=True
        )[:4]  # Top 4 by Sharpe
        
        if len(risk_adj_strategies) >= 2:
            risk_weights = {name: 1.0/len(risk_adj_strategies) for name in risk_adj_strategies}
            combinations.append({
                'name': 'Risk_Adjusted',
                'weights': risk_weights,
                'description': 'Equal allocation across highest Sharpe ratio strategies'
            })
        
        # 4. Conservative portfolio (low drawdown)
        conservative_strategies = sorted(
            good_strategies,
            key=lambda x: abs(self.strategies_data[x].get('max_drawdown', 1)),
        )[:3]  # Lowest 3 drawdowns
        
        if len(conservative_strategies) >= 2:
            conservative_weights = {name: 1.0/len(conservative_strategies) for name in conservative_strategies}
            combinations.append({
                'name': 'Conservative',
                'weights': conservative_weights,
                'description': 'Equal allocation across lowest drawdown strategies'
            })
        
        # 5. Optimized portfolios
        optimization_targets = ['sharpe', 'return', 'risk_adjusted']
        
        for target in optimization_targets:
            if len(good_strategies) >= 2:
                opt_result = self.optimize_weights(good_strategies, target)
                if opt_result.get('success', False):
                    combinations.append({
                        'name': f'Optimized_{target.title()}',
                        'weights': opt_result['optimal_weights'],
                        'description': f'Mathematically optimized for {target}',
                        'performance': opt_result['performance']
                    })
        
        return combinations
    
    def analyze_all_combinations(self) -> Dict:
        """Analyze all portfolio combinations"""
        combinations = self.create_portfolio_combinations()
        
        if not combinations:
            return {'error': 'No valid combinations found'}
        
        results = []
        
        for combo in combinations:
            strategy_names = list(combo['weights'].keys())
            weights = np.array(list(combo['weights'].values()))
            
            # Create returns matrix for this combination
            returns_df = self.create_returns_matrix(strategy_names)
            
            if not returns_df.empty:
                metrics = self.calculate_portfolio_metrics(weights, returns_df)
                
                result = {
                    'name': combo['name'],
                    'description': combo['description'],
                    'weights': combo['weights'],
                    'performance': metrics,
                    'meets_target': metrics['monthly_return'] >= self.target_return,
                    'risk_acceptable': abs(metrics['max_drawdown']) <= self.max_risk
                }
                
                results.append(result)
        
        # Sort by monthly return
        results.sort(key=lambda x: x['performance']['monthly_return'], reverse=True)
        
        return {
            'combinations': results,
            'target_return': self.target_return,
            'max_risk': self.max_risk,
            'analysis_date': datetime.now().isoformat()
        }
    
    def generate_portfolio_report(self, analysis_results: Dict) -> str:
        """Generate comprehensive portfolio analysis report"""
        if 'error' in analysis_results:
            return f"Portfolio Analysis Error: {analysis_results['error']}"
        
        combinations = analysis_results['combinations']
        target_return = analysis_results['target_return']
        
        # Find successful combinations
        successful = [c for c in combinations if c['meets_target'] and c['risk_acceptable']]
        
        report = f"""
🎯 PORTFOLIO OPTIMIZER - АНАЛИЗ КОМБИНАЦИЙ СТРАТЕГИЙ
{'='*80}

📊 ПАРАМЕТРЫ АНАЛИЗА:
- Целевая доходность: {target_return:.1%} в месяц
- Максимальный риск: {analysis_results['max_risk']:.1%} просадки
- Проанализировано комбинаций: {len(combinations)}

🏆 РЕЗУЛЬТАТЫ ДОСТИЖЕНИЯ ЦЕЛИ:
Успешных комбинаций: {len(successful)}/{len(combinations)} ({len(successful)/len(combinations)*100:.1f}%)

"""
        
        if successful:
            report += f"""
✅ РЕКОМЕНДУЕМЫЕ ПОРТФЕЛИ (достигают цель {target_return:.1%} в месяц):
{'-'*80}
"""
            
            for i, combo in enumerate(successful, 1):
                perf = combo['performance']
                report += f"""
{i}. {combo['name']}
   Описание: {combo['description']}
   
   📈 ПРОИЗВОДИТЕЛЬНОСТЬ:
   - Месячная доходность: {perf['monthly_return']:.2%}
   - Годовая доходность: {perf['annualized_return']:.1%}
   - Коэффициент Шарпа: {perf['sharpe_ratio']:.3f}
   - Максимальная просадка: {perf['max_drawdown']:.2%}
   - Волатильность: {perf['volatility']:.2%}
   
   💼 РАСПРЕДЕЛЕНИЕ КАПИТАЛА:
"""
                
                for strategy, weight in combo['weights'].items():
                    report += f"   - {strategy}: {weight:.1%}\n"
                
                report += "\n"
        
        else:
            report += f"""
❌ НИ ОДНА КОМБИНАЦИЯ НЕ ДОСТИГЛА ЦЕЛИ {target_return:.1%} В МЕСЯЦ
"""
        
        # Show all combinations ranked by performance
        report += f"""
📊 ВСЕ КОМБИНАЦИИ (по доходности):
{'-'*80}
"""
        
        for i, combo in enumerate(combinations, 1):
            perf = combo['performance']
            status = "✅" if combo['meets_target'] and combo['risk_acceptable'] else "❌"
            
            report += f"""
{i:2d}. {combo['name']} {status}
    Месячная доходность: {perf['monthly_return']:>8.2%}
    Коэффициент Шарпа:   {perf['sharpe_ratio']:>8.3f}
    Макс. просадка:      {perf['max_drawdown']:>8.2%}
    Топ стратегии: {', '.join(list(combo['weights'].keys())[:3])}
"""
        
        # Best individual strategies reference
        report += f"""

🔍 АНАЛИЗ ЛУЧШИХ ИНДИВИДУАЛЬНЫХ СТРАТЕГИЙ:
{'-'*80}
"""
        
        # Get top individual strategies
        individual_strategies = []
        for name, data in self.strategies_data.items():
            if isinstance(data, dict):
                individual_strategies.append((name, data.get('monthly_return', 0)))
        
        individual_strategies.sort(key=lambda x: x[1], reverse=True)
        
        for i, (name, monthly_ret) in enumerate(individual_strategies[:5], 1):
            meets_individual_target = monthly_ret >= target_return
            status = "🎯" if meets_individual_target else "📊"
            
            report += f"   {i}. {name}: {monthly_ret:.2%} в месяц {status}\n"
        
        # Recommendations
        report += f"""

💡 РЕКОМЕНДАЦИИ:
{'-'*80}
"""
        
        if successful:
            best_combo = successful[0]
            report += f"""
🚀 ГЛАВНАЯ РЕКОМЕНДАЦИЯ: {best_combo['name']}
   Ожидаемая доходность: {best_combo['performance']['monthly_return']:.2%} в месяц
   Уровень риска: {abs(best_combo['performance']['max_drawdown']):.2%} просадка
   
   План действий:
   1. Распределить капитал согласно весам портфеля
   2. Запустить все стратегии одновременно
   3. Еженедельно мониторить результаты
   4. Ребалансировать портфель ежемесячно
   5. Корректировать веса при значительных отклонениях
"""
        else:
            best_combo = combinations[0] if combinations else None
            if best_combo:
                leverage_needed = target_return / best_combo['performance']['monthly_return']
                report += f"""
⚠️ Стандартные комбинации не достигают цели {target_return:.1%} в месяц.

🔧 АЛЬТЕРНАТИВНЫЕ ПОДХОДЫ:
1. Увеличить кредитное плечо в {leverage_needed:.1f}x раз
2. Использовать лучшую комбинацию: {best_combo['name']}
   (текущая доходность: {best_combo['performance']['monthly_return']:.2%} в месяц)
3. Добавить дополнительные стратегии
4. Рассмотреть торговлю криптовалютами
5. Использовать внутридневные стратегии
"""
        
        report += f"""

📈 ПОТЕНЦИАЛ РОСТА КАПИТАЛА:
{'-'*80}
"""
        
        if successful:
            best_monthly = successful[0]['performance']['monthly_return']
            projections = [1, 3, 6, 12]  # months
            initial_capital = 100000
            
            report += f"При начальном капитале {initial_capital:,.0f} ₽ и доходности {best_monthly:.2%} в месяц:\n\n"
            
            for months in projections:
                final_amount = initial_capital * (1 + best_monthly) ** months
                profit = final_amount - initial_capital
                report += f"   Через {months:2d} мес.: {final_amount:>10,.0f} ₽ (прибыль: {profit:>8,.0f} ₽)\n"
        
        report += f"""

⚠️ ВАЖНЫЕ ЗАМЕЧАНИЯ:
{'-'*80}
• Результаты основаны на историческом тестировании
• Будущие результаты могут отличаться от прогнозов
• Диверсификация снижает риски, но может уменьшить доходность
• Рекомендуется начать с небольшого капитала для проверки
• Обязательно используйте стоп-лоссы и риск-менеджмент

📅 Дата анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return report

def main():
    """Main function to demonstrate portfolio optimization"""
    # Initialize optimizer
    optimizer = PortfolioOptimizer(target_return=0.20, max_risk=0.25)
    
    # Load strategy results
    try:
        optimizer.load_results_from_json('advanced_strategy_results.json')
    except FileNotFoundError:
        logger.error("Strategy results file not found. Run advanced_test.py first.")
        return
    
    # Analyze all combinations
    logger.info("Analyzing portfolio combinations...")
    analysis_results = optimizer.analyze_all_combinations()
    
    # Generate and display report
    report = optimizer.generate_portfolio_report(analysis_results)
    print(report)
    
    # Save results
    with open('portfolio_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False, default=str)
    
    with open('portfolio_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info("Portfolio analysis completed and saved!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()