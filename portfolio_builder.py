#!/usr/bin/env python3
"""
Advanced Portfolio Builder for Risk-Based Project Allocation
Builds portfolios of trading strategies/projects based on risk tolerance and investment volume
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from scipy.optimize import minimize
from scipy.stats import norm
import json
from datetime import datetime, timedelta
import warnings
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk tolerance levels"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    VERY_AGGRESSIVE = "very_aggressive"

class ProjectType(Enum):
    """Types of projects/strategies"""
    ARBITRAGE = "arbitrage"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    ML_PREDICTIVE = "ml_predictive"
    TECHNICAL = "technical"
    HEDGE = "hedge"
    SCALPING = "scalping"

@dataclass
class ProjectProfile:
    """Profile for a trading project/strategy"""
    name: str
    project_type: ProjectType
    expected_return: float  # Annual return
    volatility: float  # Annual volatility
    max_drawdown: float
    correlation_matrix: Optional[pd.DataFrame] = None
    min_investment: float = 1000
    max_investment: float = 1000000
    liquidity_score: float = 1.0  # 0-1, higher = more liquid
    complexity_score: float = 0.5  # 0-1, higher = more complex
    success_rate: float = 0.6  # Historical success rate
    
    def __post_init__(self):
        if self.correlation_matrix is None:
            self.correlation_matrix = pd.DataFrame()

@dataclass
class PortfolioConstraints:
    """Portfolio construction constraints"""
    total_capital: float
    risk_level: RiskLevel
    max_projects: int = 10
    min_projects: int = 3
    max_single_allocation: float = 0.4  # 40% max per project
    min_single_allocation: float = 0.05  # 5% min per project
    target_return: Optional[float] = None
    max_volatility: Optional[float] = None
    max_drawdown: Optional[float] = None
    liquidity_requirement: float = 0.7  # Minimum average liquidity
    complexity_limit: float = 0.8  # Maximum average complexity

class RiskProfileManager:
    """Manages risk profiles for different risk tolerance levels"""
    
    RISK_PROFILES = {
        RiskLevel.CONSERVATIVE: {
            'max_volatility': 0.15,
            'max_drawdown': 0.10,
            'target_return': 0.08,
            'risk_free_rate': 0.03,
            'utility_aversion': 2.0
        },
        RiskLevel.MODERATE: {
            'max_volatility': 0.25,
            'max_drawdown': 0.20,
            'target_return': 0.12,
            'risk_free_rate': 0.03,
            'utility_aversion': 1.5
        },
        RiskLevel.AGGRESSIVE: {
            'max_volatility': 0.40,
            'max_drawdown': 0.30,
            'target_return': 0.18,
            'risk_free_rate': 0.03,
            'utility_aversion': 1.0
        },
        RiskLevel.VERY_AGGRESSIVE: {
            'max_volatility': 0.60,
            'max_drawdown': 0.45,
            'target_return': 0.25,
            'risk_free_rate': 0.03,
            'utility_aversion': 0.5
        }
    }
    
    @classmethod
    def get_risk_profile(cls, risk_level: RiskLevel) -> Dict:
        """Get risk profile for given risk level"""
        return cls.RISK_PROFILES.get(risk_level, cls.RISK_PROFILES[RiskLevel.MODERATE])

class PortfolioBuilder:
    """Advanced portfolio builder for risk-based project allocation"""
    
    def __init__(self, risk_manager=None):
        self.projects: Dict[str, ProjectProfile] = {}
        self.risk_manager = risk_manager
        self.optimization_history = []
        
    def add_project(self, project: ProjectProfile):
        """Add a project to the portfolio builder"""
        self.projects[project.name] = project
        logger.info(f"Added project: {project.name} ({project.project_type.value})")
    
    def load_projects_from_strategies(self, strategy_results_file: str):
        """Load projects from existing strategy results"""
        try:
            with open(strategy_results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for strategy_name, strategy_data in data.items():
                if isinstance(strategy_data, dict):
                    # Determine project type based on strategy name
                    project_type = self._determine_project_type(strategy_name)
                    
                    # Extract performance metrics
                    monthly_return = strategy_data.get('monthly_return', 0)
                    annual_return = (1 + monthly_return) ** 12 - 1
                    volatility = strategy_data.get('volatility', 0.02) * np.sqrt(252)
                    max_dd = abs(strategy_data.get('max_drawdown', 0))
                    
                    # Create project profile
                    project = ProjectProfile(
                        name=strategy_name,
                        project_type=project_type,
                        expected_return=annual_return,
                        volatility=volatility,
                        max_drawdown=max_dd,
                        success_rate=strategy_data.get('win_rate', 0.6),
                        min_investment=1000,
                        max_investment=100000
                    )
                    
                    self.add_project(project)
            
            logger.info(f"Loaded {len(self.projects)} projects from strategy results")
            
        except Exception as e:
            logger.error(f"Error loading projects from {strategy_results_file}: {e}")
    
    def _determine_project_type(self, strategy_name: str) -> ProjectType:
        """Determine project type from strategy name"""
        name_lower = strategy_name.lower()
        
        if any(word in name_lower for word in ['arbitrage', 'spread', 'pairs']):
            return ProjectType.ARBITRAGE
        elif any(word in name_lower for word in ['momentum', 'trend', 'breakout']):
            return ProjectType.MOMENTUM
        elif any(word in name_lower for word in ['mean_reversion', 'reversion', 'oscillator']):
            return ProjectType.MEAN_REVERSION
        elif any(word in name_lower for word in ['ml', 'neural', 'predictive', 'ai']):
            return ProjectType.ML_PREDICTIVE
        elif any(word in name_lower for word in ['technical', 'rsi', 'macd', 'bollinger']):
            return ProjectType.TECHNICAL
        elif any(word in name_lower for word in ['hedge', 'hedging']):
            return ProjectType.HEDGE
        elif any(word in name_lower for word in ['scalping', 'scalp']):
            return ProjectType.SCALPING
        else:
            return ProjectType.TECHNICAL  # Default
    
    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """Calculate correlation matrix for all projects"""
        if len(self.projects) < 2:
            return pd.DataFrame()
        
        # Create synthetic returns for correlation calculation
        n_days = 252
        returns_data = {}
        
        for name, project in self.projects.items():
            # Generate synthetic daily returns based on project characteristics
            daily_return = project.expected_return / 252
            daily_vol = project.volatility / np.sqrt(252)
            
            # Add some correlation based on project type
            base_returns = np.random.normal(daily_return, daily_vol, n_days)
            
            # Add project type correlation
            if project.project_type in [ProjectType.ARBITRAGE, ProjectType.HEDGE]:
                # Lower correlation with market
                returns = base_returns * 0.3
            elif project.project_type in [ProjectType.MOMENTUM, ProjectType.TECHNICAL]:
                # Higher correlation with market
                returns = base_returns * 1.2
            else:
                returns = base_returns
            
            returns_data[name] = returns
        
        returns_df = pd.DataFrame(returns_data)
        correlation_matrix = returns_df.corr()
        
        # Update correlation matrices for all projects
        for project in self.projects.values():
            project.correlation_matrix = correlation_matrix
        
        return correlation_matrix
    
    def filter_projects_by_constraints(self, constraints: PortfolioConstraints) -> List[str]:
        """Filter projects based on portfolio constraints"""
        filtered_projects = []
        
        for name, project in self.projects.items():
            # Check investment limits - make them more flexible
            min_allocation_amount = constraints.total_capital * constraints.min_single_allocation
            max_allocation_amount = constraints.total_capital * constraints.max_single_allocation
            
            # Allow projects if they can fit within allocation limits
            if project.min_investment > max_allocation_amount:
                continue
            
            # Check risk constraints - make them more flexible
            risk_profile = RiskProfileManager.get_risk_profile(constraints.risk_level)
            
            # Only apply strict volatility filter for conservative portfolios
            if constraints.risk_level == RiskLevel.CONSERVATIVE and project.volatility > 0.25:
                continue
            
            # Only apply strict drawdown filter for conservative portfolios
            if constraints.risk_level == RiskLevel.CONSERVATIVE and project.max_drawdown > 0.15:
                continue
            
            # Relax liquidity requirement for smaller portfolios
            min_liquidity = constraints.liquidity_requirement
            if constraints.total_capital < 500000:  # Less than 500K
                min_liquidity *= 0.7  # Reduce liquidity requirement by 30%
            
            if project.liquidity_score < min_liquidity:
                continue
            
            # Relax complexity limit for larger portfolios
            max_complexity = constraints.complexity_limit
            if constraints.total_capital > 2000000:  # More than 2M
                max_complexity *= 1.2  # Increase complexity limit by 20%
            
            if project.complexity_score > max_complexity:
                continue
            
            filtered_projects.append(name)
        
        logger.info(f"Filtered {len(filtered_projects)} projects from {len(self.projects)}")
        return filtered_projects
    
    def calculate_portfolio_metrics(self, weights: np.array, project_names: List[str]) -> Dict:
        """Calculate portfolio performance metrics"""
        if len(weights) != len(project_names):
            raise ValueError("Weights and project names must have same length")
        
        # Portfolio return and risk
        portfolio_return = sum(weights[i] * self.projects[name].expected_return 
                              for i, name in enumerate(project_names))
        
        # Portfolio volatility with correlation
        portfolio_variance = 0
        for i, name1 in enumerate(project_names):
            for j, name2 in enumerate(project_names):
                vol1 = self.projects[name1].volatility
                vol2 = self.projects[name2].volatility
                weight1 = weights[i]
                weight2 = weights[j]
                
                if i == j:
                    correlation = 1.0
                else:
                    # Get correlation from matrix
                    corr_matrix = self.projects[name1].correlation_matrix
                    if not corr_matrix.empty and name2 in corr_matrix.columns:
                        correlation = corr_matrix.loc[name1, name2]
                    else:
                        correlation = 0.3  # Default correlation
                
                portfolio_variance += weight1 * weight2 * vol1 * vol2 * correlation
        
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Calculate Value at Risk (VaR)
        confidence_level = 0.95
        var_95 = norm.ppf(1 - confidence_level, portfolio_return, portfolio_volatility)
        
        # Calculate Conditional Value at Risk (CVaR)
        cvar_95 = portfolio_return - portfolio_volatility * norm.pdf(norm.ppf(1 - confidence_level)) / (1 - confidence_level)
        
        # Calculate Sharpe ratio
        risk_free_rate = RiskProfileManager.get_risk_profile(RiskLevel.MODERATE)['risk_free_rate']
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Calculate Sortino ratio (downside deviation)
        downside_returns = [max(0, risk_free_rate - portfolio_return)]
        downside_deviation = np.sqrt(np.mean(downside_returns)) if downside_returns else 0
        sortino_ratio = (portfolio_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # Calculate maximum drawdown (simplified)
        max_drawdown = max(self.projects[name].max_drawdown * weights[i] 
                          for i, name in enumerate(project_names))
        
        return {
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'max_drawdown': max_drawdown,
            'diversification_ratio': self._calculate_diversification_ratio(weights, project_names, portfolio_volatility)
        }
    
    def _calculate_diversification_ratio(self, weights: np.array, project_names: List[str], portfolio_volatility: float = None) -> float:
        """Calculate diversification ratio"""
        weighted_vol = sum(weights[i] * self.projects[name].volatility 
                          for i, name in enumerate(project_names))
        
        if portfolio_volatility is None:
            # Calculate portfolio volatility directly to avoid recursion
            portfolio_variance = 0
            for i, name1 in enumerate(project_names):
                for j, name2 in enumerate(project_names):
                    vol1 = self.projects[name1].volatility
                    vol2 = self.projects[name2].volatility
                    weight1 = weights[i]
                    weight2 = weights[j]
                    
                    if i == j:
                        correlation = 1.0
                    else:
                        correlation = 0.3  # Default correlation
                    
                    portfolio_variance += weight1 * weight2 * vol1 * vol2 * correlation
            
            portfolio_volatility = np.sqrt(portfolio_variance)
        
        return weighted_vol / portfolio_volatility if portfolio_volatility > 0 else 1.0
    
    def objective_function(self, weights: np.array, project_names: List[str], 
                          constraints: PortfolioConstraints, objective_type: str = 'utility') -> float:
        """Objective function for portfolio optimization"""
        # Calculate basic metrics without recursion
        portfolio_return = sum(weights[i] * self.projects[name].expected_return 
                              for i, name in enumerate(project_names))
        
        # Simple volatility calculation
        portfolio_variance = 0
        for i, name1 in enumerate(project_names):
            for j, name2 in enumerate(project_names):
                vol1 = self.projects[name1].volatility
                vol2 = self.projects[name2].volatility
                weight1 = weights[i]
                weight2 = weights[j]
                
                if i == j:
                    correlation = 1.0
                else:
                    correlation = 0.3  # Default correlation
                
                portfolio_variance += weight1 * weight2 * vol1 * vol2 * correlation
        
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        if objective_type == 'utility':
            # Utility function approach
            risk_profile = RiskProfileManager.get_risk_profile(constraints.risk_level)
            risk_aversion = risk_profile['utility_aversion']
            
            utility = portfolio_return - 0.5 * risk_aversion * portfolio_volatility ** 2
            return -utility  # Negative because we minimize
        
        elif objective_type == 'sharpe':
            risk_free_rate = 0.03
            sharpe = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
            return -sharpe
        
        elif objective_type == 'sortino':
            risk_free_rate = 0.03
            sortino = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
            return -sortino
        
        elif objective_type == 'risk_parity':
            # Risk parity objective
            risk_contributions = []
            for i, name in enumerate(project_names):
                risk_contrib = weights[i] * self.projects[name].volatility
                risk_contributions.append(risk_contrib)
            
            target_risk = np.mean(risk_contributions)
            risk_parity_penalty = sum((rc - target_risk) ** 2 for rc in risk_contributions)
            return risk_parity_penalty
        
        else:
            return -portfolio_return
    
    def optimize_portfolio(self, constraints: PortfolioConstraints, 
                          objective_type: str = 'utility') -> Dict:
        """Optimize portfolio allocation"""
        # Filter projects
        available_projects = self.filter_projects_by_constraints(constraints)
        
        if len(available_projects) < constraints.min_projects:
            return {
                'success': False,
                'error': f"Not enough projects available. Need {constraints.min_projects}, have {len(available_projects)}"
            }
        
        # Limit number of projects
        if len(available_projects) > constraints.max_projects:
            # Select best projects based on Sharpe ratio
            project_scores = []
            for name in available_projects:
                project = self.projects[name]
                sharpe = (project.expected_return - 0.03) / project.volatility if project.volatility > 0 else 0
                project_scores.append((name, sharpe))
            
            project_scores.sort(key=lambda x: x[1], reverse=True)
            available_projects = [name for name, _ in project_scores[:constraints.max_projects]]
        
        # Calculate correlation matrix
        self.calculate_correlation_matrix()
        
        n_projects = len(available_projects)
        
        # Initial weights (equal allocation)
        initial_weights = np.array([1.0 / n_projects] * n_projects)
        
        # Constraints
        constraints_list = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # Weights sum to 1
        ]
        
        # Add target return constraint if specified
        if constraints.target_return:
            constraints_list.append({
                'type': 'ineq',
                'fun': lambda w: sum(w[i] * self.projects[name].expected_return 
                                   for i, name in enumerate(available_projects)) - constraints.target_return
            })
        
        # Add volatility constraint if specified
        if constraints.max_volatility:
            def volatility_constraint(weights):
                portfolio_variance = 0
                for i, name1 in enumerate(available_projects):
                    for j, name2 in enumerate(available_projects):
                        vol1 = self.projects[name1].volatility
                        vol2 = self.projects[name2].volatility
                        weight1 = weights[i]
                        weight2 = weights[j]
                        
                        if i == j:
                            correlation = 1.0
                        else:
                            correlation = 0.3
                        
                        portfolio_variance += weight1 * weight2 * vol1 * vol2 * correlation
                
                portfolio_volatility = np.sqrt(portfolio_variance)
                return constraints.max_volatility - portfolio_volatility
            
            constraints_list.append({
                'type': 'ineq',
                'fun': volatility_constraint
            })
        
        # Bounds
        bounds = [(constraints.min_single_allocation, constraints.max_single_allocation) 
                 for _ in range(n_projects)]
        
        try:
            # Optimize
            result = minimize(
                fun=self.objective_function,
                x0=initial_weights,
                args=(available_projects, constraints, objective_type),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_list,
                options={'maxiter': 1000}
            )
            
            if result.success:
                optimal_weights = result.x
                
                # Calculate final metrics
                final_metrics = self.calculate_portfolio_metrics(optimal_weights, available_projects)
                
                # Create allocation plan
                allocation_plan = {}
                for i, name in enumerate(available_projects):
                    allocation_amount = constraints.total_capital * optimal_weights[i]
                    allocation_plan[name] = {
                        'weight': optimal_weights[i],
                        'amount': allocation_amount,
                        'project_type': self.projects[name].project_type.value,
                        'expected_return': self.projects[name].expected_return,
                        'volatility': self.projects[name].volatility
                    }
                
                optimization_result = {
                    'success': True,
                    'allocation_plan': allocation_plan,
                    'portfolio_metrics': final_metrics,
                    'constraints': constraints,
                    'objective_type': objective_type,
                    'optimization_date': datetime.now().isoformat()
                }
                
                # Store in history
                self.optimization_history.append(optimization_result)
                
                logger.info(f"Portfolio optimization successful. Expected return: {final_metrics['expected_return']:.2%}")
                return optimization_result
            
            else:
                logger.error(f"Optimization failed: {result.message}")
                return {'success': False, 'error': result.message}
                
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            return {'success': False, 'error': str(e)}
    
    def create_risk_based_portfolios(self, total_capital: float) -> Dict:
        """Create portfolios for different risk levels"""
        portfolios = {}
        
        for risk_level in RiskLevel:
            constraints = PortfolioConstraints(
                total_capital=total_capital,
                risk_level=risk_level
            )
            
            # Set constraints based on risk profile
            risk_profile = RiskProfileManager.get_risk_profile(risk_level)
            constraints.target_return = risk_profile['target_return']
            constraints.max_volatility = risk_profile['max_volatility']
            constraints.max_drawdown = risk_profile['max_drawdown']
            
            # Optimize portfolio
            result = self.optimize_portfolio(constraints, 'utility')
            portfolios[risk_level.value] = result
        
        return portfolios
    
    def generate_allocation_report(self, optimization_result: Dict) -> str:
        """Generate comprehensive allocation report"""
        if not optimization_result.get('success', False):
            return f"Optimization failed: {optimization_result.get('error', 'Unknown error')}"
        
        allocation_plan = optimization_result['allocation_plan']
        metrics = optimization_result['portfolio_metrics']
        constraints = optimization_result['constraints']
        
        report = f"""
🎯 ПОРТФЕЛЬ ПРОЕКТОВ - ОТЧЕТ О РАСПРЕДЕЛЕНИИ КАПИТАЛА
{'='*80}

📊 ПАРАМЕТРЫ ПОРТФЕЛЯ:
- Общий капитал: {constraints.total_capital:,.0f} ₽
- Уровень риска: {constraints.risk_level.value.upper()}
- Количество проектов: {len(allocation_plan)}
- Тип оптимизации: {optimization_result['objective_type']}

📈 ОЖИДАЕМЫЕ ПОКАЗАТЕЛИ:
- Ожидаемая доходность: {metrics['expected_return']:.2%} годовых
- Волатильность: {metrics['volatility']:.2%} годовых
- Коэффициент Шарпа: {metrics['sharpe_ratio']:.3f}
- Коэффициент Сортино: {metrics['sortino_ratio']:.3f}
- Value at Risk (95%): {metrics['var_95']:.2%}
- Максимальная просадка: {metrics['max_drawdown']:.2%}
- Коэффициент диверсификации: {metrics['diversification_ratio']:.3f}

💼 ПЛАН РАСПРЕДЕЛЕНИЯ КАПИТАЛА:
{'-'*80}
"""
        
        # Sort by allocation amount
        sorted_allocations = sorted(
            allocation_plan.items(),
            key=lambda x: x[1]['amount'],
            reverse=True
        )
        
        for i, (project_name, allocation) in enumerate(sorted_allocations, 1):
            report += f"""
{i:2d}. {project_name}
    Тип проекта: {allocation['project_type']}
    Доля в портфеле: {allocation['weight']:.1%}
    Сумма инвестиций: {allocation['amount']:>10,.0f} ₽
    Ожидаемая доходность: {allocation['expected_return']:.2%}
    Волатильность: {allocation['volatility']:.2%}
"""
        
        # Risk analysis by project type
        report += f"""

📊 АНАЛИЗ ПО ТИПАМ ПРОЕКТОВ:
{'-'*80}
"""
        
        type_analysis = {}
        for allocation in allocation_plan.values():
            project_type = allocation['project_type']
            if project_type not in type_analysis:
                type_analysis[project_type] = {'weight': 0, 'amount': 0, 'count': 0}
            
            type_analysis[project_type]['weight'] += allocation['weight']
            type_analysis[project_type]['amount'] += allocation['amount']
            type_analysis[project_type]['count'] += 1
        
        for project_type, data in type_analysis.items():
            report += f"""
{project_type.upper()}:
  - Доля в портфеле: {data['weight']:.1%}
  - Сумма инвестиций: {data['amount']:,.0f} ₽
  - Количество проектов: {data['count']}
"""
        
        # Recommendations
        report += f"""

💡 РЕКОМЕНДАЦИИ ПО УПРАВЛЕНИЮ:
{'-'*80}
1. МОНИТОРИНГ:
   - Еженедельный анализ доходности каждого проекта
   - Ежемесячная ребалансировка портфеля
   - Контроль корреляции между проектами

2. РИСК-МЕНЕДЖМЕНТ:
   - Установить стоп-лоссы на уровне {metrics['var_95']:.2%}
   - Диверсифицировать по времени входа в проекты
   - Контролировать общую просадку портфеля

3. ОПТИМИЗАЦИЯ:
   - Пересматривать веса при значительных отклонениях
   - Добавлять новые проекты при появлении возможностей
   - Учитывать изменения в рыночных условиях

📈 ПРОГНОЗ РОСТА КАПИТАЛА:
{'-'*80}
При начальном капитале {constraints.total_capital:,.0f} ₽ и доходности {metrics['expected_return']:.2%} годовых:

"""
        
        projections = [1, 3, 6, 12]  # months
        for months in projections:
            final_amount = constraints.total_capital * (1 + metrics['expected_return']) ** (months/12)
            profit = final_amount - constraints.total_capital
            report += f"   Через {months:2d} мес.: {final_amount:>10,.0f} ₽ (прибыль: {profit:>8,.0f} ₽)\n"
        
        report += f"""

⚠️ ВАЖНЫЕ ЗАМЕЧАНИЯ:
{'-'*80}
• Результаты основаны на исторических данных и прогнозах
• Будущая доходность может отличаться от ожидаемой
• Диверсификация снижает риски, но не гарантирует прибыль
• Рекомендуется начать с небольшого капитала для тестирования
• Обязательно используйте риск-менеджмент и стоп-лоссы

📅 Дата создания: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return report
    
    def save_portfolio_analysis(self, optimization_result: Dict, filename: str = None):
        """Save portfolio analysis to file"""
        if filename is None:
            filename = f"portfolio_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Prepare data for saving
        save_data = {
            'optimization_result': optimization_result,
            'projects_data': {
                name: {
                    'project_type': project.project_type.value,
                    'expected_return': project.expected_return,
                    'volatility': project.volatility,
                    'max_drawdown': project.max_drawdown
                }
                for name, project in self.projects.items()
            },
            'analysis_date': datetime.now().isoformat()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Portfolio analysis saved to {filename}")

def main():
    """Main function to demonstrate portfolio building"""
    # Initialize portfolio builder
    builder = PortfolioBuilder()
    
    # Load projects from existing strategy results
    try:
        builder.load_projects_from_strategies('advanced_strategy_results.json')
    except FileNotFoundError:
        logger.error("Strategy results file not found. Run advanced_test.py first.")
        return
    
    # Create portfolios for different risk levels
    total_capital = 1000000  # 1 million rubles
    
    logger.info("Creating risk-based portfolios...")
    portfolios = builder.create_risk_based_portfolios(total_capital)
    
    # Generate reports for each risk level
    for risk_level, result in portfolios.items():
        if result.get('success', False):
            print(f"\n{'='*80}")
            print(f"ПОРТФЕЛЬ ДЛЯ УРОВНЯ РИСКА: {risk_level.upper()}")
            print(f"{'='*80}")
            
            report = builder.generate_allocation_report(result)
            print(report)
            
            # Save individual portfolio analysis
            builder.save_portfolio_analysis(result, f"portfolio_{risk_level}.json")
    
    # Save comprehensive analysis
    comprehensive_analysis = {
        'portfolios': portfolios,
        'total_capital': total_capital,
        'analysis_date': datetime.now().isoformat()
    }
    
    with open('comprehensive_portfolio_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(comprehensive_analysis, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info("Portfolio analysis completed!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
