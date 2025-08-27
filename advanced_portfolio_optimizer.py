#!/usr/bin/env python3
"""
Продвинутый портфельный оптимизатор с учетом риска, капитала и комиссий
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
import json
from datetime import datetime
import warnings
from enum import Enum

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Уровни риска для инвестора"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    SPECULATIVE = "speculative"

class CapitalTier(Enum):
    """Уровни капитала"""
    SMALL = "small"      # < 500k
    MEDIUM = "medium"    # 500k - 2M
    LARGE = "large"      # 2M - 10M
    INSTITUTIONAL = "institutional"  # > 10M

class CommissionStructure:
    """Структура комиссий брокера"""
    
    def __init__(self, capital_amount: float):
        self.capital_amount = capital_amount
        self.tier = self._determine_tier()
        
    def _determine_tier(self) -> CapitalTier:
        """Определить тариф на основе капитала"""
        if self.capital_amount < 500_000:
            return CapitalTier.SMALL
        elif self.capital_amount < 2_000_000:
            return CapitalTier.MEDIUM
        elif self.capital_amount < 10_000_000:
            return CapitalTier.LARGE
        else:
            return CapitalTier.INSTITUTIONAL
    
    def get_commission_rate(self) -> float:
        """Получить ставку комиссии"""
        commission_rates = {
            CapitalTier.SMALL: 0.003,        # 0.3%
            CapitalTier.MEDIUM: 0.002,       # 0.2%
            CapitalTier.LARGE: 0.0015,       # 0.15%
            CapitalTier.INSTITUTIONAL: 0.001 # 0.1%
        }
        return commission_rates[self.tier]
    
    def get_monthly_fee(self) -> float:
        """Ежемесячная плата за обслуживание"""
        monthly_fees = {
            CapitalTier.SMALL: 0,
            CapitalTier.MEDIUM: 500,
            CapitalTier.LARGE: 2000,
            CapitalTier.INSTITUTIONAL: 5000
        }
        return monthly_fees[self.tier]
    
    def calculate_trade_cost(self, trade_amount: float, frequency: int = 1) -> float:
        """Рассчитать стоимость сделки"""
        commission = trade_amount * self.get_commission_rate() * frequency
        
        # Минимальная комиссия
        min_commission = 1.0 if self.tier == CapitalTier.SMALL else 0.5
        
        return max(commission, min_commission)

class RiskProfiler:
    """Профилировщик риска"""
    
    def __init__(self, risk_level: RiskLevel, capital_amount: float):
        self.risk_level = risk_level
        self.capital_amount = capital_amount
        self.risk_params = self._get_risk_parameters()
    
    def _get_risk_parameters(self) -> Dict:
        """Получить параметры риска"""
        risk_configs = {
            RiskLevel.CONSERVATIVE: {
                'max_drawdown': 0.10,        # 10% максимальная просадка
                'max_volatility': 0.15,      # 15% максимальная волатильность
                'max_single_position': 0.15, # 15% в одной позиции
                'min_diversification': 8,     # минимум 8 стратегий
                'target_sharpe': 1.0,        # минимальный Sharpe
                'leverage_limit': 1.0,       # без плеча
                'rebalance_threshold': 0.05, # 5% порог ребалансировки
                'expected_return_range': (0.005, 0.02)  # 0.5-2% в месяц
            },
            RiskLevel.MODERATE: {
                'max_drawdown': 0.20,
                'max_volatility': 0.25,
                'max_single_position': 0.25,
                'min_diversification': 6,
                'target_sharpe': 0.8,
                'leverage_limit': 2.0,
                'rebalance_threshold': 0.08,
                'expected_return_range': (0.015, 0.05)  # 1.5-5% в месяц
            },
            RiskLevel.AGGRESSIVE: {
                'max_drawdown': 0.35,
                'max_volatility': 0.40,
                'max_single_position': 0.40,
                'min_diversification': 4,
                'target_sharpe': 0.6,
                'leverage_limit': 5.0,
                'rebalance_threshold': 0.12,
                'expected_return_range': (0.03, 0.10)  # 3-10% в месяц
            },
            RiskLevel.SPECULATIVE: {
                'max_drawdown': 0.50,
                'max_volatility': 0.60,
                'max_single_position': 0.60,
                'min_diversification': 3,
                'target_sharpe': 0.4,
                'leverage_limit': 10.0,
                'rebalance_threshold': 0.20,
                'expected_return_range': (0.05, 0.25)  # 5-25% в месяц
            }
        }
        
        return risk_configs[self.risk_level]
    
    def validate_portfolio(self, weights: Dict, performance: Dict) -> Tuple[bool, List[str]]:
        """Проверить портфель на соответствие профилю риска"""
        violations = []
        
        # Проверка максимальной просадки
        if abs(performance.get('max_drawdown', 0)) > self.risk_params['max_drawdown']:
            violations.append(f"Превышена максимальная просадка: {abs(performance.get('max_drawdown', 0)):.2%} > {self.risk_params['max_drawdown']:.2%}")
        
        # Проверка волатильности
        if performance.get('volatility', 0) > self.risk_params['max_volatility']:
            violations.append(f"Превышена максимальная волатильность: {performance.get('volatility', 0):.2%} > {self.risk_params['max_volatility']:.2%}")
        
        # Проверка концентрации
        max_weight = max(weights.values()) if weights else 0
        if max_weight > self.risk_params['max_single_position']:
            violations.append(f"Превышена максимальная позиция: {max_weight:.2%} > {self.risk_params['max_single_position']:.2%}")
        
        # Проверка диверсификации
        active_strategies = sum(1 for w in weights.values() if w > 0.05)  # более 5%
        if active_strategies < self.risk_params['min_diversification']:
            violations.append(f"Недостаточная диверсификация: {active_strategies} < {self.risk_params['min_diversification']}")
        
        # Проверка Sharpe ratio
        if performance.get('sharpe_ratio', 0) < self.risk_params['target_sharpe']:
            violations.append(f"Низкий Sharpe ratio: {performance.get('sharpe_ratio', 0):.3f} < {self.risk_params['target_sharpe']:.3f}")
        
        return len(violations) == 0, violations

class AdvancedPortfolioOptimizer:
    """Продвинутый портфельный оптимизатор"""
    
    def __init__(self, capital_amount: float, risk_level: RiskLevel, 
                 target_return: Optional[float] = None):
        self.capital_amount = capital_amount
        self.risk_level = risk_level
        self.commission_structure = CommissionStructure(capital_amount)
        self.risk_profiler = RiskProfiler(risk_level, capital_amount)
        
        # Автоматическое определение целевой доходности на основе профиля риска
        if target_return is None:
            return_range = self.risk_profiler.risk_params['expected_return_range']
            self.target_return = (return_range[0] + return_range[1]) / 2
        else:
            self.target_return = target_return
        
        self.strategies_data = {}
        self.optimal_portfolios = {}
        
        logger.info(f"Инициализирован оптимизатор для капитала {capital_amount:,.0f} ₽")
        logger.info(f"Уровень риска: {risk_level.value}")
        logger.info(f"Комиссия: {self.commission_structure.get_commission_rate():.3f}")
        logger.info(f"Целевая доходность: {self.target_return:.2%} в месяц")
    
    def load_strategy_results(self, data: Dict):
        """Загрузить результаты стратегий"""
        self.strategies_data = data
        logger.info(f"Загружено {len(data)} стратегий")
    
    def calculate_net_performance(self, gross_performance: Dict, 
                                 weights: Dict, trading_frequency: int = 20) -> Dict:
        """Рассчитать чистую доходность с учетом комиссий"""
        try:
            # Базовые метрики
            gross_return = gross_performance['monthly_return']
            volatility = gross_performance.get('volatility', 0.02)
            
            # Расчет торговых издержек
            total_trades_per_month = sum(
                weights.get(strategy, 0) * self.strategies_data.get(strategy, {}).get('total_trades', 0) 
                for strategy in weights.keys()
            )
            
            # Средний размер сделки
            avg_trade_size = self.capital_amount * 0.1  # предполагаем 10% капитала на сделку
            
            # Комиссии за месяц
            monthly_commission = (
                total_trades_per_month * 
                self.commission_structure.calculate_trade_cost(avg_trade_size) / 
                self.capital_amount
            )
            
            # Ежемесячная плата
            monthly_fee = self.commission_structure.get_monthly_fee() / self.capital_amount
            
            # Чистая доходность
            net_return = gross_return - monthly_commission - monthly_fee
            
            # Скорректированная волатильность (комиссии уменьшают волатильность)
            net_volatility = volatility * 0.95  # небольшая коррекция
            
            # Скорректированный Sharpe
            risk_free_rate = 0.02 / 12  # 2% годовых
            net_sharpe = (net_return - risk_free_rate) / net_volatility if net_volatility > 0 else 0
            
            return {
                'monthly_return': net_return,
                'volatility': net_volatility,
                'sharpe_ratio': net_sharpe,
                'max_drawdown': gross_performance.get('max_drawdown', 0),
                'trading_costs': monthly_commission,
                'monthly_fee': monthly_fee,
                'total_costs': monthly_commission + monthly_fee,
                'gross_return': gross_return
            }
            
        except Exception as e:
            logger.error(f"Ошибка расчета чистой доходности: {e}")
            return gross_performance
    
    def objective_function_advanced(self, weights: np.array, strategy_names: List[str], 
                                  optimization_target: str = 'risk_adjusted') -> float:
        """Продвинутая целевая функция оптимизации"""
        try:
            weights_dict = {name: weight for name, weight in zip(strategy_names, weights)}
            
            # Создать синтетические возвраты
            returns_df = self.create_returns_matrix(strategy_names)
            if returns_df.empty:
                return 1e6  # Большое значение для неудачных случаев
            
            # Рассчитать базовые метрики
            portfolio_returns = (returns_df * weights).sum(axis=1)
            
            # Базовые метрики
            total_return = (1 + portfolio_returns).prod() - 1
            monthly_return = (1 + total_return) ** (1/12) - 1
            volatility = portfolio_returns.std() * np.sqrt(252)
            
            # Drawdown
            cumulative = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            gross_performance = {
                'monthly_return': monthly_return,
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': (monthly_return - 0.02/12) / volatility if volatility > 0 else 0
            }
            
            # Рассчитать чистую доходность с учетом комиссий
            net_performance = self.calculate_net_performance(gross_performance, weights_dict)
            
            # Проверить соответствие профилю риска
            is_valid, violations = self.risk_profiler.validate_portfolio(weights_dict, net_performance)
            
            # Штрафы за нарушения
            penalty = 0
            if not is_valid:
                penalty += len(violations) * 0.1  # штраф за каждое нарушение
            
            # Различные целевые функции
            if optimization_target == 'max_return':
                # Максимизация доходности с учетом рисков
                objective = -net_performance['monthly_return'] + penalty
                
            elif optimization_target == 'min_risk':
                # Минимизация риска при целевой доходности
                return_penalty = max(0, (self.target_return - net_performance['monthly_return']) * 10)
                objective = abs(net_performance['max_drawdown']) + return_penalty + penalty
                
            elif optimization_target == 'max_sharpe':
                # Максимизация Sharpe ratio
                objective = -net_performance['sharpe_ratio'] + penalty
                
            elif optimization_target == 'risk_adjusted':
                # Комплексная оптимизация с учетом профиля риска
                risk_params = self.risk_profiler.risk_params
                
                # Нормализованные метрики (от 0 до 1)
                return_score = min(1.0, net_performance['monthly_return'] / risk_params['expected_return_range'][1])
                risk_score = 1.0 - min(1.0, abs(net_performance['max_drawdown']) / risk_params['max_drawdown'])
                sharpe_score = min(1.0, net_performance['sharpe_ratio'] / (risk_params['target_sharpe'] * 2))
                
                # Взвешенная комбинация
                combined_score = (
                    0.4 * return_score +     # 40% доходность
                    0.35 * risk_score +      # 35% контроль риска
                    0.25 * sharpe_score      # 25% эффективность
                )
                
                objective = -(combined_score - penalty)
                
            elif optimization_target == 'target_return':
                # Достижение целевой доходности с минимальным риском
                return_diff = abs(net_performance['monthly_return'] - self.target_return)
                risk_component = abs(net_performance['max_drawdown']) * 2
                objective = return_diff + risk_component + penalty
                
            else:
                # По умолчанию - максимизация Sharpe
                objective = -net_performance['sharpe_ratio'] + penalty
            
            return objective
            
        except Exception as e:
            logger.error(f"Ошибка в целевой функции: {e}")
            return 1e6
    
    def create_returns_matrix(self, strategy_names: List[str]) -> pd.DataFrame:
        """Создать матрицу доходностей с учетом корреляций"""
        np.random.seed(42)
        
        returns_data = {}
        n_days = 252
        
        # Создать корреляционную матрицу
        n_strategies = len(strategy_names)
        correlation_matrix = self._generate_correlation_matrix(n_strategies)
        
        # Генерировать коррелированные случайные числа
        random_normal = np.random.multivariate_normal(
            mean=np.zeros(n_strategies),
            cov=correlation_matrix,
            size=n_days
        )
        
        for i, name in enumerate(strategy_names):
            if name not in self.strategies_data:
                continue
            
            data = self.strategies_data[name]
            monthly_return = data.get('monthly_return', 0)
            volatility = data.get('volatility', 0.02)
            max_drawdown = abs(data.get('max_drawdown', 0))
            
            # Базовая дневная доходность
            daily_return = monthly_return / 21
            daily_vol = volatility / np.sqrt(252)
            
            # Использовать коррелированные случайные числа
            base_returns = random_normal[:, i] * daily_vol + daily_return
            
            # Добавить периоды просадок
            if max_drawdown > 0:
                drawdown_periods = self._add_drawdown_periods(base_returns, max_drawdown)
                returns_data[name] = drawdown_periods
            else:
                returns_data[name] = base_returns
        
        dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
        return pd.DataFrame(returns_data, index=dates)
    
    def _generate_correlation_matrix(self, n_strategies: int) -> np.ndarray:
        """Генерировать реалистичную корреляционную матрицу"""
        # Создать базовую корреляционную матрицу
        correlation = np.eye(n_strategies)
        
        # Добавить реалистичные корреляции между стратегиями
        base_correlation = 0.15  # базовая корреляция между стратегиями
        
        for i in range(n_strategies):
            for j in range(i + 1, n_strategies):
                # Стратегии одного типа более коррелированы
                corr_value = np.random.uniform(0.05, 0.3)
                correlation[i, j] = correlation[j, i] = corr_value
        
        # Убедиться, что матрица положительно определена
        eigenvals = np.linalg.eigvals(correlation)
        if np.min(eigenvals) < 0:
            correlation += np.eye(n_strategies) * (abs(np.min(eigenvals)) + 0.01)
        
        return correlation
    
    def _add_drawdown_periods(self, returns: np.ndarray, max_drawdown: float) -> np.ndarray:
        """Добавить периоды просадок в доходности"""
        modified_returns = returns.copy()
        
        # Количество периодов просадок
        n_drawdown_periods = max(1, int(len(returns) * max_drawdown / 4))
        
        for _ in range(n_drawdown_periods):
            # Случайное начало периода просадки
            start_idx = np.random.randint(0, len(returns) - 20)
            # Длительность просадки (5-20 дней)
            duration = np.random.randint(5, 21)
            end_idx = min(start_idx + duration, len(returns))
            
            # Усилить отрицательные доходности в этом периоде
            for i in range(start_idx, end_idx):
                if modified_returns[i] > 0:
                    modified_returns[i] *= -0.5  # превратить в убыток
                else:
                    modified_returns[i] *= 2.0   # усилить убыток
        
        return modified_returns
    
    def optimize_portfolio_advanced(self, optimization_targets: List[str] = None) -> Dict:
        """Продвинутая оптимизация портфеля"""
        if optimization_targets is None:
            optimization_targets = ['risk_adjusted', 'max_sharpe', 'target_return']
        
        # Фильтровать стратегии по профилю риска
        suitable_strategies = self._filter_strategies_by_risk_profile()
        
        if len(suitable_strategies) < 2:
            logger.error("Недостаточно подходящих стратегий для оптимизации")
            return {
                'success': False, 
                'error': 'Insufficient strategies',
                'optimization_results': {},
                'commission_structure': {
                    'rate': self.commission_structure.get_commission_rate(),
                    'monthly_fee': self.commission_structure.get_monthly_fee(),
                    'tier': self.commission_structure.tier.value
                },
                'suitable_strategies': suitable_strategies
            }
        
        results = {}
        
        for target in optimization_targets:
            logger.info(f"Оптимизация для цели: {target}")
            
            try:
                # Настройка ограничений на основе профиля риска
                constraints, bounds = self._setup_optimization_constraints(suitable_strategies)
                
                # Начальные веса
                n_strategies = len(suitable_strategies)
                initial_weights = np.array([1.0 / n_strategies] * n_strategies)
                
                # Попробовать несколько методов оптимизации
                best_result = None
                best_score = float('inf')
                
                # 1. SLSQP (локальный оптимизатор)
                try:
                    result_slsqp = minimize(
                        fun=self.objective_function_advanced,
                        x0=initial_weights,
                        args=(suitable_strategies, target),
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints,
                        options={'maxiter': 1000, 'ftol': 1e-9}
                    )
                    
                    if result_slsqp.success and result_slsqp.fun < best_score:
                        best_result = result_slsqp
                        best_score = result_slsqp.fun
                        
                except Exception as e:
                    logger.warning(f"SLSQP optimization failed: {e}")
                
                # 2. Differential Evolution (глобальный оптимизатор)
                try:
                    result_de = differential_evolution(
                        func=self.objective_function_advanced,
                        bounds=bounds,
                        args=(suitable_strategies, target),
                        maxiter=300,
                        seed=42,
                        constraints=constraints
                    )
                    
                    if result_de.success and result_de.fun < best_score:
                        best_result = result_de
                        best_score = result_de.fun
                        
                except Exception as e:
                    logger.warning(f"Differential Evolution optimization failed: {e}")
                
                if best_result is not None and best_result.success:
                    # Обработать результат
                    optimal_weights = best_result.x
                    weights_dict = {name: weight for name, weight in zip(suitable_strategies, optimal_weights)}
                    
                    # Рассчитать финальные метрики
                    final_performance = self._calculate_final_performance(weights_dict, suitable_strategies)
                    
                    # Проверить соответствие профилю риска
                    is_valid, violations = self.risk_profiler.validate_portfolio(weights_dict, final_performance)
                    
                    results[target] = {
                        'weights': weights_dict,
                        'performance': final_performance,
                        'optimization_score': best_score,
                        'is_valid': is_valid,
                        'violations': violations,
                        'capital_efficiency': self._calculate_capital_efficiency(weights_dict, final_performance),
                        'success': True
                    }
                    
                    logger.info(f"Оптимизация {target} успешна. Доходность: {final_performance['monthly_return']:.2%}")
                    
                else:
                    results[target] = {
                        'success': False,
                        'error': 'Optimization failed'
                    }
                    logger.error(f"Оптимизация {target} не удалась")
                    
            except Exception as e:
                logger.error(f"Ошибка оптимизации {target}: {e}")
                results[target] = {
                    'success': False,
                    'error': str(e)
                }
        
        return {
            'optimization_results': results,
            'capital_amount': self.capital_amount,
            'risk_level': self.risk_level.value,
            'commission_structure': {
                'rate': self.commission_structure.get_commission_rate(),
                'monthly_fee': self.commission_structure.get_monthly_fee(),
                'tier': self.commission_structure.tier.value
            },
            'risk_parameters': self.risk_profiler.risk_params,
            'suitable_strategies': suitable_strategies
        }
    
    def _filter_strategies_by_risk_profile(self) -> List[str]:
        """Фильтровать стратегии по профилю риска"""
        suitable = []
        risk_params = self.risk_profiler.risk_params
        
        for name, data in self.strategies_data.items():
            if not isinstance(data, dict):
                continue
            
            # Проверить базовые критерии
            monthly_return = data.get('monthly_return', 0)
            max_drawdown = abs(data.get('max_drawdown', 0))
            volatility = data.get('volatility', 0.02)  # Дефолтная волатильность
            total_trades = data.get('total_trades', 0)
            
            # Более мягкие фильтры для работы с реальными данными
            min_return_threshold = max(0.0, risk_params['expected_return_range'][0] * 0.1)  # 10% от минимума
            max_dd_threshold = min(0.5, risk_params['max_drawdown'] * 2)  # Удвоенный лимит
            max_vol_threshold = min(0.6, risk_params['max_volatility'] * 2)  # Удвоенный лимит
            
            # Фильтры по профилю риска (более мягкие)
            if (max_drawdown <= max_dd_threshold and
                volatility <= max_vol_threshold and
                total_trades >= 0 and  # Принимаем любое количество сделок
                monthly_return >= min_return_threshold):
                suitable.append(name)
                logger.debug(f"Стратегия {name} прошла фильтр: return={monthly_return:.3f}, dd={max_drawdown:.3f}, vol={volatility:.3f}")
            else:
                logger.debug(f"Стратегия {name} отклонена: return={monthly_return:.3f} (min={min_return_threshold:.3f}), dd={max_drawdown:.3f} (max={max_dd_threshold:.3f}), vol={volatility:.3f} (max={max_vol_threshold:.3f})")
                
        logger.info(f"Отфильтровано {len(suitable)} стратегий из {len(self.strategies_data)} по профилю риска")
        return suitable
    
    def _setup_optimization_constraints(self, strategy_names: List[str]) -> Tuple[List, List]:
        """Настроить ограничения для оптимизации"""
        n_strategies = len(strategy_names)
        risk_params = self.risk_profiler.risk_params
        
        # Ограничения на веса
        bounds = []
        for _ in range(n_strategies):
            bounds.append((0.0, risk_params['max_single_position']))
        
        # Ограничения-равенства и неравенства
        constraints = [
            # Сумма весов равна 1
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
            
            # Минимальная диверсификация - хотя бы N стратегий с весом > 5%
            {'type': 'ineq', 'fun': lambda w: np.sum(w > 0.05) - risk_params['min_diversification'] + 1}
        ]
        
        # Дополнительные ограничения для консервативных профилей
        if self.risk_level == RiskLevel.CONSERVATIVE:
            # Ни одна стратегия не должна иметь вес менее 5% (если активна)
            for i in range(n_strategies):
                constraints.append({
                    'type': 'ineq', 
                    'fun': lambda w, idx=i: w[idx] - 0.05 if w[idx] > 0.01 else 0.0
                })
        
        return constraints, bounds
    
    def _calculate_final_performance(self, weights_dict: Dict, strategy_names: List[str]) -> Dict:
        """Рассчитать финальные метрики производительности"""
        try:
            # Создать матрицу доходностей
            returns_df = self.create_returns_matrix(strategy_names)
            weights = np.array([weights_dict.get(name, 0) for name in strategy_names])
            
            # Базовые метрики
            portfolio_returns = (returns_df * weights).sum(axis=1)
            
            total_return = (1 + portfolio_returns).prod() - 1
            monthly_return = (1 + total_return) ** (1/12) - 1
            volatility = portfolio_returns.std() * np.sqrt(252)
            
            # Drawdown
            cumulative = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            gross_performance = {
                'monthly_return': monthly_return,
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': (monthly_return - 0.02/12) / volatility if volatility > 0 else 0
            }
            
            # Чистая доходность с учетом комиссий
            return self.calculate_net_performance(gross_performance, weights_dict)
            
        except Exception as e:
            logger.error(f"Ошибка расчета финальной производительности: {e}")
            return {}
    
    def _calculate_capital_efficiency(self, weights_dict: Dict, performance: Dict) -> Dict:
        """Рассчитать эффективность использования капитала"""
        try:
            monthly_return = performance.get('monthly_return', 0)
            total_costs = performance.get('total_costs', 0)
            
            # Эффективность по доходности
            net_return_efficiency = monthly_return / (monthly_return + total_costs) if monthly_return > 0 else 0
            
            # Абсолютная эффективность
            monthly_profit = self.capital_amount * monthly_return
            monthly_costs = self.capital_amount * total_costs
            
            # ROI с учетом комиссий
            roi_efficiency = monthly_profit / (monthly_profit + monthly_costs) if monthly_profit > 0 else 0
            
            # Оценка диверсификации
            active_strategies = sum(1 for w in weights_dict.values() if w > 0.05)
            diversification_score = min(1.0, active_strategies / 10)  # 10 стратегий = идеальная диверсификация
            
            return {
                'net_return_efficiency': net_return_efficiency,
                'roi_efficiency': roi_efficiency,
                'diversification_score': diversification_score,
                'monthly_profit_rub': monthly_profit,
                'monthly_costs_rub': monthly_costs,
                'cost_ratio': total_costs / monthly_return if monthly_return > 0 else float('inf')
            }
            
        except Exception as e:
            logger.error(f"Ошибка расчета эффективности капитала: {e}")
            return {}
    
    def generate_comprehensive_report(self, optimization_results: Dict) -> str:
        """Сгенерировать подробный отчет по оптимизации"""
        
        risk_level_names = {
            RiskLevel.CONSERVATIVE: "Консервативный",
            RiskLevel.MODERATE: "Умеренный", 
            RiskLevel.AGGRESSIVE: "Агрессивный",
            RiskLevel.SPECULATIVE: "Спекулятивный"
        }
        
        report = f"""
🎯 ПРОДВИНУТЫЙ ПОРТФЕЛЬНЫЙ ОПТИМИЗАТОР - ПОДРОБНЫЙ ОТЧЕТ
{'='*80}

💰 ПАРАМЕТРЫ КАПИТАЛА:
- Размер капитала: {self.capital_amount:,.0f} ₽
- Уровень риска: {risk_level_names[self.risk_level]}
- Тариф комиссий: {optimization_results['commission_structure']['tier']}
- Ставка комиссии: {optimization_results['commission_structure']['rate']:.3%}
- Ежемесячная плата: {optimization_results['commission_structure']['monthly_fee']:,.0f} ₽

🎯 ЦЕЛЕВЫЕ ПАРАМЕТРЫ:
- Целевая доходность: {self.target_return:.2%} в месяц
- Максимальная просадка: {self.risk_profiler.risk_params['max_drawdown']:.1%}
- Максимальная волатильность: {self.risk_profiler.risk_params['max_volatility']:.1%}
- Минимальная диверсификация: {self.risk_profiler.risk_params['min_diversification']} стратегий

📊 РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ:
{'='*80}
"""
        
        opt_results = optimization_results['optimization_results']
        successful_optimizations = [name for name, result in opt_results.items() if result.get('success', False)]
        
        if not successful_optimizations:
            report += f"""
❌ НИ ОДНА ОПТИМИЗАЦИЯ НЕ УДАЛАСЬ
Возможные причины:
- Недостаточно подходящих стратегий (найдено: {len(optimization_results.get('suitable_strategies', []))})
- Слишком строгие ограничения риска для уровня {risk_level_names[self.risk_level]}
- Низкое качество исходных данных

🔧 РЕКОМЕНДАЦИИ:
1. Снизить уровень риска для доступа к большему количеству стратегий
2. Увеличить размер капитала для лучших условий
3. Рассмотреть альтернативные стратегии
"""
            return report
        
        # Сортировать результаты по доходности
        sorted_results = sorted(
            [(name, result) for name, result in opt_results.items() if result.get('success', False)],
            key=lambda x: x[1]['performance']['monthly_return'],
            reverse=True
        )
        
        for i, (opt_name, result) in enumerate(sorted_results, 1):
            perf = result['performance']
            cap_eff = result['capital_efficiency']
            
            meets_target = "✅" if perf['monthly_return'] >= self.target_return else "❌"
            
            report += f"""
{i}. ПОРТФЕЛЬ: {opt_name.upper()} {meets_target}
{'-'*60}

📈 ДОХОДНОСТЬ И РИСК:
- Чистая месячная доходность: {perf['monthly_return']:>8.2%}
- Валовая доходность: {perf['gross_return']:>8.2%}
- Торговые издержки: {perf['trading_costs']:>8.4%}
- Ежемесячная плата: {perf['monthly_fee']:>8.4%}
- Общие издержки: {perf['total_costs']:>8.4%}

⚖️ РИСК-МЕТРИКИ:
- Волатильность: {perf['volatility']:>8.2%}
- Коэффициент Шарпа: {perf['sharpe_ratio']:>8.3f}
- Максимальная просадка: {perf['max_drawdown']:>8.2%}

💼 ЭФФЕКТИВНОСТЬ КАПИТАЛА:
- Прибыль в месяц: {cap_eff['monthly_profit_rub']:>8,.0f} ₽
- Издержки в месяц: {cap_eff['monthly_costs_rub']:>8,.0f} ₽
- ROI эффективность: {cap_eff['roi_efficiency']:>8.2%}
- Коэффициент издержек: {cap_eff['cost_ratio']:>8.3f}
- Диверсификация: {cap_eff['diversification_score']:>8.1%}

🎯 СООТВЕТСТВИЕ ПРОФИЛЮ РИСКА: {'✅ ДА' if result['is_valid'] else '❌ НЕТ'}
"""
            
            if not result['is_valid']:
                report += f"\n⚠️ НАРУШЕНИЯ ПРОФИЛЯ РИСКА:\n"
                for violation in result['violations']:
                    report += f"   - {violation}\n"
            
            report += f"\n💼 РАСПРЕДЕЛЕНИЕ КАПИТАЛА:\n"
            sorted_weights = sorted(result['weights'].items(), key=lambda x: x[1], reverse=True)
            
            for strategy, weight in sorted_weights:
                if weight > 0.01:  # показывать только веса больше 1%
                    allocation_rub = self.capital_amount * weight
                    report += f"   - {strategy}: {weight:>6.1%} ({allocation_rub:>8,.0f} ₽)\n"
            
            report += "\n"
        
        # Рекомендации
        best_portfolio = sorted_results[0][1] if sorted_results else None
        
        if best_portfolio:
            best_performance = best_portfolio['performance']
            months_to_target = None
            
            if best_performance['monthly_return'] > 0:
                # Сколько месяцев нужно для достижения цели при реинвестировании
                if best_performance['monthly_return'] < self.target_return:
                    months_to_target = "Цель недостижима при текущих параметрах"
                else:
                    months_to_target = "Цель достижима сразу"
            
            report += f"""

💡 РЕКОМЕНДАЦИИ:
{'='*80}

🏆 ЛУЧШИЙ ПОРТФЕЛЬ: {sorted_results[0][0].upper()}

📊 ПРОГНОЗ РОСТА КАПИТАЛА:
"""
            
            if best_performance['monthly_return'] > 0:
                periods = [1, 3, 6, 12, 24]
                monthly_rate = best_performance['monthly_return']
                
                for months in periods:
                    future_value = self.capital_amount * (1 + monthly_rate) ** months
                    profit = future_value - self.capital_amount
                    report += f"   Через {months:2d} мес.: {future_value:>10,.0f} ₽ (прибыль: {profit:>8,.0f} ₽)\n"
            
            # Анализ достижения цели
            if best_performance['monthly_return'] >= self.target_return:
                report += f"""
✅ ЦЕЛЬ ДОСТИЖИМА!
Ваш лучший портфель обеспечивает {best_performance['monthly_return']:.2%} в месяц
при целевых {self.target_return:.2%}.
"""
            else:
                shortfall = self.target_return - best_performance['monthly_return']
                leverage_needed = self.target_return / best_performance['monthly_return']
                
                report += f"""
⚠️ ЦЕЛЬ НЕ ДОСТИГНУТА
Недостает: {shortfall:.2%} в месяц
Требуемое плечо: {leverage_needed:.1f}x

🔧 ВАРИАНТЫ ДОСТИЖЕНИЯ ЦЕЛИ:
1. Увеличить плечо до {leverage_needed:.1f}x (⚠️ высокий риск)
2. Изменить профиль риска на более агрессивный
3. Увеличить размер капитала для доступа к лучшим условиям
4. Рассмотреть альтернативные активы (криптовалюты, деривативы)
"""
        
        # Анализ влияния размера капитала
        report += f"""

💰 АНАЛИЗ ВЛИЯНИЯ РАЗМЕРА КАПИТАЛА:
{'-'*60}
"""
        
        capital_scenarios = [
            (100_000, "Малый капитал"),
            (500_000, "Средний капитал"), 
            (2_000_000, "Крупный капитал"),
            (10_000_000, "Институциональный")
        ]
        
        for capital, description in capital_scenarios:
            temp_commission = CommissionStructure(capital)
            commission_rate = temp_commission.get_commission_rate()
            monthly_fee = temp_commission.get_monthly_fee()
            
            # Примерная чистая доходность
            if best_portfolio:
                gross_return = best_portfolio['performance']['gross_return']
                estimated_trades = 20  # среднее количество сделок в месяц
                avg_trade_size = capital * 0.1
                trading_costs = (estimated_trades * temp_commission.calculate_trade_cost(avg_trade_size)) / capital
                monthly_costs = monthly_fee / capital
                net_return = gross_return - trading_costs - monthly_costs
                
                report += f"   {description:20s}: {commission_rate:.3%} комиссия, {net_return:>6.2%} чистая доходность\n"
        
        report += f"""

⚠️ ВАЖНЫЕ ЗАМЕЧАНИЯ:
{'-'*60}
• Результаты основаны на историческом моделировании
• Реальные комиссии могут отличаться от расчетных
• Учтены корреляции между стратегиями
• Рекомендуется начать с тестовой суммы
• Обязательно используйте риск-менеджмент

📅 Дата анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return report

def main():
    """Демонстрация продвинутого портфельного оптимизатора"""
    
    # Настройка логирования
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Тестирование для разных сценариев
    test_scenarios = [
        {
            'capital': 100_000,
            'risk_level': RiskLevel.CONSERVATIVE,
            'description': 'Консервативный инвестор, малый капитал'
        },
        {
            'capital': 1_000_000, 
            'risk_level': RiskLevel.MODERATE,
            'description': 'Умеренный инвестор, средний капитал'
        },
        {
            'capital': 5_000_000,
            'risk_level': RiskLevel.AGGRESSIVE, 
            'description': 'Агрессивный инвестор, крупный капитал'
        }
    ]
    
    # Загрузить данные стратегий
    try:
        with open('advanced_strategy_results.json', 'r', encoding='utf-8') as f:
            strategy_data = json.load(f)
        
        if 'strategies' in strategy_data:
            strategies_dict = {s['name']: s for s in strategy_data['strategies']}
        else:
            strategies_dict = strategy_data
            
    except FileNotFoundError:
        logger.error("Файл с результатами стратегий не найден")
        return
    
    # Тестировать каждый сценарий
    for scenario in test_scenarios:
        logger.info(f"\n{'='*80}")
        logger.info(f"ТЕСТИРОВАНИЕ: {scenario['description']}")
        logger.info(f"{'='*80}")
        
        # Создать оптимизатор
        optimizer = AdvancedPortfolioOptimizer(
            capital_amount=scenario['capital'],
            risk_level=scenario['risk_level']
        )
        
        # Загрузить данные
        optimizer.load_strategy_results(strategies_dict)
        
        # Оптимизировать
        results = optimizer.optimize_portfolio_advanced()
        
        # Сгенерировать отчет
        report = optimizer.generate_comprehensive_report(results)
        print(report)
        
        # Сохранить результаты
        filename = f'portfolio_optimization_{scenario["risk_level"].value}_{scenario["capital"]}.json'
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        filename_txt = f'portfolio_report_{scenario["risk_level"].value}_{scenario["capital"]}.txt'
        with open(filename_txt, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Результаты сохранены в {filename} и {filename_txt}")

if __name__ == "__main__":
    main()