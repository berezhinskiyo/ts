#!/usr/bin/env python3
"""
Parameter Optimizer for Aggressive Strategy
Оптимизатор параметров агрессивной стратегии
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import os
import sys
from typing import Dict, List, Optional, Tuple
import optuna
from optuna.samplers import TPESampler

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ParameterOptimizer:
    """Оптимизатор параметров агрессивной стратегии"""
    
    def __init__(self, config_path: str = "config/parameters/aggressive_config.py"):
        self.config = self._load_config(config_path)
        self.optimization_history = []
        self.best_parameters = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """Загрузка конфигурации"""
        try:
            # В реальном проекте здесь будет загрузка из файла
            return {
                'leverage_range': (5.0, 50.0),
                'profit_target_range': (0.005, 0.05),
                'stop_loss_range': (0.002, 0.02),
                'trend_threshold_range': (0.01, 0.05),
                'volatility_threshold_range': (0.02, 0.05),
                'position_size_range': (0.05, 0.2),
                'optimization_trials': 100,
                'optimization_metric': 'sharpe_ratio',
                'backtest_period': 252,  # 1 год
                'min_trades': 10
            }
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def generate_test_data(self, days: int = 252) -> pd.DataFrame:
        """Генерация тестовых данных для оптимизации"""
        logger.info(f"📊 Generating {days} days of test data")
        
        try:
            np.random.seed(42)
            
            dates = pd.date_range(
                start=datetime.now() - timedelta(days=days),
                end=datetime.now(),
                freq='D'
            )
            
            # Генерируем более реалистичные данные с трендами
            data = []
            base_price = 100
            
            for i, date in enumerate(dates):
                # Добавляем тренды и волатильность
                if i < days // 3:
                    # Восходящий тренд
                    trend = 0.002
                    volatility = 0.02
                elif i < 2 * days // 3:
                    # Боковой тренд
                    trend = 0.000
                    volatility = 0.015
                else:
                    # Нисходящий тренд
                    trend = -0.001
                    volatility = 0.025
                
                price_change = np.random.normal(trend, volatility)
                base_price *= (1 + price_change)
                
                data.append({
                    'date': date,
                    'open': base_price * (1 + np.random.normal(0, 0.003)),
                    'high': base_price * (1 + abs(np.random.normal(0, 0.008))),
                    'low': base_price * (1 - abs(np.random.normal(0, 0.008))),
                    'close': base_price,
                    'volume': int(1000000 * np.random.uniform(0.5, 2.0))
                })
            
            df = pd.DataFrame(data)
            df.set_index('date', inplace=True)
            
            logger.info(f"✅ Generated {len(df)} data points")
            return df
            
        except Exception as e:
            logger.error(f"Error generating test data: {e}")
            return pd.DataFrame()
    
    def backtest_strategy(self, parameters: Dict, data: pd.DataFrame) -> Dict:
        """Бэктестинг стратегии с заданными параметрами"""
        try:
            # Извлечение параметров
            leverage = parameters['leverage']
            profit_target = parameters['profit_target']
            stop_loss = parameters['stop_loss']
            trend_threshold = parameters['trend_threshold']
            volatility_threshold = parameters['volatility_threshold']
            position_size = parameters['position_size']
            
            # Инициализация
            capital = 100000  # Начальный капитал
            positions = {}
            trades = []
            daily_values = []
            
            # Расчет технических индикаторов
            data['sma_short'] = data['close'].rolling(10).mean()
            data['sma_long'] = data['close'].rolling(30).mean()
            data['volatility'] = data['close'].pct_change().rolling(20).std()
            
            # Торговая логика
            for i in range(30, len(data)):  # Начинаем после 30 дней
                current_date = data.index[i]
                current_price = data['close'].iloc[i]
                
                # Расчет тренда
                sma_short = data['sma_short'].iloc[i]
                sma_long = data['sma_long'].iloc[i]
                trend_strength = (sma_short - sma_long) / sma_long
                
                # Расчет волатильности
                volatility = data['volatility'].iloc[i]
                
                # Условия для входа
                if (abs(trend_strength) > trend_threshold and 
                    volatility < volatility_threshold):
                    
                    # Определение направления сделки
                    if trend_strength > 0:
                        action = 'buy'
                    else:
                        action = 'sell'
                    
                    # Расчет размера позиции
                    position_value = capital * position_size * leverage
                    quantity = int(position_value / current_price)
                    
                    if quantity > 0:
                        # Выполнение сделки
                        if action == 'buy':
                            capital -= quantity * current_price
                            positions[current_date] = {
                                'action': 'buy',
                                'quantity': quantity,
                                'price': current_price,
                                'leverage': leverage
                            }
                        else:
                            capital += quantity * current_price
                            positions[current_date] = {
                                'action': 'sell',
                                'quantity': quantity,
                                'price': current_price,
                                'leverage': leverage
                            }
                        
                        trades.append({
                            'date': current_date,
                            'action': action,
                            'price': current_price,
                            'quantity': quantity,
                            'leverage': leverage
                        })
                
                # Управление позициями (упрощенное)
                for pos_date, position in list(positions.items()):
                    days_held = (current_date - pos_date).days
                    
                    # Take profit
                    if position['action'] == 'buy':
                        current_return = (current_price - position['price']) / position['price']
                        if current_return > profit_target:
                            capital += position['quantity'] * current_price
                            del positions[pos_date]
                    else:
                        current_return = (position['price'] - current_price) / position['price']
                        if current_return > profit_target:
                            capital -= position['quantity'] * current_price
                            del positions[pos_date]
                    
                    # Stop loss
                    if position['action'] == 'buy':
                        current_return = (current_price - position['price']) / position['price']
                        if current_return < -stop_loss:
                            capital += position['quantity'] * current_price
                            del positions[pos_date]
                    else:
                        current_return = (position['price'] - current_price) / position['price']
                        if current_return < -stop_loss:
                            capital -= position['quantity'] * current_price
                            del positions[pos_date]
                
                # Расчет текущей стоимости портфеля
                portfolio_value = capital
                for pos_date, position in positions.items():
                    if position['action'] == 'buy':
                        portfolio_value += position['quantity'] * current_price
                    else:
                        portfolio_value -= position['quantity'] * current_price
                
                daily_values.append({
                    'date': current_date,
                    'value': portfolio_value
                })
            
            # Расчет метрик
            if len(daily_values) > 1:
                final_value = daily_values[-1]['value']
                total_return = (final_value - 100000) / 100000
                
                # Расчет дневных доходностей
                daily_returns = []
                for i in range(1, len(daily_values)):
                    daily_return = (daily_values[i]['value'] - daily_values[i-1]['value']) / daily_values[i-1]['value']
                    daily_returns.append(daily_return)
                
                if daily_returns:
                    avg_daily_return = np.mean(daily_returns)
                    daily_volatility = np.std(daily_returns)
                    sharpe_ratio = (avg_daily_return / daily_volatility) * np.sqrt(252) if daily_volatility > 0 else 0
                else:
                    sharpe_ratio = 0
                
                # Расчет максимальной просадки
                peak_value = 100000
                max_drawdown = 0
                for dv in daily_values:
                    if dv['value'] > peak_value:
                        peak_value = dv['value']
                    else:
                        drawdown = (peak_value - dv['value']) / peak_value
                        max_drawdown = max(max_drawdown, drawdown)
                
                # Расчет win rate
                profitable_trades = 0
                for i in range(1, len(trades)):
                    if trades[i]['action'] == 'sell' and trades[i-1]['action'] == 'buy':
                        if trades[i]['price'] > trades[i-1]['price']:
                            profitable_trades += 1
                    elif trades[i]['action'] == 'buy' and trades[i-1]['action'] == 'sell':
                        if trades[i]['price'] < trades[i-1]['price']:
                            profitable_trades += 1
                
                win_rate = profitable_trades / max(1, len(trades) - 1)
                
                return {
                    'total_return': total_return,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'win_rate': win_rate,
                    'total_trades': len(trades),
                    'final_value': final_value
                }
            else:
                return {
                    'total_return': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'win_rate': 0,
                    'total_trades': 0,
                    'final_value': 100000
                }
                
        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'total_trades': 0,
                'final_value': 100000
            }
    
    def objective_function(self, trial, data: pd.DataFrame) -> float:
        """Целевая функция для оптимизации"""
        try:
            # Предложение параметров
            parameters = {
                'leverage': trial.suggest_float('leverage', *self.config['leverage_range']),
                'profit_target': trial.suggest_float('profit_target', *self.config['profit_target_range']),
                'stop_loss': trial.suggest_float('stop_loss', *self.config['stop_loss_range']),
                'trend_threshold': trial.suggest_float('trend_threshold', *self.config['trend_threshold_range']),
                'volatility_threshold': trial.suggest_float('volatility_threshold', *self.config['volatility_threshold_range']),
                'position_size': trial.suggest_float('position_size', *self.config['position_size_range'])
            }
            
            # Бэктестинг
            results = self.backtest_strategy(parameters, data)
            
            # Проверка минимального количества сделок
            if results['total_trades'] < self.config['min_trades']:
                return 0.0
            
            # Возврат метрики для оптимизации
            metric = self.config['optimization_metric']
            return results.get(metric, 0.0)
            
        except Exception as e:
            logger.error(f"Error in objective function: {e}")
            return 0.0
    
    def optimize_parameters(self, data: pd.DataFrame = None) -> Dict:
        """Оптимизация параметров стратегии"""
        logger.info("🔧 Starting parameter optimization")
        
        try:
            # Генерация данных если не предоставлены
            if data is None:
                data = self.generate_test_data(self.config['backtest_period'])
            
            if data.empty:
                logger.error("No data available for optimization")
                return {}
            
            # Создание исследования Optuna
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=42)
            )
            
            # Запуск оптимизации
            study.optimize(
                lambda trial: self.objective_function(trial, data),
                n_trials=self.config['optimization_trials']
            )
            
            # Получение лучших параметров
            best_params = study.best_params
            best_value = study.best_value
            
            # Детальное тестирование лучших параметров
            detailed_results = self.backtest_strategy(best_params, data)
            
            # Сохранение результатов
            optimization_result = {
                'timestamp': datetime.now(),
                'best_parameters': best_params,
                'best_value': best_value,
                'detailed_results': detailed_results,
                'optimization_trials': self.config['optimization_trials'],
                'study_summary': {
                    'n_trials': len(study.trials),
                    'best_trial': study.best_trial.number
                }
            }
            
            self.optimization_history.append(optimization_result)
            self.best_parameters = best_params
            
            # Сохранение в файл
            self._save_optimization_results(optimization_result)
            
            logger.info(f"✅ Optimization completed: {self.config['optimization_metric']} = {best_value:.3f}")
            logger.info(f"Best parameters: {best_params}")
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"Error in optimization: {e}")
            return {}
    
    def _save_optimization_results(self, results: Dict):
        """Сохранение результатов оптимизации"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"strategies/aggressive/optimization_results_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str, ensure_ascii=False)
            
            logger.info(f"💾 Optimization results saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving optimization results: {e}")
    
    def get_optimization_report(self) -> Dict:
        """Получение отчета по оптимизации"""
        if not self.optimization_history:
            return {'error': 'No optimization history available'}
        
        latest_optimization = self.optimization_history[-1]
        
        report = {
            'latest_optimization': latest_optimization,
            'total_optimizations': len(self.optimization_history),
            'best_parameters': self.best_parameters,
            'optimization_trend': self._analyze_optimization_trend()
        }
        
        return report
    
    def _analyze_optimization_trend(self) -> Dict:
        """Анализ тренда оптимизации"""
        if len(self.optimization_history) < 2:
            return {'trend': 'insufficient_data'}
        
        # Анализ улучшения метрики
        values = [opt['best_value'] for opt in self.optimization_history]
        
        if values[-1] > values[0]:
            trend = 'improving'
            improvement = values[-1] - values[0]
        else:
            trend = 'declining'
            improvement = values[-1] - values[0]
        
        return {
            'trend': trend,
            'improvement': improvement,
            'latest_value': values[-1],
            'initial_value': values[0]
        }
    
    def apply_best_parameters(self, config_path: str = None):
        """Применение лучших параметров к конфигурации"""
        if not self.best_parameters:
            logger.warning("No best parameters available")
            return False
        
        try:
            # В реальном проекте здесь будет обновление конфигурационного файла
            logger.info("🔄 Applying best parameters to configuration")
            logger.info(f"New parameters: {self.best_parameters}")
            
            # Сохранение в файл конфигурации
            if config_path:
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(self.best_parameters, f, indent=2, ensure_ascii=False)
            
            logger.info("✅ Best parameters applied successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error applying parameters: {e}")
            return False

def main():
    """Основная функция для тестирования"""
    optimizer = ParameterOptimizer()
    
    # Запуск оптимизации
    logger.info("🧪 Running parameter optimization test")
    results = optimizer.optimize_parameters()
    
    if results:
        logger.info("✅ Optimization test completed successfully")
        
        # Генерация отчета
        report = optimizer.get_optimization_report()
        print("\n📊 OPTIMIZATION REPORT")
        print("="*50)
        print(f"Best {optimizer.config['optimization_metric']}: {results['best_value']:.3f}")
        print(f"Best parameters: {results['best_parameters']}")
        print(f"Total return: {results['detailed_results']['total_return']:.2%}")
        print(f"Sharpe ratio: {results['detailed_results']['sharpe_ratio']:.3f}")
        print(f"Max drawdown: {results['detailed_results']['max_drawdown']:.2%}")
        print(f"Win rate: {results['detailed_results']['win_rate']:.2%}")
        print(f"Total trades: {results['detailed_results']['total_trades']}")
        
    else:
        logger.error("❌ Optimization test failed")

if __name__ == "__main__":
    main()

