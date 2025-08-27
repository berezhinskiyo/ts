#!/usr/bin/env python3
"""
Главный автотрейдер T-Bank с комплексным портфелем стратегий
"""

import asyncio
import sys
import os
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from core.api_client import TBankAPIClient
from core.risk_manager import RiskManager
from data.data_provider import DataProvider
from portfolio_optimizer import PortfolioOptimizer

# Import strategies
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
    RandomForestStrategy, GradientBoostingStrategy, EnsembleStrategy
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'autotrader_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutoTrader:
    """Главный класс автотрейдера"""
    
    def __init__(self):
        self.config = Config()
        self.api_client = None
        self.data_provider = None
        self.risk_manager = RiskManager(Config.INITIAL_CAPITAL)
        self.portfolio_optimizer = PortfolioOptimizer(target_return=0.20)
        
        # Trading state
        self.active_strategies = {}
        self.portfolio_weights = {}
        self.current_positions = {}
        self.trading_enabled = True
        self.last_rebalance = None
        
        # Performance tracking
        self.daily_pnl = []
        self.trade_history = []
        self.performance_metrics = {}
        
        # Monitoring
        self.status = {
            'running': False,
            'last_update': None,
            'total_trades': 0,
            'current_value': Config.INITIAL_CAPITAL,
            'daily_return': 0.0,
            'monthly_return': 0.0
        }
    
    async def initialize(self):
        """Инициализация автотрейдера"""
        try:
            logger.info("🚀 Инициализация автотрейдера T-Bank...")
            
            # Initialize API client
            self.api_client = TBankAPIClient()
            await self.api_client.__aenter__()
            
            # Initialize data provider
            self.data_provider = DataProvider(use_tbank=True)
            await self.data_provider.__aenter__()
            
            # Load optimal portfolio configuration
            await self.load_optimal_portfolio()
            
            # Initialize active strategies
            self.setup_strategies()
            
            logger.info("✅ Автотрейдер успешно инициализирован")
            self.status['running'] = True
            self.status['last_update'] = datetime.now()
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации: {e}")
            raise
    
    async def cleanup(self):
        """Очистка ресурсов"""
        try:
            if self.api_client:
                await self.api_client.__aexit__(None, None, None)
            
            if self.data_provider:
                await self.data_provider.__aexit__(None, None, None)
            
            logger.info("🔧 Ресурсы очищены")
            
        except Exception as e:
            logger.error(f"Ошибка при очистке ресурсов: {e}")
    
    async def load_optimal_portfolio(self):
        """Загрузка оптимальной конфигурации портфеля"""
        try:
            # Load portfolio optimization results
            if os.path.exists('portfolio_analysis.json'):
                with open('portfolio_analysis.json', 'r', encoding='utf-8') as f:
                    portfolio_data = json.load(f)
                
                # Find best combination
                combinations = portfolio_data.get('combinations', [])
                if combinations:
                    best_combo = combinations[0]  # Already sorted by performance
                    self.portfolio_weights = best_combo['weights']
                    
                    logger.info(f"📊 Загружен оптимальный портфель: {best_combo['name']}")
                    logger.info(f"📈 Ожидаемая доходность: {best_combo['performance']['monthly_return']:.2%} в месяц")
                    
                    for strategy, weight in self.portfolio_weights.items():
                        logger.info(f"   - {strategy}: {weight:.1%}")
                else:
                    logger.warning("⚠️ Не найдена оптимальная конфигурация, использую равные веса")
                    self.portfolio_weights = self.get_default_weights()
            else:
                logger.warning("⚠️ Файл portfolio_analysis.json не найден, использую равные веса")
                self.portfolio_weights = self.get_default_weights()
                
        except Exception as e:
            logger.error(f"Ошибка загрузки портфеля: {e}")
            self.portfolio_weights = self.get_default_weights()
    
    def get_default_weights(self) -> Dict[str, float]:
        """Получить веса по умолчанию"""
        strategies = [
            'RSI_Conservative',
            'MA_Crossover_Standard', 
            'MACD_Trend_Following',
            'Mean_Reversion_BB',
            'Momentum_Breakout'
        ]
        
        weight = 1.0 / len(strategies)
        return {strategy: weight for strategy in strategies}
    
    def setup_strategies(self):
        """Настройка активных стратегий"""
        strategy_configs = {
            'RSI_Conservative': RSIStrategy({'rsi_period': 21, 'oversold_threshold': 20, 'overbought_threshold': 80}),
            'RSI_Momentum_Aggressive': RSIStrategy({'rsi_period': 10, 'oversold_threshold': 25, 'overbought_threshold': 75}),
            'MA_Crossover_Standard': MovingAverageCrossoverStrategy({'fast_period': 10, 'slow_period': 30}),
            'MA_Crossover_Fast': MovingAverageCrossoverStrategy({'fast_period': 5, 'slow_period': 15}),
            'MACD_Trend_Following': MACDStrategy({'fast_period': 12, 'slow_period': 26, 'signal_period': 9}),
            'BB_Standard': BollingerBandsStrategy({'period': 20, 'std_dev': 2}),
            'Momentum_Breakout': MomentumStrategy({'lookback_period': 10, 'momentum_threshold': 0.02}),
            'Mean_Reversion_BB': MeanReversionStrategy({'lookback_period': 20, 'deviation_threshold': 2.0}),
            'Volatility_Arbitrage': VolatilityArbitrageStrategy({'volatility_window': 20}),
            'RandomForest_ML': RandomForestStrategy({'n_estimators': 50, 'max_depth': 8}),
            'Ensemble_ML': EnsembleStrategy()
        }
        
        # Activate strategies that are in portfolio weights
        for strategy_name, weight in self.portfolio_weights.items():
            if strategy_name in strategy_configs and weight > 0:
                self.active_strategies[strategy_name] = {
                    'strategy': strategy_configs[strategy_name],
                    'weight': weight,
                    'enabled': True,
                    'last_signal': None,
                    'performance': {'trades': 0, 'pnl': 0.0}
                }
                
                logger.info(f"✅ Активирована стратегия: {strategy_name} (вес: {weight:.1%})")
        
        logger.info(f"📊 Всего активных стратегий: {len(self.active_strategies)}")
    
    async def get_trading_instruments(self) -> List[str]:
        """Получить список инструментов для торговли"""
        # В реальной реализации здесь будет поиск инструментов через API
        return [
            'SBER',  # Сбербанк
            'GAZP',  # Газпром  
            'LKOH',  # ЛУКОЙЛ
            'YNDX',  # Яндекс
            'ROSN',  # Роснефть
            'NVTK',  # НОВАТЭК
            'MTSS',  # МТС
            'MGNT',  # Магнит
        ]
    
    async def execute_trading_cycle(self):
        """Выполнить один цикл торговли"""
        try:
            logger.info("🔄 Начало торгового цикла...")
            
            # Get trading instruments
            instruments = await self.get_trading_instruments()
            
            # Get portfolio info
            portfolio_info = await self.get_portfolio_info()
            
            # Process each instrument
            for instrument in instruments:
                await self.process_instrument(instrument, portfolio_info)
            
            # Update performance metrics
            await self.update_performance_metrics()
            
            # Check if rebalancing is needed
            await self.check_rebalancing()
            
            self.status['last_update'] = datetime.now()
            logger.info("✅ Торговый цикл завершен")
            
        except Exception as e:
            logger.error(f"❌ Ошибка в торговом цикле: {e}")
    
    async def process_instrument(self, instrument: str, portfolio_info: Dict):
        """Обработать один инструмент"""
        try:
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=100)  # 100 days of data
            
            # In production, this would fetch real data from T-Bank API
            # For now, using synthetic data for demonstration
            data = self.data_provider.generate_synthetic_data(
                instrument, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
            )
            
            if data.empty:
                logger.warning(f"⚠️ Нет данных для {instrument}")
                return
            
            current_price = data['close'].iloc[-1]
            
            # Collect signals from all active strategies
            signals = {}
            for strategy_name, strategy_info in self.active_strategies.items():
                if not strategy_info['enabled']:
                    continue
                
                try:
                    strategy = strategy_info['strategy']
                    signal = strategy.generate_signal(data, current_price, portfolio_info)
                    
                    # Weight the signal by strategy weight
                    if signal['confidence'] > 0:
                        signal['weighted_confidence'] = signal['confidence'] * strategy_info['weight']
                        signals[strategy_name] = signal
                        strategy_info['last_signal'] = signal
                    
                except Exception as e:
                    logger.error(f"Ошибка в стратегии {strategy_name} для {instrument}: {e}")
            
            # Aggregate signals
            if signals:
                await self.process_aggregated_signals(instrument, signals, current_price, portfolio_info)
            
        except Exception as e:
            logger.error(f"Ошибка обработки инструмента {instrument}: {e}")
    
    async def process_aggregated_signals(self, instrument: str, signals: Dict, 
                                       current_price: float, portfolio_info: Dict):
        """Обработать агрегированные сигналы"""
        try:
            # Calculate weighted signals
            buy_weight = sum(s['weighted_confidence'] for s in signals.values() if s['action'] == 'buy')
            sell_weight = sum(s['weighted_confidence'] for s in signals.values() if s['action'] == 'sell')
            
            # Determine action
            min_confidence = 0.3  # Minimum confidence threshold
            
            if buy_weight > sell_weight and buy_weight > min_confidence:
                action = 'buy'
                confidence = buy_weight
            elif sell_weight > buy_weight and sell_weight > min_confidence:
                action = 'sell' 
                confidence = sell_weight
            else:
                action = 'hold'
                confidence = 0
            
            # Execute trade if conditions are met
            if action != 'hold' and confidence > min_confidence:
                await self.execute_trade(instrument, action, confidence, current_price, portfolio_info)
            
        except Exception as e:
            logger.error(f"Ошибка обработки сигналов для {instrument}: {e}")
    
    async def execute_trade(self, instrument: str, action: str, confidence: float,
                          current_price: float, portfolio_info: Dict):
        """Выполнить торговую операцию"""
        try:
            # Risk management check
            signal = {'action': action, 'confidence': confidence, 'figi': instrument}
            is_valid, reason = self.risk_manager.validate_trade(signal, portfolio_info, current_price)
            
            if not is_valid:
                logger.debug(f"🚫 Сделка отклонена для {instrument}: {reason}")
                return
            
            # Calculate position size
            stop_loss_price = self.risk_manager.calculate_stop_loss(current_price, action)
            position_size = self.risk_manager.calculate_position_size(
                portfolio_info.get('total_amount', Config.INITIAL_CAPITAL),
                current_price,
                stop_loss_price
            )
            
            if position_size <= 0:
                logger.debug(f"🚫 Нулевой размер позиции для {instrument}")
                return
            
            # In production, this would execute real trade via T-Bank API
            # For now, simulate the trade
            trade_info = {
                'timestamp': datetime.now(),
                'instrument': instrument,
                'action': action,
                'price': current_price,
                'quantity': position_size,
                'confidence': confidence,
                'stop_loss': stop_loss_price,
                'take_profit': self.risk_manager.calculate_take_profit(current_price, action)
            }
            
            self.trade_history.append(trade_info)
            self.status['total_trades'] += 1
            
            logger.info(f"📈 {action.upper()} {instrument}: {position_size} @ {current_price:.2f} "
                       f"(confidence: {confidence:.2f})")
            
        except Exception as e:
            logger.error(f"Ошибка выполнения сделки {instrument}: {e}")
    
    async def get_portfolio_info(self) -> Dict:
        """Получить информацию о портфеле"""
        try:
            # In production, this would fetch real portfolio from T-Bank API
            # For now, return simulated portfolio
            return {
                'total_amount': self.status['current_value'],
                'positions': [
                    {
                        'figi': instrument,
                        'quantity': 100,  # Simulated
                        'current_price': 100,
                    }
                    for instrument in self.current_positions.keys()
                ]
            }
            
        except Exception as e:
            logger.error(f"Ошибка получения портфеля: {e}")
            return {'total_amount': Config.INITIAL_CAPITAL, 'positions': []}
    
    async def update_performance_metrics(self):
        """Обновить метрики производительности"""
        try:
            current_value = self.status['current_value']
            
            # Update risk manager
            self.risk_manager.update_portfolio_metrics(current_value)
            
            # Calculate returns
            if self.daily_pnl:
                yesterday_value = self.daily_pnl[-1]['value'] if self.daily_pnl else Config.INITIAL_CAPITAL
                daily_return = (current_value - yesterday_value) / yesterday_value
                self.status['daily_return'] = daily_return
            
            # Calculate monthly return
            start_of_month_value = Config.INITIAL_CAPITAL  # Simplified
            monthly_return = (current_value - start_of_month_value) / start_of_month_value
            self.status['monthly_return'] = monthly_return
            
            # Add to daily PnL
            self.daily_pnl.append({
                'date': datetime.now().date(),
                'value': current_value,
                'daily_return': self.status['daily_return'],
                'monthly_return': monthly_return
            })
            
            # Keep only last 30 days
            if len(self.daily_pnl) > 30:
                self.daily_pnl = self.daily_pnl[-30:]
            
        except Exception as e:
            logger.error(f"Ошибка обновления метрик: {e}")
    
    async def check_rebalancing(self):
        """Проверить необходимость ребалансировки"""
        try:
            # Rebalance weekly
            if (self.last_rebalance is None or 
                (datetime.now() - self.last_rebalance).days >= 7):
                
                await self.rebalance_portfolio()
                self.last_rebalance = datetime.now()
                
        except Exception as e:
            logger.error(f"Ошибка ребалансировки: {e}")
    
    async def rebalance_portfolio(self):
        """Ребалансировка портфеля"""
        try:
            logger.info("⚖️ Начало ребалансировки портфеля...")
            
            # Reload optimal weights (could be updated based on recent performance)
            await self.load_optimal_portfolio()
            
            # Update strategy weights
            for strategy_name, strategy_info in self.active_strategies.items():
                new_weight = self.portfolio_weights.get(strategy_name, 0)
                old_weight = strategy_info['weight']
                
                if abs(new_weight - old_weight) > 0.05:  # 5% threshold
                    strategy_info['weight'] = new_weight
                    logger.info(f"🔄 Обновлен вес {strategy_name}: {old_weight:.1%} → {new_weight:.1%}")
            
            logger.info("✅ Ребалансировка завершена")
            
        except Exception as e:
            logger.error(f"Ошибка ребалансировки: {e}")
    
    def get_status_report(self) -> str:
        """Получить отчет о состоянии"""
        try:
            current_time = datetime.now()
            
            # Performance summary
            total_return = (self.status['current_value'] - Config.INITIAL_CAPITAL) / Config.INITIAL_CAPITAL
            
            report = f"""
🤖 АВТОТРЕЙДЕР T-BANK - СТАТУС
{'='*50}

⏰ Время: {current_time.strftime('%Y-%m-%d %H:%M:%S')}
🔄 Статус: {'🟢 Активен' if self.status['running'] else '🔴 Остановлен'}

💰 ПРОИЗВОДИТЕЛЬНОСТЬ:
- Текущий капитал: {self.status['current_value']:,.0f} ₽
- Общая доходность: {total_return:.2%}
- Дневная доходность: {self.status['daily_return']:.2%}
- Месячная доходность: {self.status['monthly_return']:.2%}

📊 ТОРГОВАЯ АКТИВНОСТЬ:
- Всего сделок: {self.status['total_trades']}
- Активных стратегий: {len([s for s in self.active_strategies.values() if s['enabled']])}
- Последнее обновление: {self.status['last_update'].strftime('%H:%M:%S') if self.status['last_update'] else 'Никогда'}

🎯 АКТИВНЫЕ СТРАТЕГИИ:
"""
            
            for name, info in self.active_strategies.items():
                if info['enabled']:
                    last_signal = info['last_signal']
                    signal_info = ""
                    if last_signal:
                        signal_info = f"({last_signal['action']}, {last_signal['confidence']:.2f})"
                    
                    report += f"  - {name}: {info['weight']:.1%} {signal_info}\n"
            
            # Recent trades
            if self.trade_history:
                report += f"\n📈 ПОСЛЕДНИЕ СДЕЛКИ:\n"
                for trade in self.trade_history[-5:]:  # Last 5 trades
                    report += f"  - {trade['timestamp'].strftime('%H:%M')} {trade['action'].upper()} " \
                             f"{trade['instrument']} @ {trade['price']:.2f}\n"
            
            report += f"\n⚠️ Риск-менеджмент: {'🟢 ОК' if not self.risk_manager.should_stop_trading()[0] else '🔴 СТОП'}"
            
            return report
            
        except Exception as e:
            return f"Ошибка генерации отчета: {e}"
    
    async def run_forever(self, cycle_interval: int = 300):
        """Запустить автотрейдер в непрерывном режиме"""
        logger.info(f"🚀 Запуск автотрейдера (интервал: {cycle_interval} сек)")
        
        try:
            while self.trading_enabled:
                cycle_start = time.time()
                
                # Execute trading cycle
                await self.execute_trading_cycle()
                
                # Print status
                if self.status['total_trades'] % 10 == 0:  # Every 10 trades
                    print(self.get_status_report())
                
                # Sleep until next cycle
                cycle_duration = time.time() - cycle_start
                sleep_time = max(0, cycle_interval - cycle_duration)
                
                if sleep_time > 0:
                    logger.info(f"😴 Сон {sleep_time:.1f} сек до следующего цикла...")
                    await asyncio.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("⏹️ Получен сигнал остановки...")
            self.trading_enabled = False
        except Exception as e:
            logger.error(f"❌ Критическая ошибка в главном цикле: {e}")
            raise
        finally:
            await self.save_state()
    
    async def save_state(self):
        """Сохранить состояние автотрейдера"""
        try:
            state = {
                'timestamp': datetime.now().isoformat(),
                'status': self.status,
                'portfolio_weights': self.portfolio_weights,
                'daily_pnl': self.daily_pnl[-30:],  # Last 30 days
                'recent_trades': self.trade_history[-100:],  # Last 100 trades
                'performance_metrics': self.risk_manager.get_performance_metrics()
            }
            
            with open(f'autotrader_state_{datetime.now().strftime("%Y%m%d")}.json', 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            logger.info("💾 Состояние автотрейдера сохранено")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения состояния: {e}")

async def main():
    """Главная функция"""
    logger.info("🚀 Запуск автотрейдера T-Bank...")
    
    # Create autotrader
    autotrader = AutoTrader()
    
    try:
        # Initialize
        await autotrader.initialize()
        
        # Show initial status
        print(autotrader.get_status_report())
        
        # Run forever (or until interrupted)
        await autotrader.run_forever(cycle_interval=60)  # 1 minute cycles for demo
        
    except KeyboardInterrupt:
        logger.info("👋 Автотрейдер остановлен пользователем")
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        raise
    finally:
        # Cleanup
        await autotrader.cleanup()
        logger.info("🏁 Автотрейдер завершен")

if __name__ == "__main__":
    # Run the autotrader
    asyncio.run(main())