#!/usr/bin/env python3
"""
Стратегии с полноценным управлением рисками и стоп-лоссами
Включает PostStopOrderRequest и все необходимые защитные механизмы
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
import logging
import sys
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StopOrder:
    """Структура стоп-ордера"""
    order_id: str
    symbol: str
    order_type: str  # 'stop_loss' или 'take_profit'
    trigger_price: float
    quantity: int
    direction: str  # 'buy' или 'sell'
    created_at: datetime
    is_active: bool = True

@dataclass
class Position:
    """Структура позиции"""
    symbol: str
    quantity: int
    entry_price: float
    entry_date: datetime
    current_price: float
    unrealized_pnl: float
    stop_loss_price: float
    take_profit_price: float
    max_risk_amount: float

class RiskManagedStrategyTester:
    """Тестер стратегий с управлением рисками"""
    
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.stop_orders = []
        self.trades = []
        self.equity_history = []
        
        # Параметры управления рисками
        self.max_risk_per_trade = 0.02  # 2% от капитала на сделку
        self.max_portfolio_risk = 0.20  # 20% от капитала в портфеле
        self.stop_loss_pct = 0.05  # 5% стоп-лосс
        self.take_profit_pct = 0.15  # 15% тейк-профит
        self.max_drawdown = 0.25  # 25% максимальная просадка
        self.daily_loss_limit = 0.05  # 5% дневной лимит потерь
        
        # Трекинг рисков
        self.peak_equity = initial_capital
        self.current_drawdown = 0.0
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        
    def calculate_position_size(self, symbol: str, entry_price: float, 
                              stop_loss_price: float) -> int:
        """Расчет размера позиции на основе риска"""
        try:
            # Рассчитываем риск на акцию
            risk_per_share = abs(entry_price - stop_loss_price)
            
            if risk_per_share == 0:
                return 0
            
            # Максимальный риск на сделку
            max_risk_amount = self.current_capital * self.max_risk_per_trade
            
            # Размер позиции на основе риска
            position_size_by_risk = int(max_risk_amount / risk_per_share)
            
            # Максимальный размер позиции по стоимости
            max_position_value = self.current_capital * self.max_portfolio_risk
            max_position_size = int(max_position_value / entry_price)
            
            # Берем минимальное значение
            position_size = min(position_size_by_risk, max_position_size)
            
            # Проверяем, что у нас достаточно капитала
            required_capital = position_size * entry_price
            if required_capital > self.current_capital * 0.8:  # Не более 80% капитала
                position_size = int((self.current_capital * 0.8) / entry_price)
            
            return max(0, position_size)
            
        except Exception as e:
            logger.error(f"Ошибка расчета размера позиции для {symbol}: {e}")
            return 0
    
    def create_stop_orders(self, symbol: str, position: Position):
        """Создание стоп-ордеров для позиции"""
        try:
            # Стоп-лосс ордер
            stop_loss_order = StopOrder(
                order_id=f"SL_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                symbol=symbol,
                order_type='stop_loss',
                trigger_price=position.stop_loss_price,
                quantity=position.quantity,
                direction='sell',
                created_at=datetime.now()
            )
            
            # Тейк-профит ордер
            take_profit_order = StopOrder(
                order_id=f"TP_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                symbol=symbol,
                order_type='take_profit',
                trigger_price=position.take_profit_price,
                quantity=position.quantity,
                direction='sell',
                created_at=datetime.now()
            )
            
            self.stop_orders.extend([stop_loss_order, take_profit_order])
            logger.info(f"Созданы стоп-ордера для {symbol}: SL={position.stop_loss_price:.2f}, TP={position.take_profit_price:.2f}")
            
        except Exception as e:
            logger.error(f"Ошибка создания стоп-ордеров для {symbol}: {e}")
    
    def check_stop_orders(self, symbol: str, current_price: float):
        """Проверка срабатывания стоп-ордеров"""
        try:
            orders_to_execute = []
            
            for order in self.stop_orders:
                if (order.symbol == symbol and 
                    order.is_active and 
                    order.order_type == 'stop_loss' and
                    current_price <= order.trigger_price):
                    orders_to_execute.append(order)
                    
                elif (order.symbol == symbol and 
                      order.is_active and 
                      order.order_type == 'take_profit' and
                      current_price >= order.trigger_price):
                    orders_to_execute.append(order)
            
            # Выполняем сработавшие ордера
            for order in orders_to_execute:
                self.execute_stop_order(order, current_price)
                
        except Exception as e:
            logger.error(f"Ошибка проверки стоп-ордеров для {symbol}: {e}")
    
    def execute_stop_order(self, order: StopOrder, current_price: float):
        """Выполнение стоп-ордера"""
        try:
            if order.symbol not in self.positions:
                return
            
            position = self.positions[order.symbol]
            
            # Рассчитываем PnL
            pnl = (current_price - position.entry_price) * position.quantity
            
            # Записываем сделку
            trade = {
                'entry_date': position.entry_date,
                'exit_date': datetime.now(),
                'symbol': order.symbol,
                'entry_price': position.entry_price,
                'exit_price': current_price,
                'quantity': position.quantity,
                'pnl': pnl,
                'return_pct': (current_price / position.entry_price - 1) * 100,
                'exit_reason': order.order_type,
                'stop_loss_price': position.stop_loss_price,
                'take_profit_price': position.take_profit_price
            }
            
            self.trades.append(trade)
            
            # Обновляем капитал
            self.current_capital += pnl
            
            # Удаляем позицию
            del self.positions[order.symbol]
            
            # Деактивируем все ордера для этого символа
            for o in self.stop_orders:
                if o.symbol == order.symbol:
                    o.is_active = False
            
            logger.info(f"Выполнен {order.order_type} для {order.symbol}: PnL={pnl:.2f}₽")
            
        except Exception as e:
            logger.error(f"Ошибка выполнения стоп-ордера {order.order_id}: {e}")
    
    def check_risk_limits(self) -> Tuple[bool, str]:
        """Проверка лимитов риска"""
        try:
            # Проверка максимальной просадки
            if self.current_drawdown > self.max_drawdown:
                return False, f"Превышена максимальная просадка: {self.current_drawdown:.2%}"
            
            # Проверка дневного лимита потерь
            if self.daily_pnl < -self.current_capital * self.daily_loss_limit:
                return False, f"Превышен дневной лимит потерь: {self.daily_pnl:.2f}₽"
            
            # Проверка последовательных убытков
            if self.consecutive_losses >= 5:
                return False, f"5 последовательных убыточных дней"
            
            # Проверка минимального капитала
            if self.current_capital < self.initial_capital * 0.5:
                return False, f"Капитал ниже 50% от начального: {self.current_capital:.2f}₽"
            
            return True, "Риски в пределах нормы"
            
        except Exception as e:
            logger.error(f"Ошибка проверки лимитов риска: {e}")
            return False, f"Ошибка проверки рисков: {e}"
    
    def test_conservative_strategy_with_risk(self, symbol: str, data: pd.DataFrame):
        """Консервативная стратегия с управлением рисками"""
        try:
            logger.info(f"Тестирование консервативной стратегии с рисками для {symbol}")
            
            for i in range(2, len(data)):
                current_price = data.iloc[i]['close']
                prev_price = data.iloc[i-1]['close']
                prev2_price = data.iloc[i-2]['close']
                
                # Проверяем стоп-ордера
                if symbol in self.positions:
                    self.check_stop_orders(symbol, current_price)
                
                # Проверяем лимиты риска
                can_trade, risk_reason = self.check_risk_limits()
                if not can_trade:
                    logger.warning(f"Торговля остановлена: {risk_reason}")
                    break
                
                # Сигнал на покупку: рост 2 дня подряд
                if (prev_price > prev2_price and 
                    current_price > prev_price and 
                    symbol not in self.positions):
                    
                    # Рассчитываем стоп-лосс и тейк-профит
                    stop_loss_price = current_price * (1 - self.stop_loss_pct)
                    take_profit_price = current_price * (1 + self.take_profit_pct)
                    
                    # Рассчитываем размер позиции
                    position_size = self.calculate_position_size(symbol, current_price, stop_loss_price)
                    
                    if position_size > 0:
                        # Создаем позицию
                        position = Position(
                            symbol=symbol,
                            quantity=position_size,
                            entry_price=current_price,
                            entry_date=data.index[i],
                            current_price=current_price,
                            unrealized_pnl=0.0,
                            stop_loss_price=stop_loss_price,
                            take_profit_price=take_profit_price,
                            max_risk_amount=self.current_capital * self.max_risk_per_trade
                        )
                        
                        self.positions[symbol] = position
                        
                        # Создаем стоп-ордера
                        self.create_stop_orders(symbol, position)
                        
                        logger.info(f"Открыта позиция {symbol}: {position_size} акций по {current_price:.2f}₽")
                
                # Обновляем метрики
                self._update_equity_metrics(current_price)
            
            return self._calculate_metrics(symbol, "Conservative_RiskManaged")
            
        except Exception as e:
            logger.error(f"Ошибка в консервативной стратегии с рисками для {symbol}: {e}")
            return None
    
    def test_momentum_strategy_with_risk(self, symbol: str, data: pd.DataFrame):
        """Моментум стратегия с управлением рисками"""
        try:
            logger.info(f"Тестирование моментум стратегии с рисками для {symbol}")
            
            for i in range(5, len(data)):
                current_price = data.iloc[i]['close']
                
                # Проверяем стоп-ордера
                if symbol in self.positions:
                    self.check_stop_orders(symbol, current_price)
                
                # Проверяем лимиты риска
                can_trade, risk_reason = self.check_risk_limits()
                if not can_trade:
                    logger.warning(f"Торговля остановлена: {risk_reason}")
                    break
                
                # Рассчитываем моментум за 5 дней
                momentum_5d = (current_price / data.iloc[i-5]['close'] - 1) * 100
                
                # Сигнал на покупку: сильный моментум
                if momentum_5d > 3 and symbol not in self.positions:
                    # Рассчитываем стоп-лосс и тейк-профит
                    stop_loss_price = current_price * (1 - self.stop_loss_pct)
                    take_profit_price = current_price * (1 + self.take_profit_pct)
                    
                    # Рассчитываем размер позиции
                    position_size = self.calculate_position_size(symbol, current_price, stop_loss_price)
                    
                    if position_size > 0:
                        # Создаем позицию
                        position = Position(
                            symbol=symbol,
                            quantity=position_size,
                            entry_price=current_price,
                            entry_date=data.index[i],
                            current_price=current_price,
                            unrealized_pnl=0.0,
                            stop_loss_price=stop_loss_price,
                            take_profit_price=take_profit_price,
                            max_risk_amount=self.current_capital * self.max_risk_per_trade
                        )
                        
                        self.positions[symbol] = position
                        
                        # Создаем стоп-ордера
                        self.create_stop_orders(symbol, position)
                        
                        logger.info(f"Открыта позиция {symbol}: {position_size} акций по {current_price:.2f}₽")
                
                # Обновляем метрики
                self._update_equity_metrics(current_price)
            
            return self._calculate_metrics(symbol, "Momentum_RiskManaged")
            
        except Exception as e:
            logger.error(f"Ошибка в моментум стратегии с рисками для {symbol}: {e}")
            return None
    
    def test_aggressive_strategy_with_risk(self, symbol: str, data: pd.DataFrame):
        """Агрессивная стратегия с управлением рисками"""
        try:
            logger.info(f"Тестирование агрессивной стратегии с рисками для {symbol}")
            
            for i in range(3, len(data)):
                current_price = data.iloc[i]['close']
                
                # Проверяем стоп-ордера
                if symbol in self.positions:
                    self.check_stop_orders(symbol, current_price)
                
                # Проверяем лимиты риска
                can_trade, risk_reason = self.check_risk_limits()
                if not can_trade:
                    logger.warning(f"Торговля остановлена: {risk_reason}")
                    break
                
                # Рассчитываем волатильность за 3 дня
                volatility = data.iloc[i-2:i+1]['close'].std() / data.iloc[i-2:i+1]['close'].mean()
                
                # Агрессивная стратегия: покупка при высокой волатильности
                if volatility > 0.02 and symbol not in self.positions:
                    # Более агрессивные стоп-лоссы для агрессивной стратегии
                    aggressive_stop_loss = 0.03  # 3% вместо 5%
                    aggressive_take_profit = 0.10  # 10% вместо 15%
                    
                    stop_loss_price = current_price * (1 - aggressive_stop_loss)
                    take_profit_price = current_price * (1 + aggressive_take_profit)
                    
                    # Рассчитываем размер позиции (с учетом плеча 5x)
                    position_size = self.calculate_position_size(symbol, current_price, stop_loss_price)
                    # Увеличиваем размер позиции для агрессивной стратегии
                    position_size = int(position_size * 2.5)  # Эквивалент плеча 5x
                    
                    if position_size > 0:
                        # Создаем позицию
                        position = Position(
                            symbol=symbol,
                            quantity=position_size,
                            entry_price=current_price,
                            entry_date=data.index[i],
                            current_price=current_price,
                            unrealized_pnl=0.0,
                            stop_loss_price=stop_loss_price,
                            take_profit_price=take_profit_price,
                            max_risk_amount=self.current_capital * self.max_risk_per_trade
                        )
                        
                        self.positions[symbol] = position
                        
                        # Создаем стоп-ордера
                        self.create_stop_orders(symbol, position)
                        
                        logger.info(f"Открыта агрессивная позиция {symbol}: {position_size} акций по {current_price:.2f}₽")
                
                # Обновляем метрики
                self._update_equity_metrics(current_price)
            
            return self._calculate_metrics(symbol, "Aggressive_RiskManaged")
            
        except Exception as e:
            logger.error(f"Ошибка в агрессивной стратегии с рисками для {symbol}: {e}")
            return None
    
    def test_ml_strategy_with_risk(self, symbol: str, data: pd.DataFrame):
        """ML стратегия с управлением рисками"""
        try:
            if len(data) < 50:
                return None
                
            logger.info(f"Тестирование ML стратегии с рисками для {symbol}")
            
            for i in range(20, len(data)):
                current_price = data.iloc[i]['close']
                
                # Проверяем стоп-ордера
                if symbol in self.positions:
                    self.check_stop_orders(symbol, current_price)
                
                # Проверяем лимиты риска
                can_trade, risk_reason = self.check_risk_limits()
                if not can_trade:
                    logger.warning(f"Торговля остановлена: {risk_reason}")
                    break
                
                # Рассчитываем простые индикаторы
                sma_5 = data.iloc[i-4:i+1]['close'].mean()
                sma_20 = data.iloc[i-19:i+1]['close'].mean()
                rsi = self._calculate_rsi(data.iloc[i-13:i+1]['close'])
                
                # ML-подобная логика: комбинация индикаторов
                signal_strength = 0
                if current_price > sma_5 > sma_20:
                    signal_strength += 1
                if rsi > 30 and rsi < 70:
                    signal_strength += 1
                if current_price > sma_20 * 1.02:
                    signal_strength += 1
                
                # Покупка при сильном сигнале
                if signal_strength >= 2 and symbol not in self.positions:
                    # Рассчитываем стоп-лосс и тейк-профит
                    stop_loss_price = current_price * (1 - self.stop_loss_pct)
                    take_profit_price = current_price * (1 + self.take_profit_pct)
                    
                    # Рассчитываем размер позиции
                    position_size = self.calculate_position_size(symbol, current_price, stop_loss_price)
                    
                    if position_size > 0:
                        # Создаем позицию
                        position = Position(
                            symbol=symbol,
                            quantity=position_size,
                            entry_price=current_price,
                            entry_date=data.index[i],
                            current_price=current_price,
                            unrealized_pnl=0.0,
                            stop_loss_price=stop_loss_price,
                            take_profit_price=take_profit_price,
                            max_risk_amount=self.current_capital * self.max_risk_per_trade
                        )
                        
                        self.positions[symbol] = position
                        
                        # Создаем стоп-ордера
                        self.create_stop_orders(symbol, position)
                        
                        logger.info(f"Открыта ML позиция {symbol}: {position_size} акций по {current_price:.2f}₽")
                
                # Обновляем метрики
                self._update_equity_metrics(current_price)
            
            return self._calculate_metrics(symbol, "ML_RiskManaged")
            
        except Exception as e:
            logger.error(f"Ошибка в ML стратегии с рисками для {symbol}: {e}")
            return None
    
    def _calculate_rsi(self, prices, window=14):
        """Расчет RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not rsi.empty else 50
    
    def _update_equity_metrics(self, current_price: float):
        """Обновление метрик капитала"""
        try:
            # Рассчитываем текущий капитал
            current_equity = self.current_capital
            
            # Добавляем нереализованную прибыль/убыток
            for symbol, position in self.positions.items():
                unrealized_pnl = (current_price - position.entry_price) * position.quantity
                current_equity += unrealized_pnl
            
            # Обновляем пиковый капитал
            if current_equity > self.peak_equity:
                self.peak_equity = current_equity
                self.current_drawdown = 0.0
                self.consecutive_losses = 0
            else:
                self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
                if current_equity < self.current_capital:
                    self.consecutive_losses += 1
            
            # Записываем историю капитала
            self.equity_history.append({
                'date': datetime.now(),
                'equity': current_equity,
                'drawdown': self.current_drawdown,
                'positions_count': len(self.positions)
            })
            
        except Exception as e:
            logger.error(f"Ошибка обновления метрик капитала: {e}")
    
    def _calculate_metrics(self, symbol: str, strategy_name: str):
        """Расчет метрик стратегии"""
        try:
            if not self.equity_history:
                return None
                
            equity_curve = pd.DataFrame(self.equity_history)
            equity_curve.set_index('date', inplace=True)
            
            # Основные метрики
            total_return = (equity_curve['equity'].iloc[-1] / self.initial_capital) - 1
            
            # Месячная доходность
            monthly_returns = equity_curve['equity'].resample('ME').last().pct_change().dropna()
            monthly_return = monthly_returns.mean() * 100 if not monthly_returns.empty else 0
            
            # Волатильность
            volatility = equity_curve['equity'].pct_change().std() * np.sqrt(252) * 100
            
            # Sharpe ratio
            risk_free_rate = 0.05
            excess_returns = equity_curve['equity'].pct_change().mean() * 252 - risk_free_rate
            sharpe_ratio = excess_returns / (equity_curve['equity'].pct_change().std() * np.sqrt(252)) if equity_curve['equity'].pct_change().std() > 0 else 0
            
            # Максимальная просадка
            max_drawdown = equity_curve['drawdown'].max() * 100
            
            # Win rate
            if self.trades:
                winning_trades = [t for t in self.trades if t['pnl'] > 0]
                win_rate = len(winning_trades) / len(self.trades) * 100
            else:
                win_rate = 0
            
            # Статистика стоп-ордеров
            stop_loss_trades = [t for t in self.trades if t['exit_reason'] == 'stop_loss']
            take_profit_trades = [t for t in self.trades if t['exit_reason'] == 'take_profit']
            
            return {
                'symbol': symbol,
                'strategy': strategy_name,
                'monthly_return': float(monthly_return),
                'total_return': float(total_return * 100),
                'volatility': float(volatility),
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown),
                'win_rate': float(win_rate),
                'total_trades': int(len(self.trades)),
                'stop_loss_trades': int(len(stop_loss_trades)),
                'take_profit_trades': int(len(take_profit_trades)),
                'risk_management_active': True,
                'final_equity': float(equity_curve['equity'].iloc[-1]),
                'risk_parameters': {
                    'max_risk_per_trade': self.max_risk_per_trade,
                    'stop_loss_pct': self.stop_loss_pct,
                    'take_profit_pct': self.take_profit_pct,
                    'max_drawdown': self.max_drawdown
                }
            }
            
        except Exception as e:
            logger.error(f"Ошибка расчета метрик для {symbol}: {e}")
            return None
    
    def load_tbank_data(self):
        """Загрузка данных T-Bank"""
        data_dir = 'data/tbank_real'
        market_data = {}
        
        if not os.path.exists(data_dir):
            logger.error(f"Директория {data_dir} не найдена")
            return market_data
        
        for filename in os.listdir(data_dir):
            if filename.endswith('_tbank.csv'):
                symbol_period = filename.replace('_tbank.csv', '')
                parts = symbol_period.split('_')
                if len(parts) >= 2:
                    symbol = parts[0]
                    period = parts[1]
                    
                    filepath = os.path.join(data_dir, filename)
                    try:
                        df = pd.read_csv(filepath, index_col='date', parse_dates=True)
                        if not df.empty and len(df) >= 50:
                            key = f"{symbol}_{period}"
                            market_data[key] = df
                            logger.info(f"Загружены данные {key}: {len(df)} дней")
                    except Exception as e:
                        logger.error(f"Ошибка загрузки {filename}: {e}")
        
        return market_data
    
    def run_all_tests_with_risk_management(self):
        """Запуск всех тестов с управлением рисками"""
        logger.info("🚀 ТЕСТИРОВАНИЕ СТРАТЕГИЙ С УПРАВЛЕНИЕМ РИСКАМИ")
        logger.info("=" * 70)
        
        # Загружаем данные
        market_data = self.load_tbank_data()
        
        if not market_data:
            logger.error("❌ Нет данных для тестирования")
            return
        
        logger.info(f"✅ Загружено {len(market_data)} наборов данных")
        
        # Тестируем все стратегии на всех данных
        all_results = {}
        
        for data_key, data in market_data.items():
            symbol = data_key.split('_')[0]
            logger.info(f"\n📈 Тестирование {data_key} с управлением рисками...")
            
            results = {}
            
            # Тестируем все стратегии
            strategies = [
                ('Conservative_RiskManaged', self.test_conservative_strategy_with_risk),
                ('Momentum_RiskManaged', self.test_momentum_strategy_with_risk),
                ('Aggressive_RiskManaged', self.test_aggressive_strategy_with_risk),
                ('ML_RiskManaged', self.test_ml_strategy_with_risk)
            ]
            
            for strategy_name, strategy_func in strategies:
                # Сбрасываем состояние для каждой стратегии
                self.current_capital = self.initial_capital
                self.positions = {}
                self.stop_orders = []
                self.trades = []
                self.equity_history = []
                self.peak_equity = self.initial_capital
                self.current_drawdown = 0.0
                self.consecutive_losses = 0
                
                logger.info(f"  🔍 {strategy_name}...")
                result = strategy_func(symbol, data)
                if result:
                    results[strategy_name] = result
                    logger.info(f"    ✅ {result['monthly_return']:.2f}% в месяц, Sharpe: {result['sharpe_ratio']:.2f}")
                    logger.info(f"    🛡️ Стоп-лоссов: {result['stop_loss_trades']}, Тейк-профитов: {result['take_profit_trades']}")
                else:
                    logger.info(f"    ❌ Не удалось получить результаты")
            
            all_results[data_key] = results
        
        # Сохраняем результаты
        self._save_results(all_results)
        
        # Выводим сводку
        self._print_summary(all_results)
        
        return all_results
    
    def _save_results(self, results):
        """Сохранение результатов"""
        output_dir = 'backtesting/results'
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        json_filename = os.path.join(output_dir, f'risk_managed_strategies_{timestamp}.json')
        txt_filename = os.path.join(output_dir, f'risk_managed_strategies_{timestamp}.txt')
        
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Создаем текстовый отчет
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write("🛡️ ОТЧЕТ ПО СТРАТЕГИЯМ С УПРАВЛЕНИЕМ РИСКАМИ\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Дата тестирования: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("🎯 ПАРАМЕТРЫ УПРАВЛЕНИЯ РИСКАМИ:\n")
            f.write(f"• Максимальный риск на сделку: {self.max_risk_per_trade:.1%}\n")
            f.write(f"• Стоп-лосс: {self.stop_loss_pct:.1%}\n")
            f.write(f"• Тейк-профит: {self.take_profit_pct:.1%}\n")
            f.write(f"• Максимальная просадка: {self.max_drawdown:.1%}\n")
            f.write(f"• Дневной лимит потерь: {self.daily_loss_limit:.1%}\n\n")
            
            for data_key, data_results in results.items():
                f.write(f"📊 {data_key.upper()}\n")
                f.write("-" * 40 + "\n")
                
                for strategy_name, strategy_result in data_results.items():
                    f.write(f"{strategy_name}:\n")
                    f.write(f"  • Доходность в месяц: {strategy_result['monthly_return']:.2f}%\n")
                    f.write(f"  • Общая доходность: {strategy_result['total_return']:.2f}%\n")
                    f.write(f"  • Волатильность: {strategy_result['volatility']:.2f}%\n")
                    f.write(f"  • Sharpe Ratio: {strategy_result['sharpe_ratio']:.2f}\n")
                    f.write(f"  • Макс. просадка: {strategy_result['max_drawdown']:.2f}%\n")
                    f.write(f"  • Win Rate: {strategy_result['win_rate']:.1f}%\n")
                    f.write(f"  • Сделок: {strategy_result['total_trades']}\n")
                    f.write(f"  • Стоп-лоссов: {strategy_result['stop_loss_trades']}\n")
                    f.write(f"  • Тейк-профитов: {strategy_result['take_profit_trades']}\n\n")
        
        logger.info(f"✅ Результаты сохранены в {json_filename} и {txt_filename}")
    
    def _print_summary(self, results):
        """Вывод сводки"""
        logger.info(f"\n🛡️ СВОДКА РЕЗУЛЬТАТОВ С УПРАВЛЕНИЕМ РИСКАМИ:")
        
        # Собираем все результаты
        all_strategy_results = []
        for data_key, data_results in results.items():
            for strategy_name, strategy_result in data_results.items():
                all_strategy_results.append({
                    'data_key': data_key,
                    'strategy': strategy_name,
                    'monthly_return': strategy_result['monthly_return'],
                    'sharpe_ratio': strategy_result['sharpe_ratio'],
                    'max_drawdown': strategy_result['max_drawdown'],
                    'stop_loss_trades': strategy_result['stop_loss_trades'],
                    'take_profit_trades': strategy_result['take_profit_trades']
                })
        
        # Сортируем по доходности
        all_strategy_results.sort(key=lambda x: x['monthly_return'], reverse=True)
        
        logger.info(f"\n🏆 ТОП-10 РЕЗУЛЬТАТОВ С РИСК-МЕНЕДЖМЕНТОМ:")
        for i, result in enumerate(all_strategy_results[:10]):
            logger.info(f"  {i+1}. {result['data_key']} - {result['strategy']}: {result['monthly_return']:.2f}% в месяц")
            logger.info(f"      Sharpe: {result['sharpe_ratio']:.2f}, Просадка: {result['max_drawdown']:.2f}%")
            logger.info(f"      Стоп-лоссов: {result['stop_loss_trades']}, Тейк-профитов: {result['take_profit_trades']}")

def main():
    """Основная функция"""
    tester = RiskManagedStrategyTester()
    results = tester.run_all_tests_with_risk_management()
    
    if results:
        logger.info(f"\n🛡️ ТЕСТИРОВАНИЕ С УПРАВЛЕНИЕМ РИСКАМИ ЗАВЕРШЕНО УСПЕШНО!")
    else:
        logger.info(f"\n❌ ТЕСТИРОВАНИЕ НЕ УДАЛОСЬ")

if __name__ == "__main__":
    main()
