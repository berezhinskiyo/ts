#!/usr/bin/env python3
"""
Тестирование стратегий на реальных данных T-Bank API
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConservativeStrategy:
    """Консервативная стратегия с ограниченным риском"""
    
    def __init__(self, initial_capital=100000, max_leverage=2.0):
        self.initial_capital = initial_capital
        self.max_leverage = max_leverage
        self.equity_history = []
        self.closed_trades = []
        self.current_capital = initial_capital
        self.positions = {}
        
    def run_backtest(self, symbol, data):
        """Запуск бэктеста"""
        try:
            if data.empty:
                logger.warning(f"Нет данных для {symbol}")
                return None
                
            logger.info(f"Тестируем {symbol} на {len(data)} днях данных")
            
            # Сбрасываем состояние
            self.current_capital = self.initial_capital
            self.equity_history = []
            self.closed_trades = []
            self.positions = {}
            
            # Простая стратегия: покупаем при росте цены на 2 дня подряд
            for i in range(2, len(data)):
                current_price = data.iloc[i]['close']
                prev_price = data.iloc[i-1]['close']
                prev2_price = data.iloc[i-2]['close']
                
                # Сигнал на покупку: рост 2 дня подряд
                if prev_price > prev2_price and current_price > prev_price:
                    if symbol not in self.positions:
                        # Покупаем с плечом 2x
                        position_size = (self.current_capital * self.max_leverage) / current_price
                        self.positions[symbol] = {
                            'size': position_size,
                            'entry_price': current_price,
                            'entry_date': data.index[i]
                        }
                        logger.debug(f"Покупка {symbol} по цене {current_price}")
                
                # Сигнал на продажу: падение цены
                elif symbol in self.positions and current_price < prev_price:
                    position = self.positions[symbol]
                    exit_price = current_price
                    pnl = (exit_price - position['entry_price']) * position['size']
                    
                    self.closed_trades.append({
                        'symbol': symbol,
                        'entry_date': position['entry_date'],
                        'exit_date': data.index[i],
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'size': position['size'],
                        'pnl': pnl,
                        'return_pct': (exit_price / position['entry_price'] - 1) * 100
                    })
                    
                    self.current_capital += pnl
                    del self.positions[symbol]
                    logger.debug(f"Продажа {symbol} по цене {exit_price}, PnL: {pnl:.2f}")
                
                # Записываем текущий капитал
                current_equity = self.current_capital
                if symbol in self.positions:
                    position = self.positions[symbol]
                    unrealized_pnl = (current_price - position['entry_price']) * position['size']
                    current_equity += unrealized_pnl
                
                self.equity_history.append({
                    'date': data.index[i],
                    'equity': current_equity,
                    'price': current_price
                })
            
            return self._calculate_results(symbol)
            
        except Exception as e:
            logger.error(f"Ошибка в стратегии для {symbol}: {e}")
            return None
    
    def _calculate_results(self, symbol):
        """Расчет результатов"""
        try:
            if not self.equity_history:
                return None
                
            equity_curve = pd.DataFrame(self.equity_history)
            equity_curve.set_index('date', inplace=True)
            
            # Основные метрики
            total_return = (equity_curve['equity'].iloc[-1] / self.initial_capital) - 1
            
            # Месячная доходность
            monthly_returns = equity_curve['equity'].resample('M').last().pct_change().dropna()
            monthly_return = monthly_returns.mean() * 100 if not monthly_returns.empty else 0
            
            # Волатильность
            volatility = equity_curve['equity'].pct_change().std() * np.sqrt(252) * 100
            
            # Sharpe ratio (предполагаем безрисковую ставку 5%)
            risk_free_rate = 0.05
            excess_returns = equity_curve['equity'].pct_change().mean() * 252 - risk_free_rate
            sharpe_ratio = excess_returns / (equity_curve['equity'].pct_change().std() * np.sqrt(252)) if equity_curve['equity'].pct_change().std() > 0 else 0
            
            # Максимальная просадка
            rolling_max = equity_curve['equity'].expanding().max()
            drawdown = (equity_curve['equity'] - rolling_max) / rolling_max
            max_drawdown = drawdown.min() * 100
            
            # Win rate
            if self.closed_trades:
                winning_trades = [t for t in self.closed_trades if t['pnl'] > 0]
                win_rate = len(winning_trades) / len(self.closed_trades) * 100
            else:
                win_rate = 0
            
            return {
                'symbol': symbol,
                'monthly_return': float(monthly_return),
                'total_return': float(total_return * 100),
                'volatility': float(volatility),
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown),
                'win_rate': float(win_rate),
                'total_trades': int(len(self.closed_trades)),
                'period_start': equity_curve.index[0].strftime('%Y-%m-%d'),
                'period_end': equity_curve.index[-1].strftime('%Y-%m-%d'),
                'period_days': int((equity_curve.index[-1] - equity_curve.index[0]).days),
                'final_equity': float(equity_curve['equity'].iloc[-1])
            }
            
        except Exception as e:
            logger.error(f"Ошибка расчета результатов для {symbol}: {e}")
            return None

class MomentumStrategy:
    """Моментум стратегия"""
    
    def __init__(self, initial_capital=100000, max_leverage=2.0):
        self.initial_capital = initial_capital
        self.max_leverage = max_leverage
        self.equity_history = []
        self.closed_trades = []
        self.current_capital = initial_capital
        self.positions = {}
        
    def run_backtest(self, symbol, data):
        """Запуск бэктеста"""
        try:
            if data.empty:
                return None
                
            # Сбрасываем состояние
            self.current_capital = self.initial_capital
            self.equity_history = []
            self.closed_trades = []
            self.positions = {}
            
            # Моментум стратегия: покупаем при сильном росте
            for i in range(5, len(data)):
                current_price = data.iloc[i]['close']
                
                # Рассчитываем моментум за 5 дней
                momentum_5d = (current_price / data.iloc[i-5]['close'] - 1) * 100
                
                # Сигнал на покупку: сильный моментум
                if momentum_5d > 3 and symbol not in self.positions:
                    position_size = (self.current_capital * self.max_leverage) / current_price
                    self.positions[symbol] = {
                        'size': position_size,
                        'entry_price': current_price,
                        'entry_date': data.index[i]
                    }
                
                # Сигнал на продажу: отрицательный моментум
                elif symbol in self.positions and momentum_5d < -1:
                    position = self.positions[symbol]
                    exit_price = current_price
                    pnl = (exit_price - position['entry_price']) * position['size']
                    
                    self.closed_trades.append({
                        'symbol': symbol,
                        'entry_date': position['entry_date'],
                        'exit_date': data.index[i],
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'size': position['size'],
                        'pnl': pnl,
                        'return_pct': (exit_price / position['entry_price'] - 1) * 100
                    })
                    
                    self.current_capital += pnl
                    del self.positions[symbol]
                
                # Записываем текущий капитал
                current_equity = self.current_capital
                if symbol in self.positions:
                    position = self.positions[symbol]
                    unrealized_pnl = (current_price - position['entry_price']) * position['size']
                    current_equity += unrealized_pnl
                
                self.equity_history.append({
                    'date': data.index[i],
                    'equity': current_equity,
                    'price': current_price
                })
            
            return self._calculate_results(symbol)
            
        except Exception as e:
            logger.error(f"Ошибка в моментум стратегии для {symbol}: {e}")
            return None
    
    def _calculate_results(self, symbol):
        """Расчет результатов (аналогично ConservativeStrategy)"""
        try:
            if not self.equity_history:
                return None
                
            equity_curve = pd.DataFrame(self.equity_history)
            equity_curve.set_index('date', inplace=True)
            
            total_return = (equity_curve['equity'].iloc[-1] / self.initial_capital) - 1
            monthly_returns = equity_curve['equity'].resample('M').last().pct_change().dropna()
            monthly_return = monthly_returns.mean() * 100 if not monthly_returns.empty else 0
            volatility = equity_curve['equity'].pct_change().std() * np.sqrt(252) * 100
            
            risk_free_rate = 0.05
            excess_returns = equity_curve['equity'].pct_change().mean() * 252 - risk_free_rate
            sharpe_ratio = excess_returns / (equity_curve['equity'].pct_change().std() * np.sqrt(252)) if equity_curve['equity'].pct_change().std() > 0 else 0
            
            rolling_max = equity_curve['equity'].expanding().max()
            drawdown = (equity_curve['equity'] - rolling_max) / rolling_max
            max_drawdown = drawdown.min() * 100
            
            if self.closed_trades:
                winning_trades = [t for t in self.closed_trades if t['pnl'] > 0]
                win_rate = len(winning_trades) / len(self.closed_trades) * 100
            else:
                win_rate = 0
            
            return {
                'symbol': symbol,
                'monthly_return': float(monthly_return),
                'total_return': float(total_return * 100),
                'volatility': float(volatility),
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown),
                'win_rate': float(win_rate),
                'total_trades': int(len(self.closed_trades)),
                'period_start': equity_curve.index[0].strftime('%Y-%m-%d'),
                'period_end': equity_curve.index[-1].strftime('%Y-%m-%d'),
                'period_days': int((equity_curve.index[-1] - equity_curve.index[0]).days),
                'final_equity': float(equity_curve['equity'].iloc[-1])
            }
            
        except Exception as e:
            logger.error(f"Ошибка расчета результатов для {symbol}: {e}")
            return None

def load_tbank_data():
    """Загрузка данных T-Bank"""
    data_dir = 'data/tbank_real'
    market_data = {}
    
    if not os.path.exists(data_dir):
        logger.error(f"Директория {data_dir} не найдена")
        return market_data
    
    # Загружаем все CSV файлы
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
                    if not df.empty:
                        key = f"{symbol}_{period}"
                        market_data[key] = df
                        logger.info(f"Загружены данные {key}: {len(df)} дней, период {df.index[0].date()} - {df.index[-1].date()}")
                except Exception as e:
                    logger.error(f"Ошибка загрузки {filename}: {e}")
    
    return market_data

def main():
    """Основная функция"""
    logger.info("🚀 ТЕСТИРОВАНИЕ СТРАТЕГИЙ НА РЕАЛЬНЫХ ДАННЫХ T-BANK")
    logger.info("=" * 60)
    
    # Загружаем данные
    logger.info("📊 Загрузка реальных данных T-Bank...")
    market_data = load_tbank_data()
    
    if not market_data:
        logger.error("❌ Нет данных для тестирования")
        return
    
    logger.info(f"✅ Загружено {len(market_data)} наборов данных")
    
    # Инициализируем стратегии
    conservative_strategy = ConservativeStrategy()
    momentum_strategy = MomentumStrategy()
    
    all_results = {}
    
    # Тестируем на каждом наборе данных
    for data_key, data in market_data.items():
        logger.info(f"\n📈 Тестирование {data_key}...")
        
        symbol = data_key.split('_')[0]
        
        # Тестируем консервативную стратегию
        logger.info(f"  🔍 Консервативная стратегия...")
        conservative_results = conservative_strategy.run_backtest(symbol, data)
        
        # Тестируем моментум стратегию
        logger.info(f"  🔍 Моментум стратегия...")
        momentum_results = momentum_strategy.run_backtest(symbol, data)
        
        if conservative_results or momentum_results:
            all_results[data_key] = {
                'conservative': conservative_results,
                'momentum': momentum_results
            }
            
            if conservative_results:
                logger.info(f"    ✅ Консервативная: {conservative_results['monthly_return']:.2f}% в месяц")
            if momentum_results:
                logger.info(f"    ✅ Моментум: {momentum_results['monthly_return']:.2f}% в месяц")
    
    # Сохраняем результаты
    output_dir = 'backtesting/results'
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    json_filename = os.path.join(output_dir, f'tbank_real_test_{timestamp}.json')
    txt_filename = os.path.join(output_dir, f'tbank_real_test_{timestamp}.txt')
    
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # Создаем текстовый отчет
    with open(txt_filename, 'w', encoding='utf-8') as f:
        f.write("🚀 ОТЧЕТ ПО ТЕСТИРОВАНИЮ СТРАТЕГИЙ НА РЕАЛЬНЫХ ДАННЫХ T-BANK\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Дата тестирования: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Количество наборов данных: {len(market_data)}\n\n")
        
        for data_key, results in all_results.items():
            f.write(f"📊 {data_key.upper()}\n")
            f.write("-" * 40 + "\n")
            
            if results['conservative']:
                c = results['conservative']
                f.write(f"Консервативная стратегия:\n")
                f.write(f"  • Доходность в месяц: {c['monthly_return']:.2f}%\n")
                f.write(f"  • Общая доходность: {c['total_return']:.2f}%\n")
                f.write(f"  • Волатильность: {c['volatility']:.2f}%\n")
                f.write(f"  • Sharpe Ratio: {c['sharpe_ratio']:.2f}\n")
                f.write(f"  • Макс. просадка: {c['max_drawdown']:.2f}%\n")
                f.write(f"  • Win Rate: {c['win_rate']:.1f}%\n")
                f.write(f"  • Сделок: {c['total_trades']}\n")
                f.write(f"  • Период: {c['period_start']} - {c['period_end']}\n\n")
            
            if results['momentum']:
                m = results['momentum']
                f.write(f"Моментум стратегия:\n")
                f.write(f"  • Доходность в месяц: {m['monthly_return']:.2f}%\n")
                f.write(f"  • Общая доходность: {m['total_return']:.2f}%\n")
                f.write(f"  • Волатильность: {m['volatility']:.2f}%\n")
                f.write(f"  • Sharpe Ratio: {m['sharpe_ratio']:.2f}\n")
                f.write(f"  • Макс. просадка: {m['max_drawdown']:.2f}%\n")
                f.write(f"  • Win Rate: {m['win_rate']:.1f}%\n")
                f.write(f"  • Сделок: {m['total_trades']}\n")
                f.write(f"  • Период: {m['period_start']} - {m['period_end']}\n\n")
    
    logger.info(f"✅ Результаты сохранены в {json_filename} и {txt_filename}")
    
    # Выводим сводку
    logger.info(f"\n📊 СВОДКА РЕЗУЛЬТАТОВ:")
    for data_key, results in all_results.items():
        logger.info(f"\n🎯 {data_key.upper()}:")
        if results['conservative']:
            logger.info(f"  Консервативная: {results['conservative']['monthly_return']:.2f}% в месяц")
        if results['momentum']:
            logger.info(f"  Моментум: {results['momentum']['monthly_return']:.2f}% в месяц")

if __name__ == "__main__":
    main()
