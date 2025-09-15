#!/usr/bin/env python3
"""
Тестирование ВСЕХ стратегий на реальных данных T-Bank API
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import logging
import sys

# Добавляем пути для импорта стратегий
sys.path.append('strategies')
sys.path.append('strategies/ml')
sys.path.append('strategies/aggressive')
sys.path.append('strategies/combined')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UniversalStrategyTester:
    """Универсальный тестер для всех стратегий"""
    
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.results = {}
        
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
                        if not df.empty and len(df) >= 50:  # Минимум 50 дней
                            key = f"{symbol}_{period}"
                            market_data[key] = df
                            logger.info(f"Загружены данные {key}: {len(df)} дней")
                    except Exception as e:
                        logger.error(f"Ошибка загрузки {filename}: {e}")
        
        return market_data
    
    def test_conservative_strategy(self, symbol, data):
        """Тестирование консервативной стратегии"""
        try:
            current_capital = self.initial_capital
            positions = {}
            trades = []
            equity_history = []
            
            for i in range(2, len(data)):
                current_price = data.iloc[i]['close']
                prev_price = data.iloc[i-1]['close']
                prev2_price = data.iloc[i-2]['close']
                
                # Сигнал на покупку: рост 2 дня подряд
                if prev_price > prev2_price and current_price > prev_price:
                    if symbol not in positions:
                        position_size = (current_capital * 2.0) / current_price
                        positions[symbol] = {
                            'size': position_size,
                            'entry_price': current_price,
                            'entry_date': data.index[i]
                        }
                
                # Сигнал на продажу: падение цены
                elif symbol in positions and current_price < prev_price:
                    position = positions[symbol]
                    exit_price = current_price
                    pnl = (exit_price - position['entry_price']) * position['size']
                    
                    trades.append({
                        'entry_date': position['entry_date'],
                        'exit_date': data.index[i],
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'return_pct': (exit_price / position['entry_price'] - 1) * 100
                    })
                    
                    current_capital += pnl
                    del positions[symbol]
                
                # Записываем текущий капитал
                current_equity = current_capital
                if symbol in positions:
                    position = positions[symbol]
                    unrealized_pnl = (current_price - position['entry_price']) * position['size']
                    current_equity += unrealized_pnl
                
                equity_history.append({
                    'date': data.index[i],
                    'equity': current_equity
                })
            
            return self._calculate_metrics(symbol, equity_history, trades, "Conservative")
            
        except Exception as e:
            logger.error(f"Ошибка в консервативной стратегии для {symbol}: {e}")
            return None
    
    def test_momentum_strategy(self, symbol, data):
        """Тестирование моментум стратегии"""
        try:
            current_capital = self.initial_capital
            positions = {}
            trades = []
            equity_history = []
            
            for i in range(5, len(data)):
                current_price = data.iloc[i]['close']
                
                # Рассчитываем моментум за 5 дней
                momentum_5d = (current_price / data.iloc[i-5]['close'] - 1) * 100
                
                # Сигнал на покупку: сильный моментум
                if momentum_5d > 3 and symbol not in positions:
                    position_size = (current_capital * 2.0) / current_price
                    positions[symbol] = {
                        'size': position_size,
                        'entry_price': current_price,
                        'entry_date': data.index[i]
                    }
                
                # Сигнал на продажу: отрицательный моментум
                elif symbol in positions and momentum_5d < -1:
                    position = positions[symbol]
                    exit_price = current_price
                    pnl = (exit_price - position['entry_price']) * position['size']
                    
                    trades.append({
                        'entry_date': position['entry_date'],
                        'exit_date': data.index[i],
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'return_pct': (exit_price / position['entry_price'] - 1) * 100
                    })
                    
                    current_capital += pnl
                    del positions[symbol]
                
                # Записываем текущий капитал
                current_equity = current_capital
                if symbol in positions:
                    position = positions[symbol]
                    unrealized_pnl = (current_price - position['entry_price']) * position['size']
                    current_equity += unrealized_pnl
                
                equity_history.append({
                    'date': data.index[i],
                    'equity': current_equity
                })
            
            return self._calculate_metrics(symbol, equity_history, trades, "Momentum")
            
        except Exception as e:
            logger.error(f"Ошибка в моментум стратегии для {symbol}: {e}")
            return None
    
    def test_aggressive_strategy(self, symbol, data):
        """Тестирование агрессивной стратегии"""
        try:
            current_capital = self.initial_capital
            positions = {}
            trades = []
            equity_history = []
            
            for i in range(3, len(data)):
                current_price = data.iloc[i]['close']
                
                # Рассчитываем волатильность за 3 дня
                volatility = data.iloc[i-2:i+1]['close'].std() / data.iloc[i-2:i+1]['close'].mean()
                
                # Агрессивная стратегия: покупка при высокой волатильности
                if volatility > 0.02 and symbol not in positions:
                    # Используем плечо 5x для агрессивной стратегии
                    position_size = (current_capital * 5.0) / current_price
                    positions[symbol] = {
                        'size': position_size,
                        'entry_price': current_price,
                        'entry_date': data.index[i]
                    }
                
                # Быстрая продажа при снижении волатильности
                elif symbol in positions and volatility < 0.01:
                    position = positions[symbol]
                    exit_price = current_price
                    pnl = (exit_price - position['entry_price']) * position['size']
                    
                    trades.append({
                        'entry_date': position['entry_date'],
                        'exit_date': data.index[i],
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'return_pct': (exit_price / position['entry_price'] - 1) * 100
                    })
                    
                    current_capital += pnl
                    del positions[symbol]
                
                # Записываем текущий капитал
                current_equity = current_capital
                if symbol in positions:
                    position = positions[symbol]
                    unrealized_pnl = (current_price - position['entry_price']) * position['size']
                    current_equity += unrealized_pnl
                
                equity_history.append({
                    'date': data.index[i],
                    'equity': current_equity
                })
            
            return self._calculate_metrics(symbol, equity_history, trades, "Aggressive")
            
        except Exception as e:
            logger.error(f"Ошибка в агрессивной стратегии для {symbol}: {e}")
            return None
    
    def test_ml_strategy(self, symbol, data):
        """Тестирование ML стратегии (упрощенная версия)"""
        try:
            if len(data) < 50:
                return None
                
            current_capital = self.initial_capital
            positions = {}
            trades = []
            equity_history = []
            
            # Простая ML-подобная стратегия на основе технических индикаторов
            for i in range(20, len(data)):
                current_price = data.iloc[i]['close']
                
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
                if signal_strength >= 2 and symbol not in positions:
                    position_size = (current_capital * 2.0) / current_price
                    positions[symbol] = {
                        'size': position_size,
                        'entry_price': current_price,
                        'entry_date': data.index[i]
                    }
                
                # Продажа при слабом сигнале
                elif symbol in positions and signal_strength <= 1:
                    position = positions[symbol]
                    exit_price = current_price
                    pnl = (exit_price - position['entry_price']) * position['size']
                    
                    trades.append({
                        'entry_date': position['entry_date'],
                        'exit_date': data.index[i],
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'return_pct': (exit_price / position['entry_price'] - 1) * 100
                    })
                    
                    current_capital += pnl
                    del positions[symbol]
                
                # Записываем текущий капитал
                current_equity = current_capital
                if symbol in positions:
                    position = positions[symbol]
                    unrealized_pnl = (current_price - position['entry_price']) * position['size']
                    current_equity += unrealized_pnl
                
                equity_history.append({
                    'date': data.index[i],
                    'equity': current_equity
                })
            
            return self._calculate_metrics(symbol, equity_history, trades, "ML")
            
        except Exception as e:
            logger.error(f"Ошибка в ML стратегии для {symbol}: {e}")
            return None
    
    def _calculate_rsi(self, prices, window=14):
        """Расчет RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not rsi.empty else 50
    
    def _calculate_metrics(self, symbol, equity_history, trades, strategy_name):
        """Расчет метрик"""
        try:
            if not equity_history:
                return None
                
            equity_curve = pd.DataFrame(equity_history)
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
            rolling_max = equity_curve['equity'].expanding().max()
            drawdown = (equity_curve['equity'] - rolling_max) / rolling_max
            max_drawdown = drawdown.min() * 100
            
            # Win rate
            if trades:
                winning_trades = [t for t in trades if t['pnl'] > 0]
                win_rate = len(winning_trades) / len(trades) * 100
            else:
                win_rate = 0
            
            return {
                'symbol': symbol,
                'strategy': strategy_name,
                'monthly_return': float(monthly_return),
                'total_return': float(total_return * 100),
                'volatility': float(volatility),
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown),
                'win_rate': float(win_rate),
                'total_trades': int(len(trades)),
                'period_start': equity_curve.index[0].strftime('%Y-%m-%d'),
                'period_end': equity_curve.index[-1].strftime('%Y-%m-%d'),
                'period_days': int((equity_curve.index[-1] - equity_curve.index[0]).days),
                'final_equity': float(equity_curve['equity'].iloc[-1])
            }
            
        except Exception as e:
            logger.error(f"Ошибка расчета метрик для {symbol}: {e}")
            return None
    
    def run_all_tests(self):
        """Запуск всех тестов"""
        logger.info("🚀 ТЕСТИРОВАНИЕ ВСЕХ СТРАТЕГИЙ НА РЕАЛЬНЫХ ДАННЫХ T-BANK")
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
            logger.info(f"\n📈 Тестирование {data_key}...")
            
            results = {}
            
            # Тестируем все стратегии
            strategies = [
                ('Conservative', self.test_conservative_strategy),
                ('Momentum', self.test_momentum_strategy),
                ('Aggressive', self.test_aggressive_strategy),
                ('ML', self.test_ml_strategy)
            ]
            
            for strategy_name, strategy_func in strategies:
                logger.info(f"  🔍 {strategy_name} стратегия...")
                result = strategy_func(symbol, data)
                if result:
                    results[strategy_name] = result
                    logger.info(f"    ✅ {result['monthly_return']:.2f}% в месяц")
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
        json_filename = os.path.join(output_dir, f'all_strategies_test_{timestamp}.json')
        txt_filename = os.path.join(output_dir, f'all_strategies_test_{timestamp}.txt')
        
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Создаем текстовый отчет
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write("🚀 ПОЛНЫЙ ОТЧЕТ ПО ТЕСТИРОВАНИЮ ВСЕХ СТРАТЕГИЙ\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Дата тестирования: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for data_key, data_results in results.items():
                f.write(f"📊 {data_key.upper()}\n")
                f.write("-" * 40 + "\n")
                
                for strategy_name, strategy_result in data_results.items():
                    f.write(f"{strategy_name} стратегия:\n")
                    f.write(f"  • Доходность в месяц: {strategy_result['monthly_return']:.2f}%\n")
                    f.write(f"  • Общая доходность: {strategy_result['total_return']:.2f}%\n")
                    f.write(f"  • Волатильность: {strategy_result['volatility']:.2f}%\n")
                    f.write(f"  • Sharpe Ratio: {strategy_result['sharpe_ratio']:.2f}\n")
                    f.write(f"  • Макс. просадка: {strategy_result['max_drawdown']:.2f}%\n")
                    f.write(f"  • Win Rate: {strategy_result['win_rate']:.1f}%\n")
                    f.write(f"  • Сделок: {strategy_result['total_trades']}\n")
                    f.write(f"  • Период: {strategy_result['period_start']} - {strategy_result['period_end']}\n\n")
        
        logger.info(f"✅ Результаты сохранены в {json_filename} и {txt_filename}")
    
    def _print_summary(self, results):
        """Вывод сводки"""
        logger.info(f"\n📊 СВОДКА РЕЗУЛЬТАТОВ:")
        
        # Собираем все результаты
        all_strategy_results = []
        for data_key, data_results in results.items():
            for strategy_name, strategy_result in data_results.items():
                all_strategy_results.append({
                    'data_key': data_key,
                    'strategy': strategy_name,
                    'monthly_return': strategy_result['monthly_return'],
                    'sharpe_ratio': strategy_result['sharpe_ratio']
                })
        
        # Сортируем по доходности
        all_strategy_results.sort(key=lambda x: x['monthly_return'], reverse=True)
        
        logger.info(f"\n🏆 ТОП-10 РЕЗУЛЬТАТОВ:")
        for i, result in enumerate(all_strategy_results[:10]):
            logger.info(f"  {i+1}. {result['data_key']} - {result['strategy']}: {result['monthly_return']:.2f}% в месяц (Sharpe: {result['sharpe_ratio']:.2f})")

def main():
    """Основная функция"""
    tester = UniversalStrategyTester()
    results = tester.run_all_tests()
    
    if results:
        logger.info(f"\n🎯 ТЕСТИРОВАНИЕ ВСЕХ СТРАТЕГИЙ ЗАВЕРШЕНО УСПЕШНО!")
    else:
        logger.info(f"\n❌ ТЕСТИРОВАНИЕ НЕ УДАЛОСЬ")

if __name__ == "__main__":
    main()

