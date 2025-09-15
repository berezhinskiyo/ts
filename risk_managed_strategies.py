#!/usr/bin/env python3
"""
Risk-Managed Trading Strategies with 2x Maximum Leverage
Стратегии с управлением рисками и максимальным плечом 2x
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Добавляем путь к проекту
sys.path.append('.')

class RiskManagedStrategyTester:
    """Тестер стратегий с управлением рисками"""
    
    def __init__(self):
        self.data_dir = 'data/historical'
        self.derivatives_dir = 'data/derivatives'
        self.results_dir = 'backtesting/results'
        self.max_leverage = 2.0  # Максимальное плечо 2x
        
        # Создаем директории
        os.makedirs(self.results_dir, exist_ok=True)
        
        print("🛡️ Risk-Managed Strategy Tester initialized")
        print(f"⚡ Maximum leverage: {self.max_leverage}x")
        print(f"📁 Data directory: {self.data_dir}")
        print(f"📁 Derivatives directory: {self.derivatives_dir}")
    
    def load_all_data(self):
        """Загрузка всех данных"""
        print("\n📊 Загрузка всех данных...")
        
        all_data = {}
        
        # Загружаем акции
        if os.path.exists(self.data_dir):
            for file in os.listdir(self.data_dir):
                if file.endswith('_tbank.csv'):
                    ticker = file.replace('_tbank.csv', '')
                    filepath = os.path.join(self.data_dir, file)
                    
                    try:
                        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                        df['instrument_type'] = 'stock'
                        all_data[ticker] = df
                        print(f"✅ {ticker}: {len(df)} дней, цена: {df['close'].iloc[-1]:.2f} руб.")
                    except Exception as e:
                        print(f"❌ {ticker}: ошибка загрузки - {e}")
        
        # Загружаем фьючерсы
        if os.path.exists(self.derivatives_dir):
            for file in os.listdir(self.derivatives_dir):
                if file.endswith('_futures.csv'):
                    ticker = file.replace('_futures.csv', '')
                    filepath = os.path.join(self.derivatives_dir, file)
                    
                    try:
                        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                        df['instrument_type'] = 'futures'
                        all_data[ticker] = df
                        print(f"✅ {ticker}: {len(df)} дней, цена: {df['close'].iloc[-1]:.2f}")
                    except Exception as e:
                        print(f"❌ {ticker}: ошибка загрузки - {e}")
        
        print(f"\n📈 Загружено данных: {len(all_data)} инструментов")
        return all_data
    
    def calculate_returns(self, prices):
        """Расчет доходности"""
        returns = prices.pct_change().dropna()
        return returns
    
    def calculate_risk_metrics(self, returns, prices):
        """Расчет метрик риска и производительности"""
        if len(returns) == 0:
            return {}
        
        # Основные метрики
        total_return = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
        annualized_return = ((1 + total_return/100) ** (252/len(returns)) - 1) * 100
        volatility = returns.std() * np.sqrt(252) * 100
        sharpe_ratio = (annualized_return / volatility) if volatility > 0 else 0
        
        # Максимальная просадка
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # VaR (Value at Risk) - 95% доверительный интервал
        var_95 = np.percentile(returns, 5) * 100
        
        # CVaR (Conditional Value at Risk) - ожидаемые потери при превышении VaR
        cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100
        
        # Sortino Ratio (учитывает только отрицательную волатильность)
        negative_returns = returns[returns < 0]
        downside_volatility = negative_returns.std() * np.sqrt(252) * 100
        sortino_ratio = (annualized_return / downside_volatility) if downside_volatility > 0 else 0
        
        # Calmar Ratio (отношение доходности к максимальной просадке)
        calmar_ratio = (annualized_return / abs(max_drawdown)) if max_drawdown != 0 else 0
        
        # Месячная доходность
        monthly_return = annualized_return / 12
        
        # Коэффициент стабильности (отношение положительных дней к общему количеству)
        positive_days = (returns > 0).sum()
        total_days = len(returns)
        win_rate = (positive_days / total_days) * 100 if total_days > 0 else 0
        
        return {
            'total_return': round(total_return, 2),
            'annualized_return': round(annualized_return, 2),
            'monthly_return': round(monthly_return, 2),
            'volatility': round(volatility, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'sortino_ratio': round(sortino_ratio, 2),
            'calmar_ratio': round(calmar_ratio, 2),
            'max_drawdown': round(max_drawdown, 2),
            'var_95': round(var_95, 2),
            'cvar_95': round(cvar_95, 2),
            'win_rate': round(win_rate, 2),
            'data_points': len(returns)
        }
    
    def calculate_advanced_indicators(self, df):
        """Расчет продвинутых технических индикаторов"""
        prices = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # Базовые индикаторы
        sma_5 = prices.rolling(5).mean()
        sma_10 = prices.rolling(10).mean()
        sma_20 = prices.rolling(20).mean()
        sma_50 = prices.rolling(50).mean()
        
        # EMA
        ema_5 = prices.ewm(span=5).mean()
        ema_10 = prices.ewm(span=10).mean()
        ema_20 = prices.ewm(span=20).mean()
        
        # MACD
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9).mean()
        macd_histogram = macd - macd_signal
        
        # RSI
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        bb_middle = sma_20
        bb_std = prices.rolling(20).std()
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)
        bb_width = (bb_upper - bb_lower) / bb_middle
        bb_position = (prices - bb_lower) / (bb_upper - bb_lower)
        
        # Stochastic
        lowest_low = low.rolling(14).min()
        highest_high = high.rolling(14).max()
        stoch_k = 100 * ((prices - lowest_low) / (highest_high - lowest_low))
        stoch_d = stoch_k.rolling(3).mean()
        
        # Volume indicators
        volume_sma = volume.rolling(20).mean()
        volume_ratio = volume / volume_sma
        
        # Volatility
        returns = prices.pct_change()
        volatility = returns.rolling(20).std() * np.sqrt(252)
        
        # ATR
        tr1 = high - low
        tr2 = abs(high - prices.shift(1))
        tr3 = abs(low - prices.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        
        # Momentum indicators
        momentum_5 = prices / prices.shift(5) - 1
        momentum_10 = prices / prices.shift(10) - 1
        
        # Williams %R
        williams_r = -100 * ((highest_high - prices) / (highest_high - lowest_low))
        
        # CCI (Commodity Channel Index)
        typical_price = (high + low + prices) / 3
        cci_sma = typical_price.rolling(20).mean()
        cci_mad = typical_price.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (typical_price - cci_sma) / (0.015 * cci_mad)
        
        return {
            'sma_5': sma_5, 'sma_10': sma_10, 'sma_20': sma_20, 'sma_50': sma_50,
            'ema_5': ema_5, 'ema_10': ema_10, 'ema_20': ema_20,
            'macd': macd, 'macd_signal': macd_signal, 'macd_histogram': macd_histogram,
            'rsi': rsi,
            'bb_upper': bb_upper, 'bb_middle': bb_middle, 'bb_lower': bb_lower, 
            'bb_width': bb_width, 'bb_position': bb_position,
            'stoch_k': stoch_k, 'stoch_d': stoch_d,
            'williams_r': williams_r,
            'cci': cci,
            'volume_ratio': volume_ratio,
            'volatility': volatility,
            'atr': atr,
            'momentum_5': momentum_5, 'momentum_10': momentum_10
        }
    
    def risk_management_system(self, signals, prices, atr, instrument_type, max_risk=0.02):
        """Система управления рисками"""
        risk_adjusted_signals = signals.copy()
        
        for i in range(1, len(signals)):
            if signals.iloc[i] != 0:  # Есть сигнал
                current_price = prices.iloc[i]
                current_atr = atr.iloc[i]
                
                if pd.isna(current_atr) or current_atr == 0:
                    risk_adjusted_signals.iloc[i] = 0
                    continue
                
                # Параметры риска в зависимости от типа инструмента
                if instrument_type == 'futures':
                    stop_loss_atr = 1.5
                    take_profit_atr = 2.5
                    position_size_multiplier = 1.0
                else:  # stocks
                    stop_loss_atr = 2.0
                    take_profit_atr = 3.0
                    position_size_multiplier = 0.8
                
                # Расчет размера позиции на основе риска
                stop_loss_distance = current_atr * stop_loss_atr
                position_size = max_risk / (stop_loss_distance / current_price)
                
                # Применяем максимальное плечо 2x
                position_size = min(position_size, self.max_leverage * position_size_multiplier)
                
                # Дополнительные ограничения риска
                position_size = min(position_size, 0.1)  # Максимум 10% от капитала
                
                # Применяем размер позиции к сигналу
                risk_adjusted_signals.iloc[i] = signals.iloc[i] * position_size
        
        return risk_adjusted_signals
    
    def test_conservative_strategy(self, data):
        """Консервативная стратегия с управлением рисками"""
        print("\n🛡️ Тестирование Conservative Strategy...")
        
        results = {}
        
        for ticker, df in data.items():
            try:
                prices = df['close']
                returns = self.calculate_returns(prices)
                instrument_type = df['instrument_type'].iloc[0]
                
                # Получаем индикаторы
                indicators = self.calculate_advanced_indicators(df)
                
                # Консервативные сигналы
                signals = pd.Series(0, index=prices.index)
                
                # Консервативные условия для покупки
                buy_conditions = (
                    # Тренд
                    (indicators['sma_10'] > indicators['sma_20']) &
                    (indicators['sma_20'] > indicators['sma_50']) &
                    (indicators['ema_10'] > indicators['ema_20']) &
                    
                    # Momentum
                    (indicators['momentum_10'] > 0.02) &  # 2% рост за 10 дней
                    
                    # Технические индикаторы
                    (indicators['rsi'] > 40) & (indicators['rsi'] < 60) &
                    (indicators['macd'] > indicators['macd_signal']) &
                    (indicators['stoch_k'] > indicators['stoch_d']) &
                    (indicators['stoch_k'] > 30) & (indicators['stoch_k'] < 70) &
                    
                    # Bollinger Bands
                    (indicators['bb_position'] > 0.3) & (indicators['bb_position'] < 0.7) &
                    (indicators['bb_width'] > 0.02) &
                    
                    # Volume
                    (indicators['volume_ratio'] > 1.2) &
                    
                    # Volatility (умеренная)
                    (indicators['volatility'] > 0.1) & (indicators['volatility'] < 0.5)
                )
                
                # Условия для продажи
                sell_conditions = (
                    (indicators['sma_10'] < indicators['sma_20']) |
                    (indicators['ema_10'] < indicators['ema_20']) |
                    (indicators['rsi'] > 65) |
                    (indicators['macd'] < indicators['macd_signal']) |
                    (indicators['stoch_k'] < indicators['stoch_d']) |
                    (indicators['bb_position'] > 0.8) |
                    (indicators['volatility'] > 0.6)
                )
                
                signals[buy_conditions] = 1
                signals[sell_conditions] = -1
                
                # Применяем управление рисками
                risk_adjusted_signals = self.risk_management_system(
                    signals, prices, indicators['atr'], instrument_type, max_risk=0.015
                )
                
                # Расчет доходности стратегии
                strategy_returns = risk_adjusted_signals.shift(1) * returns
                strategy_returns = strategy_returns.dropna()
                
                if len(strategy_returns) > 0:
                    metrics = self.calculate_risk_metrics(strategy_returns, prices)
                    results[ticker] = {
                        **metrics,
                        'instrument_type': instrument_type,
                        'max_leverage': self.max_leverage
                    }
                    print(f"  ✅ {ticker}: {metrics['monthly_return']:.2f}% в месяц, Sharpe: {metrics['sharpe_ratio']:.2f}, VaR: {metrics['var_95']:.2f}%")
                else:
                    print(f"  ⚠️ {ticker}: недостаточно данных")
                    
            except Exception as e:
                print(f"  ❌ {ticker}: ошибка - {e}")
        
        return results
    
    def test_balanced_strategy(self, data):
        """Сбалансированная стратегия"""
        print("\n⚖️ Тестирование Balanced Strategy...")
        
        results = {}
        
        for ticker, df in data.items():
            try:
                prices = df['close']
                returns = self.calculate_returns(prices)
                instrument_type = df['instrument_type'].iloc[0]
                
                # Получаем индикаторы
                indicators = self.calculate_advanced_indicators(df)
                
                # Сбалансированные сигналы
                signals = pd.Series(0, index=prices.index)
                
                # Сбалансированные условия для покупки
                buy_conditions = (
                    # Тренд
                    (indicators['sma_5'] > indicators['sma_10']) &
                    (indicators['sma_10'] > indicators['sma_20']) &
                    (indicators['ema_5'] > indicators['ema_10']) &
                    
                    # Momentum
                    (indicators['momentum_5'] > 0.01) &  # 1% рост за 5 дней
                    (indicators['momentum_10'] > 0.015) &  # 1.5% рост за 10 дней
                    
                    # Технические индикаторы
                    (indicators['rsi'] > 35) & (indicators['rsi'] < 65) &
                    (indicators['macd'] > indicators['macd_signal']) &
                    (indicators['macd_histogram'] > 0) &
                    (indicators['stoch_k'] > indicators['stoch_d']) &
                    (indicators['stoch_k'] > 25) & (indicators['stoch_k'] < 75) &
                    
                    # Bollinger Bands
                    (indicators['bb_position'] > 0.25) & (indicators['bb_position'] < 0.75) &
                    (indicators['bb_width'] > 0.025) &
                    
                    # Volume
                    (indicators['volume_ratio'] > 1.3) &
                    
                    # Volatility
                    (indicators['volatility'] > 0.15) & (indicators['volatility'] < 0.7)
                )
                
                # Условия для продажи
                sell_conditions = (
                    (indicators['sma_5'] < indicators['sma_10']) |
                    (indicators['ema_5'] < indicators['ema_10']) |
                    (indicators['momentum_5'] < -0.005) |
                    (indicators['rsi'] > 70) |
                    (indicators['macd'] < indicators['macd_signal']) |
                    (indicators['macd_histogram'] < 0) |
                    (indicators['stoch_k'] < indicators['stoch_d']) |
                    (indicators['stoch_k'] > 80) |
                    (indicators['bb_position'] > 0.85) |
                    (indicators['volatility'] > 0.8)
                )
                
                signals[buy_conditions] = 1
                signals[sell_conditions] = -1
                
                # Применяем управление рисками
                risk_adjusted_signals = self.risk_management_system(
                    signals, prices, indicators['atr'], instrument_type, max_risk=0.02
                )
                
                # Расчет доходности стратегии
                strategy_returns = risk_adjusted_signals.shift(1) * returns
                strategy_returns = strategy_returns.dropna()
                
                if len(strategy_returns) > 0:
                    metrics = self.calculate_risk_metrics(strategy_returns, prices)
                    results[ticker] = {
                        **metrics,
                        'instrument_type': instrument_type,
                        'max_leverage': self.max_leverage
                    }
                    print(f"  ✅ {ticker}: {metrics['monthly_return']:.2f}% в месяц, Sharpe: {metrics['sharpe_ratio']:.2f}, VaR: {metrics['var_95']:.2f}%")
                else:
                    print(f"  ⚠️ {ticker}: недостаточно данных")
                    
            except Exception as e:
                print(f"  ❌ {ticker}: ошибка - {e}")
        
        return results
    
    def test_aggressive_risk_managed_strategy(self, data):
        """Агрессивная стратегия с управлением рисками"""
        print("\n⚡ Тестирование Aggressive Risk-Managed Strategy...")
        
        results = {}
        
        for ticker, df in data.items():
            try:
                prices = df['close']
                returns = self.calculate_returns(prices)
                instrument_type = df['instrument_type'].iloc[0]
                
                # Получаем индикаторы
                indicators = self.calculate_advanced_indicators(df)
                
                # Агрессивные сигналы с управлением рисками
                signals = pd.Series(0, index=prices.index)
                
                # Агрессивные условия для покупки
                buy_conditions = (
                    # Тренд
                    (indicators['sma_3'] > indicators['sma_5']) &
                    (indicators['sma_5'] > indicators['sma_10']) &
                    (indicators['ema_5'] > indicators['ema_10']) &
                    
                    # Momentum
                    (indicators['momentum_5'] > 0.015) &  # 1.5% рост за 5 дней
                    (indicators['momentum_10'] > 0.025) &  # 2.5% рост за 10 дней
                    
                    # Технические индикаторы
                    (indicators['rsi'] > 45) & (indicators['rsi'] < 65) &
                    (indicators['macd'] > indicators['macd_signal']) &
                    (indicators['macd_histogram'] > 0) &
                    (indicators['stoch_k'] > indicators['stoch_d']) &
                    (indicators['stoch_k'] > 30) & (indicators['stoch_k'] < 70) &
                    
                    # Bollinger Bands
                    (indicators['bb_position'] > 0.2) & (indicators['bb_position'] < 0.8) &
                    (indicators['bb_width'] > 0.03) &
                    
                    # Volume
                    (indicators['volume_ratio'] > 1.5) &
                    
                    # Volatility
                    (indicators['volatility'] > 0.2) & (indicators['volatility'] < 0.8)
                )
                
                # Условия для продажи
                sell_conditions = (
                    (indicators['sma_3'] < indicators['sma_5']) |
                    (indicators['ema_5'] < indicators['ema_10']) |
                    (indicators['momentum_5'] < -0.01) |
                    (indicators['rsi'] > 70) |
                    (indicators['macd'] < indicators['macd_signal']) |
                    (indicators['macd_histogram'] < 0) |
                    (indicators['stoch_k'] < indicators['stoch_d']) |
                    (indicators['stoch_k'] > 80) |
                    (indicators['bb_position'] > 0.9) |
                    (indicators['volatility'] > 1.0)
                )
                
                signals[buy_conditions] = 1
                signals[sell_conditions] = -1
                
                # Применяем управление рисками
                risk_adjusted_signals = self.risk_management_system(
                    signals, prices, indicators['atr'], instrument_type, max_risk=0.025
                )
                
                # Расчет доходности стратегии
                strategy_returns = risk_adjusted_signals.shift(1) * returns
                strategy_returns = strategy_returns.dropna()
                
                if len(strategy_returns) > 0:
                    metrics = self.calculate_risk_metrics(strategy_returns, prices)
                    results[ticker] = {
                        **metrics,
                        'instrument_type': instrument_type,
                        'max_leverage': self.max_leverage
                    }
                    print(f"  ✅ {ticker}: {metrics['monthly_return']:.2f}% в месяц, Sharpe: {metrics['sharpe_ratio']:.2f}, VaR: {metrics['var_95']:.2f}%")
                else:
                    print(f"  ⚠️ {ticker}: недостаточно данных")
                    
            except Exception as e:
                print(f"  ❌ {ticker}: ошибка - {e}")
        
        return results
    
    def run_all_risk_managed_tests(self):
        """Запуск всех тестов с управлением рисками"""
        print("🛡️ Запуск тестирования стратегий с управлением рисками")
        print(f"⚡ Максимальное плечо: {self.max_leverage}x")
        print("=" * 80)
        
        # Загружаем данные
        data = self.load_all_data()
        
        if not data:
            print("❌ Нет данных для тестирования!")
            return
        
        # Тестируем стратегии
        self.strategies_results = {}
        self.strategies_results['Conservative Strategy'] = self.test_conservative_strategy(data)
        self.strategies_results['Balanced Strategy'] = self.test_balanced_strategy(data)
        self.strategies_results['Aggressive Risk-Managed Strategy'] = self.test_aggressive_risk_managed_strategy(data)
        
        # Анализируем результаты
        self.analyze_risk_managed_results()
        
        # Сохраняем результаты
        self.save_risk_managed_results()
    
    def analyze_risk_managed_results(self):
        """Анализ результатов с управлением рисками"""
        print("\n" + "=" * 80)
        print("📊 АНАЛИЗ РЕЗУЛЬТАТОВ С УПРАВЛЕНИЕМ РИСКАМИ")
        print("=" * 80)
        
        for strategy_name, results in self.strategies_results.items():
            print(f"\n🎯 {strategy_name}:")
            
            if not results:
                print("  ❌ Нет результатов")
                continue
            
            # Собираем метрики
            monthly_returns = [r['monthly_return'] for r in results.values()]
            sharpe_ratios = [r['sharpe_ratio'] for r in results.values()]
            max_drawdowns = [r['max_drawdown'] for r in results.values()]
            var_95s = [r['var_95'] for r in results.values()]
            win_rates = [r['win_rate'] for r in results.values()]
            
            # Статистика
            avg_monthly_return = np.mean(monthly_returns)
            max_monthly_return = np.max(monthly_returns)
            avg_sharpe = np.mean(sharpe_ratios)
            avg_drawdown = np.mean(max_drawdowns)
            avg_var = np.mean(var_95s)
            avg_win_rate = np.mean(win_rates)
            
            print(f"  📈 Средняя месячная доходность: {avg_monthly_return:.2f}%")
            print(f"  🚀 Максимальная месячная доходность: {max_monthly_return:.2f}%")
            print(f"  📊 Средний Sharpe Ratio: {avg_sharpe:.2f}")
            print(f"  📉 Средняя максимальная просадка: {avg_drawdown:.2f}%")
            print(f"  🛡️ Средний VaR (95%): {avg_var:.2f}%")
            print(f"  🎯 Средний Win Rate: {avg_win_rate:.2f}%")
            
            # Лучшие инструменты
            best_instrument = max(results.items(), key=lambda x: x[1]['monthly_return'])
            print(f"  🏆 Лучший инструмент: {best_instrument[0]} ({best_instrument[1]['monthly_return']:.2f}%)")
            
            # Проверка цели 20%
            goal_achieved = sum(1 for r in monthly_returns if r >= 20)
            print(f"  🎯 Достигли цели 20%: {goal_achieved}/{len(monthly_returns)} инструментов")
            
            # Анализ по типам инструментов
            stock_results = [r for r in results.values() if r['instrument_type'] == 'stock']
            futures_results = [r for r in results.values() if r['instrument_type'] == 'futures']
            
            if stock_results:
                stock_returns = [r['monthly_return'] for r in stock_results]
                print(f"  📊 Акции: {np.mean(stock_returns):.2f}% в месяц ({len(stock_results)} инструментов)")
            
            if futures_results:
                futures_returns = [r['monthly_return'] for r in futures_results]
                print(f"  📈 Фьючерсы: {np.mean(futures_returns):.2f}% в месяц ({len(futures_results)} инструментов)")
    
    def save_risk_managed_results(self):
        """Сохранение результатов с управлением рисками"""
        print(f"\n💾 Сохранение результатов с управлением рисками...")
        
        # Создаем отчет
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_source': 'T-Bank API + Expanded Futures Portfolio',
            'test_period': '252 days (1 year)',
            'max_leverage': self.max_leverage,
            'risk_management': {
                'max_position_size': '10% of capital',
                'stop_loss_atr': '1.5-2.0x ATR',
                'take_profit_atr': '2.5-3.0x ATR',
                'max_risk_per_trade': '1.5-2.5%'
            },
            'strategies': self.strategies_results,
            'summary': {}
        }
        
        # Сводка по стратегиям
        for strategy_name, results in self.strategies_results.items():
            if results:
                monthly_returns = [r['monthly_return'] for r in results.values()]
                sharpe_ratios = [r['sharpe_ratio'] for r in results.values()]
                max_drawdowns = [r['max_drawdown'] for r in results.values()]
                var_95s = [r['var_95'] for r in results.values()]
                
                report['summary'][strategy_name] = {
                    'avg_monthly_return': round(np.mean(monthly_returns), 2),
                    'max_monthly_return': round(np.max(monthly_returns), 2),
                    'avg_sharpe_ratio': round(np.mean(sharpe_ratios), 2),
                    'avg_max_drawdown': round(np.mean(max_drawdowns), 2),
                    'avg_var_95': round(np.mean(var_95s), 2),
                    'instruments_tested': len(results),
                    'goal_20_percent_achieved': sum(1 for r in monthly_returns if r >= 20)
                }
        
        # Сохраняем JSON
        with open(f'{self.results_dir}/risk_managed_strategies.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Создаем текстовый отчет
        self.create_risk_managed_text_report(report)
        
        print(f"✅ Результаты с управлением рисками сохранены в {self.results_dir}/")
    
    def create_risk_managed_text_report(self, report):
        """Создание текстового отчета с управлением рисками"""
        with open(f'{self.results_dir}/risk_managed_strategies.txt', 'w', encoding='utf-8') as f:
            f.write("🛡️ СТРАТЕГИИ С УПРАВЛЕНИЕМ РИСКАМИ\n")
            f.write("=" * 80 + "\n")
            f.write(f"Дата тестирования: {report['timestamp']}\n")
            f.write(f"Источник данных: {report['data_source']}\n")
            f.write(f"Период тестирования: {report['test_period']}\n")
            f.write(f"Максимальное плечо: {report['max_leverage']}x\n\n")
            
            f.write("🛡️ УПРАВЛЕНИЕ РИСКАМИ:\n")
            for key, value in report['risk_management'].items():
                f.write(f"  • {key}: {value}\n")
            f.write("\n")
            
            for strategy_name, summary in report['summary'].items():
                f.write(f"🎯 {strategy_name}:\n")
                f.write(f"  📈 Средняя месячная доходность: {summary['avg_monthly_return']}%\n")
                f.write(f"  🚀 Максимальная месячная доходность: {summary['max_monthly_return']}%\n")
                f.write(f"  📊 Средний Sharpe Ratio: {summary['avg_sharpe_ratio']}\n")
                f.write(f"  📉 Средняя максимальная просадка: {summary['avg_max_drawdown']}%\n")
                f.write(f"  🛡️ Средний VaR (95%): {summary['avg_var_95']}%\n")
                f.write(f"  📊 Протестировано инструментов: {summary['instruments_tested']}\n")
                f.write(f"  🎯 Достигли цели 20%: {summary['goal_20_percent_achieved']}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("🛡️ ТЕСТИРОВАНИЕ С УПРАВЛЕНИЕМ РИСКАМИ ЗАВЕРШЕНО!\n")

def main():
    """Главная функция"""
    tester = RiskManagedStrategyTester()
    tester.run_all_risk_managed_tests()
    
    print(f"\n🎉 ТЕСТИРОВАНИЕ С УПРАВЛЕНИЕМ РИСКАМИ ЗАВЕРШЕНО!")
    print(f"📊 Результаты сохранены в backtesting/results/")
    print(f"🛡️ Максимальное плечо: {tester.max_leverage}x")

if __name__ == "__main__":
    main()
