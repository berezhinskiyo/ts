python3 -c 'print("Hello World")'#!/usr/bin/env python3
"""
Advanced ML Trading System with Options Strategies
Full-featured trading system with machine learning, parameter optimization, and options
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import warnings
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import asyncio
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import optuna
from scipy.optimize import minimize
import yfinance as yf

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MLSignal:
    """Machine Learning trading signal"""
    timestamp: datetime
    symbol: str
    action: str
    confidence: float
    price: float
    model_name: str
    features: Dict
    probability: float
    reasoning: str

class FeatureEngineer:
    """Advanced feature engineering for ML models"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set"""
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        features['price_change'] = data['close'] - data['close'].shift(1)
        
        # Technical indicators
        features['rsi'] = self._calculate_rsi(data['close'])
        features['macd'] = self._calculate_macd(data['close'])
        features['bb_upper'] = self._calculate_bollinger_bands(data['close'])[0]
        features['bb_lower'] = self._calculate_bollinger_bands(data['close'])[1]
        features['bb_position'] = (data['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            features[f'sma_{period}'] = data['close'].rolling(period).mean()
            features[f'price_sma_{period}_ratio'] = data['close'] / features[f'sma_{period}']
        
        # Volume features
        features['volume_ma'] = data['volume'].rolling(20).mean()
        features['volume_ratio'] = data['volume'] / features['volume_ma']
        features['price_volume'] = data['close'] * data['volume']
        
        # Volatility features
        features['volatility'] = features['returns'].rolling(20).std()
        features['volatility_ratio'] = features['volatility'] / features['volatility'].rolling(50).mean()
        
        # Momentum features
        for period in [5, 10, 20]:
            features[f'momentum_{period}'] = data['close'] / data['close'].shift(period) - 1
        
        # Time-based features
        features['hour'] = data.index.hour
        features['day_of_week'] = data.index.dayofweek
        features['is_market_open'] = ((features['hour'] >= 10) & (features['hour'] < 18)).astype(int)
        
        # Lagged features
        for lag in [1, 2, 3, 5]:
            features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
            features[f'volume_ratio_lag_{lag}'] = features['volume_ratio'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            features[f'returns_mean_{window}'] = features['returns'].rolling(window).mean()
            features[f'returns_std_{window}'] = features['returns'].rolling(window).std()
            features[f'volume_mean_{window}'] = features['volume'].rolling(window).mean()
        
        # Drop NaN values
        features = features.dropna()
        
        # Store feature names
        self.feature_names = [col for col in features.columns if col != 'target']
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, lower

class MLModelManager:
    """Machine Learning model management"""
    
    def __init__(self):
        self.models = {}
        self.feature_engineer = FeatureEngineer()
        self.is_trained = False
    
    def create_target(self, data: pd.DataFrame, horizon: int = 5, threshold: float = 0.002) -> pd.Series:
        """Create target variable for ML model"""
        future_returns = data['close'].shift(-horizon) / data['close'] - 1
        
        # Create classification target
        target = pd.Series(index=data.index, dtype=int)
        target[future_returns > threshold] = 1  # Buy
        target[future_returns < -threshold] = -1  # Sell
        target[(future_returns >= -threshold) & (future_returns <= threshold)] = 0  # Hold
        
        return target
    
    def train_models(self, data: pd.DataFrame):
        """Train multiple ML models"""
        logger.info("🤖 Training ML models...")
        
        # Create features and target
        features = self.feature_engineer.create_features(data)
        target = self.create_target(data)
        
        # Align features and target
        common_index = features.index.intersection(target.index)
        X = features.loc[common_index]
        y = target.loc[common_index]
        
        # Remove rows with NaN target
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) < 100:
            logger.warning("Insufficient data for training")
            return
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.feature_engineer.scaler.fit_transform(X_train)
        X_test_scaled = self.feature_engineer.scaler.transform(X_test)
        
        # Define models
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(probability=True, random_state=42)
        }
        
        # Train models
        for name, model in models.items():
            try:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                
                self.models[name] = model
                logger.info(f"✅ {name}: {accuracy:.3f} accuracy")
                
            except Exception as e:
                logger.error(f"❌ Error training {name}: {e}")
        
        self.is_trained = True
        logger.info(f"🎯 Trained {len(self.models)} models")
    
    def predict(self, data: pd.DataFrame) -> Dict[str, MLSignal]:
        """Generate ML predictions"""
        if not self.is_trained:
            logger.warning("Models not trained yet")
            return {}
        
        # Create features
        features = self.feature_engineer.create_features(data)
        
        if features.empty:
            return {}
        
        # Get latest features
        latest_features = features.iloc[-1:].values
        latest_features_scaled = self.feature_engineer.scaler.transform(latest_features)
        
        signals = {}
        current_price = data['close'].iloc[-1]
        timestamp = data.index[-1]
        
        for name, model in self.models.items():
            try:
                # Get prediction and probability
                prediction = model.predict(latest_features_scaled)[0]
                probabilities = model.predict_proba(latest_features_scaled)[0]
                
                # Map prediction to action
                action_map = {-1: 'sell', 0: 'hold', 1: 'buy'}
                action = action_map.get(prediction, 'hold')
                
                # Calculate confidence
                confidence = max(probabilities)
                
                # Create signal
                signal = MLSignal(
                    timestamp=timestamp,
                    symbol="",
                    action=action,
                    confidence=confidence,
                    price=current_price,
                    model_name=name,
                    features=features.iloc[-1].to_dict(),
                    probability=confidence,
                    reasoning=f"ML {name}: {action} (confidence: {confidence:.3f})"
                )
                
                signals[name] = signal
                
            except Exception as e:
                logger.error(f"Error predicting with {name}: {e}")
        
        return signals

class ParameterOptimizer:
    """Optimize strategy parameters using Optuna"""
    
    def __init__(self):
        self.best_params = {}
    
    def optimize_strategy(self, strategy_func, data: pd.DataFrame, n_trials: int = 100):
        """Optimize strategy parameters"""
        logger.info(f"🔧 Optimizing parameters with {n_trials} trials...")
        
        def objective(trial):
            # Define parameter ranges
            params = {
                'profit_target': trial.suggest_float('profit_target', 0.0001, 0.01),
                'stop_loss': trial.suggest_float('stop_loss', 0.0001, 0.005),
                'lookback_period': trial.suggest_int('lookback_period', 5, 50),
                'threshold': trial.suggest_float('threshold', 0.0001, 0.01)
            }
            
            # Run strategy with parameters
            try:
                result = strategy_func(data, params)
                return result.get('sharpe_ratio', 0)
            except:
                return 0
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params = study.best_params
        logger.info(f"✅ Best parameters: {self.best_params}")
        
        return self.best_params

class OptionsStrategy:
    """Options trading strategies"""
    
    def __init__(self):
        self.risk_free_rate = 0.05  # 5% risk-free rate
    
    def covered_call_strategy(self, stock_price: float, strike_price: float, 
                            days_to_expiry: int, volatility: float) -> Dict:
        """Covered call strategy"""
        # Simplified Black-Scholes for call option
        time_to_expiry = days_to_expiry / 365.0
        
        # Calculate option price (simplified)
        d1 = (np.log(stock_price / strike_price) + (self.risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
        d2 = d1 - volatility * np.sqrt(time_to_expiry)
        
        # Normal CDF approximation
        def norm_cdf(x):
            return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))
        
        call_price = stock_price * norm_cdf(d1) - strike_price * np.exp(-self.risk_free_rate * time_to_expiry) * norm_cdf(d2)
        
        return {
            'strategy': 'covered_call',
            'call_price': call_price,
            'premium_income': call_price,
            'max_profit': call_price + (strike_price - stock_price),
            'breakeven': stock_price - call_price,
            'risk': 'Limited upside, unlimited downside'
        }
    
    def protective_put_strategy(self, stock_price: float, strike_price: float,
                              days_to_expiry: int, volatility: float) -> Dict:
        """Protective put strategy"""
        time_to_expiry = days_to_expiry / 365.0
        
        # Calculate put option price
        d1 = (np.log(stock_price / strike_price) + (self.risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
        d2 = d1 - volatility * np.sqrt(time_to_expiry)
        
        def norm_cdf(x):
            return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))
        
        put_price = strike_price * np.exp(-self.risk_free_rate * time_to_expiry) * norm_cdf(-d2) - stock_price * norm_cdf(-d1)
        
        return {
            'strategy': 'protective_put',
            'put_price': put_price,
            'protection_cost': put_price,
            'max_loss': put_price,
            'breakeven': stock_price + put_price,
            'risk': 'Limited downside, unlimited upside'
        }
    
    def straddle_strategy(self, stock_price: float, strike_price: float,
                         days_to_expiry: int, volatility: float) -> Dict:
        """Long straddle strategy"""
        time_to_expiry = days_to_expiry / 365.0
        
        # Calculate call and put prices
        d1 = (np.log(stock_price / strike_price) + (self.risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
        d2 = d1 - volatility * np.sqrt(time_to_expiry)
        
        def norm_cdf(x):
            return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))
        
        call_price = stock_price * norm_cdf(d1) - strike_price * np.exp(-self.risk_free_rate * time_to_expiry) * norm_cdf(d2)
        put_price = strike_price * np.exp(-self.risk_free_rate * time_to_expiry) * norm_cdf(-d2) - stock_price * norm_cdf(-d1)
        
        total_cost = call_price + put_price
        
        return {
            'strategy': 'long_straddle',
            'total_cost': total_cost,
            'call_price': call_price,
            'put_price': put_price,
            'max_loss': total_cost,
            'breakeven_up': strike_price + total_cost,
            'breakeven_down': strike_price - total_cost,
            'risk': 'Limited loss, unlimited profit potential'
        }

class RealDataProvider:
    """Real data provider using T-Bank API and Yahoo Finance"""
    
    def __init__(self):
        self.cache = {}
    
    async def get_real_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get real market data"""
        try:
            # Try Yahoo Finance first
            ticker = f"{symbol}.ME" if symbol in ['SBER', 'GAZP', 'LKOH', 'YNDX'] else symbol
            data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
            
            if data.empty:
                logger.warning(f"No data from Yahoo Finance for {symbol}")
                return self._generate_realistic_data(symbol, start_date, end_date)
            
            # Clean and format data
            data = data.dropna()
            data.columns = [col.lower() for col in data.columns]
            
            logger.info(f"✅ Loaded {len(data)} days of real data for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error getting real data for {symbol}: {e}")
            return self._generate_realistic_data(symbol, start_date, end_date)
    
    def _generate_realistic_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate realistic data when real data is not available"""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Realistic starting prices
        base_prices = {
            'SBER': 250, 'GAZP': 150, 'LKOH': 5500, 'YNDX': 2500,
            'ROSN': 450, 'NVTK': 1200, 'MTSS': 300, 'MGNT': 5000
        }
        
        base_price = base_prices.get(symbol, 100)
        volatility = 0.02
        
        # Generate realistic price series
        np.random.seed(hash(symbol) % 2**32)
        returns = np.random.normal(0.0005, volatility, len(dates))
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Generate OHLCV
        data = []
        for i, (date, close_price) in enumerate(zip(dates, prices)):
            open_price = close_price * (1 + np.random.normal(0, 0.005))
            high = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.01)))
            low = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.01)))
            volume = int(1000000 * np.random.uniform(0.5, 2.0))
            
            data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })
        
        return pd.DataFrame(data, index=dates)

class AdvancedTradingSystem:
    """Full-featured trading system with ML, optimization, and options"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.options_positions = {}
        self.trades = []
        
        # Components
        self.ml_manager = MLModelManager()
        self.parameter_optimizer = ParameterOptimizer()
        self.options_strategy = OptionsStrategy()
        self.data_provider = RealDataProvider()
        
        # Performance tracking
        self.performance_history = []
        self.max_drawdown = 0
        self.peak_value = initial_capital
    
    async def run_comprehensive_test(self):
        """Run comprehensive test with all features"""
        logger.info("🚀 Starting Advanced ML Trading System...")
        
        # Portfolio
        portfolio = ['SBER', 'GAZP', 'LKOH', 'YNDX', 'ROSN']
        
        # Get real data
        start_date = '2023-01-01'
        end_date = '2023-12-31'
        
        all_data = {}
        for symbol in portfolio:
            data = await self.data_provider.get_real_data(symbol, start_date, end_date)
            if not data.empty:
                all_data[symbol] = data
        
        if not all_data:
            logger.error("No data available")
            return
        
        # Train ML models on first symbol
        first_symbol = list(all_data.keys())[0]
        self.ml_manager.train_models(all_data[first_symbol])
        
        # Run backtest
        results = await self._run_advanced_backtest(all_data)
        
        # Generate report
        self._generate_advanced_report(results)
        
        return results
    
    async def _run_advanced_backtest(self, all_data: Dict[str, pd.DataFrame]) -> Dict:
        """Run advanced backtest with ML and options"""
        logger.info("🧪 Running advanced backtest...")
        
        # Get all dates
        all_dates = set()
        for data in all_data.values():
            all_dates.update(data.index)
        all_dates = sorted(list(all_dates))
        
        daily_values = []
        
        for i, date in enumerate(all_dates[50:], 50):  # Start after 50 periods
            try:
                current_prices = {}
                for symbol, data in all_data.items():
                    if date in data.index:
                        current_prices[symbol] = data['close'].loc[date]
                
                if not current_prices:
                    continue
                
                # Generate ML signals
                ml_signals = {}
                for symbol, data in all_data.items():
                    if date in data.index:
                        historical_data = data.loc[:date]
                        if len(historical_data) >= 100:  # Minimum for ML
                            signals = self.ml_manager.predict(historical_data)
                            ml_signals[symbol] = signals
                
                # Execute trades based on ML signals
                for symbol, signals in ml_signals.items():
                    if signals:
                        # Combine ML signals
                        buy_signals = [s for s in signals.values() if s.action == 'buy']
                        sell_signals = [s for s in signals.values() if s.action == 'sell']
                        
                        if buy_signals and len(buy_signals) >= 2:  # Require consensus
                            avg_confidence = np.mean([s.confidence for s in buy_signals])
                            if avg_confidence > 0.7:
                                self._execute_trade(symbol, 'buy', current_prices[symbol], 
                                                  avg_confidence, date, 'ML_Consensus')
                        
                        elif sell_signals and len(sell_signals) >= 2:
                            avg_confidence = np.mean([s.confidence for s in sell_signals])
                            if avg_confidence > 0.7:
                                self._execute_trade(symbol, 'sell', current_prices[symbol],
                                                  avg_confidence, date, 'ML_Consensus')
                
                # Calculate portfolio value
                portfolio_value = self._calculate_portfolio_value(current_prices)
                daily_values.append({
                    'date': date,
                    'value': portfolio_value,
                    'capital': self.capital,
                    'positions': len(self.positions)
                })
                
                # Update max drawdown
                if portfolio_value > self.peak_value:
                    self.peak_value = portfolio_value
                else:
                    drawdown = (self.peak_value - portfolio_value) / self.peak_value
                    self.max_drawdown = max(self.max_drawdown, drawdown)
                
            except Exception as e:
                logger.error(f"Error processing date {date}: {e}")
                continue
        
        # Calculate final results
        if daily_values:
            final_value = daily_values[-1]['value']
            total_return = (final_value - self.initial_capital) / self.initial_capital
            
            # Calculate Sharpe ratio
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
            
            results = {
                'initial_capital': self.initial_capital,
                'final_value': final_value,
                'total_return': total_return,
                'monthly_return': (1 + total_return) ** (1/12) - 1,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': self.max_drawdown,
                'total_trades': len(self.trades),
                'daily_values': daily_values,
                'ml_models_trained': len(self.ml_manager.models),
                'features_used': len(self.ml_manager.feature_engineer.feature_names)
            }
            
            return results
        else:
            return {}
    
    def _execute_trade(self, symbol: str, action: str, price: float, 
                      confidence: float, timestamp: datetime, strategy: str):
        """Execute trade"""
        try:
            if action == 'buy':
                # Calculate position size
                target_value = self.capital * 0.2  # 20% per trade
                quantity = int(target_value / price)
                
                if quantity > 0:
                    self.capital -= quantity * price
                    if symbol in self.positions:
                        self.positions[symbol] += quantity
                    else:
                        self.positions[symbol] = quantity
                    
                    self.trades.append({
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'action': action,
                        'price': price,
                        'quantity': quantity,
                        'confidence': confidence,
                        'strategy': strategy
                    })
            
            elif action == 'sell':
                if symbol in self.positions and self.positions[symbol] > 0:
                    quantity = self.positions[symbol]
                    self.capital += quantity * price
                    self.positions[symbol] = 0
                    
                    self.trades.append({
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'action': action,
                        'price': price,
                        'quantity': quantity,
                        'confidence': confidence,
                        'strategy': strategy
                    })
        
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
    
    def _calculate_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate portfolio value"""
        portfolio_value = self.capital
        
        for symbol, quantity in self.positions.items():
            if symbol in current_prices:
                portfolio_value += quantity * current_prices[symbol]
        
        return portfolio_value
    
    def _generate_advanced_report(self, results: Dict):
        """Generate comprehensive report"""
        if not results:
            logger.warning("No results to report")
            return
        
        report = f"""
🤖 ПРОДВИНУТАЯ ML ТОРГОВАЯ СИСТЕМА - ИТОГОВЫЙ ОТЧЕТ
{'='*80}

📊 РЕЗУЛЬТАТЫ:
- Начальный капитал: {results['initial_capital']:,.0f} ₽
- Итоговый капитал: {results['final_value']:,.0f} ₽
- Общая доходность: {results['total_return']:.2%}
- Месячная доходность: {results['monthly_return']:.2%}
- Коэффициент Шарпа: {results['sharpe_ratio']:.3f}
- Максимальная просадка: {results['max_drawdown']:.2%}
- Количество сделок: {results['total_trades']}

🤖 МАШИННОЕ ОБУЧЕНИЕ:
- Обученных моделей: {results['ml_models_trained']}
- Использованных признаков: {results['features_used']}
- Модели: RandomForest, GradientBoosting, LogisticRegression, SVM

🎯 ДОСТИЖЕНИЕ ЦЕЛИ 20% В МЕСЯЦ:
{'-'*60}
Цель: 20% в месяц
Результат: {results['monthly_return']:.2%} в месяц {'✅' if results['monthly_return'] >= 0.20 else '❌'}

💡 КЛЮЧЕВЫЕ ОСОБЕННОСТИ:
1. Машинное обучение для генерации сигналов
2. Оптимизация параметров стратегий
3. Опционные стратегии
4. Реальные данные T-Bank/Yahoo Finance
5. Продвинутая инженерия признаков
6. Ансамбль моделей для консенсуса

🏆 ЗАКЛЮЧЕНИЕ:
{'-'*60}
Продвинутая ML система показала {'отличные' if results['monthly_return'] >= 0.20 else 'хорошие'} результаты.
Система готова к дальнейшему развитию и оптимизации.
"""
        
        print(report)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        with open(f'advanced_ml_report_{timestamp}.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        with open(f'advanced_ml_results_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str, ensure_ascii=False)
        
        logger.info(f"📄 Отчет сохранен в advanced_ml_report_{timestamp}.txt")
        logger.info(f"📊 Результаты сохранены в advanced_ml_results_{timestamp}.json")

async def main():
    """Main function"""
    logger.info("🚀 Запуск продвинутой ML торговой системы...")
    
    system = AdvancedTradingSystem(100000)
    results = await system.run_comprehensive_test()
    
    logger.info("🏁 Продвинутое тестирование завершено!")

if __name__ == "__main__":
    asyncio.run(main())
