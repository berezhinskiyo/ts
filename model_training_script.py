#!/usr/bin/env python3
"""
Отдельный скрипт для обучения ML моделей на исторических данных
Оптимизирован для быстрого обучения и сохранения моделей
"""

import os
import sys
import json
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# ML импорты
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import SVR
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import joblib

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Быстрый класс для расчета технических индикаторов"""
    
    @staticmethod
    def sma(data: pd.Series, window: int) -> pd.Series:
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data: pd.Series, window: int) -> pd.Series:
        return data.ewm(span=window).mean()
    
    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data: pd.Series, window: int = 20, std_dev: float = 2):
        sma = TechnicalIndicators.sma(data, window)
        std = data.rolling(window=window).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=window).mean()
    
    @staticmethod
    def create_features(df: pd.DataFrame) -> pd.DataFrame:
        """Создание основных технических индикаторов"""
        df = df.copy()
        
        # Основные скользящие средние
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = TechnicalIndicators.sma(df['close'], window)
            df[f'ema_{window}'] = TechnicalIndicators.ema(df['close'], window)
        
        # RSI
        df['rsi_14'] = TechnicalIndicators.rsi(df['close'], 14)
        df['rsi_21'] = TechnicalIndicators.rsi(df['close'], 21)
        
        # MACD
        macd, signal, histogram = TechnicalIndicators.macd(df['close'])
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_histogram'] = histogram
        
        # Полосы Боллинджера
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(df['close'])
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # ATR
        df['atr'] = TechnicalIndicators.atr(df['high'], df['low'], df['close'])
        
        # Дополнительные признаки
        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        df['body_size'] = abs(df['close'] - df['open']) / df['open']
        
        # Волатильность
        df['volatility_5'] = df['close'].rolling(5).std()
        df['volatility_20'] = df['close'].rolling(20).std()
        
        # Временные признаки
        df['hour'] = df['begin'].dt.hour
        df['day_of_week'] = df['begin'].dt.dayofweek
        df['month'] = df['begin'].dt.month
        
        # Объемные индикаторы
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df

class FastMLTrainer:
    """Быстрый класс для обучения ML моделей"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.model_scores = {}
        self.models_dir = "trained_models"
        os.makedirs(self.models_dir, exist_ok=True)
    
    def prepare_data(self, df: pd.DataFrame, target_column: str = 'future_return', 
                    lookback: int = 30, forecast_horizon: int = 5) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Подготовка данных для обучения (оптимизированная версия)"""
        # Создаем целевую переменную
        df['future_price'] = df['close'].shift(-forecast_horizon)
        df['future_return'] = (df['future_price'] / df['close'] - 1) * 100
        
        # Убираем NaN
        df = df.dropna()
        
        # Выбираем числовые признаки
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = [col for col in numeric_columns if col not in ['future_price', 'future_return', 'value']]
        
        # Создаем скользящие окна (уменьшенный размер для скорости)
        X = []
        y = []
        
        for i in range(lookback, len(df) - forecast_horizon):
            window_data = df[feature_columns].iloc[i-lookback:i].values
            target_value = df[target_column].iloc[i]
            
            if not np.isnan(target_value):
                X.append(window_data.flatten())
                y.append(target_value)
        
        return np.array(X), np.array(y), feature_columns
    
    def train_models_fast(self, X: np.ndarray, y: np.ndarray, symbol: str, feature_columns: List[str]):
        """Быстрое обучение моделей с оптимизацией"""
        logger.info(f"🤖 Быстрое обучение моделей для {symbol}...")
        
        # Разделяем данные
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Нормализация
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Быстрый выбор признаков
        feature_selector = SelectKBest(score_func=f_regression, k=min(30, X_train_scaled.shape[1]))
        X_train_selected = feature_selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = feature_selector.transform(X_test_scaled)
        
        # Оптимизированные модели для быстрого обучения
        models_config = {
            'random_forest': RandomForestRegressor(
                n_estimators=50,  # Уменьшено для скорости
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=50,  # Уменьшено для скорости
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'ridge': Ridge(alpha=1.0),
            'linear_regression': LinearRegression()
        }
        
        results = {}
        
        for name, model in models_config.items():
            try:
                logger.info(f"  🔄 Обучаем {name}...")
                start_time = time.time()
                
                # Обучение
                if name in ['ridge', 'linear_regression']:
                    model.fit(X_train_selected, y_train)
                    y_pred = model.predict(X_test_selected)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                training_time = time.time() - start_time
                
                # Оценка
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                # Быстрая кросс-валидация (меньше фолдов)
                if name in ['ridge', 'linear_regression']:
                    cv_scores = cross_val_score(model, X_train_selected, y_train, cv=3, scoring='r2')
                else:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='r2')
                
                results[name] = {
                    'model': model,
                    'mse': mse,
                    'r2': r2,
                    'mae': mae,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'training_time': training_time,
                    'predictions': y_pred,
                    'actual': y_test
                }
                
                logger.info(f"    ✅ {name}: R²={r2:.4f}, MAE={mae:.4f}, CV={cv_scores.mean():.4f}, Время={training_time:.2f}с")
                
            except Exception as e:
                logger.error(f"    ❌ Ошибка обучения {name}: {e}")
        
        # Сохраняем результаты
        self.models[symbol] = results
        self.scalers[symbol] = scaler
        self.feature_selectors[symbol] = feature_selector
        
        # Сохраняем модели на диск
        self.save_models(symbol, results, scaler, feature_selector)
        
        return results
    
    def save_models(self, symbol: str, models: Dict, scaler: StandardScaler, feature_selector: SelectKBest):
        """Сохранение обученных моделей"""
        symbol_dir = os.path.join(self.models_dir, symbol)
        os.makedirs(symbol_dir, exist_ok=True)
        
        # Сохраняем каждую модель
        for model_name, model_data in models.items():
            model_path = os.path.join(symbol_dir, f"{model_name}.joblib")
            joblib.dump(model_data['model'], model_path)
            logger.info(f"  💾 Сохранена модель {model_name} для {symbol}")
        
        # Сохраняем scaler и feature_selector
        scaler_path = os.path.join(symbol_dir, "scaler.joblib")
        selector_path = os.path.join(symbol_dir, "feature_selector.joblib")
        
        joblib.dump(scaler, scaler_path)
        joblib.dump(feature_selector, selector_path)
        
        # Сохраняем метрики
        metrics = {name: {k: v for k, v in data.items() if k != 'model'} 
                  for name, data in models.items()}
        
        metrics_path = os.path.join(symbol_dir, "metrics.json")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)
    
    def load_models(self, symbol: str) -> Dict:
        """Загрузка обученных моделей"""
        symbol_dir = os.path.join(self.models_dir, symbol)
        
        if not os.path.exists(symbol_dir):
            logger.warning(f"Модели для {symbol} не найдены")
            return {}
        
        models = {}
        
        # Загружаем каждую модель
        for model_name in ['random_forest', 'gradient_boosting', 'ridge', 'linear_regression']:
            model_path = os.path.join(symbol_dir, f"{model_name}.joblib")
            if os.path.exists(model_path):
                models[model_name] = joblib.load(model_path)
        
        # Загружаем scaler и feature_selector
        scaler_path = os.path.join(symbol_dir, "scaler.joblib")
        selector_path = os.path.join(symbol_dir, "feature_selector.joblib")
        
        if os.path.exists(scaler_path):
            self.scalers[symbol] = joblib.load(scaler_path)
        if os.path.exists(selector_path):
            self.feature_selectors[symbol] = joblib.load(selector_path)
        
        # Загружаем метрики
        metrics_path = os.path.join(symbol_dir, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
                logger.info(f"📊 Загружены метрики для {symbol}: {list(metrics.keys())}")
        
        return models

class UnsupervisedTrainer:
    """Быстрый класс для обучения моделей без учителя"""
    
    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.clusterer = KMeans(n_clusters=5, random_state=42)
        self.pca = PCA(n_components=5)  # Уменьшено для скорости
        self.scaler = StandardScaler()
        self.is_trained = False
        self.models_dir = "unsupervised_models"
        os.makedirs(self.models_dir, exist_ok=True)
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Подготовка признаков для обучения без учителя"""
        feature_columns = [
            'sma_20', 'ema_20', 'rsi_14', 'macd', 'bb_width', 'atr',
            'volatility_20', 'volume_ratio', 'price_change', 'high_low_ratio'
        ]
        
        available_columns = [col for col in feature_columns if col in df.columns]
        
        if not available_columns:
            return np.array([])
        
        features = df[available_columns].fillna(0).values
        return features
    
    def train_unsupervised_models(self, df: pd.DataFrame):
        """Быстрое обучение моделей без учителя"""
        logger.info("🤖 Быстрое обучение моделей без учителя...")
        
        features = self.prepare_features(df)
        if len(features) < 100:
            logger.warning("⚠️ Недостаточно данных для обучения без учителя")
            return
        
        # Нормализация
        features_scaled = self.scaler.fit_transform(features)
        
        # PCA для снижения размерности
        features_pca = self.pca.fit_transform(features_scaled)
        
        # Обучение детектора аномалий
        self.anomaly_detector.fit(features_pca)
        
        # Обучение кластеризации
        self.clusterer.fit(features_pca)
        
        self.is_trained = True
        
        # Сохраняем модели
        self.save_models()
        
        logger.info("✅ Модели без учителя обучены и сохранены")
    
    def save_models(self):
        """Сохранение моделей без учителя"""
        joblib.dump(self.anomaly_detector, os.path.join(self.models_dir, "anomaly_detector.joblib"))
        joblib.dump(self.clusterer, os.path.join(self.models_dir, "clusterer.joblib"))
        joblib.dump(self.pca, os.path.join(self.models_dir, "pca.joblib"))
        joblib.dump(self.scaler, os.path.join(self.models_dir, "scaler.joblib"))
    
    def load_models(self):
        """Загрузка моделей без учителя"""
        try:
            self.anomaly_detector = joblib.load(os.path.join(self.models_dir, "anomaly_detector.joblib"))
            self.clusterer = joblib.load(os.path.join(self.models_dir, "clusterer.joblib"))
            self.pca = joblib.load(os.path.join(self.models_dir, "pca.joblib"))
            self.scaler = joblib.load(os.path.join(self.models_dir, "scaler.joblib"))
            self.is_trained = True
            logger.info("✅ Модели без учителя загружены")
        except Exception as e:
            logger.warning(f"⚠️ Не удалось загрузить модели без учителя: {e}")

def load_data_from_files(symbols: List[str], data_dir: str = "data/3year_minute_data") -> Dict[str, pd.DataFrame]:
    """Загрузка данных из файлов"""
    logger.info("📥 Загружаем данные из файлов...")
    
    all_data = {}
    
    for symbol in symbols:
        file_path = os.path.join(data_dir, f"{symbol}_3year_minute.csv")
        
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                df['begin'] = pd.to_datetime(df['begin'])
                df['end'] = pd.to_datetime(df['end'])
                
                # Удаляем дубликаты
                df = df.drop_duplicates(subset=['begin']).reset_index(drop=True)
                
                all_data[symbol] = df
                logger.info(f"✅ {symbol}: Загружено {len(df)} записей")
                
            except Exception as e:
                logger.error(f"❌ Ошибка загрузки {symbol}: {e}")
        else:
            logger.warning(f"⚠️ Файл {file_path} не найден")
    
    return all_data

def main():
    """Основная функция обучения моделей"""
    logger.info("🚀 ЗАПУСК ОБУЧЕНИЯ ML МОДЕЛЕЙ")
    logger.info("=" * 50)
    
    # Инструменты для обучения
    symbols = ['GAZP', 'SBER', 'PIKK', 'IRAO', 'SGZH']
    
    # Загрузка данных
    all_data = load_data_from_files(symbols)
    
    if not all_data:
        logger.error("❌ Не удалось загрузить данные. Завершение работы.")
        return
    
    # Инициализация
    trainer = FastMLTrainer()
    unsupervised_trainer = UnsupervisedTrainer()
    indicators = TechnicalIndicators()
    
    # Обучение моделей для каждого символа
    trained_models = {}
    
    for symbol, df in all_data.items():
        logger.info(f"📊 Обрабатываем {symbol}...")
        
        # Создаем технические индикаторы
        df_with_features = indicators.create_features(df)
        
        # Подготавливаем данные для обучения
        X, y, feature_columns = trainer.prepare_data(df_with_features)
        
        if len(X) > 100:  # Минимум данных для обучения
            # Обучаем модели
            models = trainer.train_models_fast(X, y, symbol, feature_columns)
            trained_models[symbol] = {
                'models': models,
                'data': df_with_features
            }
        else:
            logger.warning(f"⚠️ {symbol}: Недостаточно данных для обучения ({len(X)} записей)")
    
    # Обучение моделей без учителя
    if trained_models:
        logger.info("🤖 Обучение моделей без учителя...")
        all_training_data = []
        for model_data in trained_models.values():
            all_training_data.append(model_data['data'])
        
        combined_df = pd.concat(all_training_data, ignore_index=True)
        unsupervised_trainer.train_unsupervised_models(combined_df)
    
    # Создаем сводку результатов
    summary_data = []
    
    for symbol, model_data in trained_models.items():
        models = model_data['models']
        
        for model_name, model_info in models.items():
            summary_data.append({
                'Symbol': symbol,
                'Model': model_name,
                'R2': f"{model_info['r2']:.4f}",
                'MAE': f"{model_info['mae']:.4f}",
                'CV_Mean': f"{model_info['cv_mean']:.4f}",
                'CV_Std': f"{model_info['cv_std']:.4f}",
                'Training_Time': f"{model_info['training_time']:.2f}s"
            })
    
    # Сохраняем сводку
    if summary_data:
        results_df = pd.DataFrame(summary_data)
        results_df.to_csv('model_training_results.csv', index=False)
        
        logger.info("\n📋 СВОДКА ОБУЧЕНИЯ МОДЕЛЕЙ:")
        logger.info("=" * 50)
        print(results_df.to_string(index=False))
        
        # Статистика
        r2_scores = [float(r['R2']) for r in summary_data]
        mae_scores = [float(r['MAE']) for r in summary_data]
        
        logger.info(f"\n📊 ОБЩАЯ СТАТИСТИКА:")
        logger.info(f"  Средний R²: {np.mean(r2_scores):.4f}")
        logger.info(f"  Средний MAE: {np.mean(mae_scores):.4f}")
        logger.info(f"  Лучший R²: {np.max(r2_scores):.4f}")
        logger.info(f"  Худший R²: {np.min(r2_scores):.4f}")
        
        # Лучшие модели по символам
        logger.info(f"\n🏆 ЛУЧШИЕ МОДЕЛИ ПО СИМВОЛАМ:")
        for symbol in symbols:
            symbol_results = [r for r in summary_data if r['Symbol'] == symbol]
            if symbol_results:
                best_model = max(symbol_results, key=lambda x: float(x['R2']))
                logger.info(f"  {symbol}: {best_model['Model']} (R²={best_model['R2']})")
    
    logger.info(f"\n💾 Модели сохранены в директории:")
    logger.info(f"  📁 trained_models/ - обученные ML модели")
    logger.info(f"  📁 unsupervised_models/ - модели без учителя")
    logger.info(f"  📄 model_training_results.csv - сводка результатов")
    logger.info(f"  📄 model_training.log - логи обучения")
    
    logger.info("\n✅ ОБУЧЕНИЕ МОДЕЛЕЙ ЗАВЕРШЕНО!")

if __name__ == "__main__":
    main()
