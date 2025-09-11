#!/usr/bin/env python3
"""
ML Strategy Retraining Scheduler
Планировщик переобучения ML моделей
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import os
import sys
from typing import Dict, List, Optional
import schedule
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLRetrainScheduler:
    """Планировщик переобучения ML моделей"""
    
    def __init__(self, config_path: str = "config/parameters/ml_config.py"):
        self.config = self._load_config(config_path)
        self.retrain_history = []
        self.model_versions = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """Загрузка конфигурации ML"""
        try:
            # В реальном проекте здесь будет загрузка из файла
            return {
                'models': ['RandomForest', 'GradientBoosting', 'LogisticRegression', 'SVM'],
                'features': 50,
                'confidence_threshold': 0.7,
                'retrain_frequency': 'daily',
                'lookback_period': 252,
                'min_data_points': 1000,
                'validation_split': 0.2,
                'performance_threshold': 0.6,
                'market_regime_detection': True
            }
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def collect_recent_data(self, days: int = 30) -> pd.DataFrame:
        """Сбор новых данных для переобучения"""
        logger.info(f"📊 Collecting recent data for {days} days")
        
        try:
            # В реальном проекте здесь будет загрузка реальных данных
            # Пока используем генерацию тестовых данных
            np.random.seed(42)
            
            dates = pd.date_range(
                start=datetime.now() - timedelta(days=days),
                end=datetime.now(),
                freq='D'
            )
            
            # Генерируем тестовые данные
            data = []
            base_price = 100
            
            for date in dates:
                price_change = np.random.normal(0.001, 0.02)
                base_price *= (1 + price_change)
                
                data.append({
                    'date': date,
                    'open': base_price * (1 + np.random.normal(0, 0.005)),
                    'high': base_price * (1 + abs(np.random.normal(0, 0.01))),
                    'low': base_price * (1 - abs(np.random.normal(0, 0.01))),
                    'close': base_price,
                    'volume': int(1000000 * np.random.uniform(0.5, 2.0))
                })
            
            df = pd.DataFrame(data)
            df.set_index('date', inplace=True)
            
            logger.info(f"✅ Collected {len(df)} data points")
            return df
            
        except Exception as e:
            logger.error(f"Error collecting data: {e}")
            return pd.DataFrame()
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Валидация данных для переобучения"""
        logger.info("🔍 Validating data quality")
        
        try:
            # Проверка минимального количества данных
            if len(data) < self.config['min_data_points']:
                logger.warning(f"Insufficient data: {len(data)} < {self.config['min_data_points']}")
                return False
            
            # Проверка на пропущенные значения
            if data.isnull().sum().sum() > 0:
                logger.warning("Data contains missing values")
                return False
            
            # Проверка на аномальные значения
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if (data[col] <= 0).any():
                    logger.warning(f"Invalid values in column {col}")
                    return False
            
            # Проверка логики OHLC
            if not (data['high'] >= data['low']).all():
                logger.warning("Invalid OHLC data: high < low")
                return False
            
            logger.info("✅ Data validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error validating data: {e}")
            return False
    
    def detect_market_regime_change(self, new_data: pd.DataFrame) -> bool:
        """Обнаружение изменения рыночного режима"""
        logger.info("🔍 Detecting market regime changes")
        
        try:
            # Простой анализ волатильности
            returns = new_data['close'].pct_change().dropna()
            recent_volatility = returns.rolling(20).std().iloc[-1]
            
            # Загружаем историческую волатильность
            historical_volatility = self._get_historical_volatility()
            
            if historical_volatility is None:
                logger.info("No historical volatility data, assuming regime change")
                return True
            
            # Проверяем значительное изменение волатильности
            volatility_change = abs(recent_volatility - historical_volatility) / historical_volatility
            
            if volatility_change > 0.5:  # 50% изменение
                logger.info(f"Market regime change detected: volatility change {volatility_change:.2%}")
                return True
            
            logger.info("No significant market regime change detected")
            return False
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return True  # В случае ошибки предполагаем изменение
    
    def _get_historical_volatility(self) -> Optional[float]:
        """Получение исторической волатильности"""
        try:
            # В реальном проекте здесь будет загрузка из базы данных
            return 0.02  # 2% волатильность по умолчанию
        except:
            return None
    
    def train_models(self, data: pd.DataFrame) -> Dict:
        """Обучение ML моделей"""
        logger.info("🤖 Training ML models")
        
        try:
            # Упрощенная версия обучения (в реальном проекте будет полная реализация)
            models = {}
            
            for model_name in self.config['models']:
                logger.info(f"Training {model_name}...")
                
                # Симуляция обучения
                model_performance = {
                    'accuracy': np.random.uniform(0.6, 0.9),
                    'precision': np.random.uniform(0.6, 0.9),
                    'recall': np.random.uniform(0.6, 0.9),
                    'f1_score': np.random.uniform(0.6, 0.9)
                }
                
                models[model_name] = {
                    'model': f"Trained_{model_name}",
                    'performance': model_performance,
                    'training_date': datetime.now(),
                    'data_points': len(data)
                }
                
                logger.info(f"✅ {model_name} trained: accuracy={model_performance['accuracy']:.3f}")
            
            return models
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return {}
    
    def test_models(self, models: Dict, validation_data: pd.DataFrame) -> float:
        """Тестирование моделей на валидационной выборке"""
        logger.info("🧪 Testing models on validation data")
        
        try:
            # Симуляция тестирования
            total_score = 0
            model_count = 0
            
            for model_name, model_data in models.items():
                performance = model_data['performance']
                # Используем F1-score как общую метрику
                model_score = performance['f1_score']
                total_score += model_score
                model_count += 1
                
                logger.info(f"{model_name} validation score: {model_score:.3f}")
            
            if model_count > 0:
                average_score = total_score / model_count
                logger.info(f"Average validation score: {average_score:.3f}")
                return average_score
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error testing models: {e}")
            return 0.0
    
    def replace_models(self, new_models: Dict):
        """Замена старых моделей новыми"""
        logger.info("🔄 Replacing models with new versions")
        
        try:
            # Сохранение старых моделей (бэкап)
            self._backup_current_models()
            
            # Замена моделей
            self.model_versions = new_models
            
            # Сохранение новых моделей
            self._save_models(new_models)
            
            logger.info("✅ Models replaced successfully")
            
        except Exception as e:
            logger.error(f"Error replacing models: {e}")
    
    def _backup_current_models(self):
        """Создание бэкапа текущих моделей"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = f"strategies/ml/backups/models_backup_{timestamp}.json"
            
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(self.model_versions, f, indent=2, default=str, ensure_ascii=False)
            
            logger.info(f"📦 Models backed up to {backup_path}")
            
        except Exception as e:
            logger.error(f"Error backing up models: {e}")
    
    def _save_models(self, models: Dict):
        """Сохранение моделей"""
        try:
            models_path = "strategies/ml/models/current_models.json"
            
            with open(models_path, 'w', encoding='utf-8') as f:
                json.dump(models, f, indent=2, default=str, ensure_ascii=False)
            
            logger.info(f"💾 Models saved to {models_path}")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def retrain_ml_models(self):
        """Основная функция переобучения"""
        logger.info("🚀 Starting ML model retraining")
        
        try:
            # 1. Сбор новых данных
            new_data = self.collect_recent_data(days=30)
            
            if new_data.empty:
                logger.error("No new data available for retraining")
                return False
            
            # 2. Валидация данных
            if not self.validate_data(new_data):
                logger.error("Data validation failed")
                return False
            
            # 3. Обучение новых моделей
            new_models = self.train_models(new_data)
            
            if not new_models:
                logger.error("Model training failed")
                return False
            
            # 4. Тестирование на валидационной выборке
            validation_score = self.test_models(new_models, new_data)
            
            # 5. Получение текущей производительности
            current_score = self._get_current_model_performance()
            
            # 6. Принятие решения о замене
            if validation_score > current_score:
                self.replace_models(new_models)
                self._log_retraining_success(validation_score, current_score)
                return True
            else:
                self._log_retraining_failed(validation_score, current_score)
                return False
                
        except Exception as e:
            logger.error(f"Error in retraining process: {e}")
            return False
    
    def _get_current_model_performance(self) -> float:
        """Получение текущей производительности моделей"""
        try:
            # В реальном проекте здесь будет загрузка из базы данных
            return 0.75  # 75% по умолчанию
        except:
            return 0.0
    
    def _log_retraining_success(self, new_score: float, old_score: float):
        """Логирование успешного переобучения"""
        log_entry = {
            'timestamp': datetime.now(),
            'status': 'SUCCESS',
            'new_score': new_score,
            'old_score': old_score,
            'improvement': new_score - old_score
        }
        
        self.retrain_history.append(log_entry)
        logger.info(f"✅ Retraining successful: {old_score:.3f} → {new_score:.3f} (+{new_score-old_score:.3f})")
    
    def _log_retraining_failed(self, new_score: float, old_score: float):
        """Логирование неудачного переобучения"""
        log_entry = {
            'timestamp': datetime.now(),
            'status': 'FAILED',
            'new_score': new_score,
            'old_score': old_score,
            'improvement': new_score - old_score
        }
        
        self.retrain_history.append(log_entry)
        logger.info(f"❌ Retraining failed: {old_score:.3f} → {new_score:.3f} ({new_score-old_score:.3f})")
    
    def schedule_retraining(self):
        """Планирование переобучения"""
        logger.info("📅 Setting up retraining schedule")
        
        # Ежедневное переобучение в 2:00 ночи
        schedule.every().day.at("02:00").do(self.retrain_ml_models)
        
        # Переобучение при снижении качества (каждые 6 часов)
        schedule.every(6).hours.do(self._check_performance_and_retrain)
        
        # Переобучение при изменении рынка (каждые 4 часа)
        schedule.every(4).hours.do(self._check_market_regime_and_retrain)
        
        logger.info("✅ Retraining schedule configured")
        
        # Запуск планировщика
        while True:
            schedule.run_pending()
            time.sleep(60)  # Проверка каждую минуту
    
    def _check_performance_and_retrain(self):
        """Проверка производительности и переобучение при необходимости"""
        try:
            current_performance = self._get_current_model_performance()
            
            if current_performance < self.config['performance_threshold']:
                logger.warning(f"Performance below threshold: {current_performance:.3f} < {self.config['performance_threshold']}")
                self.retrain_ml_models()
                
        except Exception as e:
            logger.error(f"Error checking performance: {e}")
    
    def _check_market_regime_and_retrain(self):
        """Проверка рыночного режима и переобучение при необходимости"""
        try:
            if self.config.get('market_regime_detection', False):
                recent_data = self.collect_recent_data(days=7)
                
                if not recent_data.empty and self.detect_market_regime_change(recent_data):
                    logger.info("Market regime change detected, initiating retraining")
                    self.retrain_ml_models()
                    
        except Exception as e:
            logger.error(f"Error checking market regime: {e}")

def main():
    """Основная функция"""
    scheduler = MLRetrainScheduler()
    
    # Тестовое переобучение
    logger.info("🧪 Running test retraining")
    success = scheduler.retrain_ml_models()
    
    if success:
        logger.info("✅ Test retraining completed successfully")
    else:
        logger.info("❌ Test retraining failed")
    
    # Запуск планировщика (в реальном проекте это будет отдельный процесс)
    # scheduler.schedule_retraining()

if __name__ == "__main__":
    main()

