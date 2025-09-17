#!/usr/bin/env python3
"""
Менеджер для сохранения и загрузки обученных моделей
"""

import os
import pickle
import joblib
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ModelManager:
    """Класс для управления обученными моделями"""
    
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.ensure_directories()
    
    def ensure_directories(self):
        """Создание необходимых директорий"""
        directories = [
            self.models_dir,
            os.path.join(self.models_dir, "arima"),
            os.path.join(self.models_dir, "lstm"),
            os.path.join(self.models_dir, "ensemble"),
            os.path.join(self.models_dir, "sarima"),
            os.path.join(self.models_dir, "metadata")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def save_model(self, model, model_name: str, symbol: str, strategy_type: str, 
                   metadata: Dict[str, Any] = None) -> str:
        """Сохранение модели"""
        try:
            # Создаем имя файла
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_{model_name}_{timestamp}.pkl"
            filepath = os.path.join(self.models_dir, strategy_type, filename)
            
            # Сохраняем модель
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            
            # Сохраняем метаданные
            if metadata is None:
                metadata = {}
            
            metadata.update({
                'model_name': model_name,
                'symbol': symbol,
                'strategy_type': strategy_type,
                'timestamp': timestamp,
                'filepath': filepath,
                'created_at': datetime.now().isoformat()
            })
            
            metadata_file = os.path.join(self.models_dir, "metadata", f"{filename}.json")
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ Модель сохранена: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения модели {model_name}: {e}")
            return None
    
    def load_model(self, filepath: str):
        """Загрузка модели"""
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            
            logger.info(f"✅ Модель загружена: {filepath}")
            return model
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели {filepath}: {e}")
            return None
    
    def load_latest_model(self, symbol: str, strategy_type: str, model_name: str = None):
        """Загрузка последней модели для символа и стратегии"""
        try:
            strategy_dir = os.path.join(self.models_dir, strategy_type)
            
            if not os.path.exists(strategy_dir):
                logger.warning(f"Директория {strategy_dir} не найдена")
                return None
            
            # Ищем файлы для символа
            pattern = f"{symbol}_{model_name}" if model_name else symbol
            files = [f for f in os.listdir(strategy_dir) if f.startswith(pattern) and f.endswith('.pkl')]
            
            if not files:
                logger.warning(f"Модели для {symbol} ({strategy_type}) не найдены")
                return None
            
            # Сортируем по времени создания (по имени файла)
            files.sort(reverse=True)
            latest_file = files[0]
            filepath = os.path.join(strategy_dir, latest_file)
            
            return self.load_model(filepath)
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки последней модели {symbol} ({strategy_type}): {e}")
            return None
    
    def list_models(self, symbol: str = None, strategy_type: str = None) -> pd.DataFrame:
        """Список всех сохраненных моделей"""
        try:
            models_data = []
            
            for strategy in os.listdir(self.models_dir):
                if strategy == "metadata":
                    continue
                
                if strategy_type and strategy != strategy_type:
                    continue
                
                strategy_path = os.path.join(self.models_dir, strategy)
                if not os.path.isdir(strategy_path):
                    continue
                
                for filename in os.listdir(strategy_path):
                    if not filename.endswith('.pkl'):
                        continue
                    
                    # Парсим имя файла: SYMBOL_MODELNAME_TIMESTAMP.pkl
                    parts = filename.replace('.pkl', '').split('_')
                    if len(parts) >= 3:
                        file_symbol = parts[0]
                        model_name = '_'.join(parts[1:-1])
                        timestamp = parts[-1]
                        
                        if symbol and file_symbol != symbol:
                            continue
                        
                        filepath = os.path.join(strategy_path, filename)
                        file_size = os.path.getsize(filepath)
                        created_time = datetime.fromtimestamp(os.path.getctime(filepath))
                        
                        models_data.append({
                            'symbol': file_symbol,
                            'strategy': strategy,
                            'model_name': model_name,
                            'timestamp': timestamp,
                            'filepath': filepath,
                            'size_mb': round(file_size / 1024 / 1024, 2),
                            'created': created_time.strftime('%Y-%m-%d %H:%M:%S')
                        })
            
            if models_data:
                df = pd.DataFrame(models_data)
                return df.sort_values(['symbol', 'strategy', 'created'], ascending=[True, True, False])
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"❌ Ошибка получения списка моделей: {e}")
            return pd.DataFrame()
    
    def delete_old_models(self, keep_count: int = 5):
        """Удаление старых моделей, оставляя только последние"""
        try:
            models_df = self.list_models()
            
            if models_df.empty:
                return
            
            # Группируем по символу и стратегии
            for (symbol, strategy), group in models_df.groupby(['symbol', 'strategy']):
                if len(group) > keep_count:
                    # Сортируем по времени создания и удаляем старые
                    group_sorted = group.sort_values('created', ascending=False)
                    to_delete = group_sorted.iloc[keep_count:]
                    
                    for _, model_info in to_delete.iterrows():
                        try:
                            os.remove(model_info['filepath'])
                            logger.info(f"🗑️ Удалена старая модель: {model_info['filepath']}")
                        except Exception as e:
                            logger.error(f"❌ Ошибка удаления {model_info['filepath']}: {e}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка очистки старых моделей: {e}")
    
    def get_model_info(self, filepath: str) -> Dict[str, Any]:
        """Получение информации о модели"""
        try:
            # Ищем метаданные
            metadata_file = os.path.join(self.models_dir, "metadata", f"{os.path.basename(filepath)}.json")
            
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                return metadata
            else:
                # Базовая информация из файла
                stat = os.stat(filepath)
                return {
                    'filepath': filepath,
                    'size_mb': round(stat.st_size / 1024 / 1024, 2),
                    'created_at': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    'modified_at': datetime.fromtimestamp(stat.st_mtime).isoformat()
                }
                
        except Exception as e:
            logger.error(f"❌ Ошибка получения информации о модели {filepath}: {e}")
            return {}

def main():
    """Демонстрация работы ModelManager"""
    print("🤖 ДЕМОНСТРАЦИЯ MODEL MANAGER")
    print("=" * 50)
    
    manager = ModelManager()
    
    # Показываем структуру директорий
    print("\n📁 Структура директорий:")
    for root, dirs, files in os.walk(manager.models_dir):
        level = root.replace(manager.models_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Показываем только первые 5 файлов
            print(f"{subindent}{file}")
        if len(files) > 5:
            print(f"{subindent}... и еще {len(files) - 5} файлов")
    
    # Показываем список моделей
    print("\n📊 Список моделей:")
    models_df = manager.list_models()
    if not models_df.empty:
        print(models_df.to_string(index=False))
    else:
        print("Модели не найдены")
    
    # Показываем статистику
    print(f"\n📈 Статистика:")
    print(f"   Всего моделей: {len(models_df)}")
    if not models_df.empty:
        print(f"   Общий размер: {models_df['size_mb'].sum():.2f} MB")
        print(f"   Уникальных символов: {models_df['symbol'].nunique()}")
        print(f"   Уникальных стратегий: {models_df['strategy'].nunique()}")

if __name__ == "__main__":
    main()
