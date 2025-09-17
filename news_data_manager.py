#!/usr/bin/env python3
"""
Менеджер данных новостей за 3 года
Создание, сохранение и загрузка исторических новостей
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pickle

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NewsDataManager:
    """Менеджер данных новостей за 3 года"""
    
    def __init__(self, data_dir: str = "data/news_3year"):
        self.data_dir = data_dir
        self.news_file = os.path.join(data_dir, "news_3year_data.json")
        self.metadata_file = os.path.join(data_dir, "news_metadata.json")
        
        # Создаем директорию если не существует
        os.makedirs(data_dir, exist_ok=True)
        
        logger.info(f"✅ Менеджер новостей инициализирован: {data_dir}")
    
    def generate_3year_news(self, symbols: List[str], start_date: datetime, end_date: datetime) -> Dict[str, List[Dict]]:
        """Генерация новостей за 3 года для всех символов"""
        
        logger.info(f"📰 Генерация новостей за 3 года для {len(symbols)} символов...")
        logger.info(f"📅 Период: {start_date.date()} - {end_date.date()}")
        
        # Типы новостей с разным влиянием
        news_templates = [
            # Сильно позитивные новости
            {
                'title_template': '{symbol}: Рекордная прибыль превышает прогнозы на {percent}%',
                'content_template': 'Компания {symbol} показала неожиданно высокую прибыль, превысив прогнозы аналитиков на {percent}%.',
                'sentiment_score': 0.9,
                'confidence': 0.95,
                'impact': 'very_high',
                'probability': 0.05
            },
            {
                'title_template': '{symbol}: Крупные инвестиции и расширение бизнеса',
                'content_template': '{symbol} объявила о планах крупных инвестиций в развитие новых направлений на сумму {amount} млрд руб.',
                'sentiment_score': 0.8,
                'confidence': 0.85,
                'impact': 'high',
                'probability': 0.08
            },
            {
                'title_template': '{symbol}: Повышение дивидендов и выкуп акций',
                'content_template': 'Совет директоров {symbol} одобрил повышение дивидендных выплат на {percent}% и программу выкупа акций.',
                'sentiment_score': 0.7,
                'confidence': 0.8,
                'impact': 'high',
                'probability': 0.1
            },
            # Позитивные новости
            {
                'title_template': '{symbol}: Стабильные результаты в квартале',
                'content_template': 'Компания {symbol} показала стабильные результаты, соответствующие прогнозам аналитиков.',
                'sentiment_score': 0.6,
                'confidence': 0.7,
                'impact': 'medium',
                'probability': 0.15
            },
            {
                'title_template': '{symbol}: Новые контракты и партнерства',
                'content_template': '{symbol} подписала крупные контракты, что укрепит позиции компании на рынке.',
                'sentiment_score': 0.5,
                'confidence': 0.6,
                'impact': 'medium',
                'probability': 0.12
            },
            # Негативные новости
            {
                'title_template': '{symbol}: Снижение прибыли и убытки',
                'content_template': 'Компания {symbol} показала снижение прибыли на {percent}% из-за неблагоприятных условий на рынке.',
                'sentiment_score': -0.7,
                'confidence': 0.8,
                'impact': 'high',
                'probability': 0.08
            },
            {
                'title_template': '{symbol}: Регуляторные проблемы и штрафы',
                'content_template': 'На {symbol} наложены штрафы регулятором за нарушения в отчетности на сумму {amount} млн руб.',
                'sentiment_score': -0.8,
                'confidence': 0.85,
                'impact': 'high',
                'probability': 0.05
            },
            {
                'title_template': '{symbol}: Снижение рейтинга аналитиками',
                'content_template': 'Ведущие аналитики понизили рейтинг {symbol} с "покупать" до "держать" из-за ухудшения перспектив.',
                'sentiment_score': -0.6,
                'confidence': 0.7,
                'impact': 'medium',
                'probability': 0.1
            },
            # Нейтральные новости
            {
                'title_template': '{symbol}: Обычные торговые сессии',
                'content_template': 'Торги {symbol} прошли в обычном режиме без значительных изменений.',
                'sentiment_score': 0.1,
                'confidence': 0.4,
                'impact': 'low',
                'probability': 0.2
            },
            {
                'title_template': '{symbol}: Плановые корпоративные события',
                'content_template': 'Компания {symbol} провела плановое собрание акционеров.',
                'sentiment_score': 0.0,
                'confidence': 0.3,
                'impact': 'low',
                'probability': 0.07
            }
        ]
        
        all_news = {}
        
        for symbol in symbols:
            logger.info(f"  📊 Генерация новостей для {symbol}...")
            symbol_news = []
            current_date = start_date
            
            while current_date <= end_date:
                # Вероятность новости в день
                news_probability = 0.6  # 60% вероятность новости в день
                
                if np.random.random() < news_probability:
                    # Количество новостей в день (1-3)
                    num_news = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
                    
                    for _ in range(num_news):
                        # Выбираем шаблон новости на основе вероятности
                        template = np.random.choice(news_templates, p=[t['probability'] for t in news_templates])
                        
                        # Генерируем параметры для шаблона
                        params = {
                            'symbol': symbol,
                            'percent': np.random.randint(5, 25),
                            'amount': np.random.randint(1, 10)
                        }
                        
                        # Создаем новость
                        news_item = {
                            'title': template['title_template'].format(**params),
                            'content': template['content_template'].format(**params),
                            'published_at': current_date + timedelta(hours=np.random.randint(9, 18)),
                            'source': 'Financial News',
                            'symbol': symbol,
                            'sentiment_score': template['sentiment_score'],
                            'confidence': template['confidence'],
                            'impact': template['impact'],
                            'category': self._categorize_news(template['sentiment_score']),
                            'id': f"{symbol}_{current_date.strftime('%Y%m%d')}_{len(symbol_news)}"
                        }
                        
                        symbol_news.append(news_item)
                
                current_date += timedelta(days=1)
            
            all_news[symbol] = symbol_news
            logger.info(f"    ✅ {symbol}: {len(symbol_news)} новостей")
        
        logger.info(f"📰 Всего сгенерировано новостей: {sum(len(news) for news in all_news.values())}")
        return all_news
    
    def _categorize_news(self, sentiment_score: float) -> str:
        """Категоризация новостей по настроению"""
        if sentiment_score > 0.5:
            return 'very_positive'
        elif sentiment_score > 0.2:
            return 'positive'
        elif sentiment_score > -0.2:
            return 'neutral'
        elif sentiment_score > -0.5:
            return 'negative'
        else:
            return 'very_negative'
    
    def save_news_data(self, news_data: Dict[str, List[Dict]], metadata: Dict[str, Any] = None):
        """Сохранение данных новостей"""
        
        logger.info(f"💾 Сохранение данных новостей в {self.news_file}...")
        
        try:
            # Конвертируем datetime объекты в строки
            def convert_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {k: convert_datetime(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_datetime(item) for item in obj]
                else:
                    return obj
            
            converted_data = convert_datetime(news_data)
            
            # Сохраняем основные данные
            with open(self.news_file, 'w', encoding='utf-8') as f:
                json.dump(converted_data, f, ensure_ascii=False, indent=2)
            
            # Создаем метаданные
            if metadata is None:
                metadata = {
                    'created_at': datetime.now().isoformat(),
                    'total_symbols': len(news_data),
                    'total_news': sum(len(news) for news in news_data.values()),
                    'symbols': list(news_data.keys()),
                    'date_range': {
                        'start': min(min(news['published_at'] for news in news_list) for news_list in news_data.values()).isoformat(),
                        'end': max(max(news['published_at'] for news in news_list) for news_list in news_data.values()).isoformat()
                    }
                }
            
            # Сохраняем метаданные
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ Данные новостей сохранены:")
            logger.info(f"  📁 Файл: {self.news_file}")
            logger.info(f"  📊 Символов: {metadata['total_symbols']}")
            logger.info(f"  📰 Новостей: {metadata['total_news']}")
            logger.info(f"  📅 Период: {metadata['date_range']['start']} - {metadata['date_range']['end']}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения данных новостей: {e}")
    
    def load_news_data(self) -> Optional[Dict[str, List[Dict]]]:
        """Загрузка данных новостей"""
        
        if not os.path.exists(self.news_file):
            logger.warning(f"⚠️ Файл новостей не найден: {self.news_file}")
            return None
        
        try:
            logger.info(f"📂 Загрузка данных новостей из {self.news_file}...")
            
            with open(self.news_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Конвертируем строки обратно в datetime
            def convert_string_to_datetime(obj):
                if isinstance(obj, str) and 'T' in obj:
                    try:
                        return datetime.fromisoformat(obj)
                    except:
                        return obj
                elif isinstance(obj, dict):
                    return {k: convert_string_to_datetime(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_string_to_datetime(item) for item in obj]
                else:
                    return obj
            
            converted_data = convert_string_to_datetime(data)
            
            # Загружаем метаданные
            metadata = self.load_metadata()
            if metadata:
                logger.info(f"✅ Данные новостей загружены:")
                logger.info(f"  📊 Символов: {metadata['total_symbols']}")
                logger.info(f"  📰 Новостей: {metadata['total_news']}")
                logger.info(f"  📅 Период: {metadata['date_range']['start']} - {metadata['date_range']['end']}")
            
            return converted_data
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки данных новостей: {e}")
            return None
    
    def load_metadata(self) -> Optional[Dict[str, Any]]:
        """Загрузка метаданных"""
        
        if not os.path.exists(self.metadata_file):
            return None
        
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки метаданных: {e}")
            return None
    
    def get_news_for_period(self, symbol: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Получение новостей для конкретного символа и периода"""
        
        news_data = self.load_news_data()
        if not news_data or symbol not in news_data:
            return []
        
        symbol_news = news_data[symbol]
        
        # Фильтруем по периоду
        filtered_news = [
            news for news in symbol_news
            if start_date <= news['published_at'] <= end_date
        ]
        
        return filtered_news
    
    def get_news_statistics(self) -> Dict[str, Any]:
        """Получение статистики по новостям"""
        
        news_data = self.load_news_data()
        if not news_data:
            return {}
        
        stats = {
            'total_symbols': len(news_data),
            'total_news': sum(len(news) for news in news_data.values()),
            'symbols': {}
        }
        
        for symbol, news_list in news_data.items():
            if not news_list:
                continue
            
            # Статистика по символу
            symbol_stats = {
                'total_news': len(news_list),
                'categories': {},
                'sentiment_distribution': {
                    'very_positive': 0,
                    'positive': 0,
                    'neutral': 0,
                    'negative': 0,
                    'very_negative': 0
                },
                'impact_distribution': {
                    'very_high': 0,
                    'high': 0,
                    'medium': 0,
                    'low': 0
                },
                'avg_sentiment': 0.0,
                'avg_confidence': 0.0
            }
            
            # Анализируем новости
            sentiment_scores = []
            confidence_scores = []
            
            for news in news_list:
                # Категории по настроению
                category = news.get('category', 'neutral')
                symbol_stats['sentiment_distribution'][category] += 1
                
                # Категории по влиянию
                impact = news.get('impact', 'low')
                symbol_stats['impact_distribution'][impact] += 1
                
                # Собираем метрики
                sentiment_scores.append(news.get('sentiment_score', 0.0))
                confidence_scores.append(news.get('confidence', 0.0))
            
            # Рассчитываем средние значения
            if sentiment_scores:
                symbol_stats['avg_sentiment'] = np.mean(sentiment_scores)
                symbol_stats['avg_confidence'] = np.mean(confidence_scores)
            
            stats['symbols'][symbol] = symbol_stats
        
        return stats
    
    def export_to_csv(self, output_file: str = None):
        """Экспорт новостей в CSV формат"""
        
        if output_file is None:
            output_file = os.path.join(self.data_dir, "news_3year_data.csv")
        
        news_data = self.load_news_data()
        if not news_data:
            logger.error("❌ Нет данных для экспорта")
            return
        
        logger.info(f"📊 Экспорт новостей в CSV: {output_file}")
        
        # Подготавливаем данные для CSV
        csv_data = []
        for symbol, news_list in news_data.items():
            for news in news_list:
                csv_data.append({
                    'symbol': symbol,
                    'title': news['title'],
                    'content': news['content'],
                    'published_at': news['published_at'],
                    'source': news['source'],
                    'sentiment_score': news['sentiment_score'],
                    'confidence': news['confidence'],
                    'impact': news['impact'],
                    'category': news.get('category', 'neutral'),
                    'id': news.get('id', '')
                })
        
        # Создаем DataFrame и сохраняем
        df = pd.DataFrame(csv_data)
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        logger.info(f"✅ Экспорт завершен: {len(csv_data)} записей в {output_file}")

def main():
    """Основная функция для создания и сохранения новостей за 3 года"""
    
    # Символы для генерации новостей
    symbols = ['SBER', 'GAZP', 'LKOH', 'NVTK', 'ROSN', 'TATN', 'MGNT', 'MTSS', 'PIKK', 'IRAO', 'SGZH']
    
    # Период 3 года
    start_date = datetime(2022, 9, 19)
    end_date = datetime(2025, 9, 2)
    
    # Создаем менеджер
    manager = NewsDataManager()
    
    # Проверяем, есть ли уже данные
    existing_data = manager.load_news_data()
    if existing_data:
        logger.info("📂 Данные новостей уже существуют")
        
        # Показываем статистику
        stats = manager.get_news_statistics()
        logger.info(f"📊 Статистика существующих данных:")
        logger.info(f"  📰 Всего новостей: {stats['total_news']}")
        logger.info(f"  📊 Символов: {stats['total_symbols']}")
        
        # Экспортируем в CSV
        manager.export_to_csv()
        
    else:
        logger.info("🔄 Генерация новых данных новостей...")
        
        # Генерируем новости
        news_data = manager.generate_3year_news(symbols, start_date, end_date)
        
        # Сохраняем данные
        manager.save_news_data(news_data)
        
        # Экспортируем в CSV
        manager.export_to_csv()
        
        # Показываем статистику
        stats = manager.get_news_statistics()
        logger.info(f"📊 Статистика сгенерированных данных:")
        logger.info(f"  📰 Всего новостей: {stats['total_news']}")
        logger.info(f"  📊 Символов: {stats['total_symbols']}")

if __name__ == "__main__":
    main()
