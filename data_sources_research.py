#!/usr/bin/env python3
"""
Исследование источников тиковых данных ММВБ
"""

import requests
import pandas as pd
import json
from datetime import datetime, timedelta
import os
import time

class MOEXDataSources:
    """Класс для работы с различными источниками данных ММВБ"""
    
    def __init__(self):
        self.sources = {
            'moex_official': {
                'name': 'Московская биржа (официальный)',
                'url': 'https://iss.moex.com/iss/',
                'description': 'Официальный API Московской биржи',
                'tick_data': True,
                'cost': 'Платный',
                'api_docs': 'https://iss.moex.com/iss/reference/'
            },
            'finam': {
                'name': 'Финам',
                'url': 'https://www.finam.ru/',
                'description': 'Брокер с API для исторических данных',
                'tick_data': True,
                'cost': 'Платный',
                'api_docs': 'https://www.finam.ru/profile/moex-akcii/'
            },
            'quik': {
                'name': 'QUIK',
                'url': 'https://www.quik.ru/',
                'description': 'Торговая платформа с историческими данными',
                'tick_data': True,
                'cost': 'Платный',
                'api_docs': 'https://www.quik.ru/support/api'
            },
            'ticktrack': {
                'name': 'TickTrack',
                'url': 'https://ticktrack.ru/',
                'description': 'Специализированный поставщик тиковых данных',
                'tick_data': True,
                'cost': 'Платный',
                'api_docs': 'https://ticktrack.ru/ru'
            },
            'xtick': {
                'name': 'XTick',
                'url': 'https://xtick.plan.ru/',
                'description': 'Программа технического анализа',
                'tick_data': True,
                'cost': 'Платный',
                'api_docs': 'https://xtick.plan.ru/tikker.html'
            },
            'smart_lab': {
                'name': 'Smart-Lab',
                'url': 'https://smart-lab.ru/',
                'description': 'Форум трейдеров с обменом данными',
                'tick_data': True,
                'cost': 'Бесплатный (ограниченный)',
                'api_docs': 'https://smart-lab.ru/blog/752317.php'
            }
        }
    
    def get_moex_instruments(self):
        """Получение списка инструментов с ММВБ"""
        try:
            url = "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities.json"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                securities = data['securities']['data']
                columns = data['securities']['columns']
                
                df = pd.DataFrame(securities, columns=columns)
                return df[['SECID', 'SHORTNAME', 'LOTSIZE', 'MINSTEP']].head(20)
            else:
                print(f"Ошибка получения данных: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Ошибка при получении инструментов ММВБ: {e}")
            return None
    
    def test_moex_api(self):
        """Тестирование доступности API ММВБ"""
        try:
            # Тестируем получение данных по SBER
            url = "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/SBER/candles.json"
            params = {
                'from': '2024-09-01',
                'till': '2024-09-15',
                'interval': 1  # 1 минута
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                candles = data['candles']['data']
                columns = data['candles']['columns']
                
                if candles:
                    df = pd.DataFrame(candles, columns=columns)
                    print(f"✅ API ММВБ доступен. Получено {len(df)} свечей по SBER")
                    return True
                else:
                    print("⚠️ API ММВБ доступен, но данные пустые")
                    return False
            else:
                print(f"❌ API ММВБ недоступен: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Ошибка тестирования API ММВБ: {e}")
            return False
    
    def get_tick_data_sources(self):
        """Получение информации об источниках тиковых данных"""
        print("📊 ИСТОЧНИКИ ТИКОВЫХ ДАННЫХ ММВБ:")
        print("=" * 60)
        
        for key, source in self.sources.items():
            print(f"\n🔹 {source['name']}")
            print(f"   URL: {source['url']}")
            print(f"   Описание: {source['description']}")
            print(f"   Тиковые данные: {'✅' if source['tick_data'] else '❌'}")
            print(f"   Стоимость: {source['cost']}")
            if 'api_docs' in source:
                print(f"   Документация: {source['api_docs']}")
    
    def create_data_download_script(self):
        """Создание скрипта для загрузки данных"""
        script_content = '''#!/usr/bin/env python3
"""
Скрипт для загрузки тиковых данных ММВБ
"""

import requests
import pandas as pd
import json
from datetime import datetime, timedelta
import os
import time

class MOEXTickDataDownloader:
    """Класс для загрузки тиковых данных с ММВБ"""
    
    def __init__(self, output_dir="data/tick_data"):
        self.output_dir = output_dir
        self.base_url = "https://iss.moex.com/iss/"
        os.makedirs(output_dir, exist_ok=True)
    
    def download_candles(self, symbol, days=30):
        """Загрузка минутных данных (аналог тиковых)"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            url = f"{self.base_url}engines/stock/markets/shares/boards/TQBR/securities/{symbol}/candles.json"
            params = {
                'from': start_date.strftime('%Y-%m-%d'),
                'till': end_date.strftime('%Y-%m-%d'),
                'interval': 1  # 1 минута
            }
            
            print(f"Загрузка данных для {symbol}...")
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                candles = data['candles']['data']
                columns = data['candles']['columns']
                
                if candles:
                    df = pd.DataFrame(candles, columns=columns)
                    df['datetime'] = pd.to_datetime(df['begin'])
                    
                    # Сохраняем данные
                    filename = f"{self.output_dir}/{symbol}_1min_{days}d.csv"
                    df.to_csv(filename, index=False)
                    print(f"✅ Сохранено {len(df)} записей в {filename}")
                    return df
                else:
                    print(f"⚠️ Нет данных для {symbol}")
                    return None
            else:
                print(f"❌ Ошибка загрузки {symbol}: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Ошибка при загрузке {symbol}: {e}")
            return None
    
    def download_multiple_symbols(self, symbols, days=30):
        """Загрузка данных для нескольких инструментов"""
        results = {}
        
        for symbol in symbols:
            df = self.download_candles(symbol, days)
            if df is not None:
                results[symbol] = df
            time.sleep(1)  # Пауза между запросами
        
        return results

if __name__ == "__main__":
    # Основные инструменты ММВБ
    symbols = ['SBER', 'GAZP', 'LKOH', 'NVTK', 'ROSN', 'TATN', 'MGNT', 'MTSS']
    
    downloader = MOEXTickDataDownloader()
    results = downloader.download_multiple_symbols(symbols, days=7)
    
    print(f"\\n📊 Загружено данных для {len(results)} инструментов")
'''
        
        with open('download_tick_data.py', 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        print("✅ Создан скрипт download_tick_data.py для загрузки данных")
    
    def generate_recommendations(self):
        """Генерация рекомендаций по источникам данных"""
        recommendations = {
            'free_sources': [
                {
                    'name': 'MOEX API (ограниченный)',
                    'description': 'Официальный API с ограниченным доступом к историческим данным',
                    'pros': ['Официальный источник', 'Бесплатный', 'Надежный'],
                    'cons': ['Ограниченная глубина истории', 'Нет тиковых данных', 'Rate limits']
                }
            ],
            'paid_sources': [
                {
                    'name': 'TickTrack',
                    'description': 'Специализированный поставщик тиковых данных',
                    'pros': ['Полные тиковые данные', 'Высокое качество', 'API доступ'],
                    'cons': ['Платный', 'Требует подписки'],
                    'cost': 'От 5000₽/месяц'
                },
                {
                    'name': 'Финам API',
                    'description': 'API брокера Финам для исторических данных',
                    'pros': ['Доступ к тиковым данным', 'Интеграция с торговлей', 'Поддержка'],
                    'cons': ['Платный', 'Требует брокерский счет'],
                    'cost': 'Зависит от тарифа'
                }
            ],
            'alternative_sources': [
                {
                    'name': 'Smart-Lab форум',
                    'description': 'Сообщество трейдеров, обменивающихся данными',
                    'pros': ['Бесплатный', 'Разнообразные данные', 'Сообщество'],
                    'cons': ['Неофициальный', 'Качество может варьироваться', 'Ограниченный объем']
                }
            ]
        }
        
        print("\n🎯 РЕКОМЕНДАЦИИ ПО ИСТОЧНИКАМ ДАННЫХ:")
        print("=" * 60)
        
        for category, sources in recommendations.items():
            print(f"\n📂 {category.upper().replace('_', ' ')}:")
            for source in sources:
                print(f"\n🔹 {source['name']}")
                print(f"   {source['description']}")
                print(f"   ✅ Плюсы: {', '.join(source['pros'])}")
                print(f"   ❌ Минусы: {', '.join(source['cons'])}")
                if 'cost' in source:
                    print(f"   💰 Стоимость: {source['cost']}")

def main():
    """Основная функция"""
    print("🔍 ИССЛЕДОВАНИЕ ИСТОЧНИКОВ ДАННЫХ ММВБ")
    print("=" * 60)
    
    moex = MOEXDataSources()
    
    # Показываем доступные источники
    moex.get_tick_data_sources()
    
    # Тестируем API ММВБ
    print("\n🧪 ТЕСТИРОВАНИЕ API ММВБ:")
    print("-" * 30)
    moex.test_moex_api()
    
    # Получаем список инструментов
    print("\n📋 ИНСТРУМЕНТЫ ММВБ:")
    print("-" * 30)
    instruments = moex.get_moex_instruments()
    if instruments is not None:
        print(instruments.to_string(index=False))
    
    # Создаем скрипт для загрузки
    print("\n📝 СОЗДАНИЕ СКРИПТА ЗАГРУЗКИ:")
    print("-" * 30)
    moex.create_data_download_script()
    
    # Генерируем рекомендации
    moex.generate_recommendations()
    
    print("\n✅ Исследование завершено!")
    print("\n📁 Созданы директории:")
    print("   - models/ (для сохранения обученных моделей)")
    print("   - data/tick_data/ (для тиковых данных)")
    print("   - download_tick_data.py (скрипт загрузки)")

if __name__ == "__main__":
    main()
