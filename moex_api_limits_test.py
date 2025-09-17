#!/usr/bin/env python3
"""
Тестирование ограничений и лимитов MOEX API
"""

import requests
import time
import json
from datetime import datetime, timedelta
import pandas as pd
import asyncio
import aiohttp

class MOEXAPILimitsTester:
    """Класс для тестирования ограничений MOEX API"""
    
    def __init__(self):
        self.base_url = "https://iss.moex.com/iss/"
        self.results = {}
    
    def test_rate_limits(self):
        """Тестирование лимитов запросов в секунду"""
        print("🔄 Тестирование лимитов запросов...")
        
        # Тестируем разные интервалы между запросами
        intervals = [0.1, 0.2, 0.5, 1.0, 2.0]
        url = f"{self.base_url}engines/stock/markets/shares/boards/TQBR/securities/SBER/candles.json"
        params = {
            'from': '2024-09-15',
            'till': '2024-09-16',
            'interval': 1
        }
        
        for interval in intervals:
            print(f"  Тестируем интервал {interval}с...")
            success_count = 0
            error_count = 0
            start_time = time.time()
            
            for i in range(10):  # 10 запросов
                try:
                    response = requests.get(url, params=params, timeout=10)
                    if response.status_code == 200:
                        success_count += 1
                    else:
                        error_count += 1
                        print(f"    Ошибка {response.status_code} на запросе {i+1}")
                except Exception as e:
                    error_count += 1
                    print(f"    Исключение на запросе {i+1}: {e}")
                
                if i < 9:  # Не ждем после последнего запроса
                    time.sleep(interval)
            
            elapsed_time = time.time() - start_time
            success_rate = (success_count / 10) * 100
            
            print(f"    Результат: {success_count}/10 успешных ({success_rate:.1f}%) за {elapsed_time:.1f}с")
            
            self.results[f'rate_limit_{interval}s'] = {
                'success_count': success_count,
                'error_count': error_count,
                'success_rate': success_rate,
                'elapsed_time': elapsed_time
            }
    
    def test_data_limits(self):
        """Тестирование ограничений на объем данных"""
        print("\n📊 Тестирование ограничений объема данных...")
        
        # Тестируем разные периоды
        periods = [1, 7, 30, 90, 365]  # дни
        
        for days in periods:
            print(f"  Тестируем период {days} дней...")
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            url = f"{self.base_url}engines/stock/markets/shares/boards/TQBR/securities/SBER/candles.json"
            params = {
                'from': start_date.strftime('%Y-%m-%d'),
                'till': end_date.strftime('%Y-%m-%d'),
                'interval': 1  # 1 минута
            }
            
            try:
                start_time = time.time()
                response = requests.get(url, params=params, timeout=30)
                elapsed_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    candles = data['candles']['data']
                    
                    print(f"    ✅ Успешно: {len(candles)} записей за {elapsed_time:.2f}с")
                    
                    self.results[f'data_limit_{days}d'] = {
                        'success': True,
                        'records_count': len(candles),
                        'response_time': elapsed_time,
                        'status_code': response.status_code
                    }
                else:
                    print(f"    ❌ Ошибка {response.status_code}")
                    self.results[f'data_limit_{days}d'] = {
                        'success': False,
                        'status_code': response.status_code,
                        'response_time': elapsed_time
                    }
                    
            except Exception as e:
                print(f"    ❌ Исключение: {e}")
                self.results[f'data_limit_{days}d'] = {
                    'success': False,
                    'error': str(e)
                }
    
    def test_concurrent_requests(self):
        """Тестирование одновременных запросов"""
        print("\n⚡ Тестирование одновременных запросов...")
        
        async def make_request(session, symbol, i):
            url = f"{self.base_url}engines/stock/markets/shares/boards/TQBR/securities/{symbol}/candles.json"
            params = {
                'from': '2024-09-15',
                'till': '2024-09-16',
                'interval': 1
            }
            
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        candles = data['candles']['data']
                        return {'success': True, 'records': len(candles), 'request_id': i}
                    else:
                        return {'success': False, 'status': response.status, 'request_id': i}
            except Exception as e:
                return {'success': False, 'error': str(e), 'request_id': i}
        
        async def test_concurrent():
            symbols = ['SBER', 'GAZP', 'LKOH', 'NVTK', 'ROSN']
            
            # Тестируем разное количество одновременных запросов
            for concurrent_count in [1, 3, 5, 10]:
                print(f"  Тестируем {concurrent_count} одновременных запросов...")
                
                start_time = time.time()
                
                async with aiohttp.ClientSession() as session:
                    tasks = []
                    for i in range(concurrent_count):
                        symbol = symbols[i % len(symbols)]
                        task = make_request(session, symbol, i)
                        tasks.append(task)
                    
                    results = await asyncio.gather(*tasks)
                
                elapsed_time = time.time() - start_time
                success_count = sum(1 for r in results if r['success'])
                
                print(f"    Результат: {success_count}/{concurrent_count} успешных за {elapsed_time:.2f}с")
                
                self.results[f'concurrent_{concurrent_count}'] = {
                    'success_count': success_count,
                    'total_count': concurrent_count,
                    'success_rate': (success_count / concurrent_count) * 100,
                    'elapsed_time': elapsed_time,
                    'results': results
                }
        
        # Запускаем асинхронный тест
        try:
            asyncio.run(test_concurrent())
        except Exception as e:
            print(f"    ❌ Ошибка асинхронного тестирования: {e}")
    
    def test_different_endpoints(self):
        """Тестирование разных эндпоинтов API"""
        print("\n🔗 Тестирование разных эндпоинтов...")
        
        endpoints = [
            {
                'name': 'Свечи (candles)',
                'url': 'engines/stock/markets/shares/boards/TQBR/securities/SBER/candles.json',
                'params': {'from': '2024-09-15', 'till': '2024-09-16', 'interval': 1}
            },
            {
                'name': 'История (history)',
                'url': 'engines/stock/markets/shares/boards/TQBR/securities/SBER/history.json',
                'params': {'from': '2024-09-15', 'till': '2024-09-16'}
            },
            {
                'name': 'Список инструментов',
                'url': 'engines/stock/markets/shares/boards/TQBR/securities.json',
                'params': {}
            },
            {
                'name': 'Статистика торгов',
                'url': 'engines/stock/markets/shares/boards/TQBR/securities/SBER/trades.json',
                'params': {'from': '2024-09-15', 'till': '2024-09-16'}
            }
        ]
        
        for endpoint in endpoints:
            print(f"  Тестируем {endpoint['name']}...")
            
            try:
                start_time = time.time()
                response = requests.get(
                    f"{self.base_url}{endpoint['url']}", 
                    params=endpoint['params'], 
                    timeout=10
                )
                elapsed_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    # Подсчитываем количество записей в первом ключе
                    first_key = list(data.keys())[0] if data else None
                    records_count = len(data[first_key]['data']) if first_key and 'data' in data[first_key] else 0
                    
                    print(f"    ✅ Успешно: {records_count} записей за {elapsed_time:.2f}с")
                    
                    self.results[f'endpoint_{endpoint["name"].replace(" ", "_").lower()}'] = {
                        'success': True,
                        'records_count': records_count,
                        'response_time': elapsed_time,
                        'status_code': response.status_code
                    }
                else:
                    print(f"    ❌ Ошибка {response.status_code}")
                    self.results[f'endpoint_{endpoint["name"].replace(" ", "_").lower()}'] = {
                        'success': False,
                        'status_code': response.status_code,
                        'response_time': elapsed_time
                    }
                    
            except Exception as e:
                print(f"    ❌ Исключение: {e}")
                self.results[f'endpoint_{endpoint["name"].replace(" ", "_").lower()}'] = {
                    'success': False,
                    'error': str(e)
                }
    
    def test_historical_depth(self):
        """Тестирование глубины исторических данных"""
        print("\n📅 Тестирование глубины исторических данных...")
        
        # Тестируем разные годы
        years = [2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015]
        
        for year in years:
            print(f"  Тестируем данные за {year} год...")
            
            start_date = f"{year}-01-01"
            end_date = f"{year}-01-07"  # Первая неделя года
            
            url = f"{self.base_url}engines/stock/markets/shares/boards/TQBR/securities/SBER/candles.json"
            params = {
                'from': start_date,
                'till': end_date,
                'interval': 1
            }
            
            try:
                start_time = time.time()
                response = requests.get(url, params=params, timeout=15)
                elapsed_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    candles = data['candles']['data']
                    
                    print(f"    ✅ Успешно: {len(candles)} записей за {elapsed_time:.2f}с")
                    
                    self.results[f'historical_{year}'] = {
                        'success': True,
                        'records_count': len(candles),
                        'response_time': elapsed_time,
                        'status_code': response.status_code
                    }
                else:
                    print(f"    ❌ Ошибка {response.status_code}")
                    self.results[f'historical_{year}'] = {
                        'success': False,
                        'status_code': response.status_code,
                        'response_time': elapsed_time
                    }
                    
            except Exception as e:
                print(f"    ❌ Исключение: {e}")
                self.results[f'historical_{year}'] = {
                    'success': False,
                    'error': str(e)
                }
    
    def generate_report(self):
        """Генерация отчета по ограничениям"""
        print("\n📋 ОТЧЕТ ПО ОГРАНИЧЕНИЯМ MOEX API")
        print("=" * 60)
        
        # Анализ лимитов запросов
        print("\n🔄 ЛИМИТЫ ЗАПРОСОВ:")
        rate_limits = {k: v for k, v in self.results.items() if k.startswith('rate_limit_')}
        if rate_limits:
            for interval, result in rate_limits.items():
                interval_val = interval.replace('rate_limit_', '').replace('s', '')
                print(f"  {interval_val}с между запросами: {result['success_rate']:.1f}% успешных")
        
        # Анализ ограничений данных
        print("\n📊 ОГРАНИЧЕНИЯ ОБЪЕМА ДАННЫХ:")
        data_limits = {k: v for k, v in self.results.items() if k.startswith('data_limit_')}
        if data_limits:
            for period, result in data_limits.items():
                days = period.replace('data_limit_', '').replace('d', '')
                if result['success']:
                    print(f"  {days} дней: ✅ {result['records_count']} записей за {result['response_time']:.2f}с")
                else:
                    print(f"  {days} дней: ❌ Ошибка {result.get('status_code', 'Unknown')}")
        
        # Анализ одновременных запросов
        print("\n⚡ ОДНОВРЕМЕННЫЕ ЗАПРОСЫ:")
        concurrent = {k: v for k, v in self.results.items() if k.startswith('concurrent_')}
        if concurrent:
            for count, result in concurrent.items():
                req_count = count.replace('concurrent_', '')
                print(f"  {req_count} запросов: {result['success_rate']:.1f}% успешных за {result['elapsed_time']:.2f}с")
        
        # Анализ эндпоинтов
        print("\n🔗 ДОСТУПНЫЕ ЭНДПОИНТЫ:")
        endpoints = {k: v for k, v in self.results.items() if k.startswith('endpoint_')}
        if endpoints:
            for endpoint, result in endpoints.items():
                name = endpoint.replace('endpoint_', '').replace('_', ' ').title()
                if result['success']:
                    print(f"  {name}: ✅ {result['records_count']} записей")
                else:
                    print(f"  {name}: ❌ Ошибка {result.get('status_code', 'Unknown')}")
        
        # Анализ исторических данных
        print("\n📅 ГЛУБИНА ИСТОРИЧЕСКИХ ДАННЫХ:")
        historical = {k: v for k, v in self.results.items() if k.startswith('historical_')}
        if historical:
            available_years = []
            for year, result in historical.items():
                year_val = year.replace('historical_', '')
                if result['success']:
                    available_years.append(int(year_val))
                    print(f"  {year_val}: ✅ {result['records_count']} записей")
                else:
                    print(f"  {year_val}: ❌ Недоступно")
            
            if available_years:
                print(f"\n  📊 Доступны данные с {min(available_years)} по {max(available_years)} год")
        
        # Рекомендации
        print("\n💡 РЕКОМЕНДАЦИИ:")
        print("  1. Используйте интервал не менее 0.5-1 секунды между запросами")
        print("  2. Ограничьте период запроса 30-90 днями для оптимальной производительности")
        print("  3. Не делайте более 3-5 одновременных запросов")
        print("  4. Кэшируйте данные для избежания повторных запросов")
        print("  5. Используйте эндпоинт 'candles' для получения OHLCV данных")
        
        # Сохраняем результаты
        with open('moex_api_limits_results.json', 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Результаты сохранены в moex_api_limits_results.json")

def main():
    """Основная функция тестирования"""
    print("🧪 ТЕСТИРОВАНИЕ ОГРАНИЧЕНИЙ MOEX API")
    print("=" * 60)
    
    tester = MOEXAPILimitsTester()
    
    # Запускаем все тесты
    tester.test_rate_limits()
    tester.test_data_limits()
    tester.test_concurrent_requests()
    tester.test_different_endpoints()
    tester.test_historical_depth()
    
    # Генерируем отчет
    tester.generate_report()
    
    print("\n✅ Тестирование завершено!")

if __name__ == "__main__":
    main()
