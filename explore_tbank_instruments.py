#!/usr/bin/env python3
"""
Исследование доступных инструментов в T-Bank API
Поиск срочных инструментов (фьючерсы, опционы)
"""

import asyncio
import json
import os
from datetime import datetime
from typing import List, Dict
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Импорты для T-Bank API
try:
    from tinkoff_invest import ProductionSession, SandboxSession
    from tinkoff_invest.exceptions import RequestProcessingError
    from dotenv import load_dotenv
    load_dotenv('config/environments/.env')
except ImportError as e:
    logger.error(f"Ошибка импорта: {e}")
    logger.error("Установите зависимости: pip install tinkoff-invest python-dotenv")
    exit(1)

class TBankInstrumentExplorer:
    """Исследователь инструментов T-Bank API"""
    
    def __init__(self):
        self.token = os.getenv('TBANK_TOKEN')
        self.sandbox_token = os.getenv('TBANK_SANDBOX_TOKEN')
        self.use_sandbox = os.getenv('USE_SANDBOX', 'True').lower() == 'true'
        self.client = None
        
        if not self.token and not self.sandbox_token:
            logger.error("❌ Не найдены токены T-Bank API в .env файле")
            exit(1)
    
    async def __aenter__(self):
        token = self.sandbox_token if self.use_sandbox else self.token
        if self.use_sandbox:
            self.client = SandboxSession(token)
        else:
            self.client = ProductionSession(token)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # ProductionSession не имеет метода close
        pass
    
    async def search_instruments(self, query: str) -> List[Dict]:
        """Поиск инструментов по запросу"""
        try:
            # Пробуем найти по тикеру
            try:
                instrument = await self.client.get_instrument_by_ticker(ticker=query)
                if instrument:
                    return [{
                        'figi': instrument.figi,
                        'ticker': instrument.ticker,
                        'name': instrument.name,
                        'currency': instrument.currency,
                        'instrument_type': instrument.instrument_type,
                        'class_code': getattr(instrument, 'class_code', ''),
                        'exchange': getattr(instrument, 'exchange', ''),
                        'lot': getattr(instrument, 'lot', 1),
                        'min_price_increment': getattr(instrument, 'min_price_increment', None)
                    }]
            except:
                pass
            
            # Если не найден по тикеру, возвращаем пустой список
            return []
        except Exception as e:
            logger.error(f"Ошибка поиска инструментов '{query}': {e}")
            return []
    
    async def get_all_instrument_types(self) -> Dict[str, List[Dict]]:
        """Получение всех типов инструментов"""
        logger.info("🔍 Поиск всех типов инструментов...")
        
        # Поисковые запросы для разных типов инструментов
        search_queries = [
            # Акции
            "SBER", "GAZP", "LKOH", "YNDX", "ROSN",
            
            # Фьючерсы
            "фьючерс", "futures", "FUT", "Si", "RTS", "BR", "GZ",
            
            # Опционы
            "опцион", "option", "OPT", "call", "put",
            
            # Облигации
            "облигация", "bond", "OFZ", "корпоративная",
            
            # Валютные пары
            "USD", "EUR", "валютная пара", "currency",
            
            # ETF
            "ETF", "фонд", "индексный фонд",
            
            # Криптовалюты
            "биткоин", "bitcoin", "BTC", "эфириум", "ethereum", "ETH",
            
            # Товары
            "золото", "gold", "нефть", "oil", "серебро", "silver"
        ]
        
        all_instruments = {}
        
        for query in search_queries:
            logger.info(f"🔍 Поиск: {query}")
            instruments = await self.search_instruments(query)
            
            if instruments:
                # Группируем по типам инструментов
                for instrument in instruments:
                    instrument_type = instrument['instrument_type']
                    if instrument_type not in all_instruments:
                        all_instruments[instrument_type] = []
                    all_instruments[instrument_type].append(instrument)
        
        return all_instruments
    
    async def get_futures_instruments(self) -> List[Dict]:
        """Получение фьючерсов"""
        logger.info("📈 Поиск фьючерсов...")
        
        futures_queries = [
            "Si",      # Доллар/рубль
            "Eu",      # Евро/рубль
            "BR",      # Нефть Brent
            "GZ",      # Газпром
            "SBER",    # Сбербанк
            "RTS",     # Индекс RTS
            "MX",      # Индекс МосБиржи
            "GD",      # Золото
            "SV",      # Серебро
            "CU",      # Медь
            "WZ",      # Пшеница
            "CO",      # Кукуруза
            "SU",      # Сахар
            "futures", "фьючерс"
        ]
        
        futures = []
        for query in futures_queries:
            instruments = await self.search_instruments(query)
            for instrument in instruments:
                if 'futures' in instrument['instrument_type'].lower() or \
                   'фьючерс' in instrument['name'].lower() or \
                   instrument['ticker'].endswith('F') or \
                   'FUT' in instrument['ticker']:
                    futures.append(instrument)
        
        # Удаляем дубликаты
        unique_futures = []
        seen_figis = set()
        for future in futures:
            if future['figi'] not in seen_figis:
                unique_futures.append(future)
                seen_figis.add(future['figi'])
        
        return unique_futures
    
    async def get_options_instruments(self) -> List[Dict]:
        """Получение опционов"""
        logger.info("📊 Поиск опционов...")
        
        options_queries = [
            "опцион", "option", "OPT", "call", "put",
            "SBER", "GAZP", "LKOH", "RTS"
        ]
        
        options = []
        for query in options_queries:
            instruments = await self.search_instruments(query)
            for instrument in instruments:
                if 'option' in instrument['instrument_type'].lower() or \
                   'опцион' in instrument['name'].lower() or \
                   'OPT' in instrument['ticker']:
                    options.append(instrument)
        
        # Удаляем дубликаты
        unique_options = []
        seen_figis = set()
        for option in options:
            if option['figi'] not in seen_figis:
                unique_options.append(option)
                seen_figis.add(option['figi'])
        
        return unique_options
    
    async def get_historical_data_sample(self, figi: str, days: int = 30) -> Dict:
        """Получение образца исторических данных"""
        try:
            from datetime import timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            response = await self.client.get_candles(
                figi=figi,
                from_=start_date,
                to=end_date,
                interval='CANDLE_INTERVAL_DAY'
            )
            
            if response.candles:
                return {
                    'figi': figi,
                    'candles_count': len(response.candles),
                    'first_candle': {
                        'date': response.candles[0].time.isoformat(),
                        'open': response.candles[0].open.units + response.candles[0].open.nano / 1e9,
                        'close': response.candles[0].close.units + response.candles[0].close.nano / 1e9,
                        'volume': response.candles[0].volume
                    },
                    'last_candle': {
                        'date': response.candles[-1].time.isoformat(),
                        'open': response.candles[-1].open.units + response.candles[-1].open.nano / 1e9,
                        'close': response.candles[-1].close.units + response.candles[-1].close.nano / 1e9,
                        'volume': response.candles[-1].volume
                    }
                }
            else:
                return {'figi': figi, 'error': 'No historical data available'}
                
        except Exception as e:
            return {'figi': figi, 'error': str(e)}
    
    async def explore_all(self):
        """Полное исследование всех инструментов"""
        logger.info("🚀 Начинаем исследование инструментов T-Bank API...")
        
        # Получаем все типы инструментов
        all_instruments = await self.get_all_instrument_types()
        
        # Получаем фьючерсы отдельно
        futures = await self.get_futures_instruments()
        
        # Получаем опционы отдельно
        options = await self.get_options_instruments()
        
        # Собираем результаты
        results = {
            'timestamp': datetime.now().isoformat(),
            'use_sandbox': self.use_sandbox,
            'total_instrument_types': len(all_instruments),
            'instrument_types': {},
            'futures': futures,
            'options': options,
            'summary': {
                'total_futures': len(futures),
                'total_options': len(options),
                'total_instruments_by_type': {k: len(v) for k, v in all_instruments.items()}
            }
        }
        
        # Добавляем информацию о типах инструментов
        for instrument_type, instruments in all_instruments.items():
            results['instrument_types'][instrument_type] = {
                'count': len(instruments),
                'sample': instruments[:5]  # Первые 5 для примера
            }
        
        # Тестируем исторические данные для нескольких инструментов
        logger.info("📊 Тестируем исторические данные...")
        test_instruments = []
        
        # Берем по одному инструменту каждого типа
        for instrument_type, instruments in all_instruments.items():
            if instruments:
                test_instruments.append(instruments[0]['figi'])
        
        # Добавляем фьючерсы и опционы
        if futures:
            test_instruments.append(futures[0]['figi'])
        if options:
            test_instruments.append(options[0]['figi'])
        
        # Тестируем исторические данные
        historical_data_tests = []
        for figi in test_instruments[:10]:  # Ограничиваем до 10 тестов
            logger.info(f"📈 Тестируем исторические данные для {figi}")
            test_result = await self.get_historical_data_sample(figi)
            historical_data_tests.append(test_result)
        
        results['historical_data_tests'] = historical_data_tests
        
        return results

async def main():
    """Основная функция"""
    print("🔍 ИССЛЕДОВАНИЕ ИНСТРУМЕНТОВ T-BANK API")
    print("=" * 50)
    
    async with TBankInstrumentExplorer() as explorer:
        results = await explorer.explore_all()
        
        # Сохраняем результаты
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tbank_instruments_exploration_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Выводим краткую сводку
        print(f"\n📊 РЕЗУЛЬТАТЫ ИССЛЕДОВАНИЯ:")
        print(f"✅ Используется: {'Sandbox' if results['use_sandbox'] else 'Production'}")
        print(f"📈 Всего типов инструментов: {results['total_instrument_types']}")
        print(f"🎯 Фьючерсов найдено: {results['summary']['total_futures']}")
        print(f"📊 Опционов найдено: {results['summary']['total_options']}")
        
        print(f"\n📋 ТИПЫ ИНСТРУМЕНТОВ:")
        for instrument_type, info in results['instrument_types'].items():
            print(f"  • {instrument_type}: {info['count']} инструментов")
        
        if results['futures']:
            print(f"\n🎯 НАЙДЕННЫЕ ФЬЮЧЕРСЫ:")
            for future in results['futures'][:10]:  # Показываем первые 10
                print(f"  • {future['ticker']} - {future['name']} ({future['instrument_type']})")
        
        if results['options']:
            print(f"\n📊 НАЙДЕННЫЕ ОПЦИОНЫ:")
            for option in results['options'][:10]:  # Показываем первые 10
                print(f"  • {option['ticker']} - {option['name']} ({option['instrument_type']})")
        
        print(f"\n💾 Полные результаты сохранены в: {filename}")

if __name__ == "__main__":
    asyncio.run(main())
