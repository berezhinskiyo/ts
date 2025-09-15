#!/usr/bin/env python3
"""
Проверка доступности высокочастотных данных (тики, секунды, минуты)
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv('config/environments/.env')

async def check_high_frequency_data():
    """Проверка высокочастотных данных"""
    try:
        from tinkoff.invest import (
            AsyncClient, 
            AssetsRequest, 
            InstrumentType, 
            InstrumentStatus,
            CandleInterval
        )
        
        token = os.getenv('TBANK_TOKEN')
        if not token:
            print("❌ Не найден токен T-Bank API")
            return
        
        print("🚀 ПРОВЕРКА ВЫСОКОЧАСТОТНЫХ ДАННЫХ T-BANK API")
        print("=" * 60)
        
        async with AsyncClient(token) as client:
            results = {}
            
            # Тестируем разные типы инструментов
            test_types = [
                (InstrumentType.INSTRUMENT_TYPE_SHARE, "Акции"),
                (InstrumentType.INSTRUMENT_TYPE_FUTURES, "Фьючерсы"),
                (InstrumentType.INSTRUMENT_TYPE_CURRENCY, "Валютные пары"),
            ]
            
            # Интервалы для проверки
            intervals = [
                (CandleInterval.CANDLE_INTERVAL_1_MIN, "1 минута"),
                (CandleInterval.CANDLE_INTERVAL_5_MIN, "5 минут"),
                (CandleInterval.CANDLE_INTERVAL_15_MIN, "15 минут"),
                (CandleInterval.CANDLE_INTERVAL_HOUR, "1 час"),
                (CandleInterval.CANDLE_INTERVAL_DAY, "1 день"),
            ]
            
            for instrument_type, type_name in test_types:
                print(f"\n🔍 Тестируем {type_name}...")
                
                try:
                    request = AssetsRequest(
                        instrument_type=instrument_type,
                        instrument_status=InstrumentStatus.INSTRUMENT_STATUS_BASE,
                    )
                    
                    response = await client.instruments.get_assets(request=request)
                    
                    if response.assets:
                        print(f"✅ Найдено {len(response.assets)} {type_name.lower()}")
                        
                        # Тестируем первые 3 инструмента
                        sample_size = min(3, len(response.assets))
                        instruments_data = []
                        
                        for i, asset in enumerate(response.assets[:sample_size]):
                            if asset.instruments:
                                instrument = asset.instruments[0]
                                print(f"  📊 {instrument.ticker} - проверяем интервалы...")
                                
                                instrument_intervals = {}
                                
                                for interval, interval_name in intervals:
                                    try:
                                        # Проверяем разные периоды для каждого интервала
                                        periods = [
                                            (1, "1 день"),
                                            (7, "1 неделя"),
                                            (30, "1 месяц"),
                                        ]
                                        
                                        interval_data = {}
                                        
                                        for days, period_name in periods:
                                            try:
                                                end_date = datetime.now()
                                                start_date = end_date - timedelta(days=days)
                                                
                                                candles = await client.market_data.get_candles(
                                                    figi=instrument.figi,
                                                    from_=start_date,
                                                    to=end_date,
                                                    interval=interval
                                                )
                                                
                                                if candles.candles:
                                                    first_candle = candles.candles[0]
                                                    last_candle = candles.candles[-1]
                                                    
                                                    interval_data[period_name] = {
                                                        'candles_count': len(candles.candles),
                                                        'first_date': first_candle.time.isoformat(),
                                                        'last_date': last_candle.time.isoformat(),
                                                        'first_price': first_candle.close.units + first_candle.close.nano / 1e9,
                                                        'last_price': last_candle.close.units + last_candle.close.nano / 1e9,
                                                    }
                                                    
                                                    print(f"    ✅ {interval_name} ({period_name}): {len(candles.candles)} свечей")
                                                else:
                                                    interval_data[period_name] = {
                                                        'candles_count': 0,
                                                        'error': 'Нет данных'
                                                    }
                                                    print(f"    ❌ {interval_name} ({period_name}): нет данных")
                                                    
                                            except Exception as e:
                                                interval_data[period_name] = {
                                                    'candles_count': 0,
                                                    'error': str(e)
                                                }
                                                print(f"    ❌ {interval_name} ({period_name}): ошибка - {e}")
                                        
                                        instrument_intervals[interval_name] = interval_data
                                        
                                    except Exception as e:
                                        print(f"    ❌ {interval_name}: ошибка - {e}")
                                        instrument_intervals[interval_name] = {'error': str(e)}
                                
                                instruments_data.append({
                                    'ticker': instrument.ticker,
                                    'figi': instrument.figi,
                                    'class_code': instrument.class_code,
                                    'instrument_type': instrument.instrument_type,
                                    'intervals': instrument_intervals
                                })
                        
                        results[type_name] = {
                            'total_count': len(response.assets),
                            'sample_analyzed': sample_size,
                            'instruments': instruments_data
                        }
                        
                    else:
                        print(f"❌ {type_name} не найдены")
                        
                except Exception as e:
                    print(f"❌ Ошибка при анализе {type_name}: {e}")
            
            # Сохраняем результаты
            filename = f"high_frequency_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Результаты сохранены в: {filename}")
            
            # Анализируем результаты
            print(f"\n📊 АНАЛИЗ ВЫСОКОЧАСТОТНЫХ ДАННЫХ:")
            
            for type_name, type_data in results.items():
                if type_data['instruments']:
                    print(f"\n🎯 {type_name}:")
                    
                    for instrument in type_data['instruments']:
                        print(f"  📈 {instrument['ticker']}:")
                        
                        for interval_name, interval_data in instrument['intervals'].items():
                            if 'error' not in interval_data:
                                # Находим максимальное количество свечей
                                max_candles = 0
                                max_period = ""
                                
                                for period_name, period_data in interval_data.items():
                                    if period_data.get('candles_count', 0) > max_candles:
                                        max_candles = period_data['candles_count']
                                        max_period = period_name
                                
                                if max_candles > 0:
                                    print(f"    ✅ {interval_name}: до {max_candles} свечей ({max_period})")
                                else:
                                    print(f"    ❌ {interval_name}: нет данных")
                            else:
                                print(f"    ❌ {interval_name}: {interval_data['error']}")
            
            return results
            
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
    except Exception as e:
        print(f"❌ Общая ошибка: {e}")

async def main():
    """Основная функция"""
    results = await check_high_frequency_data()
    
    if results:
        print(f"\n🎯 АНАЛИЗ ЗАВЕРШЕН УСПЕШНО!")
        
        # Создаем сводку по интервалам
        print(f"\n📋 СВОДКА ПО ИНТЕРВАЛАМ:")
        
        interval_summary = {}
        for type_name, type_data in results.items():
            if type_data['instruments']:
                for instrument in type_data['instruments']:
                    for interval_name, interval_data in instrument['intervals'].items():
                        if interval_name not in interval_summary:
                            interval_summary[interval_name] = {'available': 0, 'total': 0}
                        
                        interval_summary[interval_name]['total'] += 1
                        
                        if 'error' not in interval_data:
                            # Проверяем, есть ли хотя бы один период с данными
                            has_data = any(
                                period_data.get('candles_count', 0) > 0 
                                for period_data in interval_data.values() 
                                if isinstance(period_data, dict)
                            )
                            if has_data:
                                interval_summary[interval_name]['available'] += 1
        
        for interval_name, summary in interval_summary.items():
            percentage = (summary['available'] / summary['total']) * 100 if summary['total'] > 0 else 0
            print(f"  {interval_name}: {summary['available']}/{summary['total']} ({percentage:.1f}%)")
    else:
        print(f"\n❌ АНАЛИЗ НЕ УДАЛСЯ")

if __name__ == "__main__":
    asyncio.run(main())
