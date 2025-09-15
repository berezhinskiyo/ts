#!/usr/bin/env python3
"""
Полный анализ всех доступных инструментов в T-Bank API
с проверкой исторических данных
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv('config/environments/.env')

async def analyze_all_instruments():
    """Анализ всех доступных инструментов"""
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
        
        print("🚀 ПОЛНЫЙ АНАЛИЗ ИНСТРУМЕНТОВ T-BANK API")
        print("=" * 60)
        
        async with AsyncClient(token) as client:
            all_instruments = {}
            
            # Список типов инструментов для анализа
            instrument_types = [
                (InstrumentType.INSTRUMENT_TYPE_SHARE, "Акции"),
                (InstrumentType.INSTRUMENT_TYPE_BOND, "Облигации"),
                (InstrumentType.INSTRUMENT_TYPE_FUTURES, "Фьючерсы"),
                (InstrumentType.INSTRUMENT_TYPE_OPTION, "Опционы"),
                (InstrumentType.INSTRUMENT_TYPE_CURRENCY, "Валютные пары"),
                (InstrumentType.INSTRUMENT_TYPE_ETF, "ETF"),
                (InstrumentType.INSTRUMENT_TYPE_COMMODITY, "Товары"),
            ]
            
            for instrument_type, type_name in instrument_types:
                print(f"\n🔍 Анализируем {type_name}...")
                
                try:
                    # Получаем все инструменты данного типа
                    request = AssetsRequest(
                        instrument_type=instrument_type,
                        instrument_status=InstrumentStatus.INSTRUMENT_STATUS_BASE,
                    )
                    
                    response = await client.instruments.get_assets(request=request)
                    
                    if response.assets:
                        print(f"✅ Найдено {len(response.assets)} {type_name.lower()}")
                        
                        # Анализируем первые 10 активов для проверки истории
                        sample_size = min(10, len(response.assets))
                        instruments_with_history = []
                        
                        for i, asset in enumerate(response.assets[:sample_size]):
                            # Каждый asset содержит список instruments
                            if asset.instruments:
                                instrument = asset.instruments[0]  # Берем первый инструмент из актива
                                print(f"  📊 Проверяем {i+1}/{sample_size}: {instrument.ticker}")
                                
                                # Проверяем исторические данные
                                history_info = await check_historical_data(client, instrument.figi, instrument.ticker)
                            
                                if history_info:
                                    instruments_with_history.append({
                                        'ticker': instrument.ticker,
                                        'figi': instrument.figi,
                                        'class_code': instrument.class_code,
                                        'instrument_type': instrument.instrument_type,
                                        'instrument_kind': instrument.instrument_kind,
                                        'uid': instrument.uid,
                                        'history': history_info
                                    })
                        
                        all_instruments[type_name] = {
                            'total_count': len(response.assets),
                            'sample_analyzed': sample_size,
                            'with_history': len(instruments_with_history),
                            'instruments': instruments_with_history
                        }
                        
                        print(f"  📈 Из {sample_size} проверенных, {len(instruments_with_history)} имеют исторические данные")
                    else:
                        print(f"❌ {type_name} не найдены")
                        all_instruments[type_name] = {
                            'total_count': 0,
                            'sample_analyzed': 0,
                            'with_history': 0,
                            'instruments': []
                        }
                        
                except Exception as e:
                    print(f"❌ Ошибка при анализе {type_name}: {e}")
                    all_instruments[type_name] = {
                        'total_count': 0,
                        'sample_analyzed': 0,
                        'with_history': 0,
                        'instruments': [],
                        'error': str(e)
                    }
            
            # Сохраняем результаты
            results = {
                'timestamp': datetime.now().isoformat(),
                'analysis_summary': {
                    'total_types_analyzed': len(instrument_types),
                    'types_with_instruments': len([t for t in all_instruments.values() if t['total_count'] > 0]),
                    'total_instruments_found': sum(t['total_count'] for t in all_instruments.values()),
                    'total_with_history': sum(t['with_history'] for t in all_instruments.values())
                },
                'instruments_by_type': all_instruments
            }
            
            filename = f"full_instruments_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Результаты сохранены в: {filename}")
            
            # Выводим сводку
            print(f"\n📊 СВОДКА АНАЛИЗА:")
            print(f"✅ Типов инструментов проанализировано: {results['analysis_summary']['total_types_analyzed']}")
            print(f"✅ Типов с инструментами: {results['analysis_summary']['types_with_instruments']}")
            print(f"✅ Всего инструментов найдено: {results['analysis_summary']['total_instruments_found']}")
            print(f"✅ С историческими данными: {results['analysis_summary']['total_with_history']}")
            
            return results
            
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        print("Установите: pip install git+https://github.com/RussianInvestments/invest-python.git")
    except Exception as e:
        print(f"❌ Общая ошибка: {e}")

async def check_historical_data(client, figi, ticker):
    """Проверка исторических данных для инструмента"""
    try:
        from tinkoff.invest import CandleInterval
        
        # Проверяем разные периоды
        periods = [
            (30, "30 дней"),
            (90, "3 месяца"),
            (365, "1 год"),
            (1095, "3 года")  # 3 года
        ]
        
        history_info = {
            'figi': figi,
            'ticker': ticker,
            'periods': {}
        }
        
        for days, period_name in periods:
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                candles = await client.market_data.get_candles(
                    figi=figi,
                    from_=start_date,
                    to=end_date,
                    interval=CandleInterval.CANDLE_INTERVAL_DAY
                )
                
                if candles.candles:
                    # Анализируем данные
                    first_candle = candles.candles[0]
                    last_candle = candles.candles[-1]
                    
                    # Проверяем качество данных
                    valid_candles = 0
                    for candle in candles.candles:
                        if (candle.open.units > 0 and candle.close.units > 0 and 
                            candle.high.units > 0 and candle.low.units > 0):
                            valid_candles += 1
                    
                    data_quality = valid_candles / len(candles.candles) if candles.candles else 0
                    
                    history_info['periods'][period_name] = {
                        'candles_count': len(candles.candles),
                        'valid_candles': valid_candles,
                        'data_quality': round(data_quality * 100, 2),
                        'first_date': first_candle.time.date().isoformat(),
                        'last_date': last_candle.time.date().isoformat(),
                        'first_price': first_candle.close.units + first_candle.close.nano / 1e9,
                        'last_price': last_candle.close.units + last_candle.close.nano / 1e9,
                        'price_change': round(
                            ((last_candle.close.units + last_candle.close.nano / 1e9) / 
                             (first_candle.close.units + first_candle.close.nano / 1e9) - 1) * 100, 2
                        ) if first_candle.close.units > 0 else 0
                    }
                    
                    print(f"    ✅ {period_name}: {len(candles.candles)} свечей, качество {data_quality*100:.1f}%")
                else:
                    history_info['periods'][period_name] = {
                        'candles_count': 0,
                        'error': 'Нет данных'
                    }
                    print(f"    ❌ {period_name}: нет данных")
                    
            except Exception as e:
                history_info['periods'][period_name] = {
                    'candles_count': 0,
                    'error': str(e)
                }
                print(f"    ❌ {period_name}: ошибка - {e}")
        
        # Определяем максимальную глубину истории
        max_period = None
        max_candles = 0
        
        for period_name, period_data in history_info['periods'].items():
            if period_data.get('candles_count', 0) > max_candles:
                max_candles = period_data['candles_count']
                max_period = period_name
        
        history_info['max_history_period'] = max_period
        history_info['max_candles'] = max_candles
        
        return history_info if max_candles > 0 else None
        
    except Exception as e:
        print(f"    ❌ Ошибка проверки истории: {e}")
        return None

async def main():
    """Основная функция"""
    results = await analyze_all_instruments()
    
    if results:
        print(f"\n🎯 АНАЛИЗ ЗАВЕРШЕН УСПЕШНО!")
        
        # Показываем лучшие инструменты по глубине истории
        print(f"\n🏆 ИНСТРУМЕНТЫ С НАИБОЛЬШЕЙ ГЛУБИНОЙ ИСТОРИИ:")
        
        all_with_history = []
        for type_name, type_data in results['instruments_by_type'].items():
            for instrument in type_data['instruments']:
                if instrument['history']:
                    all_with_history.append({
                        'type': type_name,
                        'ticker': instrument['ticker'],
                        'figi': instrument['figi'],
                        'max_candles': instrument['history']['max_candles'],
                        'max_period': instrument['history']['max_history_period']
                    })
        
        # Сортируем по количеству свечей
        all_with_history.sort(key=lambda x: x['max_candles'], reverse=True)
        
        for i, instrument in enumerate(all_with_history[:10]):  # Топ-10
            print(f"  {i+1}. {instrument['ticker']} ({instrument['type']}) - {instrument['max_candles']} свечей ({instrument['max_period']})")
    else:
        print(f"\n❌ АНАЛИЗ НЕ УДАЛСЯ")

if __name__ == "__main__":
    asyncio.run(main())
