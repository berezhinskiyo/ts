#!/usr/bin/env python3
"""
Обновление реальных данных из T-Bank API для лучших инструментов
"""

import asyncio
import json
import os
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv('config/environments/.env')

async def update_tbank_data():
    """Обновление данных из T-Bank API"""
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
        
        print("🚀 ОБНОВЛЕНИЕ РЕАЛЬНЫХ ДАННЫХ T-BANK API")
        print("=" * 50)
        
        async with AsyncClient(token) as client:
            # Лучшие инструменты для тестирования
            target_instruments = [
                {'ticker': 'PIKK', 'type': 'futures', 'description': 'Фьючерс PIKK - лучший для высокочастотной торговли'},
                {'ticker': 'IRAO', 'type': 'futures', 'description': 'Фьючерс IRAO - отличные данные'},
                {'ticker': 'SGZH', 'type': 'futures', 'description': 'Фьючерс SGZH - хорошие данные'},
                {'ticker': 'GAZP', 'type': 'share', 'description': 'Акция Газпром - популярная российская акция'},
                {'ticker': 'SBER', 'type': 'share', 'description': 'Акция Сбербанк - ликвидная акция'},
            ]
            
            # Создаем директорию для данных
            data_dir = 'data/tbank_real'
            os.makedirs(data_dir, exist_ok=True)
            
            updated_instruments = []
            
            for instrument_info in target_instruments:
                ticker = instrument_info['ticker']
                print(f"\n🔍 Ищем {ticker} ({instrument_info['description']})...")
                
                try:
                    # Определяем тип инструмента
                    if instrument_info['type'] == 'futures':
                        instrument_type = InstrumentType.INSTRUMENT_TYPE_FUTURES
                    else:
                        instrument_type = InstrumentType.INSTRUMENT_TYPE_SHARE
                    
                    # Ищем инструмент
                    request = AssetsRequest(
                        instrument_type=instrument_type,
                        instrument_status=InstrumentStatus.INSTRUMENT_STATUS_BASE,
                    )
                    
                    response = await client.instruments.get_assets(request=request)
                    
                    found_instrument = None
                    if response.assets:
                        for asset in response.assets:
                            if asset.instruments:
                                for instrument in asset.instruments:
                                    if instrument.ticker == ticker:
                                        found_instrument = instrument
                                        break
                                if found_instrument:
                                    break
                    
                    if found_instrument:
                        print(f"✅ Найден {ticker}: {found_instrument.figi}")
                        
                        # Получаем исторические данные за разные периоды
                        periods = [
                            (30, "30 дней", "1M"),
                            (90, "3 месяца", "3M"),
                            (365, "1 год", "1Y"),
                        ]
                        
                        instrument_data = {
                            'ticker': ticker,
                            'figi': found_instrument.figi,
                            'class_code': found_instrument.class_code,
                            'instrument_type': found_instrument.instrument_type,
                            'description': instrument_info['description'],
                            'periods': {}
                        }
                        
                        for days, period_name, period_code in periods:
                            print(f"  📊 Загружаем {period_name}...")
                            
                            try:
                                end_date = datetime.now()
                                start_date = end_date - timedelta(days=days)
                                
                                # Получаем дневные данные
                                candles = await client.market_data.get_candles(
                                    figi=found_instrument.figi,
                                    from_=start_date,
                                    to=end_date,
                                    interval=CandleInterval.CANDLE_INTERVAL_DAY
                                )
                                
                                if candles.candles:
                                    # Конвертируем в DataFrame
                                    data = []
                                    for candle in candles.candles:
                                        data.append({
                                            'date': candle.time.date(),
                                            'open': candle.open.units + candle.open.nano / 1e9,
                                            'high': candle.high.units + candle.high.nano / 1e9,
                                            'low': candle.low.units + candle.low.nano / 1e9,
                                            'close': candle.close.units + candle.close.nano / 1e9,
                                            'volume': candle.volume
                                        })
                                    
                                    df = pd.DataFrame(data)
                                    df.set_index('date', inplace=True)
                                    df.sort_index(inplace=True)
                                    
                                    # Сохраняем данные
                                    filename = f"{ticker}_{period_code}_tbank.csv"
                                    filepath = os.path.join(data_dir, filename)
                                    df.to_csv(filepath)
                                    
                                    instrument_data['periods'][period_code] = {
                                        'days': days,
                                        'candles': len(candles.candles),
                                        'first_date': df.index[0].strftime('%Y-%m-%d'),
                                        'last_date': df.index[-1].strftime('%Y-%m-%d'),
                                        'filename': filename
                                    }
                                    
                                    print(f"    ✅ {len(candles.candles)} свечей, период: {df.index[0].date()} - {df.index[-1].date()}")
                                else:
                                    print(f"    ❌ Нет данных за {period_name}")
                                    
                            except Exception as e:
                                print(f"    ❌ Ошибка загрузки {period_name}: {e}")
                        
                        updated_instruments.append(instrument_data)
                        
                    else:
                        print(f"❌ {ticker} не найден")
                        
                except Exception as e:
                    print(f"❌ Ошибка поиска {ticker}: {e}")
            
            # Сохраняем метаданные
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'total_instruments': len(updated_instruments),
                'data_directory': data_dir,
                'instruments': updated_instruments
            }
            
            metadata_file = os.path.join(data_dir, 'metadata.json')
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Метаданные сохранены в: {metadata_file}")
            
            # Выводим сводку
            print(f"\n📊 СВОДКА ОБНОВЛЕНИЯ:")
            print(f"✅ Инструментов обновлено: {len(updated_instruments)}")
            
            for instrument in updated_instruments:
                print(f"\n🎯 {instrument['ticker']} ({instrument['description']}):")
                for period_code, period_data in instrument['periods'].items():
                    print(f"  • {period_code}: {period_data['candles']} свечей ({period_data['first_date']} - {period_data['last_date']})")
            
            return updated_instruments
            
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
    except Exception as e:
        print(f"❌ Общая ошибка: {e}")

async def main():
    """Основная функция"""
    instruments = await update_tbank_data()
    
    if instruments:
        print(f"\n🎯 ОБНОВЛЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        print(f"📁 Данные сохранены в: data/tbank_real/")
    else:
        print(f"\n❌ ОБНОВЛЕНИЕ НЕ УДАЛОСЬ")

if __name__ == "__main__":
    asyncio.run(main())

