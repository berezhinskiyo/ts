#!/usr/bin/env python3
"""
Update Test Data from T-Bank API
Обновление тестовых данных из T-Bank API
"""

import asyncio
import os
import requests
import pandas as pd
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv('config/environments/.env')

class TBankDataUpdater:
    """Обновление данных из T-Bank API"""
    
    def __init__(self):
        self.token = os.getenv('TBANK_TOKEN')
        self.base_url = "https://invest-public-api.tinkoff.ru/rest/"
        self.headers = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        }
        
        # FIGI коды российских акций
        self.figi_mapping = {
            'SBER': 'BBG004730N88',     # Сбербанк
            'GAZP': 'BBG004730ZJ9',     # Газпром
            'LKOH': 'BBG004730JJ5',     # ЛУКОЙЛ
            'YNDX': 'BBG006L8G4H1',     # Яндекс
            'ROSN': 'BBG0047315Y7',     # Роснефть
            'NVTK': 'BBG00475KKY8',     # НОВАТЭК
            'TATN': 'BBG004RVFFC0',     # Татнефть
            'MTSS': 'BBG004730N88',     # МТС
            'MGNT': 'BBG004RVFFC0',     # Магнит
            'RTKM': 'BBG004RVFFC0',     # Ростелеком
            'OZON': 'BBG006L8G4H1',     # OZON
            'FIVE': 'BBG004RVFFC0'      # X5 Retail Group
        }
        
        print(f"🔧 T-Bank Data Updater initialized")
        print(f"🔑 Token: {self.token[:20]}...")
        print(f"📊 Instruments: {len(self.figi_mapping)}")
    
    def make_request(self, endpoint, payload=None):
        """Выполнение запроса к API"""
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.post(url, headers=self.headers, json=payload or {})
            return response
        except Exception as e:
            print(f"❌ Ошибка запроса: {e}")
            return None
    
    def get_historical_data(self, figi, days=252):
        """Получение исторических данных"""
        print(f"📈 Получение данных для {figi} за {days} дней...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        payload = {
            "figi": figi,
            "from": start_date.isoformat() + "Z",
            "to": end_date.isoformat() + "Z",
            "interval": "CANDLE_INTERVAL_DAY"
        }
        
        response = self.make_request("tinkoff.public.invest.api.contract.v1.MarketDataService/GetCandles", payload)
        
        if response and response.status_code == 200:
            try:
                data = response.json()
                if 'candles' in data and data['candles']:
                    candles = []
                    for candle in data['candles']:
                        candles.append({
                            'date': candle['time'][:10],  # YYYY-MM-DD
                            'open': float(candle['open']['units']),
                            'high': float(candle['high']['units']),
                            'low': float(candle['low']['units']),
                            'close': float(candle['close']['units']),
                            'volume': int(candle['volume'])
                        })
                    
                    df = pd.DataFrame(candles)
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    df.sort_index(inplace=True)
                    
                    print(f"✅ Получено {len(df)} свечей")
                    return df
                else:
                    print(f"⚠️ Данные не найдены")
                    return None
            except Exception as e:
                print(f"❌ Ошибка обработки данных: {e}")
                return None
        else:
            print(f"❌ Ошибка API: {response.status_code if response else 'No response'}")
            return None
    
    def get_last_prices(self, figis):
        """Получение последних цен"""
        print(f"💲 Получение последних цен для {len(figis)} инструментов...")
        
        payload = {"figi": figis}
        response = self.make_request("tinkoff.public.invest.api.contract.v1.MarketDataService/GetLastPrices", payload)
        
        if response and response.status_code == 200:
            try:
                data = response.json()
                prices = {}
                if 'lastPrices' in data:
                    for price in data['lastPrices']:
                        prices[price['figi']] = float(price['price']['units'])
                
                print(f"✅ Получено цен: {len(prices)}")
                return prices
            except Exception as e:
                print(f"❌ Ошибка обработки цен: {e}")
                return {}
        else:
            print(f"❌ Ошибка API: {response.status_code if response else 'No response'}")
            return {}
    
    def update_all_data(self):
        """Обновление всех данных"""
        print("🚀 Начинаем обновление данных из T-Bank API...")
        print("=" * 60)
        
        all_data = {}
        last_prices = {}
        
        # Получаем исторические данные для каждого инструмента
        for ticker, figi in self.figi_mapping.items():
            print(f"\n📊 Обработка {ticker} ({figi})...")
            
            # Получаем исторические данные
            df = self.get_historical_data(figi, days=252)  # 1 год данных
            
            if df is not None:
                all_data[ticker] = df
                print(f"✅ {ticker}: {len(df)} дней, цена: {df['close'].iloc[-1]:.2f} руб.")
            else:
                print(f"❌ {ticker}: данные не получены")
        
        # Получаем последние цены
        print(f"\n💲 Получение последних цен...")
        figis = list(self.figi_mapping.values())
        last_prices = self.get_last_prices(figis)
        
        # Создаем обратное соответствие FIGI -> Ticker
        figi_to_ticker = {v: k for k, v in self.figi_mapping.items()}
        
        print(f"\n📋 Последние цены:")
        for figi, price in last_prices.items():
            ticker = figi_to_ticker.get(figi, 'Unknown')
            print(f"  {ticker}: {price:.2f} руб.")
        
        # Сохраняем данные
        self.save_data(all_data, last_prices)
        
        return all_data, last_prices
    
    def save_data(self, all_data, last_prices):
        """Сохранение данных"""
        print(f"\n💾 Сохранение данных...")
        
        # Создаем директорию для данных
        os.makedirs('data/real_time', exist_ok=True)
        os.makedirs('data/historical', exist_ok=True)
        
        # Сохраняем исторические данные
        for ticker, df in all_data.items():
            filename = f"data/historical/{ticker}_tbank.csv"
            df.to_csv(filename)
            print(f"✅ {ticker}: сохранено в {filename}")
        
        # Сохраняем последние цены
        prices_data = {
            'timestamp': datetime.now().isoformat(),
            'prices': last_prices,
            'figi_mapping': self.figi_mapping
        }
        
        with open('data/real_time/last_prices.json', 'w', encoding='utf-8') as f:
            json.dump(prices_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Последние цены сохранены в data/real_time/last_prices.json")
        
        # Создаем сводный отчет
        self.create_summary_report(all_data, last_prices)
    
    def create_summary_report(self, all_data, last_prices):
        """Создание сводного отчета"""
        print(f"\n📊 Создание сводного отчета...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_source': 'T-Bank API',
            'instruments': {},
            'summary': {
                'total_instruments': len(all_data),
                'successful_downloads': len([d for d in all_data.values() if d is not None]),
                'total_data_points': sum(len(df) for df in all_data.values() if df is not None)
            }
        }
        
        # Информация по каждому инструменту
        for ticker, df in all_data.items():
            if df is not None:
                report['instruments'][ticker] = {
                    'figi': self.figi_mapping[ticker],
                    'data_points': len(df),
                    'date_range': {
                        'start': df.index[0].isoformat(),
                        'end': df.index[-1].isoformat()
                    },
                    'price_range': {
                        'min': float(df['low'].min()),
                        'max': float(df['high'].max()),
                        'last': float(df['close'].iloc[-1])
                    },
                    'volume_avg': int(df['volume'].mean())
                }
        
        # Сохраняем отчет
        with open('data/real_time/data_summary.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Сводный отчет сохранен в data/real_time/data_summary.json")
        
        # Выводим краткую статистику
        print(f"\n📈 СТАТИСТИКА ОБНОВЛЕНИЯ:")
        print(f"  📊 Инструментов: {report['summary']['total_instruments']}")
        print(f"  ✅ Успешно загружено: {report['summary']['successful_downloads']}")
        print(f"  📅 Всего точек данных: {report['summary']['total_data_points']}")
        
        return report

def main():
    """Главная функция"""
    print("🔄 T-Bank Data Updater")
    print("=" * 60)
    
    # Создаем обновлятор
    updater = TBankDataUpdater()
    
    # Обновляем данные
    all_data, last_prices = updater.update_all_data()
    
    print(f"\n🎉 ОБНОВЛЕНИЕ ЗАВЕРШЕНО!")
    print(f"✅ Данные обновлены из T-Bank API")
    print(f"📊 Готово для тестирования стратегий")
    
    return all_data, last_prices

if __name__ == "__main__":
    main()
