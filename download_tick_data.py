#!/usr/bin/env python3
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
    
    print(f"\n📊 Загружено данных для {len(results)} инструментов")
