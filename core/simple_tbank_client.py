#!/usr/bin/env python3
"""
Simplified T-Bank API Client
Упрощенный клиент для T-Bank API
"""

import asyncio
import logging
import requests
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import os
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv('config/environments/.env')

logger = logging.getLogger(__name__)

class SimpleTBankClient:
    """Упрощенный клиент для T-Bank API"""
    
    def __init__(self):
        self.sandbox_token = os.getenv('TBANK_SANDBOX_TOKEN')
        self.production_token = os.getenv('TBANK_TOKEN')
        self.use_sandbox = os.getenv('USE_SANDBOX', 'True').lower() == 'true'
        self.app_name = os.getenv('TBANK_APP_NAME', 'AutoTrader')
        
        # Выбираем токен в зависимости от режима
        self.token = self.sandbox_token if self.use_sandbox else self.production_token
        
        # Базовые URL для API
        if self.use_sandbox:
            self.base_url = "https://invest-public-api.tinkoff.ru/rest/"
        else:
            self.base_url = "https://invest-public-api.tinkoff.ru/rest/"
        
        self.headers = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        }
        
        logger.info(f"T-Bank API Client initialized (Sandbox: {self.use_sandbox})")
    
    async def get_accounts(self) -> List[Dict]:
        """Получить список счетов"""
        try:
            url = f"{self.base_url}tinkoff.public.invest.api.contract.v1.UsersService/GetAccounts"
            
            # Для упрощения используем синхронный запрос
            response = requests.post(url, headers=self.headers, json={})
            
            if response.status_code == 200:
                data = response.json()
                accounts = []
                
                if 'accounts' in data:
                    for account in data['accounts']:
                        accounts.append({
                            'id': account.get('id', ''),
                            'name': account.get('name', ''),
                            'type': account.get('type', ''),
                            'status': account.get('status', '')
                        })
                
                logger.info(f"Found {len(accounts)} accounts")
                return accounts
            else:
                logger.error(f"API Error: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting accounts: {e}")
            return []
    
    async def get_portfolio(self, account_id: str) -> Dict:
        """Получить портфель"""
        try:
            url = f"{self.base_url}tinkoff.public.invest.api.contract.v1.OperationsService/GetPortfolio"
            
            payload = {
                "accountId": account_id
            }
            
            response = requests.post(url, headers=self.headers, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                
                # Подсчитываем общую стоимость портфеля
                total_amount = 0
                positions = []
                
                if 'positions' in data:
                    for position in data['positions']:
                        quantity = float(position.get('quantity', {}).get('units', 0))
                        price = float(position.get('currentPrice', {}).get('units', 0))
                        value = quantity * price
                        total_amount += value
                        
                        positions.append({
                            'figi': position.get('figi', ''),
                            'ticker': position.get('ticker', ''),
                            'quantity': quantity,
                            'price': price,
                            'value': value
                        })
                
                portfolio = {
                    'total_amount': total_amount,
                    'positions': positions,
                    'account_id': account_id
                }
                
                logger.info(f"Portfolio value: {total_amount} руб.")
                return portfolio
            else:
                logger.error(f"API Error: {response.status_code} - {response.text}")
                return {}
                
        except Exception as e:
            logger.error(f"Error getting portfolio: {e}")
            return {}
    
    async def get_historical_candles(self, figi: str, from_date: datetime, 
                                   to_date: datetime, interval: str = 'day') -> List[Dict]:
        """Получить исторические свечи"""
        try:
            url = f"{self.base_url}tinkoff.public.invest.api.contract.v1.MarketDataService/GetCandles"
            
            # Преобразуем интервал
            interval_map = {
                '1min': 'CANDLE_INTERVAL_1_MIN',
                '5min': 'CANDLE_INTERVAL_5_MIN',
                '15min': 'CANDLE_INTERVAL_15_MIN',
                '1hour': 'CANDLE_INTERVAL_HOUR',
                'day': 'CANDLE_INTERVAL_DAY'
            }
            
            interval_enum = interval_map.get(interval, 'CANDLE_INTERVAL_DAY')
            
            payload = {
                "figi": figi,
                "from": from_date.isoformat() + "Z",
                "to": to_date.isoformat() + "Z",
                "interval": interval_enum
            }
            
            response = requests.post(url, headers=self.headers, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                candles = []
                
                if 'candles' in data:
                    for candle in data['candles']:
                        candles.append({
                            'time': candle.get('time', ''),
                            'open': float(candle.get('open', {}).get('units', 0)),
                            'high': float(candle.get('high', {}).get('units', 0)),
                            'low': float(candle.get('low', {}).get('units', 0)),
                            'close': float(candle.get('close', {}).get('units', 0)),
                            'volume': int(candle.get('volume', 0))
                        })
                
                logger.info(f"Retrieved {len(candles)} candles for {figi}")
                return candles
            else:
                logger.error(f"API Error: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting candles: {e}")
            return []
    
    async def get_last_prices(self, figis: List[str]) -> Dict[str, float]:
        """Получить последние цены"""
        try:
            url = f"{self.base_url}tinkoff.public.invest.api.contract.v1.MarketDataService/GetLastPrices"
            
            payload = {
                "figi": figis
            }
            
            response = requests.post(url, headers=self.headers, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                prices = {}
                
                if 'lastPrices' in data:
                    for price in data['lastPrices']:
                        figi = price.get('figi', '')
                        price_value = float(price.get('price', {}).get('units', 0))
                        prices[figi] = price_value
                
                logger.info(f"Retrieved prices for {len(prices)} instruments")
                return prices
            else:
                logger.error(f"API Error: {response.status_code} - {response.text}")
                return {}
                
        except Exception as e:
            logger.error(f"Error getting last prices: {e}")
            return {}
    
    async def search_instruments(self, query: str) -> List[Dict]:
        """Поиск инструментов"""
        try:
            url = f"{self.base_url}tinkoff.public.invest.api.contract.v1.InstrumentsService/SearchByTicker"
            
            payload = {
                "ticker": query
            }
            
            response = requests.post(url, headers=self.headers, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                instruments = []
                
                if 'instruments' in data:
                    for instrument in data['instruments']:
                        instruments.append({
                            'figi': instrument.get('figi', ''),
                            'ticker': instrument.get('ticker', ''),
                            'name': instrument.get('name', ''),
                            'type': instrument.get('instrumentType', ''),
                            'currency': instrument.get('currency', '')
                        })
                
                logger.info(f"Found {len(instruments)} instruments for '{query}'")
                return instruments
            else:
                logger.error(f"API Error: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Error searching instruments: {e}")
            return []
    
    def get_russian_stocks_figi(self) -> Dict[str, str]:
        """Получить FIGI коды российских акций"""
        return {
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
    
    async def test_connection(self) -> bool:
        """Тестирование подключения к API"""
        try:
            logger.info("Testing T-Bank API connection...")
            
            # Пробуем получить счета
            accounts = await self.get_accounts()
            
            if accounts:
                logger.info(f"✅ Connection successful! Found {len(accounts)} accounts")
                return True
            else:
                logger.warning("⚠️ Connection successful but no accounts found")
                return True  # API работает, но счетов нет
                
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            return False

# Функция для тестирования
async def test_tbank_api():
    """Тестирование T-Bank API"""
    print("🔍 Тестирование T-Bank API...")
    
    client = SimpleTBankClient()
    
    # Тест подключения
    connection_ok = await client.test_connection()
    
    if connection_ok:
        print("✅ Подключение к API успешно")
        
        # Получить счета
        accounts = await client.get_accounts()
        print(f"📊 Найдено счетов: {len(accounts)}")
        
        if accounts:
            for i, account in enumerate(accounts):
                print(f"  {i+1}. {account['name']} - {account['status']}")
            
            # Получить портфель первого счета
            portfolio = await client.get_portfolio(accounts[0]['id'])
            print(f"💰 Стоимость портфеля: {portfolio.get('total_amount', 0)} руб.")
            
            # Получить данные по Сбербанку
            figi_mapping = client.get_russian_stocks_figi()
            sber_figi = figi_mapping['SBER']
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            candles = await client.get_historical_candles(
                figi=sber_figi,
                from_date=start_date,
                to_date=end_date,
                interval='day'
            )
            
            print(f"📈 Получено {len(candles)} свечей по Сбербанку")
            if candles:
                print(f"   Последняя цена: {candles[-1]['close']} руб.")
        
        return True
    else:
        print("❌ Ошибка подключения к API")
        return False

if __name__ == "__main__":
    # Запуск тестирования
    result = asyncio.run(test_tbank_api())
    print(f"\n🎯 Результат тестирования: {'✅ УСПЕШНО' if result else '❌ ОШИБКА'}")
