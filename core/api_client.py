import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from tinkoff.invest import Client, RequestError, InstrumentIdType
from tinkoff.invest.schemas import (
    OrderDirection, OrderType, CandleInterval, 
    HistoricCandle, Quotation
)
from config import Config

logger = logging.getLogger(__name__)

class TBankAPIClient:
    """Client for T-Bank API interaction"""
    
    def __init__(self):
        self.token = Config.TBANK_SANDBOX_TOKEN if Config.USE_SANDBOX else Config.TBANK_TOKEN
        self.app_name = Config.TBANK_APP_NAME
        self.client = None
        
    async def __aenter__(self):
        self.client = Client(self.token, app_name=self.app_name)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.close()
    
    def quotation_to_decimal(self, quotation: Quotation) -> float:
        """Convert Quotation to decimal"""
        return quotation.units + quotation.nano / 1e9
    
    def decimal_to_quotation(self, value: float) -> Quotation:
        """Convert decimal to Quotation"""
        units = int(value)
        nano = int((value - units) * 1e9)
        return Quotation(units=units, nano=nano)
    
    async def get_accounts(self) -> List[Dict]:
        """Get all trading accounts"""
        try:
            response = await self.client.users.get_accounts()
            return [
                {
                    'id': account.id,
                    'name': account.name,
                    'type': account.type,
                    'status': account.status
                }
                for account in response.accounts
            ]
        except RequestError as e:
            logger.error(f"Error getting accounts: {e}")
            return []
    
    async def get_portfolio(self, account_id: str) -> Dict:
        """Get portfolio information"""
        try:
            response = await self.client.operations.get_portfolio(account_id=account_id)
            
            total_amount = self.quotation_to_decimal(response.total_amount_shares)
            expected_yield = self.quotation_to_decimal(response.expected_yield)
            
            positions = []
            for position in response.positions:
                positions.append({
                    'figi': position.figi,
                    'instrument_type': position.instrument_type,
                    'quantity': self.quotation_to_decimal(position.quantity),
                    'average_position_price': self.quotation_to_decimal(position.average_position_price),
                    'expected_yield': self.quotation_to_decimal(position.expected_yield),
                    'current_nkd': self.quotation_to_decimal(position.current_nkd),
                    'current_price': self.quotation_to_decimal(position.current_price)
                })
            
            return {
                'total_amount': total_amount,
                'expected_yield': expected_yield,
                'positions': positions
            }
        except RequestError as e:
            logger.error(f"Error getting portfolio: {e}")
            return {}
    
    async def get_historical_candles(self, figi: str, from_date: datetime, 
                                   to_date: datetime, interval: CandleInterval) -> List[Dict]:
        """Get historical candles for instrument"""
        try:
            response = await self.client.market_data.get_candles(
                figi=figi,
                from_=from_date,
                to=to_date,
                interval=interval
            )
            
            candles = []
            for candle in response.candles:
                candles.append({
                    'time': candle.time,
                    'open': self.quotation_to_decimal(candle.open),
                    'high': self.quotation_to_decimal(candle.high),
                    'low': self.quotation_to_decimal(candle.low),
                    'close': self.quotation_to_decimal(candle.close),
                    'volume': candle.volume
                })
            
            return candles
        except RequestError as e:
            logger.error(f"Error getting candles for {figi}: {e}")
            return []
    
    async def get_last_prices(self, figis: List[str]) -> Dict[str, float]:
        """Get last prices for instruments"""
        try:
            response = await self.client.market_data.get_last_prices(figi=figis)
            
            prices = {}
            for price in response.last_prices:
                prices[price.figi] = self.quotation_to_decimal(price.price)
            
            return prices
        except RequestError as e:
            logger.error(f"Error getting last prices: {e}")
            return {}
    
    async def search_instruments(self, query: str) -> List[Dict]:
        """Search for instruments"""
        try:
            response = await self.client.instruments.find_instrument(query=query)
            
            instruments = []
            for instrument in response.instruments:
                instruments.append({
                    'figi': instrument.figi,
                    'ticker': instrument.ticker,
                    'name': instrument.name,
                    'currency': instrument.currency,
                    'instrument_type': instrument.instrument_type
                })
            
            return instruments
        except RequestError as e:
            logger.error(f"Error searching instruments: {e}")
            return []
    
    async def post_order(self, account_id: str, figi: str, quantity: int, 
                        direction: OrderDirection, order_type: OrderType,
                        price: Optional[float] = None) -> Dict:
        """Post trading order"""
        try:
            order_request = {
                'figi': figi,
                'quantity': quantity,
                'direction': direction,
                'account_id': account_id,
                'order_type': order_type,
                'order_id': f"order_{datetime.now().timestamp()}"
            }
            
            if price is not None:
                order_request['price'] = self.decimal_to_quotation(price)
            
            response = await self.client.orders.post_order(**order_request)
            
            return {
                'order_id': response.order_id,
                'execution_report_status': response.execution_report_status,
                'lots_requested': response.lots_requested,
                'lots_executed': response.lots_executed,
                'initial_order_price': self.quotation_to_decimal(response.initial_order_price),
                'executed_order_price': self.quotation_to_decimal(response.executed_order_price),
                'total_order_amount': self.quotation_to_decimal(response.total_order_amount),
                'figi': response.figi,
                'direction': response.direction,
                'initial_security_price': self.quotation_to_decimal(response.initial_security_price)
            }
        except RequestError as e:
            logger.error(f"Error posting order: {e}")
            return {}
    
    async def cancel_order(self, account_id: str, order_id: str) -> bool:
        """Cancel order"""
        try:
            await self.client.orders.cancel_order(account_id=account_id, order_id=order_id)
            return True
        except RequestError as e:
            logger.error(f"Error canceling order: {e}")
            return False
    
    async def get_orders(self, account_id: str) -> List[Dict]:
        """Get active orders"""
        try:
            response = await self.client.orders.get_orders(account_id=account_id)
            
            orders = []
            for order in response.orders:
                orders.append({
                    'order_id': order.order_id,
                    'figi': order.figi,
                    'direction': order.direction,
                    'initial_security_price': self.quotation_to_decimal(order.initial_security_price),
                    'lots_requested': order.lots_requested,
                    'lots_executed': order.lots_executed,
                    'initial_order_price': self.quotation_to_decimal(order.initial_order_price)
                })
            
            return orders
        except RequestError as e:
            logger.error(f"Error getting orders: {e}")
            return []