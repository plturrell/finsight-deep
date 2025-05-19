"""Market Data Provider for real-time financial data integration"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
import aiohttp
import websockets
from abc import ABC, abstractmethod

# Financial data providers
import yfinance as yf
import finnhub
import alpaca_trade_api as tradeapi
from polygon import RESTClient as PolygonClient
from binance import AsyncClient as BinanceClient
import ccxt.async_support as ccxt


@dataclass
class MarketQuote:
    """Market quote data"""
    symbol: str
    price: float
    bid: float
    ask: float
    volume: int
    timestamp: datetime
    source: str
    metadata: Dict[str, Any] = None


@dataclass
class HistoricalData:
    """Historical market data"""
    symbol: str
    timeframe: str
    data: pd.DataFrame
    source: str
    metadata: Dict[str, Any] = None


@dataclass
class MarketEvent:
    """Real-time market event"""
    event_type: str
    symbol: str
    data: Dict[str, Any]
    timestamp: datetime
    source: str


class DataProvider(str, Enum):
    """Available data providers"""
    YAHOO = "yahoo"
    FINNHUB = "finnhub"
    ALPACA = "alpaca"
    POLYGON = "polygon"
    BINANCE = "binance"
    COINBASE = "coinbase"
    INTERACTIVE_BROKERS = "ib"


class MarketDataProvider:
    """Unified market data provider with multiple sources"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.providers = {}
        self.active_subscriptions = {}
        self.event_handlers = {}
        self.cache = {}
        self.cache_ttl = config.get("cache_ttl", 60)  # seconds
        
        # Initialize providers
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize data providers based on configuration"""
        
        # Yahoo Finance (free, but limited)
        if self.config.get("yahoo", {}).get("enabled", True):
            self.providers[DataProvider.YAHOO] = YahooFinanceProvider()
        
        # Finnhub (free tier available)
        if finnhub_key := self.config.get("finnhub", {}).get("api_key"):
            self.providers[DataProvider.FINNHUB] = FinnhubProvider(finnhub_key)
        
        # Alpaca (free tier for market data)
        if alpaca_config := self.config.get("alpaca"):
            self.providers[DataProvider.ALPACA] = AlpacaProvider(
                api_key=alpaca_config.get("api_key"),
                api_secret=alpaca_config.get("api_secret"),
                base_url=alpaca_config.get("base_url", "https://paper-api.alpaca.markets")
            )
        
        # Polygon.io
        if polygon_key := self.config.get("polygon", {}).get("api_key"):
            self.providers[DataProvider.POLYGON] = PolygonProvider(polygon_key)
        
        # Binance (for crypto)
        if binance_config := self.config.get("binance"):
            self.providers[DataProvider.BINANCE] = BinanceProvider(
                api_key=binance_config.get("api_key"),
                api_secret=binance_config.get("api_secret")
            )
    
    # Real-time data methods
    async def get_real_time_quotes(self, symbols: List[str]) -> List[MarketQuote]:
        """Get real-time quotes for multiple symbols"""
        quotes = []
        
        # Check cache first
        cached_quotes = []
        uncached_symbols = []
        
        for symbol in symbols:
            cache_key = f"quote:{symbol}"
            if cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                if datetime.now() - timestamp < timedelta(seconds=self.cache_ttl):
                    cached_quotes.append(cached_data)
                    continue
            uncached_symbols.append(symbol)
        
        # Fetch uncached quotes
        if uncached_symbols:
            # Determine appropriate provider for each symbol
            provider_groups = self._group_symbols_by_provider(uncached_symbols)
            
            # Fetch from each provider asynchronously
            tasks = []
            for provider, provider_symbols in provider_groups.items():
                if provider in self.providers:
                    tasks.append(
                        self.providers[provider].get_quotes(provider_symbols)
                    )
            
            # Gather results
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, list):
                    for quote in result:
                        quotes.append(quote)
                        # Cache the quote
                        cache_key = f"quote:{quote.symbol}"
                        self.cache[cache_key] = (quote, datetime.now())
        
        return cached_quotes + quotes
    
    async def get_historical_data(
        self,
        symbol: str,
        period: str,
        interval: str = "1d",
        provider: Optional[DataProvider] = None
    ) -> HistoricalData:
        """Get historical market data"""
        # Check cache
        cache_key = f"historical:{symbol}:{period}:{interval}"
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < timedelta(seconds=self.cache_ttl * 10):
                return cached_data
        
        # Determine provider
        if not provider:
            provider = self._get_best_provider_for_symbol(symbol)
        
        if provider not in self.providers:
            raise ValueError(f"Provider {provider} not available")
        
        # Fetch data
        historical_data = await self.providers[provider].get_historical_data(
            symbol, period, interval
        )
        
        # Cache the data
        self.cache[cache_key] = (historical_data, datetime.now())
        
        return historical_data
    
    async def subscribe_to_updates(
        self,
        symbols: List[str],
        callback: Callable[[MarketEvent], None],
        event_types: Optional[List[str]] = None
    ) -> str:
        """Subscribe to real-time market updates"""
        subscription_id = f"sub_{datetime.now().timestamp()}"
        
        # Group symbols by provider
        provider_groups = self._group_symbols_by_provider(symbols)
        
        # Subscribe with each provider
        for provider, provider_symbols in provider_groups.items():
            if provider in self.providers and hasattr(self.providers[provider], 'subscribe'):
                await self.providers[provider].subscribe(
                    provider_symbols,
                    lambda event: self._handle_market_event(event, callback),
                    event_types
                )
        
        # Store subscription info
        self.active_subscriptions[subscription_id] = {
            "symbols": symbols,
            "callback": callback,
            "event_types": event_types,
            "providers": list(provider_groups.keys())
        }
        
        return subscription_id
    
    async def unsubscribe(self, subscription_id: str):
        """Unsubscribe from market updates"""
        if subscription_id not in self.active_subscriptions:
            return
        
        subscription = self.active_subscriptions[subscription_id]
        
        # Unsubscribe from each provider
        for provider in subscription["providers"]:
            if provider in self.providers and hasattr(self.providers[provider], 'unsubscribe'):
                await self.providers[provider].unsubscribe(subscription["symbols"])
        
        del self.active_subscriptions[subscription_id]
    
    # Technical indicators
    async def get_technical_indicators(
        self,
        symbol: str,
        indicators: List[str],
        period: str = "1y",
        interval: str = "1d"
    ) -> Dict[str, pd.Series]:
        """Calculate technical indicators"""
        # Get historical data
        historical = await self.get_historical_data(symbol, period, interval)
        df = historical.data
        
        results = {}
        
        for indicator in indicators:
            if indicator == "sma":
                results["sma_20"] = self._calculate_sma(df["close"], 20)
                results["sma_50"] = self._calculate_sma(df["close"], 50)
                results["sma_200"] = self._calculate_sma(df["close"], 200)
            
            elif indicator == "ema":
                results["ema_12"] = self._calculate_ema(df["close"], 12)
                results["ema_26"] = self._calculate_ema(df["close"], 26)
            
            elif indicator == "rsi":
                results["rsi"] = self._calculate_rsi(df["close"])
            
            elif indicator == "macd":
                macd, signal, histogram = self._calculate_macd(df["close"])
                results["macd"] = macd
                results["macd_signal"] = signal
                results["macd_histogram"] = histogram
            
            elif indicator == "bollinger":
                upper, middle, lower = self._calculate_bollinger_bands(df["close"])
                results["bb_upper"] = upper
                results["bb_middle"] = middle
                results["bb_lower"] = lower
            
            elif indicator == "atr":
                results["atr"] = self._calculate_atr(df)
        
        return results
    
    # Market data aggregation
    async def get_market_summary(self) -> Dict[str, Any]:
        """Get overall market summary"""
        # Major indices
        indices = ["^GSPC", "^DJI", "^IXIC", "^VIX"]  # S&P 500, Dow, Nasdaq, VIX
        
        # Get quotes
        quotes = await self.get_real_time_quotes(indices)
        
        # Get sector performance
        sectors = [
            "XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLB", "XLRE", "XLU"
        ]
        sector_quotes = await self.get_real_time_quotes(sectors)
        
        # Calculate market breadth
        # In production, this would query advance/decline data
        
        return {
            "indices": {quote.symbol: {
                "price": quote.price,
                "change": self._calculate_change(quote),
                "volume": quote.volume
            } for quote in quotes},
            "sectors": {quote.symbol: {
                "price": quote.price,
                "change": self._calculate_change(quote)
            } for quote in sector_quotes},
            "market_breadth": {
                "advancing": 1500,  # Mock data
                "declining": 1000,
                "unchanged": 500
            },
            "timestamp": datetime.now().isoformat()
        }
    
    # News and sentiment
    async def get_market_news(
        self,
        symbols: Optional[List[str]] = None,
        category: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get market news"""
        news_items = []
        
        # Aggregate news from available providers
        for provider in self.providers.values():
            if hasattr(provider, 'get_news'):
                provider_news = await provider.get_news(symbols, category, limit)
                news_items.extend(provider_news)
        
        # Sort by timestamp and limit
        news_items.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return news_items[:limit]
    
    # Helper methods
    def _group_symbols_by_provider(self, symbols: List[str]) -> Dict[DataProvider, List[str]]:
        """Group symbols by appropriate provider"""
        groups = {}
        
        for symbol in symbols:
            # Determine provider based on symbol type
            if symbol.startswith("^"):  # Index
                provider = DataProvider.YAHOO
            elif "-USD" in symbol or "BTC" in symbol or "ETH" in symbol:  # Crypto
                provider = DataProvider.BINANCE
            else:  # Stocks
                # Use the most reliable available provider
                if DataProvider.POLYGON in self.providers:
                    provider = DataProvider.POLYGON
                elif DataProvider.ALPACA in self.providers:
                    provider = DataProvider.ALPACA
                else:
                    provider = DataProvider.YAHOO
            
            if provider not in groups:
                groups[provider] = []
            groups[provider].append(symbol)
        
        return groups
    
    def _get_best_provider_for_symbol(self, symbol: str) -> DataProvider:
        """Get the best available provider for a symbol"""
        if symbol.startswith("^"):
            return DataProvider.YAHOO
        elif "-USD" in symbol or "BTC" in symbol or "ETH" in symbol:
            if DataProvider.BINANCE in self.providers:
                return DataProvider.BINANCE
            else:
                return DataProvider.YAHOO
        else:
            if DataProvider.POLYGON in self.providers:
                return DataProvider.POLYGON
            elif DataProvider.ALPACA in self.providers:
                return DataProvider.ALPACA
            else:
                return DataProvider.YAHOO
    
    def _handle_market_event(self, event: MarketEvent, callback: Callable):
        """Handle market event from provider"""
        # Apply any filtering or transformation
        callback(event)
    
    def _calculate_change(self, quote: MarketQuote) -> Dict[str, float]:
        """Calculate price change (mock implementation)"""
        # In production, this would compare with previous close
        change_amount = 0.0
        change_percent = 0.0
        
        return {
            "amount": change_amount,
            "percent": change_percent
        }
    
    # Technical indicator calculations
    def _calculate_sma(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return prices.rolling(window=period).mean()
    
    def _calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return prices.ewm(span=period, adjust=False).mean()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(
        self,
        prices: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = self._calculate_ema(prices, fast_period)
        ema_slow = self._calculate_ema(prices, slow_period)
        
        macd = ema_fast - ema_slow
        signal = self._calculate_ema(macd, signal_period)
        histogram = macd - signal
        
        return macd, signal, histogram
    
    def _calculate_bollinger_bands(
        self,
        prices: pd.Series,
        period: int = 20,
        std_dev: int = 2
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        middle = self._calculate_sma(prices, period)
        std = prices.rolling(window=period).std()
        
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        
        return upper, middle, lower
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        # True Range calculation
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr


# Provider implementations
class YahooFinanceProvider:
    """Yahoo Finance data provider"""
    
    async def get_quotes(self, symbols: List[str]) -> List[MarketQuote]:
        """Get quotes from Yahoo Finance"""
        quotes = []
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                quote = MarketQuote(
                    symbol=symbol,
                    price=info.get("regularMarketPrice", 0),
                    bid=info.get("bid", 0),
                    ask=info.get("ask", 0),
                    volume=info.get("regularMarketVolume", 0),
                    timestamp=datetime.now(),
                    source="yahoo",
                    metadata=info
                )
                quotes.append(quote)
            except Exception as e:
                print(f"Error fetching {symbol} from Yahoo: {e}")
        
        return quotes
    
    async def get_historical_data(
        self,
        symbol: str,
        period: str,
        interval: str
    ) -> HistoricalData:
        """Get historical data from Yahoo Finance"""
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        # Rename columns to standard format
        df.columns = df.columns.str.lower()
        
        return HistoricalData(
            symbol=symbol,
            timeframe=interval,
            data=df,
            source="yahoo"
        )


class FinnhubProvider:
    """Finnhub data provider"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = finnhub.Client(api_key=api_key)
    
    async def get_quotes(self, symbols: List[str]) -> List[MarketQuote]:
        """Get quotes from Finnhub"""
        quotes = []
        
        # Use asyncio for concurrent requests
        async with aiohttp.ClientSession() as session:
            tasks = []
            for symbol in symbols:
                url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={self.api_key}"
                tasks.append(self._fetch_quote(session, url, symbol))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, MarketQuote):
                    quotes.append(result)
        
        return quotes
    
    async def _fetch_quote(
        self,
        session: aiohttp.ClientSession,
        url: str,
        symbol: str
    ) -> MarketQuote:
        """Fetch single quote"""
        async with session.get(url) as response:
            data = await response.json()
            
            return MarketQuote(
                symbol=symbol,
                price=data.get("c", 0),  # Current price
                bid=data.get("b", 0),    # Bid
                ask=data.get("a", 0),    # Ask
                volume=data.get("v", 0), # Volume
                timestamp=datetime.fromtimestamp(data.get("t", 0)),
                source="finnhub",
                metadata=data
            )
    
    async def get_news(
        self,
        symbols: Optional[List[str]] = None,
        category: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get news from Finnhub"""
        if symbols and len(symbols) == 1:
            # Company news
            news = self.client.company_news(
                symbols[0],
                _from=(datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
                to=datetime.now().strftime("%Y-%m-%d")
            )
        else:
            # General news
            news = self.client.general_news(category or "general")
        
        return news[:limit]
    
    async def subscribe(
        self,
        symbols: List[str],
        callback: Callable,
        event_types: Optional[List[str]] = None
    ):
        """Subscribe to WebSocket updates"""
        async with websockets.connect(f'wss://ws.finnhub.io?token={self.api_key}') as websocket:
            # Subscribe to symbols
            for symbol in symbols:
                await websocket.send(json.dumps({
                    'type': 'subscribe',
                    'symbol': symbol
                }))
            
            # Listen for updates
            async for message in websocket:
                data = json.loads(message)
                if data.get('type') == 'trade':
                    for trade in data.get('data', []):
                        event = MarketEvent(
                            event_type='trade',
                            symbol=trade['s'],
                            data=trade,
                            timestamp=datetime.fromtimestamp(trade['t'] / 1000),
                            source='finnhub'
                        )
                        callback(event)


class AlpacaProvider:
    """Alpaca data provider"""
    
    def __init__(self, api_key: str, api_secret: str, base_url: str):
        self.api = tradeapi.REST(api_key, api_secret, base_url)
        self.stream = tradeapi.Stream(api_key, api_secret, base_url)
    
    async def get_quotes(self, symbols: List[str]) -> List[MarketQuote]:
        """Get quotes from Alpaca"""
        quotes = []
        
        # Get latest quotes
        latest_quotes = self.api.get_latest_quotes(symbols)
        
        for symbol, quote in latest_quotes.items():
            quotes.append(MarketQuote(
                symbol=symbol,
                price=quote.ask_price,  # Use ask price as current
                bid=quote.bid_price,
                ask=quote.ask_price,
                volume=quote.bid_size + quote.ask_size,
                timestamp=quote.timestamp,
                source="alpaca"
            ))
        
        return quotes
    
    async def get_historical_data(
        self,
        symbol: str,
        period: str,
        interval: str
    ) -> HistoricalData:
        """Get historical data from Alpaca"""
        # Convert period to start/end dates
        end = datetime.now()
        if period == "1d":
            start = end - timedelta(days=1)
        elif period == "1w":
            start = end - timedelta(weeks=1)
        elif period == "1mo":
            start = end - timedelta(days=30)
        elif period == "1y":
            start = end - timedelta(days=365)
        else:
            start = end - timedelta(days=30)
        
        # Get bars
        bars = self.api.get_bars(
            symbol,
            interval,
            start=start.isoformat(),
            end=end.isoformat()
        ).df
        
        return HistoricalData(
            symbol=symbol,
            timeframe=interval,
            data=bars,
            source="alpaca"
        )
    
    async def subscribe(
        self,
        symbols: List[str],
        callback: Callable,
        event_types: Optional[List[str]] = None
    ):
        """Subscribe to real-time updates"""
        @self.stream.on_bar(*symbols)
        async def on_bar(bar):
            event = MarketEvent(
                event_type='bar',
                symbol=bar.symbol,
                data={
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume,
                    'timestamp': bar.timestamp
                },
                timestamp=bar.timestamp,
                source='alpaca'
            )
            callback(event)
        
        @self.stream.on_trade(*symbols)
        async def on_trade(trade):
            event = MarketEvent(
                event_type='trade',
                symbol=trade.symbol,
                data={
                    'price': trade.price,
                    'size': trade.size,
                    'timestamp': trade.timestamp
                },
                timestamp=trade.timestamp,
                source='alpaca'
            )
            callback(event)
        
        # Start streaming
        self.stream.run()


class PolygonProvider:
    """Polygon.io data provider"""
    
    def __init__(self, api_key: str):
        self.client = PolygonClient(api_key)
    
    async def get_quotes(self, symbols: List[str]) -> List[MarketQuote]:
        """Get quotes from Polygon"""
        quotes = []
        
        for symbol in symbols:
            try:
                # Get snapshot
                snapshot = self.client.get_snapshot(symbol)
                
                quote = MarketQuote(
                    symbol=symbol,
                    price=snapshot.last_trade['p'],
                    bid=snapshot.last_quote['P'],
                    ask=snapshot.last_quote['A'],
                    volume=snapshot.day['v'],
                    timestamp=datetime.fromtimestamp(snapshot.updated / 1000),
                    source="polygon"
                )
                quotes.append(quote)
            except Exception as e:
                print(f"Error fetching {symbol} from Polygon: {e}")
        
        return quotes
    
    async def get_historical_data(
        self,
        symbol: str,
        period: str,
        interval: str
    ) -> HistoricalData:
        """Get historical data from Polygon"""
        # Convert period and interval to Polygon format
        end = datetime.now()
        
        if period == "1d":
            start = end - timedelta(days=1)
        elif period == "1w":
            start = end - timedelta(weeks=1)
        elif period == "1mo":
            start = end - timedelta(days=30)
        elif period == "1y":
            start = end - timedelta(days=365)
        else:
            start = end - timedelta(days=30)
        
        # Map interval to timespan
        timespan_map = {
            "1m": "minute",
            "5m": "minute",
            "15m": "minute",
            "1h": "hour",
            "1d": "day"
        }
        
        timespan = timespan_map.get(interval, "day")
        
        # Get aggregates
        aggs = self.client.get_aggs(
            ticker=symbol,
            multiplier=1,
            timespan=timespan,
            from_=start.strftime("%Y-%m-%d"),
            to=end.strftime("%Y-%m-%d")
        )
        
        # Convert to DataFrame
        data = pd.DataFrame([{
            'timestamp': datetime.fromtimestamp(a.timestamp / 1000),
            'open': a.open,
            'high': a.high,
            'low': a.low,
            'close': a.close,
            'volume': a.volume
        } for a in aggs])
        
        data.set_index('timestamp', inplace=True)
        
        return HistoricalData(
            symbol=symbol,
            timeframe=interval,
            data=data,
            source="polygon"
        )


class BinanceProvider:
    """Binance data provider for cryptocurrency"""
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.client = None
        self.api_key = api_key
        self.api_secret = api_secret
    
    async def _get_client(self):
        """Get or create Binance client"""
        if not self.client:
            self.client = await BinanceClient.create(self.api_key, self.api_secret)
        return self.client
    
    async def get_quotes(self, symbols: List[str]) -> List[MarketQuote]:
        """Get quotes from Binance"""
        client = await self._get_client()
        quotes = []
        
        # Get tickers
        tickers = await client.get_all_tickers()
        ticker_map = {t['symbol']: t for t in tickers}
        
        for symbol in symbols:
            # Convert symbol format (BTC-USD -> BTCUSDT)
            binance_symbol = symbol.replace('-', '').replace('USD', 'USDT')
            
            if binance_symbol in ticker_map:
                ticker = ticker_map[binance_symbol]
                
                # Get order book for bid/ask
                order_book = await client.get_order_book(symbol=binance_symbol, limit=5)
                
                quote = MarketQuote(
                    symbol=symbol,
                    price=float(ticker['price']),
                    bid=float(order_book['bids'][0][0]) if order_book['bids'] else 0,
                    ask=float(order_book['asks'][0][0]) if order_book['asks'] else 0,
                    volume=float(ticker.get('volume', 0)),
                    timestamp=datetime.now(),
                    source="binance"
                )
                quotes.append(quote)
        
        return quotes
    
    async def get_historical_data(
        self,
        symbol: str,
        period: str,
        interval: str
    ) -> HistoricalData:
        """Get historical data from Binance"""
        client = await self._get_client()
        
        # Convert symbol format
        binance_symbol = symbol.replace('-', '').replace('USD', 'USDT')
        
        # Map interval
        interval_map = {
            "1m": client.KLINE_INTERVAL_1MINUTE,
            "5m": client.KLINE_INTERVAL_5MINUTE,
            "15m": client.KLINE_INTERVAL_15MINUTE,
            "1h": client.KLINE_INTERVAL_1HOUR,
            "1d": client.KLINE_INTERVAL_1DAY
        }
        
        binance_interval = interval_map.get(interval, client.KLINE_INTERVAL_1DAY)
        
        # Get klines
        klines = await client.get_klines(
            symbol=binance_symbol,
            interval=binance_interval,
            limit=1000
        )
        
        # Convert to DataFrame
        data = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert types
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            data[col] = data[col].astype(float)
        
        data.set_index('timestamp', inplace=True)
        
        return HistoricalData(
            symbol=symbol,
            timeframe=interval,
            data=data[['open', 'high', 'low', 'close', 'volume']],
            source="binance"
        )