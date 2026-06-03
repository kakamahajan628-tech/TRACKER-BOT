import os
import asyncio
import logging
import sys
import time
import html
import hashlib
import json
from collections import deque
import numpy as np
import pandas as pd
import ccxt.async_support as ccxt  
import ta
from scipy.signal import find_peaks  
import aiohttp
import aiosqlite
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes, CallbackQueryHandler, MessageHandler, filters

# Logging setup
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    stream=sys.stdout
)

TOKEN = os.getenv("TELEGRAM_TOKEN")
raw_chat_id = os.getenv("USER_CHAT_ID")
USER_CHAT_ID = int(raw_chat_id) if (raw_chat_id and raw_chat_id.strip().isdigit()) else None

if not TOKEN or not USER_CHAT_ID:
    logging.critical("ENVIRONMENT CONFIGURATION ERROR: System credentials mapping failed.")
    sys.exit(1)

DB_FILE = "quant_sniper.db"
CG_JSON_FILE = "coingecko_map.json"
EXCHANGES = []
EXCHANGE_MARKETS = {}
MARKETS_LOADED = False

# Rate-Limiting Hardware Isolation Pools
GATEIO_SEMAPHORE = asyncio.Semaphore(5)
MEXC_SEMAPHORE = asyncio.Semaphore(3)
COINGECKO_SEMAPHORE = asyncio.Semaphore(1) 
DEFAULT_SEMAPHORE = asyncio.Semaphore(2)
GLOBAL_SCAN_SEMAPHORE = asyncio.Semaphore(10)

VALID_SYMBOLS = set()
SYMBOL_TO_EXCHANGE = {}  
TIMEFRAMES = ['5m', '15m', '1h']  # Permanently Locked: Strictly 3 core intervals, 4h expunged forever
CACHE_TTL = 900  
CACHE_UNKNOWN_TTL = 86400  
MAX_PAIRS_PER_USER = 50  
MONITOR_TASK = None  
COINGECKO_LOCKS = {}  

def safe_float(v):
    try:
        if v is None: return 0.0
        return float(v)
    except:
        return 0.0

try:
    gateio = ccxt.gateio({'enableRateLimit': True})
    mexc = ccxt.mexc({'enableRateLimit': True, 'options': {'defaultType': 'spot'}})
    EXCHANGES.append(gateio)
    EXCHANGES.append(mexc)
    logging.info("Asynchronous exchanges registered safely inside operational tracking arrays.")
except Exception as e:
    logging.error(f"Exchange adapter initialization breakdown: {e}")

# ============================================================================
# SERVICE ORIENTED STATE BLUEPRINT CAPSULE
# ============================================================================

class BotState:
    def __init__(self):
        self.db = None  
        self.session = None  
        self.tracked_pairs = {}
        self.waiting_for_coin = {}
        self.alert_cooldown = {}
        self.report_cooldown = {}
        self.active_exchange_cache = {}
        self.coingecko_cache = {}
        self.ohlcv_cache = {}
        self.orderbook_cache = {}  
        self.coingecko_unknown_cache = {}
        
        self.exchange_failures = {}
        self.exchange_disabled_until = {}
        self.symbol_to_exchange_snapshot = {}
        self.pending_db_commit = False
        
        self.computed_signals_matrix = {}
        self.symbol_active_counts = {}
        self.symbol_locks = {}
        self.user_command_cooldowns = {}
        
        self.worker_queue = asyncio.Queue(maxsize=500)
        self.orderbook_snapshot_history = {}
        
        self.coingecko_id_map = {
            "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana",
            "XRP": "ripple", "ADA": "cardano", "DOT": "polkadot",
            "DOGE": "dogecoin", "AVAX": "avalanche-2", "LINK": "chainlink"
        }
        
        # Thread Synchronization Locks Matrix Ordering Priority Hierarchy Chain
        self.tracked_pairs_lock = asyncio.Lock()
        self.symbol_registry_lock = asyncio.Lock()  
        self.computed_signals_lock = asyncio.Lock()  
        self.alert_lock = asyncio.Lock()
        
        self.waiting_lock = asyncio.Lock()
        self.ohlcv_cache_lock = asyncio.Lock()
        self.orderbook_cache_lock = asyncio.Lock()
        self.report_cooldown_lock = asyncio.Lock()
        self.coingecko_cache_lock = asyncio.Lock()
        self.markets_validation_lock = asyncio.Lock()  
        self.db_queue_lock = asyncio.Lock()  

STATE = BotState()

def get_exchange_semaphore(exchange):
    if exchange.id == "gateio":
        return GATEIO_SEMAPHORE
    elif exchange.id == "mexc":
        return MEXC_SEMAPHORE
    return DEFAULT_SEMAPHORE

# ============================================================================
# CIRCUITS BREAKER LOGIC COUPLING
# ============================================================================

def exchange_available(exchange):
    disabled_until = STATE.exchange_disabled_until.get(exchange.id, 0.0)
    return time.time() > disabled_until

def mark_exchange_failure(exchange):
    count = STATE.exchange_failures.get(exchange.id, 0) + 1
    STATE.exchange_failures[exchange.id] = count
    logging.warning(f"Connection failure logged on {exchange.id}. Anomalies tally: {count}/5")
    if count >= 5:
        STATE.exchange_disabled_until[exchange.id] = time.time() + 600
        STATE.exchange_failures[exchange.id] = 0
        logging.critical(f"CIRCUIT BREAKER ENGAGED: Isolated for 10 minutes: {exchange.id}")

def mark_exchange_success(exchange):
    STATE.exchange_failures[exchange.id] = 0

# ============================================================================
# NON-BLOCKING ASYNC SQL STORAGE MATRIX (WAL Mode Setup)
# ============================================================================

async def init_db_async():
    STATE.db = await aiosqlite.connect(DB_FILE)
    await STATE.db.execute("""
        CREATE TABLE IF NOT EXISTS tracked_pairs (
            chat_id INTEGER,
            symbol TEXT,
            PRIMARY KEY (chat_id, symbol)
        )
    """)
    await STATE.db.execute("""
        CREATE TABLE IF NOT EXISTS signal_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            signal TEXT,
            price REAL,
            score INTEGER,
            timestamp INTEGER
        )
    """)
    await STATE.db.execute("PRAGMA journal_mode=WAL")
    await STATE.db.execute("PRAGMA synchronous=NORMAL")
    await STATE.db.commit()

async def db_load_tracked_pairs_async():
    await init_db_async()
    try:
        async with STATE.db.execute("SELECT chat_id, symbol FROM tracked_pairs") as cursor:
            rows = await cursor.fetchall()
                
        async with STATE.tracked_pairs_lock:
            for chat_id, symbol in rows:
                if chat_id not in STATE.tracked_pairs:
                    STATE.tracked_pairs[chat_id] = set()
                STATE.tracked_pairs[chat_id].add(symbol)
                STATE.symbol_active_counts[symbol] = STATE.symbol_active_counts.get(symbol, 0) + 1
                async with STATE.symbol_registry_lock:
                    if symbol not in STATE.symbol_locks:
                        STATE.symbol_locks[symbol] = asyncio.Lock()
        logging.info("aiosqlite database configuration synchronized seamlessly.")
    except Exception as e:
        logging.error(f"Failed to populate storage fields: {e}")

async def db_add_pair_async(chat_id, symbol):
    try:
        await STATE.db.execute("INSERT OR IGNORE INTO tracked_pairs (chat_id, symbol) VALUES (?, ?)", (chat_id, symbol))
        async with STATE.db_queue_lock:
            STATE.pending_db_commit = True
    except Exception as e:
        logging.error(f"Database insertion write dropout context: {e}")

async def db_remove_pair_async(chat_id, symbol):
    try:
        await STATE.db.execute("DELETE FROM tracked_pairs WHERE chat_id = ? AND symbol = ?", (chat_id, symbol))
        async with STATE.db_queue_lock:
            STATE.pending_db_commit = True
    except Exception as e:
        logging.error(f"Database erasure write dropout context: {e}")

async def db_log_signal_history_async(symbol, signal, price, score):
    try:
        await STATE.db.execute(
            "INSERT INTO signal_history (symbol, signal, price, score, timestamp) VALUES (?, ?, ?, ?, ?)",
            (symbol, signal, price, int(score), int(time.time()))
        )
        async with STATE.db_queue_lock:
            STATE.pending_db_commit = True
    except Exception as e:
        logging.error(f"Failed to append quantitative history records: {e}")

async def flush_database_commits_batch():
    async with STATE.db_queue_lock:
        if not STATE.pending_db_commit: return
        has_elements = True
    if has_elements:
        try:
            await STATE.db.commit()
            async with STATE.db_queue_lock:
                STATE.pending_db_commit = False
        except Exception as e:
            logging.error(f"Storage array compilation commit flush failure: {e}")

# ============================================================================
# ACCELERATED ADAPTER ASYNC PIPELINES LAYER
# ============================================================================

async def load_exchange_markets():
    global MARKETS_LOADED, VALID_SYMBOLS, EXCHANGE_MARKETS, SYMBOL_TO_EXCHANGE
    new_symbols = set()
    new_markets = {}
    new_routing_map = {}
    best_volume = {}
    
    for exchange in EXCHANGES:
        if not exchange_available(exchange): continue
        try:
            logging.info(f"{exchange.name} market parameters downloading async...")
            markets = await exchange.load_markets()
            new_markets[exchange.id] = markets
            
            for symbol, market_obj in markets.items():
                clean_sym = symbol.upper()
                new_symbols.add(clean_sym)
                
                raw_vol = market_obj.get("quoteVolume") or market_obj.get("baseVolume") or market_obj.get("info", {}).get("quoteVolume") or market_obj.get("info", {}).get("volume24h") or 0.0
                volume = safe_float(raw_vol)
                
                if clean_sym not in best_volume or volume > best_volume[clean_sym]:
                    best_volume[clean_sym] = volume
                    new_routing_map[clean_sym] = exchange
                    
            mark_exchange_success(exchange)
        except Exception as e:
            logging.error(f"Markets tracking layout extraction failed on {exchange.name}: {e}")
            mark_exchange_failure(exchange)
            continue  
            
    if not new_symbols: return
    async with STATE.markets_validation_lock:
        VALID_SYMBOLS = new_symbols
        EXCHANGE_MARKETS = new_markets
        SYMBOL_TO_EXCHANGE = new_routing_map
        MARKETS_LOADED = True
    logging.info("Atomic market matrices swapped successfully into core tracking parameters.")

async def validate_market_symbol(symbol):
    async with STATE.markets_validation_lock:
        return symbol.upper() in VALID_SYMBOLS

async def fetch_ohlcv_permitted(symbol, timeframe, exchange_target, limit=500):
    if not exchange_available(exchange_target): return None
    market_symbol = symbol.upper()
    cache_key = f"{exchange_target.id}:{market_symbol}:{timeframe}"
    current_time = time.time()
    tf_ttl = 60 if timeframe == '5m' else (300 if timeframe == '15m' else 900)
    
    async with STATE.ohlcv_cache_lock:
        if cache_key in STATE.ohlcv_cache:
            cached_data, timestamp = STATE.ohlcv_cache[cache_key]
            if current_time - timestamp < tf_ttl:
                return cached_data

    async with get_exchange_semaphore(exchange_target):
        for attempt in range(4):
            try:
                ohlcv = await asyncio.wait_for(
                    exchange_target.fetch_ohlcv(market_symbol, timeframe, limit=limit),
                    timeout=4.0
                )
                if ohlcv and len(ohlcv) >= 100:
                    async with STATE.ohlcv_cache_lock:
                        if len(STATE.ohlcv_cache) > 500: 
                            STATE.ohlcv_cache.pop(next(iter(STATE.ohlcv_cache)), None)
                        STATE.ohlcv_cache[cache_key] = (ohlcv, current_time)
                    mark_exchange_success(exchange_target)
                    return ohlcv
                break
            except Exception as e:
                if attempt < 3:
                    await asyncio.sleep(2 ** attempt)
                    continue
                logging.debug(f"OHLCV async network drop context anchor on symbol {symbol}: {e}")
                mark_exchange_failure(exchange_target)
        return None

async def fetch_orderbook_async_safe(exchange, symbol):
    if not exchange_available(exchange): return None, "DEAD_EXCHANGE"
    market_symbol = symbol.upper()
    cache_key = f"{exchange.id}:{market_symbol}"
    current_time = time.time()
    
    async with STATE.orderbook_cache_lock:
        if cache_key in STATE.orderbook_cache:
            cached_data, timestamp = STATE.orderbook_cache[cache_key]
            if current_time - timestamp < 30:
                return cached_data

    async with get_exchange_semaphore(exchange):
        for attempt in range(4):
            try:
                res = await fetch_orderbook_advanced_metrics_async(exchange, symbol)
                if res and res[0] is not None:
                    async with STATE.orderbook_cache_lock:
                        if len(STATE.orderbook_cache) > 200:
                            STATE.orderbook_cache.pop(next(iter(STATE.orderbook_cache)), None)
                        STATE.orderbook_cache[cache_key] = (res, current_time)
                    mark_exchange_success(exchange)
                    return res
                elif res and res[1] == "WIDE_SPREAD":
                    return None, "WIDE_SPREAD"
                break
            except Exception as e:
                if attempt < 3:
                    await asyncio.sleep(2 ** attempt)
                    continue
                logging.error(f"Advanced async order book tracking collapse on {symbol}: {e}")
                mark_exchange_failure(exchange)
        return None, "NO_BOOK"

async def fetch_orderbook_advanced_metrics_async(exchange, symbol):
    market_symbol = symbol.upper()
    orderbook = await exchange.fetch_order_book(market_symbol, limit=20)
    bids, asks = orderbook['bids'], orderbook['asks']
    
    if not bids or not asks: return None, "NO_BOOK"
    best_bid, best_ask = bids[0][0], asks[0][0]
    mid_price = (best_bid + best_ask) / 2.0
    spread = max(best_ask - best_bid, mid_price * 0.0001)
    
    if mid_price <= 0: return None, "NO_BOOK"
    if (spread / mid_price) * 100 > 0.3: return None, "WIDE_SPREAD"
        
    top_bid_val = sum(p * q for p, q in bids[:5])
    top_ask_val = sum(p * q for p, q in asks[:5])
    
    orderbook_signal = "NORMAL"
    if top_bid_val > (top_ask_val * 2.5): orderbook_signal = "STRONG_BUY_PRESSURE"
    elif top_ask_val > (top_bid_val * 2.5): orderbook_signal = "STRONG_SELL_PRESSURE"
    
    if symbol not in STATE.orderbook_snapshot_history:
        STATE.orderbook_snapshot_history[symbol] = deque(maxlen=3)
    STATE.orderbook_snapshot_history[symbol].append(orderbook_signal)
    
    history_array = list(STATE.orderbook_snapshot_history[symbol])
    if len(history_array) == 3 and history_array[0] == history_array[1] == history_array[2]:
        persistent_signal = history_array[-1]
    else:
        persistent_signal = "NORMAL"
        
    weighted_bids_volume = 0.0
    for price, qty in bids[:10]:
        weighted_bids_volume += qty * np.exp(-abs(price - mid_price) / (spread * 5))
        
    weighted_asks_volume = 0.0
    for price, qty in asks[:10]:
        weighted_asks_volume += qty * np.exp(-abs(price - mid_price) / (spread * 5))
        
    total_weighted_volume = weighted_bids_volume + weighted_asks_volume
    if total_weighted_volume <= 0: return 50.0, persistent_signal
        
    bid_ratio = (weighted_bids_volume / total_weighted_volume) * 100
    return bid_ratio, persistent_signal

async def prefetch_coingecko_id_matrix_map():
    if os.path.exists(CG_JSON_FILE):
        try:
            with open(CG_JSON_FILE, "r") as f:
                local_data = json.load(f)
            async with STATE.coingecko_cache_lock:
                STATE.coingecko_id_map.update(local_data)
            logging.info(f"CoinGecko id maps preloaded flawlessly out of local JSON disk fields. Count: {len(STATE.coingecko_id_map)}")
        except Exception as read_err:
            logging.error(f"Failed to read local CoinGecko storage mappings: {read_err}")

async def scrape_public_onchain_intel(symbol):
    clean_ticker = symbol.split('/')[0].upper()
    current_time = time.time()
    
    async with STATE.coingecko_cache_lock:
        if clean_ticker in STATE.coingecko_unknown_cache:
            if current_time - STATE.coingecko_unknown_cache[clean_ticker] < CACHE_UNKNOWN_TTL:
                return "UNKNOWN"
        if clean_ticker in STATE.coingecko_cache:
            cache_data, timestamp = STATE.coingecko_cache[clean_ticker]
            if current_time - timestamp < CACHE_TTL:
                return cache_data

        if clean_ticker not in COINGECKO_LOCKS:
            COINGECKO_LOCKS[clean_ticker] = {"lock": asyncio.Lock(), "timestamp": current_time}
        COINGECKO_LOCKS[clean_ticker]["timestamp"] = current_time
        target_lock = COINGECKO_LOCKS[clean_ticker]["lock"]
        
    async with target_lock:
        async with STATE.coingecko_cache_lock:
            if clean_ticker in STATE.coingecko_cache:
                cache_data, ts = STATE.coingecko_cache[clean_ticker]
                if current_time - ts < CACHE_TTL: return cache_data

        try:
            async with STATE.coingecko_cache_lock:
                coin_id = STATE.coingecko_id_map.get(clean_ticker)
            
            if not coin_id:
                search_url = f"https://api.coingecko.com/api/v3/search?query={clean_ticker}"
                async with COINGECKO_SEMAPHORE:
                    async with STATE.session.get(search_url, timeout=5) as response:
                        res = await response.json()
                    
                if 'coins' in res and len(res['coins']) > 0:
                    highest_rank = float('inf')
                    for coin_node in res['coins']:
                        node_symbol = coin_node.get('symbol', '').upper()
                        if node_symbol == clean_ticker:
                            rank = coin_node.get('market_cap_rank')
                            if rank and rank < highest_rank:
                                highest_rank = rank
                                coin_id = coin_node['id']
                    
                    if coin_id:
                        async with STATE.coingecko_cache_lock: STATE.coingecko_id_map[clean_ticker] = coin_id
            
            if not coin_id:
                async with STATE.coingecko_cache_lock: STATE.coingecko_unknown_cache[clean_ticker] = current_time
                return "UNKNOWN"
                
            detail_url = f"https://api.coingecko.com/api/v3/coins/{coin_id}?localization=false&tickers=false&market_data=true&community_data=false&developer_data=false&sparkline=false"
            async with COINGECKO_SEMAPHORE:
                async with STATE.session.get(detail_url, timeout=5) as detail_response:
                    coin_data = await detail_response.json()
                
            mcap = coin_data.get('market_data', {}).get('market_cap', {}).get('usd', 0)
            vol_24h = coin_data.get('market_data', {}).get('total_volume', {}).get('usd', 0)
            price_change = coin_data.get('market_data', {}).get('price_change_percentage_24h', 0)
            
            if vol_24h > 0 and mcap > 0:
                v2m_ratio = vol_24h / mcap
                status = "HOLDERS_OK"
                if price_change < -10 and v2m_ratio > 0.35: status = "WHALE_BUY"
                elif v2m_ratio > 0.40: status = "WHALE_ACCUM"
                elif v2m_ratio > 0.20: status = "VOLUME_FLOW"
                
                async with STATE.coingecko_cache_lock: STATE.coingecko_cache[clean_ticker] = (status, current_time)
                return status
        except Exception as e:
            logging.error(f"CoinGecko API cluster failure on {clean_ticker}: {e}")
            
        return "HOLDERS_OK"

# ============================================================================
# BI-REGIME DUAL-DIRECTIONAL SIGNAL ENGINE MATRIX (9.6+/10 Tier Blueprint)
# ============================================================================

async def evaluate_bitcoin_macro_regime_lock() -> bool:
    try:
        btc_exchange_node = STATE.symbol_to_exchange_snapshot.get("BTC/USDT")
        if not btc_exchange_node: return False
        
        ohlcv_btc = await fetch_ohlcv_permitted("BTC/USDT", '1h', btc_exchange_node, limit=250)
        if not ohlcv_btc: return False
        
        df_btc = pd.DataFrame(ohlcv_btc, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']).dropna().reset_index(drop=True)
        if len(df_btc) < 50: return False
        
        btc_prices = df_btc['close'].to_numpy()
        btc_ema50 = ta.trend.ema_indicator(close=df_btc['close'], window=50).to_numpy()
        btc_ema200 = ta.trend.ema_indicator(close=df_btc['close'], window=200).to_numpy()
        btc_macd_diff = ta.trend.MACD(close=df_btc['close']).macd_diff().to_numpy()
        
        if len(btc_ema50) == 0 or len(btc_ema200) == 0 or len(btc_macd_diff) == 0: return False
        
        if btc_prices[-1] < btc_ema50[-1] or btc_ema50[-1] < btc_ema200[-1] or btc_macd_diff[-1] < 0:
            return False
        return True
    except Exception as btc_err:
        logging.error(f"Failed to extract Bitcoin benchmark regime filters: {btc_err}")
        return False

def evaluate_quant_signal_scoring(df, bid_pct, order_flow_status, symbol):
    df = df.dropna().reset_index(drop=True)
    prices = df['close'].to_numpy()
    highs = df['high'].to_numpy()
    lows = df['low'].to_numpy()
    
    if len(prices) < 30: return "SCAN", 0, "Clear", 0.0
    
    vol_mas = df['volume'].rolling(window=10).mean().to_numpy()
    rsis = ta.momentum.rsi(close=df['close'], window=14).to_numpy()
    
    macd = ta.trend.MACD(close=df['close'])
    macd_lines = macd.macd().to_numpy()
    macd_hists = macd.macd_diff().to_numpy()
    
    bb = ta.volatility.BollingerBands(close=df['close'], window=20)
    bb_highs = bb.bollinger_hband().to_numpy()
    bb_lows = bb.bollinger_lband().to_numpy()
    adx_vals = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close']).adx().to_numpy()
    atr_series = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range().to_numpy()
    vwap_series = ta.volume.VolumeWeightedAveragePrice(high=df['high'], low=df['low'], close=df['close'], volume=df['volume']).volume_weighted_average_price().to_numpy()
    
    ema50 = ta.trend.ema_indicator(close=df['close'], window=50).to_numpy()
    ema200 = ta.trend.ema_indicator(close=df['close'], window=200).to_numpy()
    
    if pd.isna(vol_mas[-1]) or pd.isna(rsis[-1]) or pd.isna(macd_hists[-1]) or pd.isna(adx_vals[-1]) or pd.isna(atr_series[-1]) or pd.isna(vwap_series[-1]):
        return "SCAN", 0, "Clear", 0.0
        
    last_price, last_rsi, last_adx, last_bb_high, last_bb_low = prices[-1], rsis[-1], adx_vals[-1], bb_highs[-1], bb_lows[-1]
    last_vol, last_vol_ma = df['volume'].iloc[-1], vol_mas[-1]
    last_hist, last_atr, last_vwap = macd_hists[-1], atr_series[-1], vwap_series[-1]
    
    if last_price <= 0: return "SCAN", 0, "Clear", 0.0
    
    atr_pct = (last_atr / last_price) * 100
    clamped_atr_pct = max(0.3, min(5.0, atr_pct))
    
    has_ema50 = len(ema50) > 0 and pd.notna(ema50[-1])
    has_ema200 = len(ema200) > 0 and pd.notna(ema200[-1])
    
    is_macro_bullish = (ema50[-1] > ema200[-1]) and (last_price > ema50[-1]) and (last_price > ema200[-1]) if (has_ema50 and has_ema200) else False
    is_macro_bearish = has_ema50 and has_ema200 and (ema50[-1] < ema200[-1])
    
    ema_slope_down = has_ema50 and (ema50[-1] < ema50[-2] < ema50[-3])
    price_below_ema = has_ema50 and (last_price < ema50[-1])
    macd_negative = last_hist < 0
    is_crashing_hard = (ema_slope_down and price_below_ema and macd_negative)
    
    is_pumping_hard = has_ema50 and (ema50[-1] > ema50[-2] > ema50[-3]) and (last_price > ema50[-1]) and (last_hist > 0)

    if last_adx > 30:
        market_regime = "TRENDING_UP" if is_macro_bullish else ("TRENDING_DOWN" if is_macro_bearish else "TRENDING")
    else:
        market_regime = "RANGING"
        
    bull_structure_confirmed = False
    if len(lows) >= 6 and len(highs) >= 6:
        higher_low = lows[-1] > lows[-5]
        higher_high = highs[-1] > highs[-5]
        bull_structure_confirmed = (higher_low and higher_high)
        
    volume_ratio = last_vol / last_vol_ma if last_vol_ma > 0 else 1.0
    volume_expansion_valid = volume_ratio >= 1.5
    
    liquidity_sweep_confirmed = False
    if len(lows) >= 12:
        local_structural_floor = min(lows[-10:-2])
        liquidity_sweep_confirmed = (lows[-1] < local_structural_floor) and (prices[-1] > df['close'].iloc[-2])

    score = 0
    if market_regime == "TRENDING_UP" or is_pumping_hard:
        score += 25  
        if last_price > last_vwap: score += 15
        if last_hist > 0: score += 15
        if volume_expansion_valid: score += 15
        if order_flow_status == "STRONG_BUY_PRESSURE": score += 5  
        if len(macd_hists) >= 5 and (macd_hists[-1] > macd_hists[-2] > macd_hists[-3] > macd_hists[-4] > macd_hists[-5]): score += 15
    elif market_regime == "TRENDING_DOWN" or is_crashing_hard:
        score -= 25
        if last_price < last_vwap: score -= 15
        if last_hist < 0: score -= 15
        if last_vol > last_vol_ma: score -= 15
        if order_flow_status == "STRONG_SELL_PRESSURE": score -= 5  
        if len(macd_hists) >= 5 and (macd_hists[-1] < macd_hists[-2] < macd_hists[-3] < macd_hists[-4] < macd_hists[-5]): score -= 15
            
    else: 
        bull_structure = has_ema50 and (last_price > ema50[-1])
        if bull_structure:
            if last_rsi <= 25: score += 20
            if last_rsi <= 30: score += 10
            if last_price < last_bb_low: score += 25
            if order_flow_status == "STRONG_BUY_PRESSURE": score += 5
            if liquidity_sweep_confirmed: score += 20  
        
        if not bull_structure:
            if last_rsi >= 75: score -= 20
            if last_rsi >= 80: score -= 10
            if last_price > last_bb_high: score -= 25
            if order_flow_status == "STRONG_SELL_PRESSURE": score -= 5

    adaptive_prominence = max(0.05, last_atr * 0.05)
    macd_vector = np.asarray(macd_lines, dtype=np.float64)
    peaks, _ = find_peaks(macd_vector, prominence=adaptive_prominence)
    troughs, _ = find_peaks(-macd_vector, prominence=adaptive_prominence)
    
    anomaly = "Clear"
    recent_window_floor = max(0, len(prices) - 100)
    
    if len(troughs) >= 2 and len(peaks) >= 2:
        low1_idx, low2_idx = troughs[-2], troughs[-1]
        high1_idx, high2_idx = peaks[-2], peaks[-1]
        
        if low2_idx >= recent_window_floor and low1_idx >= recent_window_floor:
            if low2_idx < len(prices) and low1_idx < len(prices) and low2_idx < len(macd_lines) and low1_idx < len(macd_lines):
                if prices[low2_idx] < prices[low1_idx] and macd_lines[low2_idx] > macd_lines[low1_idx] and abs(prices[low2_idx] - prices[low1_idx]) > (last_atr * 0.5): 
                    if not is_crashing_hard and last_price > (ema50[-1] if has_ema50 else last_price):
                        anomaly = "BUY_DIV"
                        score += 20
        if high2_idx >= recent_window_floor and high1_idx >= recent_window_floor:
            if high2_idx < len(prices) and high1_idx < len(prices) and high2_idx < len(macd_lines) and high1_idx < len(macd_lines):
                if prices[high2_idx] > prices[high1_idx] and macd_lines[high2_idx] < macd_lines[high1_idx] and abs(prices[high2_idx] - prices[high1_idx]) > (last_atr * 0.5): 
                    if not is_pumping_hard:
                        anomaly = "SELL_DIV"
                        score -= 20

    if has_ema200:
        if last_price < ema200[-1] and score > 0:
            score = min(score, 20)  

    dynamic_threshold = max(50.0, min(80.0, 50.0 + (last_adx / 2.0)))

    future_pred = "SCAN"
    if score >= dynamic_threshold and bull_structure_confirmed and volume_expansion_valid:
        if len(prices) >= 2 and prices[-1] > highs[-2]:  
            future_pred = "LONG_THOKO"
    elif score <= -dynamic_threshold and not bull_structure_confirmed:
        if len(prices) >= 2 and prices[-1] < lows[-2]:  
            future_pred = "SHORT_THOKO"
            
    if future_pred == "LONG_THOKO":
        if is_crashing_hard or not volume_expansion_valid or last_price < (ema200[-1] if has_ema200 else last_price):
            future_pred = "SCAN"
    elif future_pred == "SHORT_THOKO":
        if is_pumping_hard or last_price > (ema200[-1] if has_ema200 else last_price):
            future_pred = "SCAN"
    
    return future_pred, score, anomaly, last_atr

def analyze_predictive_metrics(ohlcv_converted, bid_pct, order_flow_status, symbol):
    df = pd.DataFrame(ohlcv_converted, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    bb = ta.volatility.BollingerBands(close=df['close'], window=20)
    ema50 = ta.trend.ema_indicator(close=df['close'], window=50).to_numpy()
    
    last_price = df['close'].iloc[-1]
    has_ema = len(ema50) > 0 and pd.notna(ema50[-1])
    
    structure_trend = "CH_UP" if (has_ema and last_price > ema50[-1]) else "CH_DN"
    
    mid_band = bb.bollinger_mavg().replace(0, np.nan).to_numpy()
    bb_high = bb.bollinger_hband().to_numpy()
    bb_low = bb.bollinger_lband().to_numpy()
    bbw_val = ((bb_high[-1] - bb_low[-1]) / mid_band[-1]) if (len(mid_band) > 0 and pd.notna(mid_band[-1])) else 0.05
    squeeze = "MOVE_IN" if (bbw_val <= 0.02) else "SHANT"
    
    prediction, score, anomaly, last_atr = evaluate_quant_signal_scoring(df, bid_pct, order_flow_status, symbol)
    return structure_trend, squeeze, anomaly, prediction, last_price, score, last_atr

async def process_single_timeframe_isolated(symbol, tf, exchange_obj, bid_pct, order_flow_status, df_base_5m):
    try:
        if tf == "5m":
            df = df_base_5m.copy()
        else:
            resample_rule = '15min' if tf == '15m' else '60min'
            df = df_base_5m.resample(resample_rule, on='timestamp').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
            }).dropna().reset_index()
            
        if len(df) < 30: return tf, None
        ohlcv_converted = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].values.tolist()
        
        structure_trend, squeeze, anomaly, prediction, price, score, last_atr = analyze_predictive_metrics(ohlcv_converted, bid_pct, order_flow_status, symbol)
        return tf, (squeeze, order_flow_status, prediction, structure_trend, price, score, exchange_obj.name, anomaly, last_atr)
    except Exception as e:
        logging.error(f"Timeframe isolation processing fault on {symbol} {tf}: {e}")
    return tf, None

async def safe_tf_runner(symbol, tf, exchange_obj, bid_pct, order_flow_status, df_base_5m):
    try:
        return await asyncio.wait_for(
            process_single_timeframe_isolated(symbol, tf, exchange_obj, bid_pct, order_flow_status, df_base_5m),
            timeout=15.0
        )
    except Exception as e:
        logging.error(f"Execution node context timed out for timeframe processing on {symbol} {tf}: {e}")
        return tf, None

async def queue_worker_consumer_engine():
    while True:
        symbol, loop_start_time = await STATE.worker_queue.get()
        try:
            res = await analyze_target_asset_data_stream_isolated_worker(symbol, loop_start_time)
            if res:
                async with STATE.computed_signals_lock:
                    STATE.computed_signals_matrix[symbol] = res
        except Exception as queue_err:
            logging.error(f"Distributed queue worker engine error logging asset {symbol}: {queue_err}")
        finally:
            STATE.worker_queue.task_done()

async def analyze_target_asset_data_stream_isolated_worker(symbol, loop_start_time):
    target_exchange_node = STATE.symbol_to_exchange_snapshot.get(symbol)
    if not target_exchange_node: return None
    
    async with STATE.markets_validation_lock:
        market_metrics = EXCHANGE_MARKETS.get(target_exchange_node.id, {}).get(symbol, {})
        
    quote_volume_raw = (
        market_metrics.get("quoteVolume")
        or market_metrics.get("baseVolume")
        or market_metrics.get("info", {}).get("quoteVolume")
        or market_metrics.get("info", {}).get("volume24h")
        or market_metrics.get("info", {}).get("vol", 0)
    )
    try:
        if float(quote_volume_raw) < 500000: return None
    except Exception: pass
        
    ohlcv_5m = await fetch_ohlcv_permitted(symbol, '5m', target_exchange_node, limit=300)
    if not ohlcv_5m: return None
    
    bid_pct, order_flow_status = await fetch_orderbook_async_safe(target_exchange_node, symbol)
    if bid_pct is None or order_flow_status == "NO_BOOK" or order_flow_status == "WIDE_SPREAD": return None

    df_base_5m = pd.DataFrame(ohlcv_5m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df_base_5m['timestamp'] = pd.to_datetime(df_base_5m['timestamp'], unit='ms', utc=True)

    tasks = [safe_tf_runner(symbol, tf, target_exchange_node, bid_pct, order_flow_status, df_base_5m) for tf in TIMEFRAMES]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    timeframe_data, tf_scores, trigger_atr_5m = {}, [], 0.0
    last_price, node_source, last_prediction, master_anomaly, has_data = 0.0, "Unknown", "SCAN", "Clear", False
    
    for item in results:
        if isinstance(item, Exception) or item is None or item[1] is None: continue
        tf, res = item
        squeeze, order_flow_status, prediction, structure_trend, price, score, active_exchange_name, anomaly, atr = res
        if tf == "5m":
            last_price, node_source, last_prediction, master_anomaly, trigger_atr_5m = price, active_exchange_name, prediction, anomaly, atr
        tf_scores.append(score)
        timeframe_data[tf] = (squeeze, order_flow_status, prediction, structure_trend, anomaly)
        has_data = True

    if has_data:
        return {
            "timeframe_data": timeframe_data, "tf_scores": tf_scores,
            "onchain_intel": await scrape_public_onchain_intel(symbol), "last_price": last_price,
            "node_source": node_source, "last_prediction": last_prediction,
            "trigger_atr_5m": trigger_atr_5m, "order_flow_status": order_flow_status
        }
    return None

async def process_single_symbol_concurrency_block(symbol, chat_id, loop_start_time, application):
    async with STATE.symbol_registry_lock:
        target_lock = STATE.symbol_locks.get(symbol)
        
    if not target_lock: return 
    
    async with target_lock:
        async with STATE.computed_signals_lock:
            symbol_computed_matrix = STATE.computed_signals_matrix.get(symbol)
        if symbol_computed_matrix:
            await execute_alert_dispatch_layer(chat_id, symbol, symbol_computed_matrix, loop_start_time, application)

# ============================================================================
# INITIALIZATION HOOKS PLATFORM LIFECYCLE
# ============================================================================

async def startup_sequence(application: Application):
    global MONITOR_TASK
    STATE.session = aiohttp.ClientSession()
    
    await db_load_tracked_pairs_async()
    await prefetch_coingecko_id_matrix_map() 
    await load_exchange_markets()
    
    for _ in range(10):
        asyncio.create_task(queue_worker_consumer_engine())
        
    if USER_CHAT_ID:
        try:
            await application.bot.send_message(
                chat_id=USER_CHAT_ID,
                text="🚀 <b>QUANT TERMINAL v50.0 MASTER TIERS ONLINE</b>\nMulti Timeframe constraints aligned perfectly. 4H blocks completely removed. Use /panel.",
                parse_mode="HTML"
            )
            logging.info("Startup sequence telegram validation alert broadcast successfully.")
        except Exception as e: 
            logging.error(f"Graceful initialization fallback trace block failure: {e}")
            
    MONITOR_TASK = asyncio.create_task(monitoring_job(application))

async def shutdown_sequence(application: Application):
    logging.info("Shutting down active processes safely. Dropping network sockets pools...")
    global MONITOR_TASK
    if MONITOR_TASK and not MONITOR_TASK.done():
        MONITOR_TASK.cancel()
        try: await MONITOR_TASK
        except asyncio.CancelledError: pass
            
    if STATE.db: await STATE.db.close()
    if STATE.session: await STATE.session.close()
            
    for exchange in EXCHANGES:
        try: await exchange.close()
        except Exception as socket_err: logging.error(f"Socket decoupling tracking collapse for {exchange.name}: {socket_err}")

# ============================================================================
# TELEGRAM APP USER PLATFORM INTERACTION LAYER
# ============================================================================

def build_control_panel(chat_id):
    pairs = STATE.tracked_pairs.get(chat_id, set())
    keyboard = []
    for symbol in sorted(list(pairs)):
        escaped_symbol = html.escape(str(symbol))
        symbol_hash = hashlib.md5(symbol.encode()).hexdigest()[:8]
        keyboard.append([
            InlineKeyboardButton(f"📊 {escaped_symbol}", callback_data=f"view_{symbol_hash}"),
            InlineKeyboardButton("🛑 STOP", callback_data=f"stop_{symbol_hash}")
        ])
    keyboard.append([InlineKeyboardButton("➕ Naya Coin Add Karo", callback_data="add_coin_click")])
    return InlineKeyboardMarkup(keyboard)

async def check_command_rate_limiting(chat_id):
    current_time = time.time()
    last_hit = STATE.user_command_cooldowns.get(chat_id, 0.0)
    if current_time - last_hit < 2.0: 
        return False
    STATE.user_command_cooldowns[chat_id] = current_time
    return True

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id != USER_CHAT_ID: return
    if not await check_command_rate_limiting(update.effective_chat.id): return
    await update.message.reply_text("⚡ <b>WELCOME TO QUANT SNIPER TERMINAL</b>\n\n👉 /panel - Control Panel Menu Kholein", parse_mode="HTML")

async def show_panel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id != USER_CHAT_ID: return
    if not await check_command_rate_limiting(update.effective_chat.id): return
    chat_id = update.effective_chat.id
    if not MARKETS_LOADED:
        await update.message.reply_text("⏳ <b>Markets abhi load ho rahe hain.</b> Thodi der baad try karo, Bhai!", parse_mode="HTML")
        return
        
    async with STATE.tracked_pairs_lock:
        if chat_id not in STATE.tracked_pairs: STATE.tracked_pairs[chat_id] = set()
        
    await update.message.reply_text("🎛️ <b>QUANT SNIPER CONTROL PANEL</b>\n\nSelect an option below:", reply_markup=build_control_panel(chat_id), parse_mode="HTML")

async def handle_button_clicks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id != USER_CHAT_ID: return
    query = update.callback_query
    chat_id = query.message.chat_id
    data = query.data
    await query.answer()

    async with STATE.tracked_pairs_lock:
        if chat_id not in STATE.tracked_pairs: STATE.tracked_pairs[chat_id] = set()

    if data == "add_coin_click":
        async with STATE.tracked_pairs_lock: pairs_count = len(STATE.tracked_pairs.get(chat_id, set()))
        if pairs_count >= MAX_PAIRS_PER_USER:
            await query.message.reply_text("❌ <b>Pair tracking cap ceiling reached!</b> Cannot trace past 50 active instruments.", parse_mode="HTML")
            return
            
        async with STATE.waiting_lock: STATE.waiting_for_coin[chat_id] = time.time()
        await query.message.reply_text("📝 Pair ka naam send karo (Ex: <code>SOL/USDT</code>):", parse_mode="HTML")
    elif data.startswith("stop_") or data.startswith("view_"):
        is_stop = data.startswith("stop_")
        hash_fragment = data.replace("stop_", "") if is_stop else data.replace("view_", "")
        target_symbol = None
        
        async with STATE.tracked_pairs_lock:
            for s in STATE.tracked_pairs.get(chat_id, set()):
                if hashlib.md5(s.encode()).hexdigest()[:8] == hash_fragment:
                    target_symbol = s
                    break
                    
            if is_stop and target_symbol:
                STATE.tracked_pairs[chat_id].remove(target_symbol)
                await db_remove_pair_async(chat_id, target_symbol)
                STATE.symbol_active_counts[target_symbol] = max(0, STATE.symbol_active_counts.get(target_symbol, 1) - 1)
                
        if target_symbol and is_stop:
            alert_key = f"{chat_id}:{target_symbol}"
            report_routing_key = f"{chat_id}_{target_symbol}"
            async with STATE.alert_lock: STATE.alert_cooldown.pop(alert_key, None)
            async with STATE.report_cooldown_lock: STATE.report_cooldown.pop(report_routing_key, None)
            await query.message.reply_text(f"🛑 <b>{html.escape(str(target_symbol))}</b> list se hat gaya aur database se saaf ho gaya.", parse_mode="HTML")
        await query.edit_message_reply_markup(reply_markup=build_control_panel(chat_id))

async def handle_text_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id != USER_CHAT_ID: return
    chat_id = update.effective_chat.id
    text_received = update.message.text.strip().upper()

    async with STATE.waiting_lock:
        has_waiting_context = chat_id in STATE.waiting_for_coin
        waiting_timestamp = STATE.waiting_for_coin.get(chat_id, 0)

    if has_waiting_context:
        if time.time() - waiting_timestamp > 300:
            async with STATE.waiting_lock: STATE.waiting_for_coin.pop(chat_id, None)
            return

        if "/" not in text_received:
            await update.message.reply_text("❌ Format galat hai! Use: <code>SOL/USDT</code>", parse_mode="HTML")
            return
            
        if not await validate_market_symbol(text_received):
            await update.message.reply_text(f"❌ <b>{html.escape(str(text_received))}</b> exchange list mein nahi mila! Sahi format use karo.", parse_mode="HTML")
            return

        async with STATE.tracked_pairs_lock:
            if chat_id not in STATE.tracked_pairs: STATE.tracked_pairs[chat_id] = set()
            STATE.tracked_pairs[chat_id].add(text_received)
            await db_add_pair_async(chat_id, text_received)
            
            STATE.symbol_active_counts[text_received] = STATE.symbol_active_counts.get(text_received, 0) + 1
            async with STATE.symbol_registry_lock:
                if text_received not in STATE.symbol_locks: STATE.symbol_locks[text_received] = asyncio.Lock()
            
        async with STATE.waiting_lock: STATE.waiting_for_coin.pop(chat_id, None)
        await update.message.reply_text(f"✅ <b>{html.escape(str(text_received))}</b> add ho gaya aur permanent database mein save ho gaya!", reply_markup=build_control_panel(chat_id), parse_mode="HTML")

# ============================================================================
# BROADCAST & SUBSCRIBER ROUTING CONNECTIONS LAYER
# ============================================================================

async def execute_alert_dispatch_layer(chat_id, symbol, data_matrix, loop_start_time, application):
    timeframes_list = list(data_matrix["timeframe_data"].keys())
    
    long_weighted_votes = 0
    short_weighted_votes = 0
    
    for tf in timeframes_list:
        pred = data_matrix["timeframe_data"][tf][2]
        weight = 1 if tf == "5m" else (2 if tf == "15m" else 4)
        if pred == "LONG_THOKO": long_weighted_votes += weight
        elif pred == "SHORT_THOKO": short_weighted_votes += weight

    macro_1h_prediction = data_matrix["timeframe_data"].get("1h", ["", "", "SCAN"])[2]
    
    if long_weighted_votes >= 4 and macro_1h_prediction == "LONG_THOKO": 
        final_signal = "LONG_THOKO"
    elif short_weighted_votes >= 4 and macro_1h_prediction == "SHORT_THOKO": 
        final_signal = "SHORT_THOKO"
    else: 
        return  
        
    if final_signal == "LONG_THOKO" and symbol != "BTC/USDT":
        if not await evaluate_bitcoin_macro_regime_lock():
            logging.info(f"ALT LONG REJECTED via global Bitcoin Dump Lock-Switch: {symbol}")
            return

    last_price = data_matrix["last_price"]
    trigger_atr_5m = data_matrix["trigger_atr_5m"]
    node_source = data_matrix["node_source"]
    order_flow_status = data_matrix["order_flow_status"]
    
    score = int(sum(data_matrix["tf_scores"]) / len(data_matrix["tf_scores"])) if data_matrix["tf_scores"] else 0
    
    atr_pct = (trigger_atr_5m / last_price) * 100
    clamped_atr_pct = max(1.0, min(10.0, atr_pct))
    effective_atr = last_price * (clamped_atr_pct / 100.0)
    
    alert_key = f"{chat_id}:{symbol}"
    async with STATE.alert_lock: allowed_alert = loop_start_time > STATE.alert_cooldown.get(alert_key, 0)
        
    if allowed_alert:
        if final_signal == "LONG_THOKO":
            stop_loss_val = last_price - (1.5 * effective_atr)
            take_profit_val = last_price + (3.0 * effective_atr)
            direction_label = "LONG 🟢"
        else:
            stop_loss_val = last_price + (1.5 * effective_atr)
            take_profit_val = last_price - (3.0 * effective_atr)
            direction_label = "SHORT 🔴"
            
        await db_log_signal_history_async(symbol, final_signal, last_price, score)
        
        sniper_msg = f"🎯 <b>🚨 MTF SNIPER ELITE TRIGGER: {html.escape(str(symbol))} ({direction_label})</b>\n"
        sniper_msg += f"• Composite Avg Score: <code>{score:+.1f}/100 🔥</code>\n"
        sniper_msg += f"• Execution Price: ${last_price:,.4f}\n"
        sniper_msg += f"• Target Node: <code>{html.escape(str(node_source))}</code>\n"
        sniper_msg += f"• Book Walls: <code>{html.escape(str(order_flow_status))}</code>\n\n"
        sniper_msg += f"🛡️ <b>RISK MATRIX RATIO SPECIFICATIONS:</b>\n"
        sniper_msg += f"🛑 Target Stop Loss: ${stop_loss_val:,.4f} (1.5x ATR Clamped)\n"
        sniper_msg += f"💰 Target Take Profit: ${take_profit_val:,.4f} (3x ATR Clamped)\n\n"
        sniper_msg += "• <i>Anti-overtrading channel cooldown activated for 30 minutes.</i>"
        try: 
            await asyncio.wait_for(application.bot.send_message(chat_id=chat_id, text=sniper_msg, parse_mode="HTML"), timeout=6.0)
            async with STATE.alert_lock:
                STATE.alert_cooldown[alert_key] = loop_start_time + 1800
        except Exception as e: logging.error(f"Alert output node pipeline breakdown: {e}")

async def send_global_dashboard_summary_report(chat_id, symbols_list, application):
    current_time = time.time()
    report_routing_key = f"{chat_id}_global_dashboard"
    
    async with STATE.report_cooldown_lock:
        if current_time < STATE.report_cooldown.get(report_routing_key, 0.0): return

    msg = "🛰️ <b>QUANT SNAPSHOT LIVING DASHBOARD MENU</b>\n"
    msg += f"⏱️ Time Stamp: <code>{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))}</code>\n"
    msg += "==================================\n"
    msg += "<code>SYMBOL       BIAS          TREND    BOOK</code>\n"
    msg += "----------------------------------\n"
    
    has_active_rows = False
    async with STATE.computed_signals_lock:
        for symbol in symbols_list:
            data_matrix = STATE.computed_signals_matrix.get(symbol)
            if not data_matrix: continue
            
            tf_scores = data_matrix["tf_scores"]
            timeframe_data = data_matrix["timeframe_data"]
            avg_score = sum(tf_scores) / len(tf_scores) if tf_scores else 0.0
            
            direction_bias = "LONG 🟢" if avg_score >= 35 else ("SHORT 🔴" if avg_score <= -35 else "SIDE ⏳")
            
            trend_label = timeframe_data.get("5m", ["SHANT", "NORMAL", "SCAN", "CH_DN", 0.0])[3]
            book_label = "BUY 📈" if "BUY" in data_matrix["order_flow_status"] else "SELL 📉"
            
            msg += f"<code>{html.escape(str(symbol)):<13}{direction_bias:<13}{trend_label:<9}{book_label}</code>\n"
            has_active_rows = True
            
    msg += "==================================\n"
    msg += "💡 <i>Updates posted concurrently every 15 minutes. Use /panel for controls.</i>"
    
    if has_active_rows:
        try:
            await asyncio.wait_for(application.bot.send_message(chat_id=chat_id, text=msg, parse_mode="HTML"), timeout=6.0)
            async with STATE.report_cooldown_lock:
                STATE.report_cooldown[report_routing_key] = current_time + 900
        except Exception as e:
            logging.error(f"Failed to post summary status dashboard: {e}")

# ============================================================================
# CONCURRENT MONITORING PIPELINE BACKGROUND ENGINE
# ============================================================================

async def monitoring_job(application: Application):
    last_market_refresh = time.time()
    while True:
        loop_start_time = time.time()
        
        if loop_start_time - last_market_refresh > 3600:
            try:
                await load_exchange_markets()
                for exchange in EXCHANGES:
                    if not exchange_available(exchange):
                        try: await exchange.fetch_time()
                        except: pass
                last_market_refresh = loop_start_time
            except Exception as e: logging.error(f"Hourly parameter reload crash context trace: {e}")

        async with STATE.ohlcv_cache_lock:
            stale_ohlcv = [k for k, v in STATE.ohlcv_cache.items() if loop_start_time - v[1] > 1800]
            for k in stale_ohlcv: STATE.ohlcv_cache.pop(k, None)
            
        async with STATE.coingecko_cache_lock:
            stale_cg_cache = [k for k, v in STATE.coingecko_cache.items() if loop_start_time - v[1] > CACHE_TTL]
            for k in stale_cg_cache: STATE.coingecko_cache.pop(k, None)
            if len(STATE.coingecko_id_map) > 2000:
                stale_keys = list(STATE.coingecko_id_map.keys())[:500]
                for k in stale_keys: STATE.coingecko_id_map.pop(k, None)

        async with STATE.orderbook_cache_lock:
            stale_ob = [k for k, v in STATE.orderbook_cache.items() if loop_start_time - v[1] > 30]
            for k in stale_ob: STATE.orderbook_cache.pop(k, None)
        async with STATE.waiting_lock:
            stale_waiting = [k for k, v in STATE.waiting_for_coin.items() if loop_start_time - v > 300]
            for k in stale_waiting: STATE.waiting_for_coin.pop(k, None)

        async with STATE.alert_lock:
            expired_alerts = [k for k, v in STATE.alert_cooldown.items() if loop_start_time > v]
            for k in expired_alerts: STATE.alert_cooldown.pop(k, None)
        async with STATE.report_cooldown_lock:
            expired_reports = [k for k, v in STATE.report_cooldown.items() if loop_start_time > v]
            for k in expired_reports: STATE.report_cooldown.pop(k, None)

        async with STATE.coingecko_cache_lock:
            stale_cg_locks = [k for k, v in COINGECKO_LOCKS.items() if loop_start_time - v["timestamp"] > 3600]
            for k in stale_cg_locks: COINGECKO_LOCKS.pop(k, None)

        async with STATE.tracked_pairs_lock:
            tracked_copy = list(STATE.tracked_pairs.items())
            symbols_list = sorted({sym for _, sub_set in tracked_copy for sym in sub_set})

        async with STATE.computed_signals_lock:
            active_symbols_set = set(symbols_list)
            stale_signals = [k for k in list(STATE.computed_signals_matrix.keys()) if k not in active_symbols_set]
            for k in stale_signals: STATE.computed_signals_matrix.pop(k, None)

        async with STATE.markets_validation_lock:
            STATE.symbol_to_exchange_snapshot = SYMBOL_TO_EXCHANGE.copy()

        async with STATE.symbol_registry_lock:
            for k in list(STATE.symbol_locks.keys()):
                if k not in symbols_list and STATE.symbol_active_counts.get(k, 0) <= 0:
                    STATE.symbol_locks.pop(k, None)
                    STATE.symbol_active_counts.pop(k, None)

        symbol_processing_list = list(symbols_list)
        if "BTC/USDT" not in symbol_processing_list and MARKETS_LOADED:
            symbol_processing_list.append("BTC/USDT")

        pending_symbols_set = set()
        for symbol in symbol_processing_list:
            if symbol not in pending_symbols_set and STATE.worker_queue.qsize() < 400:
                await STATE.worker_queue.put((symbol, loop_start_time))
                pending_symbols_set.add(symbol)
            
        await STATE.worker_queue.join()

        # SUBSCRIBER ROUTING LAYER
        parallel_subscriber_tasks = []
        for chat_id, pairs in tracked_copy:
            for symbol in list(pairs):
                parallel_subscriber_tasks.append(process_single_symbol_concurrency_block(symbol, chat_id, loop_start_time, application))
            parallel_subscriber_tasks.append(send_global_dashboard_summary_report(chat_id, list(pairs), application))
                
        if parallel_subscriber_tasks:
            await asyncio.gather(*parallel_subscriber_tasks, return_exceptions=True)

        await flush_database_commits_batch()

        next_cycle_target = loop_start_time + 300
        await asyncio.sleep(max(0.1, next_cycle_target - time.time()))

def main():
    application = Application.builder().token(TOKEN).post_init(startup_sequence).post_stop(shutdown_sequence).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("panel", show_panel))
    application.add_handler(CallbackQueryHandler(handle_button_clicks))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_input))
    
    application.run_polling(drop_pending_updates=True)

if __name__ == '__main__':
    main()
