import os
import asyncio
import logging
import sys
import numpy as np
import pandas as pd
import ccxt
import ta
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from threading import Thread
from flask import Flask

# Logging setup to track everything in Render console
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    stream=sys.stdout
)

TOKEN = os.getenv("TELEGRAM_TOKEN")
raw_chat_id = os.getenv("USER_CHAT_ID")
USER_CHAT_ID = int(raw_chat_id) if (raw_chat_id and raw_chat_id.strip().isdigit()) else None

# Initialize ONLY Gate.io safely
GATEIO = None
try:
    GATEIO = ccxt.gateio({'enableRateLimit': True})
    logging.info("Gate.io exchange interface initialized as the sole data provider.")
except Exception as e:
    logging.error(f"Error initializing Gate.io: {e}")

TRACKED_PAIRS = {}
TIMEFRAMES = ['5m', '15m', '1h', '4h']

def fetch_ohlcv_gate(symbol, timeframe, limit=150):
    try:
        if not GATEIO: return None
        market_symbol = symbol.upper()
        ohlcv = GATEIO.fetch_ohlcv(market_symbol, timeframe, limit=limit)
        if ohlcv and len(ohlcv) >= 40:
            return ohlcv
    except Exception as e:
        logging.warning(f"Failed fetching {symbol} from Gate.io: {e}")
    return None

def fetch_orderbook_imbalance_gate(symbol):
    try:
        if not GATEIO: return "Scanning..."
        market_symbol = symbol.upper()
        orderbook = GATEIO.fetch_order_book(market_symbol, limit=20)
        total_bids = sum([bid[1] for bid in orderbook['bids']])
        total_asks = sum([ask[1] for ask in orderbook['asks']])
        total_volume = total_bids + total_asks
        if total_volume == 0: return "50% Net"
        bid_ratio = (total_bids / total_volume) * 100
        if bid_ratio >= 65: return f"{bid_ratio:.0f}% Bid Heavy 🟢"
        elif bid_ratio <= 35: return f"{(100 - bid_ratio):.0f}% Ask Heavy 🔴"
        return "Neutral 🟡"
    except Exception:
        return "Scanning..."

def find_peaks_and_troughs(price, indicator, window=3):
    p_peaks, p_troughs = [], []
    i_peaks, i_troughs = [], []
    for i in range(window, len(price) - window):
        if price[i] == max(price[i-window:i+window+1]): p_peaks.append((i, price[i]))
        if price[i] == min(price[i-window:i+window+1]): p_troughs.append((i, price[i]))
        if indicator[i] == max(indicator[i-window:i+window+1]): i_peaks.append((i, indicator[i]))
        if indicator[i] == min(indicator[i-window:i+window+1]): i_troughs.append((i, indicator[i]))
    return p_peaks, p_troughs, i_peaks, i_troughs

def analyze_predictive_metrics(ohlcv_data, symbol):
    """
    Predictive HFT Mathematical Core running exclusively on Gate.io streams
    """
    try:
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # 1. Structural Mathematics & Micro MACD
        macd = ta.trend.MACD(close=df['close'], window_fast=12, window_slow=26, window_sign=9)
        df['macd_line'] = macd.macd()
        df['ema_200'] = ta.trend.ema_indicator(close=df['close'], window=200)
        
        # Anchored VWAP proxy
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        
        # Bollinger Band Width
        bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bbw'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
        
        df = df.dropna().reset_index(drop=True)
        
        if len(df) < 15: 
            return "Neutral", "Low Data", "Scanning...", "Clear", "📉 Data Squeeze", df['close'].iloc[-1] if len(df) > 0 else 0.0
            
        prices = df['close'].to_numpy()
        highs = df['high'].to_numpy()
        lows = df['low'].to_numpy()
        ind_vals = df['macd_line'].to_numpy()
        bbws = df['bbw'].to_numpy()
        vwaps = df['vwap'].to_numpy()
        
        last_price = prices[-1]
        last_bbw = bbws[-1]
        last_vwap = vwaps[-1]
        
        trend = "🟢 BULL" if last_price >= df['ema_200'].iloc[-1] else "🔴 BEAR"
        
        # Volatility Squeeze Matrix
        if last_bbw <= np.percentile(bbws[-20:], 20): squeeze_status = "SQUEEZE ⚡"
        elif last_bbw >= np.percentile(bbws[-20:], 85): squeeze_status = "EXPANDING 🌊"
        else: squeeze_status = "Stable"
        
        order_flow = fetch_orderbook_imbalance_gate(symbol)
        
        future_pred = "📉 Scanning"
        p_p, p_t, _, _ = find_peaks_and_troughs(prices, ind_vals, window=2)
        
        if len(p_p) >= 2 and len(p_t) >= 2:
            recent_high = p_p[-1][1]
            recent_low = p_t[-1][1]
            
            if last_price > recent_high and trend == "🔴 BEAR": future_pred = "🚀 BULLISH MSB"
            elif last_price < recent_low and trend == "🟢 BULL": future_pred = "💥 BEARISH MSB"
            elif lows[-1] < recent_low and last_price > recent_low: future_pred = "🟢 LIQ SWEEP (PUMP)"
            elif highs[-1] > recent_high and last_price < recent_high: future_pred = "🔴 LIQ SWEEP (DUMP)"
                
        if future_pred == "📉 Scanning":
            if last_price > (last_vwap * 1.06): future_pred = "🧲 VWAP DROP PRED"
            elif last_price < (last_vwap * 0.94): future_pred = "🧲 VWAP PUMP PRED"

        anomaly_status = "Clear"
        if len(p_t) >= 2 and len(ind_vals) >= 2:
            if p_t[-1][1] < p_t[-2][1] and ind_vals[-1] > ind_vals[-2]: anomaly_status = "🔥 REG_BULL"
            elif p_t[-1][1] > p_t[-2][1] and ind_vals[-1] < ind_vals[-2]: anomaly_status = "⚡ HID_BULL"
        if len(p_p) >= 2 and len(ind_vals) >= 2:
            if p_p[-1][1] > p_p[-2][1] and ind_vals[-1] < ind_vals[-2]: anomaly_status = "💥 REG_BEAR"
            elif p_p[-1][1] < p_p[-2][1] and ind_vals[-1] > ind_vals[-2]: anomaly_status = "❄️ HID_BEAR"
                
        return trend, squeeze_status, order_flow, anomaly_status, future_pred, last_price
    except Exception as e:
        logging.error(f"Error inside math execution: {e}")
        return "Error", "Error", "Error", "Error", "Error", 0.0

# Startup Signal Function
async def send_startup_message(application: Application):
    if USER_CHAT_ID:
        try:
            await asyncio.sleep(3)
            await application.bot.send_message(
                chat_id=USER_CHAT_ID,
                text="🚀 *PREDICTIVE GATE.IO QUANT TERMINAL ONLINE!*\nBinance/Bybit nodes removed completely. Streamlined polling active. Use `/track`.",
                parse_mode="Markdown"
            )
        except Exception as e: logging.error(f"Startup fail: {e}")

# Telegram Command Handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "⚡ *PREDICTIVE GATE TERMINAL V8.0*\n\n"
        "Commands:\n"
        "`/track BTC/USDT` - Load pair to predictive structural array\n"
        "`/stop BTC/USDT` - Unmap tracking vectors\n"
        "`/status` - View watchlist", 
        parse_mode="Markdown"
    )

async def track_coin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not context.args:
        await update.message.reply_text("❌ Specify asset pair matrix. Ex: `/track BTC/USDT`")
        return
    symbol = context.args[0].upper()
    if chat_id not in TRACKED_PAIRS: TRACKED_PAIRS[chat_id] = set()
    TRACKED_PAIRS[chat_id].add(symbol)
    await update.message.reply_text(f"✅ Mapped *{symbol}* to Gate.io Predictive Operational Matrix.", parse_mode="Markdown")

async def stop_coin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not context.args: return
    symbol = context.args[0].upper()
    if chat_id in TRACKED_PAIRS and symbol in TRACKED_PAIRS[chat_id]:
        TRACKED_PAIRS[chat_id].remove(symbol)
        await update.message.reply_text(f"🛑 Unmapped *{symbol}*.", parse_mode="Markdown")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    pairs = TRACKED_PAIRS.get(chat_id, set())
    if not pairs: await update.message.reply_text("Watchlist empty.")
    else: await update.message.reply_text(f"📋 *Active Watchlist (Gate.io Only):*\n" + "\n".join([f"• {p}" for p in pairs]), parse_mode="Markdown")

# Background Monitoring Core Execution Loop
async def monitoring_job(application: Application):
    while True:
        await asyncio.sleep(60)
        for chat_id, pairs in list(TRACKED_PAIRS.items()):
            for symbol in list(pairs):
                timeframe_data = {}
                trigger_alert = False
                last_price = 0.0
                macro_trend = "Unknown"

                for tf in TIMEFRAMES:
                    try:
                        ohlcv = fetch_ohlcv_gate(symbol, tf, limit=150)
                        if ohlcv is None: continue
                        
                        trend, squeeze, order_flow, anomaly, prediction, price = analyze_predictive_metrics(ohlcv, symbol)
                        if trend == "Error": continue

                        last_price = price
                        macro_trend = trend
                        timeframe_data[tf] = (squeeze, order_flow, anomaly, prediction)

                        # Trigger alerts on valid predictive confluences or tracking states
                        if "MSB" in prediction or "LIQ" in prediction or "SQUEEZE" in squeeze or "REG_" in anomaly or "HID_" in anomaly or "Scanning" in prediction:
                            trigger_alert = True
                    except Exception as loop_err: logging.error(f"Processing loop err: {loop_err}")

                if trigger_alert and timeframe_data:
                    is_msb = any("MSB" in data[3] for data in timeframe_data.values())
                    header = "🔥 ALERT: STRUCTURE BREAK DETECTED" if is_msb else "🛰️ QUANT PREDICTIVE MATRIX VECTOR"
