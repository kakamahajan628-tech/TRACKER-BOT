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

# Initialize ONLY Permitted Exchanges (Gate.io + MEXC)
EXCHANGES = []
try:
    EXCHANGES.append(ccxt.gateio({'enableRateLimit': True}))
    EXCHANGES.append(ccxt.mexc({'enableRateLimit': True, 'options': {'defaultType': 'spot'}}))
    logging.info("Permitted exchange nodes (Gate.io & MEXC) initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing permitted nodes: {e}")

TRACKED_PAIRS = {}
TIMEFRAMES = ['5m', '15m', '1h', '4h']

def fetch_ohlcv_permitted(symbol, timeframe, limit=150):
    for exchange in EXCHANGES:
        try:
            market_symbol = symbol.upper()
            ohlcv = exchange.fetch_ohlcv(market_symbol, timeframe, limit=limit)
            if ohlcv and len(ohlcv) >= 40:
                return ohlcv, exchange
        except Exception:
            continue
    return None, None

def fetch_orderbook_imbalance(exchange, symbol):
    try:
        market_symbol = symbol.upper()
        orderbook = exchange.fetch_order_book(market_symbol, limit=20)
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

def analyze_predictive_metrics(ohlcv_data, exchange, symbol):
    try:
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        macd = ta.trend.MACD(close=df['close'], window_fast=12, window_slow=26, window_sign=9)
        df['macd_line'] = macd.macd()
        df['ema_200'] = ta.trend.ema_indicator(close=df['close'], window=200)
        
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        
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
        
        if last_bbw <= np.percentile(bbws[-20:], 20): squeeze_status = "SQUEEZE ⚡"
        elif last_bbw >= np.percentile(bbws[-20:], 85): squeeze_status = "EXPANDING 🌊"
        else: squeeze_status = "Stable"
        
        order_flow = fetch_orderbook_imbalance(exchange, symbol)
        
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
                text="🚀 *PREDICTIVE QUANT TERMINAL ONLINE!*\nContinuous matrix feed set to 5-minute intervals. Use `/track`.",
                parse_mode="Markdown"
            )
        except Exception as e: logging.error(f"Startup fail: {e}")

# Telegram Command Handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "⚡ *PREDICTIVE COIL TERMINAL V9.5*\n\n"
        "Commands:\n"
        "`/track BTC/USDT` - Map asset matrix to real-time loops\n"
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
    await update.message.reply_text(f"✅ Mapped *{symbol}* to 5-Minute Continuous Matrix Loop. Updates streaming soon.", parse_mode="Markdown")

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
    else: await update.message.reply_text(f"📋 *Active Watchlist (Permitted Nodes):*\n" + "\n".join([f"• {p}" for p in pairs]), parse_mode="Markdown")

# Background Monitoring Loop (MODIFIED: Runs strictly every 5 minutes / 300 seconds)
async def monitoring_job(application: Application):
    while True:
        await asyncio.sleep(300) # 5 minutes interval delay loop
        for chat_id, pairs in list(TRACKED_PAIRS.items()):
            for symbol in list(pairs):
                timeframe_data = {}
                last_price = 0.0
                node_source = "Unknown"
                macro_trend = "Unknown"
                has_data = False

                for tf in TIMEFRAMES:
                    try:
                        ohlcv, exchange_obj = fetch_ohlcv_permitted(symbol, tf, limit=150)
                        if ohlcv is None: continue
                        
                        trend, squeeze, order_flow, anomaly, prediction, price = analyze_predictive_metrics(ohlcv, exchange_obj, symbol)
                        if trend == "Error": continue

                        last_price = price
                        node_source = exchange_obj.name
                        macro_trend = trend
                        timeframe_data[tf] = (squeeze, order_flow, anomaly, prediction)
                        has_data = True
                    except Exception as loop_err: logging.error(f"Processing loop err: {loop_err}")

                if has_data and timeframe_data:
                    is_msb = any("MSB" in data[3] for data in timeframe_data.values())
                    header = "🔥 ALERT: STRUCTURE BREAK DETECTED" if is_msb else "🛰️ QUANT PREDICTIVE LIVE MATRIX"
                    
                    msg = f"*{header}: {symbol}*\n"
                    msg += f"• *Price:* ${last_price:,.4f}\n"
                    msg += f"• *Macro Trend (EMA 200):* {macro_trend}\n"
                    msg += f"• *Execution Node:* {node_source}\n"
                    msg += "▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔\n"
                    msg += "`TF    │ SQUEEZE  │ ORDERBOOK FLOW   │ FUTURE PRED`\n"
                    msg += "─────────────────────────────────────────\n"
                    
                    for tf in TIMEFRAMES:
                        if tf in timeframe_data:
                            squeeze, order_flow, anomaly, prediction = timeframe_data[tf]
                            
                            if anomaly != "Clear":
                                display_pred = prediction if "Scanning" not in prediction else anomaly
                                if "Scanning" not in prediction and anomaly != "Clear":
                                    display_pred = f"{prediction} ({anomaly})"
                            else:
                                display_pred = prediction
                                
                            msg += f"`{tf:<6}│ {squeeze:<9}│ {order_flow:<17}│` {display_pred}\n"
                    
                    msg += "─────────────────────────────────────────\n"
                    msg += "💡 *Predictive Key:* _MSB rules map mid-term trend flips. LIQ SWEEPS detect institutional stop hunting._"
                    
                    try:
                        await application.bot.send_message(chat_id=chat_id, text=msg, parse_mode="Markdown")
                    except Exception as send_err: logging.error(f"Telegram block delivery fail: {send_err}")

# Web Server for Render Keep-Alive
app = Flask(__name__)
@app.route('/')
def health_check(): return "Continuous 5-Minute Quant Feed Active", 200

def run_web_server():
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

def main():
    if not TOKEN: sys.exit(1)
    Thread(target=run_web_server, daemon=True).start()
    application = Application.builder().token(TOKEN).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("track", track_coin))
    application.add_handler(CommandHandler("stop", stop_coin))
    application.add_handler(CommandHandler("status", status))
    
    loop = asyncio.get_event_loop()
    loop.create_task(send_startup_message(application))
    loop.create_task(monitoring_job(application))
    application.run_polling()

if __name__ == '__main__':
    main()
