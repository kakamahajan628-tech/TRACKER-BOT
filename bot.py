import os
import asyncio
import logging
import sys
import numpy as np
import pandas as pd
import ccxt
import ta
import requests
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
            if ohlcv and len(ohlcv) >= 4:
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
        if total_volume == 0: return 50.0, "Neutral"
        bid_ratio = (total_bids / total_volume) * 100
        if bid_ratio >= 65: return bid_ratio, f"{bid_ratio:.0f}% Bid Heavy"
        elif bid_ratio <= 35: return bid_ratio, f"{(100 - bid_ratio):.0f}% Ask Heavy"
        return bid_ratio, "Neutral"
    except Exception:
        return 50.0, "Scanning"

def scrape_public_onchain_intel(symbol):
    try:
        clean_ticker = symbol.split('/')[0].upper()
        url = f"https://api.coingecko.com/api/v3/search?query={clean_ticker}"
        res = requests.get(url, timeout=5).json()
        
        if 'coins' in res and len(res['coins']) > 0:
            coin_id = res['coins'][0]['id']
            detail_url = f"https://api.coingecko.com/api/v3/coins/{coin_id}?localization=false&tickers=false&market_data=true&community_data=false&developer_data=false&sparkline=false"
            coin_data = requests.get(detail_url, timeout=5).json()
            
            mcap = coin_data.get('market_data', {}).get('market_cap', {}).get('usd', 0)
            vol_24h = coin_data.get('market_data', {}).get('total_volume', {}).get('usd', 0)
            price_change = coin_data.get('market_data', {}).get('price_change_percentage_24h', 0)
            
            if vol_24h > 0 and mcap > 0:
                v2m_ratio = vol_24h / mcap
                if price_change < -10 and v2m_ratio > 0.35: return "UNDERGROUND_ACC"
                elif v2m_ratio > 0.40: return "HEAVY_WHALE_ACC"
                elif v2m_ratio > 0.20: return "ACTIVE_FLOW"
        return "STABLE_HOLD"
    except Exception:
        return "ROUTING"

def find_peaks_and_troughs(price, indicator, window=2):
    p_peaks, p_troughs = [], []
    i_peaks, i_troughs = [], []
    if len(price) < (window * 2 + 1): return p_peaks, p_troughs, i_peaks, i_troughs
    for i in range(window, len(price) - window):
        if price[i] == max(price[i-window:i+window+1]): p_peaks.append((i, price[i]))
        if price[i] == min(price[i-window:i+window+1]): p_troughs.append((i, price[i]))
        if indicator[i] == max(indicator[i-window:i+window+1]): i_peaks.append((i, indicator[i]))
        if indicator[i] == min(indicator[i-window:i+window+1]): i_troughs.append((i, indicator[i]))
    return p_peaks, p_troughs, i_peaks, i_troughs

def analyze_predictive_metrics(ohlcv_data, exchange, symbol):
    try:
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        total_candles = len(df)
        
        df['vol_ma'] = df['volume'].rolling(window=min(10, total_candles)).mean()
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        
        # Core Indicators: RSI & MACD
        df['rsi'] = ta.momentum.rsi(close=df['close'], window=min(14, max(4, total_candles-1)))
        macd_w = min(12, max(2, int(total_candles/3)))
        macd = ta.trend.MACD(close=df['close'], window_fast=macd_w, window_slow=macd_w*2, window_sign=3)
        df['macd_line'] = macd.macd()
        
        df['ema_50'] = ta.trend.ema_indicator(close=df['close'], window=min(50, total_candles))
        if total_candles >= 20:
            df['ema_200'] = ta.trend.ema_indicator(close=df['close'], window=min(200, total_candles))
            bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
            df['bbw'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
        else:
            df['ema_200'] = df['close'].rolling(window=total_candles).mean()
            df['bbw'] = df['close'].pct_change().rolling(window=total_candles-1).std()

        df = df.dropna().reset_index(drop=True)
        if len(df) == 0:
            return "Discovery", "Discovery", "Scanning", "Clear", "Velocity Load", ohlcv_data[-1][4], 0

        prices = df['close'].to_numpy()
        highs = df['high'].to_numpy()
        lows = df['low'].to_numpy()
        volumes = df['volume'].to_numpy()
        vol_mas = df['vol_ma'].to_numpy()
        ind_vals = df['macd_line'].to_numpy() if 'macd_line' in df else prices
        bbws = df['bbw'].to_numpy() if 'bbw' in df else np.zeros(len(df))
        vwaps = df['vwap'].to_numpy()
        rsis = df['rsi'].to_numpy()
        
        last_price = prices[-1]
        last_vol = volumes[-1]
        last_vol_ma = vol_mas[-1] if len(vol_mas) > 0 and not pd.isna(vol_mas[-1]) else 1.0
        last_vwap = vwaps[-1]
        last_rsi = rsis[-1] if len(rsis) > 0 else 50.0
        
        c_score = 0
        
        last_ema50 = df['ema_50'].iloc[-1] if 'ema_50' in df and len(df['ema_50']) > 0 else last_price
        last_ema200 = df['ema_200'].iloc[-1] if 'ema_200' in df and len(df['ema_200']) > 0 else last_price
        
        p_p, p_t, _, _ = find_peaks_and_troughs(prices, ind_vals, window=2)
        
        structure_trend = "Neutral"
        if total_candles <= 25:
            if last_price > last_ema50: 
                structure_trend = "MICRO_BULL"
                c_score += 1
            elif last_price < last_ema50: 
                structure_trend = "MICRO_BEAR"
                c_score -= 1
        else:
            if len(p_p) >= 2 and len(p_t) >= 2:
                higher_high = p_p[-1][1] > p_p[-2][1]
                higher_low = p_t[-1][1] > p_t[-2][1]
                lower_high = p_p[-1][1] < p_p[-2][1]
                lower_low = p_t[-1][1] < p_t[-2][1]
                
                if higher_high and higher_low and last_price > last_ema200: 
                    structure_trend = "STRG_BULL"
                    c_score += 2
                elif lower_high and lower_low and last_price < last_ema200: 
                    structure_trend = "STRG_BEAR"
                    c_score -= 2
                elif last_price > last_ema200: 
                    structure_trend = "WK_BULL"
                    c_score += 1
                elif last_price < last_ema200: 
                    structure_trend = "WK_BEAR"
                    c_score -= 1
            else:
                if last_price >= last_ema200: 
                    structure_trend = "BULL"
                    c_score += 1
                else: 
                    structure_trend = "BEAR"
                    c_score -= 1

        squeeze_status = "Stable"
        if total_candles >= 20 and len(bbws) >= 20:
            if bbws[-1] <= np.percentile(bbws[-20:], 20): squeeze_status = "SQUEEZE"
            elif bbws[-1] >= np.percentile(bbws[-20:], 85): squeeze_status = "EXPANDING"
        else:
            squeeze_status = "Discovery"
            
        bid_pct, order_flow = fetch_orderbook_imbalance(exchange, symbol)
        if "Bid" in order_flow: c_score += 1
        elif "Ask" in order_flow: c_score -= 1
        
        future_pred = "Scanning"
        
        # 🛡️ THE ANTI-LIQUIDATION RSI EXHAUSTION CORE
        if last_rsi >= 75:
            # Check for Volume Climax Exhaustion & Heavy Sell Walls (Ask Heavy)
            is_vol_exhausted = last_vol < last_vol_ma
            is_sell_wall_heavy = bid_pct <= 35
            is_rsi_hooked = len(rsis) >= 2 and rsis[-1] < rsis[-2] # RSI bending down
            
            if is_vol_exhausted and is_sell_wall_heavy and is_rsi_hooked:
                future_pred = "🚨 EXHAUST_SHORT_CONFIRM"
                c_score -= 3
            elif last_vol > (last_vol_ma * 2.5):
                future_pred = "⚠️ FOOL_BULL_RUSH (DO NOT SHORT)"
                c_score += 2
            else:
                future_pred = "⏳ RSI_HIGH (WAIT_PIVOT)"
        else:
            if total_candles <= 25:
                if last_vol > (last_vol_ma * 3.0):
                    if last_price >= prices[-2]: 
                        future_pred = "VOL_LAUNCH (PUMP)"
                        c_score += 2
                    else: 
                        future_pred = "DUMP_BURST (CRASH)"
                        c_score -= 2
                elif highs[-1] > last_price and (highs[-1] - last_price) > (last_price - lows[-1]) * 2:
                    future_pred = "TOP WHALE SELLING"
                    c_score -= 1
            else:
                if len(p_p) >= 2 and len(p_t) >= 2:
                    recent_high = p_p[-1][1]
                    recent_low = p_t[-1][1]
                    
                    if last_price > recent_high and bid_pct <= 35:
                        future_pred = "BREAKOUT_TRAP (SHORT)"
                        c_score -= 2
                    elif last_price < recent_low and bid_pct >= 65:
                        future_pred = "BREAKOUT_TRAP (LONG)"
                        c_score += 2
                    elif last_price > recent_high and "BEAR" in structure_trend: 
                        future_pred = "BULLISH MSB"
                        c_score += 2
                    elif last_price < recent_low and "BULL" in structure_trend: 
                        future_pred = "BEARISH MSB"
                        c_score -= 2
                    elif lows[-1] < recent_low and last_price > recent_low: 
                        future_pred = "LIQ SWEEP (PUMP)"
                        c_score += 2
                    elif highs[-1] > recent_high and last_price < recent_high: 
                        future_pred = "LIQ SWEEP (DUMP)"
                        c_score -= 2

        if future_pred == "Scanning":
            if last_price > (last_vwap * 1.05): future_pred = "VWAP REV DOWN"
            elif last_price < (last_vwap * 0.95): future_pred = "VWAP REV UP"

        anomaly_status = "Clear"
        if total_candles >= 15:
            if len(p_t) >= 2 and len(ind_vals) >= 2:
                if p_t[-1][1] < p_t[-2][1] and ind_vals[-1] > ind_vals[-2]: 
                    anomaly_status = "REG_BULL"
                    c_score += 1
            if len(p_p) >= 2 and len(ind_vals) >= 2:
                if p_p[-1][1] > p_p[-2][1] and ind_vals[-1] < ind_vals[-2]: 
                    anomaly_status = "REG_BEAR"
                    c_score -= 1
                
        return structure_trend, squeeze_status, order_flow, anomaly_status, future_pred, last_price, c_score
    except Exception as e:
        logging.error(f"Error inside predictive quant engine: {e}")
        return "Error", "Error", "Error", "Error", "Error", 0.0, 0

# Startup Signal Function
async def send_startup_message(application: Application):
    if USER_CHAT_ID:
        try:
            await asyncio.sleep(3)
            await application.bot.send_message(
                chat_id=USER_CHAT_ID,
                text="🚀 <b>EXHAUSTION PRO QUANT ENGINE TERMINAL ONLINE</b>\nRSI-Trap dynamic protection activated. Orderbook sell-walls filter live. Use /track.",
                parse_mode="HTML"
            )
        except Exception as e: logging.error(f"Startup fail: {e}")

# Telegram Command Handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "⚡ <b>ALPHA PIVOT EXHAUSTION TERMINAL</b>\n\n"
        "Commands:\n"
        "/track COIN/USDT - Map pair into trend exhaustion loop\n"
        "/stop COIN/USDT - Unmap tracking vectors\n"
        "/status - View dynamic watchlist dashboard", 
        parse_mode="HTML"
    )

async def track_coin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not context.args:
        await update.message.reply_text("❌ Specify asset pair matrix. Ex: /track SOL/USDT")
        return
    symbol = context.args[0].upper()
    if chat_id not in TRACKED_PAIRS: TRACKED_PAIRS[chat_id] = set()
    TRACKED_PAIRS[chat_id].add(symbol)
    await update.message.reply_text(f"✅ Mapped <b>{symbol}</b> to Institutional Exhaustion Matrix. Streaming cycles active at 5m.", parse_mode="HTML")

async def stop_coin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not context.args: return
    symbol = context.args[0].upper()
    if chat_id in TRACKED_PAIRS and symbol in TRACKED_PAIRS[chat_id]:
        TRACKED_PAIRS[chat_id].remove(symbol)
        await update.message.reply_text(f"🛑 Unmapped <b>{symbol}</b> from terminal core.", parse_mode="HTML")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    pairs = TRACKED_PAIRS.get(chat_id, set())
    if not pairs: await update.message.reply_text("Watchlist array is empty.")
    else: 
        watchlist_str = "\n".join([f"• {p}" for p in pairs])
        await update.message.reply_text(f"📋 <b>Active Watchlist:</b>\n{watchlist_str}", parse_mode="HTML")

# Background Monitoring 5-Minute Execution Loop
async def monitoring_job(application: Application):
    while True:
        await asyncio.sleep(300) # Strict 5-Minute loop block
        for chat_id, pairs in list(TRACKED_PAIRS.items()):
            for symbol in list(pairs):
                timeframe_data = {}
                last_price = 0.0
                node_source = "Unknown"
                has_data = False
                total_confluence_score = 0

                onchain_intel = scrape_public_onchain_intel(symbol)

                for tf in TIMEFRAMES:
                    try:
                        ohlcv, exchange_obj = fetch_ohlcv_permitted(symbol, tf, limit=150)
                        if ohlcv is None: continue
                        
                        structure_trend, squeeze, order_flow, anomaly, prediction, price, score = analyze_predictive_metrics(ohlcv, exchange_obj, symbol)
                        if structure_trend == "Error": continue

                        last_price = price
                        node_source = exchange_obj.name
                        total_confluence_score += score
                        timeframe_data[tf] = (squeeze, order_flow, anomaly, prediction, structure_trend)
                        has_data = True
                    except Exception as loop_err: logging.error(f"Processing loop err: {loop_err}")

                if has_data and timeframe_data:
                    if total_confluence_score >= 4: global_bias = "EXTREME_BUY 🚀"
                    elif total_confluence_score >= 1: global_bias = "MODERATE_BULL 🟢"
                    elif total_confluence_score <= -4: global_bias = "EXTREME_SHORT 💥"
                    elif total_confluence_score <= -1: global_bias = "MODERATE_BEAR 🔴"
                    else: global_bias = "NEUTRAL CONGESTION ⏳"
                    
                    # Highlight Top Urgent Warnings
                    is_exhausted = any("EXHAUST_SHORT_CONFIRM" in data[3] for data in timeframe_data.values())
                    is_fool_rush = any("FOOL_BULL_RUSH" in data[3] for data in timeframe_data.values())
                    
                    if is_exhausted: header = "🚨 COIN EXHAUSTION DETECTED (SAFE TO SHORT)"
                    elif is_fool_rush: header = "⚠️ FOOLISH BULL RUSH (DO NOT SHORT)"
                    else: header = "🛰️ QUANT PREDICTIVE MATRIX FEED"
                    
                    msg = f"<b>{header}: {symbol}</b>\n"
                    msg += f"• Price: ${last_price:,.4f}\n"
                    msg += f"• <b>ON-CHAIN INTEL:</b> <code>{onchain_intel}</code>\n"
                    msg += f"• <b>NET QUANT SCORE:</b> <code>{total_confluence_score:+} ({global_bias})</code>\n"
                    msg += f"• Execution Node: {node_source}\n"
                    msg += "▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔\n"
                    msg += "<code>TF    │ TREND     │ SQUEEZE   │ ORDERBOOK        │ FUTURE PRED</code>\n"
                    msg += "───────────────────────────────────────────────────────\n"
                    
                    for tf in TIMEFRAMES:
                        if tf in timeframe_data:
                            squeeze, order_flow, anomaly, prediction, structure_trend = timeframe_data[tf]
                            
                            if anomaly != "Clear":
                                display_pred = prediction if "Scanning" not in prediction else anomaly
                                if "Scanning" not in prediction and anomaly != "Clear":
                                    display_pred = f"{prediction} ({anomaly})"
                            else:
                                display_pred = prediction
                                
                            msg += f"<code>{tf:<6}│ {structure_trend:<10}│ {squeeze:<10}│ {order_flow:<17}│</code> {display_pred}\n"
                    
                    msg += "───────────────────────────────────────────────────────\n"
                    msg += "💡 <i>Predictive Key: EXHAUST_SHORT filters false tops using Orderbook walls & volume decay. Do not fight FOOL_BULL_RUSH.</i>"
                    
                    try:
                        await application.bot.send_message(chat_id=chat_id, text=msg, parse_mode="HTML")
                    except Exception as send_err: logging.error(f"Telegram HTML alpha matrix dispatch fail: {send_err}")

# Web Server for Render Keep-Alive
app = Flask(__name__)
@app.route('/')
def health_check(): return "Anti Liquidation Engine Live", 200

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
