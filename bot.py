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

# Advanced Autonomous Loop with 4-second hard network timeout barrier
async def fetch_ohlcv_permitted(symbol, timeframe, limit=150):
    for exchange in EXCHANGES:
        try:
            market_symbol = symbol.upper()
            ohlcv = await asyncio.wait_for(
                asyncio.to_thread(exchange.fetch_ohlcv, market_symbol, timeframe, limit=limit),
                timeout=4.0
            )
            if ohlcv and len(ohlcv) >= 4:
                return ohlcv, exchange
        except Exception:
            continue
    return None, None

def fetch_orderbook_advanced_metrics(exchange, symbol):
    try:
        market_symbol = symbol.upper()
        orderbook = exchange.fetch_order_book(market_symbol, limit=20)
        
        bids = orderbook['bids']
        asks = orderbook['asks']
        
        total_bids = sum([bid[1] for bid in bids])
        total_asks = sum([ask[1] for ask in asks])
        total_volume = total_bids + total_asks
        
        bid_ratio = (total_bids / total_volume) * 100 if total_volume > 0 else 50.0
        
        if len(bids) > 0 and len(asks) > 0:
            best_bid = bids[0][0]
            best_ask = asks[0][0]
            spread_pct = ((best_ask - best_bid) / best_bid) * 100
        else:
            spread_pct = 0.0
            
        if spread_pct >= 0.25: return bid_ratio, "KHATRA_GAP"
        if bid_ratio >= 65: return bid_ratio, f"{bid_ratio:.0f}%BUYER"
        elif bid_ratio <= 35: return bid_ratio, f"{(100 - bid_ratio):.0f}%SELLER"
        return bid_ratio, "NORMAL"
    except Exception:
        return 50.0, "SCANNING"

def scrape_public_onchain_intel(symbol):
    try:
        clean_ticker = symbol.split('/')[0].upper()
        url = f"https://api.coingecko.com/api/v3/search?query={clean_ticker}"
        res = requests.get(url, timeout=4).json()
        
        if 'coins' in res and len(res['coins']) > 0:
            coin_id = res['coins'][0]['id']
            detail_url = f"https://api.coingecko.com/api/v3/coins/{coin_id}?localization=false&tickers=false&market_data=true&community_data=false&developer_data=false&sparkline=false"
            coin_data = requests.get(detail_url, timeout=4).json()
            
            mcap = coin_data.get('market_data', {}).get('market_cap', {}).get('usd', 0)
            vol_24h = coin_data.get('market_data', {}).get('total_volume', {}).get('usd', 0)
            price_change = coin_data.get('market_data', {}).get('price_change_percentage_24h', 0)
            
            if vol_24h > 0 and mcap > 0:
                v2m_ratio = vol_24h / mcap
                if price_change < -10 and v2m_ratio > 0.35: return "WHALE_CHUPCHAP_BUY"
                elif v2m_ratio > 0.40: return "WHALE_MAL_IKATHA_KR_RHI"
                elif v2m_ratio > 0.20: return "HEAVY_VOLUME_FLOW"
        return "STABLE_HOLDERS"
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

def analyze_predictive_metrics(ohlcv_data, bid_pct, order_flow_status, symbol):
    try:
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        total_candles = len(df)
        
        df['vol_ma'] = df['volume'].rolling(window=min(10, total_candles)).mean()
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        
        df['candle_delta'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 0.000001) * df['volume']
        df['cumulative_delta_ma'] = df['candle_delta'].rolling(window=min(5, total_candles)).mean()
        
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
            return "NORMAL", "NORMAL", "Clear", "SCANNING", ohlcv_data[-1][4], 0

        prices = df['close'].to_numpy()
        highs = df['high'].to_numpy()
        lows = df['low'].to_numpy()
        volumes = df['volume'].to_numpy()
        vol_mas = df['vol_ma'].to_numpy()
        ind_vals = df['macd_line'].to_numpy() if 'macd_line' in df else prices
        bbws = df['bbw'].to_numpy() if 'bbw' in df else np.zeros(len(df))
        vwaps = df['vwap'].to_numpy()
        rsis = df['rsi'].to_numpy()
        deltas = df['cumulative_delta_ma'].to_numpy()
        
        last_price = prices[-1]
        last_vol = volumes[-1]
        last_vol_ma = vol_mas[-1] if len(vol_mas) > 0 and not pd.isna(vol_mas[-1]) else 1.0
        last_vwap = vwaps[-1]
        last_rsi = rsis[-1] if len(rsis) > 0 else 50.0
        last_delta = deltas[-1] if len(deltas) > 0 else 0.0
        
        c_score = 0
        
        last_ema50 = df['ema_50'].iloc[-1] if 'ema_50' in df and len(df['ema_50']) > 0 else last_price
        last_ema200 = df['ema_200'].iloc[-1] if 'ema_200' in df and len(df['ema_200']) > 0 else last_price
        
        p_p, p_t, _, _ = find_peaks_and_troughs(prices, ind_vals, window=2)
        
        # Pure Hinglish Trend Output
        structure_trend = "NORMAL"
        if total_candles <= 25:
            if last_price > last_ema50: 
                structure_trend = "UP_CHOTA"
                c_score += 1
            elif last_price < last_ema50: 
                structure_trend = "DN_CHOTA"
                c_score -= 1
        else:
            if len(p_p) >= 2 and len(p_t) >= 2:
                higher_high = p_p[-1][1] > p_p[-2][1]
                higher_low = p_t[-1][1] > p_t[-2][1]
                lower_high = p_p[-1][1] < p_p[-2][1]
                lower_low = p_t[-1][1] < p_t[-2][1]
                
                if higher_high and higher_low and last_price > last_ema200: 
                    structure_trend = "TEZ_UP"
                    c_score += 2
                elif lower_high and lower_low and last_price < last_ema200: 
                    structure_trend = "TEZ_DOWN"
                    c_score -= 2
                elif last_price > last_ema200: 
                    structure_trend = "UP_HALKA"
                    c_score += 1
                elif last_price < last_ema200: 
                    structure_trend = "DN_HALKA"
                    c_score -= 1
            else:
                if last_price >= last_ema200: 
                    structure_trend = "TEJI"
                    c_score += 1
                else: 
                    structure_trend = "MANDI"
                    c_score -= 1

        squeeze_status = "SHANT"
        if total_candles >= 20 and len(bbws) >= 20:
            if bbws[-1] <= np.percentile(bbws[-20:], 20): squeeze_status = "BADA_MOVE"
            elif bbws[-1] >= np.percentile(bbws[-20:], 85): squeeze_status = "BHAAG_RHA"
        else:
            squeeze_status = "NAYA_COIN"
            
        if "BUYER" in order_flow_status: c_score += 1
        elif "SELLER" in order_flow_status: c_score -= 1
        
        future_pred = "SCANNING"
        
        if last_rsi >= 75:
            is_vol_exhausted = last_vol < last_vol_ma
            is_sell_wall_heavy = bid_pct <= 35
            is_rsi_hooked = len(rsis) >= 2 and rsis[-1] < rsis[-2]
            is_delta_divergent = last_delta < 0
            is_velocity_decaying = len(rsis) >= 3 and (rsis[-1] - rsis[-2]) < (rsis[-2] - rsis[-3])
            
            if order_flow_status == "KHATRA_GAP":
                future_pred = "🎰 VOID_TRAP_MATH"
                c_score -= 3
            elif is_vol_exhausted and is_sell_wall_heavy and is_rsi_hooked and is_delta_divergent and is_velocity_decaying:
                future_pred = "🎯 SHORT_THOKO_ABHI"
                c_score -= 5
            elif is_vol_exhausted and is_sell_wall_heavy and is_rsi_hooked:
                future_pred = "🚨 DUM_KHTM_SHORT"
                c_score -= 3
            elif last_vol > (last_vol_ma * 2.5) and last_delta > 0:
                future_pred = "⚠️ FAKE_PUMP_MAT_SHORT"
                c_score += 2
            else:
                future_pred = "⏳ RUKO_SET_HONE_DO"
        else:
            if total_candles <= 25:
                if last_vol > (last_vol_ma * 3.0):
                    if last_price >= prices[-2]: 
                        future_pred = "VOLUME_PUMP_AAYA"
                        c_score += 2
                    else: 
                        future_pred = "VOLUME_DUMP_AAYA"
                        c_score -= 2
                elif highs[-1] > last_price and (highs[-1] - last_price) > (last_price - lows[-1]) * 2:
                    future_pred = "WHALE_MAL_BECH_RHI"
                    c_score -= 1
            else:
                if len(p_p) >= 2 and len(p_t) >= 2:
                    recent_high = p_p[-1][1]
                    recent_low = p_t[-1][1]
                    
                    if last_price > recent_high and bid_pct <= 35:
                        future_pred = "TRAP_HA_SHORT_KRO"
                        c_score -= 2
                    elif last_price < recent_low and bid_pct >= 65:
                        future_pred = "TRAP_HA_LONG_KRO"
                        c_score += 2
                    elif last_price > recent_high and "MANDI" in structure_trend: 
                        future_pred = "TREND_PALTA_UP"
                        c_score += 2
                    elif last_price < recent_low and "TEJI" in structure_trend: 
                        future_pred = "TREND_PALTA_DN"
                        c_score -= 2
                    elif lows[-1] < recent_low and last_price > recent_low: 
                        future_pred = "STOP_HUNT_PUMP"
                        c_score += 2
                    elif highs[-1] > recent_high and last_price < recent_high: 
                        future_pred = "STOP_HUNT_DUMP"
                        c_score -= 2

        if future_pred == "SCANNING":
            if last_price > (last_vwap * 1.05): future_pred = "BHUT_UPAR_AGYA"
            elif last_price < (last_vwap * 0.95): future_pred = "BHUT_NICHE_AGYA"

        anomaly_status = "Clear"
        if total_candles >= 15:
            if len(p_t) >= 2 and len(ind_vals) >= 2:
                if p_t[-1][1] < p_t[-2][1] and ind_vals[-1] > ind_vals[-2]: 
                    anomaly_status = "CHUPI_BUY_DIVERG"
                    c_score += 1
            if len(p_p) >= 2 and len(ind_vals) >= 2:
                if p_p[-1][1] > p_p[-2][1] and ind_vals[-1] < ind_vals[-2]: 
                    anomaly_status = "CHUPI_SELL_DIVERG"
                    c_score -= 1
                
        return structure_trend, squeeze_status, anomaly_status, future_pred, last_price, c_score
    except Exception as e:
        logging.error(f"Error inside predictive quant engine: {e}")
        return "Error", "Error", "Error", "Error", 0.0, 0

# Startup Signal Function
async def send_startup_message(application: Application):
    if USER_CHAT_ID:
        try:
            await asyncio.sleep(3)
            await application.bot.send_message(
                chat_id=USER_CHAT_ID,
                text="🚀 <b>HINGLISH TERMINAL SYSTEM ONLINE</b>\nAb saari reports ekdum aasan Hinglish words mein aayengi bhai. Use /track.",
                parse_mode="HTML"
            )
        except Exception as e: logging.error(f"Startup fail: {e}")

# Telegram Command Handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "⚡ <b>HINGLISH SAFE QUANT TERMINAL</b>\n\n"
        "Commands:\n"
        "/track COIN/USDT - Coin ko list mein add karein\n"
        "/stop COIN/USDT - Coin ko list se hatayein\n"
        "/status - Apni watchlist check karein", 
        parse_mode="HTML"
    )

async def track_coin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not context.args:
        await update.message.reply_text("❌ Coin ka naam dalo. Ex: /track SOL/USDT")
        return
    symbol = context.args[0].upper()
    if chat_id not in TRACKED_PAIRS: TRACKED_PAIRS[chat_id] = set()
    TRACKED_PAIRS[chat_id].add(symbol)
    await update.message.reply_text(f"✅ <b>{symbol}</b> load ho gaya hai bhai. Har 5 minute mein automatic report aayegi.", parse_mode="HTML")

async def stop_coin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not context.args: return
    symbol = context.args[0].upper()
    if chat_id in TRACKED_PAIRS and symbol in TRACKED_PAIRS[chat_id]:
        TRACKED_PAIRS[chat_id].remove(symbol)
        await update.message.reply_text(f"🛑 <b>{symbol}</b> ko list se hata diya hai.", parse_mode="HTML")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    pairs = TRACKED_PAIRS.get(chat_id, set())
    if not pairs: await update.message.reply_text("Watchlist abhi khali hai bhai.")
    else: 
        watchlist_str = "\n".join([f"• {p}" for p in pairs])
        await update.message.reply_text(f"📋 <b>Active Watchlist:</b>\n{watchlist_str}", parse_mode="HTML")

# Background Monitoring 5-Minute Execution Loop
async def monitoring_job(application: Application):
    while True:
        await asyncio.sleep(300) # Pure 5-Minute loop block
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
                        ohlcv_res = await fetch_ohlcv_permitted(symbol, tf, limit=150)
                        if ohlcv_res is None or ohlcv_res[0] is None: continue
                        ohlcv, exchange_obj = ohlcv_res
                        
                        bid_pct, order_flow_status = fetch_orderbook_advanced_metrics(exchange_obj, symbol)
                        
                        structure_trend, squeeze, anomaly, prediction, price, score = analyze_predictive_metrics(ohlcv, bid_pct, order_flow_status, symbol)
                        if structure_trend == "Error": continue

                        last_price = price
                        node_source = exchange_obj.name
                        total_confluence_score += score
                        timeframe_data[tf] = (squeeze, order_flow_status, anomaly, prediction, structure_trend)
                        has_data = True
                    except Exception as loop_err: logging.error(f"Processing loop err: {loop_err}")

                if has_data and timeframe_data:
                    if total_confluence_score >= 5: global_bias = "POORA_TEZ_BUY 🚀"
                    elif total_confluence_score >= 1: global_bias = "UP_RUKH 🟢"
                    elif total_confluence_score <= -5: global_bias = "POORA_MANDI_SHORT 💥"
                    elif total_confluence_score <= -1: global_bias = "DN_RUKH 🔴"
                    else: global_bias = "SIDEWAYS ⏳"
                    
                    is_quant_top = any("SHORT_THOKO_ABHI" in data[3] for data in timeframe_data.values())
                    is_void = any("KHATRA_GAP" in data[1] for data in timeframe_data.values())
                    is_fool_rush = any("FAKE_PUMP" in data[3] for data in timeframe_data.values())
                    
                    if is_quant_top: header = "🎯🎯 ALFA TRIGGER: SHORT THOKO ABHI"
                    elif is_void: header = "🎰 WARNING: ORDERBOOK KHALI HA (VOID)"
                    elif is_fool_rush: header = "⚠️ DOOR RAHO: FAKE UP RUSH ACTIVE"
                    else: header = "🛰️ LIVE QUANT MATRIX REPORT"
                    
                    # Mobile Clean Grid Formatter
                    msg = f"<b>{header}: {symbol}</b>\n"
                    msg += f"• Price: ${last_price:,.4f} ({node_source})\n"
                    msg += f"• Whales Intel: <code>{onchain_intel}</code>\n"
                    msg += f"• Net Score: <code>{total_confluence_score:+} ({global_bias})</code>\n"
                    msg += "====================================\n"
                    msg += "<code>TF  │TREND   │MOVE│ORDERBK  │FORECAST</code>\n"
                    msg += "------------------------------------\n"
                    
                    for tf in TIMEFRAMES:
                        if tf in timeframe_data:
                            squeeze, order_flow_status, anomaly, prediction, structure_trend = timeframe_data[tf]
                            
                            short_pred = prediction
                            if len(short_pred) > 18:
                                short_pred = short_pred[:17] + ".."
                                
                            msg += f"<code>{tf:<4}│{structure_trend:<8}│{squeeze:<4}│{order_flow_status:<9}│</code>{short_pred}\n"
                    
                    msg += "====================================\n"
                    msg += "💡 <i>Short Guide: Jab top par SHORT_THOKO_ABHI likha aaye aur Score heavy negative ho, tabhi entry banani hai bhai.</i>"
                    
                    try:
                        await application.bot.send_message(chat_id=chat_id, text=msg, parse_mode="HTML")
                    except Exception as send_err: logging.error(f"Telegram HTML compact matrix dispatch fail: {send_err}")

# Web Server for Render Keep-Alive
app = Flask(__name__)
@app.route('/')
def health_check(): return "Hinglish Quant Core Alive", 200

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
