# 选定币种，只跑这5个
SYMBOLS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "LTC/USDT", "DOGE/USDT"]

# ================== 主循环 ==================
def main():
    ex = build_exchange()
    ex.load_markets()
    mode_txt = "实盘" if LIVE_TRADE==1 else "纸面"
    log(f"启动Bot {EXCHANGE_NAME}/{MARKET_TYPE} 模式={mode_txt}")
    tg_send(f"🤖 Bot启动 {EXCHANGE_NAME}/{MARKET_TYPE} 模式={mode_txt}")

    if MARKET_TYPE == "future":
        for sym in SYMBOLS:
            try:
                set_symbol_leverage(ex, sym)
            except Exception:
                pass

    last_hourly_push = 0

    while True:
        loop_start = time.time()
        try:
            report_lines = []  # 一小时的汇总
            for symbol in SYMBOLS:
                sides = []
                detail_map = {}

                for tf in TIMEFRAMES:
                    try:
                        df = fetch_df(ex, symbol, tf, limit=300)
                        side, det = indicators_and_side(df)
                        detail_map[tf] = (side, det, df)
                        sides.append(side)
                        log(summarize(tf, side, det))
                    except Exception as e_tf:
                        log(f"❌ {symbol} {tf} 指标失败: {e_tf}")
                        detail_map[tf] = (None, None, None)
                        sides.append(None)

                bull = sum(1 for s in sides if s=="多")
                bear = sum(1 for s in sides if s=="空")
                final_side = None
                if bull >= REQUIRED_CONFIRMS and bull > bear:
                    final_side = "多"
                elif bear >= REQUIRED_CONFIRMS and bear > bull:
                    final_side = "空"

                # 如果有信号，准备止盈止损数据
                entry, sl, tp = "-", "-", "-"
                if final_side:
                    s1h, d1h, df1h = detail_map["1h"]
                    if d1h and df1h is not None:
                        entry = d1h["entry"]
                        atr = compute_atr(df1h, period=14)
                        if final_side == "多":
                            sl = entry - SL_ATR_MULT * atr
                            tp = entry + TP_ATR_MULT * atr
                        else:
                            sl = entry + SL_ATR_MULT * atr
                            tp = entry - TP_ATR_MULT * atr

                        # 真正下单
                        place_order_and_brackets(ex, symbol, final_side, entry, df1h)

                report_lines.append(
                    f"{symbol} → {final_side or '无信号'} "
                    f"入:{format_price(entry)} SL:{format_price(sl)} TP:{format_price(tp)}"
                )

            # 整点推送一次 TG
            now_ts = int(time.time())
            if now_ts - last_hourly_push >= 3600:
                tg_send("📊 每小时交易报告\n" + "\n".join(report_lines))
                last_hourly_push = now_ts

        except Exception as e:
            log(f"[主循环异常] {e}\n{traceback.format_exc()}")

        used = time.time() - loop_start
        time.sleep(max(1, POLL_INTERVAL - int(used)))
