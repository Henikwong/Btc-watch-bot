# ================== 主循环（改良版） ==================
kline_cache = {}
last_send = datetime.utcnow() - timedelta(hours=1)
prev_signals = {}

while True:
    now = datetime.utcnow()
    try:
        coins = main_coins + meme_coins
        kline_cache = {c: {"60min": {}, "4hour": {}, "1day": {}} for c in coins}

        # 抓取 K 线数据
        for coin in coins:
            # Huobi
            for period in ["60min", "4hour", "1day"]:
                kline_cache[coin][period]["huobi"] = get_kline_huobi(coin, period)
            # Binance
            for period, interval in zip(["60min", "4hour", "1day"], ["1h","4h","1d"]):
                kline_cache[coin][period]["binance"] = get_kline_binance(coin, interval)
            # Bybit
            for period, interval in zip(["60min", "4hour", "1day"], ["60","240","D"]):
                kline_cache[coin][period]["bybit"] = get_kline_bybit(coin, interval)

        # 每小时发送一次信号
        if (now - last_send).total_seconds() >= 3600:
            messages = []
            for coin in coins:
                period_signals, period_entries = {}, {}

                for period in main_periods:
                    dfs = kline_cache[coin].get(period, {})
                    signals, entries = [], []

                    for ex, df in dfs.items():
                        if df is not None and len(df) > 35:
                            sig, entry = calc_signal(df)
                            if sig:
                                signals.append(sig)
                                entries.append(entry)

                    if signals:
                        final_sig = max(set(signals), key=signals.count)
                        period_signals[period] = final_sig
                        period_entries[period] = sum(entries)/len(entries)

                if period_signals:
                    sig_values = list(period_signals.values())
                    unique_count = len(set(sig_values))
                    color = "🟢 绿色"
                    if unique_count == 1 and len(sig_values) == 3:
                        color = "🔴 红色"
                    elif len(sig_values) >= 2:
                        color = "🟡 黄色"

                    msg_lines = [f"📊 {coin.upper()} 信号 ({color})"]
                    for p in main_periods:
                        if p in period_signals:
                            entry = period_entries[p]
                            stop_loss = None
                            dfs_ref = kline_cache[coin][p].get("huobi") or next((df for df in kline_cache[coin][p].values() if df is not None), None)
                            if dfs_ref is not None:
                                stop_loss = calc_stop_loss(dfs_ref, period_signals[p], entry)
                            target = entry*(1.01 if "多" in period_signals[p] else 0.99)
                            line = f"{p} → {period_signals[p]} | 入场:{format_price(entry)} 目标:{format_price(target)} 止损:{format_price(stop_loss)}"
                            prev_sig = prev_signals.get(coin, {}).get(p)
                            if prev_sig and prev_sig != period_signals[p]:
                                line += " ⚡ 信号变化"
                                if dfs_ref is not None:
                                    analysis = gpt_analysis(coin, dfs_ref, period_signals[p])
                                    send_telegram_message(f"🧠 突发 GPT 分析\n{analysis[:3000]}")
                            msg_lines.append(line)

                    if unique_count == 1 and len(sig_values) == 3:
                        msg_lines.append("🌟 强信号！三周期一致")

                    messages.append("\n".join(msg_lines))
                    prev_signals[coin] = period_signals

                    # GPT 综合分析
                    try:
                        dfs_ref = kline_cache[coin]["60min"].get("huobi") or next((df for df in kline_cache[coin]["60min"].values() if df is not None), None)
                        if dfs_ref is not None:
                            analysis = gpt_analysis(coin, dfs_ref, period_signals)
                            send_telegram_message(f"🧠 GPT 综合分析\n{analysis[:3000]}")
                    except Exception as e:
                        print(f"[GPT ERROR] {e}")

            if messages:
                send_telegram_message("\n\n".join(messages))
            last_send = now

    except Exception as e:
        print(f"循环错误: {e}")

    time.sleep(900)
