# ====== 主循环核心部分（修正版） ======
while True:
    try:
        coins = main_coins + meme_coins
        now = datetime.now()
        hourly_report_lines = ["📢 每小时普通信息（含 1h /4h /1d /1w 指标&一致性）"]
        strong_alerts = []

        for coin in coins:
            coin_upper = coin.upper()
            per_period_results = {}

            # 抓取各周期各交易所 K 线并计算指标
            for period_label in ["60min", "4hour", "1day", "1week"]:
                huobi_df = get_kline_huobi(coin, period=period_label)
                binance_df = get_kline_binance(coin, interval=period_map[period_label]["binance"])
                okx_df = get_kline_okx(coin, bar=period_map[period_label]["okx"])

                h_ind = calc_indicators(huobi_df)
                b_ind = calc_indicators(binance_df)
                o_ind = calc_indicators(okx_df)

                per_period_results[period_label] = {
                    "huobi_df": huobi_df,
                    "binance_df": binance_df,
                    "okx_df": okx_df,
                    "huobi": h_ind,
                    "binance": b_ind,
                    "okx": o_ind
                }

            # ---- 高度动向一致性判断 ----
            consistent_counts = 0
            per_period_consistent = {}

            for p in ["60min", "4hour", "1day"]:
                # ✅ 这里修复了 '[' 未闭合问题
                inds = [
                    per_period_results[p].get("huobi"),
                    per_period_results[p].get("binance"),
                    per_period_results[p].get("okx")
                ]

                ok, reason = indicators_agree(inds)
                per_period_consistent[p] = {"ok": ok, "reason": reason, "inds": inds}

                if ok:
                    consistent_counts += 1

                log(f"{coin_upper} {p} 指标一致性: {ok} ({reason})")

            # 根据一致性生成 Telegram 消息
            if consistent_counts >= 1:
                chosen_period = next((p for p, v in per_period_consistent.items() if v["ok"]), None)
                if chosen_period:
                    ind_ref = per_period_results[chosen_period]["huobi"]
                    if ind_ref:
                        side = ind_ref["ema_trend"]
                        entry = ind_ref["entry"]
                        stop, target = compute_stop_target_from_df(per_period_results[chosen_period]["huobi_df"], side, entry)
                        block = build_consistency_block(coin_upper, side, entry, target, stop, consistent_counts)
                        strong_alerts.append(block)

        # 每小时发送一次普通信息
        if now.minute == 0 and last_hour_msg != now.hour:
            for msg in strong_alerts:
                send_telegram_message(msg)
            last_hour_msg = now.hour

        time.sleep(POLL_INTERVAL)
    except Exception as e:
        log(f"主循环异常: {e}")
        time.sleep(10)
