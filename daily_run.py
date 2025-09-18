import os, sys, io, math, time, json, datetime as dt, traceback, yaml
import pandas as pd, numpy as np
import yfinance as yf
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from jinja2 import Template
from rules import (score_buffett, score_graham, score_greenblatt, score_raschke,
                   score_lynch, score_icahn, score_ackman, classify_signal, safe, clamp)

# --------- utils ---------
def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def fetch_history(tickers, period="400d", interval="1d"):
    data = yf.download(tickers, period=period, interval=interval, auto_adjust=False, group_by="ticker", threads=True, progress=False)
    return data

def read_positions(path="positions.csv"):
    """
    Načte reálné pozice z CSV.
    Podporuje oddělovače , ; \t, ignoruje mezery za oddělovačem
    a převádí desetinné čárky na tečky.
    Vrací DataFrame: ticker(str), shares(float), cost_basis(float).
    """
    import csv, os
    if not os.path.exists(path):
        return pd.DataFrame(columns=["ticker", "shares", "cost_basis"])

    # rozpoznání oddělovače
    with open(path, "r", encoding="utf-8") as f:
        sample = f.read(2048)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        sep = dialect.delimiter
    except Exception:
        sep = ","

    # načtení + ignorování mezer za oddělovačem
    df = pd.read_csv(path, sep=sep, skipinitialspace=True)

    # tolerantní mapování názvů sloupců
    cols = {c.strip().lower(): c for c in df.columns}
    def col(*names, default=None):
        for n in names:
            if n in cols:
                return cols[n]
        return default

    col_ticker = col("ticker", default=(df.columns[0] if len(df.columns) >= 1 else None))
    col_shares = col("shares", "quantity", "qty", default=(df.columns[1] if len(df.columns) >= 2 else None))
    col_cost   = col("cost_basis", "avg_cost", "avgprice", "avg_price",
                     default=(df.columns[2] if len(df.columns) >= 3 else None))

    out = pd.DataFrame({
        "ticker": df[col_ticker].astype(str).str.strip() if col_ticker else "",
        "shares": df[col_shares] if col_shares else 0,
        "cost_basis": df[col_cost] if col_cost else 0,
    })

    # převod „českých“ čárek na tečky + odstranění mezer
    out["shares"] = (out["shares"].astype(str)
                                  .str.replace(" ", "", regex=False)
                                  .str.replace(",", ".", regex=False))
    out["cost_basis"] = (out["cost_basis"].astype(str)
                                        .str.replace(" ", "", regex=False)
                                        .str.replace(",", ".", regex=False))

    out["shares"] = pd.to_numeric(out["shares"], errors="coerce").fillna(0.0)
    out["cost_basis"] = pd.to_numeric(out["cost_basis"], errors="coerce").fillna(0.0)
    out["ticker"] = out["ticker"].str.strip()
    out = out[out["ticker"] != ""]
    return out

def compute_indicators(df):  # df: MultiIndex columns (ticker, field)
    rows = []
    for tkr in sorted(set(k for k,_ in df.columns)):
        try:
            sub = df[tkr].dropna().copy()
            if len(sub) < 60:
                continue
            sub["sma50"] = SMAIndicator(sub["Close"], 50).sma_indicator()
            sub["sma200"] = SMAIndicator(sub["Close"], 200).sma_indicator()
            sub["rsi"] = RSIIndicator(sub["Close"], 14).rsi()
            atr = AverageTrueRange(high=sub["High"], low=sub["Low"], close=sub["Close"], window=14)
            sub["atr"] = atr.average_true_range()
            sub["vol20"] = sub["Volume"].rolling(20).mean()
            sub["vol_ratio"] = sub["Volume"] / (sub["vol20"] + 1e-9)
            sub["hi_52w"] = sub["High"].rolling(252, min_periods=60).max()
            sub["lo_52w"] = sub["Low"].rolling(252, min_periods=60).min()
            last = sub.iloc[-1]
            close = float(last["Close"])
            sma50 = float(last["sma50"])
            sma200 = float(last["sma200"]) if not math.isnan(last["sma200"]) else None
            rsi = float(last["rsi"])
            atr_last = float(last["atr"])
            vol_ratio = float(last["vol_ratio"])
            hi_52w = float(last["hi_52w"])
            lo_52w = float(last["lo_52w"])
            dist_hi = (hi_52w - close) / hi_52w if hi_52w and hi_52w>0 else None
            near_hi = (hi_52w and close >= 0.99*hi_52w)
            mom_1m = close / sub["Close"].iloc[-21] - 1 if len(sub)>=21 else None
            mom_3m = close / sub["Close"].iloc[-63] - 1 if len(sub)>=63 else None
            rows.append(dict(ticker=tkr, close=close, sma50=sma50, sma200=sma200 or np.nan, rsi=rsi,
                             atr=atr_last, vol_ratio=vol_ratio, hi_52w=hi_52w, lo_52w=lo_52w,
                             dist_52w_hi=dist_hi or 0, dist_52w_lo=(close - lo_52w)/lo_52w if lo_52w else None,
                             near_52w_high=bool(near_hi), mom_1m=mom_1m, mom_3m=mom_3m))
        except Exception as e:
            print("indicator fail:", tkr, e)
    return pd.DataFrame(rows)

def fetch_info(ticker):
    t = yf.Ticker(ticker)
    info = {}
    try:
        info = t.get_info() or {}
    except Exception:
        try:
            info = t.fast_info or {}
        except Exception:
            info = {}
    # Basics we care about
    keys = ["marketCap","enterpriseValue","trailingPE","forwardPE","enterpriseToEbitda",
            "returnOnEquity","returnOnCapital","freeCashflow","operatingCashflow",
            "totalRevenue","grossMargins","revenueGrowth","earningsGrowth","debtToEquity",
            "sharesOutstanding"]
    out = {k: info.get(k) for k in keys}
    return out

def compute_fundamentals_row(ticker):
    info = fetch_info(ticker)
    mc = safe(info.get("marketCap"), None)
    fcf = safe(info.get("freeCashflow"), None)
    ocf = safe(info.get("operatingCashflow"), None)
    totrev = safe(info.get("totalRevenue"), None)
    ev = safe(info.get("enterpriseValue"), None)
    ebitda_mult = safe(info.get("enterpriseToEbitda"), None)
    # proxies
    pfcf = (mc / fcf) if mc and fcf and fcf>0 else None
    fcf_margin = (fcf / totrev) if fcf and totrev and totrev>0 else None
    ev_ebitda = ebitda_mult if ebitda_mult and ebitda_mult>0 else None
    ev_ebit = None  # not available free; keep None
    # shareholder yield = dividend + net buyback approx (we approximate buyback from change in shares; skip if missing)
    shareholder_yield = None
    # FCF conversion
    fcf_conv = (fcf / ocf) if fcf and ocf and ocf>0 else None
    # growth proxies
    eps_growth = safe(info.get("earningsGrowth"), None)   # YoY fraction
    rev_growth = safe(info.get("revenueGrowth"), None)    # YoY fraction
    # valuation discount vs "median" (unknown): leave None
    valuation_discount = None
    return {
        "info": info, "pfcf": pfcf, "fcf_margin": fcf_margin,
        "ev_ebitda": ev_ebitda, "ev_ebit": ev_ebit,
        "shareholder_yield": shareholder_yield,
        "fcf_conv": fcf_conv, "eps_growth": eps_growth, "rev_growth": rev_growth
    }

def earnings_blackout(ticker, days=1):
    try:
        t = yf.Ticker(ticker)
        # get upcoming/last earnings dates; here we treat blackout if today is within +/- days of last date
        ed = t.get_earnings_dates(limit=1)
        if ed is not None and len(ed)>0:
            edate = ed.index[0].to_pydatetime().date()
            today = dt.date.today()
            return abs((today - edate)).days <= days
    except Exception:
        pass
    return False

def guru_mix_scores(tech_row, fund_row, weights):
    info = fund_row["info"]
    # Individual scores
    buffett = score_buffett(info, fund_row["fcf_margin"] if fund_row else None)
    graham = score_graham(info, fund_row["pfcf"], fund_row["ev_ebitda"], None)
    greenblatt = score_greenblatt(info, fund_row["ev_ebit"], fund_row["ev_ebitda"])
    raschke = score_raschke(tech_row["close"]>tech_row["sma50"],
                            (tech_row["close"]>tech_row["sma200"]) if not math.isnan(tech_row["sma200"]) else False,
                            tech_row["rsi"], tech_row["dist_52w_hi"])
    lynch = score_lynch(info.get("trailingPE"), fund_row["eps_growth"])
    icahn = score_icahn(fund_row["pfcf"], fund_row["ev_ebitda"], fund_row["shareholder_yield"], False)
    ackman = score_ackman(info.get("returnOnEquity"), fund_row["fcf_conv"], fund_row["rev_growth"])

    comp = dict(buffett=buffett, graham=graham, greenblatt=greenblatt,
                raschke=raschke, lynch=lynch, icahn=icahn, ackman=ackman)
    total = 0.0
    for k,v in comp.items():
        total += v * (weights.get(k,0)/100.0)
    return clamp(total, 0, 100), comp
def portfolio_value(positions, prices):
    pv = 0.0
    for p in positions:
        price = prices.get(p["ticker"])
        if price:
            pv += float(p["shares"]) * float(price)
    return pv

def compute_targets_and_plan(cfg, positions, metrics_df, prices):
    """
    metrics_df: DataFrame s alespoň ['ticker','guru_mix','signal']
    positions : list(dict) z positions.csv  (ticker, shares, cost_basis)
    prices    : dict {ticker: last_close}
    """
    # aktivní tickery + jejich skóre
    active = metrics_df.set_index("ticker")

    risk  = cfg.get("risk", {})
    alloc = cfg.get("allocation", {})
    max_pos   = float(risk.get("max_pos", 0.08))     # max váha 8 %/pozice
    min_trade = float(alloc.get("min_trade", 100.0)) # min. notional na obchod (CZK/EUR/USD)

    # cílové váhy z guru_mix (ořez na max_pos a renormalizace)
    gm = active["guru_mix"].clip(lower=0.0)
    tot = gm.sum()
    if tot <= 0:
        target_w = pd.Series(0.0, index=active.index)
    else:
        target_w = (gm / tot).clip(upper=max_pos)
        target_w = target_w / target_w.sum()

    targets = pd.DataFrame({"ticker": active.index, "target_w": target_w.values})

    # současné váhy portfolia
    port_val = portfolio_value(positions, prices)
    cur = pd.DataFrame([{
        "ticker": p["ticker"],
        "shares": float(p["shares"]),
        "cur_w": (float(p["shares"]) * float(prices.get(p["ticker"], 0.0))) / port_val if port_val > 0 else 0.0
    } for p in positions]).set_index("ticker")

    merged = targets.set_index("ticker").join(cur, how="left").fillna(0.0)
    merged["price"] = [prices.get(t, np.nan) for t in merged.index]
    merged["diff_w"] = merged["target_w"] - merged["cur_w"]
    merged["trade_shares"] = (merged["diff_w"] * port_val / merged["price"]).round(3)
    merged["notional"] = (merged["trade_shares"] * merged["price"]).abs()
    merged["action"] = np.where(merged["trade_shares"] > 0, "BUY",
                         np.where(merged["trade_shares"] < 0, "TRIM", "HOLD"))
    # odfiltrovat drobky
    plan = merged[merged["notional"] >= min_trade].reset_index()

    return targets.reset_index(), plan


def load_template():
    tmpl_path = os.path.join("templates", "report.html")
    with open(tmpl_path, "r", encoding="utf-8") as f:
        return Template(f.read())

def main():
    cfg = load_config()
  
    universe = cfg["universe"]
    tickers = sorted(set(universe.get("portfolio", []) + universe.get("watchlist", [])))
    if not tickers:
        print("No tickers in config.yaml")
        return
    # Fetch prices
    hist = fetch_history(tickers)
    tech = compute_indicators(hist)
    tech = tech.set_index("ticker")

    # Merge fundamentals & scores
    rows = []
    for tkr, row in tech.iterrows():
        fund = compute_fundamentals_row(tkr)
        gm, comp = guru_mix_scores(row, fund, cfg.get("weights", {}))
        e_blk = earnings_blackout(tkr, cfg.get("watchlist_rules", {}).get("earnings_blackout_days",1))
        signal = classify_signal(dict(
            close=row["close"], sma50=row["sma50"], rsi=row["rsi"],
            dist_52w_hi=row["dist_52w_hi"], near_52w_high=row["near_52w_high"]
        ), earnings_window=e_blk)
        why = f"Close>{'SMA50' if row['close']>row['sma50'] else 'SMA50?'}; RSI={row['rsi']:.1f}; {row['dist_52w_hi']*100:.1f}% pod 52W high"
        bucket = "portfolio" if tkr in universe.get("portfolio", []) else "watchlist"
        rows.append({
            "ticker": tkr, "bucket": bucket, "signal": signal, "guru_mix": gm,
            "buffett": comp["buffett"], "graham": comp["graham"], "greenblatt": comp["greenblatt"],
            "raschke": comp["raschke"], "lynch": comp["lynch"], "icahn": comp["icahn"], "ackman": comp["ackman"],
            "close": row["close"], "sma50": row["sma50"], "sma200": row["sma200"], "rsi": row["rsi"],
            "dist_hi_pct": row["dist_52w_hi"]*100.0, "near_52w_high": row["near_52w_high"],
            "mom_1m": row["mom_1m"] if row["mom_1m"] is not None else np.nan,
            "mom_3m": row["mom_3m"] if row["mom_3m"] is not None else np.nan,
            "why": why, "earnings_blackout": e_blk
        })

    out = pd.DataFrame(rows).sort_values(["bucket", "signal", "guru_mix"], ascending=[True, True, False])
        # uložím výstupy
    os.makedirs("report", exist_ok=True)
    out.to_csv(os.path.join("report", "report.csv"), index=False)
    targets.to_csv(os.path.join("report", "targets.csv"), index=False)
    plan.to_csv(os.path.join("report", "trade_plan.csv"), index=False)

    # Render HTML (tohle musí být stále uvnitř def main():, tedy s jedním odsazením)
    tmpl = load_template()
    html = tmpl.render(
        updated=dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        rows=out.to_dict("records"),
        plan=plan.to_dict("records"),
        targets=targets.to_dict("records"),
    )
    with open(os.path.join("report", "report.html"), "w", encoding="utf-8") as f:
        f.write(html)

    print("Done. Wrote report/report.csv and report/report.html")

if __name__ == "__main__":
    main()

