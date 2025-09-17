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
    os.makedirs("report", exist_ok=True)
    csv_path = os.path.join("report", "report.csv")
    out.to_csv(csv_path, index=False)

    # Render HTML
    tmpl = load_template()
    html = tmpl.render(updated=dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
                       rows=out.to_dict("records"))
    with open(os.path.join("report", "report.html"), "w", encoding="utf-8") as f:
        f.write(html)

    print("Done. Wrote report/report.csv and report/report.html")

if __name__ == "__main__":
    main()
