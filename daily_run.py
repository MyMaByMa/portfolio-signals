import os, sys, io, math, time, json, datetime as dt, traceback, yaml, csv
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
  data = yf.download(
    tickers, period=period, interval=interval,
    auto_adjust=False, group_by="ticker", threads=True, progress=False
  )
  return data

def compute_indicators(df):  # df: MultiIndex columns (ticker, field)
  rows = []
  tickers = sorted(set(k for k, _ in df.columns))
  for tkr in tickers:
    try:
      sub = df[tkr].dropna().copy()
      if len(sub) < 60:
        continue
      sub["sma50"]  = SMAIndicator(sub["Close"], 50).sma_indicator()
      sub["sma200"] = SMAIndicator(sub["Close"], 200).sma_indicator()
      sub["rsi"]    = RSIIndicator(sub["Close"], 14).rsi()
      atr           = AverageTrueRange(high=sub["High"], low=sub["Low"], close=sub["Close"], window=14)
      sub["atr"]    = atr.average_true_range()
      sub["vol20"]  = sub["Volume"].rolling(20).mean()
      sub["vol_ratio"] = sub["Volume"] / (sub["vol20"] + 1e-9)
      sub["hi_52w"] = sub["High"].rolling(252, min_periods=60).max()
      sub["lo_52w"] = sub["Low"].rolling(252, min_periods=60).min()

      last = sub.iloc[-1]
      close  = float(last["Close"])
      sma50  = float(last["sma50"])
      sma200 = float(last["sma200"]) if not math.isnan(last["sma200"]) else np.nan
      rsi    = float(last["rsi"])
      atr14  = float(last["atr"])
      vol_r  = float(last["vol_ratio"])
      hi_52w = float(last["hi_52w"])
      lo_52w = float(last["lo_52w"])

      dist_hi = (hi_52w - close) / hi_52w if hi_52w and hi_52w > 0 else np.nan
      near_hi = bool(hi_52w and close >= 0.99 * hi_52w)
      mom_1m  = close / sub["Close"].iloc[-21] - 1 if len(sub) >= 21 else np.nan
      mom_3m  = close / sub["Close"].iloc[-63] - 1 if len(sub) >= 63 else np.nan

      rows.append(dict(
        ticker=tkr, close=close, sma50=sma50, sma200=sma200, rsi=rsi,
        atr=atr14, vol_ratio=vol_r, hi_52w=hi_52w, lo_52w=lo_52w,
        dist_52w_hi=dist_hi, dist_52w_lo=(close - lo_52w)/lo_52w if lo_52w else np.nan,
        near_52w_high=near_hi, mom_1m=mom_1m, mom_3m=mom_3m
      ))
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
    keys = [
        "shortName", "longName",                    # ← PŘIDÁNO
        "marketCap","enterpriseValue","trailingPE","forwardPE","enterpriseToEbitda",
        "returnOnEquity","returnOnCapital","freeCashflow","operatingCashflow",
        "totalRevenue","grossMargins","revenueGrowth","earningsGrowth","debtToEquity",
        "sharesOutstanding"
    ]
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
    ev_ebit = None
    shareholder_yield = None
    fcf_conv = (fcf / ocf) if fcf and ocf and ocf>0 else None
    eps_growth = safe(info.get("earningsGrowth"), None)
    rev_growth = safe(info.get("revenueGrowth"), None)
name = info.get("shortName") or info.get("longName") or ticker

return {
    "info": info,
    "name": name,                    # ← PŘIDÁNO
    "pfcf": pfcf, "fcf_margin": fcf_margin,
    "ev_ebitda": ev_ebitda, "ev_ebit": ev_ebit,
    "shareholder_yield": shareholder_yield,
    "fcf_conv": fcf_conv, "eps_growth": eps_growth, "rev_growth": rev_growth
}

def earnings_blackout(ticker, days=1):
  try:
    t = yf.Ticker(ticker)
    ed = t.get_earnings_dates(limit=1)
    if ed is not None and len(ed) > 0:
      edate = ed.index[0].to_pydatetime().date()
      today = dt.date.today()
      return abs((today - edate)).days <= days
  except Exception:
    pass
  return False

def guru_mix_scores(tech_row, fund_row, weights):
  info = fund_row["info"]
  buffett   = score_buffett(info, fund_row["fcf_margin"])
  graham    = score_graham(info, fund_row["pfcf"], fund_row["ev_ebitda"], None)
  greenblatt= score_greenblatt(info, fund_row["ev_ebit"], fund_row["ev_ebitda"])
  raschke   = score_raschke(
                tech_row["close"] > tech_row["sma50"],
                (tech_row["close"] > tech_row["sma200"]) if not math.isnan(tech_row["sma200"]) else False,
                tech_row["rsi"],
                tech_row["dist_52w_hi"])
  lynch     = score_lynch(info.get("trailingPE"), fund_row["eps_growth"])
  icahn     = score_icahn(fund_row["pfcf"], fund_row["ev_ebitda"], fund_row["shareholder_yield"], False)
  ackman    = score_ackman(info.get("returnOnEquity"), fund_row["fcf_conv"], fund_row["rev_growth"])

  comp = dict(buffett=buffett, graham=graham, greenblatt=greenblatt,
              raschke=raschke, lynch=lynch, icahn=icahn, ackman=ackman)
  total = 0.0
  for k, v in comp.items():
    total += v * (weights.get(k, 0) / 100.0)
  return clamp(total, 0, 100), comp

def load_template():
  tmpl_path = os.path.join("templates", "report.html")
  with open(tmpl_path, "r", encoding="utf-8") as f:
    return Template(f.read())

def _parse_num(s):
  if s is None:
    return 0.0
  if isinstance(s, (int, float)):
    return float(s)
  s = str(s).strip().replace(" ", "")
  s = s.replace(",", ".")
  try:
    return float(s)
  except Exception:
    return 0.0

def read_positions(path="positions.csv"):
  """
  Načte pozice z CSV. Toleruje:
  - oddělovače čárka/semicolon
  - sloupce: ticker | (shares|quantity|qty) | (cost_basis|avg_cost|avgprice|avg_price)
  - desetinné čárky
  Vrací list(dict(ticker, shares, cost_basis)).
  """
  if not os.path.exists(path):
    return []

  # detekce delimiteru
  with open(path, "r", encoding="utf-8") as f:
    sample = f.read(2048)
  try:
    dialect = csv.Sniffer().sniff(sample, delimiters=";, \t")
    sep = dialect.delimiter
  except Exception:
    sep = ","

  rows = []
  with open(path, "r", encoding="utf-8", newline="") as f:
    r = csv.DictReader(f, skipinitialspace=True, delimiter=sep)
    for row in r:
      if not row:
        continue
      t = (row.get("ticker") or row.get("Ticker") or "").strip()
      if not t:
        # fallback: když DictReader nechytil hlavičku
        vals = list(row.values())
        if vals:
          t = str(vals[0]).strip()
      if not t:
        continue
      sh = row.get("shares") or row.get("quantity") or row.get("qty")
      cb = row.get("cost_basis") or row.get("avg_cost") or row.get("avgprice") or row.get("avg_price")
      rows.append({"ticker": t, "shares": _parse_num(sh), "cost_basis": _parse_num(cb)})
  return rows

# --------- main ---------
def main():
  cfg = load_config()

  # --- universe
  universe = cfg.get("universe", {})
  tickers = sorted(set(universe.get("portfolio", []) + universe.get("watchlist", [])))
  if not tickers:
    print("No tickers in config.yaml")
    return

  # --- fetch prices & indicators
  hist = fetch_history(tickers)
  tech = compute_indicators(hist).set_index("ticker")

  # --- rows for report
  rows = []
  for tkr, row in tech.iterrows():
    fund = compute_fundamentals_row(tkr)
    name = fund.get("name") or tkr                    # ← jméno z fundamentů
    gm, comp = guru_mix_scores(row, fund, cfg.get("weights", {}))
    e_blk = earnings_blackout(tkr, cfg.get("watchlist_rules", {}).get("earnings_blackout_days",1))

    signal = classify_signal(dict(
        close=row["close"], sma50=row["sma50"], rsi=row["rsi"],
        dist_52w_hi=row["dist_52w_hi"], near_52w_high=row["near_52w_high"]
    ), earnings_window=e_blk)

    why = ...
    bucket = "portfolio" if tkr in universe.get("portfolio", []) else "watchlist"

    rows.append({
        "name": name,                                   # ← PŘIDÁNO (název firmy)
        "ticker": tkr,
        "bucket": bucket, "signal": signal, "guru_mix": gm,
        "buffett": comp["buffett"], "graham": comp["graham"], "greenblatt": comp["greenblatt"],
        "raschke": comp["raschke"], "lynch": comp["lynch"], "icahn": comp["icahn"], "ackman": comp["ackman"],
        "close": row["close"], "sma50": row["sma50"], "sma200": row["sma200"], "rsi": row["rsi"],
        "dist_hi_pct": row["dist_52w_hi"]*100.0, "near_52w_high": row["near_52w_high"],
        "mom_1m": row["mom_1m"] if row["mom_1m"] is not None else np.nan,
        "mom_3m": row["mom_3m"] if row["mom_3m"] is not None else np.nan,
        "why": why, "earnings_blackout": e_blk
    })

  out = pd.DataFrame(rows).sort_values(["bucket", "signal", "guru_mix"], ascending=[True, True, False])

  # --- uložit hlavní CSV
  os.makedirs("report", exist_ok=True)
  out.to_csv(os.path.join("report", "report.csv"), index=False)

  # -------------------------------
  #   TARGETS & TRADE PLAN
  # -------------------------------
  # gm→0–20 váha (jemné škálování pro alokace)
  if "guru_mix" in out.columns:
    out["gm20"] = np.clip(out["guru_mix"] / 5.0, 0, 20)
  else:
    out["gm20"] = 0.0

  # jen BUY tickery
  buy = out[out["signal"] == "BUY"].set_index("ticker").copy()

  # načti aktuální pozice
  pos_list = read_positions("positions.csv")
  cur_shares = {str(p["ticker"]).strip(): float(p["shares"]) for p in pos_list}

  if buy.empty:
    targets = pd.DataFrame(columns=["ticker", "target_w", "target_val", "target_shares"])
    plan    = pd.DataFrame(columns=["ticker", "action", "qty", "note"])
  else:
    # — bezpečné načtení z configu (risk vs Risk) —
    risk_cfg  = cfg.get("risk") or cfg.get("Risk") or {}
    alloc_cfg = cfg.get("allocation") or {}
    exec_cfg  = cfg.get("execution") or {}

    cap_val = risk_cfg.get("capital_total")
    if cap_val is None:
      raise RuntimeError("config.yaml: chybí risk.capital_total")

    capital = float(cap_val)
    max_w   = float(alloc_cfg.get("max_pos_weight", 0.12))
    min_w   = float(alloc_cfg.get("min_pos_weight", 0.01))
    lot     = int(exec_cfg.get("lot_size", 1))

    # surové váhy z gm20
    w_raw = buy["gm20"]
    w = w_raw / w_raw.sum()

    # omez min/max
    w = w.clip(lower=min_w, upper=max_w)

    # renormalizace aby suma <= 1.0
    s = w.sum()
    if s > 1.0:
      w = w / s

    # cílová hodnota a počty kusů
    tgt_val = (capital * w).rename("target_val")
    tgt_sh  = (tgt_val / buy["close"])
    if lot > 1:
      tgt_sh = (tgt_sh / lot).round() * lot
    else:
      tgt_sh = tgt_sh.round()

    targets = pd.DataFrame({
      "ticker": buy.index,
      "target_w": w.values.round(4),
      "target_val": tgt_val.values.round(2),
      "target_shares": tgt_sh.astype(int).values
    })

    # plán = rozdíl proti aktuálním pozicím
    orders = []
    for tkr, t_sh in tgt_sh.items():
      have = int(round(cur_shares.get(tkr, 0)))
      d = int(t_sh) - have
      if d > 0:
        orders.append((tkr, "BUY",  d, f"to reach {int(t_sh)}"))
      elif d < 0:
        orders.append((tkr, "SELL", -d, f"to reduce to {int(t_sh)}"))
    plan = pd.DataFrame(orders, columns=["ticker", "action", "qty", "note"])

  # --- uložit i CSV s plánem a cíli
  targets.to_csv(os.path.join("report", "targets.csv"), index=False)
  plan.to_csv(os.path.join("report", "trade_plan.csv"), index=False)

  # --- Render HTML
  tmpl = load_template()
  html = tmpl.render(
    updated=dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    rows=out.to_dict("records"),
    plan=plan.to_dict("records"),
    targets=targets.to_dict("records")
  )
  with open(os.path.join("report", "report.html"), "w", encoding="utf-8") as f:
    f.write(html)

  print("Done. Wrote report/report.csv, report/targets.csv, report/trade_plan.csv and report/report.html")

if __name__ == "__main__":
  main()
