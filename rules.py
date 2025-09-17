import math

def safe(val, default=None):
    try:
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return default
        return val
    except Exception:
        return default

def clamp(x, lo=0, hi=100):
    return max(lo, min(hi, x))

def score_buffett(info, fcf_margin):
    # Kvalita: ROE, FCF margin, leverage
    s = 0
    # ROE proxy
    roe = safe(info.get("returnOnEquity"), 0)  # fraction
    if roe is not None:
        s += clamp((roe or 0)*100/2, 0, 15)  # 15 b max around 30% ROE
    # FCF margin (0-10 b)
    if fcf_margin is not None:
        s += clamp(fcf_margin*2, 0, 10)  # 20% FCF margin ~ 10b
    # Leverage (0-5 b) - better if low
    nde = safe(info.get("debtToEquity"), None)
    if nde is not None:
        if nde < 30: s += 5
        elif nde < 60: s += 3
        elif nde < 120: s += 1
    return clamp(s, 0, 35)

def score_graham(info, pfcf, ev_ebitda, valuation_discount):
    s = 0
    # Value: low multiples
    pe = safe(info.get("trailingPE"), None)
    fwdpe = safe(info.get("forwardPE"), None)
    for mult, w in ((pe,6),(fwdpe,6)):
        if mult is None: continue
        if mult <= 10: s += w
        elif mult <= 15: s += w*0.7
        elif mult <= 20: s += w*0.4
        elif mult <= 25: s += w*0.2
    if pfcf is not None:
        if pfcf <= 12: s += 6
        elif pfcf <= 20: s += 3
    if ev_ebitda is not None:
        if ev_ebitda <= 8: s += 7
        elif ev_ebitda <= 12: s += 3
    if valuation_discount is not None:
        # discount vs 5Y median multiple (%)
        if valuation_discount >= 20: s += 6
        elif valuation_discount >= 10: s += 3
    return clamp(s, 0, 25)

def score_greenblatt(info, ev_ebit, ev_ebitda):
    s = 0
    roic = safe(info.get("returnOnCapital"), None) or safe(info.get("returnOnEquity"), 0)
    if roic is not None:
        if roic >= 0.20: s += 10
        elif roic >= 0.12: s += 7
        elif roic >= 0.08: s += 4
    if ev_ebit is not None:
        if ev_ebit <= 10: s += 10
        elif ev_ebit <= 14: s += 7
        elif ev_ebit <= 18: s += 4
    elif ev_ebitda is not None:
        if ev_ebitda <= 8: s += 8
        elif ev_ebitda <= 12: s += 5
        elif ev_ebitda <= 16: s += 3
    return clamp(s, 0, 20)

def score_raschke(close_gt_sma50, close_gt_sma200, rsi, dist_hi):
    s = 0
    if close_gt_sma50: s += 5
    if close_gt_sma200: s += 5
    if rsi is not None:
        if 45 <= rsi <= 65: s += 5
        elif rsi > 70: s -= 5
    if dist_hi is not None and 0.05 <= dist_hi <= 0.15:
        s += 5
    return clamp(s, 0, 20)

def score_lynch(trailing_pe, eps_growth):
    # PEG heuristic: trailing PE / (EPS growth * 100)
    if eps_growth is None or eps_growth <= 0 or trailing_pe is None or trailing_pe <= 0:
        return 0
    peg = trailing_pe / (eps_growth*100.0)
    if peg <= 1.0: return 15
    if peg <= 1.5: return 10
    if peg <= 2.0: return 6
    return 0

def score_icahn(pfcf, ev_ebitda, shareholder_yield, catalyst_flag):
    s = 0
    if pfcf is not None and pfcf <= 12: s += 5
    if ev_ebitda is not None and ev_ebitda <= 6: s += 5
    if shareholder_yield is not None:
        if shareholder_yield >= 0.08: s += 5
        elif shareholder_yield >= 0.06: s += 3
    if catalyst_flag: s += 5
    return clamp(s, 0, 20)

def score_ackman(roe, fcf_conv, rev_growth):
    s = 0
    if roe is not None:
        if roe >= 0.20: s += 8
        elif roe >= 0.12: s += 5
    if fcf_conv is not None:
        if fcf_conv >= 0.9: s += 7
        elif fcf_conv >= 0.8: s += 5
    if rev_growth is not None:
        if rev_growth >= 0.12: s += 5
        elif rev_growth >= 0.08: s += 3
    return clamp(s, 0, 20)

def classify_signal(row, earnings_window=False):
    buy = (row["close"] > row["sma50"]) and (40 <= row["rsi"] <= 65) and (row["dist_52w_hi"] >= 0.05) and (not earnings_window)
    trim = (row["rsi"] > 70) and row["near_52w_high"]
    if buy: return "BUY"
    if trim: return "TRIM"
    return "HOLD"
