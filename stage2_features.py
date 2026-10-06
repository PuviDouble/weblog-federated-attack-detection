"""
=============================================================
STAGE 2 — FEATURE EXTRACTION
=============================================================
Extracts 41 features from each HTTP request.
Works for all 3 dataset formats.
=============================================================
"""

import re
import math
import joblib
import numpy as np
import pandas as pd
from urllib.parse import unquote, urlparse, parse_qs
from sklearn.preprocessing import StandardScaler

# ── Attack pattern regexes ────────────────────────────────────

SQLI_RE    = re.compile(
    r"(union.*select|select.*from|drop\s+table|insert\s+into|"
    r"delete\s+from|or\s+1\s*=\s*1|' or '|--|;--|/\*|\*/|"
    r"0x[0-9a-f]+|benchmark\s*\(|sleep\s*\(|"
    r"information_schema|%27|%3b|%2b|waitfor\s+delay|"
    r"extractvalue|updatexml)", re.I)

XSS_RE     = re.compile(
    r"(<script|</script|javascript:|onerror\s*=|onload\s*=|"
    r"onclick\s*=|alert\s*\(|document\.cookie|<iframe|<svg|"
    r"eval\s*\(|%3cscript|%3csvg|&#|onmouseover)", re.I)

PATH_RE    = re.compile(
    r"(\.\./|\.\.\\|%2e%2e|%252e|/etc/passwd|/etc/shadow|"
    r"windows[\\/]system32|%5c\.\.|/proc/self|"
    r"\.\.%2f|\.\.%5c)", re.I)

CMD_RE     = re.compile(
    r"(%3b|%7c|%26|%60|\||;|`|\$\(|&&|\|\||"
    r"cat\s+/|ls\s+-|wget\s+|curl\s+http|"
    r"/bin/sh|/bin/bash|cmd\.exe|/usr/bin)", re.I)

RFI_RE     = re.compile(
    r"(https?://[^/]+\.[a-z]{2,}/.*\.(php|asp|jsp|py|sh)|"
    r"ftp://|php://|data://|phar://)", re.I)

SSRF_RE    = re.compile(
    r"(169\.254\.169\.254|localhost|"
    r"127\.\d+\.\d+\.\d+|0\.0\.0\.0|::1|"
    r"file://|dict://|gopher://|"
    r"metadata\.google\.internal|"
    r"100\.100\.100\.200)", re.I)

SCANNER_RE = re.compile(
    r"(sqlmap|nikto|nmap|masscan|nessus|dirbuster|"
    r"gobuster|wfuzz|hydra|metasploit|acunetix|"
    r"burpsuite|python-requests|curl/[0-9]|"
    r"wget/[0-9]|libwww|havij|w3af)", re.I)

# ── Feature lists ─────────────────────────────────────────────

CORE_FEATURES = [
    'url_length','url_entropy','url_special_chars',
    'url_param_count','url_max_param_len','url_dir_depth',
    'url_is_encoded','url_double_encoded',
    'sqli_flag','xss_flag','traversal_flag','cmdinj_flag',
    'rfi_flag','ssrf_flag','scanner_ua','attack_flag_sum',
    'method_encoded','is_post','is_suspicious_method',
    'status_2xx','status_3xx','status_4xx','status_5xx',
    'status_403','status_500',
    'content_length_feat','content_entropy',
    'content_sqli','content_xss',
    'bytes_feat','latency_feat',
    'ua_length','ua_is_empty','ua_is_bot',
    'has_referrer','ext_referrer',
]

WINDOW_FEATURES = [
    'req_rate','error_rate','repeat_url_ratio',
    'unique_endpoints','sqli_rate_window',
]

ALL_FEATURES = CORE_FEATURES + WINDOW_FEATURES


# ── Helper functions ──────────────────────────────────────────

def _entropy(s):
    s = str(s)
    if len(s) < 2:
        return 0.0
    freq = [s.count(c)/len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in freq if p > 0)

def _max_param_len(url):
    try:
        params = parse_qs(urlparse(str(url)).query)
        if not params:
            return 0
        return max(len(str(v[0])) for v in params.values())
    except:
        return 0

def _safe(df, col, default):
    """Safely get column as Series with default."""
    if col in df.columns:
        return df[col].fillna(default)
    return pd.Series([default] * len(df), index=df.index)


# ── Main feature extraction ───────────────────────────────────

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    print(f"[Stage 2] Extracting features from {len(df)} rows...")
    df = df.copy()

    # Ensure all required columns
    df['endpoint']       = _safe(df,'endpoint','/').astype(str)
    df['method']         = _safe(df,'method','GET').astype(str).str.upper()
    df['user_agent']     = _safe(df,'user_agent','').astype(str)
    df['referrer']       = _safe(df,'referrer','').astype(str)
    df['content']        = _safe(df,'content','').astype(str)
    df['status']         = pd.to_numeric(
        _safe(df,'status',200), errors='coerce').fillna(200).astype(int)
    df['bytes']          = pd.to_numeric(
        _safe(df,'bytes',0), errors='coerce').fillna(0).astype(int)
    df['content_length'] = pd.to_numeric(
        _safe(df,'content_length',0), errors='coerce').fillna(0)
    df['latency']        = pd.to_numeric(
        _safe(df,'latency',0), errors='coerce').fillna(0.0)

    ep       = df['endpoint']
    ua       = df['user_agent']
    combined = (ep + ' ' + df['content']).apply(
        lambda s: unquote(str(s)))

    METHOD_MAP = {'GET':0,'POST':1,'PUT':2,'DELETE':3,
                  'HEAD':4,'OPTIONS':5,'PATCH':6}

    # URL features
    df['url_length']        = ep.apply(len)
    df['url_entropy']       = ep.apply(
        lambda u: _entropy(unquote(str(u))))
    df['url_special_chars'] = ep.apply(
        lambda u: sum(1 for c in unquote(str(u))
                      if c in "';\"<>(){}[]\\|`~!@#%^&*=+,"))
    df['url_param_count']   = ep.apply(lambda u: str(u).count('='))
    df['url_max_param_len'] = ep.apply(_max_param_len)
    df['url_dir_depth']     = ep.apply(lambda u: str(u).count('/'))
    df['url_is_encoded']    = ep.apply(
        lambda u: 1 if re.search(r'%[0-9a-fA-F]{2}', str(u)) else 0)
    df['url_double_encoded']= ep.apply(
        lambda u: 1 if re.search(r'%25[0-9a-fA-F]{2}', str(u)) else 0)

    # Attack flags
    df['sqli_flag']      = combined.apply(
        lambda u: 1 if SQLI_RE.search(u) else 0)
    df['xss_flag']       = combined.apply(
        lambda u: 1 if XSS_RE.search(u) else 0)
    df['traversal_flag'] = combined.apply(
        lambda u: 1 if PATH_RE.search(u) else 0)
    df['cmdinj_flag']    = combined.apply(
        lambda u: 1 if CMD_RE.search(u) else 0)
    df['rfi_flag']       = combined.apply(
        lambda u: 1 if RFI_RE.search(u) else 0)
    df['ssrf_flag']      = combined.apply(
        lambda u: 1 if SSRF_RE.search(u) else 0)
    df['scanner_ua']     = ua.apply(
        lambda u: 1 if SCANNER_RE.search(str(u)) else 0)
    df['attack_flag_sum']= (df['sqli_flag'] + df['xss_flag'] +
                             df['traversal_flag'] + df['cmdinj_flag'] +
                             df['rfi_flag'] + df['ssrf_flag'])

    # HTTP method
    df['method_encoded']       = df['method'].map(
        METHOD_MAP).fillna(7).astype(int)
    df['is_post']              = (df['method']=='POST').astype(int)
    df['is_suspicious_method'] = df['method'].apply(
        lambda m: 1 if m in ['TRACE','TRACK','CONNECT','PROPFIND']
        else 0)

    # Status codes
    df['status_2xx'] = df['status'].apply(
        lambda s: 1 if 200<=s<300 else 0)
    df['status_3xx'] = df['status'].apply(
        lambda s: 1 if 300<=s<400 else 0)
    df['status_4xx'] = df['status'].apply(
        lambda s: 1 if 400<=s<500 else 0)
    df['status_5xx'] = df['status'].apply(
        lambda s: 1 if 500<=s<600 else 0)
    df['status_403'] = (df['status']==403).astype(int)
    df['status_500'] = (df['status']==500).astype(int)

    # Content features
    df['content_length_feat'] = df['content_length']
    df['content_entropy']     = df['content'].apply(
        lambda c: _entropy(unquote(str(c))))
    df['content_sqli']        = df['content'].apply(
        lambda c: 1 if SQLI_RE.search(unquote(str(c))) else 0)
    df['content_xss']         = df['content'].apply(
        lambda c: 1 if XSS_RE.search(unquote(str(c))) else 0)

    # Response
    df['bytes_feat']   = df['bytes']
    df['latency_feat'] = df['latency']

    # User agent
    df['ua_length']   = ua.apply(len)
    df['ua_is_empty'] = (ua.str.strip()=='').astype(int)
    df['ua_is_bot']   = ua.apply(
        lambda u: 1 if re.search(
            r'(bot|crawler|spider|slurp|bingbot|googlebot)',
            str(u), re.I) else 0)

    # Referrer
    df['has_referrer'] = (
        df['referrer'].str.strip()!='').astype(int)
    df['ext_referrer'] = df['referrer'].apply(
        lambda r: 0 if str(r).strip()=='' or
        'localhost' in str(r) else 1)

    # Window features default (filled by extract_window_features)
    for col in WINDOW_FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    print(f"  Done. Shape: {df.shape} | "
          f"SQLi:{df['sqli_flag'].sum()} "
          f"XSS:{df['xss_flag'].sum()} "
          f"Traversal:{df['traversal_flag'].sum()} "
          f"Cmd:{df['cmdinj_flag'].sum()} "
          f"RFI:{df['rfi_flag'].sum()} "
          f"SSRF:{df['ssrf_flag'].sum()}")
    return df


def extract_window_features(df: pd.DataFrame,
                              window_sec=60) -> pd.DataFrame:
    """Rolling window features per IP (needs timestamps)."""
    if ('timestamp' not in df.columns or
            df['timestamp'].isna().mean() > 0.8):
        print("[Stage 2] Window features skipped (no timestamps)")
        return df

    print(f"[Stage 2] Computing {window_sec}s window features...")
    df = df.copy()
    df['timestamp'] = pd.to_datetime(
        df['timestamp'], errors='coerce', utc=True)
    df['ts_unix'] = df['timestamp'].astype('int64') // 10**9

    if 'ip' not in df.columns:
        df['ip'] = '0.0.0.0'

    rates, errs, repeats, uniq_eps, sqli_rates = [],[],[],[],[]

    for ip, grp in df.groupby('ip'):
        grp = grp.sort_values('ts_unix').reset_index()
        for i in range(len(grp)):
            t0 = grp.loc[i, 'ts_unix']
            w  = grp[(grp['ts_unix'] >= t0 - window_sec) &
                     (grp['ts_unix'] <= t0)]
            rates.append(len(w))
            errs.append(float(w['status_4xx'].mean())
                        if len(w) else 0)
            vc = w['endpoint'].value_counts()
            repeats.append(float(vc.iloc[0])/len(w)
                           if len(w) > 0 else 0)
            uniq_eps.append(int(w['endpoint'].nunique()))
            sqli_rates.append(float(w['sqli_flag'].mean())
                               if len(w) else 0)

    df['req_rate']         = rates
    df['error_rate']       = errs
    df['repeat_url_ratio'] = repeats
    df['unique_endpoints'] = uniq_eps
    df['sqli_rate_window'] = sqli_rates
    print("  Window features done.")
    return df


# ── Scaler ────────────────────────────────────────────────────

def get_X(df: pd.DataFrame) -> np.ndarray:
    """Get raw feature matrix from DataFrame."""
    for c in ALL_FEATURES:
        if c not in df.columns:
            df[c] = 0
    return df[ALL_FEATURES].fillna(0).values

def fit_and_save_scaler(df: pd.DataFrame) -> StandardScaler:
    """Fit scaler on all data and save."""
    X = get_X(df.copy())
    scaler = StandardScaler()
    scaler.fit(X)
    joblib.dump(scaler, "models/scaler.pkl")
    print("[Stage 2] Scaler fitted and saved → models/scaler.pkl")
    return scaler

def scale(df: pd.DataFrame,
          scaler: StandardScaler) -> np.ndarray:
    """Apply saved scaler to DataFrame."""
    X = get_X(df.copy())
    return scaler.transform(X)
