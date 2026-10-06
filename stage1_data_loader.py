"""
=============================================================
STAGE 1 — DATA LOADER
=============================================================
Loads all 3 datasets:
  1. access.log           (unlabeled Apache log)
  2. csic_binary.csv      (binary labeled)
  3. multiclass_attacks.csv (8 attack types)
=============================================================
"""

import re
import pandas as pd
from datetime import datetime

APACHE_REGEX = re.compile(
    r'(?P<ip>\S+)\s+-\s+-\s+'
    r'\[(?P<timestamp>[^\]]+)\]\s+'
    r'"(?P<method>\S+)\s+(?P<endpoint>\S+)\s+(?P<http_version>[^"]+)"\s+'
    r'(?P<status>\d{3})\s+'
    r'(?P<bytes>\S+)\s+'
    r'"(?P<referrer>[^"]*)"\s+'
    r'"(?P<user_agent>[^"]*)"'
)


# ── Loader 1: Unlabeled Apache log ────────────────────────────

def load_unlabeled_apache(filepath="data/access.log"):
    print(f"[Stage 1] Loading unlabeled Apache log: {filepath}")
    records, skipped = [], 0
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                m = APACHE_REGEX.match(line.strip())
                if not m:
                    skipped += 1
                    continue
                r = m.groupdict()
                try:
                    r['timestamp'] = datetime.strptime(
                        r['timestamp'].strip(), "%d/%b/%Y:%H:%M:%S %z")
                except:
                    r['timestamp'] = None
                r['status']         = int(r['status'])
                r['bytes']          = int(r['bytes']) if r['bytes'] != '-' else 0
                r['referrer']       = '' if r['referrer'] == '-' else r['referrer']
                r['content']        = ''
                r['content_length'] = 0
                r['latency']        = 0.0
                r['label']          = -1
                r['attack_type']    = 'unknown'
                r['source']         = 'unlabeled'
                records.append(r)
    except FileNotFoundError:
        print(f"  [Warning] File not found: {filepath}")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    print(f"  Loaded: {len(df)} rows | Skipped: {skipped} lines")
    return df


# ── Loader 2: Binary labeled CSIC ─────────────────────────────

def load_binary_csic(filepath="data/csic_binary.csv", sample_n=None):
    print(f"[Stage 1] Loading binary labeled CSIC: {filepath}")
    try:
        df = pd.read_csv(filepath, on_bad_lines='skip')
    except FileNotFoundError:
        print(f"  [Warning] File not found: {filepath}")
        return pd.DataFrame()

    df.columns = [c.strip().lower().replace('-','_').replace(' ','_')
                  for c in df.columns]

    # Find label column
    label_col = None
    for candidate in ['classification', 'label', 'class', 'type']:
        if candidate in df.columns:
            label_col = candidate
            break
    if label_col is None:
        first_col = df.columns[0]
        vals = df[first_col].astype(str).str.strip().str.lower().unique()
        if any(v in ['normal','anomaly','attack'] for v in vals):
            label_col = first_col

    if label_col:
        print(f"  Label column: '{label_col}' | "
              f"Values: {df[label_col].astype(str).str.strip().unique()[:5]}")
        df['label'] = df[label_col].astype(str).str.strip().apply(
            lambda x: 0 if x.lower() in
            ['normal','0','false','benign','legitimate'] else 1
        )
    else:
        print("  [Warning] No label column found")
        df['label'] = -1

    df['attack_type'] = df['label'].map(
        {0:'Normal', 1:'Attack', -1:'unknown'})

    # Find URL column
    url_col = next((c for c in ['url','request_url','uri']
                    if c in df.columns), None)
    if url_col:
        df['endpoint'] = df[url_col].astype(str).apply(
            lambda u: re.sub(r'https?://[^/]+','',
                             u.split(' HTTP')[0].strip()) or '/')
    else:
        df['endpoint'] = '/'

    # Ensure columns
    for col, default in [('method','GET'),('ip','127.0.0.1'),
                          ('user_agent',''),('referrer',''),
                          ('content',''),('http_version','HTTP/1.1')]:
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default).astype(str)

    df['status']         = 200
    df['bytes']          = pd.to_numeric(
        df.get('lenght', df.get('length', 0)),
        errors='coerce').fillna(0)
    df['content_length'] = df['bytes'].copy()
    df['latency']        = 0.0
    df['timestamp']      = pd.NaT
    df['source']         = 'binary_labeled'

    if sample_n and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)

    print(f"  Loaded: {len(df)} rows | "
          f"Normal: {(df['label']==0).sum()} | "
          f"Attack: {(df['label']==1).sum()}")
    return df


# ── Loader 3: Multiclass labeled ──────────────────────────────

def load_multiclass(filepath="data/multiclass_attacks.csv",
                    sample_n=50000):
    print(f"[Stage 1] Loading multiclass labeled: {filepath}")
    try:
        df = pd.read_csv(filepath, on_bad_lines='skip')
    except FileNotFoundError:
        print(f"  [Warning] File not found: {filepath}")
        return pd.DataFrame()

    df.columns = [c.strip().lower() for c in df.columns]

    # Parse timestamp if exists
    if 'timestamp' in df.columns:
        def parse_ts(t):
            for fmt in ["%d/%b/%Y:%H:%M:%S %z",
                        "%Y-%m-%d %H:%M:%S",
                        "%d/%b/%Y:%H:%M:%S"]:
                try:
                    return datetime.strptime(str(t).strip(), fmt)
                except:
                    pass
            return None
        df['timestamp'] = df['timestamp'].apply(parse_ts)

    df['attack_type'] = df.get(
        'attack_type', pd.Series(['Normal']*len(df))
    ).astype(str).str.strip()

    df['label'] = df['attack_type'].apply(
        lambda x: 0 if x.lower() == 'normal' else 1)

    # Numeric fields
    df['status']  = pd.to_numeric(
        df.get('status', 200), errors='coerce').fillna(200).astype(int)
    df['bytes']   = pd.to_numeric(
        df.get('bytes', 0), errors='coerce').fillna(0).astype(int)
    df['latency'] = pd.to_numeric(
        df.get('latency', 0), errors='coerce').fillna(0.0)

    for col, default in [('ip','0.0.0.0'),('method','GET'),
                          ('endpoint','/'),('http_version','HTTP/1.1'),
                          ('referrer',''),('user_agent',''),
                          ('content',''),('content_length', 0)]:
        if col not in df.columns:
            df[col] = default

    df['source'] = 'multiclass_labeled'

    # Stratified sample
    if sample_n and len(df) > sample_n:
        df = df.groupby('attack_type', group_keys=False).apply(
            lambda x: x.sample(
                min(len(x), max(1, int(sample_n * len(x) / len(df)))),
                random_state=42)
        ).reset_index(drop=True)
        print(f"  Sampled: {len(df)} rows (stratified)")

    print(f"  Loaded: {len(df)} rows")
    print(f"  Distribution:\n"
          f"{df['attack_type'].value_counts().to_string()}")
    return df
