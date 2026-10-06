"""
=============================================================
STAGE 5 — INFERENCE & ALERTS
=============================================================
Runs trained models on new requests.
Generates structured alerts with MITRE mappings.
=============================================================
"""

import json
import math
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional

# ── MITRE ATT&CK mappings ─────────────────────────────────────

MITRE = {
    'SQLi':             'T1190 - SQL Injection',
    'XSS':              'T1059.007 - Cross-Site Scripting',
    'CommandInjection': 'T1059 - Command Injection',
    'PathTraversal':    'T1083 - Path Traversal',
    'LFI':              'T1083 - Local File Inclusion',
    'RFI':              'T1105 - Remote File Inclusion',
    'SSRF':             'T1090 - Server-Side Request Forgery',
    'Attack':           'T1190 - Exploit Public-Facing App',
    'Normal':            None,
    'unknown':           None,
}

REMEDIATION = {
    'SQLi':             'Use parameterized queries. Enable WAF. Block source IP.',
    'XSS':              'Enable CSP header. Sanitize inputs. Encode output.',
    'CommandInjection': 'Disable OS command execution. Use allowlist validation.',
    'PathTraversal':    'Canonicalize paths. Restrict file system access.',
    'LFI':              'Disable allow_url_include. Validate file parameters.',
    'RFI':              'Block outbound HTTP. Validate all URL parameters.',
    'SSRF':             'Block internal IPs. Use allowlist for external URLs.',
    'Attack':           'Investigate request. Block source IP immediately.',
    'Normal':            None,
    'unknown':          'Monitor source IP. Escalate for investigation.',
}

SEVERITY_BOOST = {
    'CommandInjection': 0.10,
    'SQLi':             0.08,
    'RFI':              0.10,
    'SSRF':             0.10,
    'LFI':              0.05,
    'PathTraversal':    0.07,
    'XSS':              0.05,
}


# ── Alert dataclass ───────────────────────────────────────────

@dataclass
class Alert:
    alert_id:     str
    timestamp:    str
    host_service: str
    severity:     str
    score:        float
    attack_type:  str
    confidence:   str
    key_reasons:  list
    source_ip:    str
    method:       str
    endpoint:     str
    status:       int
    mitre:        Optional[str]
    action:       Optional[str]


# ── Helper functions ──────────────────────────────────────────

def get_severity(score, atype):
    boosted = min(1.0, score + SEVERITY_BOOST.get(atype, 0))
    if boosted >= 0.85: return 'CRITICAL'
    if boosted >= 0.70: return 'HIGH'
    if boosted >= 0.50: return 'MEDIUM'
    return 'LOW'

def get_reasons(row) -> list:
    reasons = []
    if row.get('sqli_flag',0):
        reasons.append("SQL injection pattern in URL/body")
    if row.get('xss_flag',0):
        reasons.append("XSS payload detected")
    if row.get('traversal_flag',0):
        reasons.append("Path traversal attempt (../)")
    if row.get('cmdinj_flag',0):
        reasons.append("Command injection detected")
    if row.get('rfi_flag',0):
        reasons.append("Remote file inclusion URL")
    if row.get('ssrf_flag',0):
        reasons.append("SSRF to internal resource")
    if row.get('scanner_ua',0):
        reasons.append("Known attack tool user-agent")
    if row.get('url_double_encoded',0):
        reasons.append("Double URL encoding (evasion attempt)")
    if row.get('url_entropy',0) > 4.5:
        reasons.append(
            f"High URL entropy: {row.get('url_entropy',0):.2f}")
    if row.get('status_500',0):
        reasons.append("Server error 500 (possible exploit success)")
    if row.get('status_403',0):
        reasons.append("Access forbidden (blocked attempt)")
    return reasons[:5] if reasons else [
        "Anomalous pattern detected by ML model"]


# ── Main inference function ───────────────────────────────────

def run_inference(X, df_orig, ae_model, mlp_binary,
                   mlp_multi, fusion,
                   threshold=0.50,
                   host='apache-web-server-node'):

    print(f"[Stage 5] Running inference on "
          f"{len(X)} requests...")

    # Get scores from all 3 models
    ae_scores  = ae_model.score(X)
    mlp_scores = mlp_binary.predict_proba(X)
    atk_types  = mlp_multi.predict(X)

    # Fuse scores
    fused = fusion.fuse(ae_scores, mlp_scores, atk_types)

    alerts, counter, normal = [], 0, 0

    for i, row in fused.iterrows():
        score = row['final_score']
        atype = str(row['attack_type'])

        # Skip if below threshold
        if score < threshold:
            normal += 1
            continue

        # Skip if supervised says Normal
        if atype == 'Normal' and row['mlp_score'] < 0.5:
            normal += 1
            continue

        counter += 1
        orig = df_orig.iloc[i]

        # Timestamp
        ts = orig.get('timestamp', None)
        if (ts is None or
                (isinstance(ts, float) and math.isnan(ts))):
            ts = datetime.now()

        alerts.append(Alert(
            alert_id     = f"WEB-{counter:06d}",
            timestamp    = str(ts),
            host_service = host,
            severity     = get_severity(score, atype),
            score        = round(float(score), 4),
            attack_type  = atype,
            confidence   = str(row['confidence']),
            key_reasons  = get_reasons(orig),
            source_ip    = str(orig.get('ip', 'unknown')),
            method       = str(orig.get('method', 'GET')),
            endpoint     = str(orig.get('endpoint', '/'))[:120],
            status       = int(orig.get('status', 0)),
            mitre        = MITRE.get(atype),
            action       = REMEDIATION.get(
                atype, 'Investigate request.')
        ))

    print(f"  → {len(alerts)} ALERTS | {normal} Normal\n")
    return alerts


# ── Report printing ───────────────────────────────────────────

def print_report(alerts):
    icons = {'CRITICAL':'🔴','HIGH':'🟠',
             'MEDIUM':'🟡','LOW':'🟢'}

    sev_counts = {s: sum(1 for a in alerts if a.severity==s)
                  for s in ['CRITICAL','HIGH','MEDIUM','LOW']}

    print("\n" + "═"*60)
    print("  🛡️   APACHE WEB SERVER — THREAT REPORT")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═"*60)
    print(f"\n  Total Alerts : {len(alerts)}")
    for s, c in sev_counts.items():
        if c:
            print(f"  {icons[s]} {s:<10}: {c}")

    if alerts:
        atk_dist = pd.Series(
            [a.attack_type for a in alerts]
        ).value_counts()
        print(f"\n  Attack Types Detected:")
        for t, n in atk_dist.items():
            print(f"    • {t:<22}: {n}")

        print(f"\n  Top 5 Highest Score Alerts:")
        print("  " + "─"*55)
        top5 = sorted(alerts,
                      key=lambda x: x.score,
                      reverse=True)[:5]
        for a in top5:
            print(f"\n  [{a.severity}] {a.alert_id} "
                  f"| Score: {a.score:.3f}")
            print(f"  IP: {a.source_ip} "
                  f"| {a.method} | Status: {a.status}")
            print(f"  URL: {a.endpoint[:55]}")
            print(f"  Type: {a.attack_type} "
                  f"| Confidence: {a.confidence}")
            print(f"  Reason: "
                  f"{a.key_reasons[0] if a.key_reasons else 'N/A'}")
            if a.mitre:
                print(f"  MITRE: {a.mitre}")
            if a.action:
                print(f"  Action: {a.action[:60]}")

    print("\n" + "═"*60 + "\n")


# ── Save outputs ──────────────────────────────────────────────

def save_outputs(alerts, host):
    import os
    os.makedirs("outputs", exist_ok=True)

    # JSON
    out = {
        "generated_at":    datetime.now().isoformat(),
        "host":            host,
        "total_alerts":    len(alerts),
        "severity_counts": {
            s: sum(1 for a in alerts if a.severity==s)
            for s in ['CRITICAL','HIGH','MEDIUM','LOW']
        },
        "attack_type_counts": pd.Series(
            [a.attack_type for a in alerts]
        ).value_counts().to_dict(),
        "alerts": [asdict(a) for a in alerts]
    }
    with open("outputs/threat_alerts.json", 'w') as f:
        json.dump(out, f, indent=2, default=str)

    # CSV
    rows = []
    for a in alerts:
        d = asdict(a)
        d['key_reasons'] = ' | '.join(a.key_reasons)
        rows.append(d)
    pd.DataFrame(rows).to_csv(
        "outputs/threat_alerts.csv", index=False)

    print("[Stage 6] Saved → outputs/threat_alerts.json")
    print("[Stage 6] Saved → outputs/threat_alerts.csv")
