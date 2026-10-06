# 🌐 Web Server Log Cyber Attack Detection — Federated Learning Node

> **Final Year Research Project Component**  
> **Project Title:** AI-Driven Log Analysis with Federated Learning  
> **Module Scope:** Web Server Log (Apache/Nginx) Security Analysis & Federated Client  
> **Author:** Member 1 — Individual Component  

---

## 📌 Project Overview

This repository contains the **Web Server Log Analysis Module**, built as part of a 4-member Final Year Research Project. The overall system leverages **Federated Learning** across heterogeneous log sources to detect cyber threats collaboratively without exposing sensitive enterprise log data.

### 👥 Team Component Breakdown
| Member | Assigned Log Subsystem | Focus Area |
| :--- | :--- | :--- |
| **Member 1 (This Repo)** | **Web Server Logs (Apache / Nginx)** | **HTTP Attack Detection & Federated Client** |
| Member 2 | System / OS Logs (Linux/Windows) | Auth Failures & Privilege Escalation |
| Member 3 | Network & Firewall Logs | Traffic Anomalies & Intrusion Detection |
| Member 4 | Cloud & Database Logs | Cloud Audit Logs & SQL Query Anomalies |

---

## ⚙️ Architecture & Pipeline Stages

The Web Server Log node processes both raw unstructured Apache logs (`access.log`) and structured security datasets (CSIC 2010 Web Attack dataset) through a 5-stage AI pipeline:

```
┌─────────────────┐    ┌────────────────────┐    ┌──────────────────────┐
│  Stage 1: Data  │ ──►│ Stage 2: Security  │ ──►│ Stage 3: Tri-Model   │
│  Ingestion &    │    │ Feature Extraction │    │ Machine Learning     │
│  Parsing        │    │ (Entropy, Window)  │    │ (AE + MLP1 + MLP2)   │
└─────────────────┘    └────────────────────┘    └──────────────────────┘
                                                            │
┌─────────────────┐    ┌────────────────────┐               │
│ Stage 5: Real   │ ◄──│ Stage 4: Federated │ ◄─────────────┘
│ Time Inference  │    │ Learning Client    │
│ & Alerting      │    │ (Flower flwr)      │
└─────────────────┘    └────────────────────┘
```

### 🧠 Tri-Model Machine Learning Engine
1. **Unsupervised Deep Autoencoder:** Learns baseline normal web traffic behavior and flags zero-day anomalies based on reconstruction loss thresholds.
2. **MLP Binary Classifier:** High-precision supervised classifier separating benign requests from web attack payloads.
3. **MLP Multiclass Classifier:** Categorizes threats into specific attack vectors:
   - SQL Injection (SQLi)
   - Cross-Site Scripting (XSS)
   - Path Traversal / LFI
   - Directory Bruteforce & Admin Probe
   - Remote Code Execution (RCE) / Command Injection

---

## 📂 Repository Structure

```
.
├── stage1_data_loader.py    # Apache access.log regex parser & dataset loader
├── stage2_features.py       # Payload entropy, URL pattern & window feature extractor
├── stage3_models.py         # Keras Autoencoder, MLP Binary & MLP Multiclass architectures
├── stage4_federated.py      # Flower (flwr) Federated Learning client implementation
├── stage5_inference.py      # Multi-model fusion engine & alert generator
├── main_pipeline.py         # End-to-end training, FL execution & evaluation script
├── test_simulation.py       # Interactive attack simulator & real-time log tester
├── requirements.txt         # Required Python packages
├── .gitignore               # Excludes large logs & binaries from Git
├── data/
│   └── access.log           # Sample Apache access log for local testing
├── models/                  # Saved model weights (.keras, .pkl) [Ignored by git]
└── outputs/                 # Threat alerts & JSON report output [Ignored by git]
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9+ 
- Virtual Environment recommended (`venv` or `conda`)

### 2. Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/YOUR_USERNAME/weblog-federated-analysis.git
cd weblog-federated-analysis
pip install -r requirements.txt
```

---

## 🏃 Execution Guide

### Option A: Run Full Pipeline
Train local models, perform feature scaling, run federated client rounds, and execute inference:
```bash
python main_pipeline.py
```

### Option B: Run Interactive Threat Simulator
Test the trained model stack against simulated cyber attacks and live log streams:
```bash
python test_simulation.py
```

### Option C: Connect to Federated Aggregation Server
To participate in global federated model aggregation led by the group FL server:
```bash
python stage4_federated.py --server_address "SERVER_IP:8080"
```

---

## 📊 Datasets & Privacy

For full model retraining on large-scale datasets:
- **CSIC 2010 Web Attack Dataset:** Place `csic_binary.csv` and `multiclass_attacks.csv` inside the `data/` folder.
- **Raw Web Server Logs:** Place full `access.log` inside the project root or `data/` directory.

*(Note: Raw log datasets and trained model binaries are excluded from Git repository via `.gitignore` to comply with storage limits and security best practices).*

---

## 📜 Authors & Citation

- **Author:** Web Log Subsystem Lead (Final Year Project)
- **Institution:** SLIIT — School of Computing
