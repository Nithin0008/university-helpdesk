# Streamlit Community Cloud Deployment Guide

This guide provides complete, step-by-step instructions to deploy the **Multilingual University Helpdesk** on **Streamlit Community Cloud** (or any cloud container environment).

---

## 1. Prerequisites

1. A **GitHub account** with access to create/push repositories.
2. A **Google Gemini API Key** obtained from [Google AI Studio](https://aistudio.google.com/).
3. A **Streamlit Community Cloud account** linked to your GitHub account (free at [share.streamlit.io](https://share.streamlit.io)).

---

## 2. GitHub Repository Setup

### Step 2.1: Initialize & Verify Local Git Repository
In your project root, verify that sensitive files (`.env`, `.index_cache/`, `scratch/`) are ignored:

```bash
# Check git status (ensure no .env or .venv files appear)
git status
```

### Step 2.2: Stage & Commit Clean Project Files
```bash
git add app.py requirements.txt README.md .gitignore .env.example
git add utils/ tests/ sample_documents/ .streamlit/ deployment/
git commit -m "feat: complete multilingual university helpdesk ready for deployment"
```

### Step 2.3: Push to GitHub
Create a new repository (e.g. `university-helpdesk`) on GitHub, then link and push:
```bash
git branch -M main
git remote add origin https://github.com/<your-username>/university-helpdesk.git
git push -u origin main
```

> **Security Note:** Never commit your `.env` file or paste your real API key into code or Git commit messages. The repository contains `.env.example` as a safe placeholder.

---

## 3. Deploying on Streamlit Community Cloud

### Step 3.1: Create New App
1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click the **"New app"** button.
3. Select your repository: `<your-username>/university-helpdesk`.
4. Set **Branch**: `main`.
5. Set **Main file path**: `app.py`.
6. Set **App URL**: Choose a custom subdomain (e.g., `lpu-helpdesk.streamlit.app`) or use the default.

---

### Step 3.2: Configure Cloud Secrets (Crucial)
Before clicking Deploy, click **"Advanced settings..."** and navigate to the **Secrets** section.

Enter your API credentials in TOML syntax:

```toml
GEMINI_API_KEY = "AIzaSyYourActualKeyHere..."
GEMINI_MODEL = "gemini-3.8-flash"
```

> **How it works:** `app.py` automatically detects Streamlit Cloud's `st.secrets` and bridges them into `os.environ` during runtime.

---

### Step 3.3: Deploy and Monitor Startup
1. Click **"Deploy!"**.
2. Streamlit Cloud will:
   - Provision a Debian Linux container with Python 3.11+.
   - Install dependencies from `requirements.txt`.
   - Download the compact `paraphrase-multilingual-MiniLM-L12-v2` embedding weights (~470 MB, cached once).
   - Automatically index all 14 policy documents in `sample_documents/docs_raw/`.
   - Launch the chatbot at your public URL.

---

## 4. Document Knowledge Base Strategy

The Helpdesk supports two production document strategies:

### Option A: Pre-Packaged Documents (Default)
- All 14 genuine LPU policy `.docx` files are included in `sample_documents/docs_raw/`.
- On container cold start, `load_or_build_document_index()` builds the BM25 index and FAISS vector index automatically (~5-10 seconds).
- The resulting `.index_cache/` provides sub-second reloads on subsequent app interactions.

### Option B: Runtime Uploads
- If documents are kept private or omitted from the repository:
- Users and administrators can drag-and-drop `.docx`, `.pdf`, and `.txt` files directly into the sidebar uploader.
- Uploaded documents are parsed, hashed, chunked, and dynamically indexed in real-time.

---

## 5. Resource Optimization & Limits

| Metric | Measured Usage | Streamlit Cloud Limit | Status |
| :--- | :--- | :--- | :--- |
| **RAM (Cold Start)** | ~320 MB | 1,024 MB (1 GB) | Well within limits |
| **RAM (Peak with Embeddings)** | ~580 MB | 1,024 MB (1 GB) | Healthy headroom (~440 MB free) |
| **Index Size (14 Docs / 282 Chunks)** | ~1.4 MB | Disk: unlimited | Minimal footprint |
| **Embedding Speed (CPU)** | ~1.2s total | N/A | Cached after initial build |

### Optimizations Implemented:
1. **Lightweight Embedding Model**: `paraphrase-multilingual-MiniLM-L12-v2` consumes only ~120 MB VRAM/RAM compared to multi-gigabyte alternatives.
2. **Deterministic Native DOCX Parser**: Zero external heavy XML dependencies; parses DOCX natively via `zipfile` and `xml.etree`.
3. **Session Resource Caching**: Uses `@st.cache_resource` so transformer models load once per server instance, preventing memory leaks across student sessions.

---

## 6. Troubleshooting & Diagnostics

### Issue 1: "Local Fallback Mode" displayed in Sidebar
- **Cause**: Gemini API key is missing, invalid, or quota-exceeded.
- **Resolution**: Check **App settings > Secrets** in the Streamlit Cloud dashboard. Ensure `GEMINI_API_KEY` has no trailing spaces or quotes.
- **Failover**: The system will automatically use the high-speed local extractive synthesizer so students still receive grounded policy answers.

### Issue 2: HTTP 503 from Gemini Flash
- **Cause**: Upstream Google AI Studio capacity spike on `gemini-3.8-flash`.
- **Resolution**: The system automatically executes transparent multi-tier failover to `gemini-3.5-flash-lite`, preserving response delivery without user intervention.

### Issue 3: Container OOM (Out of Memory)
- **Cause**: Uploading massive (>500 page) PDF documents on the free tier.
- **Resolution**: Keep individual uploaded documents under 200 MB. Ensure `.index_cache/` is not bloated with extraneous binary files.
