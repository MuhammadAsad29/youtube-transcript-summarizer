# 🎬 YouTube Transcript Summarizer

> A fast YouTube transcript summarizer powered by **Streamlit**, supporting both **100% Local Inference (Ollama)** and **Cloud Deployment (Groq Cloud API)**.

---

## ✨ Features

- 📥 **Automated Transcript Extraction**: Fetches video captions via `youtube-transcript-api` with fallback metadata support via `yt-dlp`.
- ✂️ **Token-Aware Chunking**: Intelligently chunks long transcripts (`tiktoken`) with overlapping contexts.
- 🤖 **Map-Reduce Summarization Pipeline**:
  - **Executive Summary**: Cohesive overall summary (Brief, Standard, or Detailed).
  - **Key Takeaways**: Extracted bullet points of core insights.
  - **Main Themes**: Identified key topics and section summaries.
- ⚡ **Dual AI Engines**:
  - **Local Mode (Ollama)**: 100% private, offline summarization with Llama 3.2 3B.
  - **Cloud Mode (Groq API)**: Ultra-fast cloud inference with Llama 3.1 8B (perfect for Streamlit Cloud deployment).
- 💾 **Export Options**: Download summaries directly as Markdown (`.md`) or Plain Text (`.txt`).

---

## 🏗️ Project Structure

```text
Youtube Transcript-Summarizer/
├── app.py              # Streamlit UI & interactive dashboard
├── summarizer.py       # Map-Reduce pipeline & LLM API handlers (Groq & Ollama)
├── extractor.py        # YouTube transcript & metadata fetching
├── chunker.py          # Token counting & text chunking logic
├── config.py           # Application settings & environment configuration
├── utils.py            # Formatting, validation, & logging utilities
├── tests/              # Unit test suite
├── requirements.txt    # Python package dependencies
└── README.md           # Documentation
```

---

## 🚀 Quickstart Guide

### 1. Local Mode (Using Ollama)

```bash
# Start Ollama server
ollama serve

# Pull Llama 3.2 3B model
ollama pull llama3.2:3b-instruct-q4_K_M

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

### 2. Cloud Mode (Streamlit Cloud Deployment)

1. Fork or push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and create a **New App**.
3. Select your repository and set `app.py` as the main file path.
4. Under **Advanced settings** -> **Secrets**, add your free Groq API key:
   ```toml
   GROQ_API_KEY = "gsk_your_groq_api_key_here"
   ```
5. Click **Deploy!**

---

## ⚙️ Configuration (.env)

```env
# Local Ollama Settings
OLLAMA_HOST=http://localhost:11434
MODEL_NAME=llama3.2:3b-instruct-q4_K_M

# Cloud Groq Settings (Optional for Cloud Mode)
GROQ_API_KEY=gsk_xxxxxxxxxxxx
GROQ_MODEL_NAME=llama-3.1-8b-instant

# Pipeline Parameters
CHUNK_SIZE=3000
OVERLAP=200
TEMPERATURE=0.3
TIMEOUT=120
MAX_SUMMARY_WORDS=200
```

---

## 📝 License

Distributed under the MIT License.
