import streamlit as st
import time
import gc
import config
from config import MODEL_NAME, CHUNK_SIZE, OVERLAP
from utils import (
    validate_youtube_url,
    format_output_as_markdown,
    count_words,
    truncate_text_preview
)
from extractor import extract_transcript, get_video_metadata
from chunker import split_text_into_chunks, count_tokens, get_chunking_stats
from summarizer import (
    check_ollama_connection,
    list_available_models,
    run_summarization_pipeline
)

# ─── Page Configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="YouTube Transcript Summarizer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #FF4444;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #888888;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #1E1E2E;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .status-ok {
        color: #00CC44;
        font-weight: bold;
    }
    .status-error {
        color: #FF4444;
        font-weight: bold;
    }
    .key-point {
        background-color: #1A1A2E;
        border-left: 3px solid #FF4444;
        padding: 0.5rem 1rem;
        margin: 0.4rem 0;
        border-radius: 0 8px 8px 0;
    }
    .theme-badge {
        display: inline-block;
        background-color: #2D2D44;
        color: #AAAAFF;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.9rem;
    }
    stTextArea textarea {
        font-family: monospace;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ────────────────────────────────────────────────────────────────────

from summarizer import get_groq_key

groq_active = bool(get_groq_key())
sub_text = "⚡ Powered by Groq Cloud API — Lightning Fast" if groq_active else "🔒 Powered by Local LLMs via Ollama — 100% Offline & Private"

st.markdown('<div class="main-header">🎬 YouTube Transcript Summarizer</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">{sub_text}</div>', unsafe_allow_html=True)

# ─── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Configuration")

    # Service Status
    st.subheader("🔌 System Status")
    if groq_active:
        st.success("🟢 Groq API Connected (Cloud Mode)")
    else:
        if st.button("Check Connection", use_container_width=True):
            with st.spinner("Checking..."):
                ok, msg = check_ollama_connection()
            if ok:
                st.success(msg)
            else:
                st.error(msg)
                st.code("ollama serve", language="bash")

    # Model Selection
    st.subheader("🤖 Model")
    available_models = list_available_models()
    if available_models:
        default_idx = 0
        if groq_active and config.GROQ_MODEL_NAME in available_models:
            default_idx = available_models.index(config.GROQ_MODEL_NAME)
        elif not groq_active and MODEL_NAME in available_models:
            default_idx = available_models.index(MODEL_NAME)

        selected_model = st.selectbox(
            "Select Model",
            options=available_models,
            index=default_idx,
            help="Choose model for summarization."
        )
        # Override config at runtime
        if groq_active:
            config.GROQ_MODEL_NAME = selected_model
        else:
            config.MODEL_NAME = selected_model
            import summarizer
            summarizer.GENERATE_ENDPOINT = f"{config.OLLAMA_HOST}/api/generate"
    else:
        st.warning("No models found. Pull one with Ollama or configure GROQ_API_KEY.")
        selected_model = MODEL_NAME

    # Summary Settings
    st.subheader("📝 Summary Settings")
    summary_detail = st.radio(
        "Summary Detail Level",
        options=["Brief", "Standard", "Detailed"],
        index=1,
        help="Controls the target length of the executive summary."
    )

    detail_word_map = {"Brief": 100, "Standard": 200, "Detailed": 350}
    import config as cfg
    cfg.MAX_SUMMARY_WORDS = detail_word_map[summary_detail]

    chunk_size_override = st.slider(
        "Chunk Size (tokens)",
        min_value=1000,
        max_value=4000,
        value=CHUNK_SIZE,
        step=500,
        help="Smaller = faster per chunk, more chunks. Larger = slower, fewer chunks."
    )

    # Setup Guide
    with st.expander("📖 Setup Guide"):
        st.markdown("""
**First Time Setup:**
```bash
# 1. Install Ollama
# Visit: https://ollama.ai

# 2. Start Ollama server
ollama serve

# 3. Pull the model
ollama pull llama3.2:3b-instruct-q4_K_M

# 4. Activate your venv
cd "Youtube Transcript-Summarizer"
.\\venv\\Scripts\\activate  # Windows
source venv/bin/activate   # Mac/Linux

# 5. Install dependencies
pip install -r requirements.txt

# 6. Run the app
streamlit run app.py
```
        """)

    with st.expander("ℹ️ Hardware Info"):
        import psutil
        ram = psutil.virtual_memory()
        cpu_count = psutil.cpu_count(logical=False)
        st.metric("Total RAM", f"{ram.total / (1024**3):.1f} GB")
        st.metric("Available RAM", f"{ram.available / (1024**3):.1f} GB")
        st.metric("Physical CPU Cores", cpu_count)
        st.metric("RAM Usage", f"{ram.percent}%")

# ─── Main Input Area ───────────────────────────────────────────────────────────

st.markdown("---")
col_input, col_btn = st.columns([5, 1])

with col_input:
    youtube_url = st.text_input(
        "🔗 YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
        label_visibility="collapsed"
    )

with col_btn:
    summarize_btn = st.button("▶ Summarize", type="primary", use_container_width=True)

# ─── Session State Initialization ──────────────────────────────────────────────

if "result" not in st.session_state:
    st.session_state.result = None
if "last_url" not in st.session_state:
    st.session_state.last_url = ""
if "metadata" not in st.session_state:
    st.session_state.metadata = {}

# ─── Main Processing Logic ─────────────────────────────────────────────────────

if summarize_btn and youtube_url:

    # URL Validation
    is_valid, video_id = validate_youtube_url(youtube_url)
    if not is_valid:
        st.error(
            "❌ Invalid YouTube URL. Please enter a valid YouTube video link.\n\n"
            "**Supported formats:**\n"
            "- https://www.youtube.com/watch?v=VIDEO_ID\n"
            "- https://youtu.be/VIDEO_ID\n"
            "- https://www.youtube.com/shorts/VIDEO_ID"
        )
        st.stop()

    # Ollama Check
    ok, msg = check_ollama_connection()
    if not ok:
        st.error(f"❌ {msg}")
        st.code("ollama serve", language="bash")
        st.stop()

    # Clear previous result if URL changed
    if youtube_url != st.session_state.last_url:
        st.session_state.result = None
        st.session_state.metadata = {}

    # ── Progress UI ──
    progress_placeholder = st.empty()
    status_placeholder = st.empty()

    def update_progress(stage: str, current: int, total: int):
        if stage == "chunk":
            pct = 0.1 + (current / total) * 0.7
            progress_placeholder.progress(pct, text=f"Summarizing chunk {current} of {total}...")
        elif stage == "synthesis":
            progress_placeholder.progress(0.85, text="Synthesizing final summary...")

    try:
        # Stage 1: Extraction
        status_placeholder.info("📥 Stage 1/3 — Extracting transcript from YouTube...")
        progress_placeholder.progress(0.05, text="Fetching transcript...")

        transcript, method = extract_transcript(youtube_url)
        metadata = get_video_metadata(youtube_url)
        st.session_state.metadata = metadata

        word_count = count_words(transcript)
        token_count = count_tokens(transcript)
        status_placeholder.success(
            f"✅ Transcript extracted via **{method}**. "
            f"**{word_count:,} words** / **{token_count:,} tokens**."
        )

        # Stage 2: Chunking
        progress_placeholder.progress(0.1, text="Splitting transcript into chunks...")
        status_placeholder.info("✂️ Stage 2/3 — Chunking transcript for processing...")

        chunks = split_text_into_chunks(
            transcript,
            chunk_size=chunk_size_override,
            overlap=OVERLAP
        )
        stats = get_chunking_stats(transcript, chunks)

        status_placeholder.success(
            f"✅ Transcript split into **{stats['chunk_count']} chunks** "
            f"(avg {stats['avg_tokens_per_chunk']} tokens each)."
        )

        # Stage 3: Summarization
        status_placeholder.info(
            f"🤖 Stage 3/3 — Summarizing with **{selected_model}**. "
            f"This may take {stats['chunk_count'] * 2}–{stats['chunk_count'] * 5} minutes on CPU..."
        )

        result = run_summarization_pipeline(
            transcript=transcript,
            chunks=chunks,
            video_title=metadata.get("title", ""),
            progress_callback=update_progress
        )

        progress_placeholder.progress(1.0, text="✅ Complete!")
        status_placeholder.success(
            f"🎉 Done! Processed in **{result['processing_time_seconds']}s**. "
            f"Compression ratio: **{result['compression_ratio']}x**"
        )

        st.session_state.result = result
        st.session_state.last_url = youtube_url
        gc.collect()

    except ValueError as e:
        progress_placeholder.empty()
        st.error(f"❌ Input Error: {e}")
    except RuntimeError as e:
        progress_placeholder.empty()
        st.error(f"❌ Processing Error: {e}")
    except Exception as e:
        progress_placeholder.empty()
        st.error(f"❌ Unexpected Error: {e}")
        st.exception(e)

elif summarize_btn and not youtube_url:
    st.warning("⚠️ Please enter a YouTube URL first.")

# ─── Results Display ───────────────────────────────────────────────────────────

if st.session_state.result:
    result = st.session_state.result
    metadata = st.session_state.metadata

    st.markdown("---")

    # Video Metadata Bar
    if metadata:
        st.markdown(f"### 🎬 {metadata.get('title', 'Video')}")
        meta_cols = st.columns(4)
        duration_min = metadata.get("duration_seconds", 0) // 60
        meta_cols[0].metric("Duration", f"{duration_min} min")
        meta_cols[1].metric("Uploader", metadata.get("uploader", "—")[:20])
        meta_cols[2].metric("Views", f"{metadata.get('view_count', 0):,}")
        meta_cols[3].metric("Compression", f"{result['compression_ratio']}x")

    # Stats Row
    stats_cols = st.columns(4)
    stats_cols[0].metric("Original Words", f"{result['word_count_original']:,}")
    stats_cols[1].metric("Summary Words", f"{result['word_count_summary']:,}")
    stats_cols[2].metric("Chunks", result["chunk_count"])
    stats_cols[3].metric("Processing Time", f"{result['processing_time_seconds']}s")

    st.markdown("---")

    # Results Tabs
    tab_summary, tab_keypoints, tab_themes, tab_transcript, tab_export = st.tabs([
        "📋 Summary", "🔑 Key Points", "🏷️ Themes", "📜 Full Transcript", "💾 Export"
    ])

    with tab_summary:
        st.subheader("Executive Summary")
        st.write(result["executive_summary"])
        if st.button("📋 Copy Summary", key="copy_summary"):
            st.code(result["executive_summary"])

    with tab_keypoints:
        st.subheader("Key Takeaways")
        for i, point in enumerate(result["key_points"], 1):
            st.markdown(
                f'<div class="key-point"><b>{i}.</b> {point}</div>',
                unsafe_allow_html=True
            )
        if st.button("📋 Copy Key Points", key="copy_keypoints"):
            formatted = "\n".join(
                f"{i}. {p}" for i, p in enumerate(result["key_points"], 1)
            )
            st.code(formatted)

    with tab_themes:
        st.subheader("Main Themes")
        theme_html = " ".join(
            f'<span class="theme-badge">🏷️ {theme}</span>'
            for theme in result["themes"]
        )
        st.markdown(theme_html, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Per-Chunk Summaries (Intermediate Output)")
        for i, cs in enumerate(result.get("chunk_summaries", []), 1):
            with st.expander(f"Chunk {i} Summary"):
                st.write(cs)

    with tab_transcript:
        st.subheader("Full Transcript")
        st.markdown(f"*{result['word_count_original']:,} words*")
        st.text_area(
            "Transcript",
            value=result["full_transcript"],
            height=400,
            label_visibility="collapsed"
        )

    with tab_export:
        st.subheader("Download Results")
        markdown_content = format_output_as_markdown(result)

        dl_col1, dl_col2 = st.columns(2)

        with dl_col1:
            st.download_button(
                label="⬇️ Download as Markdown (.md)",
                data=markdown_content,
                file_name=f"summary_{video_id if 'video_id' in dir() else 'video'}.md",
                mime="text/markdown",
                use_container_width=True
            )

        with dl_col2:
            st.download_button(
                label="⬇️ Download as Plain Text (.txt)",
                data=markdown_content,
                file_name=f"summary_{video_id if 'video_id' in dir() else 'video'}.txt",
                mime="text/plain",
                use_container_width=True
            )

        st.markdown("**Preview:**")
        st.text_area(
            "Export Preview",
            value=truncate_text_preview(markdown_content, 800),
            height=250,
            label_visibility="collapsed"
        )

# ─── Empty State ───────────────────────────────────────────────────────────────

elif not summarize_btn:
    st.markdown("""
    <div style="text-align:center; padding: 3rem; color: #666666;">
        <div style="font-size: 4rem;">🎬</div>
        <h3>Paste a YouTube URL above and click Summarize</h3>
        <p>Works with lectures, tutorials, podcasts, documentaries — any video with captions.</p>
        <p>All processing happens on your machine. No data leaves your laptop.</p>
    </div>
    """, unsafe_allow_html=True)