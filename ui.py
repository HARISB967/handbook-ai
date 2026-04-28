import streamlit as st
import streamlit.components.v1 as components
import httpx
import json
import time
import re
import xml.etree.ElementTree as ET
from database import (
    get_chat_history, add_chat_history,
    create_session, list_sessions, rename_session, delete_session,
    get_session_documents
)
from markdown_pdf import MarkdownPdf, Section
import tempfile
import os

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
API_URL = "http://localhost:8001"

st.set_page_config(
    page_title="HandBook.ai",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #E2E8F0; }
.stApp { background: #0D1117 !important; }

[data-testid="stSidebar"] {
    background: #161B27 !important;
    border-right: 1px solid #1E2A3B !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 1.2rem 0.9rem !important; }

.brand-title {
    font-size: 1.4rem; font-weight: 700;
    background: linear-gradient(135deg, #818CF8 0%, #C084FC 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em; margin-bottom: 0.1rem;
}
.brand-sub { font-size: 0.7rem; color: #4B5563; margin-bottom: 1rem; }

div[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
    color: #fff !important; font-weight: 600 !important;
    font-size: 0.83rem !important; border: none !important;
    border-radius: 8px !important; transition: opacity 0.2s !important;
}
div[data-testid="stButton"] button[kind="primary"]:hover { opacity: 0.85 !important; }
div[data-testid="stButton"] button:not([kind="primary"]) {
    background: transparent !important; border: none !important;
    color: #6B7280 !important; font-size: 0.82rem !important;
    padding: 0.3rem 0.4rem !important; border-radius: 6px !important;
    transition: background 0.15s, color 0.15s !important;
}
div[data-testid="stButton"] button:not([kind="primary"]):hover {
    background: #1E2A3B !important; color: #E2E8F0 !important;
}

.section-label {
    font-size: 0.65rem; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: #374151; margin: 0.9rem 0 0.4rem 0;
    padding-top: 0.75rem; border-top: 1px solid #1E2A3B;
}

.main .block-container {
    padding-top: 1rem !important; padding-left: 2rem !important;
    padding-right: 2rem !important; max-width: 860px !important;
}

.stChatMessage {
    background: #161B27 !important; border: 1px solid #1E2A3B !important;
    border-radius: 12px !important; padding: 0.9rem 1.1rem !important;
    margin-bottom: 0.6rem !important; transition: border-color 0.2s !important;
}
.stChatMessage:hover { border-color: #2D3B55 !important; }

/* Document card */
.doc-card {
    background: #161B27; border: 1px solid #1E2A3B; border-radius: 10px;
    padding: 0.75rem 1rem; margin-bottom: 0.5rem;
    display: flex; align-items: center; justify-content: space-between;
}
.doc-info { flex: 1; }
.doc-name { font-size: 0.85rem; font-weight: 500; color: #C7D2FE; }
.doc-meta { font-size: 0.7rem; color: #4B5563; margin-top: 0.15rem; }

/* Upload zone */
.upload-zone {
    background: linear-gradient(135deg, #161B27, #1a1f35);
    border: 2px dashed #2D3B55; border-radius: 12px;
    padding: 1.5rem 1rem; margin-bottom: 1rem;
    text-align: center; transition: border-color 0.2s;
}
.upload-zone:hover { border-color: #6366F1; }
.upload-title { font-size: 0.95rem; font-weight: 600; color: #818CF8; margin-bottom: 0.25rem; }
.upload-sub   { font-size: 0.78rem; color: #4B5563; }

/* Ingest banner */
.ingest-banner {
    background: linear-gradient(135deg, #1a1f35, #1e1b2e);
    border: 1px solid #3730a3; border-radius: 12px;
    padding: 1.2rem 1.5rem; margin: 1rem 0; text-align: center;
}
.ingest-title { font-size: 1rem; font-weight: 600; color: #A5B4FC; margin-bottom: 0.3rem; }
.ingest-sub   { font-size: 0.82rem; color: #6B7280; }

/* Welcome */
.welcome-wrap {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    height: 35vh; text-align: center;
}
.welcome-icon  { font-size: 2.8rem; margin-bottom: 0.6rem; }
.welcome-title { font-size: 1.25rem; font-weight: 600; color: #6B7280; margin-bottom: 0.35rem; }
.welcome-sub   { font-size: 0.85rem; color: #374151; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0D1117; }
::-webkit-scrollbar-thumb { background: #1E2A3B; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PDF Helper
# ─────────────────────────────────────────────────────────────────────────────
def generate_pdf_bytes(markdown_text: str) -> bytes:
    try:
        pdf = MarkdownPdf(toc_level=2)
        pdf.add_section(Section(markdown_text))
        
        # Save to temp file and read bytes (Windows safe)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.close()
        tmp_path = tmp.name
        
        pdf.save(tmp_path)
            
        with open(tmp_path, "rb") as f:
            pdf_bytes = f.read()
            
        os.remove(tmp_path)
        return pdf_bytes
    except Exception as e:
        print(f"PDF Gen Error: {e}")
        return b""

# ─────────────────────────────────────────────────────────────────────────────
# Session State Bootstrap
# ─────────────────────────────────────────────────────────────────────────────
def _bootstrap():
    if "bootstrapped" in st.session_state:
        return
    
    # Try to load session from URL query params
    url_session_id = st.query_params.get("session_id")
    if url_session_id:
        try:
            # Check if session actually exists
            sessions = list_sessions()
            matching_session = next((s for s in sessions if s["id"] == url_session_id), None)
            if matching_session:
                st.session_state.active_session_id   = matching_session["id"]
                st.session_state.active_session_name = matching_session["name"]
                
                # Load history and docs
                history = get_chat_history(session_id=matching_session["id"])
                st.session_state.messages = [{"role": m["role"], "content": m["content"]} for m in history]
                st.session_state.session_docs = get_session_documents(matching_session["id"])
            else:
                raise ValueError("Session ID not found in database.")
        except Exception as e:
            st.error(f"Could not load session from URL: {e}")
            url_session_id = None
            
    # If no URL session or loading failed, create a new chat
    if not url_session_id:
        try:
            ns = create_session("New Chat")
            st.session_state.active_session_id   = ns["id"]
            st.session_state.active_session_name = ns["name"]
            st.query_params["session_id"]        = ns["id"]
        except Exception as e:
            st.error(f"Could not initialise session: {e}")
            st.session_state.active_session_id   = None
            st.session_state.active_session_name = "New Chat"
        st.session_state.messages      = []
        st.session_state.session_docs  = []

    st.session_state.rename_target = None
    st.session_state.ingesting     = False
    st.session_state.ingest_queue  = []
    if "session_docs" not in st.session_state:
        st.session_state.session_docs  = []
    st.session_state.uploader_key  = 0   # Incremented to force-clear the file uploader
    st.session_state.bootstrapped  = True

_bootstrap()


def load_session(session_id: str, session_name: str):
    """Switch to a different session, loading its messages and documents."""
    st.session_state.active_session_id   = session_id
    st.session_state.active_session_name = session_name
    st.query_params["session_id"]        = session_id
    try:
        history = get_chat_history(session_id=session_id)
        st.session_state.messages = [{"role": m["role"], "content": m["content"]} for m in history]
    except Exception as e:
        st.error(f"Could not load messages: {e}")
        st.session_state.messages = []
    try:
        st.session_state.session_docs = get_session_documents(session_id)
    except Exception:
        st.session_state.session_docs = []
    st.session_state.rename_target = None


def refresh_docs():
    """Reload document list for current session from DB."""
    sid = st.session_state.active_session_id
    if sid:
        try:
            st.session_state.session_docs = get_session_documents(sid)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — Sessions only
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="brand-title">📖 HandBook.ai</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">Autonomous document intelligence</div>', unsafe_allow_html=True)

    if st.button("＋  New Chat", type="primary", use_container_width=True, key="new_chat_btn"):
        try:
            ns = create_session("New Chat")
            st.session_state.active_session_id   = ns["id"]
            st.session_state.active_session_name = ns["name"]
            st.query_params["session_id"]        = ns["id"]
            st.session_state.messages            = []
            st.session_state.session_docs        = []   # ← clear docs so previous session's list doesn't bleed in
            st.session_state.rename_target       = None
            st.session_state.uploader_key        += 1  # ← force file uploader to reset
            st.rerun()
        except Exception as e:
            st.error(f"Failed: {e}")

    st.markdown('<div class="section-label">Recent Chats</div>', unsafe_allow_html=True)
    try:
        sessions = list_sessions()
    except Exception:
        sessions = []

    for session in sessions:
        sid      = session["id"]
        sname    = session["name"]
        is_active = (sid == st.session_state.active_session_id)

        if st.session_state.rename_target == sid:
            new_name = st.text_input("Name", value=sname, key=f"ri_{sid}", label_visibility="collapsed")
            ca, cb = st.columns(2)
            with ca:
                if st.button("Save", key=f"rsave_{sid}", use_container_width=True):
                    try:
                        rename_session(sid, new_name.strip() or sname)
                        if is_active:
                            st.session_state.active_session_name = new_name.strip() or sname
                        st.session_state.rename_target = None
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
            with cb:
                if st.button("Cancel", key=f"rcancel_{sid}", use_container_width=True):
                    st.session_state.rename_target = None
                    st.rerun()
        else:
            cn, ce, cd = st.columns([7, 1, 1])
            with cn:
                prefix = "▸ " if is_active else "  "
                if st.button(prefix + sname, key=f"sel_{sid}", use_container_width=True):
                    load_session(sid, sname)
                    st.rerun()
            with ce:
                if st.button("✏", key=f"ren_{sid}", help="Rename"):
                    st.session_state.rename_target = sid
                    st.rerun()
            with cd:
                if st.button("✕", key=f"del_{sid}", help="Delete"):
                    try:
                        delete_session(sid)
                        if is_active:
                            remaining = [s for s in sessions if s["id"] != sid]
                            if remaining:
                                nxt = remaining[0]
                                load_session(nxt["id"], nxt["name"])
                            else:
                                nw = create_session("New Chat")
                                load_session(nw["id"], nw["name"])
                                st.session_state.messages     = []
                                st.session_state.session_docs = []
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

    # ── Live Backend Logs ──────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Backend Logs</div>', unsafe_allow_html=True)
    col_r, col_c = st.columns(2)
    with col_r:
        st.button("↻ Refresh", key="refresh_logs", use_container_width=True)
    with col_c:
        if st.button("✕ Clear", key="clear_logs_btn", use_container_width=True):
            try:
                httpx.get(f"{API_URL}/logs/clear", timeout=3.0)
            except Exception:
                pass
            st.rerun()

    try:
        resp = httpx.get(f"{API_URL}/logs", timeout=3.0)
        log_lines = resp.json() if resp.status_code == 200 else []
    except Exception:
        log_lines = ["⚠ Backend not reachable on port 8001"]

    def _colour_line(line: str) -> str:
        tag_colours = {
            "[API]": "#60A5FA", "[ROUTER]": "#A78BFA",
            "[INGESTION]": "#34D399", "[PDF]": "#34D399", "[EMBEDDING]": "#34D399",
            "[LLM/KG]": "#FBBF24", "[SHORT]": "#60A5FA",
            "[AGENTWRITE]": "#F472B6", "[LONGWRITER]": "#F472B6",
            "[DB]": "#94A3B8", "[APP]": "#CBD5E1", "[QUERY]": "#A78BFA",
            "ERROR": "#F87171",
        }
        colour = "#4B5563"
        for tag, c in tag_colours.items():
            if tag in line:
                colour = c
                break
        safe = line.replace("<", "&lt;").replace(">", "&gt;")
        return f'<span style="color:{colour};font-family:\'Courier New\',monospace;font-size:0.68rem;line-height:1.6;">{safe}</span>'

    tail     = log_lines[-60:] if log_lines else ["No logs yet."]
    log_html = "<br>".join(_colour_line(l) for l in tail)
    st.markdown(
        f'<div style="background:#080C12;border:1px solid #1E2A3B;border-radius:8px;'
        f'padding:0.6rem 0.75rem;max-height:230px;overflow-y:auto;">{log_html}</div>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────────
# 🕸️ Knowledge Graph Visualizer (D3.js)
# ─────────────────────────────────────────────────────────────────
def render_knowledge_graph(graphml_content: str):
    """Parses GraphML and renders a premium D3.js force-directed graph."""
    try:
        # Simple extraction of nodes and edges from GraphML
        root = ET.fromstring(graphml_content)
        ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
        
        nodes = []
        edges = []
        
        # Parse nodes
        for node in root.findall(".//g:node", ns):
            node_id = node.get("id")
            # Try to find a label or name in data keys
            label = node_id
            for data in node.findall("g:data", ns):
                if data.text:
                    label = data.text
                    break
            nodes.append({"id": node_id, "label": label})
            
        # Parse edges
        for edge in root.findall(".//g:edge", ns):
            edges.append({"source": edge.get("source"), "target": edge.get("target")})

        if not nodes:
            st.warning("The Knowledge Graph is still being built. Ingest more documents!")
            return

        # D3.js HTML/JS Template
        d3_html = f"""
        <div id="graph-container" style="width: 100%; height: 600px; background: #0D1117; border-radius: 12px; border: 1px solid #1E2A3B; overflow: hidden;">
            <svg id="canvas" style="width: 100%; height: 100%; cursor: grab;"></svg>
        </div>
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <script>
            const nodes = {json.dumps(nodes)};
            const links = {json.dumps(edges)};
            
            const svg = d3.select("#canvas");
            const width = document.getElementById('graph-container').clientWidth;
            const height = 600;
            
            const simulation = d3.forceSimulation(nodes)
                .force("link", d3.forceLink(links).id(d => d.id).distance(80))
                .force("charge", d3.forceManyBody().strength(-200))
                .force("center", d3.forceCenter(width / 2, height / 2))
                .force("collision", d3.forceCollide().radius(40));

            const g = svg.append("g");
            
            svg.call(d3.zoom().on("zoom", (event) => g.attr("transform", event.transform)));

            const link = g.append("g")
                .selectAll("line")
                .data(links)
                .join("line")
                .attr("stroke", "#2D3B55")
                .attr("stroke-opacity", 0.6)
                .attr("stroke-width", 1.5);

            const node = g.append("g")
                .selectAll("circle")
                .data(nodes)
                .join("circle")
                .attr("r", 8)
                .attr("fill", (d) => d.id.includes('chunk') ? "#818CF8" : "#C084FC")
                .attr("stroke", "#fff")
                .attr("stroke-width", 1.5)
                .call(d3.drag()
                    .on("start", (event, d) => {{
                        if (!event.active) simulation.alphaTarget(0.3).restart();
                        d.fx = d.x; d.fy = d.y;
                    }})
                    .on("drag", (event, d) => {{ d.fx = event.x; d.fy = event.y; }})
                    .on("end", (event, d) => {{
                        if (!event.active) simulation.alphaTarget(0);
                        d.fx = null; d.fy = null;
                    }}));

            const label = g.append("g")
                .selectAll("text")
                .data(nodes)
                .join("text")
                .text(d => d.label.length > 20 ? d.label.substring(0, 20) + "..." : d.label)
                .attr("font-size", "10px")
                .attr("fill", "#94A3B8")
                .attr("dx", 12)
                .attr("dy", 4)
                .attr("pointer-events", "none");

            simulation.on("tick", () => {{
                link.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
                node.attr("cx", d => d.x).attr("cy", d => d.y);
                label.attr("x", d => d.x).attr("y", d => d.y);
            }});
        </script>
        """
        components.html(d3_html, height=620)
        
    except Exception as e:
        st.error(f"Failed to render graph: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Area
# ─────────────────────────────────────────────────────────────────────────────
session_id = st.session_state.active_session_id

# ── 1. Ingestion lock ─────────────────────────────────────────────────────────
if st.session_state.ingesting and st.session_state.ingest_queue:
    st.markdown("""
    <div class="ingest-banner">
        <div class="ingest-title">🔄 Building Knowledge Graph…</div>
        <div class="ingest-sub">⛔ Chat is locked until all documents are processed.</div>
    </div>
    """, unsafe_allow_html=True)

    total_files = len(st.session_state.ingest_queue)
    all_done    = True
    status_boxes = {}
    for fname, _ in st.session_state.ingest_queue:
        status_boxes[fname] = st.status(f"📄 {fname}", expanded=True)

    for fname, fbytes in st.session_state.ingest_queue:
        sb = status_boxes[fname]
        try:
            with httpx.stream(
                "POST", f"{API_URL}/ingest/stream",
                files={"file": (fname, fbytes, "application/pdf")},
                data={"session_id": session_id},
                timeout=600.0
            ) as r:
                if r.status_code != 200:
                    sb.update(label=f"❌ {fname} — API Error {r.status_code}", state="error", expanded=True)
                    sb.write(f"Server returned error code {r.status_code}.")
                    all_done = False
                    continue

                for line in r.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    try:
                        data = json.loads(line[6:])
                        msg  = data.get("content", "")
                        kind = data.get("type")
                        if msg:
                            sb.write(msg)
                        if kind == "done":
                            if "⚠️" in msg:
                                sb.update(label=f"⚠️ {fname} (Partial)", state="complete", expanded=False)
                                all_done = False
                            else:
                                sb.update(label=f"✅ {fname}", state="complete", expanded=False)
                        elif kind == "error":
                            sb.update(label=f"❌ {fname}", state="error", expanded=True)
                            all_done = False
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            sb.update(label=f"❌ {fname} — connection error", state="error", expanded=True)
            sb.write(str(e))
            all_done = False

    st.session_state.ingesting    = False
    st.session_state.ingest_queue = []
    refresh_docs()
    st.session_state.uploader_key += 1   # Reset file uploader so it shows empty
    if all_done:
        st.success(f"✅ All {total_files} document(s) ingested! Ask away.")
    else:
        st.warning("⚠️ Some documents had issues.")
    st.rerun()

# ── 2. Normal view ────────────────────────────────────────────────────────────
else:
    # ── Main Tabs ───────────────────────────────────────────────────────────
    chat_tab, graph_tab = st.tabs(["💬 Chat Interface", "🕸️ Knowledge Graph"])

    with chat_tab:
        # ── Document management section (always shown at top) ───────────────────
        docs = st.session_state.get("session_docs", [])
        with st.expander(
            f"📚 Documents ({len(docs)} uploaded)" if docs else "📁 Upload Documents",
            expanded=(len(docs) == 0 and len(st.session_state.messages) == 0)
        ):
            if docs:
                for doc in docs:
                    status_icons = {"success": "✅", "partial": "⚠️", "duplicate": "📋", "failed": "❌"}
                    icon = status_icons.get(doc.get("status", "success"), "📄")
                    wc   = doc.get("word_count", 0)
                    pg   = doc.get("page_count", 0)
                    sz   = doc.get("file_size_bytes", 0)
                    col_name, col_del = st.columns([9, 1])
                    with col_name:
                        st.markdown(
                            f'<div class="doc-card">'
                            f'<div class="doc-info">'
                            f'<div class="doc-name">{icon} {doc["filename"]}</div>'
                            f'<div class="doc-meta">{wc:,} words · {pg} pages · {sz/1024:.0f} KB · {doc.get("status","")}</div>'
                            f'</div></div>',
                            unsafe_allow_html=True
                        )
                    with col_del:
                        if st.button("🗑", key=f"deldoc_{doc['id']}", help="Remove document record"):
                            try:
                                httpx.delete(f"{API_URL}/document/{doc['id']}", timeout=5.0)
                                refresh_docs()
                                st.rerun()
                            except Exception as e:
                                st.error(str(e))
                st.divider()

        # Upload area
        st.markdown('<div class="upload-title">➕ Add Documents</div>', unsafe_allow_html=True)
        st.caption("PDFs are ingested into this session's private knowledge base.")
        uploaded_files = st.file_uploader(
            "Upload PDFs", type=["pdf"], accept_multiple_files=True,
            label_visibility="collapsed",
            disabled=st.session_state.ingesting,
            key=f"pdf_uploader_{st.session_state.uploader_key}"   # Changes key = force clear
        )
        if st.button("⬆ Ingest PDFs", type="primary", use_container_width=True,
                     key="main_ingest_btn", disabled=st.session_state.ingesting):
            if not uploaded_files:
                st.warning("Select at least one PDF first.")
            else:
                st.session_state.ingest_queue = [(f.name, f.getvalue()) for f in uploaded_files]
                st.session_state.ingesting    = True
                st.rerun()

    # ── Chat history ─────────────────────────────────────────────────────────
    if not st.session_state.messages:
        st.markdown("""
        <div class="welcome-wrap">
            <div class="welcome-icon">💬</div>
            <div class="welcome-title">Ask anything</div>
            <div class="welcome-sub">Upload PDFs above, then ask questions about them.<br>
            Or just chat — general questions work too!</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show download buttons for historical handbooks
                if message["role"] == "assistant" and "Generated by HandBook.ai" in message["content"]:
                    wc = len(message["content"].split())
                    st.caption(f"📄 {wc:,} words generated")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "⬇ Download Markdown (.md)",
                            data=message["content"],
                            file_name="handbook.md",
                            mime="text/markdown",
                            key=f"hist_dl_md_{i}"
                        )
                    with col2:
                        pdf_data = generate_pdf_bytes(message["content"])
                        if pdf_data:
                            st.download_button(
                                "⬇ Download PDF (.pdf)",
                                data=pdf_data,
                                file_name="handbook.pdf",
                                mime="application/pdf",
                                key=f"hist_dl_pdf_{i}"
                            )

    # ── Chat input ────────────────────────────────────────────────────────────
    if prompt := st.chat_input(
        "Ask questions or request a handbook…",
        disabled=st.session_state.ingesting
    ):
        # Auto-name session on first message
        if len(st.session_state.messages) == 0 and session_id:
            auto_name = prompt[:40] + ("…" if len(prompt) > 40 else "")
            try:
                rename_session(session_id, auto_name)
                st.session_state.active_session_name = auto_name
            except Exception:
                pass

        # Capture history BEFORE adding current message
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-4:]]

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder   = st.empty()
            full_response = ""
            status_box    = None
            doc_names     = [d["filename"] for d in st.session_state.get("session_docs", [])]

            try:
                with httpx.stream(
                    "POST", f"{API_URL}/chat",
                    json={
                        "message":    prompt,
                        "session_id": session_id,
                        "doc_names":  doc_names,
                        "history":    history,
                    },
                    timeout=600.0
                ) as r:
                    for line in r.iter_lines():
                        if not line.startswith("data: "):
                            continue
                        try:
                            data     = json.loads(line[6:])
                            msg_type = data.get("type")
                            content  = data.get("content", "")

                            if msg_type == "intent" and content == "LONG":
                                status_box = st.status("🚀 AgentWrite — Qwen2.5-72B writing your handbook…", expanded=True)
                            elif msg_type == "token":
                                full_response += content
                                placeholder.markdown(full_response + "▌")
                            elif msg_type == "status":
                                if status_box:
                                    status_box.update(label=content, state="running")
                                    status_box.write(content)
                            elif msg_type == "partial_handbook":
                                # Live section preview: show growing document as each section completes
                                wc_so_far = len(content.split())
                                placeholder.markdown(
                                    f"*✍️ Generating… {wc_so_far:,} words so far*\n\n---\n\n{content}",
                                    unsafe_allow_html=False
                                )
                                full_response = content  # Keep latest partial so save works
                            elif msg_type == "handbook":
                                if status_box:
                                    status_box.update(label="✅ Handbook complete!", state="complete", expanded=False)
                                full_response = content
                                placeholder.markdown(full_response)
                            elif msg_type == "error":
                                st.error(f"Backend error: {content}")
                        except json.JSONDecodeError:
                            pass

            except httpx.ConnectError:
                full_response = f"⚠️ Cannot connect to backend at **{API_URL}**. Is `uvicorn` running on port 8001?"
                placeholder.markdown(full_response)
            except Exception as e:
                full_response = f"⚠️ Unexpected error: {e}"
                placeholder.markdown(full_response)

            if full_response:
                placeholder.markdown(full_response)

                if "Generated by HandBook.ai" in full_response:
                    wc = len(full_response.split())
                    st.caption(f"📄 {wc:,} words generated")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "⬇ Download Markdown (.md)",
                            data=full_response,
                            file_name="handbook.md",
                            mime="text/markdown",
                            key=f"dl_md_{len(st.session_state.messages)}"
                        )
                    with col2:
                        pdf_data = generate_pdf_bytes(full_response)
                        if pdf_data:
                            st.download_button(
                                "⬇ Download PDF (.pdf)",
                                data=pdf_data,
                                file_name="handbook.pdf",
                                mime="application/pdf",
                                key=f"dl_pdf_{len(st.session_state.messages)}"
                            )

                st.session_state.messages.append({"role": "assistant", "content": full_response})
                try:
                    add_chat_history("user",      prompt,        session_id=session_id)
                    add_chat_history("assistant", full_response, session_id=session_id)
                except Exception as db_err:
                    st.toast(f"⚠️ History save failed: {db_err}")

    with graph_tab:
        st.markdown(
            '<div style="margin-bottom: 1.5rem;">'
            '<h3 style="margin-bottom: 0.2rem;">🕸️ Knowledge Map</h3>'
            '<p style="font-size: 0.85rem; color: #6B7280;">Explore the entities and relationships extracted from your documents.</p>'
            '</div>',
            unsafe_allow_html=True
        )
        
        if st.button("🔄 Refresh Knowledge Graph", use_container_width=True):
            try:
                resp = httpx.get(f"{API_URL}/session/{session_id}/graph", timeout=15.0)
                if resp.status_code == 200:
                    graph_data = resp.json()
                    if "graphml" in graph_data:
                        render_knowledge_graph(graph_data["graphml"])
                    elif "error" in graph_data:
                        st.info(f"💡 {graph_data['error']}")
                    else:
                        st.info("No graph data found yet. Ingest a document first!")
                else:
                    st.error(f"Backend error ({resp.status_code}): {resp.text}")
            except json.JSONDecodeError:
                st.error(f"Backend returned invalid data: {resp.text[:200]}")
            except Exception as e:
                st.error(f"Connection error: {e}")
        else:
            st.info("Click the button above to visualize your Document Intelligence.")
