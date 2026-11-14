"""EE6405 study assistant -- chat page with a source panel.

    streamlit run src/ee6405/app/main.py      (or: make app)

Clicking a citation under an answer opens the original file on the right at the
cited place: the PDF page, the video at the timestamp, the notebook cell. When
the raw file is not on this machine the panel shows the indexed text instead.
"""

from __future__ import annotations

import html
import importlib.util
import time
import uuid

import streamlit as st
import streamlit.components.v1 as components

from ee6405.agent.chain import StudyAssistant
from ee6405.agent.llm import EchoLLM, build_llm
from ee6405.agent.memory import ConversationMemory
from ee6405.app import sources as src
from ee6405.config import settings
from ee6405.i18n import tr
from ee6405.kg.course_map import (
    Corpus,
    CourseMap,
    build_course_map,
    build_query_map,
    scan_corpus,
    triples_to_map,
)
from ee6405.kg.extract import extract_triples_llm
from ee6405.kg.render import to_obsidian_html
from ee6405.rag.retriever import build_retriever
from ee6405.rag.store import ChunkStore
from ee6405.types import Chunk, Hit

st.set_page_config(page_title=tr("EE6405 学习助手", "EE6405 Study Assistant"), page_icon=":material/menu_book:", layout="wide")

ALL_WEEKS = tr("全部周次", "All weeks")
RETRIEVERS = {"bm25": "BM25", "dense": tr("语义", "Dense"), "hybrid": tr("混合", "Hybrid")}
NEEDS_ML = {"dense", "hybrid"}
KG_HEIGHT = 560  # px, canvas height of the knowledge map
KG_HEIGHT_WIDE = 720  # when the map is expanded to most of the page
HAS_ML = importlib.util.find_spec("sentence_transformers") is not None
MODALITY_GROUPS = {
    tr("课件", "Slides"): ["slides"],
    tr("视频转写", "Lecture video"): ["transcript"],
    tr("代码", "Code"): ["notebook"],
    tr("测验", "Quizzes"): ["mcq", "assessment_pdf"],
    tr("其他", "Other"): ["misc", "clarification"],
}
EXAMPLES = [
    "What is the difference between BERT and RoBERTa?",
    tr("TF-IDF 是怎么计算的？", "How is TF-IDF calculated?"),
    tr("第 10 周讲的 RAG 由哪几部分组成？", "What are the components of RAG taught in Week 10?"),
]

CSS = """
<style>
/* user turns on the right, as a grey bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  flex-direction: row-reverse; background: transparent;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
  [data-testid="stChatMessageContent"] {
  flex: 0 1 auto; background: #F2F4F7; border-radius: 16px 16px 4px 16px; padding: 10px 16px;
}
/* citation rows read left to right like a list */
[class*="st-key-cites_"] button { justify-content: flex-start; text-align: left; }
[class*="st-key-cites_"] button p { font-size: 13px; }
.src-tr { font-size: 13px; line-height: 1.55; padding: 4px 10px; border-radius: 4px; }
.src-tr.hit { background: #FFF7D6; }
.src-tr .ts { font-family: ui-monospace, monospace; font-size: 12px; color: #2B59C3; margin-right: 10px; }
.src-tr.off { color: #59636E; }
/* headings inside an answer are section labels, not page titles */
[data-testid="stChatMessageContent"] h1,
[data-testid="stChatMessageContent"] h2,
[data-testid="stChatMessageContent"] h3,
[data-testid="stChatMessageContent"] h4 {
  font-size: 1.02rem; font-weight: 600; line-height: 1.45; padding: 0.7rem 0 0.2rem; margin: 0;
}
[data-testid="stChatMessageContent"] h1 { font-size: 1.12rem; }
/* no anchor-link icons next to headings */
[data-testid="stHeaderActionElements"] { display: none; }
[data-testid="stMainBlockContainer"] { padding: 2.6rem 2.5rem 1rem; max-width: 1600px; }
/* knowledge-map frame sits flush; the rounded dark canvas is inside it */
[data-testid="stIFrame"], [data-testid="stCustomComponentV1"] { border: 0; border-radius: 12px; }
</style>
"""


# ----------------------------------------------------------------- resources


@st.cache_resource
def get_store() -> ChunkStore:
    return ChunkStore(settings.index_path, read_only=True)


@st.cache_resource
def get_retriever(kind: str):
    return build_retriever(get_store(), kind)


@st.cache_resource
def get_llm(thinking: str):
    return build_llm(thinking=thinking)


@st.cache_data
def weeks() -> list[str]:
    rows = get_store().conn.execute(
        "SELECT DISTINCT week FROM chunks WHERE week IS NOT NULL ORDER BY week"
    ).fetchall()
    return [r[0] for r in rows]


@st.cache_data
def load_all_chunks(week: str | None = None) -> list[Chunk]:
    chunks = get_store().all_chunks()
    if week is not None:
        chunks = [c for c in chunks if c.week == week]
    return chunks


@st.cache_resource(show_spinner=False)
def get_corpus() -> Corpus:
    """Concept counts for every material; scanned once and shared by all maps."""
    return scan_corpus(load_all_chunks(None))


def _map_stats(m: CourseMap) -> tuple[int, int, int]:
    n_concepts = sum(1 for n in m.nodes if n.group == "concept")
    n_materials = sum(1 for n in m.nodes if n.group not in {"concept", "query"})
    return n_materials, n_concepts, len(m.edges)


@st.cache_resource(show_spinner=False)
def course_map_html(week: str | None = None, height: int = KG_HEIGHT) -> tuple[str, tuple[int, int, int]]:
    """Whole-course knowledge map: materials linked through the concepts they cover."""
    m = build_course_map(get_corpus(), week=week)
    return to_obsidian_html(m, height=height), _map_stats(m)


@st.cache_data(show_spinner=False)
def query_map_html(
    query: str, chunk_ids: tuple[str, ...], _chunks: list[Chunk], height: int = KG_HEIGHT
) -> tuple[str, tuple[int, int, int]]:
    """Map for one question: cited materials, their concepts, and related materials."""
    m = build_query_map(query, _chunks, get_corpus())
    return to_obsidian_html(m, height=height), _map_stats(m)


@st.cache_data(show_spinner=False)
def triple_map_html(
    chunk_ids: tuple[str, ...], _chunks: list[Chunk], llm_name: str, _llm, height: int = KG_HEIGHT
) -> tuple[str, tuple[int, int, int]]:
    """Typed concept relations the LLM extracts from the retrieved chunks (cached per question)."""
    m = triples_to_map(extract_triples_llm(_chunks, llm=_llm))
    return to_obsidian_html(m, height=height), _map_stats(m)


@st.cache_data
def page_png(path: str, page: int) -> bytes | None:
    from pathlib import Path

    return src.pdf_page_png(Path(path), page)


@st.cache_data
def page_count(path: str) -> int | None:
    from pathlib import Path

    return src.pdf_page_count(Path(path))


@st.cache_data
def nb_html(path: str, cell_idx: int) -> str:
    from pathlib import Path

    return src.notebook_html(Path(path), cell_idx)


# --------------------------------------------------------------------- state


def init_state() -> None:
    ss = st.session_state
    ss.setdefault("sid", uuid.uuid4().hex)
    ss.setdefault("messages", [])
    ss.setdefault("active", None)  # (message index, hit index)
    ss.setdefault("pdf_page", 1)
    ss.setdefault("kg_query", "")
    ss.setdefault("kg_chunks", [])
    ss.setdefault("kg_top_k_used", None)
    ss.setdefault("kg_view", "auto")  # "auto" | "global" | "local"
    ss.setdefault("map_wide", False)  # map expanded to most of the page
    if "memory" not in ss:
        ss.memory = ConversationMemory(settings.memory_path, session_id=ss.sid)


def new_chat() -> None:
    ss = st.session_state
    ss.memory.clear()
    ss.messages = []
    ss.active = None
    ss.kg_query = ""
    ss.kg_chunks = []
    ss.kg_top_k_used = None
    ss.kg_view = "auto"


def open_cite(i: int, j: int) -> None:
    ss = st.session_state
    ss.active = (i, j)
    chunk = ss.messages[i]["hits"][j].chunk
    ss.pdf_page = int(chunk.metadata.get("page") or chunk.position or 1)


def close_panel() -> None:
    st.session_state.active = None


def flip_page(delta: int) -> None:
    st.session_state.pdf_page += delta


def ask(question: str) -> None:
    st.session_state.pending = question


def active_hit() -> Hit | None:
    ss = st.session_state
    if ss.active is None:
        return None
    i, j = ss.active
    try:
        return ss.messages[i]["hits"][j]
    except (IndexError, KeyError):
        ss.active = None
        return None


# ------------------------------------------------------------------- sidebar


def sidebar() -> dict:
    with st.sidebar:
        st.markdown(tr("### EE6405 学习助手", "### EE6405 Study Assistant"))
        st.caption(tr("Table 8 · NLP 课程问答", "Table 8 · NLP course Q&A"))
        st.button(tr("新对话", "New chat"), icon=":material/add:", on_click=new_chat, width="stretch")

        st.markdown(tr("##### 检索设置", "##### Retrieval"))
        kind = st.segmented_control(
            tr("检索方式", "Retriever"), list(RETRIEVERS), format_func=RETRIEVERS.get, default="bm25",
            required=True, width="stretch",
        )
        if kind in NEEDS_ML and not HAS_ML:
            st.warning(tr("语义 / 混合检索需要先安装 requirements-ml.txt，本次改用 BM25。",
                          "Dense / hybrid retrieval needs requirements-ml.txt; using BM25 instead."))
            kind = "bm25"
        week = st.selectbox(tr("周次", "Week"), [ALL_WEEKS, *weeks()])
        groups = st.pills(
            tr("资料类型", "Material"), list(MODALITY_GROUPS), selection_mode="multi",
            default=list(MODALITY_GROUPS),
        )

        st.markdown(tr("##### 知识地图", "##### Knowledge map"))
        kg_method = st.segmented_control(
            tr("提问后的视图", "View after a question"),
            ["materials", "llm"],
            format_func=lambda x: tr("资料关联", "Materials") if x == "materials" else tr("概念关系", "Relations"),
            default="materials",
            required=True,
            width="stretch",
            help=tr("资料关联：被引用的资料、涉及的概念和相关资料。概念关系：由 LLM 从检索到的片段里抽取三元组。",
                    "Materials: cited materials, their concepts and related materials. "
                    "Relations: triples the LLM extracts from the retrieved chunks."),
        )
        kg_top_k = st.slider(
            tr("用于构图的片段数", "Chunks used for the map"), min_value=3, max_value=15, value=6,
            help=tr("取检索结果里最相关的 K 个片段来画当前问题的地图",
                    "The map for a question is drawn from its K most relevant chunks"),
        )

        st.markdown(tr("##### 模型", "##### Model"))
        thinking = st.toggle(
            tr("思考模式", "Thinking mode"), value=settings.llm_thinking == "enabled",
            help=tr("更慢，推理更充分", "Slower, reasons more"),
        )
        llm = get_llm("enabled" if thinking else "disabled")
        if isinstance(llm, EchoLLM):
            st.warning(tr("没有配置 API key，当前是离线模式，只会回显提示词。",
                          "No API key configured: offline mode, the prompt is echoed back."))
        else:
            st.caption(f"`{llm.name}`")

        st.divider()
        st.caption(tr(f"索引：{get_store().count():,} 个片段", f"Index: {get_store().count():,} chunks"))

    modality = None
    if groups and len(groups) < len(MODALITY_GROUPS):
        modality = [m for g in groups for m in MODALITY_GROUPS[g]]
    return {
        "retriever": kind,
        "week": None if week == ALL_WEEKS else week,
        "modality": modality,
        "llm": llm,
        "kg_method": kg_method,
        "kg_top_k": kg_top_k,
    }


# ---------------------------------------------------------------------- chat


def render_message(i: int, m: dict) -> None:
    if m["role"] == "user":
        with st.chat_message("user", avatar=":material/person:"):
            st.markdown(m["content"])
        return
    with st.chat_message("assistant", avatar=":material/menu_book:"):
        hits = m.get("hits", [])
        if m.get("error"):
            st.error(m["content"])
            return
        st.markdown(src.badge_citations(m["content"], len(hits)))
        st.caption(m.get("meta", ""))
        if not hits:
            return
        with st.container(key=f"cites_{i}", gap=None):
            st.caption(tr(f"引用来源 · {len(hits)} 条 · 点击在右侧查看原文",
                          f"Sources · {len(hits)} · click one to open it on the right"))
            for j, h in enumerate(hits):
                c = h.chunk
                label = (
                    f"**{j + 1}** · {src.kind_of(c).label} · {src.display_name(c)}"
                    f" · {c.week or '—'} {src.location_label(c)}"
                )
                current = st.session_state.active == (i, j)
                st.button(
                    label, key=f"cite_{i}_{j}", on_click=open_cite, args=(i, j),
                    type="primary" if current else "secondary", width="stretch",
                )


def answer(question: str, opts: dict) -> dict:
    retriever = get_retriever(opts["retriever"])
    assistant = StudyAssistant(
        retriever=retriever, llm=opts["llm"], memory=st.session_state.memory
    )
    t0 = time.time()
    try:
        ans = assistant.answer(question, week=opts["week"], modality=opts["modality"])
    except Exception as e:  # network, auth, rate limit ...
        return {"role": "assistant", "error": True, "hits": [],
                "content": tr(f"请求失败（{type(e).__name__}），请稍后重试。",
                              f"Request failed ({type(e).__name__}). Please try again.")}

    kg_k = opts["kg_top_k"]
    if ans.hits and kg_k <= len(ans.hits):
        kg_chunks = [h.chunk for h in ans.hits[:kg_k]]
    else:
        kg_hits = retriever.retrieve(
            question, top_k=kg_k, week=opts["week"], modality=opts["modality"]
        )
        kg_chunks = [h.chunk for h in kg_hits]
        
    meta = f"{ans.model} · {RETRIEVERS[opts['retriever']]} · {time.time() - t0:.1f} {tr('秒', 's')}"
    return {"role": "assistant", "content": ans.text, "hits": ans.hits, "meta": meta, "kg_chunks": kg_chunks}


# --------------------------------------------------------------------- panel


def fallback_text(chunk, note: str) -> None:
    st.info(note, icon=":material/description:")
    with st.container(border=True):
        st.markdown(chunk.text)


def panel_pdf(chunk) -> bool:
    path = src.original_file(chunk)
    if path is None or path.suffix.lower() != ".pdf":
        return False
    total = page_count(str(path))
    cited = int(chunk.metadata.get("page") or chunk.position or 1)
    ss = st.session_state
    ss.pdf_page = max(1, min(ss.pdf_page, total or ss.pdf_page))
    png = page_png(str(path), ss.pdf_page)
    if png is None:
        return False
    st.image(png, width="stretch")
    prev_col, mid, next_col = st.columns([1, 2, 1], vertical_alignment="center")
    prev_col.button(tr("上一页", "Previous"), icon=":material/chevron_left:", on_click=flip_page, args=(-1,),
                    disabled=ss.pdf_page <= 1, width="stretch")
    mid.markdown(
        "<div style='text-align:center;font-size:13px;color:#59636E'>"
        + tr(f"第 {ss.pdf_page} / {total} 页", f"Page {ss.pdf_page} of {total}")
        + (tr(" · 被引用的页", " · cited page") if ss.pdf_page == cited else "")
        + "</div>",
        unsafe_allow_html=True,
    )
    next_col.button(tr("下一页", "Next"), icon=":material/chevron_right:", icon_position="right",
                    on_click=flip_page, args=(1,), disabled=bool(total) and ss.pdf_page >= total,
                    width="stretch")
    if ss.pdf_page != cited:
        st.button(tr(f"回到被引用的第 {cited} 页", f"Back to cited page {cited}"), on_click=lambda: ss.update(pdf_page=cited),
                  type="tertiary")
    with st.expander(tr("检索命中的文字", "Text that was retrieved")):
        st.markdown(chunk.text)
    st.download_button(tr("下载 PDF", "Download PDF"), path.read_bytes(), file_name=path.name,
                       mime="application/pdf", icon=":material/download:", width="stretch")
    return True


def panel_video(chunk) -> None:
    vp = src.video_path(chunk)
    start = int(float(chunk.metadata.get("start", 0) or 0))
    if vp is not None:
        st.video(str(vp), start_time=start)
    else:
        st.info(tr("本地没有原始视频，只显示转写文字。", "The video file is not on this machine; showing the transcript."), icon=":material/videocam_off:")
    rows = src.transcript_window(chunk)
    if not rows:
        fallback_text(chunk, tr("找不到转写文件，显示索引里的文字。", "Transcript file not found; showing the indexed text."))
        return
    st.markdown(tr("**转写原文**　<span style='color:#59636E;font-size:13px'>黄色是被引用的片段</span>",
                   "**Transcript**　<span style='color:#59636E;font-size:13px'>the cited passage is highlighted</span>"),
                unsafe_allow_html=True)
    lines = "".join(
        f"<div class='src-tr {'hit' if cited else 'off'}'><span class='ts'>{ts}</span>"
        f"{html.escape(text)}</div>"
        for ts, _, text, cited in rows
    )
    with st.container(height=380, border=True):
        st.markdown(lines, unsafe_allow_html=True)


def panel_notebook(chunk) -> bool:
    path = src.resolve(chunk.source)
    if path is None or path.suffix != ".ipynb":
        return False
    idx = int(chunk.metadata.get("cell_idx", chunk.position) or 0)
    # nb_html escapes all notebook text; st.iframe runs it same-origin.
    st.iframe(nb_html(str(path), idx), height=640)
    st.download_button(tr("下载 .ipynb", "Download .ipynb"), path.read_bytes(), file_name=path.name,
                       mime="application/x-ipynb+json", icon=":material/download:",
                       width="stretch")
    return True


def render_panel(hit: Hit) -> None:
    c = hit.chunk
    i, j = st.session_state.active
    kind = src.kind_of(c)
    head, close = st.columns([5, 1], vertical_alignment="top")
    with head:
        st.badge(f"{j + 1} · {kind.label}", color=kind.color)
        st.markdown(f"**{src.display_name(c)}**")
        st.caption(f"{c.week or '—'} · {src.location_label(c)}")
    close.button(tr("关闭", "Close"), icon=":material/close:", on_click=close_panel, width="stretch")

    if c.modality in {"slides", "assessment_pdf", "misc"}:
        if not panel_pdf(c):
            fallback_text(c, tr("本地没有原始文件，下面是提取的文字。", "The original file is not on this machine; showing the extracted text."))
    elif c.modality == "transcript":
        panel_video(c)
    elif c.modality == "notebook":
        if not panel_notebook(c):
            fallback_text(c, tr("本地没有原始 notebook，下面是被引用的单元格。", "The notebook is not on this machine; showing the cited cell."))
    else:
        with st.container(border=True):
            st.markdown(c.text)

def render_kg_panel(opts: dict) -> None:
    ss = st.session_state
    if not ss.kg_query and ss.messages:
        last_u = next((m["content"] for m in reversed(ss.messages) if m["role"] == "user"), "")
        last_a = next(
            (m for m in reversed(ss.messages) if m["role"] == "assistant" and not m.get("error")),
            None,
        )
        if last_u and last_a:
            ss.kg_query = last_u
            ss.kg_chunks = last_a.get("kg_chunks") or [h.chunk for h in last_a.get("hits", [])]
            ss.kg_top_k_used = None

    has_local = bool(ss.kg_query)
    mode = "local" if (has_local and ss.kg_view != "global") else "global"

    # Sidebar settings changed after the question: re-retrieve so the map follows them.
    cur_params = (opts["kg_top_k"], opts["retriever"], opts["week"], tuple(opts["modality"] or ()))
    if mode == "local" and ss.kg_top_k_used != cur_params:
        retriever = get_retriever(opts["retriever"])
        kg_hits = retriever.retrieve(
            ss.kg_query, top_k=opts["kg_top_k"], week=opts["week"], modality=opts["modality"]
        )
        ss.kg_chunks = [h.chunk for h in kg_hits]
        ss.kg_top_k_used = cur_params

    height = KG_HEIGHT_WIDE if ss.map_wide else KG_HEIGHT
    head_col, wide_col, toggle_col = st.columns(
        [6, 2, 3] if ss.map_wide else [5, 3, 4], vertical_alignment="center"
    )
    with wide_col:
        if ss.map_wide:
            if st.button(tr("收起", "Collapse"), icon=":material/close_fullscreen:", width="stretch"):
                ss.map_wide = False
                st.rerun()
        elif st.button(tr("放大", "Expand"), icon=":material/open_in_full:", width="stretch"):
            ss.map_wide = True
            st.rerun()
    with head_col:
        st.markdown(tr("##### 课程知识地图", "##### Course knowledge map") if mode == "global"
                    else tr("##### 本次问答的知识地图", "##### Map of this answer"))
    with toggle_col:
        if has_local and mode == "local":
            if st.button(tr("全课程地图", "Whole course"), icon=":material/hub:", width="stretch"):
                ss.kg_view = "global"
                st.rerun()
        elif has_local:
            if st.button(tr("回到本次问答", "Back to this answer"), icon=":material/arrow_back:", width="stretch"):
                ss.kg_view = "local"
                st.rerun()

    note = ""
    if mode == "global":
        with st.spinner(tr("正在生成课程知识地图…", "Building the course map…")):
            html_str, stats = course_map_html(opts["week"], height)
        scope = opts["week"] or ALL_WEEKS
        what = tr("资料通过共同涉及的概念相连", "materials are linked through the concepts they share")
    else:
        if not ss.kg_chunks:
            st.info(tr("这个问题没有检索到课程片段，暂时画不出地图。", "Nothing was retrieved for this question, so there is no map to draw."), icon=":material/info:")
            return
        chunk_ids = tuple(c.chunk_id for c in ss.kg_chunks)
        llm = opts["llm"]
        use_llm = opts["kg_method"] == "llm"
        if use_llm and isinstance(llm, EchoLLM):
            note = tr("没有配置 API key，无法抽取概念关系，已改为显示资料关联。",
                      "No API key, so relations cannot be extracted; showing materials instead.")
            use_llm = False
        if use_llm:
            with st.spinner(tr("正在让模型抽取概念关系…", "Extracting relations with the model…")):
                html_str, stats = triple_map_html(
                    chunk_ids, ss.kg_chunks, getattr(llm, "name", ""), llm, height
                )
            if stats[1] == 0:
                note = tr("模型没有抽出可用的概念关系，已改为显示资料关联。",
                          "The model returned no usable relations; showing materials instead.")
                use_llm = False
        if not use_llm:
            html_str, stats = query_map_html(ss.kg_query, chunk_ids, ss.kg_chunks, height)
        scope = tr(f"{len(ss.kg_chunks)} 个检索片段", f"{len(ss.kg_chunks)} retrieved chunks")
        what = (tr("模型抽取的概念关系", "relations extracted by the model") if use_llm
                else tr("紫色圆环是被引用的资料，外圈是讲到同一概念的其他资料",
                        "ringed = cited in the answer; outer nodes = other materials on the same concepts"))

    n_materials, n_concepts, n_edges = stats
    if n_concepts == 0:
        st.info(tr("检索到的片段里没有识别出课程概念，可以换个问法或调大片段数。", "No course concept was recognised in the retrieved chunks. Try rephrasing or using more chunks."), icon=":material/info:")
        return
    counts = tr(f"{n_concepts} 个概念 · {n_edges} 条关联", f"{n_concepts} concepts · {n_edges} links")
    if n_materials:
        counts = tr(f"{n_materials} 份资料 · {counts}", f"{n_materials} materials · {counts}")
    st.caption(f"{scope} · {counts}　|　{what}")
    if note:
        st.caption(f":material/info: {note}")
    components.html(html_str, height=height + 4, scrolling=False)


# ---------------------------------------------------------------------- page


def main() -> None:
    st.html(CSS)
    init_state()
    opts = sidebar()
    ss = st.session_state

    hit = active_hit()
    # The expanded map takes most of the page; a source panel always uses the normal split.
    wide = st.session_state.map_wide and hit is None
    chat_col, panel_col = st.columns([5, 11] if wide else [11, 9], gap="large")

    with chat_col:
        first = next((m["content"] for m in ss.messages if m["role"] == "user"), None)
        st.markdown(f"#### {first[:60] if first else tr('新对话', 'New chat')}")
        st.caption(tr("回答只依据课程资料，并标注出处", "Answers come from the course material, with sources"))
        for i, m in enumerate(ss.messages):
            render_message(i, m)
        if not ss.messages:
            st.markdown(tr("**试试这些问题**", "**Try one of these**"))
            for q in EXAMPLES:
                st.button(q, on_click=ask, args=(q,), type="tertiary")

    placeholder = tr("问点 EE6405 的问题，中英文都可以…", "Ask anything about EE6405, in English or Chinese…")
    prompt = chat_col.chat_input(placeholder)
    prompt = prompt or ss.pop("pending", None)

    if prompt:
        with chat_col:
            render_message(len(ss.messages), {"role": "user", "content": prompt})
            with st.chat_message("assistant", avatar=":material/menu_book:"):
                with st.spinner(tr("正在检索课程资料并生成回答…", "Searching the course material and writing an answer…")):
                    reply = answer(prompt, opts)
        ss.messages += [{"role": "user", "content": prompt}, reply]
        if not reply.get("error"):
            ss.kg_query = prompt
            ss.kg_chunks = reply.get("kg_chunks", [])
            ss.kg_top_k_used = (
                opts["kg_top_k"], opts["retriever"], opts["week"], tuple(opts["modality"] or ())
            )
            ss.kg_view = "auto"
            ss.active = None  # 新提问后自动切回动态子图
        st.rerun()

    with panel_col:
        if hit is not None:
            render_panel(hit)
        else:
            render_kg_panel(opts)


main()