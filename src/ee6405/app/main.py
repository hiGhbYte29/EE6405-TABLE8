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
from ee6405.kg.extract import extract_triples_cooccurrence, extract_triples_llm
from ee6405.kg.graph import build_graph, to_pyvis_html, top_nodes
from ee6405.rag.retriever import build_retriever
from ee6405.rag.store import ChunkStore
from ee6405.types import Hit, Chunk

st.set_page_config(page_title="EE6405 学习助手", page_icon=":material/menu_book:", layout="wide")

RETRIEVERS = {"bm25": "BM25", "dense": "语义", "hybrid": "混合"}
NEEDS_ML = {"dense", "hybrid"}
HAS_ML = importlib.util.find_spec("sentence_transformers") is not None
MODALITY_GROUPS = {
    "课件": ["slides"],
    "视频转写": ["transcript"],
    "代码": ["notebook"],
    "测验": ["mcq", "assessment_pdf"],
    "其他": ["misc", "clarification"],
}
EXAMPLES = [
    "What is the difference between BERT and RoBERTa?",
    "TF-IDF 是怎么计算的？",
    "第 10 周讲的 RAG 由哪几部分组成？",
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
def build_global_graph_html(
    week: str | None = None,
) -> tuple[str, list[tuple[str, float]], int, int]:
    """全课程简单图谱：基于统计共现，缓存后秒加载。"""
    chunks = load_all_chunks(week)
    triples = extract_triples_cooccurrence(chunks, top_terms=20, min_count=2)
    g = build_graph(triples)
    html_str = to_pyvis_html(g, height="550px")
    return html_str, top_nodes(g, n=8), g.number_of_nodes(), g.number_of_edges()


@st.cache_data(show_spinner=False)
def build_local_graph_html(
    chunk_ids: tuple[str, ...],
    _chunks: list[Chunk],
    method: str,
    llm_name: str,
    is_echo: bool,
    _llm,
) -> tuple[str, list[tuple[str, float]], int, int, str]:
    """针对当前提问召回的 Top-K chunks 构建动态局部子图（带缓存，切换面板不重复调 LLM）。"""
    note = ""
    if method == "llm":
        if is_echo:
            note = "未配置 API Key（离线模式），已自动回退为统计共现。"
            triples = extract_triples_cooccurrence(_chunks, top_terms=30, min_count=1)
        else:
            triples = extract_triples_llm(_chunks, llm=_llm)
            if not triples:
                note = "LLM 未抽出有效三元组，已回退为统计共现。"
                triples = extract_triples_cooccurrence(_chunks, top_terms=30, min_count=1)
    else:
        triples = extract_triples_cooccurrence(_chunks, top_terms=30, min_count=2)
        if len(triples) < 3:
            # 局部片段较少时放宽共现门槛，防止子图为空
            triples = extract_triples_cooccurrence(_chunks, top_terms=30, min_count=1)

    g = build_graph(triples)
    html_str = to_pyvis_html(g, height="550px")
    return html_str, top_nodes(g, n=8), g.number_of_nodes(), g.number_of_edges(), note

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
        st.markdown("### EE6405 学习助手")
        st.caption("Table 8 · NLP 课程问答")
        st.button("新对话", icon=":material/add:", on_click=new_chat, width="stretch")

        st.markdown("##### 检索设置")
        kind = st.segmented_control(
            "检索方式", list(RETRIEVERS), format_func=RETRIEVERS.get, default="bm25",
            required=True, width="stretch",
        )
        if kind in NEEDS_ML and not HAS_ML:
            st.warning("语义 / 混合检索需要先安装 requirements-ml.txt，本次改用 BM25。")
            kind = "bm25"
        week = st.selectbox("周次", ["全部周次", *weeks()])
        groups = st.pills(
            "资料类型", list(MODALITY_GROUPS), selection_mode="multi",
            default=list(MODALITY_GROUPS),
        )

        st.markdown("##### 知识图谱")
        kg_method = st.segmented_control(
            "三元组抽取",
            ["cooccurrence", "llm"],
            format_func=lambda x: "统计共现" if x == "cooccurrence" else "LLM 抽取",
            default="cooccurrence",
            required=True,
            width="stretch",
        )
        kg_top_k = st.slider(
            "局部图谱召回 Top-K", min_value=3, max_value=15, value=6,
            help="向 RAG 检索最相关的 K 个片段用于构建当前问题的动态子图",
        )
        
        st.markdown("##### 模型")
        thinking = st.toggle(
            "思考模式", value=settings.llm_thinking == "enabled", help="更慢，推理更充分"
        )
        llm = get_llm("enabled" if thinking else "disabled")
        if isinstance(llm, EchoLLM):
            st.warning("没有配置 API key，当前是离线模式，只会回显提示词。")
        else:
            st.caption(f"`{llm.name}`")

        st.divider()
        st.caption(f"索引：{get_store().count():,} 个片段")

    modality = None
    if groups and len(groups) < len(MODALITY_GROUPS):
        modality = [m for g in groups for m in MODALITY_GROUPS[g]]
    return {
        "retriever": kind,
        "week": None if week == "全部周次" else week,
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
            st.caption(f"引用来源 · {len(hits)} 条 · 点击在右侧查看原文")
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
                "content": f"请求失败（{type(e).__name__}），请稍后重试。"}

    kg_k = opts["kg_top_k"]
    if ans.hits and kg_k <= len(ans.hits):
        kg_chunks = [h.chunk for h in ans.hits[:kg_k]]
    else:
        kg_hits = retriever.retrieve(
            question, top_k=kg_k, week=opts["week"], modality=opts["modality"]
        )
        kg_chunks = [h.chunk for h in kg_hits]
        
    meta = f"{ans.model} · {RETRIEVERS[opts['retriever']]} · {time.time() - t0:.1f} 秒"
    return {"role": "assistant", "content": ans.text, "hits": ans.hits, "meta": meta, "kg_chunks": kg_chunks}


# --------------------------------------------------------------------- panel


def fallback_text(chunk, note: str) -> None:
    st.info(note, icon=":material/description:")
    with st.container(border=True):
        st.markdown(chunk.text)


def panel_pdf(chunk) -> bool:
    path = src.resolve(chunk.source)
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
    prev_col.button("上一页", icon=":material/chevron_left:", on_click=flip_page, args=(-1,),
                    disabled=ss.pdf_page <= 1, width="stretch")
    mid.markdown(
        f"<div style='text-align:center;font-size:13px;color:#59636E'>第 {ss.pdf_page} / {total} 页"
        f"{' · 被引用的页' if ss.pdf_page == cited else ''}</div>",
        unsafe_allow_html=True,
    )
    next_col.button("下一页", icon=":material/chevron_right:", icon_position="right",
                    on_click=flip_page, args=(1,), disabled=bool(total) and ss.pdf_page >= total,
                    width="stretch")
    if ss.pdf_page != cited:
        st.button(f"回到被引用的第 {cited} 页", on_click=lambda: ss.update(pdf_page=cited),
                  type="tertiary")
    with st.expander("检索命中的文字"):
        st.markdown(chunk.text)
    st.download_button("下载 PDF", path.read_bytes(), file_name=path.name,
                       mime="application/pdf", icon=":material/download:", width="stretch")
    return True


def panel_video(chunk) -> None:
    vp = src.video_path(chunk)
    start = int(float(chunk.metadata.get("start", 0) or 0))
    if vp is not None:
        st.video(str(vp), start_time=start)
    else:
        st.info("本地没有原始视频，只显示转写文字。", icon=":material/videocam_off:")
    rows = src.transcript_window(chunk)
    if not rows:
        fallback_text(chunk, "找不到转写文件，显示索引里的文字。")
        return
    st.markdown("**转写原文**　<span style='color:#59636E;font-size:13px'>黄色是被引用的片段</span>",
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
    st.download_button("下载 .ipynb", path.read_bytes(), file_name=path.name,
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
    close.button("关闭", icon=":material/close:", on_click=close_panel, width="stretch")

    if c.modality in {"slides", "assessment_pdf", "misc"}:
        if not panel_pdf(c):
            fallback_text(c, "本地没有原始文件，下面是提取的文字。")
    elif c.modality == "transcript":
        panel_video(c)
    elif c.modality == "notebook":
        if not panel_notebook(c):
            fallback_text(c, "本地没有原始 notebook，下面是被引用的单元格。")
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

    # 若用户在提问后拖动了侧边栏 Top-K 滑块/检索方式/周次，实时按新条件重召回子图
    cur_params = (opts["kg_top_k"], opts["retriever"], opts["week"], tuple(opts["modality"] or ()))
    if mode == "local" and ss.kg_top_k_used != cur_params:
        retriever = get_retriever(opts["retriever"])
        kg_hits = retriever.retrieve(
            ss.kg_query, top_k=opts["kg_top_k"], week=opts["week"], modality=opts["modality"]
        )
        ss.kg_chunks = [h.chunk for h in kg_hits]
        ss.kg_top_k_used = cur_params

    head_col, toggle_col = st.columns([3, 2], vertical_alignment="center")
    with head_col:
        if mode == "local":
            st.markdown("##### 动态局部知识图谱")
            q_preview = ss.kg_query[:26] + ("…" if len(ss.kg_query) > 26 else "")
            st.caption(f"问题：「{q_preview}」· {len(ss.kg_chunks)} 个相关片段")
        else:
            st.markdown("##### 全课程知识图谱")
            st.caption(f"范围：{opts['week'] or '全部周次'} · 统计共现全局概览")

    with toggle_col:
        if has_local:
            if mode == "local":
                if st.button("查看全课程图谱", width="stretch"):
                    ss.kg_view = "global"
                    st.rerun()
            else:
                if st.button("回到当前问答子图", type="primary", width="stretch"):
                    ss.kg_view = "local"
                    st.rerun()

    if mode == "global":
        with st.spinner("正在构建全课程图谱…"):
            html_str, top_concepts, n_nodes, n_edges = build_global_graph_html(opts["week"])
        note = ""
    else:
        if not ss.kg_chunks:
            st.warning("当前问题未召回到相关课程片段，无法构建局部子图。")
            return
        llm = opts["llm"]
        chunk_ids = tuple(c.chunk_id for c in ss.kg_chunks)
        with st.spinner("正在提取三元组并构建局部子图…"):
            html_str, top_concepts, n_nodes, n_edges, note = build_local_graph_html(
                chunk_ids=chunk_ids,
                _chunks=ss.kg_chunks,
                method=opts["kg_method"],
                llm_name=getattr(llm, "name", "unknown"),
                is_echo=isinstance(llm, EchoLLM),
                _llm=llm,
            )

    if note:
        st.info(note, icon=":material/info:")

    if n_nodes == 0:
        st.warning("未能从当前片段提取出实体关系，可以尝试调大 Top-K 或更换问题。")
        return

    st.caption(f"节点：{n_nodes} · 关系边：{n_edges}　|　鼠标悬停节点/连线可查看详情与出处")
    components.html(html_str, height=570, scrolling=False)

    if top_concepts:
        with st.expander("核心概念排行 (Degree Centrality)"):
            st.markdown(" · ".join(f"`{name}`" for name, _ in top_concepts))

# ---------------------------------------------------------------------- page


def main() -> None:
    st.html(CSS)
    init_state()
    opts = sidebar()
    ss = st.session_state

    hit = active_hit()
    chat_col, panel_col = st.columns([3, 2], gap="large")

    with chat_col:
        first = next((m["content"] for m in ss.messages if m["role"] == "user"), None)
        st.markdown(f"#### {first[:60] if first else '新对话'}")
        st.caption("回答只依据课程资料，并标注出处")
        for i, m in enumerate(ss.messages):
            render_message(i, m)
        if not ss.messages:
            st.markdown("**试试这些问题**")
            for q in EXAMPLES:
                st.button(q, on_click=ask, args=(q,), type="tertiary")

    placeholder = "问点 EE6405 的问题，中英文都可以…"
    prompt = chat_col.chat_input(placeholder)
    prompt = prompt or ss.pop("pending", None)

    if prompt:
        with chat_col:
            render_message(len(ss.messages), {"role": "user", "content": prompt})
            with st.chat_message("assistant", avatar=":material/menu_book:"):
                with st.spinner("正在检索课程资料并生成回答…"):
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