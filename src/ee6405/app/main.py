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

from ee6405.agent.chain import StudyAssistant
from ee6405.agent.llm import EchoLLM, build_llm
from ee6405.agent.memory import ConversationMemory
from ee6405.app import sources as src
from ee6405.config import settings
from ee6405.rag.retriever import build_retriever
from ee6405.rag.store import ChunkStore
from ee6405.types import Hit

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
    if "memory" not in ss:
        ss.memory = ConversationMemory(settings.memory_path, session_id=ss.sid)


def new_chat() -> None:
    ss = st.session_state
    ss.memory.clear()
    ss.messages = []
    ss.active = None


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
    meta = f"{ans.model} · {RETRIEVERS[opts['retriever']]} · {time.time() - t0:.1f} 秒"
    return {"role": "assistant", "content": ans.text, "hits": ans.hits, "meta": meta}


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


# ---------------------------------------------------------------------- page


def main() -> None:
    st.html(CSS)
    init_state()
    opts = sidebar()
    ss = st.session_state

    hit = active_hit()
    if hit is not None:
        chat_col, panel_col = st.columns([3, 2], gap="large")
    else:
        chat_col, panel_col = st.container(), None

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
    prompt = chat_col.chat_input(placeholder) if panel_col else st.chat_input(placeholder)
    prompt = prompt or ss.pop("pending", None)

    if prompt:
        with chat_col:
            render_message(len(ss.messages), {"role": "user", "content": prompt})
            with st.chat_message("assistant", avatar=":material/menu_book:"):
                with st.spinner("正在检索课程资料并生成回答…"):
                    reply = answer(prompt, opts)
        ss.messages += [{"role": "user", "content": prompt}, reply]
        st.rerun()

    if panel_col is not None and hit is not None:
        with panel_col:
            render_panel(hit)


main()
