"""Render a :class:`CourseMap` as an Obsidian-style interactive graph.

One self-contained HTML document for ``st.components.v1.html``: dark canvas,
small dots sized by how connected they are, hairline edges, labels that appear
as you zoom in, and hover / click that lights up a node's neighbourhood while
the rest of the map fades back.

vis-network is inlined from the copy that ships inside ``pyvis`` (already a
dependency), so the map also draws with no internet connection.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from ee6405.i18n import tr
from ee6405.kg.course_map import CourseMap

# group -> (colour on the dark canvas, legend label)
GROUPS: dict[str, tuple[str, str]] = {
    "concept": ("#D7D5CF", tr("概念", "Concept")),
    "slides": ("#5B9CF5", tr("课件", "Slides")),
    "video": ("#F0A04B", tr("视频转写", "Lecture video")),
    "code": ("#57C785", tr("代码", "Code")),
    "quiz": ("#E879A6", tr("测验", "Quiz")),
    "other": ("#8A8F98", tr("其他", "Other")),
    "query": ("#A882FF", tr("当前问题", "Your question")),
}


@lru_cache(maxsize=1)
def _vis_js() -> str:
    import pyvis

    lib = Path(pyvis.__file__).parent / "lib"
    candidates = sorted(lib.glob("vis-*/vis-network.min.js"), reverse=True)
    if not candidates:
        raise RuntimeError("vis-network.min.js not found inside the pyvis package")
    return candidates[0].read_text(encoding="utf-8")


def _payload(course_map: CourseMap) -> dict:
    nodes = [
        {
            "id": n.id, "label": n.label, "group": n.group, "value": float(n.value),
            "week": n.week or "", "detail": n.detail, "ring": bool(n.highlight),
        }
        for n in course_map.nodes
    ]
    known = {n["id"] for n in nodes}
    edges = [
        {"from": e.source, "to": e.target, "w": float(e.weight), "label": e.label}
        for e in course_map.edges
        if e.source in known and e.target in known
    ]
    return {
        "nodes": nodes,
        "edges": edges,
        "groups": {g: {"color": c, "label": lab} for g, (c, lab) in GROUPS.items()},
        "t": {
            "linked": tr("相连的 {n} 个节点", "{n} linked nodes"),
            "more": tr("… 还有 {n} 个", "… {n} more"),
            "cited": tr("被引用", "Cited"),
        },
    }


def to_obsidian_html(course_map: CourseMap, height: int = 560) -> str:
    """Interactive HTML for the map. ``height`` is the canvas height in px."""
    # "</" must not appear inside the inline <script>.
    data = json.dumps(_payload(course_map), ensure_ascii=False).replace("</", "<\\/")
    return (
        _TEMPLATE.replace("__HEIGHT__", str(int(height)))
        .replace("__DATA__", data)
        .replace("__T_SEARCH__", tr("搜索节点…", "Search…"))
        .replace("__T_HINT__", tr("拖拽 · 滚轮缩放 · 点击节点", "Drag · scroll to zoom · click a node"))
        .replace("__VIS_JS__", _vis_js())
    )


_TEMPLATE = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<style>
  :root { --bg:#1E1E22; --panel:rgba(32,32,38,.92); --line:#34343C; --text:#DADADA; --muted:#8B8B95; --accent:#A882FF; }
  * { box-sizing: border-box; }
  html, body { margin:0; padding:0; background:transparent;
    font-family: Inter, "Noto Sans SC", "Microsoft YaHei", system-ui, -apple-system, "Segoe UI", sans-serif; }
  #wrap { position:relative; height:__HEIGHT__px; background:var(--bg); border-radius:12px; overflow:hidden; }
  #net { position:absolute; inset:0; }
  #search { position:absolute; top:12px; right:12px; width:170px; padding:6px 10px; font-size:12px;
    color:var(--text); background:var(--panel); border:1px solid var(--line); border-radius:8px; outline:none; }
  #search::placeholder { color:var(--muted); }
  #search:focus { border-color:var(--accent); }
  #legend { position:absolute; left:12px; bottom:10px; display:flex; flex-wrap:wrap; gap:4px 12px;
    max-width:calc(100% - 24px); font-size:11px; color:var(--muted); pointer-events:none; }
  #legend span { display:inline-flex; align-items:center; gap:5px; white-space:nowrap; }
  #legend i { width:8px; height:8px; border-radius:50%; display:inline-block; }
  #card { position:absolute; top:12px; left:12px; width:min(250px, calc(100% - 206px)); max-height:calc(100% - 60px);
    overflow:auto; padding:10px 12px; background:var(--panel); border:1px solid var(--line); border-radius:10px;
    color:var(--text); font-size:12px; line-height:1.5; display:none; }
  #card .kind { display:inline-flex; align-items:center; gap:5px; font-size:11px; color:var(--muted); }
  #card .kind i { width:8px; height:8px; border-radius:50%; display:inline-block; }
  #card h4 { margin:2px 0 2px; font-size:13.5px; font-weight:600; color:#F2F2F2; word-break:break-word; }
  #card .detail { color:var(--muted); margin-bottom:6px; word-break:break-word; }
  #card .sec { margin-top:6px; font-size:11px; color:var(--muted); }
  #card ul { list-style:none; margin:2px 0 0; padding:0; }
  #card li { display:flex; align-items:baseline; gap:6px; padding:1px 0; cursor:pointer; color:#CFCFD4; }
  #card li:hover { color:#fff; }
  #card li i { flex:0 0 6px; height:6px; border-radius:50%; transform:translateY(-1px); }
  #hint { position:absolute; right:12px; bottom:10px; font-size:11px; color:#5E5E68; pointer-events:none; }
  @media (max-width: 620px) { #hint { display:none; } }
  @media (max-width: 420px) { #search { width:130px; } #card { width:calc(100% - 24px); top:46px; } }
</style>
</head>
<body>
<div id="wrap">
  <div id="net"></div>
  <input id="search" type="text" placeholder="__T_SEARCH__" autocomplete="off" spellcheck="false">
  <div id="card"></div>
  <div id="legend"></div>
  <div id="hint">__T_HINT__</div>
</div>
<script>__VIS_JS__</script>
<script>
(function () {
  var DATA = __DATA__;
  var G = DATA.groups;
  var DIM_NODE = "rgba(120,120,130,0.16)", DIM_EDGE = "rgba(255,255,255,0.035)";
  var EDGE = "rgba(255,255,255,0.17)", EDGE_HOT = "rgba(168,130,255,0.85)";
  var LABEL = "#B9B9C0", LABEL_HOT = "#FFFFFF";

  var adj = {}, byId = {};
  DATA.nodes.forEach(function (n) { adj[n.id] = {}; byId[n.id] = n; });
  DATA.edges.forEach(function (e, i) { e.id = "e" + i; adj[e.from][e.to] = e; adj[e.to][e.from] = e; });

  var maxV = Math.max.apply(null, DATA.nodes.map(function (n) { return n.value; }).concat([1]));
  var big = DATA.nodes.length > 60;
  function size(n) {
    if (n.group === "query") return 13;
    return 5 + (big ? 13 : 7) * Math.sqrt(n.value / maxV);
  }
  // Most connected nodes keep their label at any zoom; the rest appear as you zoom in.
  var rank = {};
  DATA.nodes.slice().sort(function (a, b) { return b.value - a.value; })
      .forEach(function (n, i) { rank[n.id] = i; });
  var scaleNow = 1;
  // Labels stay a constant size on screen, so they are readable when zoomed out.
  function font(px, color) {
    return { color: color, size: Math.max(7, Math.min(34, px / scaleNow)),
             face: "Inter, 'Noto Sans SC', 'Microsoft YaHei', sans-serif",
             strokeWidth: 3, strokeColor: "#1E1E22", vadjust: 1 };
  }
  function baseNode(n) {
    var c = G[n.group] ? G[n.group].color : "#999";
    var big = n.group === "query" || n.ring;
    return {
      id: n.id, label: n.label, shape: "dot", size: size(n),
      color: { background: c, border: n.ring ? "#A882FF" : c,
               highlight: { background: c, border: "#FFFFFF" }, hover: { background: c, border: "#FFFFFF" } },
      borderWidth: n.ring ? 3 : 0, borderWidthSelected: 2,
      font: font(big ? 12.5 : 11, big ? LABEL_HOT : LABEL)
    };
  }
  function baseEdge(e) {
    return { id: e.id, from: e.from, to: e.to, color: { color: EDGE, highlight: EDGE_HOT, hover: EDGE_HOT, inherit: false },
             width: 0.7, hoverWidth: 0, selectionWidth: 0, smooth: false };
  }

  var nodes = new vis.DataSet(DATA.nodes.map(baseNode));
  var edges = new vis.DataSet(DATA.edges.map(baseEdge));
  var net = new vis.Network(document.getElementById("net"), { nodes: nodes, edges: edges }, {
    autoResize: true,
    interaction: { hover: true, tooltipDelay: 1e9, hideEdgesOnDrag: false, zoomSpeed: 0.6 },
    physics: {
      solver: "forceAtlas2Based",
      forceAtlas2Based: { gravitationalConstant: big ? -38 : -60, centralGravity: big ? 0.012 : 0.02,
                          springLength: big ? 70 : 95, springConstant: 0.07, damping: 0.55, avoidOverlap: 0.35 },
      maxVelocity: 40, minVelocity: 0.4, timestep: 0.45,
      stabilization: { enabled: true, iterations: big ? 260 : 180, fit: true }
    },
    layout: { improvedLayout: false, randomSeed: 7 },
    nodes: { chosen: false }, edges: { chosen: false }
  });

  window.__net = net;  // handle for scripted demos and tests

  // --- zoom-dependent labels ------------------------------------------------
  var labelsOn = null;
  function wantLabel(n, scale) {
    if (n.group === "query" || n.ring) return true;
    if (!big || rank[n.id] < 16) return true;
    return scale * size(n) > 9.5;          // everything once zoomed in
  }
  function refreshLabels() {
    var scale = net.getScale();
    var key = Math.round(scale * 16);
    if (key === labelsOn) return;
    labelsOn = key; scaleNow = scale;
    if (focus) { paint(focus); return; }
    nodes.update(DATA.nodes.map(function (n) {
      var b = baseNode(n);
      return { id: n.id, label: wantLabel(n, scale) ? n.label : "", font: b.font };
    }));
  }
  net.on("zoom", refreshLabels);
  // focus() animates the zoom without firing "zoom", which left labels sized for the old scale
  net.on("animationFinished", refreshLabels);
  net.once("stabilizationIterationsDone", function () { net.fit({ animation: false }); labelsOn = null; refreshLabels(); });

  // --- neighbourhood highlight ---------------------------------------------
  var focus = null, pinned = null;
  function paint(id) {
    focus = id;
    if (!id) {
      var sc = net.getScale();
      nodes.update(DATA.nodes.map(function (n) {
        var b = baseNode(n); if (!wantLabel(n, sc)) b.label = ""; return b;
      }));
      edges.update(DATA.edges.map(baseEdge));
      return;
    }
    var near = adj[id];
    // A hub with many neighbours would bury itself in labels; the card lists them instead.
    var crowded = Object.keys(near).length > 9;
    nodes.update(DATA.nodes.map(function (n) {
      var b = baseNode(n);
      if (n.id === id) { b.font = font(13.5, LABEL_HOT); return b; }
      if (near[n.id]) { b.font = font(11, LABEL_HOT); if (crowded) b.label = ""; return b; }
      b.label = ""; b.color = { background: DIM_NODE, border: DIM_NODE }; b.borderWidth = 0;
      return b;
    }));
    edges.update(DATA.edges.map(function (e) {
      var b = baseEdge(e), hot = e.from === id || e.to === id;
      b.color = { color: hot ? EDGE_HOT : DIM_EDGE, inherit: false }; b.width = hot ? 1.4 : 0.5;
      return b;
    }));
  }
  net.on("hoverNode", function (p) { if (!pinned) paint(p.node); });
  net.on("blurNode", function () { if (!pinned) paint(null); });
  net.on("click", function (p) {
    if (p.nodes.length) { pin(p.nodes[0]); } else { pin(null); }
  });
  net.on("dragStart", function (p) { if (p.nodes.length && !pinned) paint(p.nodes[0]); });

  // --- info card -----------------------------------------------------------
  var card = document.getElementById("card");
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
  function dot(group) { return '<i style="background:' + (G[group] ? G[group].color : "#999") + '"></i>'; }
  function pin(id) {
    pinned = id; paint(id);
    if (!id) { card.style.display = "none"; return; }
    var n = byId[id];
    var nb = Object.keys(adj[id]).map(function (k) { return { n: byId[k], e: adj[id][k] }; });
    nb.sort(function (a, b) { return b.e.w - a.e.w; });
    var html = '<div class="kind">' + dot(n.group) + esc(G[n.group] ? G[n.group].label : n.group) +
               (n.week ? " · " + esc(n.week) : "") + '</div><h4>' + esc(n.label) + '</h4>';
    if (n.detail) html += '<div class="detail">' + esc(n.detail) + '</div>';
    if (nb.length) {
      html += '<div class="sec">' + DATA.t.linked.replace('{n}', nb.length) + '</div><ul>';
      nb.slice(0, 14).forEach(function (x) {
        html += '<li data-id="' + esc(x.n.id) + '">' + dot(x.n.group) + '<span>' + esc(x.n.label) +
                (x.e.label ? ' <span style="color:#8B8B95">· ' + esc(x.e.label) + '</span>' : '') + '</span></li>';
      });
      if (nb.length > 14) html += '<li style="cursor:default;color:#8B8B95">' + DATA.t.more.replace('{n}', nb.length - 14) + '</li>';
      html += '</ul>';
    }
    card.innerHTML = html; card.style.display = "block"; card.scrollTop = 0;
  }
  card.addEventListener("click", function (ev) {
    var li = ev.target.closest("li[data-id]");
    if (!li) return;
    var id = li.getAttribute("data-id");
    pin(id); focusOn(id, Math.max(net.getScale(), 1.1));
  });

  // Centre a node in the space the info card leaves free, and zoom less in a narrow panel.
  function focusOn(id, scale) {
    var w = document.getElementById("wrap").clientWidth;
    var narrow = w < 760;
    net.focus(id, { scale: narrow ? Math.min(scale, 0.85) : scale,
                    offset: { x: narrow ? 120 : 0, y: narrow ? 40 : 0 },
                    animation: { duration: 600, easingFunction: "easeInOutQuad" } });
  }

  // --- search --------------------------------------------------------------
  var search = document.getElementById("search");
  search.addEventListener("keydown", function (ev) {
    if (ev.key === "Escape") { search.value = ""; pin(null); return; }
    if (ev.key !== "Enter") return;
    var q = search.value.trim().toLowerCase();
    if (!q) { pin(null); return; }
    var found = DATA.nodes.filter(function (n) { return n.label.toLowerCase().indexOf(q) >= 0; });
    found.sort(function (a, b) { return a.label.length - b.label.length; });
    if (found.length) { pin(found[0].id); focusOn(found[0].id, 1.3); }
  });

  // --- legend --------------------------------------------------------------
  var present = {};
  DATA.nodes.forEach(function (n) { present[n.group] = true; });
  var ringed = DATA.nodes.some(function (n) { return n.ring; });
  var lg = Object.keys(G).filter(function (g) { return present[g]; }).map(function (g) {
    return "<span>" + dot(g) + esc(G[g].label) + "</span>";
  });
  if (ringed) lg.push('<span><i style="background:transparent;border:2px solid #A882FF"></i>' + DATA.t.cited + '</span>');
  document.getElementById("legend").innerHTML = lg.join("");
})();
</script>
</body>
</html>
"""
