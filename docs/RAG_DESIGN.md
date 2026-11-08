# RAG 设计与实现说明

本文记录学习助手 RAG 部分的设计决策、依据和实测数据，供组内对齐与最终报告引用。

## 1. 数据集

来源：课程提供的 `EE6405_Raw_Dataset.zip`（841 MB，148 个文件），
解压在 `data/raw/EE6405_Raw_Dataset/`（已 gitignore，不入库）。

| 目录 | 内容 | 规模 |
| --- | --- | --- |
| `01_Lecture_Slides & Activities/` | 34 个 PDF（W01–W11）+ 3 个讲师澄清 txt | 954 页 |
| `02_Lecture_Videos/` | 12 个 mp4（W01–W07），已完成 12/12 转写 | 784 MB；转写 5.70 h、44,451 词 |
| `03_Code/` | 24 个 ipynb（每周 New/Old 两版） | — |
| `04_Assessments/` | 10 对 MCQ(.md)+答案(.txt)、5 个 Coding Quiz PDF、6 个 IRA/TRA PDF | 61 页 |
| `05_Miscellaneous/` | Syllabus、Group Project Details | 15 页 |

实测结论：

- **所有 PDF 均可用 pypdf 直接抽取文本，不需要 OCR。**
- MCQ 与 Answers 按 `Qn` 严格配对，**构成现成的带标注客观题评测集**（见第 5 节）。
- 文件名中的 `W01`–`W11` 是可靠的周次标记，已抽为检索过滤字段。

### 数据源数量

课程要求「3 个或更多数据集用于比较」。本项目的 6 个模态来自同一门课，
但在**载入方式、分块策略、元数据结构和检索表现**上各自独立，可按模态做对比实验。
若评分更看重「来源互相独立」，可考虑补充外部 NLP 语料作为第 7 个来源。
**此点需与组员和 Prof 确认。**


## 2. 视频转写

课程录播视频使用 `faster-whisper` 的 `large-v3` 模型转写，语言固定为英语，启用 VAD 过滤静音段，并保留每个片段的起止时间戳。脚本支持断点续传：同时存在 JSON 和 TXT 输出的视频会跳过，也可用 `--force` 重跑。正式语料的 12 个视频全部使用 `float16`，输出 JSON 保留 `source`、`week`、`lecture`、`duration`、模型/精度元数据和带时间戳的 segments；TXT 便于人工抽查。

### 精度选择：float16 优先术语准确性

在正式全量运行前，对 `W05_L01_Video.mp4` 做了同模型、同参数的精度对照：

| 配置 | 耗时 | 峰值显存 | 实时倍率 | 术语抽查 |
| --- | ---: | ---: | ---: | --- |
| `int8_float16` | 37.57 s | 3,522 MiB | 25.1× | 将 “Gated Recurrent Units” 误听为 “Data Recurrent Units” |
| `float16` | 108 s | 6,148 MiB | 8.7× | 出现 `GRUs`，但同一处仍有措辞偏差；未出现 int8 的 `Data Recurrent Units` 形式 |

最终选择 `float16`。速度和显存节省不足以抵消领域术语误听对检索的影响：BM25 是词法匹配，查询 “GRU” 或 “Gated Recurrent Units” 时，误写成 “Data Recurrent Units” 会导致词面无法匹配，因而可能零召回；这是基于检索机制的直接推论。全量转写端到端耗时 19:43，峰值显存 5,606 MiB，12/12 个正式输出均为 `large-v3`/`float16`。

运行过程中发现环境问题：conda 环境中已有 `libcublas.so.12`，但其目录未暴露在 `LD_LIBRARY_PATH` 时，CUDA 初始化报错 `Library libcublas.so.12 is not found or cannot be loaded`。脚本现在会在启动时自动加入 active conda 环境下已有的 NVIDIA CUDA 库目录，不安装新依赖。

## 3. ASR 术语误识别与后处理

全量扫描课程术语后发现，即使使用 `float16`，多个低频专名仍会系统性误识别。下表的正确计数按大小写不敏感匹配，并归并复数与所有格形式（例如 `LSTM`/`LSTMs`、`GRU`/`GRUs`；因此 `GloVe` 的大小写变体也计为正确）：

| 术语 | 正确识别 | 误听/需修正 | 说明 |
| --- | ---: | ---: | --- |
| RoBERTa | 0 | 10 | 全部转写为 `Robota` |
| ChatGPT | 8 | 51 | 全部误听为 `ChatGDP`，包括 `ChatGDP-based`；误听率 51/59≈86% |
| BLEU | 0 | 14 | 全部转写为 `blue`/`BLUE`，其中 6 处为 `blue score` |
| BERT | 37 | 0 | 未见系统性误听 |
| TF-IDF | 48 | 0 | 未见系统性误听 |
| GloVe | 8 | 0 | 大小写不敏感计数；未见系统性误听 |
| LSTM | 34 | 0 | 未见系统性误听（含复数 `LSTMs`） |
| GRU | 17 | 0 | 未见系统性误听（含复数 `GRUs`） |

这不是量化精度问题：RoBERTa 和 BLEU 在 `float16` 正式转写中仍为零正确率，说明当前 ASR 对这些课程低频专名存在系统性缺陷，提高计算精度本身不能解决。对 RAG 而言这是硬伤：BM25 查询 `RoBERTa` 永远匹配不到原文中的 `Robota`，不是排名下降，而是零词法召回；dense 检索的输入文本也会因此偏离正确术语语义。后一句关于 dense 的影响是基于文本改变对向量输入的机制推论，尚未单独做误差实验。

最终术语修正汇总如下，按误听 occurrence 数降序排列：

| 正确术语 | 原始误听 | 误听 occurrence | 正确 occurrence | 误听率 | 修正数 |
| --- | --- | ---: | ---: | ---: | ---: |
| ChatGPT | `ChatGDP` / `ChatGDP-based` | 51 | 8 | 86% | 51 |
| BLEU | `blue` / `BLUE` | 14 | 0 | 100% | 14 |
| RoBERTa | `Robota` | 10 | 0 | 100% | 10 |
| RMSprop | `RMS prop` | 5 | 0 | 100% | 5 |
| ELIZA | `ELISA` | 3 | 0 | 100% | 3 |
| DALL-E | `DALI` | 3 | 0 | 100% | 3 |
| Midjourney | `Meet Journey` | 3 | 0 | 100% | 3 |
| Weizenbaum | `Wiesenbaum` | 1 | 0 | 100% | 1 |

表中的 occurrence 按原始 segment 内的术语出现次数统计；同一 segment 出现两次时计两次。参考 syllabus 和 lecture slides 的缺席扫描结果保存在 `reports/term_absence_scan.csv`：DALL-E、Midjourney、RMSprop、RoBERTa 在转写中为零或异常偏低，GCN 与 CRF 也未在这些视频中出现，但没有找到对应的 ASR 误听形式，因此没有据此自动改写；这可能反映录播与 slides 的覆盖差异。

后处理采用上下文感知替换，并保留原始转写：`Robota` 在课程语境中修正为 `RoBERTa`；`blue` 只有在 `score`、`metric`、`translation` 等评测或机器翻译窗口中修正为 `BLEU`，避免把普通颜色词无脑替换；`ChatGDP`（包括 `ChatGDP-based`）修正为 `ChatGPT`。`GloVe` 的大小写变体按正确识别计数，是因为检索侧可以做 casefold，大小写差异不影响召回；`Robota` 则是拼写错误，casefold 无法把它变回 `RoBERTa`。

这次审计也暴露了方法局限：预定义术语表只能发现“我们想到了的”错误。`ChatGPT` 原先不在表中，因此 `ChatGDP` 的 51 处误听长期不会出现在术语报告里；通过统计转写中的高频大写 token 才发现它。现在审计采用三条路径：预定义术语表发现已知错误，高频大写 token 扫描发现 `ChatGDP` 这类单词专名，已知术语的缺席扫描则反向检查 syllabus/slides 中应出现但转写为零或异常偏低的术语。完整 top-40 列表保存在 `reports/uppercase_tokens_top40.csv`/`.json`，其中 `ChatGDP` 排名第一（51 次）；缺席扫描保存在 `reports/term_absence_scan.csv`/`.json`。

修正版 JSON 在每个 segment 增加 `text_corrected`，原始 `text` 不变；`reports/term_audit.csv` 保存术语正确/误听计数，`reports/corrections.csv` 和 `reports/corrections.json` 保存文件、时间戳及原文到修正文本的审计清单。

### Whisper 幻觉循环

除术语误听外，扫描还发现连续重复的 segment，表现为 Whisper 在静音或噪声段重复输出同一句话。12 个文件共 6,378 个原始 segment，其中 25 个属于 4 个连续重复 run（约 0.4%）：`W02_Video` 的 `NEDs.` 连续 13 次，`W03_Video` 一段 TF-IDF 句子连续 4 次，`W05_L01_Video` 两段句子各连续 4 次。重复文本会让同一内容在索引中占据多个 chunk，可能挤占 top-k 结果；这是基于检索排序机制的影响判断。

后处理将长度至少为 3 的完全相同连续 run 折叠为一条，在代表 segment 上标记 `hallucination_collapsed: true` 和重复次数；原始 JSON 不变，所有被折叠的原始 segment、时间范围和文本写入 `reports/hallucinations.json`/`.csv`，每文件汇总写入 `reports/hallucination_summary.csv`。修正版输出因此可供索引使用，也能回溯每个折叠决定。

## 4. 分块策略

统一分块会破坏引用精度，因此**按模态分别处理**：

| 模态 | 分块单位 | locator（引用定位） |
| --- | --- | --- |
| `slides` | 按页 | `page:<页码>` |
| `clarification` | 每条澄清一条 | `slide:<幻灯片页>` 或 `slide:<Topic>` |
| `notebook` | markdown cell + 其后紧邻的 code cell | `cell:<序号>` |
| `mcq` | 每题一条（题干与答案存在一起） | `q:<题号>` |
| `assessment_pdf` | 按页 | `page:<页码>` |
| `misc` | 按页 | `page:<页码>` |
| `transcript` | 时间窗口（12 个视频已转写） | `t:<mm:ss>` |

### clarification 解析的坑

三个澄清文件格式并不一致，初版解析器只覆盖了第一种，导致 5 条澄清只入库 3 条、
W06 整个文件丢失。实际格式差异：

| 文件 | 条目起始字段 | `Week` 写法 |
| --- | --- | --- |
| `W01_L01_Clarification.txt` | `Slide Reference:` | `Week: 01` |
| `W03_Clarification.txt` | 第 1 条 `Slide Reference:`，第 2 条 `Topic:` | `Week: 03` |
| `W06_Clarification.txt` | 仅 `Topic:` | `Week: W06` |

修复方式：以 `Original Clarification:` 作为唯一切分锚点（5 条全有且每条仅一个），
`Slide Reference:` / `Topic:` 降级为可选属性；`week` 统一归一化为 `W01` 形式。

## 5. 索引

SQLite 单文件（`data/processed/index.sqlite`），三张表：

- `chunks` — 正文与元数据，含可过滤列 `week`、`modality`、`locator`
- `chunks_fts` — FTS5 虚表，提供 BM25 词法检索
- `embeddings` — float32 向量 BLOB

选 SQLite 而非 FAISS/向量数据库的理由：零额外服务、索引文件可直接用 SQLite 浏览器
打开（答辩演示有用）、全库约 5 MB 便于复现。本项目语料规模（约 1500 chunk）下，
numpy 暴力余弦检索的延迟可忽略，不需要 ANN 索引。

### 实测构建结果

```
python -m ee6405.ingest.pipeline --source data/raw/EE6405_Raw_Dataset --rebuild
```

| modality | chunks | 源文件核对 |
| --- | --- | --- |
| slides | 943 | 34 个 PDF / 954 页，11 页抽取为空（纯图片页）已跳过 |
| notebook | 479 | 24 个 ipynb 的语义单元 |
| mcq | 47 | 10 个 md 共 47 题，全部入库 |
| assessment_pdf | 44 | 11 个 PDF / 61 页，空页已跳过 |
| misc | 15 | 2 个 PDF / 15 页 |
| clarification | 5 | 3 个 txt 共 5 条，全部入库 |
| **合计** | **1533** | 索引文件约 5 MB |

视频文件已完成转写，尚未并入索引；并入后将新增 `transcript` 模态。

## 6. 检索

三路检索，可独立启用，正好对应课程内容，既是功能也是消融实验：

| 检索器 | 方法 | 对应课程内容 |
| --- | --- | --- |
| `bm25` | SQLite FTS5 词法匹配 | Week 3 Term Weighting |
| `dense` | BGE-m3 向量余弦相似度 | Week 4 Word Embeddings |
| `hybrid` | 倒数排名融合（RRF） | Week 10 RAG |

选 RRF 而非分数加权：BM25 分数与余弦相似度量纲不同，RRF 只用排名，
无需校准，且对两路的分数分布不敏感。

支持按 `week` / `modality` 过滤，使「第 6 周讲了什么」这类问题可先缩小范围再检索。

### Embedding：本地 BGE-m3

- 模型 `BAAI/bge-m3`，1024 维，中英文均佳（提问可能是中文，语料是英文）
- 本地 RTX 5060（8 GB，sm_120）运行，batch size 8，CUDA 不可用时自动降级 CPU
- **选本地而非 API 的理由**：公司内网网关的 `/v1/embeddings` 端点实测返回 503
  （`No available channel`），且本地方案免费、不限量、调参时反复重建索引无成本、
  答辩可完全离线复现

## 7. 评测方案

### 5.1 检索质量

recall@k、precision@k、MRR、nDCG@k（已实现于 `src/ee6405/eval/metrics.py`）。

### 5.2 答题正确率（本项目的主要指标）

`04_Assessments/01_Practice/` 的 47 道 MCQ 自带标准答案，可直接计算准确率。
这比「RAG 召回率」更有说服力，且无需自行造题标注。

注意：MCQ 本身已入库为 `mcq` 模态。用它做评测时**必须把 mcq 模态排除出检索范围**，
否则模型会直接检索到答案，造成数据泄漏。这一点在实现评测时要落实为硬约束。

### 5.3 消融实验

| 配置 | use_rag | use_memory | retriever |
| --- | --- | --- | --- |
| base | ✓ | ✓ | hybrid |
| without rag | ✗ | ✓ | — |
| without memory | ✓ | ✗ | hybrid |
| bm25 only | ✓ | ✓ | bm25 |
| dense only | ✓ | ✓ | dense |

### 5.4 基线对比

内网网关提供 35 个模型，可在**同一套评测集和同一份 prompt** 下横向对比，
换 model id 即可，无需改代码。

## 8. 推理后端

当前开发测试使用公司内网 OpenAI 兼容网关（配置在 `.env`，已 gitignore）。

实测结论：

- `/v1/chat/completions` 可用，标准 OpenAI 格式，35 个模型（含 GPT / Claude /
  DeepSeek / Qwen / GLM / Gemini 系列）
- `/v1/embeddings` **不可用**（503），故 embedding 走本地
- 推理型模型（如 `deepseek-v4-pro`）的 `max_tokens` 需放大到数百以上，
  否则预算被 reasoning token 吃完、返回空字符串。实测 20 tokens 返回空，
  600 tokens 正常

### 后端选择（已定）

- **开发测试**：公司内网网关（免费、模型多，便于快速迭代和横向对比）
- **最终提交**：**公网 API**，与公司基础设施解耦

代码层面二者等价，均为 OpenAI 兼容接口，切换只需改 `.env` 三行
（`EE6405_LLM_BASE_URL` / `EE6405_LLM_API_KEY` / `EE6405_LLM_MODEL`），无需改动任何代码。

由此产生一条**硬约束**：报告中引用的所有评测数字，最终必须在公网 API 上复跑一遍。
内网网关外部无法访问，助教/评分者无法复现基于它的结果。因此：

- 所有评测输出（CSV）必须记录 `model` 与 `base_url`，便于区分数据来源
- 不得在代码中硬编码任何内网专有的 endpoint 或模型名

公网侧**同时接入 DeepSeek 与 Qwen 两家**，在同一套评测集、同一份 prompt 下横向对比，
作为报告的一组 baseline（课程评分鼓励多模型/多方法比较）。二者均为 OpenAI 兼容接口：

| 提供方 | base_url | 候选模型 |
| --- | --- | --- |
| DeepSeek | `https://api.deepseek.com` | `deepseek-chat` |
| Qwen（阿里百炼） | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` / `qwen-max` |

因需并存两套凭据，`.env` 的单一 `EE6405_LLM_*` 不再够用，评测配置需支持
**按系统分别指定 `base_url` 与 key 环境变量名**（见 `configs/eval.yaml`）。

## 9. 待办

- [x] 视频已转写（12 个视频、`large-v3`/`float16`）；待并入 `transcript` 模态并接入索引（带时间戳）
- [ ] 生成链路接入真实 LLM（当前 `EchoLLM` 为离线占位）
- [ ] 评测脚本接入 MCQ 评测集，落实 mcq 模态排除
- [ ] 长期 memory（`without memory` 消融依赖它）
- [ ] 知识图谱抽取与可视化
- [ ] Streamlit 前端
- [x] ~~确认数据源数量是否满足「3 个以上数据集」~~ — 暂不处理，按现有 6 模态推进
- [x] ~~确认最终提交使用的推理后端~~ — **已定：最终提交用公网 API**
- [ ] 取得 DeepSeek 与 Qwen 两家公网 key
- [ ] 评测配置改为支持多 provider（按系统指定 base_url / key 环境变量）
- [ ] 在两家公网 API 上复跑全部评测，产出报告最终数字
