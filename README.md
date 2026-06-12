# GLM-OCR 本地文档解析与 QA 流水线

基于 [GLM-OCR](https://github.com/zai-org/GLM-OCR) 的自托管文档 OCR 方案：本地 **版面检测** + **Ollama GLM-OCR** 识别，可选 **云端多模态 LLM** 对照原图审核纠错，并支持批量将题干/答案截图拼接为 QA JSON。

适用于教育类截图（题目、答案、解析）、含公式与配图的长图等场景。

---

## 架构概览

```
输入图片 / PDF
      │
      ▼
┌─────────────────┐
│ PP-DocLayoutV3  │  本地版面检测（切 text / formula / image 等区域）
└────────┬────────┘
         ▼
┌─────────────────┐
│ GLM-OCR         │  本地 Ollama（glm-ocr:latest），按区域识别
│ (Ollama)        │  temperature: 0.0（config.yaml）
└────────┬────────┘
         ▼
   model_raw.json（原始 region 输出）
         │
         ▼  （可选，LLM_REVIEWER_ENABLED=true）
┌─────────────────┐
│ LLM 审核器      │  云端多模态 API（如 mimo-v2.5）
│ 分 4 阶段流水线 │  temperature: 0（.env）
└────────┬────────┘
         ▼
   model.json + review_report.json
         │
         ▼
   {name}.json（最终结构化结果，供 QA 拼接）
```

| 组件 | 部署位置 | 配置 |
|------|----------|------|
| 版面检测 PP-DocLayoutV3 | 本地 Python（权重自动下载） | `glmocr/config.yaml` → `pipeline.layout` |
| OCR 识别 GLM-OCR | 本地 Ollama | `glmocr/config.yaml` → `pipeline.ocr_api` |
| LLM 审核（可选） | 线上 OpenAI 兼容 API | 项目根目录 `.env` |

---

## 环境要求

- **Python** ≥ 3.10（建议 3.12）
- **Ollama**：本地运行 `glm-ocr` 模型
- **磁盘**：Ollama 模型约 2GB+；版面模型首次运行从 Hugging Face 下载
- **LLM 审核（可选）**：可访问的多模态 API Key，**无需**在 Ollama 中再装审核模型

---

## 安装

```bash
git clone <仓库地址>
cd GLM-OCR

python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# 安装 SDK + 自托管依赖（版面检测、PDF 等）
pip install -e ".[selfhosted]"
```

### 安装 Ollama 并拉取模型

1. 从 [ollama.com/download](https://ollama.com/download) 安装 Ollama
2. 拉取模型：

```powershell
ollama pull glm-ocr
ollama list   # 应看到 glm-ocr:latest
```

默认 API：`http://127.0.0.1:11434`

---

## 配置

### 1. OCR 与版面：`glmocr/config.yaml`

自托管模式（不用智谱云 MaaS）：

```yaml
pipeline:
  maas:
    enabled: false

  ocr_api:
    api_host: 127.0.0.1
    api_port: 11434
    model: glm-ocr:latest
    api_mode: ollama_generate
    api_path: /api/generate
    request_timeout: 120

  page_loader:
    temperature: 0.0    # OCR 识别温度，建议保持 0

  layout:
    model_dir: PaddlePaddle/PP-DocLayoutV3_safetensors
    device: cpu           # 有 GPU 可改为 cuda 或 cuda:0
```

### 2. LLM 审核与批量路径：`.env`

```powershell
copy .env.example .env
```

**审核器（可选）** — 对照原图做四类纠错，关闭则设 `LLM_REVIEWER_ENABLED=false`：

| 阶段 | 作用 |
|------|------|
| `insert` | 漏行补全（layout 漏框的整行文字） |
| `ocr_fix` | 识别纠错（错字、角标/撇号补回等） |
| `dedup_fix` | 相邻区域切分重叠去重 |
| `reorder` | 阅读顺序纠错 |

每页按 **insert → ocr_fix → dedup_fix → reorder** 顺序各调用一次 API（启用的阶段）。

常用变量：

```env
LLM_REVIEWER_ENABLED=true
LLM_REVIEWER_BASE_URL=https://api.xiaomimimo.com/v1
LLM_REVIEWER_API_KEY=你的密钥
LLM_REVIEWER_MODEL=mimo-v2.5
LLM_REVIEWER_TEMPERATURE=0          # 建议 0，输出最稳定
LLM_REVIEWER_CONFIDENCE_THRESHOLD=0.8
LLM_REVIEWER_MAX_TOKENS=8192
LLM_REVIEWER_TIMEOUT=120
```

**批量处理路径**（`batch_process.py` 读取）：

```env
BATCH_INPUT_DIR=D:\Desktop\你的图片文件夹
BATCH_OCR_OUTPUT_DIR=output
BATCH_QA_OUTPUT_DIR=qa_output
```

相对路径以项目根目录为基准。

---

## 使用方式

### 单张 / 目录图片

```powershell
# 单张图，默认输出到 ./output/
glmocr parse "D:\path\to\image.png"

# 指定输出目录
glmocr parse "D:\path\to\image.png" -o D:\Desktop\GLM-OCR\output

# 只打印到终端、不写文件
glmocr parse "D:\path\to\image.png" --stdout --no-save

# 覆盖配置项
glmocr parse image.png --set pipeline.layout.device cuda:0
```

### 批量：题干 + 答案 → QA JSON

1. 在 `.env` 中配置 `BATCH_INPUT_DIR`、`BATCH_OCR_OUTPUT_DIR`、`BATCH_QA_OUTPUT_DIR`
2. 将成对图片放入输入目录：

```text
INPUT_DIR/
  1q1_question.png
  1q1_answer.png
  33801812_question.png
  33801812_answer.png
```

命名规则：`{前缀}_question.png` / `{前缀}_answer.png`

3. 运行：

```powershell
cd GLM-OCR
python batch_process.py
```

流程：扫描配对 → 分别 OCR → 写入 OCR 输出目录 → 拼接 QA JSON → 汇总裁剪图到 `qa_output/imgs/`。

**审核失败时**：该题 OCR 结果不保留，控制台显示失败，脚本跳过并继续下一题。

---

## 输出说明

### 单张 OCR 目录（`glmocr parse` 或批量 OCR）

以 `1q1_question` 为例，保存在 `output/1q1_question/`：

```text
1q1_question/
  1q1_question.json                      # 最终结构化结果（启用审核时为审核后内容）
  1q1_question_model_raw.json            # 审核前原始 region JSON（仅启用审核时）
  1q1_question_model.json                # 审核后 region JSON（仅启用审核时）
  1q1_question_model_review_report.json  # 审核改动报告（仅启用审核时）
  imgs/                                  # 版面中 image 区域裁剪
  layout_vis/                            # 版面检测可视化
```

`review_report.json` 常用字段：

| 字段 | 含义 |
|------|------|
| `total_changes` | 实际采纳的改动条数 |
| `ocr_fix_count` / `insert_count` 等 | 各类型改动数量 |
| `latency_ms` | 整段 LLM 审核耗时（毫秒），含所有页、所有阶段 API 调用 |
| `retry_count` | API 失败重试次数之和 |

未启用 LLM 审核时，仅生成 `{name}.json` 与 `{name}_model.json`（后者为原始模型输出，命名兼容旧版）。

### QA 输出（`batch_process.py`）

```text
qa_output/
  {前缀}.json          # 拼接后的 QA 记录
  imgs/                # 各题裁剪图汇总
```

典型字段：`question`、`answer`、`analysis`、`Detailed explanation`、`question_type`、`topic` 等（由 `glmocr/utils/qa_pair_builder.py` 解析）。

---

## 常见问题

**Ollama 连接失败**

- 确认 Ollama 桌面端或 `ollama serve` 在运行
- 检查 `config.yaml` 中 `api_host` / `api_port` 与 `ollama list` 可用时一致

**版面模型下载慢**

- `layout.model_dir` 首次从 Hugging Face 拉取 `PaddlePaddle/PP-DocLayoutV3_safetensors`，需网络畅通

**LLM 审核报错 / 提取失败**

- 检查 `.env` 中 API Key、模型名、余额与网络
- 可暂时 `LLM_REVIEWER_ENABLED=false` 先跑通 OCR
- 大图 region 多易超时或截断，可增大 `LLM_REVIEWER_MAX_TOKENS` / `LLM_REVIEWER_TIMEOUT`

**多次 parse 审核结果不一致**

- 确保 `LLM_REVIEWER_TEMPERATURE=0`；云端 API 仍可能有轻微波动
- `model_raw.json` 应稳定（OCR `temperature: 0.0`）；波动主要来自审核阶段

**批量某一题失败**

- 脚本打印 `[跳过]` 并继续；审核失败时会删除该题半成品 OCR 目录

---

## 相关文件

| 文件 | 作用 |
|------|------|
| `batch_process.py` | 批量：图片对 → OCR → QA |
| `glmocr/config.yaml` | Ollama、版面、Pipeline 主配置 |
| `.env` / `.env.example` | LLM 审核 API、批量路径 |
| `glmocr/postprocess/llm_reviewer.py` | LLM 四阶段审核逻辑 |
| `glmocr/utils/qa_pair_builder.py` | 题干/答案 JSON → QA 记录 |
| `glmocr/cli.py` | `glmocr parse` 命令行入口 |

---

## 上游与许可

- 模型与论文：[zai-org/GLM-OCR](https://github.com/zai-org/GLM-OCR)
- 本项目 SDK 许可：Apache-2.0
