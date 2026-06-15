# GLM-OCR 批量 QA 流水线

基于 [GLM-OCR](https://github.com/zai-org/GLM-OCR) 构建的本地批量 OCR 方案：**版面检测** + **GLM-OCR 识别**，可选云端多模态 LLM 对照原图审核纠错，将题干/答案截图批量拼接为结构化 QA JSON 输出。

适用于含公式、配图的教育类截图（题目+答案成对处理）。

---

## 架构概览

```
{id}_question.png + {id}_answer.png
          │
          ▼
┌──────────────────────┐
│  PP-DocLayoutV3      │  本地版面检测（text / formula / image / table …）
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  GLM-OCR (Ollama)    │  按区域逐块 OCR，temperature=0
└──────────┬───────────┘
           │  model_raw.json
           ▼  （可选，LLM_REVIEWER_ENABLED=true）
┌──────────────────────┐
│  LLM 审核器          │  云端多模态 API，分 4 阶段纠错：
│  (如 mimo-v2.5)      │  insert → ocr_fix → dedup_fix → reorder
└──────────┬───────────┘
           │  model.json + review_report.json
           ▼
  qa_output/{id}.json  ←  拼接后的 QA 结构化记录
  qa_output/imgs/      ←  裁剪图片汇总
```

| 组件 | 运行位置 | 主要配置 |
|------|----------|----------|
| 版面检测 PP-DocLayoutV3 | 本地 Python（权重自动下载） | `glmocr/config.yaml` → `pipeline.layout` |
| OCR 识别 GLM-OCR | 本地 Ollama（默认）或 vLLM/SGLang | `glmocr/config.yaml` → `pipeline.ocr_api` |
| LLM 审核（可选） | 云端 OpenAI 兼容 API | `.env` → `LLM_REVIEWER_*` |
| 批量处理入口 | 本地 Python | `.env` → `BATCH_*` |

---

## 环境要求

- **Python** ≥ 3.10（推荐 3.12）
- **Ollama** ≥ 0.24.0，已拉取 `glm-ocr:latest` 模型
- **磁盘空间**：Ollama 模型约 2 GB；PP-DocLayoutV3 版面模型首次运行从 Hugging Face 自动下载（约 300 MB），或提前下载到本地目录

> **注意**：若网络访问 Hugging Face 受限，请提前手动下载版面模型并通过 `GLMOCR_LAYOUT_MODEL_DIR` 指定本地路径（见下文配置）。

---

## 快速安装

```bash
# 1. 克隆仓库
git clone https://github.com/your-org/glm-ocr.git
cd glm-ocr

# 2. 安装 Python 依赖（包含自托管所需的 layout 扩展）
pip install -e ".[layout]"

# 3. 安装 Ollama 并拉取 GLM-OCR 模型
#    Ollama 下载：https://ollama.com/download
ollama pull glm-ocr:latest

# 4. 复制并填写配置文件
copy .env.example .env
# 然后用文本编辑器修改 .env 中的路径和 API Key
```

---

## 配置文件说明

### `.env` — 批量处理与审核配置

复制 `.env.example` 为 `.env`，按需填写：

#### 必填项

| 变量 | 说明 | 示例 |
|------|------|------|
| `BATCH_INPUT_DIR` | 存放输入图片的文件夹（绝对路径） | `D:\Desktop\input` |
| `GLMOCR_LAYOUT_MODEL_DIR` | 版面模型本地路径（留空则联网下载） | `D:\models\PP-DocLayoutV3_safetensors` |

#### 批量处理路径（可选，有合理默认值）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `BATCH_OCR_OUTPUT_DIR` | `output` | OCR 结果输出目录 |
| `BATCH_QA_OUTPUT_DIR` | `qa_output` | QA JSON 输出目录 |
| `BATCH_ERROR_DIR` | `batch_error` | 失败/跳过原图备份目录 |
| `BATCH_MAX_STORED_QUESTIONS` | `500` | output/qa_output 最多保留题数，超出时按最旧优先删除；设为-1则不删除 |

#### OCR 服务连接（默认对接本机 Ollama）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `GLMOCR_OCR_API_HOST` | `127.0.0.1` | OCR 服务地址 |
| `GLMOCR_OCR_API_PORT` | `11434` | OCR 服务端口 |
| `GLMOCR_OCR_API_PATH` | `/api/generate` | API 路径 |
| `GLMOCR_OCR_API_MODE` | `ollama_generate` | `ollama_generate` 或 `openai` |
| `GLMOCR_OCR_MODEL` | `glm-ocr:latest` | 模型名称 |

#### LLM 审核器（可选）

| 变量 | 说明 |
|------|------|
| `LLM_REVIEWER_ENABLED` | `true` 启用审核，`false` 跳过（默认 true） |
| `LLM_REVIEWER_BASE_URL` | OpenAI 兼容接口的 base URL（到 `/v1`） |
| `LLM_REVIEWER_API_KEY` | 对应平台的 API Key |
| `LLM_REVIEWER_MODEL` | 多模态模型名称（需支持 vision，如 `mimo-v2.5`、`gpt-4o`） |

其余审核器调参见 `.env.example` 中的注释。

### `glmocr/config.yaml` — 版面与 OCR 管线配置

通常不需要手动编辑，版面模型路径已统一走 `GLMOCR_LAYOUT_MODEL_DIR` 环境变量。若有高级需求（调整检测阈值、切换 GPU 设备、修改长图切分策略等），可直接编辑此文件，每个字段均有注释说明。

---

## 使用方法

### 1. 准备输入图片

将题目和答案截图按以下命名规则放入 `BATCH_INPUT_DIR` 目录：

```
{id}_question.png   ← 题目截图
{id}_answer.png     ← 答案截图
```

- `{id}` 可以是任意字符串（如题目编号 `33801700`、`1q1` 等），不能包含下划线
- 支持格式：`.jpg`、`.jpeg`、`.png`、`.bmp`、`.webp`
- **必须成对**：缺少任意一方的图片将被标记为未匹配并跳过

**示例：**
```
input/
  33801700_question.png
  33801700_answer.png
  1q1_question.png
  1q1_answer.png
```

### 2. 确认 Ollama 正在运行

```bash
ollama serve        # 若未启动则先运行此命令
ollama list         # 确认 glm-ocr:latest 已拉取
```

### 3. 运行批量处理

```bash
python batch_process.py
```

程序会输出每题的处理进度，处理完成后打印汇总结果。

---

## 输出结构

### `output/`（OCR 中间结果，每题一个子目录）

```
output/
  {id}_question/
    {id}_question.json          ← 结构化 OCR 结果（块列表）
    {id}_question_model.json    ← 审核后的最终 region 列表
    {id}_question_model_raw.json← 审核前的原始 region 列表
    {id}_question_model_review_report.json ← 审核变更报告
    imgs/                       ← 各 region 裁剪图片
    layout_vis/                 ← 版面检测可视化图
  {id}_answer/
    （同上结构）
```

### `qa_output/`（最终输出）

```
qa_output/
  {id}.json       ← 拼接好的 QA 记录
  imgs/           ← 所有题目的裁剪图片汇总
```

### QA JSON 格式

```json
[
  {
    "id": "33801700",
    "question_type": "单选题",
    "difficulty_text": "适中",
    "difficulty_score": 0.65,
    "topic": "数学·函数",
    "source": "2025年全国卷",
    "question": "题干正文，含 [image:imgs/xxx.jpg] 图片占位符",
    "answer": ["C"],
    "analysis": "分析文字",
    "Detailed explanation": "详细解题过程"
  }
]
```

> `answer` 为列表，单选题通常只有一个元素；`analysis`、`Detailed explanation` 可能为 `null`。

### `batch_error/`（失败备份）

处理失败或跳过的题目，原图会备份到此目录，并附带 `reason.txt` 说明原因：

```
batch_error/
  {id}/
    {id}_question.png
    {id}_answer.png
    reason.txt
```

---

## 处理逻辑说明

1. **扫描匹配**：扫描 `BATCH_INPUT_DIR`，按 `{id}_question.*` / `{id}_answer.*` 命名规则配对
2. **OCR**：对每张图片依次执行版面检测 + 逐区域 OCR，结果保存到 `output/`
3. **LLM 审核**（可选）：若 `LLM_REVIEWER_ENABLED=true`，调用云端模型对照原图执行 4 阶段纠错
4. **QA 拼接**：读取 question/answer 的 OCR 结果，解析元数据（题型、难度、来源），拼接为 QA 记录
5. **元数据补全**（自动）：若正文中未识别到题型/难度，会对题目截图顶部图片块补充 OCR 并再次尝试提取
6. **容量管理**：若 `qa_output/` 中的题目数超过 `BATCH_MAX_STORED_QUESTIONS`，自动删除最旧的题目（同时清理 `output/` 中对应目录）
7. **原图清理**：处理成功后自动删除输入目录中的原图；失败时先备份到 `batch_error/` 再删除

---

## 使用 Zhipu MaaS API（云端模式）

不想本地部署 Ollama 时，可改用智谱云 API：

```bash
# .env 中设置
ZHIPU_API_KEY=your_api_key_here
```

无需其他改动，SDK 检测到 API Key 后会自动切换为 MaaS 模式。

---

## 辅助脚本

| 脚本 | 用途 |
|------|------|
| `scripts/rebuild_qa_output.py` | 基于已有的 OCR JSON 重新生成 QA 输出（无需重跑 OCR） |
| `scripts/setup_dev.py` | 安装开发依赖并配置 pre-commit 钩子 |

---

## 开发与测试

```bash
# 安装开发依赖
pip install -e ".[layout,dev]"

# 运行单元测试（无需外部服务）
pytest glmocr/tests/test_unit.py -v

# 运行集成测试（需要 Ollama 服务运行中）
pytest glmocr/tests/test_integration.py -v

# 代码格式检查
black glmocr/ batch_process.py
flake8 glmocr/ batch_process.py
```

---

## 常见问题

**Q: 运行时提示 `模型权重下载失败` 或网络超时**

设置 `GLMOCR_LAYOUT_MODEL_DIR` 指向提前下载好的本地模型目录：
```bash
# 下载模型（需要 huggingface-hub）
huggingface-cli download PaddlePaddle/PP-DocLayoutV3_safetensors --local-dir ./models/PP-DocLayoutV3_safetensors
```
然后在 `.env` 中设置：
```
GLMOCR_LAYOUT_MODEL_DIR=./models/PP-DocLayoutV3_safetensors
```

**Q: OCR 结果只有一个 `image` 块，被标记为 `OCR 失败`**

可能是图片内容过于简单（纯图片，无文字）或图片质量太低。检查原图后，若确认图片有效，可尝试提高图片分辨率后重新处理。

**Q: LLM 审核器 TLS 握手失败**

设置 `LLM_REVIEWER_ENABLE_CURL_FALLBACK=true`（默认已开启），SDK 会自动降级使用系统 `curl.exe` 发送请求。

**Q: 处理速度慢**

- 调整 `glmocr/config.yaml` 中的 `pipeline.max_workers`（并发 OCR 请求数）
- Ollama 默认单 GPU 运行，可通过 `GLMOCR_LAYOUT_DEVICE=cpu` 将版面检测放到 CPU，释放 GPU 给 OCR
- 关闭 LLM 审核器（`LLM_REVIEWER_ENABLED=false`）可大幅缩短处理时间

**Q: `answer`/`analysis` 字段全为 `null`**

说明答案图片中未识别到 `【答案】`/`【分析】`/`【详解】` 等段落标记。检查答案截图是否完整，或确认截图中确实存在这些标记文字。

---

## 项目结构（核心部分）

```
GLM-OCR/
├── batch_process.py          # 批量处理入口脚本
├── glmocr/                   # 核心 OCR 库
│   ├── api.py                # Python API（GlmOcr 类）
│   ├── config.py             # 配置模型与加载逻辑
│   ├── config.yaml           # 默认配置文件
│   ├── pipeline/             # 自托管处理管线
│   ├── layout/               # 版面检测（PP-DocLayoutV3）
│   ├── postprocess/          # 后处理（结果格式化、LLM 审核）
│   ├── utils/                # 工具函数（图像、QA 构建、可视化等）
│   └── tests/                # 单元测试与集成测试
├── resources/
│   └── PingFang.ttf          # 版面可视化字体
├── scripts/
│   ├── rebuild_qa_output.py  # 重新生成 QA 输出
│   └── setup_dev.py          # 开发环境初始化
├── .env.example              # 环境变量配置模板
└── pyproject.toml            # 项目依赖与元数据
```

---

## 依赖说明

核心依赖（`pip install -e .`）：

| 包 | 用途 |
|----|------|
| `pillow` | 图像处理 |
| `numpy` | 数值计算 |
| `requests` | HTTP 请求 |
| `pydantic` | 配置模型验证 |
| `PyYAML` | YAML 配置加载 |
| `python-dotenv` | .env 文件读取 |
| `pymupdf` | PDF 转图片 |
| `portalocker` | 文件锁 |
| `tqdm` | 进度条 |

版面检测扩展（`pip install -e ".[layout]"`，自托管模式必须）：

| 包 | 用途 |
|----|------|
| `torch` / `torchvision` | 深度学习框架 |
| `transformers` | PP-DocLayoutV3 模型加载 |
| `opencv-python-headless` | 图像预处理 |
| `sentencepiece` / `accelerate` | 模型辅助依赖 |
