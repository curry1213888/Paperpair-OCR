# GLM-OCR 本地使用指南

本项目在 **自托管模式** 下运行：本地 **Ollama** 提供 GLM-OCR 识别能力，本地 **版面检测模型** 切分区域，可选 **线上 LLM API** 做结果审核纠错。本文说明环境准备、配置与运行方式。

---

## 1. 环境要求

- **Python** ≥ 3.10（建议 3.12）
- **Ollama**：本地运行 `glm-ocr` 模型（见下文）
- **磁盘**：Ollama 模型约 2GB+；版面模型首次运行会从 Hugging Face 自动下载
- **LLM 审核（可选）**：仅需能访问的线上 API（如 DeepSeek），**不需要**在 Ollama 里再装审核用大模型

---

## 2. 安装本项目

```bash
git clone <你的仓库地址>
cd GLM-OCR

# 建议使用虚拟环境
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# 安装 SDK + 自托管依赖（版面检测、PDF 等）
pip install -e ".[selfhosted]"
```

---

## 3. 安装 Ollama 并拉取 GLM-OCR 模型

### 3.1 安装 Ollama

1. 打开 [https://ollama.com/download](https://ollama.com/download) 下载 Windows 安装包并安装。
2. 安装完成后确认服务已启动：

```powershell
ollama --version
ollama list
```

默认 API 地址：`http://127.0.0.1:11434`

### 3.2 拉取 GLM-OCR 模型

```powershell
ollama pull glm-ocr
```

查看是否成功：

```powershell
ollama list
# 应能看到 glm-ocr:latest
```

### 3.3（可选）查看模型文件路径

```powershell
# 若设置了 OLLAMA_MODELS，模型在该目录；否则一般在用户目录下
echo $env:OLLAMA_MODELS
ollama show glm-ocr:latest --modelfile
# 输出里 FROM 一行即权重文件路径
```

---

## 4. 连接本地 Ollama：修改 `glmocr/config.yaml`

确保使用 **自托管模式**（不用智谱云 MaaS）：

```yaml
pipeline:
  maas:
    enabled: false   # 必须为 false

  ocr_api:
    api_host: 127.0.0.1
    api_port: 11434
    model: glm-ocr:latest          # 与 ollama list 中的名称一致
    api_mode: ollama_generate
    api_path: /api/generate
    request_timeout: 120

  layout:
    model_dir: PaddlePaddle/PP-DocLayoutV3_safetensors  # 首次运行自动下载
    device: cpu                  # 无 GPU 用 cpu；有 GPU 可改为 cuda 或 cuda:0
    batch_size: 1
```

**说明：**


| 组件                  | 部署位置              | 配置位置               |
| ------------------- | ----------------- | ------------------ |
| GLM-OCR 识别          | 本地 Ollama         | `pipeline.ocr_api` |
| 版面检测 PP-DocLayoutV3 | 本地 Python（自动下载权重） | `pipeline.layout`  |
| LLM 审核（可选）          | **线上 API**        | 项目根目录 `.env`       |


修改配置后无需重启 Ollama，重新运行 Python 命令即可。

---

## 5. LLM 审核（可选，走线上 API）

审核用于纠正 **阅读顺序** 与 **OCR 明显逻辑错误**，与 Ollama 无关。

1. 复制环境变量模板：

```powershell
copy .env.example .env
```

1. 编辑 `.env`（示例 DeepSeek）：

```env
LLM_REVIEWER_ENABLED=true
LLM_REVIEWER_BASE_URL=https://api.deepseek.com/v1
LLM_REVIEWER_API_KEY=你的密钥
LLM_REVIEWER_MODEL=deepseek-v4-flash

LLM_REVIEWER_MAX_TOKENS=8192
LLM_REVIEWER_TIMEOUT=120
LLM_REVIEWER_DISABLE_THINKING=true
LLM_REVIEWER_MAX_RETRIES=2
```

1. 关闭审核：设 `LLM_REVIEWER_ENABLED=false` 即可。

**审核失败时：** 本次图片 **不会保存 OCR 结果**，控制台显示「提取失败」，批量脚本会 **跳过该题** 继续下一题。

---

## 6. 运行方式

### 6.1 批量处理（题目 + 答案 → QA JSON）

**运行前必须先改输入/输出路径。** `python batch_process.py` **不支持**命令行传路径，所有目录都在项目根目录下的脚本里配置：

**文件位置：** `D:\Desktop\GLM-OCR\batch_process.py`（克隆到别处则改为你的项目路径）

打开该文件，修改顶部 **「配置区」**（约第 26–37 行）中的三个变量：

```python
# ============================================================
# 配置区：修改以下路径
# ============================================================

# 输入文件夹：存放 {prefix}_question.png 和 {prefix}_answer.png 的目录
INPUT_DIR = r"D:\Desktop\你的图片文件夹"

# OCR 输出文件夹：存放每张图片的识别结果（JSON、裁剪图片、版面可视化）
OCR_OUTPUT_DIR = r"D:\Desktop\GLM-OCR\output"

# QA 输出文件夹：存放最终拼接好的 QA JSON 文件
QA_OUTPUT_DIR = r"D:\Desktop\GLM-OCR\qa_output"
```

保存后再执行 `python batch_process.py`。

**输入示例：**

```text
INPUT_DIR/
  1q1_question.png
  1q1_answer.png
  2q10_question.png
  2q10_answer.png
```

**运行：**

```powershell
cd D:\Desktop\GLM-OCR
python batch_process.py
```

**流程：**

1. 扫描 `INPUT_DIR` 中成对的 `*_question` / `*_answer` 图片
2. 分别 OCR，结果写入 `OCR_OUTPUT_DIR`
3. 拼接为 QA，写入 `QA_OUTPUT_DIR/{题号}.json`
4. 裁剪图汇总到 `QA_OUTPUT_DIR/imgs/`

---

### 6.2 单张图片（命令行）

```powershell
# 单张图，结果默认保存到 ./output/
glmocr parse "D:\path\to\image.png"

# 指定输出目录
glmocr parse "D:\path\to\image.png" -o D:\Desktop\GLM-OCR\output

# 只打印到终端、不写文件
glmocr parse "D:\path\to\image.png" --stdout --no-save
```

---

## 7. 输入 / 输出路径说明

### 7.1 `batch_process.py`

路径由 `batch_process.py` 内 `INPUT_DIR`、`OCR_OUTPUT_DIR`、`QA_OUTPUT_DIR` 决定（见上文 6.1 节），与运行时的当前工作目录无关，但建议仍在项目根目录 `D:\Desktop\GLM-OCR` 下执行命令。


| 类型     | 路径                                  | 说明                                         |
| ------ | ----------------------------------- | ------------------------------------------ |
| 输入     | `INPUT_DIR/{prefix}_question.png`   | 题干截图（`INPUT_DIR` 在 `batch_process.py` 中配置） |
| 输入     | `INPUT_DIR/{prefix}_answer.png`     | 答案截图                                       |
| OCR 输出 | `OCR_OUTPUT_DIR/{prefix}_question/` | 单张题干 OCR 结果目录                              |
| OCR 输出 | `OCR_OUTPUT_DIR/{prefix}_answer/`   | 单张答案 OCR 结果目录                              |
| QA 输出  | `QA_OUTPUT_DIR/{prefix}.json`       | 拼接后的 QA 记录                                 |
| QA 附图  | `QA_OUTPUT_DIR/imgs/`               | 各题裁剪图汇总                                    |


### 7.2 单次 OCR 目录结构（`glmocr parse` 或批量 OCR 共用）

以 `1q1_question` 为例，保存在 `OCR_OUTPUT_DIR/1q1_question/`：

```text
1q1_question/
  1q1_question.json                 # 最终 OCR 结果（供 QA 拼接；启用审核时为审核后内容）
  1q1_question_model_raw.json       # 审核前原始 region JSON（仅启用 LLM 审核时）
  1q1_question_model.json           # 审核后 region JSON（仅启用 LLM 审核时）
  1q1_question_model_review_report.json  # 审核改动报告（仅启用 LLM 审核时）
  imgs/                             # 版面中的图片区域裁剪
  layout_vis/                       # 版面检测可视化
```

未启用 LLM 审核时，仅生成 `{name}.json` 与 `{name}_model.json`（后者为原始模型输出，命名与旧版兼容）。

### 7.3 QA JSON 字段（`qa_output/{prefix}.json`）

由 `glmocr/utils/qa_pair_builder.py` 根据问题/答案 OCR 的 `{prefix}.json` 拼接，典型字段：

- `question`、`answer`、`analysis`、`Detailed explanation`
- `question_type`、`topic`、`source` 等元数据

---

## 8. 常见问题

**Ollama 连接失败**

- 确认 `ollama serve` 或 Ollama 桌面端在运行  
- `config.yaml` 中 `api_host` / `api_port` 与 `ollama list` 可用时一致

**版面模型下载慢**

- `layout.model_dir` 首次会从 Hugging Face 拉取 `PaddlePaddle/PP-DocLayoutV3_safetensors`，需网络畅通

**LLM 审核报错「提取失败」**

- 检查 `.env` 中 API Key、模型名、余额  
- 可暂时 `LLM_REVIEWER_ENABLED=false` 先跑通 OCR  
- 大图（region 很多）易触发超时或输出截断，可增大 `LLM_REVIEWER_MAX_TOKENS` / `LLM_REVIEWER_TIMEOUT`

**批量时某一题失败**

- 脚本会打印 `[跳过]` 并处理下一题；审核失败时会删除该题已写的半成品 OCR 目录（若存在）

---

## 9. 相关文件速查


| 文件                                   | 作用                       |
| ------------------------------------ | ------------------------ |
| `batch_process.py`                   | 批量：图片对 → OCR → QA        |
| `glmocr/config.yaml`                 | Ollama、版面检测、Pipeline 主配置 |
| `.env`                               | LLM 审核 API（可选）           |
| `.env.example`                       | 环境变量模板                   |
| `glmocr/postprocess/llm_reviewer.py` | LLM 审核逻辑                 |


更多模型与论文介绍见上游 [GLM-OCR](https://github.com/zai-org/GLM-OCR) 官方仓库。