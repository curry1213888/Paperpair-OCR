"""LLM-based OCR result reviewer.

审核 OCR 原始输出（model_raw.json），执行两类纠错：
  1. reorder  — 根据 bbox_2d 坐标与语义关系修正阅读顺序（index）
  2. ocr_fix  — 修正会导致逻辑错误的文字识别错误（content）

配置完全通过环境变量（见 .env.example），与现有流程零耦合。

使用方式（在 Pipeline 中）：
    from glmocr.postprocess.llm_reviewer import LLMReviewer
    reviewer = LLMReviewer.from_env()   # None if disabled
    if reviewer:
        reviewed_pages, report = reviewer.review(raw_json)
        new_grouped = reviewer.apply_to_grouped(grouped, reviewed_pages)
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict
from typing import Dict, Optional, Tuple

import requests

from glmocr.utils.logging import get_logger

# 尝试加载 .env 文件（python-dotenv 已是项目依赖）
try:
    from dotenv import load_dotenv as _load_dotenv

    _load_dotenv()
except ImportError:
    pass

logger = get_logger(__name__)

_SECTION_MARKERS = ("【答案】", "【分析】", "【详解】")


def _fuzzy_replace(text: str, before: str, after: str) -> Optional[str]:
    """将 before 中的连续空白归一化为 \\s+ 后在 text 中查找并替换为 after。

    仅替换第一处匹配。若未找到则返回 None。
    """
    if not before.strip():
        return None
    tokens = before.split()
    if not tokens:
        return None
    pattern = r"\s*".join(re.escape(t) for t in tokens)
    match = re.search(pattern, text)
    if match is None:
        return None
    return text[: match.start()] + after + text[match.end() :]


class LLMReviewError(RuntimeError):
    """LLM 审核在全部重试后仍失败；调用方应中止本次 OCR，勿回退原始输出。"""

    def __init__(self, message: str, report: Optional[dict] = None):
        super().__init__(message)
        self.report = report or {}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
你是专业的OCR结构化审核员，专注于数学和教育类文档。

你的任务仅限于以下两种纠错，不做其他任何修改：

【任务1】阅读顺序纠错（reorder）
- 根据 bbox_2d 坐标（格式：[x1, y1, x2, y2]，归一化坐标0-1000）判断阅读顺序
- 通常规则：从上到下、从左到右；多栏布局时先左栏再右栏
- figure_title（图题）应紧随对应的 image（图片）之后，例如图在前图题在后
- 仅在顺序明显不合理时才修改，通过 (page, bbox_2d) 定位 item，给出新的 index 值
- 不需要为每个 item 都输出 reorder，只输出需要调整 index 的 item
- 重要：输入数据可能来自"答案图片"。答案图片中【答案】→【分析】→【详解】是固定的正确顺序（并非所有标注都会出现，但出现时顺序必定如此）。若已按此顺序排列，绝对不可对这几个标注之间的顺序做 reorder，这不是错误。
- 仅允许用“显式标签文本”判断该规则：只有当 content 明确包含【答案】/【分析】/【详解】标记时，才可据此判断标签顺序。
- 严禁语义归类重判：不得因为“看起来像详解/分析/答案”就把某段无标签文本判为其他版块并 reorder。


【任务2】OCR文字逻辑纠错（ocr_fix）
- 仅修正会导致明显逻辑错误的识别错误，例如：
  * 数学公式中变量名混淆（正文用 b 求解但结果写成 k=−5，应为 b=5）
  * 数字或运算符误识别导致等式明显矛盾
  * 单字误识别严重影响句意（需极高把握才改）
- 不确定时宁可不改，禁止猜测性修改
- 通过 (page, bbox_2d) 定位 item
- ocr_fix 必须是“最小必要改动”：只改错误字符，不改无关文本
- 严禁摘要、压缩、重写、改写语气；不得删除原有句子或段落
- 若 before 含有结构标记（如【答案】【分析】【详解】），after 必须完整保留这些标记与文本结构

硬性约束：
1. 每条 change 必须通过 (page, bbox_2d) 精确定位输入中的一个 item
2. content 为 null 的 item（图片区域）不允许做 ocr_fix
3. 输出必须是严格合法 JSON，不含任何额外自然语言或 markdown 标记
4. 低把握的修改不要输出（ocr_fix 的 confidence 需 ≥ 0.8）
5. ocr_fix 的 after 必须是 before 的完整保留版，仅做局部修正，不得截断
6. 只输出真正需要修改的 changes，无需修改时输出空数组\
"""

_USER_PROMPT_TEMPLATE = """\
请审核以下 OCR 识别结果，只输出需要修改的项目。

输入字段说明：
- page: 页码（从0开始）
- index: 当前排序序号
- label: 区域类型（text / formula / image / figure_title / header 等）
- content: 识别内容（null 表示图片区域，不可修改 content）
- bbox_2d: 边界框 [x1, y1, x2, y2]，归一化0-1000坐标

输入数据：
{items_json}

请严格按以下 JSON 格式输出，只包含真正需要修改的项，不要输出任何其他内容：
{{
  "changes": [
    {{
      "type": "reorder 或 ocr_fix",
      "page": 整数,
      "bbox_2d": 数组（用于定位要修改的 item）,
      "field": "index 或 content",
      "before": "修改前的值（字符串）",
      "after": "修改后的值（字符串）",
      "reason": "简短说明（中文）",
      "confidence": 0到1之间的小数
    }}
  ]
}}

若无需任何修改，输出 {{"changes": []}}。\
"""


# ---------------------------------------------------------------------------
# LLMReviewer
# ---------------------------------------------------------------------------


class LLMReviewer:
    """LLM-based reviewer for OCR structured results.

    从环境变量读取配置（见 .env.example），与现有流程零耦合。
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        temperature: float = 0.1,
        max_tokens: int = 8192,
        confidence_threshold: float = 0.8,
        max_retries: int = 2,
        timeout: int = 120,
        enable_reorder: bool = True,
        enable_ocr_fix: bool = True,
        disable_thinking: bool = True,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.confidence_threshold = confidence_threshold
        self.max_retries = max_retries
        self.timeout = timeout
        self.enable_reorder = enable_reorder
        self.enable_ocr_fix = enable_ocr_fix
        self.disable_thinking = disable_thinking
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> "Optional[LLMReviewer]":
        """从环境变量创建实例。未启用或配置不全时返回 None。"""
        enabled = os.environ.get("LLM_REVIEWER_ENABLED", "false").strip().lower()
        if enabled not in ("true", "1", "yes"):
            return None

        api_key = os.environ.get("LLM_REVIEWER_API_KEY", "").strip()
        base_url = os.environ.get("LLM_REVIEWER_BASE_URL", "").strip()
        model = os.environ.get("LLM_REVIEWER_MODEL", "").strip()

        if not api_key:
            logger.warning(
                "LLM_REVIEWER_ENABLED=true 但 LLM_REVIEWER_API_KEY 未设置，审核器已禁用"
            )
            return None
        if not base_url:
            logger.warning(
                "LLM_REVIEWER_ENABLED=true 但 LLM_REVIEWER_BASE_URL 未设置，审核器已禁用"
            )
            return None
        if not model:
            logger.warning(
                "LLM_REVIEWER_ENABLED=true 但 LLM_REVIEWER_MODEL 未设置，审核器已禁用"
            )
            return None

        def _bool(key: str, default: str = "true") -> bool:
            return os.environ.get(key, default).strip().lower() in ("true", "1", "yes")

        try:
            return cls(
                model=model,
                api_key=api_key,
                base_url=base_url,
                temperature=float(os.environ.get("LLM_REVIEWER_TEMPERATURE", "0.1")),
                max_tokens=int(os.environ.get("LLM_REVIEWER_MAX_TOKENS", "8192")),
                confidence_threshold=float(
                    os.environ.get("LLM_REVIEWER_CONFIDENCE_THRESHOLD", "0.8")
                ),
                max_retries=int(os.environ.get("LLM_REVIEWER_MAX_RETRIES", "2")),
                timeout=int(os.environ.get("LLM_REVIEWER_TIMEOUT", "120")),
                enable_reorder=_bool("LLM_REVIEWER_ENABLE_REORDER"),
                enable_ocr_fix=_bool("LLM_REVIEWER_ENABLE_OCR_FIX"),
                disable_thinking=_bool("LLM_REVIEWER_DISABLE_THINKING", "true"),
            )
        except Exception as e:
            logger.warning("LLM 审核器初始化失败，已禁用：%s", e)
            return None

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    def review(self, raw_json_pages: list) -> Tuple[list, dict]:
        """审核 OCR 原始 JSON，返回 (reviewed_pages, report)。

        Args:
            raw_json_pages: _build_raw_json() 输出，列表的列表：
                            [[{index, label, content, bbox_2d, polygon}, ...], ...]

        Returns:
            reviewed_pages: 与 raw_json_pages 同结构，已应用纠错。
            report:         包含改动清单、延迟、模型等元信息的字典。
        """
        if not raw_json_pages:
            return raw_json_pages, self._empty_report()

        flat_items = self._to_flat(raw_json_pages)
        t0 = time.time()
        last_error: Optional[str] = None

        for attempt in range(self.max_retries + 1):
            try:
                response_data = self._call_llm(flat_items)
                raw_changes: list = response_data.get("changes", [])

                # 兼容旧格式：如果 LLM 仍返回了 corrected_items，忽略
                if not raw_changes and response_data.get("corrected_items"):
                    logger.warning("LLM 返回了旧格式 corrected_items，已忽略")

                if not self._validate_changes(raw_changes, flat_items):
                    raise ValueError("验证失败：changes 引用了不存在的 item")

                approved_changes = self._filter_changes(raw_changes)
                corrected_flat = self._apply_approved_changes(
                    flat_items, approved_changes
                )
                reviewed_pages = self._rebuild_pages(corrected_flat, raw_json_pages)

                report = {
                    "model": self.model,
                    "base_url": self.base_url,
                    "total_items": len(flat_items),
                    "total_changes": len(approved_changes),
                    "reorder_count": sum(
                        1 for c in approved_changes if c.get("type") == "reorder"
                    ),
                    "ocr_fix_count": sum(
                        1 for c in approved_changes if c.get("type") == "ocr_fix"
                    ),
                    "changes": approved_changes,
                    "latency_ms": int((time.time() - t0) * 1000),
                    "retry_count": attempt,
                }
                logger.info(
                    "LLM 审核完成：%d 处改动（%d reorder / %d ocr_fix），耗时 %.1fs",
                    len(approved_changes),
                    report["reorder_count"],
                    report["ocr_fix_count"],
                    time.time() - t0,
                )
                return reviewed_pages, report

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    "LLM 审核第 %d/%d 次失败：%s",
                    attempt + 1,
                    self.max_retries + 1,
                    e,
                )
                if attempt < self.max_retries:
                    time.sleep(1.5 * (attempt + 1))

        report = self._empty_report()
        report.update(
            {
                "latency_ms": int((time.time() - t0) * 1000),
                "retry_count": self.max_retries,
                "error": last_error,
            }
        )
        msg = f"LLM 审核全部重试失败，提取失败：{last_error}"
        logger.error(msg)
        raise LLMReviewError(msg, report=report)

    def apply_to_grouped(self, grouped: list, reviewed_pages: list) -> list:
        """将审核纠错结果应用回 pipeline 的 grouped 结构。

        Args:
            grouped:        Pipeline 原始 grouped（含 score/task_type 等额外字段）。
            reviewed_pages: review() 返回的纠错后页面列表。

        Returns:
            纠错后的 grouped，可直接传入 result_formatter.process()。
        """
        if not reviewed_pages:
            return grouped

        # 构建 (page_idx, bbox_tuple) -> reviewed_item 查找表
        lookup: Dict[Tuple, dict] = {}
        for page_idx, page in enumerate(reviewed_pages):
            for item in page:
                bbox = tuple(item.get("bbox_2d") or [])
                lookup[(page_idx, bbox)] = item

        new_grouped = []
        for page_idx, page_regions in enumerate(grouped):
            new_page = []
            for region in page_regions:
                bbox = tuple(region.get("bbox_2d") or [])
                key = (page_idx, bbox)
                reviewed = lookup.get(key)

                new_region = dict(region)
                if reviewed is not None:
                    # 仅在内容实际发生改变时应用（null 图片区域不改）
                    rev_content = reviewed.get("content")
                    orig_content = region.get("content")
                    if rev_content is not None and rev_content != orig_content:
                        new_region["content"] = rev_content
                    new_region["_reviewer_sort"] = reviewed.get(
                        "index", region.get("index", 0)
                    )
                else:
                    new_region["_reviewer_sort"] = region.get("index", 0)
                new_page.append(new_region)

            # 按审核器给出的 index 重排，再重新连续编号
            new_page.sort(key=lambda r: r.get("_reviewer_sort", 0))
            for i, r in enumerate(new_page):
                r.pop("_reviewer_sort", None)
                r["index"] = i
            new_grouped.append(new_page)

        return new_grouped

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _to_flat(self, raw_json_pages: list) -> list:
        """将嵌套页面结构展平，加入 page 字段。"""
        flat = []
        for page_idx, page_items in enumerate(raw_json_pages):
            for item in page_items:
                flat.append(
                    {
                        "page": page_idx,
                        "index": item.get("index"),
                        "label": item.get("label"),
                        "content": item.get("content"),
                        "bbox_2d": item.get("bbox_2d"),
                    }
                )
        return flat

    def _rebuild_pages(self, corrected_flat: list, original_pages: list) -> list:
        """将展平的纠错结果重建为 raw_json_pages 同结构（含 polygon）。"""
        # 从原始页面中提取 polygon（LLM 输出不含此字段）
        polygon_lookup: dict = {}
        for page_idx, page_items in enumerate(original_pages):
            for item in page_items:
                bbox = tuple(item.get("bbox_2d") or [])
                polygon_lookup[(page_idx, bbox)] = item.get("polygon")

        # 按页分组
        page_map: Dict[int, list] = defaultdict(list)
        for item in corrected_flat:
            page_map[item.get("page", 0)].append(item)

        reviewed = []
        for page_idx in range(len(original_pages)):
            page_items = sorted(
                page_map.get(page_idx, []), key=lambda x: x.get("index", 0)
            )
            reviewed_page = []
            for i, item in enumerate(page_items):
                bbox = tuple(item.get("bbox_2d") or [])
                reviewed_page.append(
                    {
                        "index": i,
                        "label": item.get("label"),
                        "content": item.get("content"),
                        "bbox_2d": item.get("bbox_2d"),
                        "polygon": polygon_lookup.get((page_idx, bbox)),
                    }
                )
            reviewed.append(reviewed_page)
        return reviewed

    def _filter_changes(self, changes: list) -> list:
        """按类型开关和置信度阈值过滤改动。"""
        approved = []
        for change in changes:
            ctype = change.get("type", "")
            if ctype == "reorder":
                if self.enable_reorder:
                    approved.append(change)
            elif ctype == "ocr_fix":
                conf = float(change.get("confidence", 0))
                if not (self.enable_ocr_fix and conf >= self.confidence_threshold):
                    logger.debug(
                        "拒绝低置信度 ocr_fix（%.2f < %.2f）：%s",
                        conf,
                        self.confidence_threshold,
                        change.get("reason", ""),
                    )
                    continue

                before = change.get("before")
                after = change.get("after")
                if not isinstance(before, str) or not isinstance(after, str):
                    logger.debug("拒绝 ocr_fix：before/after 不是字符串")
                    continue

                before_stripped = before.strip()
                after_stripped = after.strip()
                if before_stripped and len(after_stripped) < int(len(before_stripped) * 0.85):
                    logger.debug(
                        "拒绝 ocr_fix：疑似截断（before=%d, after=%d）",
                        len(before_stripped),
                        len(after_stripped),
                    )
                    continue

                missing_markers = [
                    m for m in _SECTION_MARKERS if m in before and m not in after
                ]
                if missing_markers:
                    logger.debug(
                        "拒绝 ocr_fix：丢失段落标记 %s", ",".join(missing_markers)
                    )
                    continue

                approved.append(change)
        return approved

    def _apply_approved_changes(self, original: list, approved_changes: list) -> list:
        """在原始数据上直接应用已批准的改动。

        - reorder：将指定 item 移到目标 index，其余 item 保持相对顺序填空。
        - ocr_fix：按 (page, bbox) 精确定位，优先做子串替换（before→after），
          避免 LLM 返回局部片段时覆盖整段原文。
          替换策略（按优先级）：
            1. before 是原文精确子串 → str.replace(before, after, 1) 局部替换
            2. before 空白归一化后是原文子串 → regex 模糊替换（容忍多余空格）
            3. before 与原文完全相等 → 直接整段替换
            4. 全部失败 → 跳过并记录 warning，原文保持不变
        """
        # 构建 (page, bbox) → item 查找表
        items_by_key: Dict[tuple, dict] = {}
        for item in original:
            key = (item.get("page", 0), tuple(item.get("bbox_2d") or []))
            items_by_key[key] = dict(item)

        # 收集改动
        reorders: Dict[tuple, int] = {}
        ocr_fixes: Dict[tuple, Dict[str, str]] = {}
        for change in approved_changes:
            key = (change.get("page", 0), tuple(change.get("bbox_2d") or []))
            if change.get("type") == "reorder":
                try:
                    reorders[key] = int(change.get("after", -1))
                except (ValueError, TypeError):
                    pass
            elif change.get("type") == "ocr_fix":
                ocr_fixes[key] = {
                    "before": change.get("before", ""),
                    "after": change.get("after", ""),
                }

        # 应用 ocr_fix（子串替换优先）
        for key, fix in ocr_fixes.items():
            if key not in items_by_key:
                continue
            item = items_by_key[key]
            original_content: str = item.get("content") or ""
            before: str = fix["before"]
            after: str = fix["after"]

            if before in original_content:
                # 策略1：精确子串匹配
                item["content"] = original_content.replace(before, after, 1)
                logger.debug("ocr_fix：精确子串替换（page=%s bbox=%s）", key[0], key[1])
            elif _fuzzy_replace(original_content, before, after) is not None:
                # 策略2：空白归一化后匹配（容忍多余空格）
                item["content"] = _fuzzy_replace(original_content, before, after)
                logger.debug("ocr_fix：模糊子串替换（page=%s bbox=%s）", key[0], key[1])
            elif original_content == before:
                # 策略3：整段完全相等
                item["content"] = after
                logger.debug("ocr_fix：整段替换（page=%s bbox=%s）", key[0], key[1])
            else:
                logger.warning(
                    "ocr_fix 跳过：before 在原文中未找到，原文保持不变（page=%s bbox=%s before=%r）",
                    key[0],
                    key[1],
                    before[:80],
                )

        if not reorders:
            return list(items_by_key.values())

        # 按页分组，应用 reorder
        pages: Dict[int, list] = defaultdict(list)
        for key, item in items_by_key.items():
            page = key[0]
            if key in reorders:
                # (priority, sort_index, item) — moved items win on index conflict
                pages[page].append((0, reorders[key], item))
            else:
                pages[page].append((1, item.get("index", 0), item))

        result = []
        for page_idx in sorted(pages.keys()):
            page_items = pages[page_idx]
            page_items.sort(key=lambda x: (x[1], x[0]))
            for i, (_, _, item) in enumerate(page_items):
                item["index"] = i
                result.append(item)

        return result

    def _call_llm(self, flat_items: list) -> dict:
        """调用 LLM API，返回解析后的 JSON 字典。"""
        items_json = json.dumps(flat_items, ensure_ascii=False, indent=2)
        user_prompt = _USER_PROMPT_TEMPLATE.format(items_json=items_json)

        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }

        if self.disable_thinking:
            # DeepSeek V4：关闭 thinking，避免 finish_reason=length 且 content 为空
            payload["extra_body"] = {"thinking": {"type": "disabled"}}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        logger.info(
            "LLM 审核请求：%d 个 item，输入约 %d 字符，max_tokens=%d",
            len(flat_items),
            len(user_prompt),
            self.max_tokens,
        )

        endpoint = f"{self.base_url}/chat/completions"
        resp = self._session.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )

        if resp.status_code != 200:
            raise ValueError(f"LLM API 返回状态 {resp.status_code}：{resp.text[:400]}")

        resp_data = resp.json()
        content: str = resp_data["choices"][0]["message"]["content"]
        finish_reason = resp_data["choices"][0].get("finish_reason", "unknown")

        logger.info(
            "LLM 返回：finish_reason=%s，长度=%d 字符", finish_reason, len(content)
        )
        logger.debug("LLM 原始返回内容:\n%s", content)

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
            if match:
                return json.loads(match.group(1))

            logger.error("=" * 60)
            logger.error("LLM 返回内容解析失败 — finish_reason: %s", finish_reason)
            logger.error("返回内容长度: %d 字符", len(content))
            logger.error("完整返回内容:\n%s", content)
            logger.error("=" * 60)
            raise ValueError(
                f"LLM 返回内容不是合法 JSON（finish_reason={finish_reason}，"
                f"长度={len(content)}）：{content[-200:]}"
            )

    def _validate_changes(self, changes: list, original: list) -> bool:
        """校验 changes 中每条引用的 (page, bbox) 都在原始数据中存在。"""
        if not changes:
            return True
        valid_keys = {
            (item.get("page"), tuple(item.get("bbox_2d") or [])) for item in original
        }
        for change in changes:
            key = (change.get("page"), tuple(change.get("bbox_2d") or []))
            if key not in valid_keys:
                logger.warning(
                    "change 引用的 item 不存在：page=%s, bbox=%s",
                    change.get("page"),
                    change.get("bbox_2d"),
                )
                return False
        return True

    def _empty_report(self) -> dict:
        return {
            "model": self.model,
            "base_url": self.base_url,
            "total_items": 0,
            "total_changes": 0,
            "reorder_count": 0,
            "ocr_fix_count": 0,
            "changes": [],
            "latency_ms": 0,
            "retry_count": 0,
        }
