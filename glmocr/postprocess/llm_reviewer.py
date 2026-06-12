"""LLM-based OCR result reviewer (multimodal / 看图校对).

审核 OCR 原始输出（model_raw.json），对照**原图**执行四类纠错。
**采用分阶段流水线**：每个阶段单独调用一次模型，模型只做一种任务、不越界；返回后
代码立即应用并重新连续编号 index，再把结果交给下一阶段。执行顺序固定为：
  1. insert     — 漏行补全（layout 漏框、没进 OCR 的整行）
  2. ocr_fix    — 识别错误纠错（含角标/上下标补回，对照原图逐字校对）
  3. dedup_fix  — 切分重叠去重（在 reorder 前，保留相邻关系；在 ocr_fix 后，文字已规整）
  4. reorder    — 阅读顺序纠错（在去重后的干净序列上判序，减少误判换位）

分阶段的好处：避免多任务混在一次请求里互相干扰，也避免跨任务 index 错位
（每个阶段看到的都是上一阶段应用后重新编号的连续 index）。

输入：带 bbox 的 raw_json（给模型空间上下文）+ 对应页原图。
输出：按 page + index 定位的 changes，不含 bbox。

配置完全通过环境变量（见 .env.example），与现有流程零耦合。

使用方式（在 Pipeline 中）：
    from glmocr.postprocess.llm_reviewer import LLMReviewer
    reviewer = LLMReviewer.from_env()   # None if disabled
    if reviewer:
        reviewed_pages, report = reviewer.review(raw_json, page_images)
        new_grouped = reviewer.apply_to_grouped(grouped, reviewed_pages)
"""

from __future__ import annotations

import base64
import difflib
import io
import json
import os
import re
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

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

# insert 允许的 label（纯文本类）；禁止 image / chart / table。
_INSERT_ALLOWED_LABELS = {
    "text",
    "formula",
    "figure_title",
    "paragraph_title",
    "doc_title",
    "header",
    "footer",
    "abstract",
    "content",
    "reference_content",
    "vertical_text",
    "formula_number",
}
_INSERT_FORBIDDEN_LABELS = {"image", "chart", "table"}
# insert 置信度下限（与 prompt 一致）
_INSERT_MIN_CONFIDENCE = 0.7


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


def _has_boundary_overlap(left: str, right: str, min_len: int = 2) -> bool:
    """判断 left 的末尾与 right 的开头是否在切分边界处重叠（允许 OCR/空白差异）。"""
    if not left or not right:
        return False
    max_len = min(len(left), len(right))
    for n in range(max_len, min_len - 1, -1):
        if _texts_equivalent(left[-n:], right[:n]):
            return True
    return False


def _normalize_ws(text: str) -> str:
    """把连续空白归一化为单个空格并去首尾空白，便于宽松比对。"""
    return re.sub(r"\s+", " ", text).strip()


def _first_sentence(text: str) -> str:
    """取文本首句（按换行或句末标点切分）。"""
    if not text:
        return ""
    parts = re.split(r"[\n。！？!?]", text, maxsplit=1)
    return parts[0].strip()


def _last_sentence(text: str) -> str:
    """取文本末句（按换行或句末标点切分）。"""
    if not text:
        return ""
    for part in reversed(re.split(r"[\n。！？!?]", text)):
        if part.strip():
            return part.strip()
    return text.strip()


def _is_insert_already_in_adjacent(
    content: str, after_index: int, content_by_index: Dict[Any, Any]
) -> bool:
    """拟 insert 的内容是否已出现在 after_index 上下相邻 item 中（含合并进正文中间）。

    先查 index=X 与 index=X+1 的**完整 content**；并着重核对上面项末句、下面项首句。
    允许空格/标点轻微差异。
    """
    needle = _normalize_ws(content)
    if len(needle) < 2:
        return False
    needle_core = re.sub(r"[ .，。；、]+$", "", needle)

    neighbors: List[Optional[str]] = []
    if after_index >= 0:
        neighbors.append(content_by_index.get(after_index))
    neighbors.append(content_by_index.get(after_index + 1))

    for neighbor in neighbors:
        if not isinstance(neighbor, str) or not neighbor.strip():
            continue
        hay = _normalize_ws(neighbor)
        if needle in hay or (needle_core and needle_core in hay):
            return True
        first = _normalize_ws(_first_sentence(neighbor))
        last = _normalize_ws(_last_sentence(neighbor))
        for segment in (first, last, hay):
            if segment and (
                _texts_equivalent(needle, segment)
                or (needle_core and needle_core in segment)
            ):
                return True
    return False


def _texts_equivalent(a: str, b: str, threshold: float = 0.75) -> bool:
    """两段文本是否“基本等价”（用于判定头/尾整段重复）。

    归一化空白后：完全相等、互为前缀/被包含，或字符相似度 >= 阈值即视为等价。
    重复副本常因二次 OCR 略有出入（错字/空格），故用模糊比对而非严格相等。
    """
    na, nb = _normalize_ws(a), _normalize_ws(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    shorter, longer = (na, nb) if len(na) <= len(nb) else (nb, na)
    # 短串足够长且被长串前缀/包含，视为同段重复。
    if len(shorter) >= 4 and (longer.startswith(shorter) or shorter in longer):
        return True
    return difflib.SequenceMatcher(None, na, nb).ratio() >= threshold


def _is_adjacent_layout_dedup(
    change: dict, content_by_index: Dict[Any, Any]
) -> bool:
    """dedup_fix 仅允许处理**相邻 item** 因切分框重叠产生的边界重复。

    拒绝：与非相邻 item 内容相似/相同（如【详解】末尾复述【答案】）而整项清空。
    """
    idx = change.get("index")
    if idx is None:
        return False
    before = change.get("before")
    after = change.get("after")
    if not isinstance(before, str) or not isinstance(after, str):
        return False

    prev_c = content_by_index.get(idx - 1)
    next_c = content_by_index.get(idx + 1)
    if prev_c is None and next_c is None:
        return False

    prev_s = prev_c if isinstance(prev_c, str) else ""
    next_s = next_c if isinstance(next_c, str) else ""

    # 整项清空：当前短项整体是相邻项开头/结尾的重复副本（允许轻微 OCR 差异）。
    if after == "":
        if prev_s and (
            _texts_equivalent(before, prev_s)
            or _texts_equivalent(before, prev_s[-len(before) :])
        ):
            return True
        if next_s and (
            _texts_equivalent(before, next_s)
            or _texts_equivalent(before, next_s[: len(before)])
        ):
            return True
        return False

    if after == before or len(after) >= len(before):
        return False
    # 切分边界重叠：上一项尾 == 当前项头，或当前项尾 == 下一项头。
    if prev_s and _has_boundary_overlap(prev_s, before):
        return True
    if next_s and _has_boundary_overlap(before, next_s):
        return True
    # 头/尾整段重复：当前长项的开头(或结尾)整段复制了相邻短项的全文，
    # dedup 把这段重复前缀(后缀)删掉。判定：被删掉的那段 == 相邻项全文。
    removed = before[: len(before) - len(after)] if before.endswith(after) else ""
    removed_tail = before[len(after):] if before.startswith(after) else ""
    if removed and prev_s and _texts_equivalent(removed, prev_s):
        return True
    if removed and next_s and _texts_equivalent(removed, next_s):
        return True
    if removed_tail and next_s and _texts_equivalent(removed_tail, next_s):
        return True
    if removed_tail and prev_s and _texts_equivalent(removed_tail, prev_s):
        return True
    return False


class LLMReviewError(RuntimeError):
    """LLM 审核在全部重试后仍失败；调用方应中止本次 OCR，勿回退原始输出。"""

    def __init__(self, message: str, report: Optional[dict] = None):
        super().__init__(message)
        self.report = report or {}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 分阶段 Prompt（每个阶段只做一种任务，模型不越界）
#
# 执行顺序：insert → ocr_fix → dedup_fix → reorder。
# 每个阶段单独调用一次模型，模型只输出该阶段对应的一种 change；代码立即应用、
# 重新连续编号 index，再把结果交给下一个阶段。因此每个阶段看到的 index 始终是
# “本阶段输入时”的连续序号（0 起），不存在跨任务的 index 错位问题。
# ---------------------------------------------------------------------------

# 所有阶段共用的开场白：角色 + 输入字段 + 单任务纪律。
_PROMPT_HEADER = """\
你是专业的OCR结构化审核员，专注于数学和教育类文档。你能**看到该页的原图**，请始终对照原图进行校对。

输入是该页的 OCR 识别结果（若干 item，每个含 index / label / content / bbox_2d）。
- index: 本页当前的排序序号（从 0 开始，连续）。所有定位都用这个 index。
- label: 区域类型（text / formula / image / figure_title / header 等）。
- content: 识别内容（null 表示图片区域）。
- bbox_2d: 边界框 [x1, y1, x2, y2]，归一化 0-1000；**y 越大越靠下，x 越大越靠右**，仅作辅助参考。

**本次只负责一个专门任务，严禁做任何其他类型的修改。** 不要输出 bbox。
**index 一律使用本次输入数据中的序号，禁止使用你脑内重排后的序号。**\
"""

# ocr_fix / dedup_fix 公用的 before/after 严格要求。
_BEFORE_AFTER_RULES = """\
【关于 before / after 字段的严格要求】
- before 必须与输入数据中对应 item 的 content 字段完全一致，一字不差、一标点不差、一空格不差、一换行不差。
- 禁止对 before 做任何改动、简化、省略或重新排版。
- 禁止改变 before 中反斜杠的数量：输入里有几个反斜杠，before 里就填几个，不得增减。
- before 必须从输入 content 中原文复制，不得凭记忆或推断填写。
- after 的反斜杠转义方式与 before 保持一致，仅修改目标字符。
- 若 before 含结构标记（如【答案】【分析】【详解】），after 必须完整保留这些标记。
- **before 与 after 完全相同说明无需修改，禁止输出该项**（不要输出“核对后无需改”之类的无效条目）。\
"""

# ── 阶段1：insert ──────────────────────────────────────────────────────────
_SYSTEM_PROMPT_INSERT = (
    _PROMPT_HEADER
    + """

【本阶段唯一任务】漏行补全（insert）
你的目标是**找回 layout 漏框、没进 OCR 的整行/整块文字**。OCR 经常漏掉单独成行的公式、推导步骤、过渡句、标题等——请**主动、逐行**地把它们找出来补回，不要默认 OCR 已经识别全了。

**怎么发现漏行（按此主动排查）**：
1. **对照原图逐行扫描**：从上到下把原图里**每一行清晰可读的文字**与 items 列表逐一对应；原图里有、items 里却找不到对应 item 的整行，就是漏行。
2. **用 bbox 纵向空隙定位**：相邻两个 item 的 bbox 在 y 方向出现**明显空隙**（两项 y 间距明显大于正常行高/行距）时，该空隙处**很可能漏了一行**，重点对照原图确认。
3. 数学/解析题里**单独成行的公式、推导步骤（如 $\\because$ / $\\therefore$ 开头的行）、计算结果行**最常被漏，尤其要留意。

**确认是漏行后再补**：
- 用 after_index 指明插在哪个 item 后面：after_index = X 表示插在 index 为 X 的 item 之后；若该行应排在所有 item 之前，after_index 填 -1；漏在末尾则填最后一个 item 的 index。
- content 填写原图中该行的完整识别内容，非空。
- 同一处漏多行时，按从上到下的顺序对同一 after_index 输出多条 insert。
- 只补纯文本类内容（label 取 text / formula / figure_title / header 等），**禁止补 image / chart / table**。

**避免误补（只排除以下情况，其余该补就补）**：
- **已被相邻项合并的不算漏行**：确定 after_index = X 后，必须检查上面项(index=X)与下面项(index=X+1)的**完整 content**（不只首句/末句）；若拟补整行已作为子串出现在其中任一完整正文里（允许空格/标点轻微差异），说明 OCR 已识别、只是与邻项合并成一段，**不要 insert**。**着重**再核对上面项的**末句**与下面项的**首句**，这两处是最常见的合并位置。
- **顺序颠倒不算漏行（交给后续调序阶段）**：若拟补整行与 index=X 或 index=X+1 的**完整 content 基本相同**（整行其实已作为相邻 item 存在、只是 index 顺序不对），这是阅读顺序问题，**不要 insert**，留给后续调序阶段。**只允许比对上下相邻 1 个 item**，禁止在更远范围搜相同文本。
- **行内少字/错字/符号误识不归本阶段**：那是 ocr_fix 的事，不要用 insert 去补几个字。
- **页脚/操作栏 UI 噪声不补**（后处理会自动剔除，漏掉属正常）：①组卷统计「今日|3次组卷」「7日内…次组卷」「您最近一年使用…」；②页脚链接「相似题/纠错/详情/收藏」任意组合；③「加入试题篮」。拟补内容仅由这些构成时，一律不 insert。

- **本阶段只输出 insert**，不要纠错、不要调序、不要去重。

硬性约束：
1. 输出必须是严格合法 JSON，不含任何额外自然语言或 markdown 标记，不要输出 bbox。
2. 原图里清晰可见且 items 中确实缺失的整行，应当补（confidence 需 ≥ 0.7）；仅当原图辨认不清或无法判断插入位置时才不补。
3. 确认本页确无漏行时才输出空数组——但在此之前务必已逐行对照过原图。\
"""
)

# ── 阶段2：ocr_fix ─────────────────────────────────────────────────────────
_SYSTEM_PROMPT_OCR_FIX = (
    _PROMPT_HEADER
    + """

【本阶段唯一任务】识别错误纠错（ocr_fix）
- 对照原图，**逐 item、逐行**检查 content 是否存在**文字识别错误**（错字、漏字、多字、符号误识等）。
- **角标/上下标/撇号补回是本任务的重中之重**：
  * 典型错误：∠AGA' 被识别成 ∠AGA、A_1 被识别成 A、x² 被识别成 x2、P' 被识别成 P、点 A' 被识别成点 A 等。
  * **每一行都必须认真校对角标**：凡含角度符号（∠）、点标记（A/B/P 等）、变量、公式、撇号、上下标的位置，
    都要逐字对照原图，确认 OCR 是否漏识或误识了 '、_、^、₂、² 等标记。
  * 不得因“这一行看起来没问题”就跳过；**本页每一个文本/公式 item 都要做角标专项检查**。
  * **触发严格校对规则**：一旦在本页任意一处发现角标/上下标/撇号识别缺漏或错误，
    必须对该页**所有 item 重新逐条严格校对**，不可只改一处或只改一行就停止。
- **同一 index 可能有多处识别错误**：例如一行里既有 ∠AGA 缺撇号，又有别的错字；
  必须在**同一条** ocr_fix 的 after 里**全部**改完，不要只修第一个发现的问题就输出。
- 本任务涵盖一切需对照原图改字的纠错，包括肉眼可见的识别错误，以及变量名混淆、等式明显矛盾等**实质为识别失误**的问题。
- 改错必须以**原图比对**为准；可用**逻辑推理**辅助判断某处是否确为识别错误、拟修改是否会使前后文矛盾（如改后变量名是否与正文求解过程一致、等式是否自洽）。
  逻辑上说不通或原图不支持的不要改。不做无关的排版美化。
- 严禁以下“伪纠错”（原图与 OCR 在字符层面一致时，一律不要改）：
  * 不得统一/调整 LaTeX 公式内空格、$ 与内容之间的空格（如 $ \\sqrt{2} $ 改成 $\\sqrt{2}$）。
  * 不得重排 $ 符号包裹方式（如把 $ P A $ 改成 $PA$）。
  * 不得改写语气、压缩段落、统一格式。
- 必须是**最小必要改动**：after 完整保留 before 的原文，仅修正识别错误的字符，不得截断、改写、压缩。
- content 为 null 的 item（图片区域）不允许做 ocr_fix。
- **本阶段只输出 ocr_fix**，不要调序、不要补行、不要去重。

"""
    + _BEFORE_AFTER_RULES
    + """

硬性约束：
1. 输出必须是严格合法 JSON，不含任何额外自然语言或 markdown 标记，不要输出 bbox。
2. 低把握不改（confidence 需 ≥ 0.7）。无需修改时输出空数组。
3. 禁止因已输出一条就停止检查其余 item；同一 item 多处识别错误合并到一条 ocr_fix 一次改全。\
"""
)

# ── 阶段3：dedup_fix ───────────────────────────────────────────────────────
_SYSTEM_PROMPT_DEDUP = (
    _PROMPT_HEADER
    + """

【本阶段唯一任务】去重（dedup_fix）
- **仅处理相邻 item（index 相差 1）在切分边界处的重叠**，即 layout 两框重叠导致同一段文字被切成两半的「临界重复」：
  * 典型：上一 item 末尾与下一 item 开头出现相同片段 → 只删其中一侧的重叠片段。
  * 典型：当前 item 的**全部** content 恰好是**紧邻**上一 item 末尾或下一 item 开头的重复尾巴/脑袋 → 可将当前 item 设为 ""。
  * 典型：当前短 item 的全文，等于相邻长 item 的开头或结尾（允许少量 OCR 差异，如空格/个别错字）→ 这是重复副本，允许清空当前短 item。
  * 判定标准：**逻辑上明显是同一段内容、明显是重复识别即可**，不要求每个字完全一致（但必须是相邻 index±1 的边界重复，不能扩展到远处内容）。
- **禁止以下误判为去重**：
  * 当前 item 与**非相邻** item（隔了多个 index）内容相同或相似 —— 不算重复，不要删。
    例：【答案】里的结论，在【详解】后面又以「答：…」复述 —— 这是正常结构，**禁止** dedup_fix。
  * 仅因「全文看起来和前面某处一样」就整项清空 —— 必须确认是**与 index±1 的边界重叠**，否则不要删。
- **同一处重复只处理一侧，禁止两侧都删**：
  * 若上方 item 末尾与下方 item 开头临界重叠 → 只删上方末尾或下方开头其中一侧，**另一侧保留完整原文**。
  * 同一对相邻 item 最多输出一条 dedup_fix。
- dedup_fix 的 after 允许比 before 短很多甚至为空，但不得改正文识别错误。
- content 为 null 的 item（图片区域）不允许做 dedup_fix。
- 不确定是否为切分边界重叠时不要删。
- **本阶段只输出 dedup_fix**，不要改字、不要调序、不要补行。

"""
    + _BEFORE_AFTER_RULES
    + """

硬性约束：
1. 输出必须是严格合法 JSON，不含任何额外自然语言或 markdown 标记，不要输出 bbox。
2. 低把握不删（confidence 需 ≥ 0.65）。无需去重时输出空数组。\
"""
)

# ── 阶段4：reorder ─────────────────────────────────────────────────────────
_SYSTEM_PROMPT_REORDER = (
    _PROMPT_HEADER
    + """

【本阶段唯一任务】阅读顺序纠错（reorder）
- 对照原图**并结合 bbox_2d 坐标**，检查 items 的 index 阅读顺序是否与图中实际阅读顺序一致。
- 切分模型经常赋予错误的 index；仅在顺序**明显不合理**时才输出 reorder，把握不足时不要调整。

**通用阅读顺序**：从上到下、从左到右；多栏布局时按列从左到右，列内从上到下。结合 bbox_2d 判断各 item 的相对位置，仅在顺序**明显不合理**时才 reorder。

- 不需要为每个 item 都输出 reorder，只输出需要调整 index 的 item。
- 通过 index 定位要移动的 item，new_index 填写该 item 在**重排后整页序列里最终占据的序号**（0 起）。
- **new_index 是「最终位置编号」，不是「插在谁后面」**：把所有 item 想象成重排后从 0 开始重新数一遍，该 item 数到第几个，new_index 就是几。
- 若 item 已在正确位置，则 new_index 等于其当前 index，此时**不要输出**该条 reorder（直接输出空数组）。
- **输出前强制自检（必须通过）**：
  * 若你的 reason 含“应排在 index K 之后/后面”，则 new_index 必须是 K+1（不是 K）。
  * 若你的 reason 含“原顺序合理/无需调整/不需要移动”，则不得输出该条 reorder，必须输出空数组。
- 示例（仅用于理解 new_index 语义）：
  * 正确：index=3 的 item 应排在 index=2 后面 -> new_index=3（重排后它是第 3 个）。
  * 错误：index=3 的 item 应排在 index=2 后面 -> new_index=2（这是占了 2 的位置，会把顺序弄反）。
- 重要：输入数据可能来自“答案图片”。答案图片中【答案】→【分析】→【详解】是固定的正确顺序
  （并非所有标注都会出现，但出现时顺序必定如此）。若已按此顺序排列，绝对不可对这几个标注之间的顺序做 reorder。
- 仅允许用“显式标签文本”判断该规则：只有当 content 明确包含【答案】/【分析】/【详解】标记时，才可据此判断标签顺序。
- 严禁语义归类重判：不得因为“看起来像详解/分析/答案”就把某段无标签文本判为其他版块并 reorder。
- **本阶段只输出 reorder**，不要改 content、不要补行、不要去重；只调整顺序。

硬性约束：
1. 输出必须是严格合法 JSON，不含任何额外自然语言或 markdown 标记，不要输出 bbox。
2. reorder 不需要 before 字段；new_index 为整数最终序号（0 起）。
3. 低把握不调（confidence 需 ≥ 0.7）。无需调整时输出空数组。\
"""
)


# 各阶段 user 模板（公用输入区，仅输出 schema 不同）。
_USER_TEMPLATE_INSERT = """\
这是文档第 {page} 页的原图与 OCR 识别结果。请对照原图**逐行扫描**，主动找出 layout 漏框、没进 OCR 的整行/整块文字，**只做漏行补全（insert）**。
重点关注：①相邻 item 的 bbox 在 y 方向出现明显空隙处；②单独成行的公式、推导步骤、计算结果行；③原图清晰可见但 items 里找不到对应项的整行。
补行前必须核对 after_index=X 时上下相邻两项（index=X 与 X+1）的**完整 content**：若拟补整行已出现在其中任一完整正文里（允许空格/标点差异），说明已合并识别，**不要 insert**；并**着重**核对上面项**末句**、下面项**首句**。
若拟补整行与相邻某项全文基本相同、只是顺序颠倒，那是顺序问题，**不要 insert**，留给后续调序阶段。

输入数据（该页 items）：
{items_json}

示例（原图在 item 2 与 item 3 之间还有一行公式 $\\therefore x=2$ 未被识别）：
{{"changes": [{{"type": "insert", "page": {page}, "after_index": 2, "label": "formula", "content": "$\\therefore x=2$", "reason": "原图此处单独成行的公式未进 OCR，上下相邻项均无该内容", "confidence": 0.8}}]}}

请严格按以下 JSON 格式输出（仅 insert，无需补行时输出 {{"changes": []}}）：
{{
  "changes": [
    {{
      "type": "insert",
      "page": {page},
      "after_index": 插在此 index 之后（-1 表示插到最前）,
      "label": "text 或 formula 等文本类标签",
      "content": "原图中该行的完整内容",
      "reason": "简短说明（中文）",
      "confidence": 0到1之间的小数
    }}
  ]
}}\
"""

_USER_TEMPLATE_REORDER = """\
这是文档第 {page} 页的原图与 OCR 识别结果。请对照原图并结合 bbox_2d，**只做阅读顺序纠错（reorder）**，只输出需要移动的 item。顺序明显不合理时才调整，不要过度 reorder。
注意：new_index 表示该 item 在**重排后整页序列里最终占据的序号**（0 起），不是“插在谁后面”。若判断“应排在 index K 后面”，则 new_index=K+1。
若判断“原顺序合理/无需调整”，必须输出 {{"changes": []}}，不要输出 reorder。

输入数据（该页 items）：
{items_json}

请严格按以下 JSON 格式输出（仅 reorder，无需调整时输出 {{"changes": []}}）：
{{
  "changes": [
    {{
      "type": "reorder",
      "page": {page},
      "index": 要移动的 item 的 index,
      "new_index": 该 item 重排后最终占据的序号（整数，0 起；不是「插在谁后面」，已在正确位置时不要输出）,
      "reason": "简短说明（中文）",
      "confidence": 0到1之间的小数
    }}
  ]
}}\
"""

_USER_TEMPLATE_OCR_FIX = """\
这是文档第 {page} 页的原图与 OCR 识别结果。请对照原图，**只做识别错误纠错（ocr_fix）**，逐 item、逐行核对角标/上下标/撇号。
同一 index 若有多处识别错误，合并到一条 ocr_fix 的 after 中一次改全；改完一处后须继续检查本页其余 item。

输入数据（该页 items）：
{items_json}

请严格按以下 JSON 格式输出（仅 ocr_fix，无需修改时输出 {{"changes": []}}）：
{{
  "changes": [
    {{
      "type": "ocr_fix",
      "page": {page},
      "index": 要修改的 item 的 index,
      "before": "从输入数据中原文逐字复制该 item 的 content，一字不差，反斜杠数量不变",
      "after": "修改后的完整 content，不得省略或截断任何未改动的部分",
      "reason": "简短说明（中文）",
      "confidence": 0到1之间的小数
    }}
  ]
}}\
"""

_USER_TEMPLATE_DEDUP = """\
这是文档第 {page} 页的原图与 OCR 识别结果。请**只做去重（dedup_fix）**，**仅**处理 index 相邻（index±1）item 在切分边界处的临界重叠。
不要因与远处 item 内容相同就删（如【详解】复述【答案】）；只有紧邻上一项末尾或下一项开头的重叠才去重。同一处重复只删一侧。
若当前短 item 的全文就是相邻长 item 的开头/结尾（允许少量 OCR 差异，如空格、个别错字），这属于重复副本，可将该短 item 去重为 ""。
去重判定以“逻辑上明显同一内容”为准，不要求逐字逐符号完全一致，但必须能明确定位为相邻项边界重复。

输入数据（该页 items）：
{items_json}

请严格按以下 JSON 格式输出（仅 dedup_fix，无需去重时输出 {{"changes": []}}）：
{{
  "changes": [
    {{
      "type": "dedup_fix",
      "page": {page},
      "index": 要去重的 item 的 index,
      "before": "从输入数据中原文逐字复制该 item 的 content",
      "after": "删除重复片段后的完整 content，可为空字符串",
      "reason": "简短说明（中文）",
      "confidence": 0到1之间的小数
    }}
  ]
}}\
"""

# 阶段执行顺序：先补行 → 再改字 → 再去重 → 最后调序。
# dedup 必须在 reorder 之前（reorder 会重排、打乱 dedup 依赖的相邻关系）；
# 且放在 ocr_fix 之后（文字先规整，重复的两份才更容易匹配上）；
# reorder 放最后，在去重后的干净序列上判序，减少误判换位。
_STAGE_ORDER = ("insert", "ocr_fix", "dedup_fix", "reorder")

# 阶段 → (system_prompt, user_template)
_STAGE_PROMPTS = {
    "insert": (_SYSTEM_PROMPT_INSERT, _USER_TEMPLATE_INSERT),
    "reorder": (_SYSTEM_PROMPT_REORDER, _USER_TEMPLATE_REORDER),
    "ocr_fix": (_SYSTEM_PROMPT_OCR_FIX, _USER_TEMPLATE_OCR_FIX),
    "dedup_fix": (_SYSTEM_PROMPT_DEDUP, _USER_TEMPLATE_DEDUP),
}


# ---------------------------------------------------------------------------
# LLMReviewer
# ---------------------------------------------------------------------------


class LLMReviewer:
    """Multimodal LLM-based reviewer for OCR structured results.

    从环境变量读取配置（见 .env.example），与现有流程零耦合。
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        confidence_threshold: float = 0.8,
        max_retries: int = 2,
        timeout: int = 120,
        enable_ocr_fix: bool = True,
        enable_reorder: bool = True,
        enable_dedup_fix: bool = True,
        enable_insert: bool = True,
        multimodal: bool = True,
        image_max_side: int = 1600,
        image_jpeg_quality: int = 85,
        response_format_json: bool = True,
        enable_curl_fallback: bool = True,
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
        self.enable_ocr_fix = enable_ocr_fix
        self.enable_reorder = enable_reorder
        self.enable_dedup_fix = enable_dedup_fix
        self.enable_insert = enable_insert
        self.multimodal = multimodal
        self.image_max_side = image_max_side
        self.image_jpeg_quality = image_jpeg_quality
        self.response_format_json = response_format_json
        self.enable_curl_fallback = enable_curl_fallback
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
                temperature=float(os.environ.get("LLM_REVIEWER_TEMPERATURE", "0")),
                max_tokens=int(os.environ.get("LLM_REVIEWER_MAX_TOKENS", "8192")),
                confidence_threshold=float(
                    os.environ.get("LLM_REVIEWER_CONFIDENCE_THRESHOLD", "0.8")
                ),
                max_retries=int(os.environ.get("LLM_REVIEWER_MAX_RETRIES", "2")),
                timeout=int(os.environ.get("LLM_REVIEWER_TIMEOUT", "120")),
                enable_ocr_fix=_bool("LLM_REVIEWER_ENABLE_OCR_FIX"),
                enable_reorder=_bool("LLM_REVIEWER_ENABLE_REORDER"),
                enable_dedup_fix=_bool("LLM_REVIEWER_ENABLE_DEDUP_FIX"),
                enable_insert=_bool("LLM_REVIEWER_ENABLE_INSERT"),
                multimodal=_bool("LLM_REVIEWER_MULTIMODAL", "true"),
                image_max_side=int(
                    os.environ.get("LLM_REVIEWER_IMAGE_MAX_SIDE", "1600")
                ),
                image_jpeg_quality=int(
                    os.environ.get("LLM_REVIEWER_IMAGE_JPEG_QUALITY", "85")
                ),
                response_format_json=_bool(
                    "LLM_REVIEWER_RESPONSE_FORMAT_JSON", "true"
                ),
                enable_curl_fallback=_bool(
                    "LLM_REVIEWER_ENABLE_CURL_FALLBACK", "true"
                ),
                disable_thinking=_bool("LLM_REVIEWER_DISABLE_THINKING", "true"),
            )
        except Exception as e:
            logger.warning("LLM 审核器初始化失败，已禁用：%s", e)
            return None

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    def review(
        self, raw_json_pages: list, page_images: Optional[list] = None
    ) -> Tuple[list, dict]:
        """对照原图审核 OCR 原始 JSON，返回 (reviewed_pages, report)。

        Args:
            raw_json_pages: _build_raw_json() 输出，列表的列表：
                            [[{index, label, content, bbox_2d, polygon}, ...], ...]
            page_images:    与 raw_json_pages 一一对应的每页原图（PIL.Image），
                            缺省或某页为 None 时该页退化为纯文本审核。

        Returns:
            reviewed_pages: 与 raw_json_pages 同结构，已应用纠错（含 insert）。
                            insert 出来的行 bbox_2d=None。
            report:         包含改动清单、延迟、模型等元信息的字典。
        """
        if not raw_json_pages:
            return raw_json_pages, self._empty_report()

        page_images = page_images or []
        t0 = time.time()
        total_items = sum(len(p) for p in raw_json_pages)

        reviewed_pages: List[list] = []
        all_approved: List[dict] = []
        total_retries = 0

        for page_idx, page_items in enumerate(raw_json_pages):
            image = page_images[page_idx] if page_idx < len(page_images) else None
            reviewed_page, approved, retries = self._review_page(
                page_idx, page_items, image
            )
            reviewed_pages.append(reviewed_page)
            all_approved.extend(approved)
            total_retries += retries

        report = {
            "model": self.model,
            "base_url": self.base_url,
            "multimodal": self.multimodal,
            "total_items": total_items,
            "total_changes": len(all_approved),
            "ocr_fix_count": sum(
                1 for c in all_approved if c.get("type") == "ocr_fix"
            ),
            "reorder_count": sum(
                1 for c in all_approved if c.get("type") == "reorder"
            ),
            "dedup_fix_count": sum(
                1 for c in all_approved if c.get("type") == "dedup_fix"
            ),
            "insert_count": sum(1 for c in all_approved if c.get("type") == "insert"),
            "changes": all_approved,
            "latency_ms": int((time.time() - t0) * 1000),
            "retry_count": total_retries,
        }
        logger.info(
            "LLM 审核完成（分阶段：insert→ocr_fix→dedup_fix→reorder）："
            "%d 处改动（%d insert / %d reorder / %d ocr_fix / %d dedup_fix），%d 页，耗时 %.1fs",
            report["total_changes"],
            report["insert_count"],
            report["reorder_count"],
            report["ocr_fix_count"],
            report["dedup_fix_count"],
            len(raw_json_pages),
            time.time() - t0,
        )
        return reviewed_pages, report

    def apply_to_grouped(self, grouped: list, reviewed_pages: list) -> list:
        """将审核纠错结果应用回 pipeline 的 grouped 结构（按 index 锚定 + 支持 insert）。

        reviewed_pages 中每页 item 的顺序即最终顺序：
          - 已存在的 item 通过 (page, bbox_2d) 匹配回 grouped region，保留 score /
            task_type / polygon 等额外字段，仅覆盖 content。
          - insert 出来的 item（bbox_2d=None，匹配不到 region）新建一个纯文本 region。
        最后每页按顺序重新连续编号 index。

        Args:
            grouped:        Pipeline 原始 grouped（含 score/task_type 等额外字段）。
            reviewed_pages: review() 返回的纠错后页面列表。

        Returns:
            纠错后的 grouped，可直接传入 result_formatter.process()。
        """
        if not reviewed_pages:
            return grouped

        new_grouped = []
        for page_idx, page_regions in enumerate(grouped):
            if page_idx >= len(reviewed_pages):
                new_grouped.append([dict(r) for r in page_regions])
                continue

            # (bbox_tuple) -> 原始 region 查找表
            lookup: Dict[tuple, dict] = {}
            for region in page_regions:
                bbox = region.get("bbox_2d")
                if bbox is not None:
                    lookup[tuple(bbox)] = region

            new_page = []
            for reviewed in reviewed_pages[page_idx]:
                bbox = reviewed.get("bbox_2d")
                rev_content = reviewed.get("content")
                region = lookup.get(tuple(bbox)) if bbox is not None else None

                if region is not None:
                    # 已存在 item：保留全部字段，仅在内容变化时覆盖 content。
                    new_region = dict(region)
                    orig_content = region.get("content")
                    # 图片区域 content 为 None，保持不变；文本区域应用纠错结果
                    # （含 dedup 清空为 ""）。
                    if rev_content is not None and rev_content != orig_content:
                        new_region["content"] = rev_content
                    new_page.append(new_region)
                else:
                    # insert 出来的新行：新建纯文本 region。
                    new_page.append(
                        {
                            "label": reviewed.get("label", "text"),
                            "content": rev_content if rev_content is not None else "",
                            "bbox_2d": None,
                        }
                    )

            for i, r in enumerate(new_page):
                r["index"] = i
            new_grouped.append(new_page)

        return new_grouped

    # ------------------------------------------------------------------
    # 单页审核
    # ------------------------------------------------------------------

    def _stage_enabled(self, stage: str) -> bool:
        """该阶段是否启用（由对应 enable_* 开关控制）。"""
        return {
            "insert": self.enable_insert,
            "reorder": self.enable_reorder,
            "ocr_fix": self.enable_ocr_fix,
            "dedup_fix": self.enable_dedup_fix,
        }.get(stage, False)

    @staticmethod
    def _to_llm_items(items: list) -> list:
        """提取喂给模型的精简字段（index / label / content / bbox_2d）。"""
        return [
            {
                "index": item.get("index"),
                "label": item.get("label"),
                "content": item.get("content"),
                "bbox_2d": item.get("bbox_2d"),
            }
            for item in items
        ]

    def _review_page(
        self, page_idx: int, page_items: list, image: Any
    ) -> Tuple[list, list, int]:
        """审核单页：按 _STAGE_ORDER 顺序逐阶段调用模型。

        每个阶段只做一种任务（insert → ocr_fix → dedup_fix → reorder），模型返回后
        代码立即应用并重新连续编号 index，再把结果交给下一阶段，从而避免多任务混在
        一次请求里互相干扰、以及跨任务 index 错位。

        返回 (reviewed_page, approved_changes, total_retries)。
        任一阶段全部重试失败时抛出 LLMReviewError（调用方应中止本次 OCR）。
        空页直接返回原样。
        """
        if not page_items:
            return list(page_items), [], 0

        current_items = list(page_items)
        all_approved: List[dict] = []
        total_retries = 0

        for stage in _STAGE_ORDER:
            if not self._stage_enabled(stage):
                continue
            current_items, approved, retries = self._review_stage(
                page_idx, stage, current_items, image
            )
            all_approved.extend(approved)
            total_retries += retries

        return current_items, all_approved, total_retries

    def _review_stage(
        self, page_idx: int, stage: str, items: list, image: Any
    ) -> Tuple[list, list, int]:
        """在 items 上执行单个阶段，返回 (new_items, approved, retries)。

        new_items 已应用本阶段改动并重新连续编号 index。
        """
        if not items:
            return list(items), [], 0

        system_prompt, user_template = _STAGE_PROMPTS[stage]
        llm_items = self._to_llm_items(items)
        items_json = json.dumps(llm_items, ensure_ascii=False, indent=2)
        user_prompt = user_template.format(page=page_idx, items_json=items_json)

        last_error: Optional[str] = None
        for attempt in range(self.max_retries + 1):
            try:
                response_data = self._call_llm(
                    page_idx, image, system_prompt, user_prompt, stage, len(llm_items)
                )
                raw_changes: list = response_data.get("changes", [])
                # 只保留本阶段类型，丢弃模型越界输出的其他 type。
                raw_changes = [c for c in raw_changes if c.get("type") == stage]

                if not self._validate_changes(raw_changes, items):
                    raise ValueError("验证失败：changes 引用了不存在的 index/after_index")

                approved = self._filter_changes(raw_changes, items)
                for c in approved:
                    c.setdefault("stage", stage)
                new_items = self._apply_page_changes(items, approved)
                return new_items, approved, attempt

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    "LLM 审核第 %d/%d 次失败（page=%d, stage=%s）：%s",
                    attempt + 1,
                    self.max_retries + 1,
                    page_idx,
                    stage,
                    e,
                )
                if attempt < self.max_retries:
                    time.sleep(1.5 * (attempt + 1))

        msg = f"LLM 审核全部重试失败（page={page_idx}, stage={stage}）：{last_error}"
        logger.error(msg)
        raise LLMReviewError(msg, report={"page": page_idx, "error": last_error})

    def _apply_page_changes(self, page_items: list, approved: list) -> list:
        """在单页上应用已批准的改动，返回该页纠错后的 item 列表（含 polygon）。

        所有 change 的 index / after_index 均指**输入时**的原始序号：
        - ocr_fix / dedup_fix：按原始 index 定位，子串替换 content。
        - reorder：将原始 index 对应的 item 移动到 new_index 指定的最终序号。
        - insert：插在原始 after_index 对应 item 之后（-1 表示最前）；
          同一 after_index 多条 insert 按 approved 顺序依次向后追加。
        最后按列表顺序重新连续编号 index。
        """
        items_by_index: Dict[Any, dict] = {}
        working: List[dict] = []
        for item in page_items:
            orig_idx = item.get("index")
            new_item = {
                "index": orig_idx,
                "_orig_index": orig_idx,
                "label": item.get("label"),
                "content": item.get("content"),
                "bbox_2d": item.get("bbox_2d"),
                "polygon": item.get("polygon"),
            }
            items_by_index[orig_idx] = new_item
            working.append(new_item)

        for change in approved:
            if change.get("type") in ("ocr_fix", "dedup_fix"):
                item = items_by_index.get(change.get("index"))
                if item is not None:
                    self._apply_text_fix(item, change)

        reorders: Dict[int, int] = {}
        for change in approved:
            if change.get("type") != "reorder":
                continue
            try:
                reorders[change.get("index")] = int(
                    change.get("new_index", change.get("after", -1))
                )
            except (ValueError, TypeError):
                continue

        if reorders:

            def _reorder_key(item: dict) -> Tuple[int, int]:
                orig_idx = item["_orig_index"]
                if orig_idx in reorders:
                    return (reorders[orig_idx], 0)
                return (orig_idx, 1)

            working.sort(key=_reorder_key)

        insert_offsets: Dict[int, int] = {}
        for change in approved:
            if change.get("type") != "insert":
                continue
            try:
                after_index = int(change.get("after_index", -1))
            except (ValueError, TypeError):
                after_index = -1
            new_item = {
                "label": change.get("label", "text"),
                "content": change.get("content", ""),
                "bbox_2d": None,
                "polygon": None,
            }
            offset = insert_offsets.get(after_index, 0)
            if after_index == -1:
                working.insert(offset, new_item)
            else:
                insert_pos = next(
                    (
                        i
                        for i, it in enumerate(working)
                        if it.get("_orig_index") == after_index
                    ),
                    len(working) - 1,
                )
                working.insert(insert_pos + 1 + offset, new_item)
            insert_offsets[after_index] = offset + 1

        for i, item in enumerate(working):
            item["index"] = i
            item.pop("_orig_index", None)
        return working

    @staticmethod
    def _apply_text_fix(item: dict, change: dict) -> None:
        """对单个 item 应用 ocr_fix / dedup_fix 的 before→after 子串替换。"""
        original_content: str = item.get("content") or ""
        before: str = change.get("before", "")
        after: str = change.get("after", "")
        is_dedup: bool = change.get("type") == "dedup_fix"

        # dedup_fix 且 after 为空：整项清空（before 可能经归一化导致匹配失败）。
        if is_dedup and after == "":
            item["content"] = ""
            return

        if before in original_content:
            item["content"] = original_content.replace(before, after, 1)
            return
        fuzzy = _fuzzy_replace(original_content, before, after)
        if fuzzy is not None:
            item["content"] = fuzzy
            return
        if original_content == before:
            item["content"] = after
            return

        # 归一化 LLM 双重转义的反斜杠后重试（\\before → \before）
        norm_before = before.replace("\\\\", "\\")
        norm_after = after.replace("\\\\", "\\")
        if norm_before != before:
            if norm_before in original_content:
                item["content"] = original_content.replace(norm_before, norm_after, 1)
                return
            fuzzy = _fuzzy_replace(original_content, norm_before, norm_after)
            if fuzzy is not None:
                item["content"] = fuzzy
                return
            if original_content == norm_before:
                item["content"] = norm_after
                return

        logger.warning(
            "%s 跳过：before 在原文中未找到（含归一化重试），原文保持不变（index=%s before=%r）",
            change.get("type"),
            item.get("index"),
            before[:80],
        )

    # ------------------------------------------------------------------
    # 校验 / 过滤
    # ------------------------------------------------------------------

    def _validate_changes(self, changes: list, page_items: list) -> bool:
        """结构校验：

        - ocr_fix / dedup_fix / reorder 引用的 index 必须存在于该页。
        - reorder 的 new_index 必须落在 [0, n-1]。
        - insert 的 after_index 必须落在 [-1, n-1]。
        """
        if not changes:
            return True
        valid_indices = {item.get("index") for item in page_items}
        n = len(page_items)
        for change in changes:
            ctype = change.get("type")
            if ctype == "insert":
                after_index = change.get("after_index", None)
                try:
                    after_index = int(after_index)
                except (ValueError, TypeError):
                    logger.warning("insert 的 after_index 非法：%r", change.get("after_index"))
                    return False
                if after_index < -1 or after_index > n - 1:
                    logger.warning(
                        "insert 的 after_index 越界：%d（有效 [-1, %d]）",
                        after_index,
                        n - 1,
                    )
                    return False
            elif ctype == "reorder":
                if change.get("index") not in valid_indices:
                    logger.warning(
                        "reorder 引用的 index 不存在：%s", change.get("index")
                    )
                    return False
                raw_new = change.get("new_index", change.get("after", -1))
                try:
                    new_index = int(raw_new)
                except (ValueError, TypeError):
                    logger.warning("reorder 的 new_index 非法：%r", raw_new)
                    return False
                if new_index < 0 or new_index > n - 1:
                    logger.warning(
                        "reorder 的 new_index 越界：%d（有效 [0, %d]）",
                        new_index,
                        n - 1,
                    )
                    return False
            elif ctype in ("ocr_fix", "dedup_fix"):
                if change.get("index") not in valid_indices:
                    logger.warning(
                        "change 引用的 index 不存在：%s", change.get("index")
                    )
                    return False
        return True

    def _filter_changes(self, changes: list, page_items: list) -> list:
        """按类型开关、置信度阈值与各类护栏过滤改动。"""
        content_by_index = {item.get("index"): item.get("content") for item in page_items}
        approved = []
        for change in changes:
            ctype = change.get("type", "")
            conf = float(change.get("confidence", 0) or 0)

            if ctype == "ocr_fix":
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
                # 图片区域（content=null）不允许 ocr_fix
                if content_by_index.get(change.get("index")) is None:
                    logger.debug("拒绝 ocr_fix：目标 item content 为 null（图片区域）")
                    continue
                if before == after:
                    logger.debug("拒绝 ocr_fix：before 与 after 相同，无需修改")
                    continue
                before_stripped = before.strip()
                after_stripped = after.strip()
                if before_stripped == after_stripped:
                    logger.debug("拒绝 ocr_fix：before 与 after 去空白后相同")
                    continue
                if before_stripped and len(after_stripped) < int(
                    len(before_stripped) * 0.85
                ):
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

            elif ctype == "reorder":
                if not (self.enable_reorder and conf >= self.confidence_threshold):
                    logger.debug(
                        "拒绝低置信度 reorder（%.2f < %.2f）：%s",
                        conf,
                        self.confidence_threshold,
                        change.get("reason", ""),
                    )
                    continue
                try:
                    orig_idx = int(change.get("index"))
                    new_idx = int(change.get("new_index", change.get("after", -1)))
                except (ValueError, TypeError):
                    logger.debug("拒绝 reorder：index/new_index 不是合法整数")
                    continue
                if orig_idx == new_idx:
                    logger.debug(
                        "拒绝 reorder：new_index 与 index 相同，顺序无需调整"
                    )
                    continue
                approved.append(change)

            elif ctype == "dedup_fix":
                if not (self.enable_dedup_fix and conf >= self.confidence_threshold):
                    logger.debug(
                        "拒绝低置信度 dedup_fix（%.2f < %.2f）：%s",
                        conf,
                        self.confidence_threshold,
                        change.get("reason", ""),
                    )
                    continue
                before = change.get("before")
                after = change.get("after")
                if not isinstance(before, str) or not isinstance(after, str):
                    logger.debug("拒绝 dedup_fix：before/after 不是字符串")
                    continue
                if content_by_index.get(change.get("index")) is None:
                    logger.debug("拒绝 dedup_fix：目标 item content 为 null（图片区域）")
                    continue
                if before == after:
                    logger.debug("拒绝 dedup_fix：before 与 after 相同，无需修改")
                    continue
                # dedup_fix 允许 after 比 before 短很多甚至为空，不做长度检查
                missing_markers = [
                    m for m in _SECTION_MARKERS if m in before and m not in after
                ]
                if missing_markers:
                    logger.debug(
                        "拒绝 dedup_fix：丢失段落标记 %s", ",".join(missing_markers)
                    )
                    continue
                if not _is_adjacent_layout_dedup(change, content_by_index):
                    logger.debug(
                        "拒绝 dedup_fix：非相邻 item 切分边界重叠（index=%s）",
                        change.get("index"),
                    )
                    continue
                approved.append(change)

            elif ctype == "insert":
                insert_conf_threshold = min(
                    self.confidence_threshold, _INSERT_MIN_CONFIDENCE
                )
                if not (self.enable_insert and conf >= insert_conf_threshold):
                    logger.debug(
                        "拒绝低置信度 insert（%.2f < %.2f）：%s",
                        conf,
                        insert_conf_threshold,
                        change.get("reason", ""),
                    )
                    continue
                content = change.get("content")
                if not isinstance(content, str) or not content.strip():
                    logger.debug("拒绝 insert：content 为空")
                    continue
                label = (change.get("label") or "text").strip().lower()
                if label in _INSERT_FORBIDDEN_LABELS:
                    logger.debug("拒绝 insert：label=%s 为禁止的非文本类", label)
                    continue
                if label not in _INSERT_ALLOWED_LABELS:
                    # 未知标签归一化为 text，避免误拒
                    change = dict(change)
                    change["label"] = "text"
                try:
                    after_index = int(change.get("after_index", -1))
                except (ValueError, TypeError):
                    after_index = -1
                if _is_insert_already_in_adjacent(
                    content, after_index, content_by_index
                ):
                    logger.debug(
                        "拒绝 insert：内容已出现在相邻 item 完整正文中（after_index=%s）：%s",
                        after_index,
                        content[:60],
                    )
                    continue
                approved.append(change)

        return approved

    # ------------------------------------------------------------------
    # LLM 调用
    # ------------------------------------------------------------------

    def _encode_image(self, image: Any) -> Optional[str]:
        """将 PIL.Image 等比缩放 + JPEG 压缩后编码为 base64 data URL 内容。

        返回 base64 字符串（不含 data: 前缀）；失败或无图返回 None。
        """
        if image is None:
            return None
        try:
            from PIL import Image  # noqa: F401

            img = image
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            elif img.mode == "L":
                img = img.convert("RGB")

            w, h = img.size
            max_side = max(w, h)
            if self.image_max_side and max_side > self.image_max_side:
                scale = self.image_max_side / float(max_side)
                new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
                img = img.resize(new_size)

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=self.image_jpeg_quality)
            return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception as e:
            logger.warning("图片编码失败，本页退化为纯文本审核：%s", e)
            return None

    def _call_llm(
        self,
        page_idx: int,
        image: Any,
        system_prompt: str,
        user_prompt: str,
        stage: str,
        item_count: int,
    ) -> dict:
        """调用（多模态）LLM API，返回解析后的 JSON 字典。"""
        b64 = self._encode_image(image) if self.multimodal else None
        if b64 is not None:
            user_content: Any = [
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                },
            ]
        else:
            user_content = user_prompt

        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.response_format_json:
            payload["response_format"] = {"type": "json_object"}
        if self.disable_thinking:
            payload["extra_body"] = {"thinking": {"type": "disabled"}}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        logger.info(
            "LLM 审核请求（page=%d, stage=%s）：%d 个 item，输入约 %d 字符，带图=%s，max_tokens=%d",
            page_idx,
            stage,
            item_count,
            len(user_prompt),
            b64 is not None,
            self.max_tokens,
        )

        endpoint = f"{self.base_url}/chat/completions"
        try:
            resp = self._session.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                raise ValueError(
                    f"LLM API 返回状态 {resp.status_code}：{resp.text[:400]}"
                )
            resp_data = resp.json()
        except requests.exceptions.SSLError as ssl_err:
            if not self.enable_curl_fallback:
                raise
            logger.warning(
                "requests TLS 握手失败，切换 curl 兜底通道（page=%d, stage=%s）：%s",
                page_idx,
                stage,
                ssl_err,
            )
            resp_data = self._call_llm_via_curl(endpoint, headers, payload)
        content: str = resp_data["choices"][0]["message"]["content"]
        finish_reason = resp_data["choices"][0].get("finish_reason", "unknown")

        logger.info(
            "LLM 返回（page=%d, stage=%s）：finish_reason=%s，长度=%d 字符",
            page_idx,
            stage,
            finish_reason,
            len(content),
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

    def _call_llm_via_curl(self, endpoint: str, headers: dict, payload: dict) -> dict:
        """使用 curl.exe 发送请求，规避部分 Python TLS 环境兼容问题。"""
        payload_str = json.dumps(payload, ensure_ascii=False)
        cmd = [
            "curl.exe",
            "--silent",
            "--show-error",
            "--location",
            "--max-time",
            str(self.timeout),
            "--request",
            "POST",
            endpoint,
            "--header",
            f"Authorization: {headers['Authorization']}",
            "--header",
            "Content-Type: application/json",
            "--data-binary",
            "@-",
            "--write-out",
            "\n__HTTP_STATUS__:%{http_code}",
        ]
        proc = subprocess.run(
            cmd,
            input=payload_str.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="ignore")
            raise ValueError(f"curl 调用失败（exit={proc.returncode}）：{stderr[:400]}")

        output = proc.stdout.decode("utf-8", errors="ignore")
        if not output.strip():
            raise ValueError("curl 调用成功但返回体为空")
        marker = "\n__HTTP_STATUS__:"
        if marker not in output:
            raise ValueError(f"curl 返回缺少状态码标记：{output[:400]}")
        body, _, status_str = output.rpartition(marker)
        try:
            status_code = int(status_str.strip())
        except ValueError as e:
            raise ValueError(f"curl 返回状态码解析失败：{status_str!r}") from e
        if status_code != 200:
            raise ValueError(f"LLM API 返回状态 {status_code}：{body[:400]}")
        try:
            return json.loads(body)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"curl 返回内容不是合法 JSON：{body[:400]}"
            ) from e

    def _empty_report(self) -> dict:
        return {
            "model": self.model,
            "base_url": self.base_url,
            "multimodal": self.multimodal,
            "total_items": 0,
            "total_changes": 0,
            "ocr_fix_count": 0,
            "reorder_count": 0,
            "dedup_fix_count": 0,
            "insert_count": 0,
            "changes": [],
            "latency_ms": 0,
            "retry_count": 0,
        }
