"""Build QA records from question/answer OCR JSON streams."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

# 题型大类（仅匹配大类本身，不吞并后续文本）
_QUESTION_TYPE_RE = re.compile(r"(单选题|多选题|填空题|解答题|判断题)")
_DIFFICULTY_PAIR_RE = re.compile(
    r"(容易|适中|困难)\s*(?:[（(]\s*(0\s*\.\s*\d+)\s*[）)]|(0\s*\.\s*\d+))"
)
_QUESTION_NUM_RE = re.compile(r"(?:^|\n)\s*(?:#{1,6}\s*)?(\d+)\.\s")
_FOOTER_RE = re.compile(
    r"(?:(?:您最近一年使用)|(?:今日|昨日|7日内)[\s|]*\d+次组卷|相似题\s*纠错|详情\s*收藏|加入试题篮).*$",
    re.DOTALL,
)


def _read_items(json_path: str | Path) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON: {json_path}")
    return data


def _join_text_items(items: List[Dict[str, Any]]) -> str:
    """按 index 顺序拼接所有 text/header 块（用于元数据解析）。"""
    ordered = sorted(
        (
            i
            for i in items
            if i.get("label") in {"text", "header"}
            and str(i.get("content", "")).strip()
        ),
        key=lambda x: x.get("index", 0),
    )
    return "\n".join(str(i.get("content", "")).strip() for i in ordered)


def _join_content_items(items: List[Dict[str, Any]]) -> str:
    """按 index 顺序拼接题干相关块，图片以 [image:path] 插入原位置。"""
    ordered = sorted(
        (
            i
            for i in items
            if i.get("label") in {"text", "header", "image", "formula", "table"}
            and str(i.get("content", "")).strip()
        ),
        key=lambda x: x.get("index", 0),
    )
    parts: List[str] = []
    for item in ordered:
        label = item.get("label")
        content = str(item.get("content", "")).strip()
        if label == "image":
            parts.append(f"[image:{content}]")
        else:
            parts.append(content)
    return "\n".join(parts)


def _normalize_for_meta_parse(s: str) -> str:
    s = s.replace("\u3000", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace("（", "(").replace("）", ")")
    s = re.sub(r"(\d)\s*\.\s*(\d)", r"\1.\2", s)
    return s


def _first_line_first_sentence(text: str) -> str:
    """取第一行；若行内有句号等则取第一句。"""
    text = text.strip()
    if not text:
        return ""
    first_line = text.split("\n")[0].strip()
    for sep in ("。", "！", "？", ";", "；"):
        idx = first_line.find(sep)
        if idx != -1:
            return first_line[: idx + 1].strip()
    return first_line


def _strip_markdown_heading_prefix(text: str) -> str:
    """去掉每一行行首 Markdown 标题前缀（含单个 '#'）。"""
    return re.sub(r"(?m)^\s*#{1,6}\s*", "", text).strip()


def _strip_all_hash_chars(text: str) -> str:
    """去掉文本中任意位置的 '#' 字符。"""
    if "#" not in text:
        return text
    return text.replace("#", "").strip()


def _strip_mingxiao(text: str | None) -> str | None:
    """去掉文本中任意位置的「名校」噪声标记，并规整空白。"""
    if text is None:
        return None
    if not isinstance(text, str):
        return text
    cleaned = text.replace("名校", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or None


def _clean_topic(topic: str | None) -> str | None:
    """清理 topic 中的噪声标记。"""
    if not topic:
        return None
    cleaned = topic.replace("|", " ")
    return _strip_mingxiao(cleaned)


def _sanitize_record_text_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    """统一清理 QA 结果中文本字段里的 Markdown 标题前缀与「名校」噪声。"""
    meta_mingxiao_keys = (
        "question_type",
        "difficulty_text",
        "topic",
        "source",
    )
    for key in meta_mingxiao_keys:
        value = record.get(key)
        if isinstance(value, str):
            record[key] = _strip_mingxiao(value)

    meta_hash_keys = (
        "question_type",
        "difficulty_text",
        "topic",
        "source",
    )
    for key in meta_hash_keys:
        value = record.get(key)
        if isinstance(value, str):
            record[key] = _strip_all_hash_chars(value)

    content_hash_keys = (
        "question",
        "analysis",
        "Detailed explanation",
    )
    for key in content_hash_keys:
        value = record.get(key)
        if isinstance(value, str):
            record[key] = _strip_markdown_heading_prefix(value)

    answers = record.get("answer")
    if isinstance(answers, list):
        cleaned_answers: List[Any] = []
        for item in answers:
            if isinstance(item, str):
                cleaned_answers.append(_strip_markdown_heading_prefix(item))
            else:
                cleaned_answers.append(item)
        record["answer"] = cleaned_answers

    return record


def _parse_question_structure(full_text: str) -> Dict[str, Any]:
    """从整段题干文本中按字段模式解析元数据与正文。"""
    raw = full_text.strip()
    if not raw:
        return _empty_question_structure()

    # 保留换行用于题号定位；勿对全文做「数字.数字」合并（会把「4. 2026」误变成「4.2026」导致题号匹配失败）
    normalized = raw.replace("\u3000", " ")
    normalized = normalized.replace("（", "(").replace("）", ")")

    m_type = _QUESTION_TYPE_RE.search(normalized)
    if not m_type:
        return _empty_question_structure()

    question_type = m_type.group(1)

    before_type = normalized[: m_type.start()]
    source = _first_line_first_sentence(before_type.replace("\n", " ").strip()) or None
    source = _strip_all_hash_chars(source) if source else None

    # 顺序向后匹配：题型 -> 难度文本+难度分数（必须紧邻，仅允许空格）
    after_type = normalized[m_type.end() :]
    m_diff_pair = _DIFFICULTY_PAIR_RE.search(after_type)
    difficulty_text = m_diff_pair.group(1) if m_diff_pair else None
    difficulty_score = None
    score_end = None
    if m_diff_pair:
        score_str = (m_diff_pair.group(2) or m_diff_pair.group(3) or "").replace(
            " ", ""
        )
        difficulty_score = float(score_str)
        score_end = m_diff_pair.end()

    # 仅当题型/难度文本/难度分数都匹配成功时，才抽取 topic
    topic = None
    if (
        difficulty_text is not None
        and difficulty_score is not None
        and score_end is not None
    ):
        topic_src = after_type[score_end:]
        m_qnum = _QUESTION_NUM_RE.search(topic_src)
        if m_qnum:
            topic = _clean_topic(topic_src[: m_qnum.start()].strip())
            question = topic_src[m_qnum.start() :].strip()
        else:
            topic = _clean_topic(topic_src.strip())
            question = ""
    else:
        # 元数据不完整时，题干从题型后开始按题号提取
        m_qnum = _QUESTION_NUM_RE.search(after_type)
        if m_qnum:
            question = after_type[m_qnum.start() :].strip()
        else:
            question = ""

    question = _FOOTER_RE.sub("", question).strip()
    if question.startswith("名校"):
        question = question[2:].lstrip()

    return {
        "source": source,
        "question_type": question_type,
        "difficulty_text": difficulty_text,
        "difficulty_score": difficulty_score,
        "topic": topic,
        "question": question,
    }


def _empty_question_structure() -> Dict[str, Any]:
    return {
        "source": None,
        "question_type": None,
        "difficulty_text": None,
        "difficulty_score": None,
        "topic": None,
        "question": "",
    }


def parse_question_metadata(full_text: str) -> Dict[str, Any]:
    """Parse only metadata fields from question text.

    Returns:
        {
            "source": str | None,
            "question_type": str | None,
            "difficulty_text": str | None,
            "difficulty_score": float | None,
            "topic": str | None,
        }
    """
    parsed = _parse_question_structure(full_text)
    return {
        "source": parsed["source"],
        "question_type": parsed["question_type"],
        "difficulty_text": parsed["difficulty_text"],
        "difficulty_score": parsed["difficulty_score"],
        "topic": parsed["topic"],
    }


def _split_answers(answer_text: str) -> List[str]:
    cleaned = answer_text.strip()
    if not cleaned:
        return []

    if re.fullmatch(r"[A-D]+", cleaned):
        return [cleaned]

    parts = [p.strip() for p in re.split(r"[；;|]", cleaned) if p.strip()]
    return parts or [cleaned]


# 块内多段落头：仅匹配三种形态
# 例如「分析」只允许： 【分析】 / 分析】 / 析】（答案、详解同理）
_SECTION_HEADER_RE = re.compile(
    r"(?:【\s*(?P<long>答案|分析|详解)\s*】|(?P<mid>答案|分析|详解)\s*】|(?P<short>案|析|解)\s*】)\s*[:：]?\s*"
)


def _section_key_from_header_match(match: re.Match[str]) -> str:
    label = match.group("long") or match.group("mid")
    if label == "答案":
        return "answer"
    if label == "分析":
        return "analysis"
    if label == "详解":
        return "detail"

    short_label = match.group("short")
    if short_label == "案":
        return "answer"
    if short_label == "析":
        return "analysis"
    return "detail"


def _append_section_chunk(
    section: str,
    text: str,
    answer_chunks: List[str],
    analysis_chunks: List[str],
    detail_chunks: List[str],
) -> None:
    text = text.strip()
    if not text:
        return
    if section == "answer":
        answer_chunks.append(text)
    elif section == "analysis":
        analysis_chunks.append(text)
    else:
        detail_chunks.append(text)


def _split_content_by_section_headers(content: str) -> List[tuple[str, str]]:
    """将单块正文按【答案】/【分析】/【详解】切分为多段 (section, body)。"""
    matches = list(_SECTION_HEADER_RE.finditer(content))
    if not matches:
        return []

    segments: List[tuple[str, str]] = []
    for i, m in enumerate(matches):
        section = _section_key_from_header_match(m)
        body = content[
            m.end() : matches[i + 1].start() if i + 1 < len(matches) else len(content)
        ]
        segments.append((section, body.strip()))
    return segments


def _collect_answer_sections(items: List[Dict[str, Any]]) -> tuple[List[str], str, str]:
    answer_chunks: List[str] = []
    analysis_chunks: List[str] = []
    detail_chunks: List[str] = []
    current_section: str | None = None

    for item in sorted(items, key=lambda x: x.get("index", 0)):
        label = item.get("label")
        content = str(item.get("content", "")).strip()

        if label == "image":
            if current_section == "answer":
                answer_chunks.append(f"[image:{content}]")
            elif current_section == "detail":
                detail_chunks.append(f"[image:{content}]")
            elif current_section == "analysis":
                analysis_chunks.append(f"[image:{content}]")
            else:
                # 尚未识别到段落头时，默认把图片挂在答案段，避免内容丢失
                answer_chunks.append(f"[image:{content}]")
            continue

        if not content:
            continue

        segments = _split_content_by_section_headers(content)
        if segments:
            # 同一 OCR 块中，若标题前有前缀文本，按“前一模块”回填：
            # - 分析】前缀 -> 答案
            # - 详解】前缀 -> 分析
            first_match = _SECTION_HEADER_RE.search(content)
            prefix = content[: first_match.start()].strip() if first_match else ""
            if prefix:
                first_section = segments[0][0]
                if first_section == "analysis":
                    prefix_section = "answer"
                elif first_section == "detail":
                    prefix_section = "analysis"
                else:
                    # 答案段前缀默认归答案；若已有上下文，优先沿用上下文。
                    prefix_section = current_section or "answer"
                _append_section_chunk(
                    prefix_section,
                    prefix,
                    answer_chunks,
                    analysis_chunks,
                    detail_chunks,
                )

            for section, body in segments:
                current_section = section
                _append_section_chunk(
                    section, body, answer_chunks, analysis_chunks, detail_chunks
                )
            continue

        if current_section == "answer":
            answer_chunks.append(content)
        elif current_section == "analysis":
            analysis_chunks.append(content)
        elif current_section == "detail":
            detail_chunks.append(content)

    answer_text = "\n".join(answer_chunks).strip()
    answers = _split_answers(answer_text)
    analysis = "\n".join(analysis_chunks).strip()
    detail = "\n".join(detail_chunks).strip()
    return answers, analysis, detail


def _extract_question_with_fallback(items: List[Dict[str, Any]]) -> str:
    """Extract question text; fallback to full OCR text when structured parse fails."""
    full_content = _join_content_items(items)
    parsed_question = (
        _parse_question_structure(full_content).get("question", "").strip()
    )
    if parsed_question:
        return parsed_question

    # 元数据缺失时，直接保留 OCR 正文，避免 question 为空。
    fallback = _FOOTER_RE.sub("", full_content).strip()
    if fallback.startswith("名校"):
        fallback = fallback[2:].lstrip()
    return fallback


def _infer_record_id(question_json_path: str | Path) -> str:
    """Infer image id from question JSON filename.

    Examples:
        33157823_question.json -> 33157823
        1q9_question.json      -> 1q9
    """
    stem = Path(question_json_path).stem
    m = re.match(r"^(?P<image_id>.+)_(?:question|answer)$", stem, re.IGNORECASE)
    if m:
        return m.group("image_id")
    return stem


def build_qa_array(
    question_json_path: str | Path,
    answer_json_path: str | Path,
    record_id: str | None = None,
) -> List[Dict[str, Any]]:
    """Build one QA record array from a question JSON and an answer JSON."""
    q_items = _read_items(question_json_path)
    a_items = _read_items(answer_json_path)

    # 元数据仅从纯文本解析；题干正文（含图片占位）按 index 原位置拼接
    parsed = _parse_question_structure(_join_text_items(q_items))
    question = _extract_question_with_fallback(q_items)

    answers, analysis, detail = _collect_answer_sections(a_items)
    if not answers:
        answers = None
    if not analysis:
        analysis = None
    if not detail:
        detail = None

    qa_record_id = str(record_id).strip() if record_id is not None else ""
    if not qa_record_id:
        qa_record_id = _infer_record_id(question_json_path)

    record = {
        "id": qa_record_id,
        "question_type": parsed["question_type"],
        "difficulty_text": parsed["difficulty_text"],
        "difficulty_score": parsed["difficulty_score"],
        "topic": parsed["topic"],
        "source": parsed["source"],
        "question": question,
        "answer": answers,
        "analysis": analysis,
        "Detailed explanation": detail,
    }
    return [_sanitize_record_text_fields(record)]


def print_qa_array(
    question_json_path: str | Path, answer_json_path: str | Path
) -> None:
    """Build and print QA array to console."""
    result = build_qa_array(question_json_path, answer_json_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
