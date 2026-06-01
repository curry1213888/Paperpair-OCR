"""Build QA records from question/answer OCR JSON streams."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List

# 题型大类（后缀如「解答题-问答题」只保留大类）
_QUESTION_TYPE_RE = re.compile(
    r"(单选题|多选题|填空题|解答题|判断题)(?:[-－—][^\s(（]+)?"
)
_DIFFICULTY_RE = re.compile(
    r"(容易|适中|困难)\s*[（(]\s*(0\.\d+)\s*[）)]"
)
_QUESTION_NUM_RE = re.compile(r"(?:^|\n)\s*(\d+)\.\s")
_FOOTER_RE = re.compile(
    r"(?:今日\s*\d+次组卷|相似题\s*纠错|详情\s*收藏|加入试题篮).*$",
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
    """按 index 顺序拼接 text/header/image，图片以 [image:path] 插入原位置。"""
    ordered = sorted(
        (
            i
            for i in items
            if i.get("label") in {"text", "header", "image"}
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


def _parse_question_structure(full_text: str) -> Dict[str, Any]:
    """从整段题干文本中按字段模式解析元数据与正文。"""
    raw = full_text.strip()
    if not raw:
        return _empty_question_structure()

    # 保留换行用于题号定位，仅对单行段做空白归一化
    normalized = raw.replace("\u3000", " ")
    normalized = normalized.replace("（", "(").replace("）", ")")
    normalized = re.sub(r"(\d)\s*\.\s*(\d)", r"\1.\2", normalized)

    m_type = _QUESTION_TYPE_RE.search(normalized)
    if not m_type:
        return _empty_question_structure()

    question_type = m_type.group(1)

    before_type = normalized[: m_type.start()]
    source = _first_line_first_sentence(before_type.replace("\n", " ").strip()) or "略"

    after_type = normalized[m_type.end() :].lstrip()

    m_diff = _DIFFICULTY_RE.search(after_type)
    if m_diff:
        difficulty_text = m_diff.group(1)
        difficulty_score = float(m_diff.group(2))
        rest = after_type[m_diff.end() :].lstrip()
    else:
        difficulty_text = "略"
        difficulty_score = "略"
        rest = after_type

    m_qnum = _QUESTION_NUM_RE.search(rest)
    if m_qnum:
        topic = rest[: m_qnum.start()].strip() or "略"
        question = rest[m_qnum.start() :].strip()
    else:
        topic = rest.strip() or "略"
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
        "source": "略",
        "question_type": "略",
        "difficulty_text": "略",
        "difficulty_score": "略",
        "topic": "略",
        "question": "",
    }


def _split_answers(answer_text: str) -> List[str]:
    cleaned = answer_text.strip()
    if not cleaned:
        return []

    if re.fullmatch(r"[A-D]+", cleaned):
        return [cleaned]

    parts = [p.strip() for p in re.split(r"[；;|]", cleaned) if p.strip()]
    return parts or [cleaned]


# 块内多段落头：同一 OCR 块中可能同时含【分析】与【详解】
_SECTION_HEADER_RE = re.compile(
    r"[【\[]?\s*(答案|分析|详解)\s*[】\]]?\s*[:：]?\s*"
)


def _section_key_from_label(label: str) -> str:
    if label == "答案":
        return "answer"
    if label == "分析":
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
        section = _section_key_from_label(m.group(1))
        body = content[m.end() : matches[i + 1].start() if i + 1 < len(matches) else len(content)]
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
            if current_section == "detail":
                detail_chunks.append(f"[image:{content}]")
            elif current_section == "analysis":
                analysis_chunks.append(f"[image:{content}]")
            continue

        if not content:
            continue

        segments = _split_content_by_section_headers(content)
        if segments:
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


def build_qa_array(question_json_path: str | Path, answer_json_path: str | Path) -> List[Dict[str, Any]]:
    """Build one QA record array from a question JSON and an answer JSON."""
    q_items = _read_items(question_json_path)
    a_items = _read_items(answer_json_path)

    # 元数据仅从纯文本解析；正文（含图片占位）按 index 原位置拼接
    parsed = _parse_question_structure(_join_text_items(q_items))
    question = _parse_question_structure(_join_content_items(q_items))["question"]

    answers, analysis, detail = _collect_answer_sections(a_items)
    if not answers:
        answers = ["略"]
    if not analysis:
        analysis = "略"
    if not detail:
        detail = "略"

    record = {
        "id": str(uuid.uuid4()),
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
    return [record]


def print_qa_array(question_json_path: str | Path, answer_json_path: str | Path) -> None:
    """Build and print QA array to console."""
    result = build_qa_array(question_json_path, answer_json_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
