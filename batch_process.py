"""批量处理入口脚本

流程：
  1. 扫描输入文件夹，匹配 {image_id}_question.png / {image_id}_answer.png 图片对
  2. 对每对图片分别执行 OCR，结果保存到 OCR_OUTPUT_DIR：
       OCR_OUTPUT_DIR/
         {image_id}_question/
           {image_id}_question.json       ← OCR 结果
           {image_id}_question_model.json ← 原始模型输出
           imgs/                        ← 裁剪图片
           layout_vis/                  ← 版面可视化
         {image_id}_answer/
           （同上结构）
  3. 读取两份 JSON，拼接为 QA 记录，保存到 QA_OUTPUT_DIR/{image_id}.json
  4. 将该题问题/答案 OCR 结果中 imgs/ 下的裁剪图复制到 QA_OUTPUT_DIR/imgs/

路径通过 .env 文件配置（见 .env.example），三个变量：
  BATCH_INPUT_DIR      — 存放输入图片的文件夹
  BATCH_OCR_OUTPUT_DIR — OCR 结果输出文件夹
  BATCH_QA_OUTPUT_DIR  — QA JSON 输出文件夹
"""

import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

# ============================================================
# 路径配置：从 .env 读取，回退到合理默认值
# ============================================================

_HERE = Path(__file__).parent

def _resolve(env_key: str, default: str) -> str:
    raw = os.environ.get(env_key) or default
    p = Path(raw)
    return str(p if p.is_absolute() else _HERE / p)


INPUT_DIR = _resolve("BATCH_INPUT_DIR", "input")
OCR_OUTPUT_DIR = _resolve("BATCH_OCR_OUTPUT_DIR", "output")
QA_OUTPUT_DIR = _resolve("BATCH_QA_OUTPUT_DIR", "qa_output")

# ============================================================

_IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

_NAME_RE = re.compile(
    r"^(?P<image_id>[^_]+)_(?P<role>question|answer)$",
    re.IGNORECASE,
)


def _scan_pairs(folder: Path):
    """扫描文件夹，返回匹配对和未匹配文件列表。

    Returns:
        matched  : dict  image_id -> {"question": Path, "answer": Path}
        unmatched: list[Path]
    """
    groups: dict[str, dict[str, Path]] = {}

    for p in sorted(folder.iterdir()):
        if not p.is_file() or p.suffix.lower() not in _IMG_SUFFIXES:
            continue
        m = _NAME_RE.match(p.stem)
        if not m:
            groups.setdefault(f"__raw_{p.name}", {})["_unmatched"] = p
            continue
        image_id = m.group("image_id")
        role = m.group("role").lower()
        groups.setdefault(image_id, {})[role] = p

    matched: dict[str, dict[str, Path]] = {}
    unmatched: list[Path] = []

    for key, entry in groups.items():
        if "_unmatched" in entry:
            unmatched.append(entry["_unmatched"])
        elif "question" in entry and "answer" in entry:
            matched[key] = entry
        else:
            # 只有一边
            for p in entry.values():
                unmatched.append(p)

    return matched, unmatched


def _remove_ocr_output(ocr_output_dir: Path, stem: str) -> None:
    """删除已写入的 OCR 输出目录（审核失败时清理半成品）。"""
    target = ocr_output_dir / stem
    if target.is_dir():
        shutil.rmtree(target)


def _ocr_and_save(parser, img_path: Path, ocr_output_dir: Path) -> Path:
    """对单张图片执行 OCR，保存结果到 ocr_output_dir，返回 JSON 路径。

    保存结构：
        ocr_output_dir/{img_stem}/{img_stem}.json
        ocr_output_dir/{img_stem}/imgs/...
        ocr_output_dir/{img_stem}/layout_vis/...
    """
    result = parser.parse(str(img_path), save_layout_visualization=True)
    result.save(output_dir=str(ocr_output_dir))

    stem = img_path.stem
    json_path = ocr_output_dir / stem / f"{stem}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"OCR 结果 JSON 未找到，期望路径：{json_path}")
    return json_path


def _copy_cropped_images(
    json_paths: list[Path],
    qa_imgs_dir: Path,
    exclude_filenames: set[str] | None = None,
) -> int:
    """Copy all cropped images from OCR output dirs into qa_output/imgs/."""
    qa_imgs_dir.mkdir(parents=True, exist_ok=True)
    excludes = exclude_filenames or set()
    for name in excludes:
        stale = qa_imgs_dir / name
        if stale.is_file():
            stale.unlink()

    copied = 0
    for json_path in json_paths:
        imgs_dir = json_path.parent / "imgs"
        if not imgs_dir.is_dir():
            continue
        for src in sorted(imgs_dir.iterdir()):
            if not src.is_file() or src.suffix.lower() not in _IMG_SUFFIXES:
                continue
            if src.name in excludes:
                continue
            shutil.copy2(src, qa_imgs_dir / src.name)
            copied += 1
    return copied


def _collect_result_text(result: Any) -> str:
    """从 GlmOcr.parse 的结果中抽取可用于元数据匹配的文本。"""
    json_result = getattr(result, "json_result", None)
    if not isinstance(json_result, list):
        return ""

    parts: list[str] = []
    for page in json_result:
        if not isinstance(page, list):
            continue
        for item in page:
            if not isinstance(item, dict):
                continue
            if item.get("label") not in {"text", "header", "formula", "table"}:
                continue
            content = item.get("content")
            if isinstance(content, str) and content.strip():
                parts.append(content.strip())
    return "\n".join(parts).strip()


def _should_try_metadata_recovery(record: dict) -> bool:
    """是否需要触发顶部截图补充元数据。"""
    keys = ("question_type", "difficulty_text", "difficulty_score", "topic")
    return all(record.get(k) is None for k in keys)


def _remove_question_image_placeholder(
    record: dict, question_json_path: Path, image_path: Path
) -> None:
    """从 question 字段中删除指定 image 的占位符。"""
    question = record.get("question")
    if not isinstance(question, str) or not question:
        return
    try:
        rel_path = image_path.relative_to(question_json_path.parent).as_posix()
    except ValueError:
        rel_path = image_path.name
    tag = f"[image:{rel_path}]"
    if tag not in question:
        return
    cleaned = question.replace(tag, "")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    record["question"] = cleaned


def _fill_source_from_text_before_image_placeholder(
    record: dict, question_json_path: Path, image_path: Path
) -> None:
    """若 source 为空，则从目标图片占位符前的文本提取 source。"""
    if record.get("source") is not None:
        return
    question = record.get("question")
    if not isinstance(question, str) or not question:
        return
    try:
        rel_path = image_path.relative_to(question_json_path.parent).as_posix()
    except ValueError:
        rel_path = image_path.name
    tag = f"[image:{rel_path}]"
    idx = question.find(tag)
    if idx <= 0:
        return
    prefix = question[:idx].strip()
    if not prefix:
        return
    first_non_empty_line = next((ln.strip() for ln in prefix.splitlines() if ln.strip()), "")
    if first_non_empty_line:
        record["source"] = first_non_empty_line
        # source 回填后，删除 question 中重复的来源首行。
        q_lines = question.splitlines()
        for i, line in enumerate(q_lines):
            if not line.strip():
                continue
            if line.strip() == first_non_empty_line:
                q_lines[i] = ""
                break
            break
        cleaned_question = "\n".join(q_lines).strip()
        cleaned_question = re.sub(r"\n{3,}", "\n\n", cleaned_question)
        record["question"] = cleaned_question


def _collect_top_image_block(question_json_path: Path) -> Path | None:
    """收集 question JSON 前两个块中的首个 image。"""
    try:
        with open(question_json_path, "r", encoding="utf-8") as f:
            items = json.load(f)
    except Exception:
        return None

    if not isinstance(items, list):
        return None

    # 仅看前两个块，命中的第一个 image 即作为候选
    for item in items[:2]:
        if not isinstance(item, dict):
            continue
        if item.get("label") != "image":
            continue
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        image_path = question_json_path.parent / content
        if image_path.is_file():
            return image_path
    return None


def _try_recover_metadata_from_top_images(parser: Any, question_json_path: Path) -> dict | None:
    """对顶部截图块补充 OCR，并二次匹配元数据结构；匹配成功才返回。"""
    from glmocr.utils.qa_pair_builder import parse_question_metadata

    image_path = _collect_top_image_block(question_json_path)
    if image_path is None:
        return None

    try:
        result = parser.parse(str(image_path), save_layout_visualization=False)
    except Exception:
        return None

    merged_text = _collect_result_text(result)
    if not merged_text:
        return None

    parsed_meta = parse_question_metadata(merged_text)

    # 二次匹配门槛：题型与难度信息三项都成功，才允许回填
    required_keys = ("question_type", "difficulty_text", "difficulty_score")
    if any(parsed_meta.get(k) is None for k in required_keys):
        return None
    return parsed_meta


def _is_skipped_record(record: dict) -> tuple[bool, str]:
    """判定是否为识别失败记录。"""
    question = record.get("question")
    if not isinstance(question, str) or not question.strip():
        return True, "question 为空"

    answer_is_null = record.get("answer") is None
    analysis_is_null = record.get("analysis") is None
    detail_is_null = record.get("Detailed explanation") is None

    if answer_is_null and analysis_is_null and detail_is_null:
        return True, "answer/analysis/Detailed explanation 均为 null"

    return False, ""


def _remove_qa_output(qa_output_dir: Path, image_id: str) -> None:
    """删除 QA 输出 JSON 与该题汇总截图。"""
    out_file = qa_output_dir / f"{image_id}.json"
    if out_file.exists():
        out_file.unlink()

    qa_imgs_dir = qa_output_dir / "imgs"
    if qa_imgs_dir.is_dir():
        patterns = [
            f"{image_id}_question_idx*",
            f"{image_id}_answer_idx*",
        ]
        for pattern in patterns:
            for p in qa_imgs_dir.glob(pattern):
                if p.is_file():
                    p.unlink()


def main():
    input_dir = Path(INPUT_DIR)
    ocr_output_dir = Path(OCR_OUTPUT_DIR)
    qa_output_dir = Path(QA_OUTPUT_DIR)

    if not input_dir.exists():
        print(f"[错误] 输入文件夹不存在：{input_dir}", file=sys.stderr)
        sys.exit(1)

    ocr_output_dir.mkdir(parents=True, exist_ok=True)
    qa_output_dir.mkdir(parents=True, exist_ok=True)

    # 扫描图片对
    matched, unmatched = _scan_pairs(input_dir)

    if unmatched:
        print("以下图片没有找到匹配的配对，已跳过：")
        for p in unmatched:
            print(f"  {p.name}")
        print()

    if not matched:
        print("未找到任何匹配的图片对，程序退出。")
        return

    print(f"共找到 {len(matched)} 对匹配图片，开始处理...\n")

    try:
        from glmocr.api import GlmOcr
        from glmocr.utils.qa_pair_builder import build_qa_array
    except ImportError as e:
        print(f"[错误] 导入 glmocr 失败：{e}", file=sys.stderr)
        sys.exit(1)

    success = 0
    failed = 0
    skipped: list[tuple[str, str]] = []  # (image_id, reason)

    with GlmOcr() as parser:
        for idx, (image_id, pair) in enumerate(matched.items(), 1):
            q_img: Path = pair["question"]
            a_img: Path = pair["answer"]

            print(f"[{idx}/{len(matched)}] {image_id}")

            # --- OCR 问题图片 ---
            try:
                q_json = _ocr_and_save(parser, q_img, ocr_output_dir)
                print(f"  问题 OCR 完成 → {q_json.parent.relative_to(ocr_output_dir)}")
            except Exception as e:
                _remove_ocr_output(ocr_output_dir, q_img.stem)
                print(f"  [跳过] 问题 OCR 失败：{e}")
                failed += 1
                skipped.append((image_id, f"问题 OCR 失败: {e}"))
                continue

            # --- OCR 答案图片 ---
            try:
                a_json = _ocr_and_save(parser, a_img, ocr_output_dir)
                print(f"  答案 OCR 完成 → {a_json.parent.relative_to(ocr_output_dir)}")
            except Exception as e:
                _remove_ocr_output(ocr_output_dir, q_img.stem)
                _remove_ocr_output(ocr_output_dir, a_img.stem)
                print(f"  [跳过] 答案 OCR 失败：{e}")
                failed += 1
                skipped.append((image_id, f"答案 OCR 失败: {e}"))
                continue

            # --- 拼接 QA ---
            try:
                qa_array = build_qa_array(q_json, a_json, record_id=image_id)
            except Exception as e:
                _remove_ocr_output(ocr_output_dir, q_img.stem)
                _remove_ocr_output(ocr_output_dir, a_img.stem)
                print(f"  [跳过] QA 拼接失败：{e}")
                failed += 1
                skipped.append((image_id, f"QA 拼接失败: {e}"))
                continue

            record = qa_array[0] if isinstance(qa_array, list) and qa_array else {}
            excluded_img_names: set[str] = set()
            if _should_try_metadata_recovery(record):
                recovered = _try_recover_metadata_from_top_images(parser, q_json)
                if recovered:
                    for k in (
                        "question_type",
                        "difficulty_text",
                        "difficulty_score",
                        "topic",
                        "source",
                    ):
                        if record.get(k) is None and recovered.get(k) is not None:
                            record[k] = recovered[k]
                    top_img = _collect_top_image_block(q_json)
                    if top_img is not None:
                        _fill_source_from_text_before_image_placeholder(
                            record, q_json, top_img
                        )
                        _remove_question_image_placeholder(record, q_json, top_img)
                        excluded_img_names.add(top_img.name)
                    print("  已通过顶部截图补充元数据")

            should_skip, skip_reason = _is_skipped_record(record)
            if should_skip:
                _remove_ocr_output(ocr_output_dir, q_img.stem)
                _remove_ocr_output(ocr_output_dir, a_img.stem)
                _remove_qa_output(qa_output_dir, image_id)
                print(f"  [跳过] QA 识别失败：{skip_reason}")
                failed += 1
                skipped.append((image_id, f"QA 识别失败: {skip_reason}"))
                continue

            # --- 汇总裁剪图到 qa_output/imgs ---
            qa_imgs_dir = qa_output_dir / "imgs"
            img_count = _copy_cropped_images(
                [q_json, a_json], qa_imgs_dir, exclude_filenames=excluded_img_names
            )

            # --- 保存 QA JSON ---
            out_file = qa_output_dir / f"{image_id}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(qa_array, f, ensure_ascii=False, indent=2)

            print(f"  已复制 {img_count} 张截图 → {qa_imgs_dir}")
            print(f"  QA 已保存 → {out_file}")
            success += 1

    print(f"\n{'='*40}")
    print(f"处理完成：成功 {success} 对，跳过 {failed} 对")
    if skipped:
        print(f"\n跳过的题目：")
        for image_id, reason in skipped:
            print(f"  - {image_id}  ({reason})")
    print(f"\nOCR 结果目录：{ocr_output_dir}")
    print(f"QA  结果目录：{qa_output_dir}")
    print(f"截图汇总目录：{qa_output_dir / 'imgs'}")


if __name__ == "__main__":
    main()
