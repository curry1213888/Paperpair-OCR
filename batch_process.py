"""批量处理入口脚本

流程：
  1. 扫描输入文件夹，匹配 {prefix}_question.png / {prefix}_answer.png 图片对
  2. 对每对图片分别执行 OCR，结果保存到 OCR_OUTPUT_DIR：
       OCR_OUTPUT_DIR/
         {prefix}_question/
           {prefix}_question.json       ← OCR 结果
           {prefix}_question_model.json ← 原始模型输出
           imgs/                        ← 裁剪图片
           layout_vis/                  ← 版面可视化
         {prefix}_answer/
           （同上结构）
  3. 读取两份 JSON，拼接为 QA 记录，保存到 QA_OUTPUT_DIR/{prefix}.json
  4. 将该题问题/答案 OCR 结果中 imgs/ 下的裁剪图复制到 QA_OUTPUT_DIR/imgs/

使用前修改下方 ====== 配置区 ====== 中的路径。
"""

import json
import re
import shutil
import sys
from pathlib import Path

# ============================================================
# 配置区：修改以下路径
# ============================================================

# 输入文件夹：存放 {prefix}_question.png 和 {prefix}_answer.png 的目录
INPUT_DIR = r"D:\Desktop\11"

# OCR 输出文件夹：存放每张图片的识别结果（JSON、裁剪图片、版面可视化）
OCR_OUTPUT_DIR = r"D:\Desktop\GLM-OCR\output"

# QA 输出文件夹：存放最终拼接好的 QA JSON 文件
QA_OUTPUT_DIR = r"D:\Desktop\GLM-OCR\qa_output"

# ============================================================

_IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

_NAME_RE = re.compile(
    r"^(?P<prefix>.+)_(?P<role>question|answer)$",
    re.IGNORECASE,
)


def _scan_pairs(folder: Path):
    """扫描文件夹，返回匹配对和未匹配文件列表。

    Returns:
        matched  : dict  prefix -> {"question": Path, "answer": Path}
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
        prefix = m.group("prefix")
        role = m.group("role").lower()
        groups.setdefault(prefix, {})[role] = p

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


def _copy_cropped_images(json_paths: list[Path], qa_imgs_dir: Path) -> int:
    """Copy all cropped images from OCR output dirs into qa_output/imgs/."""
    qa_imgs_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for json_path in json_paths:
        imgs_dir = json_path.parent / "imgs"
        if not imgs_dir.is_dir():
            continue
        for src in sorted(imgs_dir.iterdir()):
            if not src.is_file() or src.suffix.lower() not in _IMG_SUFFIXES:
                continue
            shutil.copy2(src, qa_imgs_dir / src.name)
            copied += 1
    return copied


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
        from glmocr.postprocess.llm_reviewer import LLMReviewError
        from glmocr.utils.qa_pair_builder import build_qa_array
    except ImportError as e:
        print(f"[错误] 导入 glmocr 失败：{e}", file=sys.stderr)
        sys.exit(1)

    success = 0
    failed = 0

    with GlmOcr() as parser:
        for idx, (prefix, pair) in enumerate(matched.items(), 1):
            q_img: Path = pair["question"]
            a_img: Path = pair["answer"]

            print(f"[{idx}/{len(matched)}] {prefix}")

            # --- OCR 问题图片 ---
            try:
                q_json = _ocr_and_save(parser, q_img, ocr_output_dir)
                print(f"  问题 OCR 完成 → {q_json.parent.relative_to(ocr_output_dir)}")
            except LLMReviewError as e:
                _remove_ocr_output(ocr_output_dir, q_img.stem)
                print(f"  [跳过] 提取失败（问题）：{e}")
                failed += 1
                continue
            except Exception as e:
                print(f"  [跳过] 问题 OCR 失败：{e}")
                failed += 1
                continue

            # --- OCR 答案图片 ---
            try:
                a_json = _ocr_and_save(parser, a_img, ocr_output_dir)
                print(f"  答案 OCR 完成 → {a_json.parent.relative_to(ocr_output_dir)}")
            except LLMReviewError as e:
                _remove_ocr_output(ocr_output_dir, q_img.stem)
                _remove_ocr_output(ocr_output_dir, a_img.stem)
                print(f"  [跳过] 提取失败（答案）：{e}")
                failed += 1
                continue
            except Exception as e:
                _remove_ocr_output(ocr_output_dir, q_img.stem)
                print(f"  [跳过] 答案 OCR 失败：{e}")
                failed += 1
                continue

            # --- 拼接 QA ---
            try:
                qa_array = build_qa_array(q_json, a_json)
            except Exception as e:
                print(f"  [跳过] QA 拼接失败：{e}")
                failed += 1
                continue

            # --- 汇总裁剪图到 qa_output/imgs ---
            qa_imgs_dir = qa_output_dir / "imgs"
            img_count = _copy_cropped_images([q_json, a_json], qa_imgs_dir)

            # --- 保存 QA JSON ---
            out_file = qa_output_dir / f"{prefix}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(qa_array, f, ensure_ascii=False, indent=2)

            print(f"  已复制 {img_count} 张截图 → {qa_imgs_dir}")
            print(f"  QA 已保存 → {out_file}")
            success += 1

    print(f"\n{'='*40}")
    print(f"处理完成：成功 {success} 对，跳过 {failed} 对")
    print(f"OCR 结果目录：{ocr_output_dir}")
    print(f"QA  结果目录：{qa_output_dir}")
    print(f"截图汇总目录：{qa_output_dir / 'imgs'}")


if __name__ == "__main__":
    main()
