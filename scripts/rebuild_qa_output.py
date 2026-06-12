"""从已有 OCR JSON 重新组装 qa_output 下的 QA 记录。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from glmocr.utils.qa_pair_builder import build_qa_array  # noqa: E402


def main() -> None:
    qa_output_dir = _ROOT / "qa_output"
    ocr_output_dir = _ROOT / "output"

    json_files = sorted(qa_output_dir.glob("*.json"))
    if not json_files:
        print("qa_output 下没有 JSON 文件")
        return

    ok = 0
    failed: list[tuple[str, str]] = []

    for out_file in json_files:
        image_id = out_file.stem
        q_json = ocr_output_dir / f"{image_id}_question" / f"{image_id}_question.json"
        a_json = ocr_output_dir / f"{image_id}_answer" / f"{image_id}_answer.json"

        if not q_json.is_file() or not a_json.is_file():
            failed.append((image_id, "缺少 OCR JSON"))
            print(f"FAIL {image_id}: 缺少 OCR JSON")
            continue

        try:
            qa_array = build_qa_array(q_json, a_json, record_id=image_id)
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(qa_array, f, ensure_ascii=False, indent=2)
            ok += 1
            print(f"OK   {image_id}")
        except Exception as e:
            failed.append((image_id, str(e)))
            print(f"FAIL {image_id}: {e}")

    print(f"\n完成：成功 {ok}/{len(json_files)}")
    if failed:
        print("失败：")
        for image_id, reason in failed:
            print(f"  - {image_id}: {reason}")


if __name__ == "__main__":
    main()
