"""Long image pre-splitter.

Splits a single tall image into strips before feeding to the pipeline,
preventing layout detection and OCR accuracy loss caused by aggressive
downscaling of high-aspect-ratio images.

Strategy
--------
1. **Gap-based split** (primary): scan horizontal pixel rows for "blank
   bands" (high brightness + low variance).  When the accumulated strip
   height is about to exceed ``max_strip_height``, find the nearest blank
   band below the current position and cut there.  This avoids slicing
   through text or formulas.

2. **Force split** (fallback): if no blank band is found within
   ``gap_search_margin`` pixels, cut at exactly ``max_strip_height`` with
   ``overlap`` pixels of padding on both sides so that content near the
   boundary is not lost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image

from glmocr.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LongImageConfig:
    """Configuration for long-image pre-splitting."""

    enabled: bool = True
    # Trigger: split when EITHER condition is met.
    # Minimum image height (px) to trigger splitting.
    min_height_to_split: int = 3500
    # Minimum aspect ratio (height / width) to trigger splitting.
    min_aspect_ratio: float = 3.0

    # Target strip size.
    # PP-DocLayoutV3 always resizes all inputs to a fixed 800×800 square.
    # The vertical compression ratio = strip_height / 800 (independent of
    # image width). Standard A4 training pages are ~1754px tall → 800px
    # = 2.2x compression; the model handles up to ~3x well.
    #
    # max_compression_ratio_for_detector sets the maximum acceptable
    # vertical compression. Effective strip height cap = 800 * ratio.
    #   ratio=2.0 → max 1600px (conservative, 30px line → 15px in model)
    #   ratio=2.5 → max 2000px (balanced, matches A4 training distribution)
    #   ratio=3.0 → max 2400px (permissive, at A4 training limit)
    max_compression_ratio_for_detector: float = 2.5
    # Hard ceiling regardless of computed value (safety net for very tall images).
    max_strip_height: int = 2800
    # Minimum strip height; strips shorter than this are merged upward.
    min_strip_height: int = 600

    # Blank-band detection parameters.
    # A row is considered "blank" when mean brightness >= threshold AND
    # std deviation < std_threshold.
    gap_brightness_threshold: int = 240
    gap_std_threshold: float = 15.0
    # Minimum consecutive blank rows to qualify as a cut band.
    min_gap_rows: int = 8
    # How far (px) below the strip limit to search for a blank band before
    # giving up and doing a force-cut.
    gap_search_margin: int = 400

    # Cut offset from the blank band centre.  0 = cut exactly at the centre,
    # giving both strips equal margins.  Positive values shift the cut
    # downward (more whitespace on strip[0], less on strip[1]).
    padding: int = 0
    # Pixel overlap between adjacent strips when doing a force-cut.
    # Gap-based cuts do not need overlap.
    force_cut_overlap: int = 150


def _should_split(img: Image.Image, cfg: LongImageConfig) -> bool:
    """Return True if *img* is tall enough to warrant pre-splitting."""
    w, h = img.size
    aspect = h / max(w, 1)
    return h >= cfg.min_height_to_split or aspect >= cfg.min_aspect_ratio


def _find_blank_rows(
    gray_arr: np.ndarray,
    brightness_threshold: int,
    std_threshold: float,
) -> np.ndarray:
    """Return a boolean mask marking which rows are 'blank'.

    Args:
        gray_arr: 2-D uint8 array (H × W).
        brightness_threshold: Mean brightness >= this to be blank.
        std_threshold: Row std deviation < this to be blank.

    Returns:
        Boolean array of length H.
    """
    row_mean = gray_arr.mean(axis=1)
    row_std = gray_arr.std(axis=1)
    return (row_mean >= brightness_threshold) & (row_std < std_threshold)


def _find_gap_bands(
    is_blank: np.ndarray,
    min_gap_rows: int,
) -> List[Tuple[int, int]]:
    """Return a list of (start_row, end_row) pairs for blank bands.

    A blank band is a contiguous run of blank rows with length >= min_gap_rows.
    """
    bands: List[Tuple[int, int]] = []
    in_gap = False
    gap_start = 0
    h = len(is_blank)

    for y in range(h):
        if is_blank[y]:
            if not in_gap:
                in_gap = True
                gap_start = y
        else:
            if in_gap:
                in_gap = False
                if y - gap_start >= min_gap_rows:
                    bands.append((gap_start, y))

    if in_gap and h - gap_start >= min_gap_rows:
        bands.append((gap_start, h))

    return bands


def _find_best_cut_in_range(
    gap_bands: List[Tuple[int, int]],
    search_start: int,
    search_end: int,
    padding: int,
) -> int | None:
    """Return the best cut y-coordinate within [search_start, search_end].

    We prefer a blank band whose *center* falls inside the search window.
    Returns the bottom edge of the band (minus padding) so the strip ends
    in clean whitespace.

    Returns None if no suitable band exists.
    """
    candidates = []
    for gs, ge in gap_bands:
        center = (gs + ge) // 2
        if search_start <= center <= search_end:
            # cut at band center; never deeper than the search window
            cut_y = min(center + padding, search_end)
            candidates.append((center, cut_y))

    if not candidates:
        return None

    # Prefer the cut point closest to search_end (maximize strip usage).
    candidates.sort(key=lambda c: c[0], reverse=True)
    return candidates[0][1]


def split_long_image(
    img: Image.Image,
    cfg: LongImageConfig,
) -> List[Image.Image]:
    """Split *img* into strips and return them as a list.

    If no splitting is needed, returns ``[img]``.

    Args:
        img: PIL Image (any mode; converted to RGB internally for analysis).
        cfg: LongImageConfig instance.

    Returns:
        List of PIL Image strips in top-to-bottom order.
    """
    if not _should_split(img, cfg):
        return [img]

    w, h = img.size
    # Effective strip height: the smaller of the hard ceiling and the
    # compression-ratio-derived limit.
    # PP-DocLayoutV3 resizes all inputs to 800×800 (fixed, regardless of
    # input size). Vertical compression in the model = strip_height / 800.
    # To keep compression within max_compression_ratio_for_detector:
    #   max_height_from_ratio = 800 * max_compression_ratio_for_detector
    effective_max = min(
        cfg.max_strip_height,
        int(800 * cfg.max_compression_ratio_for_detector),
    )
    logger.info(
        "Long image detected (%dx%d, aspect=%.1f); pre-splitting into strips "
        "(effective_max=%d, hard_ceiling=%d, compression_limit=%d "
        "[%.1fx of 800px model height]).",
        w,
        h,
        h / max(w, 1),
        effective_max,
        cfg.max_strip_height,
        int(800 * cfg.max_compression_ratio_for_detector),
        cfg.max_compression_ratio_for_detector,
    )

    # Analyse in grayscale for speed.
    gray = img.convert("L") if img.mode != "L" else img
    gray_arr = np.array(gray, dtype=np.float32)

    is_blank = _find_blank_rows(
        gray_arr,
        cfg.gap_brightness_threshold,
        cfg.gap_std_threshold,
    )
    gap_bands = _find_gap_bands(is_blank, cfg.min_gap_rows)
    logger.debug("Found %d blank bands in long image.", len(gap_bands))

    strips: List[Image.Image] = []
    strip_top = 0

    while strip_top < h:
        ideal_bottom = strip_top + effective_max

        if ideal_bottom >= h:
            # Last strip — take everything remaining.
            tail_height = h - strip_top
            if tail_height < cfg.min_strip_height and strips:
                # Tail is too short; merge into the previous strip.
                prev = strips.pop()
                prev_top = strip_top - prev.size[1]
                merged = img.crop((0, max(prev_top, 0), w, h))
                strips.append(merged)
                logger.debug(
                    "Merged short tail (%dpx) into previous strip " "(new height=%d).",
                    tail_height,
                    merged.size[1],
                )
            else:
                strips.append(img.crop((0, strip_top, w, h)))
            break

        # Search for a blank band just before ideal_bottom.
        search_start = ideal_bottom - cfg.gap_search_margin
        search_end = ideal_bottom

        cut_y = _find_best_cut_in_range(
            gap_bands,
            search_start,
            search_end,
            cfg.padding,
        )

        if cut_y is not None:
            # Gap-based cut: no overlap needed.
            actual_bottom = min(cut_y, h)
            logger.info(
                "  Strip %d: Gap-based cut at y=%d (height=%d, "
                "compression=%.2fx) — blank band found %d px before limit.",
                len(strips),
                actual_bottom,
                actual_bottom - strip_top,
                (actual_bottom - strip_top) / 800,
                ideal_bottom - actual_bottom,
            )
        else:
            # Force-cut with overlap.
            actual_bottom = min(ideal_bottom, h)
            logger.info(
                "  Strip %d: Force-cut at y=%d (height=%d, "
                "compression=%.2fx) — no blank band in search window, "
                "overlap=%dpx applied.",
                len(strips),
                actual_bottom,
                actual_bottom - strip_top,
                (actual_bottom - strip_top) / 800,
                cfg.force_cut_overlap,
            )

        strip_height = actual_bottom - strip_top
        if strip_height < cfg.min_strip_height and strips:
            # Merge tiny tail into the previous strip rather than creating a
            # near-empty last strip.
            prev = strips.pop()
            prev_top = strip_top - prev.size[1]
            merged = img.crop((0, max(prev_top, 0), w, actual_bottom))
            strips.append(merged)
            logger.debug(
                "Merged short tail strip into previous (new height=%d).",
                merged.size[1],
            )
            strip_top = actual_bottom
            continue

        strips.append(img.crop((0, strip_top, w, actual_bottom)))

        # Next strip starts with overlap when we did a force-cut.
        if cut_y is not None:
            strip_top = actual_bottom
        else:
            strip_top = max(0, actual_bottom - cfg.force_cut_overlap)

    logger.info("Pre-split complete: %d strips from %dx%d image.", len(strips), w, h)
    return strips
