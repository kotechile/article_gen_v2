"""
Editorial Cover Typography Overlay Service.

Deterministically composites high-contrast, magazine-grade editorial typography
over generated/selected images in post-processing using Pillow (PIL).

Features:
1. 3-tier editorial copy hierarchy (Kicker, Title, Hook) with word-safe truncation.
2. Spatial placement & empty corner detection using edge energy with top-left hysteresis.
3. Proportional typography sizing (KICKER_PT=0.029h, TITLE_PT=0.086h, HOOK_PT=0.036h)
   and dynamic 2-line title wrapping.
4. Adaptive contrast with binding pixel sampling (85th/15th percentiles) and dual palettes
   (warm gold / off-white for dark; dark amber / charcoal for light).
5. Feathered gradient vignette panel (horizontal & vertical alpha ramps) tuned
   to WCAG contrast thresholds (4.5:1 base, 5.5:1 on cluttered backgrounds).
6. Dimension capping (MAX_WIDTH=1920, LANCZOS) and pristine base image preservation.
"""

from __future__ import annotations

import io
import logging
import math
import os
import re
import textwrap
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

logger = logging.getLogger(__name__)

# Constants
KICKER_MAX = 46
TITLE_MAX = 34
HOOK_MAX = 58

KICKER_PT = 0.029
TITLE_PT = 0.086
HOOK_PT = 0.036

SWITCH_MARGIN = 0.90
MAX_WIDTH = 1920

# Dual Palettes
INK_DARK = {
    "kicker": (238, 196, 120),  # Warm gold
    "title": (240, 238, 233),   # Off-white
    "hook": (214, 222, 230),    # Soft slate
    "scrim": (0, 0, 0)          # Black vignette
}

INK_LIGHT = {
    "kicker": (150, 106, 18),   # Dark amber
    "title": (45, 49, 55),      # Charcoal
    "hook": (62, 70, 80),       # Slate charcoal
    "scrim": (255, 255, 255)    # White vignette
}


@dataclass
class OverlayCopy:
    kicker: str
    title: str
    hook: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass
class OverlayResult:
    corner: str
    ink_palette: str  # "dark" or "light"
    contrast_ratio: float
    scrim_alpha: int
    wcag_target: float
    clutter: float
    kicker: str
    title: str
    hook: str
    base_width: int
    base_height: int
    final_width: int
    final_height: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def clean(text: str, max_chars: int, all_caps: bool = False) -> str:
    """
    Clean text, strip markdown formatting and smart quotes,
    and truncate at word boundaries respecting max_chars.
    """
    if not text:
        return ""

    # Replace smart quotes and dashes
    s = text.strip()
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    s = s.replace("—", " - ").replace("–", "-")

    # Strip markdown emphasis and headings
    s = re.sub(r"[*_#`~]+", "", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()

    if all_caps:
        s = s.upper()

    if len(s) <= max_chars:
        return s

    # Word-boundary truncation
    truncated = s[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > int(max_chars * 0.4):
        truncated = truncated[:last_space]

    # Clean trailing punctuation
    truncated = truncated.rstrip(" ,;:-/\\")
    return truncated


def get_system_font(bold: bool = False, size: int = 24) -> ImageFont.ImageFont:
    """
    Locates an appropriate system sans-serif font (Liberation Sans, Arial, Helvetica, DejaVu Sans).
    Falls back gracefully to default PIL font if no TrueType font is found.
    """
    candidates = [
        # Linux
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/TTF/DejaVuSans.ttf",
        # macOS
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        # Windows
        "C:\\Windows\\Fonts\\arialbd.ttf" if bold else "C:\\Windows\\Fonts\\arial.ttf",
    ]

    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=max(8, int(size)))
            except Exception:
                continue

    try:
        return ImageFont.load_default(size=max(8, int(size)))
    except Exception:
        return ImageFont.load_default()


def detect_best_corner(image: Image.Image, switch_margin: float = SWITCH_MARGIN) -> str:
    """
    Spatial placement: Evaluates 4 corner safe boxes (54% width x 50% height).
    Measures edge energy using ImageFilter.FIND_EDGES on a grayscale thumbnail.
    Top-left hysteresis: Keeps top-left unless an alternative corner is at least
    10% emptier (energy < top_left_energy * switch_margin).
    """
    w, h = image.size
    thumb_w = 256
    thumb_h = max(1, int(thumb_w * h / w))
    thumb = image.convert("L").resize((thumb_w, thumb_h), Image.Resampling.BILINEAR)
    edges = thumb.filter(ImageFilter.FIND_EDGES)
    edge_np = np.array(edges, dtype=np.float32)

    box_w = int(0.54 * thumb_w)
    box_h = int(0.50 * thumb_h)

    # 4 corner slices: (y0, y1, x0, x1)
    corners = {
        "top-left": (0, box_h, 0, box_w),
        "top-right": (0, box_h, thumb_w - box_w, thumb_w),
        "bottom-left": (thumb_h - box_h, thumb_h, 0, box_w),
        "bottom-right": (thumb_h - box_h, thumb_h, thumb_w - box_w, thumb_w),
    }

    energies = {}
    for name, (y0, y1, x0, x1) in corners.items():
        patch = edge_np[y0:y1, x0:x1]
        energies[name] = float(np.mean(patch)) if patch.size > 0 else 0.0

    tl_energy = energies["top-left"]
    best_corner = "top-left"
    lowest_energy = tl_energy

    for name in ["top-right", "bottom-left", "bottom-right"]:
        energy = energies[name]
        # Only switch if it beats top-left by at least (1 - switch_margin) = 10%
        if energy < (tl_energy * switch_margin) and energy < lowest_energy:
            lowest_energy = energy
            best_corner = name

    logger.info(f"Corner detection energies: {energies} -> Chosen: {best_corner}")
    return best_corner


def _srgb_to_lin(c: float) -> float:
    c_norm = c / 255.0
    return c_norm / 12.92 if c_norm <= 0.04045 else ((c_norm + 0.055) / 1.055) ** 2.4


def relative_luminance(rgb: Tuple[int, int, int]) -> float:
    """Calculates WCAG relative luminance."""
    r, g, b = rgb[:3]
    return 0.2126 * _srgb_to_lin(r) + 0.7152 * _srgb_to_lin(g) + 0.0722 * _srgb_to_lin(b)


def contrast_ratio(l1: float, l2: float) -> float:
    """Calculates WCAG contrast ratio between two relative luminances."""
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def _binding_rgb_and_clutter(
    image: Image.Image,
    corner: str,
    box_w: int,
    box_h: int
) -> Tuple[Tuple[int, int, int], float, bool]:
    """
    Evaluates background pixels in the corner box:
    - Calculates luminance variance (clutter).
    - Determines if overall background is dark or light.
    - Extracts binding pixel (85th percentile brightest if dark; 15th percentile darkest if light).
    """
    w, h = image.size
    if corner == "top-left":
        crop_box = (0, 0, box_w, box_h)
    elif corner == "top-right":
        crop_box = (w - box_w, 0, w, box_h)
    elif corner == "bottom-left":
        crop_box = (0, h - box_h, box_w, h)
    else:  # bottom-right
        crop_box = (w - box_w, h - box_h, w, h)

    cropped = image.convert("RGB").crop(crop_box)
    np_crop = np.array(cropped, dtype=np.float32)

    # Per-pixel relative luminance
    r = np_crop[:, :, 0]
    g = np_crop[:, :, 1]
    b = np_crop[:, :, 2]
    # Fast luminance approximation for percentile ranking
    lums = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
    flat_lums = lums.flatten()
    if flat_lums.size == 0:
        return (20, 20, 20), 0.0, True

    median_lum = float(np.median(flat_lums))
    std_lum = float(np.std(flat_lums))

    is_dark = median_lum < 0.48

    if is_dark:
        # Binding pixel is 85th percentile brightest pixel in the region
        target_val = float(np.percentile(flat_lums, 85))
    else:
        # Binding pixel is 15th percentile darkest pixel in the region
        target_val = float(np.percentile(flat_lums, 15))

    idx = int(np.argmin(np.abs(flat_lums - target_val)))
    flat_pixels = np_crop.reshape(-1, 3)
    binding_rgb = (
        int(flat_pixels[idx, 0]),
        int(flat_pixels[idx, 1]),
        int(flat_pixels[idx, 2]),
    )

    return binding_rgb, std_lum, is_dark


def _create_feathered_scrim(
    corner: str,
    box_w: int,
    box_h: int,
    scrim_alpha: int,
    scrim_color: Tuple[int, int, int]
) -> Image.Image:
    """
    Constructs a feathered vignette panel:
    - Vertical alpha ramp: fades out towards inner vertical edge.
    - Horizontal alpha ramp: fades out towards inner horizontal edge (outer 18% width).
    """
    # Vertical ramp
    vert_arr = np.zeros((box_h, box_w), dtype=np.float32)
    if "top" in corner:
        for y in range(box_h):
            vert_arr[y, :] = max(0.0, 1.0 - (y / box_h))
    else:  # bottom
        for y in range(box_h):
            vert_arr[y, :] = max(0.0, y / box_h)

    # Horizontal ramp
    horiz_arr = np.ones((box_h, box_w), dtype=np.float32)
    fade_w = max(1, int(0.18 * box_w))

    if "left" in corner:
        fade_start = box_w - fade_w
        for x in range(fade_start, box_w):
            horiz_arr[:, x] = max(0.0, 1.0 - ((x - fade_start) / fade_w))
    else:  # right
        for x in range(fade_w):
            horiz_arr[:, x] = max(0.0, x / fade_w)

    alpha_mask = np.clip(vert_arr * horiz_arr * scrim_alpha, 0, 255).astype(np.uint8)

    scrim_img = Image.new("RGBA", (box_w, box_h), (*scrim_color, 0))
    scrim_img.putalpha(Image.fromarray(alpha_mask))
    return scrim_img


def _solve_scrim_alpha(
    binding_rgb: Tuple[int, int, int],
    title_rgb: Tuple[int, int, int],
    scrim_rgb: Tuple[int, int, int],
    clutter: float
) -> Tuple[int, float, float]:
    """
    Tests scrim alpha in steps of 15 (up to 210) until WCAG contrast ratio is satisfied:
    - Base target: 4.5:1
    - High clutter (> 0.25): 5.5:1
    Returns (scrim_alpha, final_contrast_ratio, wcag_target).
    """
    wcag_target = 5.5 if clutter > 0.25 else 4.5
    title_lum = relative_luminance(title_rgb)

    best_alpha = 0
    best_ratio = 1.0

    for alpha_int in range(0, 211, 15):
        alpha = alpha_int / 255.0
        comp_r = int(scrim_rgb[0] * alpha + binding_rgb[0] * (1.0 - alpha))
        comp_g = int(scrim_rgb[1] * alpha + binding_rgb[1] * (1.0 - alpha))
        comp_b = int(scrim_rgb[2] * alpha + binding_rgb[2] * (1.0 - alpha))
        comp_lum = relative_luminance((comp_r, comp_g, comp_b))
        ratio = contrast_ratio(title_lum, comp_lum)

        best_alpha = alpha_int
        best_ratio = ratio
        if ratio >= wcag_target:
            return alpha_int, ratio, wcag_target

    return best_alpha, best_ratio, wcag_target


def _fit_single_line(
    text: str,
    bold: bool,
    initial_pt: int,
    max_w: int,
    min_pt: int = 10
) -> Tuple[ImageFont.ImageFont, int, int]:
    """Iteratively decreases font size until single line fits max_w."""
    size = int(initial_pt)
    while size >= min_pt:
        font = get_system_font(bold=bold, size=size)
        bbox = font.getbbox(text)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        if text_w <= max_w:
            return font, text_w, text_h
        size -= 1
    font = get_system_font(bold=bold, size=min_pt)
    bbox = font.getbbox(text)
    return font, bbox[2] - bbox[0], bbox[3] - bbox[1]


def _fit_wrapped_title(
    text: str,
    bold: bool,
    initial_pt: int,
    max_w: int,
    max_h: int,
    min_pt: int = 14
) -> Tuple[ImageFont.ImageFont, List[str], int, int]:
    """
    Wraps title across at most 2 lines, finding the maximum legible font size
    that fits both column width and height budget.
    """
    words = text.split()
    size = int(initial_pt)

    while size >= min_pt:
        font = get_system_font(bold=bold, size=size)

        # Try 1 line
        bbox_single = font.getbbox(text)
        single_w = bbox_single[2] - bbox_single[0]
        single_h = bbox_single[3] - bbox_single[1]
        if single_w <= max_w and single_h <= max_h:
            return font, [text], single_w, single_h

        # Try 2 lines with balanced break
        if len(words) > 1:
            best_break = None
            min_diff = 999999
            for i in range(1, len(words)):
                l1 = " ".join(words[:i])
                l2 = " ".join(words[i:])
                diff = abs(len(l1) - len(l2))
                if diff < min_diff:
                    min_diff = diff
                    best_break = (l1, l2)

            if best_break:
                l1, l2 = best_break
                b1 = font.getbbox(l1)
                b2 = font.getbbox(l2)
                w1 = b1[2] - b1[0]
                w2 = b2[2] - b2[0]
                line_h = max(b1[3] - b1[1], b2[3] - b2[1])
                total_w = max(w1, w2)
                total_h = int(line_h * 2 + size * 0.25)

                if total_w <= max_w and total_h <= max_h:
                    return font, [l1, l2], total_w, total_h

        size -= 1

    # Fallback minimal fit
    font = get_system_font(bold=bold, size=min_pt)
    if len(words) > 1:
        mid = len(words) // 2
        lines = [" ".join(words[:mid]), " ".join(words[mid:])]
    else:
        lines = [text]
    b = font.getbbox(lines[0])
    return font, lines, b[2] - b[0], (b[3] - b[1]) * len(lines)


def apply_editorial_overlay(
    image_input: Union[Image.Image, bytes, str],
    kicker: str = "",
    title: str = "",
    hook: str = "",
    corner: str = "auto",
    max_width: int = MAX_WIDTH
) -> Tuple[Image.Image, OverlayResult]:
    """
    Applies deterministic editorial cover typography over the image.

    Args:
        image_input: PIL Image, raw image bytes, or file path.
        kicker: All-caps beat/topic label (max 46 chars).
        title: All-caps headline naming story subject (max 34 chars).
        hook: Sentence-case reader stake (max 58 chars).
        corner: "auto", "top-left", "top-right", "bottom-left", "bottom-right".
        max_width: Maximum image width (LANCZOS downscaled if exceeded).

    Returns:
        (composited_image, overlay_result_metadata)
    """
    if isinstance(image_input, bytes):
        base_img = Image.open(io.BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, str):
        base_img = Image.open(image_input).convert("RGB")
    else:
        base_img = image_input.copy().convert("RGB")

    orig_w, orig_h = base_img.size

    # Cap dimensions if width exceeds max_width
    if orig_w > max_width:
        new_w = max_width
        new_h = int(orig_h * (max_width / orig_w))
        base_img = base_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        w, h = new_w, new_h
    else:
        w, h = orig_w, orig_h

    # Clean copy strings
    clean_kicker = clean(kicker, KICKER_MAX, all_caps=True)
    clean_title = clean(title, TITLE_MAX, all_caps=True)
    clean_hook = clean(hook, HOOK_MAX, all_caps=False)

    # Fallbacks if fields are empty
    if not clean_kicker and not clean_title and not clean_hook:
        clean_kicker = "EDITORIAL // FOCUS"
        clean_title = "FEATURED STORY"
        clean_hook = "Key industry analysis and strategic implications."

    # Determine placement corner
    chosen_corner = detect_best_corner(base_img) if corner == "auto" else corner
    if chosen_corner not in ["top-left", "top-right", "bottom-left", "bottom-right"]:
        chosen_corner = "top-left"

    # Box bounds (54% width x 50% height)
    box_w = max(10, int(0.54 * w))
    box_h = max(10, int(0.50 * h))

    # Padding from image canvas edges
    pad_x = max(2, int(0.045 * w))
    pad_y = max(2, int(0.045 * h))
    col_w = max(10, box_w - pad_x - max(1, int(0.02 * w)))

    # Typography target point sizes based on canvas height h
    kicker_pt = max(11, int(KICKER_PT * h))
    title_pt = max(18, int(TITLE_PT * h))
    hook_pt = max(12, int(HOOK_PT * h))

    # Font fitting
    kicker_font, kw, kh = _fit_single_line(clean_kicker, bold=True, initial_pt=kicker_pt, max_w=col_w)
    title_font, title_lines, tw, th = _fit_wrapped_title(
        clean_title, bold=True, initial_pt=title_pt, max_w=col_w, max_h=int(box_h * 0.55)
    )
    hook_font, hw, hh = _fit_single_line(clean_hook, bold=False, initial_pt=hook_pt, max_w=col_w)

    # Spacing
    gap_kicker_title = max(6, int(0.014 * h))
    gap_title_hook = max(8, int(0.018 * h))
    title_line_spacing = max(4, int(title_font.size * 0.22)) if hasattr(title_font, "size") else 6

    # Calculate total height of text stack
    rendered_title_h = 0
    for idx, tl in enumerate(title_lines):
        tb = title_font.getbbox(tl)
        line_h = tb[3] - tb[1]
        rendered_title_h += line_h
        if idx > 0:
            rendered_title_h += title_line_spacing

    total_text_h = kh + gap_kicker_title + rendered_title_h + gap_title_hook + hh

    # Corner origin for text
    if chosen_corner == "top-left":
        start_x = pad_x
        start_y = pad_y
        scrim_origin = (0, 0)
    elif chosen_corner == "top-right":
        start_x = w - box_w + pad_x
        start_y = pad_y
        scrim_origin = (w - box_w, 0)
    elif chosen_corner == "bottom-left":
        start_x = pad_x
        start_y = h - pad_y - total_text_h
        scrim_origin = (0, h - box_h)
    else:  # bottom-right
        start_x = w - box_w + pad_x
        start_y = h - pad_y - total_text_h
        scrim_origin = (w - box_w, h - box_h)

    # Adaptive contrast & binding pixel
    binding_rgb, clutter, is_dark = _binding_rgb_and_clutter(base_img, chosen_corner, box_w, box_h)
    palette_name = "dark" if is_dark else "light"
    palette = INK_DARK if is_dark else INK_LIGHT

    scrim_alpha, contrast_achieved, wcag_target = _solve_scrim_alpha(
        binding_rgb=binding_rgb,
        title_rgb=palette["title"],
        scrim_rgb=palette["scrim"],
        clutter=clutter
    )

    # Composite feathered scrim
    canvas_rgba = base_img.convert("RGBA")
    if scrim_alpha > 0:
        scrim_layer = _create_feathered_scrim(
            corner=chosen_corner,
            box_w=box_w,
            box_h=box_h,
            scrim_alpha=scrim_alpha,
            scrim_color=palette["scrim"]
        )
        canvas_rgba.paste(scrim_layer, scrim_origin, scrim_layer)

    # Render typography
    draw = ImageDraw.Draw(canvas_rgba)
    curr_y = start_y

    # 1. Kicker
    if clean_kicker:
        draw.text((start_x, curr_y), clean_kicker, font=kicker_font, fill=palette["kicker"])
        curr_y += kh + gap_kicker_title

    # 2. Title (1 or 2 lines)
    if title_lines:
        for idx, t_line in enumerate(title_lines):
            draw.text((start_x, curr_y), t_line, font=title_font, fill=palette["title"])
            tb = title_font.getbbox(t_line)
            curr_y += (tb[3] - tb[1]) + (title_line_spacing if idx < len(title_lines) - 1 else 0)
        curr_y += gap_title_hook

    # 3. Hook
    if clean_hook:
        draw.text((start_x, curr_y), clean_hook, font=hook_font, fill=palette["hook"])

    final_img = canvas_rgba.convert("RGB")

    metadata = OverlayResult(
        corner=chosen_corner,
        ink_palette=palette_name,
        contrast_ratio=round(contrast_achieved, 2),
        scrim_alpha=scrim_alpha,
        wcag_target=wcag_target,
        clutter=round(clutter, 3),
        kicker=clean_kicker,
        title=clean_title,
        hook=clean_hook,
        base_width=orig_w,
        base_height=orig_h,
        final_width=w,
        final_height=h
    )

    return final_img, metadata


def generate_overlay_copy(
    text: str,
    article_title: Optional[str] = None,
    article_context: Optional[Dict[str, Any]] = None,
    user_instructions: Optional[str] = None
) -> OverlayCopy:
    """
    Generates high-contrast 3-tier editorial copy (Kicker, Title, Hook)
    using LLM with high editorial standards or direct semantic synthesis.
    """
    article_context = article_context or {}
    art_title = article_title or article_context.get("title", "")
    art_thesis = article_context.get("thesis") or article_context.get("one_big_thing", "")
    art_vertical = article_context.get("vertical") or article_context.get("topic", "")

    # Clean fallback defaults
    fallback_kicker = clean(
        f"{art_vertical.upper()} // ANALYSIS" if art_vertical else "EXECUTIVE EDITORIAL",
        KICKER_MAX,
        all_caps=True
    )
    fallback_title = clean(
        art_title or text[:34],
        TITLE_MAX,
        all_caps=True
    )
    fallback_hook = clean(
        art_thesis or article_context.get("hook") or article_context.get("deck") or text[:58],
        HOOK_MAX,
        all_caps=False
    )

    # Prompt for LLM copy generation
    copy_prompt = f"""You are the senior editorial art director creating the cover typography for an executive magazine header image.
Read the story context and text selection below.
Produce the 3-tier typography copy adhering strictly to these rules:

1. KICKER (max 46 chars, ALL CAPS):
   - Beat/topic label and governing dynamic.
   - Format: "DOMAIN // CORE TENSION" (e.g. "LLM HARDWARE // MEMORY BOTTLENECK", "AUTOMATION // WAGE ARBITRAGE", "BATTERY DENSITY // PACK LATENCY").

2. TITLE (max 34 chars, ALL CAPS):
   - Punchy, arresting headline naming the story subject or decisive shift.
   - Fits on at most 2 short lines. No jargon soup.

3. HOOK (max 58 chars, Sentence case):
   - Articulates the reader stake or consequence. Crisp sentence fragment.

ARTICLE TITLE: {art_title}
VERTICAL: {art_vertical}
CORE THESIS: {art_thesis}
SELECTION TEXT: {text[:600]}
USER DIRECTION: {user_instructions or 'None'}

Respond ONLY with valid JSON:
{{
  "kicker": "...",
  "title": "...",
  "hook": "..."
}}
"""

    try:
        from src.services.context_image.entity_extractor import EntityExtractor
        extractor = EntityExtractor()
        raw_response = extractor._call_llm(copy_prompt)

        import json
        clean_json = raw_response.strip()
        if "```json" in clean_json:
            clean_json = clean_json.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_json:
            clean_json = clean_json.split("```")[1].split("```")[0].strip()

        data = json.loads(clean_json)
        kicker_out = clean(data.get("kicker") or fallback_kicker, KICKER_MAX, all_caps=True)
        title_out = clean(data.get("title") or fallback_title, TITLE_MAX, all_caps=True)
        hook_out = clean(data.get("hook") or fallback_hook, HOOK_MAX, all_caps=False)

        return OverlayCopy(kicker=kicker_out, title=title_out, hook=hook_out)
    except Exception as e:
        logger.warning(f"Overlay copy LLM generation failed: {e}. Using deterministic fallback.")
        return OverlayCopy(
            kicker=fallback_kicker,
            title=fallback_title,
            hook=fallback_hook
        )
