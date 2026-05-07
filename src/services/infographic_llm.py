import copy
import json
import os
import re
from datetime import datetime, timezone
from html import escape
from typing import Any, Optional

SAFE_FALLBACK_ICONS = (
    "fa-solid fa-circle-info",
    "fa-solid fa-lightbulb",
    "fa-solid fa-chart-column",
    "fa-solid fa-gears",
    "fa-solid fa-bolt",
    "fa-solid fa-star",
)

STYLE_PREFIX_ALIASES = {
    "fas": "fa-solid",
    "far": "fa-regular",
    "fab": "fa-brands",
}

ALLOWED_STYLE_PREFIXES = {"fa-solid", "fa-regular", "fa-brands"}

LEGACY_FONT_AWESOME_STYLE_TOKENS = {
    "fa",
    "fas",
    "far",
    "fab",
    "fal",
    "fat",
    "fad",
    "fass",
    "fasr",
    "fasl",
    "fast",
    "fa-solid",
    "fa-regular",
    "fa-brands",
    "fa-light",
    "fa-thin",
    "fa-duotone",
    "fa-sharp",
    "fa-sharp-duotone",
    "fa-sharp fa-solid",
}

ALLOWED_SOLID_ICON_TOKENS = {
    "fa-arrow-trend-up",
    "fa-balance-scale",
    "fa-bell",
    "fa-bolt",
    "fa-book",
    "fa-bookmark",
    "fa-brain",
    "fa-briefcase",
    "fa-bug",
    "fa-building-columns",
    "fa-bullseye",
    "fa-calendar",
    "fa-camera",
    "fa-car",
    "fa-cart-shopping",
    "fa-chart-column",
    "fa-chart-line",
    "fa-chart-pie",
    "fa-check",
    "fa-circle-check",
    "fa-circle-info",
    "fa-clock",
    "fa-cloud",
    "fa-code",
    "fa-coins",
    "fa-comment",
    "fa-comments",
    "fa-compass",
    "fa-database",
    "fa-droplet",
    "fa-envelope",
    "fa-fire",
    "fa-flag",
    "fa-flask",
    "fa-gavel",
    "fa-gears",
    "fa-gift",
    "fa-globe",
    "fa-graduation-cap",
    "fa-handshake",
    "fa-heart",
    "fa-house",
    "fa-house-laptop",
    "fa-key",
    "fa-landmark",
    "fa-laptop",
    "fa-layer-group",
    "fa-leaf",
    "fa-lightbulb",
    "fa-list-check",
    "fa-lock",
    "fa-magnifying-glass",
    "fa-medal",
    "fa-message",
    "fa-mobile-screen",
    "fa-money-bill",
    "fa-money-bill-wave",
    "fa-money-bill-trend-up",
    "fa-moon",
    "fa-pen",
    "fa-pencil",
    "fa-phone",
    "fa-piggy-bank",
    "fa-plane",
    "fa-receipt",
    "fa-robot",
    "fa-rocket",
    "fa-sack-dollar",
    "fa-scale-balanced",
    "fa-seedling",
    "fa-server",
    "fa-shield-halved",
    "fa-ship",
    "fa-shop",
    "fa-star",
    "fa-sun",
    "fa-terminal",
    "fa-train",
    "fa-tree",
    "fa-triangle-exclamation",
    "fa-truck",
    "fa-user",
    "fa-users",
    "fa-wallet",
    "fa-wand-magic-sparkles",
    "fa-wifi",
    "fa-wrench",
}

ALLOWED_REGULAR_ICON_TOKENS = {
    "fa-bell",
    "fa-bookmark",
    "fa-calendar",
    "fa-circle-check",
    "fa-circle-info",
    "fa-clock",
    "fa-comment",
    "fa-envelope",
    "fa-heart",
    "fa-lightbulb",
    "fa-message",
    "fa-star",
}

ALLOWED_BRAND_ICON_TOKENS = {
    "fa-amazon",
    "fa-apple",
    "fa-discord",
    "fa-facebook",
    "fa-github",
    "fa-google",
    "fa-instagram",
    "fa-linkedin",
    "fa-microsoft",
    "fa-pinterest",
    "fa-reddit",
    "fa-slack",
    "fa-spotify",
    "fa-tiktok",
    "fa-whatsapp",
    "fa-x-twitter",
    "fa-youtube",
}

ICON_KEYWORD_FALLBACKS = (
    (
        {
            "automation",
            "automatic",
            "bot",
            "configure",
            "gear",
            "process",
            "setup",
            "system",
            "workflow",
        },
        "fa-solid fa-gears",
    ),
    (
        {
            "analytics",
            "budget",
            "cash",
            "chart",
            "cost",
            "data",
            "expense",
            "finance",
            "growth",
            "metric",
            "money",
            "rate",
            "save",
            "savings",
            "spending",
            "trend",
        },
        "fa-solid fa-chart-column",
    ),
    (
        {
            "alert",
            "energy",
            "fast",
            "instant",
            "power",
            "risk",
            "speed",
            "urgent",
            "volatility",
            "warning",
        },
        "fa-solid fa-bolt",
    ),
    (
        {
            "guide",
            "help",
            "idea",
            "insight",
            "learn",
            "lesson",
            "tip",
        },
        "fa-solid fa-lightbulb",
    ),
)


def _allowed_tokens_for_style(style_prefix: str) -> set[str]:
    if style_prefix == "fa-regular":
        return ALLOWED_REGULAR_ICON_TOKENS
    if style_prefix == "fa-brands":
        return ALLOWED_BRAND_ICON_TOKENS
    return ALLOWED_SOLID_ICON_TOKENS


def choose_icon_fallback(item_text: str = "", description_text: str = "", index: int = 0) -> str:
    haystack = f"{item_text} {description_text}".lower()
    for keywords, fallback in ICON_KEYWORD_FALLBACKS:
        if any(keyword in haystack for keyword in keywords):
            return fallback
    return SAFE_FALLBACK_ICONS[index % len(SAFE_FALLBACK_ICONS)]


def normalize_infographic_icon(
    raw_value: Any,
    *,
    item_text: str = "",
    description_text: str = "",
    index: int = 0,
) -> tuple[str, dict[str, Any]]:
    raw_text = str(raw_value or "").strip()
    tokens = [token.strip() for token in raw_text.replace(",", " ").split() if token.strip()]

    style_prefix = ""
    icon_token = ""

    for token in tokens:
        normalized = STYLE_PREFIX_ALIASES.get(token, token)
        if normalized in ALLOWED_STYLE_PREFIXES and not style_prefix:
            style_prefix = normalized
            continue
        if normalized.startswith("fa-") and normalized not in ALLOWED_STYLE_PREFIXES and not icon_token:
            icon_token = normalized

    if not style_prefix and icon_token:
        style_prefix = "fa-brands" if icon_token in ALLOWED_BRAND_ICON_TOKENS else "fa-solid"

    allowed_tokens = _allowed_tokens_for_style(style_prefix) if style_prefix else set()
    if style_prefix and icon_token and icon_token in allowed_tokens:
        return f"{style_prefix} {icon_token}", {
            "raw": raw_text,
            "normalized": f"{style_prefix} {icon_token}",
            "status": "accepted",
        }

    fallback = choose_icon_fallback(item_text=item_text, description_text=description_text, index=index)
    return fallback, {
        "raw": raw_text,
        "normalized": fallback,
        "status": "fallback",
        "reason": "invalid_or_unknown_icon_class",
    }


def normalize_infographic_payload_icons(generated_data: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    normalized_payload = copy.deepcopy(generated_data)
    icon_audit: list[dict[str, Any]] = []

    items = normalized_payload.get("items")
    if not isinstance(items, list):
        return normalized_payload, icon_audit

    for item_index, item in enumerate(items):
        if not isinstance(item, dict):
            continue

        item_number = item_index + 1
        icon_key = f"icon{item_number}"
        item_key = f"item{item_number}"
        description_key = f"description{item_number}"
        if icon_key not in item:
            continue

        normalized_icon, audit = normalize_infographic_icon(
            item.get(icon_key),
            item_text=str(item.get(item_key) or ""),
            description_text=str(item.get(description_key) or ""),
            index=item_index,
        )
        item[icon_key] = normalized_icon
        icon_audit.append(
            {
                "field": icon_key,
                "item_index": item_index,
                **audit,
            }
        )

    return normalized_payload, icon_audit


def _helper_classes_for_icon(icon_class_string: str) -> str:
    tokens = [token for token in str(icon_class_string or "").split() if token]
    helper_tokens = ["cg-fa-icon"]
    if "fa-brands" in tokens:
        helper_tokens.append("cg-fa-icon--brands")
    elif "fa-regular" in tokens:
        helper_tokens.append("cg-fa-icon--regular")
    else:
        helper_tokens.append("cg-fa-icon--solid")
    return " ".join([*helper_tokens, *tokens])


def _filter_preserved_icon_classes(class_tokens: list[str], placeholder: str) -> list[str]:
    preserved: list[str] = []
    for token in class_tokens:
        if placeholder in token:
            continue
        if token in LEGACY_FONT_AWESOME_STYLE_TOKENS:
            continue
        if token.startswith("fa-"):
            continue
        preserved.append(token)
    return preserved


def _icon_svg_inner(icon_class_string: str) -> str:
    token_set = set(str(icon_class_string or "").split())

    if "fa-chart-column" in token_set:
        return (
            '<rect x="10" y="30" width="10" height="22" rx="2" fill="currentColor"/>'
            '<rect x="27" y="20" width="10" height="32" rx="2" fill="currentColor"/>'
            '<rect x="44" y="12" width="10" height="40" rx="2" fill="currentColor"/>'
        )
    if "fa-chart-line" in token_set:
        return (
            '<polyline points="10,42 24,30 34,34 52,18" '
            'fill="none" stroke="currentColor" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/>'
            '<circle cx="10" cy="42" r="3" fill="currentColor"/>'
            '<circle cx="24" cy="30" r="3" fill="currentColor"/>'
            '<circle cx="34" cy="34" r="3" fill="currentColor"/>'
            '<circle cx="52" cy="18" r="3" fill="currentColor"/>'
        )
    if "fa-shield-halved" in token_set:
        return (
            '<path d="M32 8 50 14v14c0 13-8 23-18 28C22 51 14 41 14 28V14z" '
            'fill="none" stroke="currentColor" stroke-width="4" stroke-linejoin="round"/>'
            '<path d="M32 12v40c8-4 14-12 14-24V17z" fill="currentColor" opacity="0.9"/>'
        )
    if "fa-gears" in token_set:
        return (
            '<circle cx="26" cy="38" r="9" fill="none" stroke="currentColor" stroke-width="5"/>'
            '<circle cx="42" cy="24" r="7" fill="none" stroke="currentColor" stroke-width="5"/>'
            '<path d="M26 22v6M26 48v6M10 38h6M36 38h6M15 27l4 4M33 45l4 4M15 49l4-4M33 31l4-4" '
            'stroke="currentColor" stroke-width="4" stroke-linecap="round"/>'
            '<path d="M42 11v4M42 33v4M29 24h4M51 24h4M33 15l3 3M48 30l3 3M33 33l3-3M48 18l3-3" '
            'stroke="currentColor" stroke-width="3" stroke-linecap="round"/>'
        )
    if "fa-bolt" in token_set:
        return '<path d="M36 8 18 34h11l-3 22 20-28H35z" fill="currentColor"/>'
    if "fa-lightbulb" in token_set:
        return (
            '<path d="M32 10c-10 0-18 8-18 18 0 7 4 12 9 16 2 1 3 3 3 5h12c0-2 1-4 3-5 5-4 9-9 9-16 0-10-8-18-18-18Z" '
            'fill="none" stroke="currentColor" stroke-width="4" stroke-linejoin="round"/>'
            '<path d="M25 52h14M26 58h12" stroke="currentColor" stroke-width="4" stroke-linecap="round"/>'
        )
    if "fa-star" in token_set:
        return '<path d="m32 9 7 14 15 2-11 10 3 15-14-8-14 8 3-15-11-10 15-2z" fill="currentColor"/>'
    if "fa-robot" in token_set:
        return (
            '<rect x="16" y="18" width="32" height="24" rx="6" fill="none" stroke="currentColor" stroke-width="4"/>'
            '<circle cx="26" cy="30" r="3" fill="currentColor"/>'
            '<circle cx="38" cy="30" r="3" fill="currentColor"/>'
            '<path d="M24 40h16M32 12v6M22 18l-4-4M42 18l4-4M20 42v8M44 42v8" '
            'stroke="currentColor" stroke-width="4" stroke-linecap="round"/>'
        )
    if "fa-lock" in token_set:
        return (
            '<rect x="18" y="28" width="28" height="24" rx="4" fill="none" stroke="currentColor" stroke-width="4"/>'
            '<path d="M24 28v-6c0-5 3-10 8-10s8 5 8 10v6" fill="none" stroke="currentColor" stroke-width="4" stroke-linecap="round"/>'
        )

    return (
        '<circle cx="32" cy="32" r="22" fill="none" stroke="currentColor" stroke-width="4"/>'
        '<path d="M32 24v16" stroke="currentColor" stroke-width="5" stroke-linecap="round"/>'
        '<circle cx="32" cy="18" r="3" fill="currentColor"/>'
    )


def build_fontawesome_icon_markup(icon_class_string: str, extra_classes: Optional[list[str]] = None) -> str:
    class_tokens = []
    if extra_classes:
        class_tokens.extend(extra_classes)
    class_tokens.extend(_helper_classes_for_icon(icon_class_string).split())
    class_attr = escape(" ".join(dict.fromkeys(class_tokens)), quote=True)
    svg_inner = _icon_svg_inner(icon_class_string)
    return (
        f'<span class="{class_attr}" aria-hidden="true">'
        f'<svg viewBox="0 0 64 64" focusable="false" aria-hidden="true">{svg_inner}</svg>'
        f"</span>"
    )


def inject_fontawesome_icon_styles(css_content: str) -> str:
    sanitized_css = re.sub(
        r"@import\s+url\((['\"]?)([^)\"']*font-?awesome[^)\"']*)\1\)\s*;?",
        "",
        css_content,
        flags=re.IGNORECASE,
    )
    helper_css = """
/* Codex infographic icon hardening */
.cg-fa-icon {
    display: inline-block;
    line-height: 1;
    font-style: normal;
    text-rendering: auto;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

.cg-fa-icon::before {
    display: inline-block;
    text-rendering: auto;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

.cg-fa-icon svg {
    display: block;
    width: 100%;
    height: 100%;
    overflow: visible;
}

.cg-fa-icon.cg-fa-icon--solid,
.cg-fa-icon.cg-fa-icon--regular,
.cg-fa-icon.cg-fa-icon--solid::before,
.cg-fa-icon.cg-fa-icon--regular::before {
    font-family: "Font Awesome 6 Free" !important;
}

.cg-fa-icon.cg-fa-icon--solid,
.cg-fa-icon.cg-fa-icon--solid::before {
    font-weight: 900 !important;
}

.cg-fa-icon.cg-fa-icon--regular,
.cg-fa-icon.cg-fa-icon--regular::before {
    font-weight: 400 !important;
}

.cg-fa-icon.cg-fa-icon--brands,
.cg-fa-icon.cg-fa-icon--brands::before {
    font-family: "Font Awesome 6 Brands" !important;
    font-weight: 400 !important;
}
"""
    if "Codex infographic icon hardening" in sanitized_css:
        return sanitized_css.strip()
    return f"{sanitized_css.rstrip()}\n\n{helper_css}".strip()


def apply_icon_markup_to_html(
    html_content: str,
    replacements: dict[str, str],
) -> tuple[str, list[dict[str, Any]]]:
    updated_html = html_content
    render_audit: list[dict[str, Any]] = []

    for key, icon_class_string in replacements.items():
        placeholder = f"+{key}+"
        helper_classes = _helper_classes_for_icon(icon_class_string)
        helper_markup = build_fontawesome_icon_markup(icon_class_string)

        placeholder_pattern = re.escape(placeholder)

        class_tag_pattern = re.compile(
            rf"<(?P<tag>[a-z0-9]+)\b(?P<before>[^>]*?)\bclass\s*=\s*(?P<quote>[\"'])(?P<class>[^\"']*{placeholder_pattern}[^\"']*)(?P=quote)(?P<after>[^>]*)>(?P<inner>.*?)</(?P=tag)>",
            re.IGNORECASE | re.DOTALL,
        )

        def _replace_class_tag(match: re.Match[str]) -> str:
            class_tokens = [token for token in match.group("class").split() if token]
            preserved = _filter_preserved_icon_classes(class_tokens, placeholder)
            return build_fontawesome_icon_markup(icon_class_string, extra_classes=preserved)

        updated_html, legacy_tag_count = class_tag_pattern.subn(_replace_class_tag, updated_html)

        attr_pattern = re.compile(
            rf'(?P<attr>\bclass\s*=\s*["\'])(?P<value>[^"\']*{placeholder_pattern}[^"\']*)(?P<quote>["\'])',
            re.IGNORECASE,
        )

        def _replace_attr(match: re.Match[str]) -> str:
            original_tokens = [token for token in match.group("value").split() if token]
            filtered_tokens = _filter_preserved_icon_classes(original_tokens, placeholder)

            value = " ".join([*filtered_tokens, *helper_classes.split()])
            value = re.sub(r"\s+", " ", value).strip()
            return f'{match.group("attr")}{value}{match.group("quote")}'

        updated_html, attr_count = attr_pattern.subn(_replace_attr, updated_html)

        text_pattern = re.compile(rf'>\s*{placeholder_pattern}\s*<')
        updated_html, text_count = text_pattern.subn(f'>{helper_markup}<', updated_html)

        plain_count = updated_html.count(placeholder)
        if plain_count:
            updated_html = updated_html.replace(placeholder, icon_class_string)

        render_audit.append(
            {
                "field": key,
                "icon_class": icon_class_string,
                "legacy_tag_replacements": legacy_tag_count,
                "attribute_replacements": attr_count,
                "text_replacements": text_count,
                "plain_replacements": plain_count,
            }
        )

    return updated_html, render_audit


def append_infographic_llm_log(log_path: str, payload: dict[str, Any]) -> None:
    if not log_path:
        return

    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **payload,
    }

    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False))
        handle.write("\n")


def write_infographic_render_debug_artifacts(
    output_dir: str,
    *,
    template_id: Any,
    request_timestamp: str,
    html_content: str,
    css_content: str,
) -> dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    safe_stamp = re.sub(r"[^0-9A-Za-z_-]+", "_", request_timestamp)
    base_name = f"template_{template_id}_{safe_stamp}"
    html_path = os.path.join(output_dir, f"{base_name}.html")
    css_path = os.path.join(output_dir, f"{base_name}.css")

    with open(html_path, "w", encoding="utf-8") as html_handle:
        html_handle.write(html_content)

    with open(css_path, "w", encoding="utf-8") as css_handle:
        css_handle.write(css_content)

    return {
        "html_path": html_path,
        "css_path": css_path,
    }
