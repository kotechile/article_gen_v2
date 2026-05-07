import copy
import json
import os
import re
from datetime import datetime, timezone
from html import escape
from typing import Any

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


def build_fontawesome_icon_markup(icon_class_string: str) -> str:
    return f'<i class="{escape(_helper_classes_for_icon(icon_class_string), quote=True)}" aria-hidden="true"></i>'


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

.cg-fa-icon.cg-fa-icon--solid,
.cg-fa-icon.cg-fa-icon--regular {
    font-family: "Font Awesome 6 Free" !important;
}

.cg-fa-icon.cg-fa-icon--solid {
    font-weight: 900 !important;
}

.cg-fa-icon.cg-fa-icon--regular {
    font-weight: 400 !important;
}

.cg-fa-icon.cg-fa-icon--brands {
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

        attr_pattern = re.compile(
            rf'(?P<attr>\bclass\s*=\s*["\'])(?P<value>[^"\']*{placeholder_pattern}[^"\']*)(?P<quote>["\'])',
            re.IGNORECASE,
        )

        def _replace_attr(match: re.Match[str]) -> str:
            original_tokens = [token for token in match.group("value").split() if token]
            filtered_tokens: list[str] = []

            for token in original_tokens:
                if placeholder in token:
                    continue
                if token in LEGACY_FONT_AWESOME_STYLE_TOKENS:
                    continue
                filtered_tokens.append(token)

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
