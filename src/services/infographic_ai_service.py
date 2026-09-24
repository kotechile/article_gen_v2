"""
Infographic AI Service for AI-driven visual diagram and infographic generation.

Supports:
1. Classic Archetypes:
   - Technical and Scientific Diagrams
   - Step-by-Step Guides and Recipes
   - Flowcharts and Whiteboard Sketches
   - Modular Explainers
   - Timelines & Chronological Overviews (modernized to avoid antique parchment looks)
   - Data Visualizations
   - Playful and Viral Menus/Listicles
2. Specific Curated Styles across 5 Categories:
   - Process & Sequential Flow (Isometric 3D, Modern Milestone, Vintage/Retro Historical, Minimalist Flat Journey, Watercolor Lifecycle)
   - Comparison & Contrast (Dark Mode Neon Duel, Clean Scandinavian Pros/Cons, Glassmorphism Venn Diagram, Bauhaus Quadrant Matrix)
   - Data & Statistics (UI/UX App Dashboard, Swiss Grid Typography Stat Sheet, Futuristic Hologram Map, Neumorphism Funnel)
   - Structure & Hierarchy (Low-Poly 3D Pyramid, Material Design Hub & Spoke, Technical Blueprint Anatomy, Hand-Drawn Mind Map)
   - Lists & Summaries (Synthwave/Outrun Checklist, Vintage Comic Pop-Art Top 10, Monochrome Typographic Cheat Sheet, Split-Tone Duotone Problem/Solution)
3. Automatic archetype detection based on text content.
"""

import logging
import re
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

ARCHETYPE_DESCRIPTIONS = {
    "technical_scientific": "Technical and Scientific Diagrams: Explanations of physics concepts, technical systems like Kubernetes pods, or biological processes using real-world data and grounding.",
    "step_by_step": "Step-by-Step Guides and Recipes: Visual instructions showing processes like cooking recipes or DIY workflows.",
    "flowchart_whiteboard": "Flowcharts and Whiteboard Sketches: Hand-drawn styles, notebook-paper flowcharts, and whiteboard layouts that mimic organic brainstorming.",
    "modular_explainer": "Modular Explainers: Central hubs with connected components designed to show how a complex system works.",
    "timeline_historical": "Timelines and Chronological Overviews: Clean modern chronological milestones tracking events or product evolution without antique parchment looks.",
    "data_visualization": "Data Visualizations: Visuals transformed from raw metrics or CSV data into structured layouts like scrum boards or financial summaries.",
    "playful_viral": "Playful and Viral Menus/Listicles: Lighthearted, illustrated menus, humorous life steps, or pop-art graphics."
}

INFOGRAPHIC_CATEGORIES = {
    "process_sequential": "Process & Sequential Flow",
    "comparison_contrast": "Comparison & Contrast",
    "data_statistics": "Data & Statistics",
    "structure_hierarchy": "Structure & Hierarchy",
    "lists_summaries": "Lists & Summaries",
    "classic_archetypes": "Classic Archetypes"
}

STYLE_PRESETS: Dict[str, Dict[str, str]] = {
    # Process & Sequential Flow
    "step_by_step_isometric": {
        "name": "The Step-by-Step Flowchart",
        "style_tag": "Isometric 3D Style",
        "category_id": "process_sequential",
        "category": "Process & Sequential Flow",
        "description": "Clean, colorful 3D blocks and miniature vector elements on a light gray background with a zigzag path.",
        "prompt_template": "Create a 3D isometric infographic explaining [TOPIC]. Layout: A sequential, left-to-right zigzag path. Style: Clean, colorful 3D blocks and miniature vector elements on a light gray background. Text: Apply Smart Brevity—use bold 1-2 word step titles with a single sentence description beneath each."
    },
    "timeline_modern": {
        "name": "The Modern Milestone Timeline",
        "style_tag": "Sleek Tech Style",
        "category_id": "process_sequential",
        "category": "Process & Sequential Flow",
        "description": "Clean, contemporary tech roadmap with distinct milestone nodes and modern typography without antique textures.",
        "prompt_template": "Design a sleek, modern chronological timeline infographic about [TOPIC]. Style: Contemporary high-tech aesthetic, clean dark slate or crisp white background, vibrant gradient milestone nodes, and sharp modern typography without any antique or parchment textures. Layout: A streamlined chronological milestone progression track with clear dates, phase badges, and concise takeaway callouts."
    },
    "timeline_historical_vintage": {
        "name": "The Historical Timeline",
        "style_tag": "Vintage/Retro Style",
        "category_id": "process_sequential",
        "category": "Process & Sequential Flow",
        "description": "Vintage/retro aesthetic with muted earthy tones (sepia, mustard, faded teal), textured paper background, and classic serif typography.",
        "prompt_template": "Design a vertical timeline infographic about [TOPIC]. Style: Vintage/retro aesthetic with muted earthy tones (sepia, mustard, faded teal), textured paper background, and classic serif typography. Layout: A central winding line with alternating left/right date nodes and minimalist icon illustrations."
    },
    "user_journey_flat": {
        "name": "The User Journey Map",
        "style_tag": "Minimalist Flat Style",
        "category_id": "process_sequential",
        "category": "Process & Sequential Flow",
        "description": "Ultra-minimalist horizontal track divided into 4 distinct phases (Awareness, Consideration, Action, Loyalty) using 3 solid colors.",
        "prompt_template": "Generate a flat-design user journey infographic for [TOPIC]. Style: Ultra-minimalist, using only 3 solid colors. Layout: A horizontal track divided into 4 distinct phases (Awareness, Consideration, Action, Loyalty). Include simple line-art icons for each phase and punchy, scannable bullet points."
    },
    "lifecycle_loop_watercolor": {
        "name": "The Lifecycle Loop",
        "style_tag": "Organic Watercolor Style",
        "category_id": "process_sequential",
        "category": "Process & Sequential Flow",
        "description": "Soft, organic watercolor textures with fluid shapes and smooth gradients in a 4-5 arrow circular loop.",
        "prompt_template": "Create a circular lifecycle infographic explaining the continuous loop of [TOPIC]. Style: Soft, organic watercolor textures with fluid shapes and smooth gradients. Layout: 4 to 5 arrows forming a closed circle, with brief text callouts radiating outward from each segment."
    },

    # Comparison & Contrast
    "side_by_side_neon": {
        "name": "The Side-by-Side Duel",
        "style_tag": "Dark Mode Neon",
        "category_id": "comparison_contrast",
        "category": "Comparison & Contrast",
        "description": "Dark mode cyberpunk aesthetic with glowing cyan and magenta accents and symmetrical split-screen comparison.",
        "prompt_template": "Design a comparison infographic analyzing [TOPIC A] vs [TOPIC B]. Style: Dark mode, cyberpunk aesthetic with glowing neon accents (cyan and magenta) against a deep black background. Layout: A symmetrical split-screen vertical layout, using stylized checkmarks and X's for feature comparisons."
    },
    "pros_cons_scandinavian": {
        "name": "The Pros and Cons Scales",
        "style_tag": "Clean Scandinavian",
        "category_id": "comparison_contrast",
        "category": "Comparison & Contrast",
        "description": "Lots of white space, muted pastel colors, and sans-serif typography with two minimalist drop-shadow card columns.",
        "prompt_template": "Generate a pros and cons infographic for [TOPIC]. Style: Clean Scandinavian design—lots of white space, muted pastel colors, and sans-serif typography. Layout: Two minimalist columns with subtle drop-shadow cards holding short, punchy bullet points."
    },
    "venn_diagram_glassmorphism": {
        "name": "The Venn Diagram",
        "style_tag": "Glassmorphism",
        "category_id": "comparison_contrast",
        "category": "Comparison & Contrast",
        "description": "Translucent frosted-glass overlapping circles with soft background blurs, high-contrast labels, and centralized core takeaway.",
        "prompt_template": "Create an intersecting Venn diagram infographic explaining [TOPIC]. Style: Glassmorphism—use translucent, frosted-glass overlapping circles with soft background blurs. Layout: Three intersecting circles with clear, high-contrast labels and a centralized core takeaway."
    },
    "quadrant_matrix_bauhaus": {
        "name": "The Quadrant Matrix",
        "style_tag": "Bauhaus",
        "category_id": "comparison_contrast",
        "category": "Comparison & Contrast",
        "description": "Bauhaus-inspired bold primary colors (red, blue, yellow), stark black lines, and 2x2 geometric precision grid.",
        "prompt_template": "Design a 2x2 quadrant matrix infographic mapping [TOPIC]. Style: Bauhaus-inspired—bold primary colors (red, blue, yellow), stark black lines, and geometric precision. Layout: A distinct grid with bold X and Y axis labels, placing clear icons in each of the four quadrants."
    },

    # Data & Statistics
    "corporate_dashboard_ui": {
        "name": "The Corporate Dashboard",
        "style_tag": "UI/UX App Style",
        "category_id": "data_statistics",
        "category": "Data & Statistics",
        "description": "Modern SaaS UI with clean cards, rounded corners, white/blue corporate palette, hero statistic, donut chart, and bar graph.",
        "prompt_template": "Generate a statistical dashboard infographic for [TOPIC]. Style: Modern SaaS application UI—clean cards, rounded corners, and a white/blue corporate palette. Layout: A grid system featuring a large hero statistic, a donut chart, and a bar graph, with bite-sized data labels."
    },
    "typography_stat_sheet_swiss": {
        "name": "The Large Typography Stat Sheet",
        "style_tag": "Swiss Grid",
        "category_id": "data_statistics",
        "category": "Data & Statistics",
        "description": "Strict Swiss grid alignment, high contrast black-and-white with vibrant accent color, focusing on massive bold numbers.",
        "prompt_template": "Create a data-driven infographic highlighting key metrics for [TOPIC]. Style: Swiss typography style—strict grid alignment, high contrast black-and-white with one vibrant accent color (e.g., bright red). Layout: Focus entirely on massive, bold numbers accompanied by micro-copy explanations."
    },
    "geographic_map_hologram": {
        "name": "The Geographic Map",
        "style_tag": "Futuristic Hologram",
        "category_id": "data_statistics",
        "category": "Data & Statistics",
        "description": "Futuristic sci-fi hologram with glowing wireframe maps, data nodes, deep blue/teal hues, and floating callout lines.",
        "prompt_template": "Design a map-based infographic showing the regional impact of [TOPIC]. Style: Futuristic sci-fi hologram—glowing wireframe maps, data nodes, and deep blue/teal hues. Layout: A central map with data callout lines pointing to floating text boxes."
    },
    "funnel_chart_neumorphism": {
        "name": "The Funnel/Conversion Chart",
        "style_tag": "Neumorphism",
        "category_id": "data_statistics",
        "category": "Data & Statistics",
        "description": "Soft extruded UI with subtle highlights and shadows in an inverted 4-layer pyramid with bold percentages at each stage.",
        "prompt_template": "Generate a funnel infographic for [TOPIC]. Style: Neumorphism (soft UI)—elements should look extruded from the background using subtle highlights and shadows. Layout: An inverted pyramid divided into 4 horizontal layers, with bold percentages at each stage."
    },

    # Structure & Hierarchy
    "pyramid_hierarchy_lowpoly": {
        "name": "The Pyramid/Hierarchy",
        "style_tag": "Low-Poly Art",
        "category_id": "structure_hierarchy",
        "category": "Structure & Hierarchy",
        "description": "Low-poly 3D art with geometric, faceted surfaces and sharp lighting in a triangle divided into horizontal slices.",
        "prompt_template": "Create a hierarchy pyramid infographic for [TOPIC]. Style: Low-poly 3D art—geometric, faceted surfaces with sharp lighting. Layout: A traditional triangle divided into horizontal slices, with the base representing foundations and the peak representing the ultimate goal."
    },
    "hub_and_spoke_material": {
        "name": "The Hub and Spoke",
        "style_tag": "Material Design",
        "category_id": "structure_hierarchy",
        "category": "Structure & Hierarchy",
        "description": "Google Material Design flat layers, intentional drop shadows, and 6 surrounding nodes with flat icons.",
        "prompt_template": "Design a hub-and-spoke infographic detailing the components of [TOPIC]. Style: Google Material Design—flat layers, intentional drop shadows, and bold, playful colors. Layout: A central core concept connected by solid lines to 6 surrounding nodes, each containing a flat icon and a short label."
    },
    "anatomy_exploded_blueprint": {
        "name": "The Anatomy/Exploded View",
        "style_tag": "Blueprint/Technical",
        "category_id": "structure_hierarchy",
        "category": "Structure & Hierarchy",
        "description": "Technical blueprint with navy blue background, thin white grid lines, and straight drafting lines to concise labels.",
        "prompt_template": "Generate an exploded-view anatomy infographic of [TOPIC]. Style: Technical blueprint—navy blue background, thin white grid lines, and precise drafting aesthetic. Layout: A central illustration broken into its core parts, with straight technical lines pointing to detailed, concise labels."
    },
    "mind_map_doodle": {
        "name": "The Mind Map",
        "style_tag": "Hand-Drawn/Doodle",
        "category_id": "structure_hierarchy",
        "category": "Structure & Hierarchy",
        "description": "Whiteboard sketch with marker textures, organic branching lines starting from a central bubble to sub-topics.",
        "prompt_template": "Create a mind map infographic exploring [TOPIC]. Style: Hand-drawn doodle aesthetic—looks like a high-quality whiteboard sketch with marker textures. Layout: Organic branching lines starting from a central bubble, leading to sub-topics."
    },

    # Lists & Summaries
    "checklist_synthwave": {
        "name": "The Checklist/Playbook",
        "style_tag": "Synthwave/Outrun",
        "category_id": "lists_summaries",
        "category": "Lists & Summaries",
        "description": "80s Synthwave/Outrun with chrome text, neon grids, purple/orange sunsets, and large stylized checkboxes.",
        "prompt_template": "Design a visual checklist infographic for [TOPIC]. Style: 80s Synthwave/Outrun—chrome text, neon grids, and purple/orange sunsets. Layout: A vertical list of large, stylized checkboxes with short, actionable steps next to them."
    },
    "top_10_listicle_popart": {
        "name": "The Top 10 Listicle",
        "style_tag": "Pop Art/Comic Book",
        "category_id": "lists_summaries",
        "category": "Lists & Summaries",
        "description": "Vintage comic book/Pop Art with halftone dot patterns, bold black outlines, speech bubbles, and cascading numbers (1-10).",
        "prompt_template": "Generate a numbered listicle infographic outlining the top facts about [TOPIC]. Style: Vintage comic book/Pop Art—halftone dot patterns, bold black outlines, and speech bubbles. Layout: A bold numbered sequence (1-10) cascading down the page."
    },
    "cheat_sheet_monochrome": {
        "name": "The Cheat Sheet",
        "style_tag": "Monochrome Typographic",
        "category_id": "lists_summaries",
        "category": "Lists & Summaries",
        "description": "Dense, highly organized multi-column grid relying on varying font weights, sizes, and spacing for quick reference.",
        "prompt_template": "Create a quick-reference cheat sheet infographic for [TOPIC]. Style: Monochrome typographic—relies entirely on varying font weights, sizes, and spacing rather than illustrations. Layout: A dense but highly organized multi-column grid, perfect for printing."
    },
    "problem_solution_duotone": {
        "name": "The Problem/Solution Layout",
        "style_tag": "Split-Tone Duotone",
        "category_id": "lists_summaries",
        "category": "Lists & Summaries",
        "description": "Split-tone duotone effect with two contrasting colors (e.g. magenta and cyan), split into top pain point and bottom solution.",
        "prompt_template": "Design a problem-and-solution infographic for [TOPIC]. Style: Split-tone duotone effect—using exactly two contrasting colors (e.g., deep magenta and bright cyan). Layout: A definitive horizontal split across the middle; the top half illustrates the pain point, and the bottom half provides the structured solution."
    },
}


class InfographicAIService:
    """Service to classify text and build high-fidelity infographic diffusion prompts."""

    @staticmethod
    def get_style_presets() -> Dict[str, Dict[str, str]]:
        return STYLE_PRESETS

    @staticmethod
    def get_categories() -> Dict[str, str]:
        return INFOGRAPHIC_CATEGORIES

    @staticmethod
    def auto_detect_archetype(text: str) -> str:
        """
        Heuristic / keyword-based classification of text into one of the 7 archetypes.
        """
        lower = text.lower()

        # 1. Timeline / History
        if any(w in lower for w in ["timeline", "history", "chronology", "century", "evolution of", "era", "decades"]) or re.search(r'\b(19\d\d|20\d\d)\b', text):
            return "timeline_historical"

        # 2. Step by Step / Recipe / Instructions
        if any(w in lower for w in ["step 1", "step 2", "first,", "then,", "recipe", "ingredients", "how to", "workflow", "instructions", "guide to", "preparation:"]):
            return "step_by_step"

        # 3. Flowchart / Decision / Logic
        if any(w in lower for w in ["flowchart", "decision tree", "if yes", "if no", "branches", "logic flow", "brainstorm", "mind map", "wireframe"]):
            return "flowchart_whiteboard"

        # 4. Data / Metrics / Financial
        if any(w in lower for w in ["%", "percent", "revenue", "metric", "statistics", "growth rate", "data:", "table:", "quarterly", "roi", "benchmark"]):
            return "data_visualization"

        # 5. Technical / Scientific
        if any(w in lower for w in ["architecture", "kubernetes", "pod", "server", "algorithm", "biology", "physics", "cellular", "neural network", "database", "infrastructure", "api", "protocol"]):
            return "technical_scientific"

        # 6. Playful / Viral / Menu
        if any(w in lower for w in ["funny", "humorous", "playful", "menu", "listicle", "tips for life", "hacks", "pop art", "cartoon", "fun facts"]):
            return "playful_viral"

        # 7. Default to Modular Explainer
        return "modular_explainer"

    @classmethod
    def synthesize_prompt(
        cls,
        text: str,
        archetype: str = "auto",
        user_instructions: Optional[str] = None,
        style: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Synthesizes a visual diffusion generation prompt tailored for Nano Banana Pro / Gemini
        to generate an aesthetic, legible infographic.

        Supports specific style presets as well as classic archetypes.
        Returns: (final_prompt, effective_archetype_or_style)
        """
        clean_text = " ".join(text.strip().split())[:800]
        extra_inst = f"\nCreative Instructions: {user_instructions.strip()}" if user_instructions else ""

        # Determine target style / archetype
        target_key = (style or archetype or "auto").strip().lower()

        # 1. Match against explicit style presets
        if target_key in STYLE_PRESETS:
            preset = STYLE_PRESETS[target_key]
            template = preset["prompt_template"]

            if "[TOPIC A] vs [TOPIC B]" in template:
                parts = re.split(r'\s+(?:vs\.?|versus|compared to)\s+', clean_text, maxsplit=1, flags=re.IGNORECASE)
                if len(parts) == 2:
                    prompt = template.replace("[TOPIC A]", f"'{parts[0].strip()}'").replace("[TOPIC B]", f"'{parts[1].strip()}'")
                else:
                    prompt = template.replace("[TOPIC A] vs [TOPIC B]", f"'{clean_text}'")
            else:
                prompt = template.replace("[TOPIC]", f"'{clean_text}'")

            return f"{prompt}{extra_inst}", target_key

        # 2. Check if archetype is a classic archetype
        effective_archetype = archetype.lower()
        if effective_archetype == "auto" or effective_archetype not in ARCHETYPE_DESCRIPTIONS:
            effective_archetype = cls.auto_detect_archetype(text)

        if effective_archetype == "technical_scientific":
            prompt = (
                f"A highly detailed, professional scientific and technical infographic diagram. "
                f"Clean schematic illustration with clear legible typographic labels, component callout boxes, "
                f"and cross-section details explaining the core concept: '{clean_text}'. "
                f"Crisp vectors, high-contrast blueprint or editorial tech aesthetic, dark slate and cyan accents, "
                f"accurate technical labeling, high resolution vector graphic, 8k visualization.{extra_inst}"
            )

        elif effective_archetype == "step_by_step":
            prompt = (
                f"A clean, modern step-by-step visual instructional infographic guide. "
                f"Numbered step cards (1, 2, 3...) arranged in a clear sequential workflow layout, "
                f"illustrating the process: '{clean_text}'. "
                f"Each step features a clear icon/illustration and concise readable heading, vibrant cohesive color palette, "
                f"instructional recipe or DIY manual design, beautiful modern typography, crisp 4k graphic.{extra_inst}"
            )

        elif effective_archetype == "flowchart_whiteboard":
            prompt = (
                f"An authentic, organic whiteboard sketch and flowchart diagram infographic. "
                f"Hand-drawn dry-erase marker aesthetic on a clean white surface with organic sketched connector arrows, "
                f"process boxes, dashed decision nodes, and handwritten-style readable notes illustrating: '{clean_text}'. "
                f"Creative brainstorming visual layout, subtle marker texture, engaging and clear concept map.{extra_inst}"
            )

        elif effective_archetype == "modular_explainer":
            prompt = (
                f"A sophisticated modular explainer infographic. "
                f"Central core hub with radiating connected modular cards and iconography explaining: '{clean_text}'. "
                f"Modern UI design system aesthetic, clean glassmorphism and subtle gradients, crisp readable title and section headers, "
                f"logical interconnecting data pathways, elegant corporate editorial graphic.{extra_inst}"
            )

        elif effective_archetype == "timeline_historical":
            # Modernized default timeline prompt to prevent unwanted antique scrolls/parchment looks
            prompt = (
                f"A sleek modern chronological timeline infographic chart. "
                f"A clean sequential progression track with milestone date markers, modern vector badges, and crisp typography "
                f"tracking the evolution of: '{clean_text}'. "
                f"Contemporary tech-forward editorial design, crisp vectors, clean minimalist background, no antique scrolls or parchment textures, high resolution graphic.{extra_inst}"
            )

        elif effective_archetype == "data_visualization":
            prompt = (
                f"A crisp data visualization and metric summary dashboard infographic. "
                f"Structured comparison columns, clean KPI metric cards, mini bar charts, and highlighted statistics based on: '{clean_text}'. "
                f"High-contrast financial and analytical infographic layout, legible numbers, clean grid alignment, modern Swiss graphic design.{extra_inst}"
            )

        elif effective_archetype == "playful_viral":
            prompt = (
                f"A vibrant, playful and viral illustrated listicle infographic menu. "
                f"Pop-art editorial illustration style, colorful retro-modern badges, punchy typography, and humorous/fun visual motifs "
                f"celebrating: '{clean_text}'. "
                f"Eye-catching social media infographic, whimsical character accents, bold legible headings, delightful visual hierarchy.{extra_inst}"
            )

        else:
            prompt = (
                f"A comprehensive modern infographic diagram explaining: '{clean_text}'. "
                f"Clean layout with clear visual hierarchy, legible headers, informative icons, and professional vector illustration.{extra_inst}"
            )

        return prompt, effective_archetype
