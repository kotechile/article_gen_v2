"""
Infographic AI Service for AI-driven visual diagram and infographic generation.

Supports 7 distinct infographic archetypes:
1. Technical and Scientific Diagrams
2. Step-by-Step Guides and Recipes
3. Flowcharts and Whiteboard Sketches
4. Modular Explainers
5. Timelines and Historical Overviews
6. Data Visualizations
7. Playful and Viral Menus/Listicles
Plus automatic archetype detection based on text content.
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
    "timeline_historical": "Timelines and Historical Overviews: Sequential charts tracking events or sports histories with precise data rendering.",
    "data_visualization": "Data Visualizations: Visuals transformed from raw metrics or CSV data into structured layouts like scrum boards or financial summaries.",
    "playful_viral": "Playful and Viral Menus/Listicles: Lighthearted, illustrated menus, humorous life steps, or pop-art graphics."
}


class InfographicAIService:
    """Service to classify text and build high-fidelity infographic diffusion prompts."""

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
        user_instructions: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Synthesizes a visual diffusion generation prompt tailored for Nano Banana Pro / Gemini
        to generate an aesthetic, legible infographic.

        Returns: (final_prompt, effective_archetype)
        """
        effective_archetype = archetype.lower()
        if effective_archetype == "auto" or effective_archetype not in ARCHETYPE_DESCRIPTIONS:
            effective_archetype = cls.auto_detect_archetype(text)

        clean_text = " ".join(text.strip().split())[:800]
        extra_inst = f"\nCreative Instructions: {user_instructions.strip()}" if user_instructions else ""

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
            prompt = (
                f"A chronological historical timeline infographic chart. "
                f"A sequential timeline path with milestone date markers, historical event badges, and illustrative vignettes "
                f"tracking the evolution and history of: '{clean_text}'. "
                f"Precise chronological layout, elegant typography, museum-grade editorial infographic design, rich visual storytelling.{extra_inst}"
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
