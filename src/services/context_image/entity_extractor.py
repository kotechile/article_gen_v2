"""
Entity Extraction & Prompt Synthesizer for Context-Aware Image Generation.

Art-direction first, generation second:
Analyzes article text excerpts (grounded in the article's core thesis, vertical, and main topic)
to commission high-impact editorial imagery following an executive publication standard.
"""

import json
import logging
import re
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List

from supabase_client import resolve_llm_provider

logger = logging.getLogger(__name__)

# What is appended to image models to strictly forbid in-image typography/rubble
IMAGE_GUARD = "Do not include: text, lettering, numbers, wordmarks, watermarks, UI."

CLICHE_BAN = (
    "light bulb", "handshake", "chess", "puzzle piece", "glowing brain", "rocket",
    "dartboard", "scales of justice", "gavel", "thumbs up"
)

# Editorial treatment catalogue
EDITORIAL_STYLES: Dict[str, Dict[str, Any]] = {
    "cinematic_still": {
        "id": "cinematic_still",
        "label": "Cinematic still",
        "when": "the article describes one decisive moment or place — a yard at dawn, a control room, a shutdown line, a handover — that a film still could hold",
        "medium": "cinematic film still, anamorphic 35mm look, wide establishing composition",
        "craft": "single strong practical light source, crisp atmospheric depth, restrained teal-and-amber palette, no people facing camera, motion implied rather than shown",
        "keywords": ("cinematic", "film still", "anamorphic", "35mm", "establishing"),
        "model_family": "photographic"
    },
    "editorial_macro": {
        "id": "editorial_macro",
        "label": "Editorial macro",
        "when": "the story turns on one physical thing — a part, a material, a component, a document — and what that thing costs, contains or crosses a border is the news",
        "medium": "extreme close-up macro photograph, 100mm macro lens, one razor-sharp focal plane",
        "craft": "shallow depth of field, visible surface texture and dust, soft directional daylight, hero object off-centre on the thirds with the background falling away",
        "keywords": ("macro", "close-up", "close up", "depth of field", "extreme close"),
        "model_family": "photographic"
    },
    "document_flatlay": {
        "id": "document_flatlay",
        "label": "Document still life",
        "when": "the story is regulatory or contractual — a filing, a mandate, a rate notice, a certificate, a purchase order that changed the economics",
        "medium": "overhead flat-lay photograph of paper documents on a plain desk surface",
        "craft": "top-down 90-degree view, even diffused daylight, one object slightly out of alignment to look handled, blank or illegibly cropped paper, muted paper tones",
        "keywords": ("flat lay", "flat-lay", "overhead", "top-down", "top down", "desk"),
        "model_family": "photographic"
    },
    "clay_render": {
        "id": "clay_render",
        "label": "Matte 3D render",
        "when": "the news is structural and abstract — a stack reordered, a layer added, a flow rerouted — and there is no literal object that carries it, so the idea is stated as a small physical assembly of recognisable parts rather than as bare shapes",
        "medium": "matte clay 3D render of a small mechanical assembly resting on a real textured surface in a real space",
        "craft": "three or four recognisable engineered parts — a modular block, a housing, a latched cover, a connector or a bay — in a clear physical arrangement that states the idea; matte surfaces with moulding seams and contact shadows resting on weathered concrete or brushed steel, one directional key light raking across them, matte muted palette, no plain spheres or wedges standing in for the subject, no text",
        "keywords": ("3d render", "clay", "matte", "studio render", "component", "modular", "assembly", "housing", "bay", "3d"),
        "model_family": "constructed"
    },
    "technical_isometric": {
        "id": "technical_isometric",
        "label": "Technical isometric cutaway",
        "when": "the article explains how a system, process or stack actually works — money flows, supply chains, pipelines, an agent assembly line",
        "medium": "clean isometric cutaway illustration, technical drawing style, axonometric projection",
        "craft": "flat muted palette with one accent colour, thin consistent line weight, laid over a real material surface — a workbench, a plant floor, a drafting table — recognisable hardware with racks, modules, trays, connectors, pipes with visible depth rather than abstract boxes, one directional light, unlabelled, generous empty margin",
        "keywords": ("isometric", "axonometric", "cutaway", "cut-away", "cross-section", "schematic"),
        "model_family": "constructed"
    },
    "component_assembly": {
        "id": "component_assembly",
        "label": "Modular component assembly",
        "when": "the story is a single number, rule, gate or shift and restraint is the point — with no scene to photograph, the frame must still be a real assembly: a modular bay, an unlatched inspection gate, a rack of blades, an interlocking connector",
        "medium": "minimalist studio composition of a modular mechanical assembly on a real surface, one directional light, generous negative space",
        "craft": "two or three recognisable engineered parts (a module, a latch, a rack rail, a bay cover, an inspection gate) in one deliberate arrangement that states the idea; hard clean edges on weathered concrete or brushed steel, one directional light throwing a long cast shadow, matte muted palette, never bare shapes, no text",
        "keywords": ("modular", "module", "component", "assembly", "connector", "rack", "chassis", "bay", "bracket", "latch"),
        "model_family": "constructed"
    },
    "paper_collage": {
        "id": "paper_collage",
        "label": "Editorial paper collage",
        "when": "the article is a purely conceptual synthesis or policy dilemma where no physical facility, machine, or supply chain exists, and an elegant abstract paper silhouette states the idea",
        "medium": "minimalist editorial cut-paper collage, crisp cut-out object silhouettes, halftone newsprint texture",
        "craft": "two or three stylized cut-paper object silhouettes (such as an hourglass, certificate, key, or mechanism) layered deliberately on a real table — hand-torn rag paper with visible fibre, physical cast shadows under each layer, one raking light; muted modern editorial palette, crisp clean edges, generous negative space, never bare geometry or random torn scraps, no legible print",
        "keywords": ("collage", "cut-paper", "cut paper", "silhouette", "cut-out", "cutout", "torn", "halftone", "newsprint"),
        "model_family": "constructed"
    },
    "long_lens_industry": {
        "id": "long_lens_industry",
        "label": "Compressed telephoto industry",
        "when": "scale is the story — a port, a refinery, a data centre hall, a rail yard, a skyline of cranes — and the reader needs to feel how big it is",
        "medium": "telephoto compression, 200mm long-lens view of industrial infrastructure",
        "craft": "stacked overlapping layers of structure with crystal-clear telephoto distance clarity, flat compressed perspective, sharp directional lighting and deep industrial contrast, no people in foreground, no fog or haze",
        "keywords": ("telephoto", "long lens", "long-lens", "compressed", "200mm"),
        "model_family": "photographic"
    },
    "studio_object": {
        "id": "studio_object",
        "label": "Studio product shot",
        "when": "the story is a product, a device, a price or a market for a thing the reader could buy — an appliance, a panel, a router, a robot arm",
        "medium": "studio product photograph of one hero object on a real studio surface, single directional light",
        "craft": "softbox key light with visible falloff and a long cast contact shadow across a textured surface, three-quarter angle, authentic product design and silhouette, catalogue clarity, no gradient sweep, no text",
        "keywords": ("studio", "studio surface", "product shot", "softbox", "three-quarter"),
        "model_family": "photographic"
    },
    "architectural_night": {
        "id": "architectural_night",
        "label": "Lit architecture at dusk",
        "when": "the story is about change after hours — automation displacing shifts, a market open all night, capacity running while people sleep",
        "medium": "architectural photograph of a modern building or plant at blue hour",
        "craft": "lit windows as the only warm light, long-exposure calm with no moving figures, deep blue ambient light, clean geometry, small human scale implied by a doorway",
        "keywords": ("blue hour", "dusk", "night", "lit windows", "long exposure", "architectural"),
        "model_family": "photographic"
    }
}


@dataclass
class EntityExtractionResult:
    has_physical_entity: bool
    main_object: str
    search_query: str
    generation_prompt: str
    object_fidelity_weight: float = 0.75
    entity_type: str = "physical"  # "physical" or "metaphorical"
    is_metaphorical: bool = False
    core_thesis: str = ""
    core_conflict: str = ""
    hero_subject: str = ""
    composition: str = ""
    style_id: str = ""
    style_label: str = ""
    alt_text: str = ""
    caption: str = ""
    title: str = ""
    negative_prompt: str = ""
    raw_response: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


SYSTEM_PROMPT = """You are an expert AI Art Director on an executive industry-analysis editorial desk. \
Your task is to commission a featured editorial image for an article based on the selected text excerpt and article context.

CREATIVE PROCESS (MAGAZINE COVER STANDARD: CONCEPTUAL, ARRESTING, NON-LITERAL):
You are an Art Director, NOT a generic search engine or stock photo generator.
The featured image is the front door to the article — it must make the reader pause, think, and want to read.

1. ANCHOR ON THE CORE TOPIC & GOVERNING CONFLICT:
   - THE HERO SUBJECT MUST DIRECTLY EMBODY THE STORY'S CORE PROTAGONIST, MACHINE, OR SYSTEMIC PHENOMENON:
     * If the headline or text centers on an electric vehicle (EV), car, cargo vessel, intermodal gantry, industrial turbine, or robotic arm: THAT MACHINE MUST BE THE HERO IN THE FRAME! Never hide or omit the vehicle/machine.
     * PRESERVE NAMED BRANDS AND MODELS: If the text or context explicitly mentions a real-world brand, model, automaker, or machine (e.g. Tesla Model Y, Rivian R1T, Porsche Taycan, Apple Watch, Nvidia H100, Boeing 787, Siemens turbine): ALWAYS EXPLICITLY NAME AND PRESERVE THE EXACT BRAND AND MODEL in the hero_subject, main_object, search_query, and generation_prompt! NEVER genericize them into 'an unbranded vehicle', 'a generic car', or 'an unnamed device'. Diffusion models (Flux, Midjourney, Imagen) are trained on authentic automotive and industrial designs and produce vastly superior, realistic results when given the exact make and model.
     * Capture the core theme, mood, and tension through dramatic environmental storytelling and conflict (e.g. a Tesla Model Y powering a dark home during a neighborhood blackout outage, or an industrial gantry moving cargo containers under floodlights), NOT boring stock photos.
     * BEWARE THE PERIPHERAL ANECDOTE TRAP: Articles frequently use minor examples, supporting anecdotes, or incidental props (e.g. a screw, a delivery van, a specific chip model, a pallet of scrap, a coffee cup, packaging tape). NEVER elevate an incidental anecdote into the hero subject!
     * BEWARE INVERTING THE PROTAGONIST: Do not swap the primary actor for a background utility box (e.g., swapping an EV for an empty meter box).
     * Identify the central dramatic tension or trade-off (e.g. rising capital expenditure vs automation payoff, write-path persistence vs token bandwidth, scale bottlenecks).

2. SELECT AN EVOCATIVE HERO OBJECT OR SCENE (CONCEPTUAL & SYMBOLIC STORYTELLING):
   - Ground the visual conceptually in the article's substantive vertical:
     * For enterprise AI / software / compute / multi-agent systems: Do NOT default to generic datacenters, blue circuit traces, or server racks as a lazy shortcut. NEVER translate software or AI into literal factory plumbing or pneumatic valves! Use conceptual, symbolic visual storytelling: optical beam-splitter prisms dividing a beam of warm light across dark obsidian stone, monolithic stone slabs in equilibrium, razor-thin blades of golden light, high-precision axonometric technical cutaways of coordinated processing modules. If specific chips or architectures are named (e.g. Nvidia Blackwell, Nvidia H100, Apple M-series), ground the design in that authentic hardware.
     * For supply chain / logistics / warehousing / freight: An evocative scene capturing balance, capacity, or flow under hard raking sunlight or twilight gantries.
     * For energy / utilities / infrastructure: Transformer substations, utility-scale battery banks, industrial copper busbars, or wind/solar installations under dramatic skies.
     * For heavy industry / manufacturing: Precision CNC machining spindles, induction heating coils, robotic welding arms, or PCB assembly benches.
     * For finance / tax / governance: Forensic audit desks with leather ledgers under focused lamps, embossed legal documents, brass balance scales, vintage vault doors.
     * For residential / home resilience / V2H: The named electric vehicle (e.g. Tesla Model Y, Ford F-150 Lightning, Rivian R1T) or home battery setup powering a residential home at dusk with warm glowing windows during a blackout.
   - EVERY scene must have physical presence, tactile context, material weight, and dramatic lighting.
   - A systemic, software or operational story is never a product shot on an empty sweep; put the hero in an evocative environment or constructed schematic.

3. WRITE IN THE IDIOM OF THE MODEL:
   - For photographic treatments: stage a PHOTOGRAPH — name the optic and falloff (35mm anamorphic wide, 100mm macro, 200mm telephoto), the room lighting, the surface texture, and real environment.
   - For constructed/schematic treatments: build a STRUCTURE — name the parts, arrangement, materials, and directional light.

4. CRAFT A CINEMATIC, HIGH-TEXTURE GENERATION PROMPT:
   - 15-120 words with rich sensory and visual details.
   - Landscape composition (16:9): wide framing with generous editorial negative space and deliberate breathing room, an ASYMMETRIC composition where the hero commands the frame off-centre (like magazine cover art, strictly without text or typography).
   - Lighting & atmosphere: chiaroscuro, single practical directional light, low-raking golden-hour sun, blue-hour twilight, deep shadows, rim highlights, crisp crystalline air.
   - Textures & materials: brushed metals, weathered concrete, copper, steel, matte polymers, tactile paper stock, polished basalt, optical glass.

DOMAIN GROUNDING & HARD RULES:
1. NEVER DEPICT A GENERIC OFFICE WORKER AT A DESK: No stock photos of a person typing at a laptop, sitting at an office desk, or in a conference room. No people facing camera.
2. NO BARE SHAPES OR ABSTRACT SCRAPS: Cubes, spheres, wedges, slabs, rectangles, and blobs are not subjects. Ground the subject in a tangible mechanism or symbolic object.
3. MATURE PROFESSIONAL B2B GROUNDING: Never depict children's toys, clip-art silhouettes, or playful cartoon nursery symbols.
4. UNIVERSAL DOMAIN SEMIOTICS: No obscure micro-metaphors. Never cross-contaminate unrelated domains (e.g. no freight trucks for pure AI algorithms, and never factory plumbing for software pipelines).
5. BRAND & PRODUCT INTEGRITY (ALWAYS PRESERVE NAMED BRANDS & MODELS):
   - When the article text, headline, or context names a specific real-world brand, model, automaker, tech device, or machine (e.g., Tesla Model Y, Apple Watch, Rivian R1T, Nvidia H100, Porsche Taycan, Boeing 787):
     * ALWAYS explicitly name the brand and exact model in the hero_subject, main_object, search_query, and generation_prompt!
     * NEVER censor or genericize named products into "an unbranded electric car", "a generic vehicle", "an unbranded gadget", or "an unnamed device".
     * Reflect authentic real-world design, body lines, and industrial proportions. (Do not request zoomed-in macro text decals or typography wordmarks, but preserve the full branded identity and silhouette of the vehicle or machine).
   - Only use generic descriptions if no specific brand or model is mentioned anywhere in the text or context.
6. CLICHE BAN: Strictly no light bulbs, handshakes, chess pieces, puzzle pieces, glowing brains, rocket ships, dartboards, scales of justice, gavels, or thumbs up.
7. STRICTLY NO TEXT, LETTERS, NUMBERS, OR UI: Generated lettering is unreadable rubble. Forbid them in negative_prompt.
8. ALT TEXT STANDARDS: <=125 characters, starting with the SUBJECT itself (e.g. "Tesla Model Y in driveway at dusk...") — never 'image of' or the medium.

Respond ONLY with a valid JSON object adhering strictly to this schema:
{
  "has_physical_entity": true/false,
  "entity_type": "physical" or "metaphorical",
  "is_metaphorical": true/false,
  "core_thesis": "1 sentence: the central assertion and insight communicating the main topic",
  "core_conflict": "the governing systemic, physical, or economic tension",
  "hero_subject": "the physical thing or evocative protagonist in the frame — never an incidental anecdote, never a bare shape",
  "main_object": "Specific physical entity or metaphorical scene description",
  "composition": "the framing rule (>=12 chars) — e.g. 'extreme asymmetry, hero off-centre with dramatic scale contrast'",
  "style_id": "suggested treatment id (e.g. cinematic_still, editorial_macro, clay_render, technical_isometric, component_assembly, paper_collage, long_lens_industry, studio_object, architectural_night)",
  "style_label": "Human readable treatment name",
  "search_query": "2 to 4 words for physical search, or empty string if metaphorical",
  "generation_prompt": "15 to 120 words cinematic diffusion prompt in the idiom of the model",
  "negative_prompt": "text, lettering, numbers, words, typography, watermark, logo, UI, low resolution, blur, deformed anatomy",
  "alt_text": "<=125 chars, the SUBJECT first — never 'image of' or the medium",
  "caption": "one sentence ending in a full stop.",
  "title": "3-100 chars, media title",
  "object_fidelity_weight": 0.75
}
"""


class EntityExtractor:
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None, api_key: Optional[str] = None):
        self.provider = provider
        self.model = model
        self.api_key = api_key

    def _get_llm_config(self) -> Dict[str, Any]:
        if self.provider and self.api_key:
            return {
                "provider": self.provider,
                "model": self.model,
                "api_key": self.api_key
            }
        resolved = resolve_llm_provider(task_role="article_generation")
        return {
            "provider": resolved.get("provider") or "gemini",
            "model": resolved.get("model") or "gemini-2.5-flash",
            "api_key": resolved.get("api_key")
        }

    def extract(
        self,
        text: str,
        user_instructions: Optional[str] = None,
        article_title: Optional[str] = None,
        article_context: Optional[Dict[str, Any]] = None,
        style_id: Optional[str] = None,
        style_name: Optional[str] = None,
        style_prompt_modifier: Optional[str] = None
    ) -> EntityExtractionResult:
        """
        Analyze article excerpt and extract entity, search query, and generation prompt
        guided by editorial art-direction standards.
        """
        if not text or not text.strip():
            return EntityExtractionResult(
                has_physical_entity=False,
                main_object="",
                search_query="",
                generation_prompt="",
                object_fidelity_weight=0.75
            )

        config = self._get_llm_config()
        prompt_content = self._build_prompt_content(
            text=text,
            user_instructions=user_instructions,
            article_title=article_title,
            article_context=article_context,
            style_id=style_id,
            style_name=style_name,
            style_prompt_modifier=style_prompt_modifier
        )

        raw_text = self._call_llm(config, prompt_content)
        return self._parse_json_response(raw_text, text, style_id=style_id)

    def _build_prompt_content(
        self,
        text: str,
        user_instructions: Optional[str] = None,
        article_title: Optional[str] = None,
        article_context: Optional[Dict[str, Any]] = None,
        style_id: Optional[str] = None,
        style_name: Optional[str] = None,
        style_prompt_modifier: Optional[str] = None
    ) -> str:
        ctx = article_context or {}
        title = article_title or ctx.get("title") or ""
        vertical = ctx.get("vertical") or ctx.get("topic") or ""
        thesis = ctx.get("thesis") or ctx.get("one_big_thing") or ""
        hook = ctx.get("hook") or ctx.get("deck") or ctx.get("excerpt") or ""

        parts = []

        # Context header
        if title or vertical or thesis:
            context_block = ["=== ARTICLE METADATA ==="]
            if title:
                context_block.append(f"Headline: {title}")
            if vertical:
                context_block.append(f"Vertical / Niche: {vertical}")
            if thesis:
                context_block.append(f"Core Thesis / One Big Thing: {thesis}")
            if hook:
                context_block.append(f"Summary / Excerpt: {hook}")
            parts.append("\n".join(context_block))

        # Selected text
        parts.append(f"=== SELECTED TEXT FROM ARTICLE ===\n\"\"\"\n{text.strip()}\n\"\"\"")

        # Style guidance
        style_key = style_id or ""
        if not style_key and style_name:
            # Try to match by name
            for sid, sdata in EDITORIAL_STYLES.items():
                if sid.lower() in style_name.lower() or sdata["label"].lower() in style_name.lower():
                    style_key = sid
                    break

        if style_key in EDITORIAL_STYLES:
            sdata = EDITORIAL_STYLES[style_key]
            parts.append(
                f"=== REQUESTED EDITORIAL TREATMENT ===\n"
                f"Style: {sdata['label']} ({sdata['id']})\n"
                f"When to use: {sdata['when']}\n"
                f"Medium: {sdata['medium']}\n"
                f"Craft: {sdata['craft']}\n"
                f"Model Family: {sdata['model_family']}"
            )
        elif style_name or style_prompt_modifier:
            parts.append(
                f"=== REQUESTED VISUAL STYLE ===\n"
                f"Style Name: {style_name or 'Custom'}\n"
                f"Style Characteristics: {style_prompt_modifier or ''}"
            )

        if user_instructions:
            parts.append(f"=== USER CREATIVE INSTRUCTIONS ===\n{user_instructions.strip()}")

        return "\n\n".join(parts)

    def _call_llm(self, config: Dict[str, Any], prompt_content: str) -> str:
        provider = (config.get("provider") or "gemini").lower()
        model = config.get("model") or "gemini-2.5-flash"
        api_key = config.get("api_key")

        # Try LiteLLM first if available
        try:
            from litellm import completion
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_content}
            ]
            litellm_model = model
            completion_kwargs: Dict[str, Any] = {
                "messages": messages,
                "api_key": api_key,
                "temperature": 0.3,
                "max_tokens": 800
            }

            if provider in ["google", "gemini"]:
                if not model.startswith("gemini/"):
                    litellm_model = f"gemini/{model}"
            elif provider == "openai":
                if not model.startswith("openai/"):
                    litellm_model = f"openai/{model}"
            elif provider == "deepseek":
                clean_model = model.replace("deepseek/", "").replace("openai/", "")
                api_model = "deepseek-chat" if clean_model in ["deepseek-v4-flash", "deepseek-v4-pro", "default"] or "flash" in clean_model else clean_model
                litellm_model = f"openai/{api_model}"
                completion_kwargs["api_base"] = "https://api.deepseek.com"
                completion_kwargs["custom_llm_provider"] = "openai"
            elif provider in ["anthropic", "claude"]:
                if not model.startswith("anthropic/"):
                    litellm_model = f"anthropic/{model}"
            elif provider in ["kimi", "moonshot"]:
                clean_model = model.replace("moonshot/", "").replace("kimi/", "")
                litellm_model = f"openai/{clean_model}"
                completion_kwargs["api_base"] = "https://api.moonshot.cn/v1"
                completion_kwargs["custom_llm_provider"] = "openai"
            else:
                if "/" not in model:
                    litellm_model = f"{provider}/{model}"

            completion_kwargs["model"] = litellm_model

            response = completion(**completion_kwargs)
            content = response.choices[0].message.content
            logger.info(f"LiteLLM entity extraction output: {content[:300]}")
            return content
        except Exception as e:
            logger.warning(f"LiteLLM completion failed in entity extractor: {e}. Trying direct HTTP fallback.")

        # Direct HTTP fallback for Gemini / DeepSeek / OpenAI
        if "gemini" in provider or "google" in provider:
            return self._call_gemini_direct(api_key, model, prompt_content)
        elif "deepseek" in provider:
            return self._call_deepseek_direct(api_key, model, prompt_content)
        elif "openai" in provider or "kimi" in provider or "moonshot" in provider:
            base_url = "https://api.moonshot.cn/v1/chat/completions" if "kimi" in provider or "moonshot" in provider else "https://api.openai.com/v1/chat/completions"
            return self._call_openai_direct(api_key, model, prompt_content, base_url=base_url)

        raise RuntimeError(f"Unable to invoke LLM provider '{provider}' for entity extraction.")

    def _call_deepseek_direct(self, api_key: str, model: str, prompt_content: str) -> str:
        import requests
        clean_model = model.replace("deepseek/", "").replace("openai/", "")
        api_model = "deepseek-chat" if clean_model in ["deepseek-v4-flash", "deepseek-v4-pro", "default"] or "flash" in clean_model else clean_model
        url = "https://api.deepseek.com/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": api_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_content}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2
        }
        res = requests.post(url, headers=headers, json=payload, timeout=25)
        res.raise_for_status()
        data = res.json()
        content = data["choices"][0]["message"]["content"]
        logger.info(f"Direct DeepSeek entity extraction output: {content[:300]}")
        return content

    def _call_gemini_direct(self, api_key: str, model: str, prompt_content: str) -> str:
        import requests
        clean_model = model.replace("gemini/", "")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model}:generateContent?key={api_key}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": f"{SYSTEM_PROMPT}\n\n{prompt_content}"}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 800,
                "responseMimeType": "application/json"
            }
        }
        res = requests.post(url, json=payload, timeout=20)
        res.raise_for_status()
        data = res.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

    def _call_openai_direct(self, api_key: str, model: str, prompt_content: str, base_url: str = "https://api.openai.com/v1/chat/completions") -> str:
        import requests
        clean_model = model.replace("openai/", "").replace("kimi/", "").replace("moonshot/", "")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": clean_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_content}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2
        }
        res = requests.post(base_url, headers=headers, json=payload, timeout=25)
        res.raise_for_status()
        data = res.json()
        return data["choices"][0]["message"]["content"]

    def _parse_json_response(self, raw_text: str, fallback_text: str, style_id: Optional[str] = None) -> EntityExtractionResult:
        try:
            logger.info(f"Raw LLM output in _parse_json_response: {raw_text[:300]}")
            # 1. Strip think tags emitted by reasoning models
            cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()

            # 2. Extract JSON object
            json_match = re.search(r"(\{[\s\S]*\})", cleaned)
            if json_match:
                cleaned = json_match.group(1).strip()
            elif cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\n", "", cleaned)
                cleaned = re.sub(r"\n```$", "", cleaned).strip()

            data = json.loads(cleaned)

            hero_sub = str(data.get("hero_subject") or data.get("main_object") or "").strip()
            main_obj = str(data.get("main_object") or hero_sub).strip()
            search_q = str(data.get("search_query") or "").strip()
            gen_prompt = str(data.get("generation_prompt") or "").strip()
            core_thesis = str(data.get("core_thesis") or "").strip()
            core_conflict = str(data.get("core_conflict") or "").strip()
            composition = str(data.get("composition") or "").strip()
            resolved_style_id = str(data.get("style_id") or style_id or "").strip()
            style_label = str(data.get("style_label") or (EDITORIAL_STYLES.get(resolved_style_id, {}).get("label") if resolved_style_id else "")).strip()

            # Alt text sanitation: <= 125 chars, start with subject, avoid "image of..."
            alt_raw = str(data.get("alt_text") or "").strip()
            alt_text = self._sanitize_alt_text(alt_raw or hero_sub or main_obj)

            caption = str(data.get("caption") or "").strip()
            if caption and not caption.endswith((".", "!", "?")):
                caption += "."

            title = str(data.get("title") or hero_sub[:50] or main_obj[:50]).strip()
            negative_prompt = str(data.get("negative_prompt") or (
                "text, lettering, numbers, words, typography, watermark, logo, UI, low resolution, blur, deformed anatomy"
            )).strip()

            if not main_obj and not hero_sub:
                raise ValueError("Parsed JSON missing 'main_object' and 'hero_subject'")

            has_physical = bool(data.get("has_physical_entity", True))
            raw_type = str(data.get("entity_type") or "").strip().lower()
            is_metaphorical = bool(data.get("is_metaphorical", raw_type == "metaphorical" or not has_physical))
            entity_type = "metaphorical" if is_metaphorical else "physical"

            # Check if generation prompt needs IMAGE_GUARD
            if gen_prompt and "text" not in gen_prompt.lower():
                gen_prompt = f"{gen_prompt}. {IMAGE_GUARD}"

            return EntityExtractionResult(
                has_physical_entity=not is_metaphorical and has_physical,
                entity_type=entity_type,
                is_metaphorical=is_metaphorical,
                main_object=main_obj,
                hero_subject=hero_sub or main_obj,
                core_thesis=core_thesis,
                core_conflict=core_conflict,
                composition=composition,
                style_id=resolved_style_id,
                style_label=style_label,
                alt_text=alt_text,
                caption=caption,
                title=title,
                negative_prompt=negative_prompt,
                search_query=search_q or (f"{hero_sub or main_obj} photo" if has_physical else ""),
                generation_prompt=gen_prompt or f"A cinematic 35mm editorial photograph of {hero_sub or main_obj}. {IMAGE_GUARD}",
                object_fidelity_weight=float(data.get("object_fidelity_weight", 0.60 if is_metaphorical else 0.75)),
                raw_response=data
            )
        except Exception as e:
            logger.error(f"Error parsing entity extractor JSON: {e}, raw text: {raw_text[:300]}")
            # Intelligent fallback heuristic
            lower_text = fallback_text.lower()
            subject = None
            query = None

            if "plumber" in lower_text:
                subject = "Plumber repairing leaking copper pipe in crawlspace"
                query = "plumber fixing pipe wrench"
            elif "electrician" in lower_text:
                subject = "Electrician inspecting electrical breaker panel"
                query = "electrician wiring tools"
            elif "boston dynamics" in lower_text or "atlas" in lower_text or "humanoid robot" in lower_text or "robot" in lower_text:
                subject = "Humanoid robot performing everyday physical tasks"
                query = "humanoid robot everyday tasks"
            elif "mechanic" in lower_text:
                subject = "Auto mechanic repairing car engine"
                query = "auto mechanic workshop"
            elif "carpenter" in lower_text:
                subject = "Carpenter cutting wood in woodworking shop"
                query = "carpenter workshop tools"
            elif "surgeon" in lower_text or "doctor" in lower_text:
                subject = "Surgeon performing surgery in modern operating room"
                query = "surgeon operating room"

            if not subject:
                sentences = [s.strip() for s in re.split(r'[.!?\n]', fallback_text) if s.strip()]
                candidate_sentence = sentences[0] if sentences else fallback_text
                words = [w for w in candidate_sentence.split() if len(w) > 3 and w.isalpha()]
                headline = " ".join(words[:3]) if words else "hands-on physical trade"
                subject = f"{headline.capitalize()} craftsmanship"
                query = f"{headline} craftsmanship photo"

            alt_clean = self._sanitize_alt_text(subject)

            return EntityExtractionResult(
                has_physical_entity=True,
                entity_type="physical",
                is_metaphorical=False,
                main_object=subject,
                hero_subject=subject,
                core_thesis=f"Substantive real-world craftsmanship: {subject}",
                core_conflict="Precision craftsmanship under operational load",
                composition="Asymmetric composition with dramatic practical directional lighting",
                style_id=style_id or "cinematic_still",
                style_label=EDITORIAL_STYLES.get(style_id or "cinematic_still", {}).get("label", "Cinematic still"),
                alt_text=alt_clean,
                caption=f"{subject} captured in authentic work environment.",
                title=subject[:60],
                negative_prompt="text, lettering, numbers, words, typography, watermark, logo, UI",
                search_query=query,
                generation_prompt=f"A cinematic 35mm editorial photograph of {subject.lower()}, authentic work environment, natural practical lighting, rich textures. {IMAGE_GUARD}",
                object_fidelity_weight=0.75
            )

    @staticmethod
    def _sanitize_alt_text(alt: str) -> str:
        """Ensure alt text describes the subject first, <= 125 chars, no 'image of' prefix."""
        cleaned = re.sub(r"^.{0,45}?\b(image|picture|photo|photograph|illustration|graphic|render)\b\s+(of|showing|depicting)\b\s*", "", alt, flags=re.I)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if len(cleaned) > 125:
            cleaned = cleaned[:122] + "..."
        return cleaned or "Editorial illustration"
