"""
Entity Extraction & Prompt Synthesizer for Context-Aware Image Generation.

Parses article sections to identify specific physical entities (e.g. electronics,
hardware, gadgets, vehicles, apparel), constructs high-precision reference search
queries, and synthesizes rich diffusion generation prompts.
"""

import json
import logging
import re
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

from supabase_client import resolve_llm_provider

logger = logging.getLogger(__name__)


@dataclass
class EntityExtractionResult:
    has_physical_entity: bool
    main_object: str
    search_query: str
    generation_prompt: str
    object_fidelity_weight: float = 0.75
    entity_type: str = "physical"  # "physical" or "metaphorical"
    is_metaphorical: bool = False
    raw_response: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


SYSTEM_PROMPT = """You are an expert AI visual art director and creative editorial photographer.
Your task is to analyze an excerpt of article text and identify or conceive the most impactful visual subject:

1. Physical Entity Detection:
   - Check if there is a dominant, specific physical object/entity mentioned (e.g., specific electronics, vehicle, wearable, hardware, tool, product, tangible gear, architecture, animal).
   - If found, extract this exact entity name and set entity_type to "physical" and is_metaphorical to false.

2. Metaphorical / Symbolic Fallback (When No Dominant Physical Object Exists):
   - If NO dominant physical entity is present (e.g., abstract or conceptual text regarding inflation, interest rates, data security, mental resilience, cloud computing, leadership, strategy):
   - DO NOT leave the subject empty or generic. Instead, devise a compelling, tangible METAPHORICAL object or symbolic scene that concretely visualizes the concept.
   - Examples:
     * Inflation / Purchasing Power -> "A melting gold sovereign coin resting on a ledger" or "An antique brass balancing scale weighing paper banknotes against a single grain of wheat"
     * Cybersecurity / Data Privacy -> "A heavy glowing neon titanium padlock locking a transparent glass sphere filled with optical fiber cables"
     * Innovation / Growth -> "A delicate green sprout breaking through a cracked slab of polished black marble"
     * Team collaboration / Synchrony -> "A pair of rowing oars striking tranquil misty water in unison"
     * Cloud Computing -> "A miniature glowing crystalline server tower floating inside an antique glass terrarium"
   - Set entity_type to "metaphorical", is_metaphorical to true, and has_physical_entity to false.

3. High-Precision Search Query:
   - Formulate a web search query designed to retrieve clean, high-resolution photography of this physical object or metaphorical scene (e.g., studio photography, clean background, editorial lighting).

4. Generation Prompt:
   - Formulate a rich, cinematic diffusion generation prompt that places the physical or metaphorical object into an engaging, realistic composition with vivid lighting, atmosphere, and 35mm editorial photography aesthetics.

5. Object Fidelity Weight:
   - Between 0.0 and 1.0 (recommended 0.75 for physical branded products, 0.60 for metaphorical objects to allow creative artistic freedom).

Respond ONLY with a valid JSON object adhering strictly to this schema:
{
  "has_physical_entity": true/false,
  "entity_type": "physical" or "metaphorical",
  "is_metaphorical": true/false,
  "main_object": "Name of physical entity or metaphorical object",
  "search_query": "Target object clean photo studio lighting high resolution",
  "generation_prompt": "Cinematic description of the object or metaphorical scene, composition, lighting, 35mm photography",
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

    def extract(self, text: str, user_instructions: Optional[str] = None) -> EntityExtractionResult:
        """
        Analyze article excerpt and extract entity, search query, and generation prompt.
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
        prompt_content = f"Article Excerpt:\n\"\"\"\n{text.strip()}\n\"\"\""
        if user_instructions:
            prompt_content += f"\n\nUser Creative Instructions:\n{user_instructions.strip()}"

        raw_text = self._call_llm(config, prompt_content)
        return self._parse_json_response(raw_text, text)

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
                "max_tokens": 600
            }

            if provider in ["google", "gemini"]:
                if not model.startswith("gemini/"):
                    litellm_model = f"gemini/{model}"
            elif provider == "openai":
                if not model.startswith("openai/"):
                    litellm_model = f"openai/{model}"
            elif provider == "deepseek":
                clean_model = model.replace("deepseek/", "").replace("openai/", "")
                # Using openai/ prefix with deepseek api_base ensures full compatibility across LiteLLM versions
                litellm_model = f"openai/{clean_model}"
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
            return response.choices[0].message.content
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
        url = "https://api.deepseek.com/chat/completions"
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
        res = requests.post(url, headers=headers, json=payload, timeout=25)
        res.raise_for_status()
        data = res.json()
        return data["choices"][0]["message"]["content"]

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
                "maxOutputTokens": 600,
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

    def _parse_json_response(self, raw_text: str, fallback_text: str) -> EntityExtractionResult:
        try:
            cleaned = raw_text.strip()
            # Strip markdown json code fence if present
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\n", "", cleaned)
                cleaned = re.sub(r"\n```$", "", cleaned)
            data = json.loads(cleaned)

            has_physical = bool(data.get("has_physical_entity", True))
            raw_type = str(data.get("entity_type") or "").strip().lower()
            is_metaphorical = bool(data.get("is_metaphorical", raw_type == "metaphorical" or not has_physical))
            entity_type = "metaphorical" if is_metaphorical else "physical"

            return EntityExtractionResult(
                has_physical_entity=not is_metaphorical and has_physical,
                entity_type=entity_type,
                is_metaphorical=is_metaphorical,
                main_object=str(data.get("main_object") or "").strip(),
                search_query=str(data.get("search_query") or "").strip(),
                generation_prompt=str(data.get("generation_prompt") or "").strip(),
                object_fidelity_weight=float(data.get("object_fidelity_weight", 0.60 if is_metaphorical else 0.75)),
                raw_response=data
            )
        except Exception as e:
            logger.error(f"Error parsing entity extractor JSON: {e}, raw text: {raw_text[:200]}")
            # Graceful fallback heuristic
            words = fallback_text.strip().split()
            headline = " ".join(words[:6])
            return EntityExtractionResult(
                has_physical_entity=True,
                entity_type="physical",
                is_metaphorical=False,
                main_object=headline,
                search_query=f"{headline} product photo",
                generation_prompt=f"A clean editorial scene showcasing {headline}, high resolution, studio lighting",
                object_fidelity_weight=0.75
            )
