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
Your task is to analyze an excerpt of article text and determine the visual approach:

1. User Instructions Priority:
   - If the user provides custom creative instructions, strictly follow their requested entity, theme, or direction.

2. Physical Entity vs. Abstract Metaphor Classification:

   A. Physical Entity Mode (has_physical_entity: true):
      - Use when the text centers on a SPECIFIC, tangible physical subject, product, or trade:
        * Consumer electronics, gadgets, wearables (e.g., Apple Watch, Sony camera, drone, smartphone).
        * Specific vehicles, machinery, hardware, robotics (e.g., Tesla Cybertruck, Boston Dynamics Atlas robot, 3D printer).
        * Specific manual trades & craftspeople (e.g., plumber repairing copper pipes, electrician at breaker panel, carpenter).
      - In this mode:
        * has_physical_entity: true
        * entity_type: "physical"
        * is_metaphorical: false
        * search_query: 2 to 4 simple, concrete words describing the physical object/subject for web image search (e.g., "plumber fixing pipe", "Apple Watch Ultra", "humanoid robot walking").
        * generation_prompt: A cinematic diffusion prompt placing this physical entity into a realistic scene.
        * object_fidelity_weight: 0.75

   B. Conceptual Metaphor Mode (has_physical_entity: false):
      - Use when there is NO specific physical object or trade centered in the text (e.g., abstract concepts like AI collaboration, career growth, strategy, inflation, data security, teamwork, cloud computing).
      - In this mode, DO NOT search for images online. Modern diffusion models (like Flux) generate stunning, realistic images directly from prompt:
        * has_physical_entity: false
        * entity_type: "metaphorical"
        * is_metaphorical: true
        * search_query: "" (leave EMPTY, as web search is skipped for abstract metaphors)
        * generation_prompt: A rich, photorealistic, cinematic prompt (using analogies from the text if available, like a driver in a car with GPS, or a compelling symbolic scene like a brass scale or climber) designed for direct text-to-image diffusion.
        * object_fidelity_weight: 0.0

Respond ONLY with a valid JSON object adhering strictly to this schema:
{
  "has_physical_entity": true/false,
  "entity_type": "physical" or "metaphorical",
  "is_metaphorical": true/false,
  "main_object": "Specific physical entity or metaphorical scene description",
  "search_query": "2 to 4 words for physical search, or empty string if metaphorical",
  "generation_prompt": "Cinematic diffusion prompt for realistic scene, lighting, 35mm photography",
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
                api_model = "deepseek-chat" if clean_model in ["deepseek-v4-flash", "deepseek-v4-pro", "default"] or "flash" in clean_model else clean_model
                # Using openai/ prefix with deepseek api_base ensures full compatibility across LiteLLM versions
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
            logger.info(f"Raw LLM output in _parse_json_response: {raw_text[:300]}")
            # 1. Strip think tags emitted by reasoning models (e.g. DeepSeek R1 / DeepSeek Reasoner)
            cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()

            # 2. Strip markdown code fence or extract JSON object substring
            json_match = re.search(r"(\{[\s\S]*\})", cleaned)
            if json_match:
                cleaned = json_match.group(1).strip()
            elif cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\n", "", cleaned)
                cleaned = re.sub(r"\n```$", "", cleaned).strip()

            data = json.loads(cleaned)
            main_obj = str(data.get("main_object") or "").strip()
            search_q = str(data.get("search_query") or "").strip()
            gen_prompt = str(data.get("generation_prompt") or "").strip()

            if not main_obj:
                raise ValueError("Parsed JSON missing 'main_object'")

            has_physical = bool(data.get("has_physical_entity", True))
            raw_type = str(data.get("entity_type") or "").strip().lower()
            is_metaphorical = bool(data.get("is_metaphorical", raw_type == "metaphorical" or not has_physical))
            entity_type = "metaphorical" if is_metaphorical else "physical"

            return EntityExtractionResult(
                has_physical_entity=not is_metaphorical and has_physical,
                entity_type=entity_type,
                is_metaphorical=is_metaphorical,
                main_object=main_obj,
                search_query=search_q or f"{main_obj} photo",
                generation_prompt=gen_prompt or f"A cinematic 35mm editorial photograph of {main_obj}",
                object_fidelity_weight=float(data.get("object_fidelity_weight", 0.60 if is_metaphorical else 0.75)),
                raw_response=data
            )
        except Exception as e:
            logger.error(f"Error parsing entity extractor JSON: {e}, raw text: {raw_text[:300]}")
            # Intelligent fallback heuristic: search for physical trade professions, crafts, or robots in text
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

            return EntityExtractionResult(
                has_physical_entity=True,
                entity_type="physical",
                is_metaphorical=False,
                main_object=subject,
                search_query=query,
                generation_prompt=f"A cinematic 35mm editorial photograph of {subject.lower()}, authentic work environment, natural practical lighting, rich textures",
                object_fidelity_weight=0.75
            )
