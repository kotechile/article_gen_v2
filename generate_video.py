import os
import sys
import json
import requests
import argparse
from bs4 import BeautifulSoup
import dotenv

# Load environment variables from the same directory
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
dotenv.load_dotenv(dotenv_path)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')

# Fallback to Supabase database if keys are not in environment
if not OPENAI_API_KEY or not ELEVENLABS_API_KEY:
    try:
        from supabase_client import get_api_key
        if not OPENAI_API_KEY:
            OPENAI_API_KEY = get_api_key('openai')
        if not ELEVENLABS_API_KEY:
            ELEVENLABS_API_KEY = get_api_key('elevenlabs')
    except Exception as e:
        pass

if not OPENAI_API_KEY:
    print("❌ ERROR: OPENAI_API_KEY not found in environment or Supabase database.")
    print("Please configure your API keys in the database to proceed.")
    sys.exit(1)


def scrape_article(url):
    """Scrapes the article URL and extracts title and main text content."""
    print(f"🌐 Scraping article from: {url}...")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to load article (Status Code: {response.status_code})")
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Try to extract the title
    title = soup.find('h1')
    title_text = title.text.strip() if title else "Untitled Article"
    
    # Extract paragraph texts
    paragraphs = soup.find_all('p')
    paragraphs_text = [p.text.strip() for p in paragraphs if len(p.text.strip()) > 30]
    
    # Join first few paragraphs for context (avoiding extremely long texts)
    body_text = "\n".join(paragraphs_text[:8])
    
    print(f"✔ Extracted Title: {title_text}")
    print(f"✔ Scraped {len(paragraphs_text)} paragraphs of text.")
    
    return title_text, body_text

def clean_json_response(text):
    """Strips markdown fences and extra whitespace to ensure clean JSON parsing."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

def generate_video_blueprint(title, body_text, primary=None, secondary=None, background=None):
    """Uses the resolved default LLM from Supabase to parse the article and return the RemotionVideoPayload JSON structure."""
    
    # 1. Resolve Default LLM
    try:
        from supabase_client import resolve_llm_provider
        from llm_client_direct import create_llm_client
        resolved = resolve_llm_provider(task_role="article_generation")
        provider = resolved.get("provider")
        model = resolved.get("model")
        api_key = resolved.get("api_key")
    except Exception as e:
        print(f"⚠️ Failed to import/use Supabase provider resolver: {e}")
        provider, model, api_key = None, None, None

    # Fallback to OpenAI env settings if Supabase resolution failed
    if not provider or not model or not api_key:
        print("⚠️ Falling back to default env-configured OpenAI GPT-4o...")
        provider = "openai"
        model = "gpt-4o"
        api_key = OPENAI_API_KEY

    print(f"🧠 Analyzing article content using resolved LLM: {provider}/{model}...")

    system_prompt = """
You are an expert automated video scriptwriter and motion graphics designer.
Your task is to take an article's title and text content, and output a highly structured JSON blueprint for a 30-second vertical or landscape video matching this exact JSON schema:

{
  "metadata": {
    "title": "Title of the video",
    "format": "vertical" | "landscape",
    "totalDurationInSeconds": 30,
    "brandColors": {
      "primary": "Hex color code matching the article mood (e.g. #8A2BE2)",
      "secondary": "Complementary accent color (e.g. #00FFFF)",
      "background": "Dark background color (e.g. #0B0C10)"
    }
  },
  "scenes": [
    {
      "sceneId": "scene_1",
      "type": "framework_hero" | "comparison_table" | "kpi_metric" | "broll_image" | "call_to_action",
      "durationInSeconds": 6.0,
      "heading": "Sleek heading for this scene",
      "subheading": "Optional subheading describing context",
      "voiceoverScript": "Specific voiceover words spoken strictly during this scene segment (around 12-15 words, matching the duration).",
      "imagePrompt": "Detailed, highly specific image prompt describing a premium conceptual photographic visual representing this scene. Focus on lighting, materials, and a refined professional style (e.g. cinematic studio lighting, dramatic shadows, dark metallic textures, textured aluminum, polished chrome, glowing sapphire accents). Avoid flat or cartoonish looks.",
      "visualKeyword": "A high-quality fallback keyword representing this scene",
      "tableData": {
        "headers": ["Header1", "Header2"],
        "rows": [["Cell1", "Cell2"], ["Cell3", "Cell4"]]
      },
      "kpiData": {
        "value": "e.g. +145% or 4.2M",
        "label": "Metric label"
      }
    }
  ]
}

Instructions:
1. You must generate exactly 5 scenes. The sum of "durationInSeconds" across all 5 scenes must be exactly 30.0 (e.g., 6.0 seconds per scene).
2. The scene "type" list must include one framework_hero, one kpi_metric, one comparison_table, one broll_image, and the final 5th scene must be a call_to_action scene to capture user attention and drive action.
3. The "voiceoverScript" represents the voiceover spoken *only* during that scene. Keep the language hook-driven, high-retention, and natural to read. Speakable word count per scene should be around 12 to 15 words maximum to match the 6.0s pacing.
4. Do not simply summarize headers. Read the body paragraphs, extract concrete metrics, analogies, or arguments, and write the voiceover and "imagePrompt" based on those specific details.
5. The "imagePrompt" must describe a premium, high-end, conceptual photographic style. Instruct the image generator with specifics like 'cinematic studio lighting', 'dramatic shadows', 'dark metallic environment', 'textured aluminum', 'polished dark chrome', and 'glowing neon/sapphire accents'. Ground metaphors physically rather than cartoonishly (e.g., an industrial balance scale instead of a flat vector scale).
6. Provide raw JSON output, without any markdown formatting wrappers or ```json tags.
"""

    user_prompt = f"Article Title: {title}\nArticle Body:\n{body_text}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        client = create_llm_client(provider=provider, model=model, api_key=api_key)
        response = client.generate(messages=messages)
        raw_content = clean_json_response(response.content)
        blueprint = json.loads(raw_content)
    except Exception as e:
        print(f"❌ LLM blueprint generation failed: {e}. Falling back to default mock blueprint.")
        raise e

    # Override colors if customized in CLI parameters
    if primary:
        blueprint['metadata']['brandColors']['primary'] = primary
    if secondary:
        blueprint['metadata']['brandColors']['secondary'] = secondary
    if background:
        blueprint['metadata']['brandColors']['background'] = background

    return blueprint

def generate_voiceover_openai(script_text, voice):
    """Uses OpenAI's TTS API to compile the voiceover track."""
    print(f"🎙 Generating voiceover using OpenAI TTS (Voice: {voice})...")
    url = "https://api.openai.com/v1/audio/speech"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "tts-1",
        "input": script_text,
        "voice": voice,
        "response_format": "mp3"
    }
    
    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 200:
        raise Exception(f"OpenAI TTS API failed: {response.text}")
    
    voiceover_path = os.path.join(os.path.dirname(__file__), '_remotion', 'public', 'voiceover.mp3')
    with open(voiceover_path, 'wb') as f:
        f.write(response.content)
    print(f"✔ Saved voiceover track to: {voiceover_path}")

def generate_voiceover_elevenlabs(script_text, voice_id):
    """Uses ElevenLabs API to generate a premium voiceover track."""
    if not ELEVENLABS_API_KEY:
        print("⚠️ WARNING: ELEVENLABS_API_KEY not found in content_generator/.env.")
        print("Fallback: Using OpenAI TTS voice 'onyx' instead.")
        return generate_voiceover_openai(script_text, "onyx")

    print(f"🎙 Generating realistic ElevenLabs voiceover (Voice ID: {voice_id})...")
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format=mp3_44100_128"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "text": script_text,
        "model_id": "eleven_v3",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }
    
    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 200:
        raise Exception(f"ElevenLabs TTS API failed (Status {response.status_code}): {response.text}")
        
    voiceover_path = os.path.join(os.path.dirname(__file__), '_remotion', 'public', 'voiceover.mp3')
    with open(voiceover_path, 'wb') as f:
        f.write(response.content)
    print(f"✔ Saved ElevenLabs voiceover track to: {voiceover_path}")

def get_flux_config():
    """Queries Supabase to find active Flux models and their API keys."""
    print("🔑 Fetching Flux API configuration from Supabase...")
    try:
        from supabase_client import get_supabase_client
        client = get_supabase_client()
        if not client:
            print("⚠️ Supabase client could not be initialized.")
            return []
            
        res = client.table('llm_providers_image').select('*').eq('is_active', True).execute()
        flux_models = []
        for row in (res.data or []):
            m_name = str(row.get('model_name') or '').lower()
            provider = str(row.get('provider') or '').lower()
            if 'flux' in m_name or 'flux' in provider or 'flix' in m_name:
                flux_models.append(row)
                
        configured_models = []
        for model in flux_models:
            key_id = model.get('api_keys_id')
            if not key_id:
                continue
            key_res = client.table('api_keys').select('key_value').eq('id', key_id).execute()
            if key_res.data and key_res.data[0].get('key_value'):
                configured_models.append({
                    'model_name': model.get('model_name'),
                    'display_name': model.get('display_name'),
                    'provider': model.get('provider'),
                    'api_key': key_res.data[0]['key_value']
                })
        return configured_models
    except Exception as e:
        print(f"⚠️ Error fetching Flux config from Supabase: {e}")
        return []

def generate_image_via_flux(model_config, prompt):
    """Triggers image generation via Flux API (supports fluxapi.ai and kie.ai) and returns image bytes."""
    provider = str(model_config.get('provider') or '').lower()
    model = model_config.get('model_name')
    api_key = model_config.get('api_key')
    
    print(f"🎨 Finally using Flux Model: '{model_config.get('display_name')}' [Model: '{model}', Provider: '{provider}']")
    print(f"   Prompt: '{prompt}'")
    
    if 'kie' in provider or 'flux-2' in model:
        # KIE implementation
        create_url = "https://api.kie.ai/api/v1/jobs/createTask"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        create_payload = {
            "model": model,
            "input": {
                "prompt": prompt,
                "aspect_ratio": "9:16",
                "resolution": "1K",
                "nsfw_checker": False
            }
        }
        r = requests.post(create_url, headers=headers, json=create_payload, timeout=30)
        r.raise_for_status()
        res_data = r.json()
        task_id = ((res_data.get("data") or {}).get("taskId") or "").strip()
        if not task_id:
            raise Exception(f"KIE did not return taskId: {res_data}")
            
        poll_url = "https://api.kie.ai/api/v1/jobs/recordInfo"
        import time
        for _ in range(60):
            time.sleep(2)
            poll_resp = requests.get(poll_url, headers={"Authorization": f"Bearer {api_key}"}, params={"taskId": task_id}, timeout=15)
            poll_resp.raise_for_status()
            poll_data = poll_resp.json()
            data = poll_data.get("data") or {}
            state = str(data.get("state") or "").lower()
            if state == "success":
                result_json = data.get("resultJson")
                parsed_result = {}
                if isinstance(result_json, dict):
                    parsed_result = result_json
                elif isinstance(result_json, str) and result_json.strip():
                    parsed_result = json.loads(result_json)
                image_url = (parsed_result.get("resultUrls") or [None])[0]
                if not image_url:
                    raise Exception(f"No resultUrls found: {poll_data}")
                img_r = requests.get(image_url, timeout=30)
                img_r.raise_for_status()
                return img_r.content
            elif state == "fail":
                raise Exception(f"KIE generation failed: {data.get('failMsg')}")
        raise Exception("KIE generation timed out")
    else:
        # fluxapi.ai implementation
        generate_url = 'https://api.fluxapi.ai/api/v1/flux/kontext/generate'
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        body = {
            "prompt": prompt,
            "enableTranslation": True,
            "aspectRatio": "9:16",
            "outputFormat": "jpeg",
            "model": model
        }
        r = requests.post(generate_url, headers=headers, json=body, timeout=30)
        r.raise_for_status()
        res_data = r.json()
        
        # Defensive check for taskId in different possible structures
        data_obj = res_data.get('data') or {}
        task_id = data_obj.get('taskId') or res_data.get('taskId')
        if not task_id:
            raise Exception(f"FluxAPI did not return taskId: {res_data}")
            
        polling_url = f"https://api.fluxapi.ai/api/v1/flux/kontext/record-info?taskId={task_id}"
        import time
        for _ in range(60):
            time.sleep(2)
            poll_resp = requests.get(polling_url, timeout=15)
            poll_resp.raise_for_status()
            poll_data = poll_resp.json()
            data_sec = poll_data.get("data") or {}
            
            # Check success flag
            success_flag = poll_data.get("success")
            if success_flag is None:
                success_flag = data_sec.get("success")
                
            status_code = poll_data.get("status") or data_sec.get("status")
            
            # If success == 1 or completed
            if success_flag == 1 or str(status_code).lower() in ("completed", "done", "success"):
                image_url = poll_data.get("imageUrl") or data_sec.get("imageUrl")
                if not image_url:
                    # check resultUrl / imgUrl
                    image_url = poll_data.get("resultUrl") or data_sec.get("resultUrl")
                if not image_url:
                    raise Exception(f"Task succeeded but image URL is missing: {poll_data}")
                img_r = requests.get(image_url, timeout=30)
                img_r.raise_for_status()
                return img_r.content
            elif success_flag == -1 or str(status_code).lower() in ("failed", "fail"):
                err_msg = poll_data.get("errorMessage") or data_sec.get("errorMessage") or "Unknown error"
                raise Exception(f"FluxAPI generation failed: {err_msg}")
        raise Exception("FluxAPI generation timed out")

def download_broll_images(blueprint):
    """Downloads custom images generated via Flux, falling back to Unsplash stock photos on failure."""
    print("🖼 Setting up B-Roll image assets for scenes...")
    
    # 1. Fetch available Flux configurations
    flux_configs = get_flux_config()
    selected_config = None
    if flux_configs:
        print(f"📢 Found {len(flux_configs)} active Flux model configuration(s) in DB.")
        # Prioritize KIE.ai model since user mentioned issues with fluxapi.ai
        kie_configs = [c for c in flux_configs if 'kie' in str(c.get('provider')).lower()]
        if kie_configs:
            selected_config = kie_configs[0]
        else:
            selected_config = flux_configs[0]
    else:
        print("⚠️ No active Flux configurations found in Supabase. Falling back to Unsplash.")

    for idx, scene in enumerate(blueprint['scenes']):
        heading = scene.get('heading', '')
        subheading = scene.get('subheading', '')
        keyword = scene.get('visualKeyword', 'business')
        
        # Check if the scene already has a user-defined custom image URL/path set
        existing_asset = scene.get('visualAssetUrl')
        if existing_asset and not existing_asset.startswith("scene_"):
            print(f"✔ Scene {idx + 1} has user-defined visual asset: {existing_asset}. Skipping auto-generation.")
            continue
            
        dest_filename = f"scene_{idx + 1}.jpg"
        dest_path = os.path.join(os.path.dirname(__file__), '_remotion', 'public', dest_filename)
        
        image_content = None
        
        # 2. Try Flux image generation first
        if selected_config:
            # Use the LLM's custom imagePrompt if available, otherwise construct a fallback
            flux_prompt = scene.get('imagePrompt')
            if not flux_prompt:
                flux_prompt = f"Professional conceptual tech photography representing '{heading}' ({subheading}). Cinematic studio lighting, dramatic shadows, dark metallic environment. Polished dark chrome, textured aluminum, glowing sapphire accents. High-end enterprise-grade layout, realistic physical textures, octanerender style."
            try:
                image_content = generate_image_via_flux(selected_config, flux_prompt)
                print(f"✔ Successfully generated custom image for Scene {idx + 1} via Flux.")
            except Exception as e:
                print(f"⚠️ Flux image generation failed for Scene {idx + 1}: {e}")
                print("Falling back to Unsplash for this scene...")
        
        # 3. Fall back to Unsplash stock photo if Flux failed or was not configured
        if image_content is None:
            unsplash_url = "https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1080&auto=format&fit=crop&q=80" # default data
            if "tech" in keyword or "code" in keyword:
                unsplash_url = "https://images.unsplash.com/photo-1517694712202-14dd9538aa97?w=1080&auto=format&fit=crop&q=80"
            elif "growth" in keyword or "chart" in keyword or "kpi" in keyword:
                unsplash_url = "https://images.unsplash.com/photo-1460925895917-afdab827c52f?w=1080&auto=format&fit=crop&q=80"
            elif "office" in keyword or "workspace" in keyword:
                unsplash_url = "https://images.unsplash.com/photo-1497366216548-37526070297c?w=1080&auto=format&fit=crop&q=80"
            elif "money" in keyword or "dollar" in keyword or "cost" in keyword:
                unsplash_url = "https://images.unsplash.com/photo-1559526324-4b87b5e36e44?w=1080&auto=format&fit=crop&q=80"
            elif "coffee" in keyword or "espresso" in keyword:
                unsplash_url = "https://images.unsplash.com/photo-1514432324607-a09d9b4aefdd?w=1080&auto=format&fit=crop&q=80"
                
            print(f"   Downloading stock Unsplash image for keyword '{keyword}' -> {dest_filename}")
            try:
                r = requests.get(unsplash_url, timeout=10)
                if r.status_code == 200:
                    image_content = r.content
                else:
                    print(f"   ⚠️ Unsplash download failed (HTTP {r.status_code})")
            except Exception as e:
                print(f"   ⚠️ Unsplash download failed. Error: {e}")
                
        # 4. Save to destination
        if image_content:
            with open(dest_path, 'wb') as f:
                f.write(image_content)
            scene['visualAssetUrl'] = dest_filename
        else:
            scene['visualAssetUrl'] = "background.mp3"  # Fallback to no-image mode if everything failed

def align_timings(blueprint, caption_position, aspect_ratio="vertical", host_url=None, music="background.mp3"):
    """Calculates frame timings for subtitles and scenes, keeping subtitles bound to their scene frame ranges."""
    fps = 30
    total_frames = 30 * fps # 30 seconds = 900 frames
    
    scenes = blueprint['scenes']
    subtitles = []
    
    current_scene_start_frame = 0
    
    for idx, sc in enumerate(scenes):
        # Calculate duration of this scene in frames
        duration_sec = sc.get('durationInSeconds', 7.5)
        scene_frames = int(duration_sec * fps)
        sc['durationInFrames'] = scene_frames
        
        scene_end_frame = current_scene_start_frame + scene_frames - 1
        
        # Parse the voiceover script for this scene and split into 2-3 word subtitle segments
        script_text = sc.get('voiceoverScript', '')
        words = script_text.split()
        chunks = []
        chunk_size = 3
        for i in range(0, len(words), chunk_size):
            chunks.append(" ".join(words[i:i+chunk_size]))
            
        if chunks:
            # Distribute subtitle segments evenly within this scene's duration
            chunk_duration = scene_frames // len(chunks)
            chunk_start = current_scene_start_frame
            for c_idx, chunk in enumerate(chunks):
                chunk_end = chunk_start + chunk_duration - 1
                if c_idx == len(chunks) - 1:
                    chunk_end = scene_end_frame
                    
                subtitles.append({
                    "text": chunk,
                    "startFrame": chunk_start,
                    "endFrame": chunk_end
                })
                chunk_start = chunk_end + 1
        else:
            # Fallback subtitle matching scene heading if script is empty
            subtitles.append({
                "text": sc.get('heading', ''),
                "startFrame": current_scene_start_frame,
                "endFrame": scene_end_frame
            })
            
        current_scene_start_frame += scene_frames
        
        # Set dynamic asset url (local vs remote host for lambda)
        asset_filename = sc.get('visualAssetUrl', f"scene_{idx + 1}.jpg")
        if host_url:
            sc['visualAssetUrl'] = f"{host_url}/api/v1/video/static/{asset_filename}"
        else:
            sc['visualAssetUrl'] = asset_filename
        
        # Clean up temporary fields
        if 'durationInSeconds' in sc:
            del sc['durationInSeconds']
        if 'voiceoverScript' in sc:
            del sc['voiceoverScript']
        if 'imagePrompt' in sc:
            del sc['imagePrompt']
        if 'visualKeyword' in sc:
            del sc['visualKeyword']

    # Finalize URLs
    final_voiceover = f"{host_url}/api/v1/video/static/voiceover.mp3" if host_url else "voiceover.mp3"
    final_music = f"{host_url}/api/v1/video/static/{music}" if host_url and not music.startswith("http") else music

    # Finalize payload
    payload = {
        "metadata": {
            "title": blueprint['metadata']['title'],
            "format": aspect_ratio,
            "totalDurationInSeconds": 30,
            "brandColors": blueprint['metadata']['brandColors'],
            "captionPosition": caption_position,
            "backgroundMusicUrl": final_music
        },
        "audioTrackUrl": final_voiceover,
        "subtitles": subtitles,
        "scenes": scenes
    }
    
    return payload

def main():
    parser = argparse.ArgumentParser(description="ArtiVids Automated Article-to-Video Engine")
    parser.add_argument("url", help="The URL of the article to turn into a video")
    parser.add_argument("--voice", default="onyx", help="Voice ID (ElevenLabs) or Voice Name (OpenAI: onyx, alloy, nova, shimmer, echo, fable)")
    parser.add_argument("--provider", default="openai", choices=["openai", "elevenlabs"], help="Voice generation service provider")
    parser.add_argument("--caption-position", default="center", choices=["center", "bottom", "top"], help="Vertical position of subtitles")
    parser.add_argument("--primary", help="Primary brand color (hex code e.g. #FF5733)")
    parser.add_argument("--secondary", help="Secondary brand color (hex code e.g. #33FF57)")
    parser.add_argument("--background", help="Background brand color (hex code e.g. #111111)")
    parser.add_argument("--aspect-ratio", default="vertical", choices=["vertical", "landscape"], help="Aspect ratio for the output video (vertical or landscape)")
    parser.add_argument("--host-url", help="Fully qualified domain of the backend server (to resolve remote assets for AWS Lambda)")
    parser.add_argument("--music", default="background.mp3", help="Background music selection")
    parser.add_argument("--render-on-lambda", action="store_true", help="Render video on AWS Lambda instead of locally")
    parser.add_argument("--blueprint-only", action="store_true", help="Only generate the blueprint JSON structure from LLM, then print to stdout and exit")
    parser.add_argument("--blueprint-payload", help="Path to a pre-existing blueprint JSON file to compile the video from directly")
    
    args = parser.parse_args()
    
    try:
        if args.blueprint_payload:
            # Skip Step 1 and 2, load the user-supplied blueprint directly
            print(f"📂 Loading pre-existing blueprint payload from: {args.blueprint_payload}...")
            with open(args.blueprint_payload, 'r') as f:
                blueprint = json.load(f)
        else:
            # Step 1: Scrape
            title, body_text = scrape_article(args.url)
            
            # Step 2: Blueprint
            blueprint = generate_video_blueprint(
                title, 
                body_text, 
                primary=args.primary, 
                secondary=args.secondary, 
                background=args.background
            )
            
            if args.blueprint_only:
                # Output raw JSON blueprint to stdout and exit
                print("\n=== BLUEPRINT_JSON_START ===")
                print(json.dumps(blueprint, indent=2))
                print("=== BLUEPRINT_JSON_END ===")
                return

        # Step 3: Voiceover
        full_script = " ".join([sc.get('voiceoverScript', '') for sc in blueprint['scenes']])
        if args.provider == "elevenlabs" or len(args.voice) > 15: # heuristic for ElevenLabs Voice ID hashes
            generate_voiceover_elevenlabs(full_script, args.voice)
        else:
            generate_voiceover_openai(full_script, args.voice)
        
        # Step 4: Download Images
        download_broll_images(blueprint)
        
        # Step 5: Align Timings & Caption Position
        payload = align_timings(
            blueprint, 
            args.caption_position, 
            aspect_ratio=args.aspect_ratio, 
            host_url=args.host_url, 
            music=args.music
        )
        
        # Step 6: Save JSON payload
        payload_path = os.path.join(os.path.dirname(__file__), '_remotion', 'src', 'mockPayload.json')
        with open(payload_path, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"✔ Saved aligned Remotion payload to: {payload_path}")
        
        # Step 7: Trigger Remotion Render
        print("\n🚀 Starting Remotion compilation engine...")
        remotion_dir = os.path.join(os.path.dirname(__file__), '_remotion')
        comp_id = args.aspect_ratio
        
        if getattr(args, 'render_on_lambda', False):
            # Your deployed S3 site serve URL
            serve_url = "https://remotionlambda-useast1-n9j3q72d18.s3.us-east-1.amazonaws.com/sites/artivids-engine/index.html"
            print(f"☁ Triggering AWS Lambda serverless render (Serve URL: {serve_url})...")
            render_cmd = f"cd {remotion_dir} && npx remotion lambda render {serve_url} {comp_id} --region=us-east-1 --props=src/mockPayload.json --concurrency=20 --timeout=300000 --function-name=remotion-render-4-0-484-mem3008mb-disk2048mb-300sec output-generated.mp4"
        else:
            print("💻 Triggering local render on the host...")
            render_cmd = f"cd {remotion_dir} && npx remotion render {comp_id} output-generated.mp4 --props src/mockPayload.json"
            
        # Load AWS credentials from _remotion/.env into the subprocess environment
        env = os.environ.copy()
        remotion_env_path = os.path.join(remotion_dir, '.env')
        if os.path.exists(remotion_env_path):
            print(f"🔑 Loading AWS credentials from {remotion_env_path}...")
            remotion_env_vars = dotenv.dotenv_values(remotion_env_path)
            for k, v in remotion_env_vars.items():
                if v:
                    env[k] = v
                    
        import subprocess
        subprocess.run(render_cmd, shell=True, check=True, env=env)
        
        print("\n==========================================")
        print("🎉 SUCCESS! Video generated successfully!")
        print(f"   Output video: _remotion/output-generated.mp4")
        print("==========================================")
        
    except Exception as e:
        print(f"\n❌ Error generating video: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
