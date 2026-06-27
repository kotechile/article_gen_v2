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

def generate_video_blueprint(title, body_text, primary=None, secondary=None, background=None):
    """Uses OpenAI GPT-4o to parse the article and return the RemotionVideoPayload JSON structure."""
    print("🧠 Analyzing article content using GPT-4o to generate script, scenes, and subtitles...")
    
    system_prompt = """
You are an expert automated video scriptwriter and motion graphics designer.
Your task is to take an article's title and text content, and output a highly structured JSON blueprint for a 30-second vertical video (YouTube Shorts layout) matching this exact JSON schema:

{
  "metadata": {
    "title": "Title of the video",
    "format": "vertical",
    "totalDurationInSeconds": 30,
    "brandColors": {
      "primary": "Hex color code matching the article mood (e.g. #8A2BE2)",
      "secondary": "Complementary accent color (e.g. #00FFFF)",
      "background": "Dark background color (e.g. #0B0C10)"
    }
  },
  "subtitles": [
    { "text": "Short word or 2-3 word phrase", "relativeWeight": 1 }
  ],
  "scenes": [
    {
      "sceneId": "unique_scene_id",
      "type": "framework_hero" | "comparison_table" | "kpi_metric" | "broll_image",
      "relativeWeight": 2,
      "heading": "Sleek heading for this scene",
      "subheading": "Optional subheading describing context",
      "visualKeyword": "A high-quality image search keyword representing this scene",
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
1. "subtitles" is a list of sequential word-by-word or short phrases (2-3 words max per subtitle) that make up the spoken voiceover script of the video. The script must be engaging, informative, and take exactly 30 seconds to speak (around 65-75 words).
2. "relativeWeight" in subtitles is the estimated relative length of that phrase. Keep it around 1-3.
3. "scenes" is a list of sequential visual layouts that display on screen. You must generate exactly 4 scenes. The types must showcase the platform features (include at least one framework_hero, one kpi_metric, one comparison_table, and one broll_image).
4. "relativeWeight" in scenes represents how long the scene displays relative to others. Ensure the sum of all scene relativeWeights maps to the 30-second duration.
5. "visualKeyword" will be used to automatically fetch high-quality stock b-roll images (e.g., "tech", "growth-chart", "workspace").
6. Provide raw JSON output, without any markdown formatting wrappers or ```json tags.
"""

    user_prompt = f"Article Title: {title}\nArticle Body:\n{body_text}"
    
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.7,
        "response_format": {"type": "json_object"}
    }
    
    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 200:
        raise Exception(f"OpenAI GPT-4o API failed: {response.text}")
        
    result = response.json()
    blueprint = json.loads(result['choices'][0]['message']['content'])

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
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "text": script_text,
        "model_id": "eleven_monolingual_v1",
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
        
        dest_filename = f"scene_{idx + 1}.jpg"
        dest_path = os.path.join(os.path.dirname(__file__), '_remotion', 'public', dest_filename)
        
        image_content = None
        
        # 2. Try Flux image generation first
        if selected_config:
            # Construct a descriptive, high-quality prompt for the scene
            flux_prompt = f"Sleek modern 3D tech graphic illustration about '{heading}'. {subheading}. Niche concept: {keyword}. Cyberpunk synthwave dark mode color scheme, high resolution, clean layout."
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

def align_timings(blueprint, caption_position):
    """Calculates frame timings for subtitles and scenes based on relative weights."""
    fps = 30
    total_frames = 30 * fps # 30 seconds = 900 frames
    
    # 1. Align Subtitles
    subtitles = blueprint['subtitles']
    total_sub_weight = sum(s.get('relativeWeight', 1) for s in subtitles)
    current_frame = 0
    for s in subtitles:
        weight = s.get('relativeWeight', 1)
        duration = int((weight / total_sub_weight) * total_frames)
        s['startFrame'] = current_frame
        s['endFrame'] = current_frame + duration
        current_frame = s['endFrame'] + 1
        
        if 'relativeWeight' in s:
            del s['relativeWeight']
            
    # 2. Align Scenes
    scenes = blueprint['scenes']
    total_scene_weight = sum(sc.get('relativeWeight', 1) for sc in scenes)
    current_frame = 0
    for sc in scenes:
        weight = sc.get('relativeWeight', 1)
        duration = int((weight / total_scene_weight) * total_frames)
        sc['durationInFrames'] = duration
        current_frame += duration
        
        if 'relativeWeight' in sc:
            del sc['relativeWeight']
        if 'visualKeyword' in sc:
            del sc['visualKeyword']

    # Finalize payload
    payload = {
        "metadata": {
            "title": blueprint['metadata']['title'],
            "format": "vertical",
            "totalDurationInSeconds": 30,
            "brandColors": blueprint['metadata']['brandColors'],
            "captionPosition": caption_position
        },
        "audioTrackUrl": "voiceover.mp3",
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
    parser.add_argument("--render-on-lambda", action="store_true", help="Render video on AWS Lambda instead of locally")
    
    args = parser.parse_args()
    
    try:
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
        
        # Step 3: Voiceover
        full_script = " ".join([s['text'] for s in blueprint['subtitles']])
        if args.provider == "elevenlabs" or len(args.voice) > 15: # heuristic for ElevenLabs Voice ID hashes
            generate_voiceover_elevenlabs(full_script, args.voice)
        else:
            generate_voiceover_openai(full_script, args.voice)
        
        # Step 4: Download Images
        download_broll_images(blueprint)
        
        # Step 5: Align Timings & Caption Position
        payload = align_timings(blueprint, args.caption_position)
        
        # Step 6: Save JSON payload
        payload_path = os.path.join(os.path.dirname(__file__), '_remotion', 'src', 'mockPayload.json')
        with open(payload_path, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"✔ Saved aligned Remotion payload to: {payload_path}")
        
        # Step 7: Trigger Remotion Render
        print("\n🚀 Starting Remotion compilation engine...")
        remotion_dir = os.path.join(os.path.dirname(__file__), '_remotion')
        
        if getattr(args, 'render_on_lambda', False):
            # Your deployed S3 site serve URL
            serve_url = "https://remotionlambda-useast1-n9j3q72d18.s3.us-east-1.amazonaws.com/sites/artivids-engine/index.html"
            print(f"☁ Triggering AWS Lambda serverless render (Serve URL: {serve_url})...")
            render_cmd = f"cd {remotion_dir} && npx remotion lambda render {serve_url} vertical --region=us-east-1 --props=src/mockPayload.json --concurrency=1 output-generated.mp4"
        else:
            print("💻 Triggering local render on the host...")
            render_cmd = f"cd {remotion_dir} && npx remotion render vertical output-generated.mp4 --props src/mockPayload.json"
            
        os.system(render_cmd)
        
        print("\n==========================================")
        print("🎉 SUCCESS! Video generated successfully!")
        print(f"   Output video: _remotion/output-generated.mp4")
        print("==========================================")
        
    except Exception as e:
        print(f"\n❌ Error generating video: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
