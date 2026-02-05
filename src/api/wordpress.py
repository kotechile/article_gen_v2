from flask import Blueprint, jsonify, request
from supabase_client import get_supabase_client
from src.utils.wordpress_client import WordPressClient
import logging

wordpress_bp = Blueprint('wordpress', __name__)
logger = logging.getLogger(__name__)

@wordpress_bp.route('/api/wordpress/sync-posts', methods=['POST'])
def sync_wordpress_posts():
    """
    Fetch posts from all configured WordPress sites for the user and save to DB.
    Expected query param: user_id (or from auth context if available, but staying simple for now)
    """
    try:
        # Get supabase client using the wrapper
        supabase = get_supabase_client()
        if not supabase:
            logger.error("Supabase client not initialized")
            return jsonify({'error': 'Internal server error: Database connection failed'}), 500

        user_id = request.args.get('user_id')
        if not user_id:
             # Try getting from JSON body if not in args
            data = request.get_json(silent=True)
            if data:
                user_id = data.get('user_id')
        
        if not user_id:
            return jsonify({'error': 'Missing user_id'}), 400

        # 1. Get WP Credentials for User
        response = supabase.table("wordPress_details").select("*").eq("user_id", user_id).execute()
        sites = response.data
        
        if not sites:
            return jsonify({'total_synced': 0, 'details': "No WordPress sites configured", 'logs': ["No sites found"]}), 200

        debug_logs = []
        debug_logs.append(f"Found {len(sites)} sites to sync")
        total_posts_saved = 0
        
        # 2. Iterate each site
        for i, site in enumerate(sites):
            domain = site.get('domain')
            debug_logs.append(f"Processing site {i+1}/{len(sites)}: {domain}")
            try:
                username = site.get('wpUserName')
                password = site.get('wordpress_key')
                site_id = site.get('id')
                
                if not domain or not username or not password:
                    debug_logs.append(f"Skipping site {domain} due to missing credentials")
                    continue
                    
                client = WordPressClient(domain, username, password)
                
                # Fetch Categories
                debug_logs.append(f"Fetching categories for {domain}...")
                try:
                    categories = client.get_categories()
                    if categories:
                         supabase.table("wordPress_details").update({"categories": categories}).eq("id", site_id).execute()
                         debug_logs.append(f"Synced {len(categories)} categories")
                except Exception as cat_err:
                     debug_logs.append(f"Error fetching categories: {cat_err}")
                
                # Fetch recent posts (e.g., last 20)
                debug_logs.append(f"Fetching posts for {domain}...")
                try:
                    posts = client.get_posts(per_page=20)
                    debug_logs.append(f"Fetched {len(posts)} posts for {domain}")
                except Exception as fetch_err:
                    debug_logs.append(f"Error fetching posts for {domain}: {str(fetch_err)}")
                    continue
                
                if not posts:
                    debug_logs.append(f"No posts found for {domain}")
                    continue
                    
                # 3. Save to Supabase
                records = []
                for post in posts:
                    title = post.get('title', {}).get('rendered', '')
                    excerpt = post.get('excerpt', {}).get('rendered', '')
                    link = post.get('link', '')
                    post_id = post.get('id')
                    
                    records.append({
                        "user_id": user_id,
                        "wordpress_detail_id": site_id,
                        "post_id": post_id,
                        "title": title,
                        "link": link,
                        "excerpt": excerpt
                    })
                
                if records:
                    debug_logs.append(f"Saving {len(records)} records for {domain}")
                    # Delete old posts for this site (simple sync)
                    supabase.table("wordpress_imported_posts").delete().eq("wordpress_detail_id", site_id).execute()
                    
                    # Insert new
                    supabase.table("wordpress_imported_posts").insert(records).execute()
                    total_posts_saved += len(records)
                    debug_logs.append(f"Saved records for {domain}")
                    
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.error(f"Failed to sync site {site.get('domain')}: {str(e)}")
                debug_logs.append(f"Error syncing {site.get('domain')}: {str(e)}")
                continue

        return jsonify({
            'total_synced': total_posts_saved, 
            'details': f"Sync completed. Processed {len(sites)} sites.",
            'logs': debug_logs
        }), 200

    except Exception as e:
        logger.error(f"Sync error: {str(e)}")
        return jsonify({'error': str(e), 'logs': [str(e)]}), 500
