from flask import Blueprint, jsonify, request
from supabase_client import get_supabase_client
from src.utils.wordpress_client import WordPressClient
import logging
import re
import requests
from datetime import datetime

wordpress_bp = Blueprint('wordpress', __name__)
logger = logging.getLogger(__name__)


def _slugify(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return re.sub(r"(^-|-$)", "", value) or "category"

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


@wordpress_bp.route('/api/wordpress/sync-project-categories', methods=['POST'])
def sync_project_categories_to_wordpress():
    """
    Synchronize a project's local categories/subcategories to its WordPress site.

    Expected JSON body:
    {
      "user_id": "<uuid>",
      "project_id": "<uuid>"
    }
    """
    try:
        supabase = get_supabase_client()
        if not supabase:
            logger.error("Supabase client not initialized")
            return jsonify({'error': 'Internal server error: Database connection failed'}), 500

        data = request.get_json(silent=True) or {}
        user_id = data.get('user_id') or request.args.get('user_id')
        project_id = data.get('project_id') or request.args.get('project_id')

        if not user_id or not project_id:
            return jsonify({'error': 'Missing user_id or project_id'}), 400

        project_resp = (
            supabase
            .table("projects")
            .select("id, user_id, domain, wpUserName, wordpress_key, app_name")
            .eq("id", project_id)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        project_rows = project_resp.data or []
        if not project_rows:
            return jsonify({'error': 'Project not found'}), 404
        project = project_rows[0]

        domain = (project.get("domain") or "").strip()
        username = (project.get("wpUserName") or "").strip()
        app_password = (project.get("wordpress_key") or "").strip()
        if not domain or not username or not app_password:
            return jsonify({'error': 'WordPress credentials are incomplete for this project'}), 400

        categories_resp = (
            supabase
            .table("project_categories")
            .select("id, name, slug, level, parent_category_id, sort_order")
            .eq("project_id", project_id)
            .eq("user_id", user_id)
            .order("level", desc=False)
            .order("sort_order", desc=False)
            .order("name", desc=False)
            .execute()
        )
        local_categories = categories_resp.data or []
        if not local_categories:
            return jsonify({
                "success": True,
                "synced": 0,
                "created": 0,
                "updated": 0,
                "details": "No local categories to sync."
            }), 200

        client = WordPressClient(domain, username, app_password)
        try:
            wp_categories = client.get_categories_detailed()
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            reason = None
            if e.response is not None:
                try:
                    body = e.response.json() or {}
                    reason = body.get("message") or body.get("code")
                except Exception:
                    reason = e.response.text[:300]
            logger.warning(
                "WordPress category fetch failed for domain=%s project=%s status=%s reason=%s",
                domain,
                project_id,
                status,
                reason,
            )
            return jsonify({
                "success": False,
                "error": "WordPress API request failed while reading categories",
                "status_code": status,
                "details": reason or str(e),
            }), 400 if status in (400, 401, 403, 404) else 502
        except requests.exceptions.RequestException as e:
            logger.warning(
                "WordPress network error for domain=%s project=%s: %s",
                domain,
                project_id,
                str(e),
            )
            return jsonify({
                "success": False,
                "error": "Failed to connect to WordPress API",
                "details": str(e),
            }), 502

        # Build quick lookup indexes.
        by_slug_parent = {}
        by_name_parent = {}
        by_id = {}
        for cat in wp_categories:
            cat_id = int(cat.get("id"))
            parent = int(cat.get("parent") or 0)
            slug = (cat.get("slug") or "").strip().lower()
            name = (cat.get("name") or "").strip().lower()
            by_id[cat_id] = cat
            if slug:
                by_slug_parent[(slug, parent)] = cat
            if name:
                by_name_parent[(name, parent)] = cat

        created_count = 0
        updated_count = 0
        synced_count = 0
        local_to_wp: dict[str, int] = {}
        update_rows = []
        sync_errors = []
        sync_details = []

        level_1 = [c for c in local_categories if int(c.get("level") or 0) == 1]
        level_2 = [c for c in local_categories if int(c.get("level") or 0) == 2]

        by_slug_global = {}
        for cat in wp_categories:
            slug = (cat.get("slug") or "").strip().lower()
            if slug:
                by_slug_global[slug] = cat

        def ensure_wp_category(local_cat, parent_wp_id: int = 0):
            nonlocal created_count, updated_count, synced_count
            name = (local_cat.get("name") or "").strip()
            slug = (local_cat.get("slug") or "").strip().lower() or _slugify(name)
            if not name:
                return None

            existing = (
                by_slug_parent.get((slug, parent_wp_id))
                or by_name_parent.get((name.lower(), parent_wp_id))
                or by_slug_global.get(slug)
            )
            if existing:
                cat_id = int(existing.get("id"))
                needs_update = (
                    (existing.get("name") or "").strip() != name
                    or (existing.get("slug") or "").strip().lower() != slug
                    or int(existing.get("parent") or 0) != int(parent_wp_id or 0)
                )
                if needs_update:
                    updated = client.update_category(cat_id, name=name, slug=slug, parent=parent_wp_id)
                    existing = updated
                    updated_count += 1
            else:
                created = client.create_category(name=name, slug=slug, parent=parent_wp_id)
                existing = created
                created_count += 1

            cat_id = int(existing.get("id"))
            # refresh indexes for child lookups
            by_id[cat_id] = existing
            by_slug_parent[(slug, int(parent_wp_id or 0))] = existing
            by_name_parent[(name.lower(), int(parent_wp_id or 0))] = existing
            by_slug_global[slug] = existing
            synced_count += 1
            return cat_id

        # 1) Sync parent categories first.
        for cat in level_1:
            local_id = str(cat.get("id"))
            try:
                wp_id = ensure_wp_category(cat, parent_wp_id=0)
            except Exception as e:
                sync_errors.append({
                    "local_category_id": local_id,
                    "name": cat.get("name"),
                    "level": 1,
                    "error": str(e),
                })
                continue
            if wp_id:
                local_to_wp[local_id] = wp_id
                update_rows.append({
                    "id": local_id,
                    "wordpress_category_id": wp_id,
                    "wordpress_parent_category_id": None,
                    "wordpress_site_domain": domain,
                    "wordpress_last_synced_at": datetime.utcnow().isoformat(),
                })
                sync_details.append({
                    "local_category_id": local_id,
                    "name": cat.get("name"),
                    "level": 1,
                    "wordpress_category_id": wp_id,
                })

        # 2) Sync subcategories, linked to mapped parent.
        for cat in level_2:
            local_id = str(cat.get("id"))
            local_parent = str(cat.get("parent_category_id") or "")
            parent_wp_id = local_to_wp.get(local_parent, 0)
            try:
                wp_id = ensure_wp_category(cat, parent_wp_id=parent_wp_id)
            except Exception as e:
                sync_errors.append({
                    "local_category_id": local_id,
                    "name": cat.get("name"),
                    "level": 2,
                    "parent_local_id": local_parent or None,
                    "parent_wordpress_id": parent_wp_id or None,
                    "error": str(e),
                })
                continue
            if wp_id:
                local_to_wp[local_id] = wp_id
                update_rows.append({
                    "id": local_id,
                    "wordpress_category_id": wp_id,
                    "wordpress_parent_category_id": parent_wp_id or None,
                    "wordpress_site_domain": domain,
                    "wordpress_last_synced_at": datetime.utcnow().isoformat(),
                })
                sync_details.append({
                    "local_category_id": local_id,
                    "name": cat.get("name"),
                    "level": 2,
                    "parent_local_id": local_parent or None,
                    "parent_wordpress_id": parent_wp_id or None,
                    "wordpress_category_id": wp_id,
                })

        # Persist mappings back to project_categories.
        mapping_update_errors = []
        for row in update_rows:
            update_payload = {
                "wordpress_category_id": row["wordpress_category_id"],
                "wordpress_parent_category_id": row["wordpress_parent_category_id"],
                "wordpress_site_domain": row["wordpress_site_domain"],
                "updated_at": datetime.utcnow().isoformat(),
            }
            # wordpress_last_synced_at may not exist on older schemas; degrade gracefully.
            try:
                update_payload["wordpress_last_synced_at"] = row["wordpress_last_synced_at"]
                supabase.table("project_categories").update(update_payload).eq("id", row["id"]).eq("user_id", user_id).execute()
            except Exception:
                # First fallback: retry without wordpress_last_synced_at for older schemas.
                try:
                    update_payload.pop("wordpress_last_synced_at", None)
                    supabase.table("project_categories").update(update_payload).eq("id", row["id"]).eq("user_id", user_id).execute()
                except Exception as persist_err:
                    # Do not fail the whole sync if mapping persistence fails.
                    mapping_update_errors.append({
                        "local_category_id": row["id"],
                        "error": str(persist_err),
                    })

        return jsonify({
            "success": True,
            "project_id": project_id,
            "domain": domain,
            "synced": synced_count,
            "created": created_count,
            "updated": updated_count,
            "errors_count": len(sync_errors),
            "errors": sync_errors,
            "mapping_update_errors_count": len(mapping_update_errors),
            "mapping_update_errors": mapping_update_errors,
            "category_results": sync_details,
            "details": (
                f"Synced {synced_count} categories to WordPress "
                f"({created_count} created, {updated_count} updated, "
                f"{len(sync_errors)} sync errors, {len(mapping_update_errors)} mapping update errors)."
            ),
        }), 200
    except Exception as e:
        logger.error(f"Project category sync error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500
