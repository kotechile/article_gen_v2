
    async def _generate_blog_idea_with_llm(
        self,
        subtopic: str,
        topic_title: str,
        selected_keywords: List[Dict[str, Any]],
        topic_id: str,
        user_id: str,
        idea_index: int
    ) -> Dict[str, Any]:
        """Generate a blog idea using LLM based on keywords"""
        try:
            logger.info(f"Generating blog idea with LLM for subtopic: {subtopic}")

            creds = await self._get_llm_credentials()
            provider_type = creds['provider_type']
            model_name = creds['model_name']
            api_key = creds['api_key']

            primary_keyword = max(selected_keywords, key=lambda k: k.get('priority_score', 0))
            keywords_list = ", ".join([k.get('keyword', '') for k in selected_keywords])

            prompt = f"""
            Generate a unique and SEO-optimized Blog Post idea for the topic "{topic_title}" focusing on the subtopic "{subtopic}".
            
            Context:
            - Primary Keyword: {primary_keyword.get('keyword', '')}
            - Secondary Keywords: {keywords_list}
            
            Requirements:
            1. Title must be catchy, SEO-friendly, and include the primary keyword if natural.
            2. Description must be a compelling hook and summary of what the article covers.
            3. Target Audience (e.g., "Beginners", "Experts").
            4. Content Angle (e.g., "Step-by-step Guide", "Case Study", "Deep Dive", "Listicle").
            
            Return JSON format ONLY:
            {{
                "title": "string",
                "description": "string",
                "target_audience": "string",
                "content_angle": "string"
            }}
            """

            content = await self._call_llm(provider_type, model_name, api_key, prompt)
            
            try:
                # Clean markdown code blocks if present
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                data = json.loads(content)
            except Exception:
                 # Fallback
                return self._generate_blog_idea_from_keywords(
                    subtopic, topic_title, selected_keywords, topic_id, user_id, idea_index
                )

            # Calculate metrics
            avg_search_volume = sum(kw.get('search_volume', 0) for kw in selected_keywords) / len(selected_keywords)
            avg_difficulty = sum(kw.get('keyword_difficulty', 0) for kw in selected_keywords) / len(selected_keywords)
            avg_cpc = sum(kw.get('cpc', 0) for kw in selected_keywords) / len(selected_keywords)
            
            seo_score = self._calculate_seo_score_from_keywords(selected_keywords)
            difficulty_level = self._map_difficulty_score(avg_difficulty)
            monetization_potential = self._calculate_monetization_potential(selected_keywords)

            return {
                "title": data.get("title"),
                "description": data.get("description"),
                "content_type": "blog",
                "category": "seo_optimized",
                "subtopic": subtopic,
                "topic_id": topic_id,
                "user_id": user_id,
                "keywords": [kw.get('keyword', '') for kw in selected_keywords],
                "keyword_metrics": {
                    "avg_search_volume": round(avg_search_volume, 0),
                    "avg_difficulty": round(avg_difficulty, 1),
                    "avg_cpc": round(avg_cpc, 2),
                    "primary_keyword": primary_keyword.get('keyword', ''),
                    "total_keywords_used": len(selected_keywords)
                },
                "seo_score": seo_score,
                "difficulty_level": difficulty_level,
                "estimated_read_time": random.randint(5, 15),
                "target_audience": data.get("target_audience", "General"),
                "content_angle": data.get("content_angle", "Guide"),
                "monetization_potential": monetization_potential,
                "generation_method": "llm_enhanced",
                "data_source": "real_keyword_metrics"
            }

        except Exception as e:
            logger.error(f"LLM blog idea generation failed: {str(e)}")
            return self._generate_blog_idea_from_keywords(
                subtopic, topic_title, selected_keywords, topic_id, user_id, idea_index
            )
