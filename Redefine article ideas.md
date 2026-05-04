Recommended process
1. Start with a topic using the existing New Reserach page. Allow to select just one topic from the list and start a research. The output of this process should still generate article titles and software ideas that can be stored in Content Ideas and released to Content-Studio (Supabase table Titles). The article ideas created this way should contain priimary and secondary keywords. The reserached keyword data can still be stored in Supabase and displayed using Keyword Intelligence (same functionality that will allow manual research of keywords). You can reuse cideas from  current implementation like angle, description, etc.

This change will eliminate the current implementation that creates Sub-topics and generates ideas for these subtopics. 

The new process is as follows:

Example topic/title idea:

“Do eco-friendly home improvements increase property value?”

Do not only search that exact phrase. Break it into seed directions:

eco friendly home improvements
green home upgrades
energy efficient home improvements
home resale value upgrades
green retrofit ROI
solar panels home value
energy efficient windows resale value
smart thermostat home value

Your goal is to create a seed set of 10–30 phrases that describe the same article area from different angles.

2. Expand directionally first

Use:

POST /v3/dataforseo_labs/google/keyword_ideas/live

This should be your first expansion API because it finds keywords in the same topic/category area, not only phrases containing your seed term. DataForSEO says Keyword Ideas selects keywords that fall into the same categories as the seed keywords and returns search volume trend, CPC, and competition values.

Use it with 10–30 seed keywords.

Suggested settings:

{
  "keywords": [
    "eco friendly home improvements",
    "green home upgrades",
    "energy efficient home improvements",
    "home resale value upgrades"
  ],
  "location_code": 2840,
  "language_code": "en",
  "limit": 1000
}

For U.S. English, location_code: 2840 and language_code: "en" are commonly used.

3. Add adjacent keywords

Use:

POST /v3/dataforseo_labs/google/related_keywords/live

This gives you related search paths. It is good for discovering “side doors” into the topic — for example, the better article angle may not be “eco-friendly home improvements,” but something like:

energy efficient upgrades that increase home value
green home tax credits
best home upgrades for resale value
solar panels resale value
heat pump ROI

Related Keywords can return up to 4,680 keyword ideas by depth and includes search volume trend, CPC, and competition values.

Use a shallow depth first:

{
  "keyword": "eco friendly home improvements",
  "location_code": 2840,
  "language_code": "en",
  "depth": 2,
  "limit": 1000
}
4. Add Google Ads keyword suggestions

Use:

POST /v3/keywords_data/google_ads/keywords_for_keywords/live

This is especially useful because your definition of “best” includes high ad competition. This API can take up to 20 seed terms and return up to 20,000 keyword suggestions with essential keyword data.

Use this to find commercially meaningful variants:

{
  "keywords": [
    "green home upgrades",
    "energy efficient home improvements",
    "home value improvements",
    "green retrofit"
  ],
  "location_code": 2840,
  "language_code": "en"
}
5. Use exact-match expansion only after that

Use:

POST /v3/dataforseo_labs/google/keyword_suggestions/live

This API gives long-tail phrases that include the seed keyword. It is useful, but I would not use it first because your goal is directional discovery, not exact phrase matching. DataForSEO describes this endpoint as returning search queries that include the specified seed keyword, with extra words before, after, or inside the phrase.

Use it after you already know which cluster is promising.

Example:

{
  "keyword": "energy efficient home upgrades",
  "location_code": 2840,
  "language_code": "en",
  "limit": 500
}
6. Clean and normalize the candidate list

After steps 2–5, you may have thousands of keywords. Clean them before scoring.

Remove:

- Duplicates
- Keywords with no search volume
- Obvious irrelevant terms
- Brand-only terms unless you want comparison content
- Local-only keywords unless location content is part of your strategy
- Keywords with mismatched intent, like “jobs,” “pdf,” “calculator,” or “near me”

Normalize similar variants:

energy efficient home improvements
energy-efficient home improvements
energy efficient improvements for homes

Keep one canonical version, but store the variants for secondary keyword use.

7. Enrich all candidates with search volume and ad competition

Use:

POST /v3/keywords_data/google_ads/search_volume/live

This endpoint gives search volume, monthly searches, competition, and other related data for up to 1,000 keywords in one request. It also returns competition, competition_index, cpc, low_top_of_page_bid, and high_top_of_page_bid.

Important: DataForSEO’s competition and competition_index are paid SERP metrics, not organic SEO difficulty. competition can be HIGH, MEDIUM, or LOW, while competition_index ranges from 0 to 100.

Batch your candidate keywords in groups of 1,000.

8. Add Keyword Difficulty

Use:

POST /v3/dataforseo_labs/google/bulk_keyword_difficulty/live

Batch keywords in groups of 1,000.

You want:

Low KD + high ad competition + enough search volume + article-friendly intent

Suggested first-pass filters:

KD <= 30
competition_index >= 60
search_volume >= 50
cpc > 0

For a newer site, be stricter:

KD <= 20
competition_index >= 50
search_volume >= 20

For a stronger site:

KD <= 40
competition_index >= 60
search_volume >= 100
9. Score the keywords

Use a scoring model like this:

Opportunity Score =
  35% low KD
+ 25% high ad competition
+ 15% search volume
+ 10% CPC / top-of-page bid
+ 10% positive trend
+ 5% topical fit

Example formula:

score =
  0.35 * (100 - KD)
+ 0.25 * competition_index
+ 0.15 * normalized_search_volume
+ 0.10 * normalized_cpc
+ 0.10 * trend_score
+ 0.05 * topical_fit

Where:

KD = lower is better
competition_index = higher is better
search_volume = higher is better, but use log scale
CPC = higher usually means stronger commercial value
trend_score = positive, flat, or declining
topical_fit = your own 1–100 judgment

Do not pick purely by volume. A 70-volume keyword with KD 12 and competition index 90 may be better than a 2,000-volume keyword with KD 67.

10. Cluster before choosing the article title

Group keywords into clusters by intent.

Example cluster:

Primary candidate:
energy efficient upgrades that increase home value

Secondary keywords:
do energy efficient windows increase home value
best green home improvements for resale
home upgrades that lower energy bills
green renovations ROI
energy efficient home improvements tax credit

This becomes one article.

Bad approach:

Write one article per keyword.

Better approach:

Write one article per intent cluster.
11. Validate the top 20–50 keywords with SERP API

Use:

POST /v3/serp/google/organic/live/advanced

Check:

- Are the top results blog posts, guides, and informational articles?
- Or are they calculators, ecommerce pages, government pages, local packs, or product pages?
- Are weak sites ranking?
- Are forums like Reddit/Quora ranking?
- Are the top pages outdated?
- Is there a featured snippet you can target?

Reject a keyword if the SERP intent does not match your article format.

Example:

Keyword: heat pump tax credit
SERP intent: government / tax-credit eligibility
Article possible: yes, but needs accuracy and citations.

Keyword: buy heat pump near me
SERP intent: local/commercial
Article possible: no, not ideal for informational blog content.
12. Pick the article angle, not just the keyword

Your final article title should be based on the best cluster, not necessarily the exact highest-scoring keyword.

Example:

Best keyword by score:

energy efficient upgrades that increase home value

Better article title:

7 Energy-Efficient Home Upgrades That Can Increase Resale Value

Supporting keywords:

green home improvements resale value
energy efficient windows home value
heat pump ROI
solar panels increase home value
smart thermostat home value

This satisfies your goal: the article is directionally in the topic area but optimized around the better keyword opportunity.

Practical workflow summary
1. Input article topic/title
2. Generate 10–30 seed phrases
3. Use Keyword Ideas for directional expansion
4. Use Related Keywords for adjacent searches
5. Use Google Ads Keywords For Keywords for commercial suggestions
6. Use Keyword Suggestions for exact long-tail expansion
7. Deduplicate and clean
8. Enrich with Google Ads Search Volume API
9. Add Bulk Keyword Difficulty
10. Filter for low KD + high ad competition
11. Score opportunities
12. Cluster by intent
13. Validate top clusters with SERP API
14. Choose one primary keyword + supporting keywords
15. Write article title around the cluster, not exact seed topic
My recommended default thresholds

For your type of content site, I would start with:

KD: 0–30
Ad competition index: 60–100
Search volume: 50+
CPC: greater than 0
Intent: informational or commercial-investigation
SERP: blog/article results visible in top 10

The sweet spot is:

KD <= 25
competition_index >= 70
search_volume >= 100
CPC >= $1
clear article intent

That is the zone where you are most likely to find keywords with organic ranking potential and commercial value.