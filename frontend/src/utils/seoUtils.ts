/**
 * Utility for SEO (Search Engine Optimization) and GEO (Generative Engine Optimization) analysis.
 * GEO in this context refers to optimizing content for Generative AI search engines (LLMs).
 */

export interface SEOResults {
    score: number;
    grade: 'A' | 'B' | 'C' | 'D' | 'F';
    checks: { label: string; passed: boolean }[];
    canPublish: boolean;
}

/**
 * Computes Generative Engine Optimization (GEO) context based on primary keyword.
 * Focuses on citation-readiness, entity density, and clear answer structure.
 */
export function computeGEOContext(primaryKeyword: string, _domain?: string) {
    const kw = (primaryKeyword || '').toLowerCase();
    
    // Check for high-intent Generative signals (questions, comparisons, definitions)
    const hasAnswerSignal = /^(what|how|why|best|vs|difference|compare|list|benefits)/i.test(kw);
    const hasDataSignal = /(statistics|data|research|study|facts|numbers)/i.test(kw);
    
    return {
        hasGEOSignal: hasAnswerSignal || hasDataSignal,
        optimizationFocus: hasAnswerSignal ? 'Direct Answer & Structure' : (hasDataSignal ? 'Data & Citations' : 'Core Entity Clarity'),
        directive: `
### GENERATIVE ENGINE OPTIMIZATION (GEO) DIRECTIVE
1. **Direct Answer Injection**: Explicitly answer the core query "${primaryKeyword}" in the first 2 paragraphs.
2. **Entity Density**: Use specific nomenclature and related concepts naturally (Latent Semantic Indexing).
3. **Citation-Ready Structure**: Use clear headings, bulleted lists, and structured data triggers (e.g., "In summary", "The key data points are").
4. **Conversational Flow**: Ensure the content is readable by LLMs as a high-authority source for synthetic answers.
`.trim()
    };
}

/**
 * Computes a publication readiness score based on SEO and GEO principles.
 */
export function computeSEOQualityScore(articleData: any): SEOResults {
    let score = 0;
    const checks: { label: string; passed: boolean }[] = [];

    const title = articleData.Title ?? articleData.title ?? '';
    const primaryKw = articleData.primary_keyword ?? articleData.primary_keywords?.[0] ?? articleData.search_phrase ?? '';
    const secondaryKws = articleData.secondary_keywords ?? [];
    const metaDesc = articleData.seo_meta_desc_optimized ?? articleData.metaDescription ?? '';
    const focusKw = articleData.focus_keyword ?? primaryKw;
    const wpCategoryId = articleData.wordpress_category_id;

    // 1. Primary Keyword Presence (30 pts)
    const hasKw = primaryKw.length > 0;
    if (hasKw) score += 30;
    checks.push({ label: 'Primary Keyword Set', passed: hasKw });

    // 2. Keyword in Title (20 pts)
    const kwInTitle = title.toLowerCase().includes(primaryKw.toLowerCase()) && primaryKw.length > 0;
    if (kwInTitle) score += 20;
    checks.push({ label: 'Keyword in Title', passed: kwInTitle });

    // 3. Meta Description Quality (15 pts)
    const metaOk = metaDesc.length >= 120 && metaDesc.length <= 160;
    if (metaOk) score += 15;
    checks.push({ label: 'Meta Description Length (120-160 chars)', passed: metaOk });

    // 4. Generative Engine Optimization (GEO) Readiness (15 pts)
    // Secondary KWs ≥3 feed semantic clustering for AI citation engines; research modes increase authority signals.
    const hasGeoData = secondaryKws.length >= 3 || articleData.claims_research_enabled || articleData.rag_enabled;
    if (hasGeoData) score += 15;
    checks.push({ label: 'GEO Readiness — AI Citation-Ready (Secondary KWs ≥3 or Research)', passed: hasGeoData });

    // 5. Taxonomy Mapping (WP Category) (10 pts)
    const hasCategory = !!wpCategoryId;
    if (hasCategory) score += 10;
    checks.push({ label: 'WordPress Category Linked', passed: hasCategory });

    // 6. Focus Keyword defined for Export (10 pts)
    const hasFocus = !!focusKw;
    if (hasFocus) score += 10;
    checks.push({ label: 'Focus Keyword Defined', passed: hasFocus });

    // Final Grade
    let grade: 'A' | 'B' | 'C' | 'D' | 'F' = 'F';
    if (score >= 90) grade = 'A';
    else if (score >= 75) grade = 'B';
    else if (score >= 60) grade = 'C';
    else if (score >= 40) grade = 'D';

    return {
        score,
        grade,
        checks,
        canPublish: score >= 40
    };
}
