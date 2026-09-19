import type { Project } from '../types';
import type { EditorialArticle } from '../services/editorial-factory.service';

export interface DomainMatchResult {
    domain: string;
    project: Project | null;
    confidence: 'high' | 'medium' | 'low' | 'none';
    matchReason?: string;
    isAutoMatched: boolean;
}

// Built-in niche keyword semantic associations for popular domain themes
const NICHE_KEYWORDS: Record<string, string[]> = {
    home: [
        'home', 'house', 'roost', 'wellroost', 'reno', 'renovation', 'remodel', 'hvac', 'heat pump',
        'insulation', 'plumbing', 'roof', 'roofing', 'garden', 'backyard', 'diy', 'interior', 'appliance',
        'kitchen', 'bathroom', 'sealed homes', 'rebate', 'weatherization', 'lawn', 'framing', 'contractor'
    ],
    tech_ai: [
        'ai', 'artificial intelligence', 'llm', 'gpu', 'nvidia', 'openai', 'anthropic', 'chips',
        'semiconductor', 'agentic', 'agent loop', 'model', 'software', 'developer', 'algorithm',
        'hardware', 'cloud', 'giniloh', 'inference', 'neural', 'automation', 'saas', 'coding'
    ],
    supply_chain: [
        'supply chain', 'freight', 'shipping', 'cma cgm', 'carrier', 'port', 'logistics', 'delay fees',
        'container', 'reshoring', 'tariff', 'import', 'export', 'cargo', 'vessel', 'warehouse'
    ],
};

function normalizeText(text: string): string {
    return (text || '')
        .toLowerCase()
        .replace(/[^a-z0-9\s_-]/g, ' ')
        .replace(/[_-]/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();
}

function extractTokens(text: string): string[] {
    const norm = normalizeText(text);
    return norm.split(' ').filter(t => t.length >= 2);
}

/**
 * Score how well an Editorial Factory article aligns with a specific Project.
 */
export function scoreArticleProjectMatch(
    article: EditorialArticle,
    project: Project
): { score: number; reasons: string[] } {
    let score = 0;
    const reasons: string[] = [];

    const projectDomain = (project.domain || '').toLowerCase().trim();
    const projectName = (project.app_name || '').toLowerCase().trim();
    const projectDesc = (project.site_description || project.websiteDescription || project.websitedescription || '').toLowerCase();
    const projectCats = (project.categories || '').toLowerCase();
    const projectKeywords = (project.target_keywords || []).map(k => k.toLowerCase());

    const projectCorpus = `${projectDomain} ${projectName} ${projectDesc} ${projectCats} ${projectKeywords.join(' ')}`;
    const projectTokens = new Set(extractTokens(projectCorpus));

    // 1. Tag Matching (High Weight)
    const articleTags = (article.tags || []).map(t => normalizeText(t));
    for (const rawTag of articleTags) {
        const tagTokens = extractTokens(rawTag);
        
        // Exact tag in project corpus
        if (rawTag && projectCorpus.includes(rawTag)) {
            score += 8;
            reasons.push(`Tag "${rawTag}" aligns with project`);
            continue;
        }

        // Sub-tokens of tag in project tokens
        for (const token of tagTokens) {
            if (projectTokens.has(token)) {
                score += 4;
                reasons.push(`Tag keyword "${token}" matches project`);
            }
        }

        // Check semantic niche keywords
        // Check Home / Wellroost match
        if (
            (projectDomain.includes('roost') || projectDomain.includes('home') || projectCorpus.includes('home') || projectCorpus.includes('reno')) &&
            NICHE_KEYWORDS.home.some(kw => rawTag.includes(kw) || kw.includes(rawTag))
        ) {
            score += 10;
            reasons.push(`Tag "${rawTag}" matches home niche (${projectDomain || projectName})`);
        }

        // Check Tech / AI / Giniloh match
        if (
            (projectDomain.includes('gini') || projectCorpus.includes('ai') || projectCorpus.includes('tech')) &&
            NICHE_KEYWORDS.tech_ai.some(kw => rawTag.includes(kw) || kw.includes(rawTag))
        ) {
            score += 10;
            reasons.push(`Tag "${rawTag}" matches AI/Tech niche (${projectDomain || projectName})`);
        }

        // Check Supply Chain match
        if (
            (projectCorpus.includes('supply') || projectCorpus.includes('logistics') || projectCorpus.includes('business')) &&
            NICHE_KEYWORDS.supply_chain.some(kw => rawTag.includes(kw) || kw.includes(rawTag))
        ) {
            score += 10;
            reasons.push(`Tag "${rawTag}" matches supply chain niche`);
        }
    }

    // 2. Title Match (Medium Weight)
    const titleNorm = normalizeText(article.title || '');
    const titleTokens = extractTokens(titleNorm);

    // Check title tokens against project tokens
    for (const token of titleTokens) {
        if (token.length >= 3 && projectTokens.has(token)) {
            score += 2.5;
            reasons.push(`Title word "${token}" matches project`);
        }
    }

    // Semantic title check for Home
    if (
        (projectDomain.includes('roost') || projectDomain.includes('home') || projectCorpus.includes('home') || projectCorpus.includes('reno')) &&
        NICHE_KEYWORDS.home.some(kw => titleNorm.includes(kw))
    ) {
        const matchedKw = NICHE_KEYWORDS.home.find(kw => titleNorm.includes(kw));
        score += 7;
        reasons.push(`Title mentions "${matchedKw}"`);
    }

    // Semantic title check for Tech/AI
    if (
        (projectDomain.includes('gini') || projectCorpus.includes('ai') || projectCorpus.includes('tech')) &&
        NICHE_KEYWORDS.tech_ai.some(kw => titleNorm.includes(kw))
    ) {
        const matchedKw = NICHE_KEYWORDS.tech_ai.find(kw => titleNorm.includes(kw));
        score += 7;
        reasons.push(`Title mentions "${matchedKw}"`);
    }

    // 3. Summary / Hook Match (Low-Medium Weight)
    const summaryNorm = normalizeText(article.summary || article.hook || '');
    if (summaryNorm) {
        if (
            (projectDomain.includes('roost') || projectDomain.includes('home') || projectCorpus.includes('home')) &&
            NICHE_KEYWORDS.home.some(kw => summaryNorm.includes(kw))
        ) {
            score += 3;
        }
        if (
            (projectDomain.includes('gini') || projectCorpus.includes('ai') || projectCorpus.includes('tech')) &&
            NICHE_KEYWORDS.tech_ai.some(kw => summaryNorm.includes(kw))
        ) {
            score += 3;
        }
    }

    return { score, reasons };
}

/**
 * Finds the best matching project domain for a given Editorial Article among all available projects.
 */
export function findBestMatchingDomain(
    article: EditorialArticle,
    projects: Project[],
    fallbackDomain?: string
): DomainMatchResult {
    if (!projects || projects.length === 0) {
        return {
            domain: fallbackDomain || '',
            project: null,
            confidence: 'none',
            matchReason: 'No projects configured',
            isAutoMatched: false,
        };
    }

    let highestScore = 0;
    let bestProject: Project | null = null;
    let bestReasons: string[] = [];

    for (const project of projects) {
        const { score, reasons } = scoreArticleProjectMatch(article, project);
        if (score > highestScore) {
            highestScore = score;
            bestProject = project;
            bestReasons = reasons;
        }
    }

    // Thresholds
    if (highestScore >= 8 && bestProject) {
        const primaryReason = bestReasons[0] || 'Matches project topics';
        return {
            domain: bestProject.domain || bestProject.app_name || '',
            project: bestProject,
            confidence: highestScore >= 14 ? 'high' : 'medium',
            matchReason: primaryReason,
            isAutoMatched: true,
        };
    }

    // Weak match threshold
    if (highestScore >= 4 && bestProject) {
        return {
            domain: bestProject.domain || bestProject.app_name || '',
            project: bestProject,
            confidence: 'low',
            matchReason: bestReasons[0] || 'Partial topic similarity',
            isAutoMatched: true,
        };
    }

    // If fallback domain matches one of the projects, use it
    const fallbackProject = projects.find(p => (p.domain || p.app_name) === fallbackDomain) || projects[0];
    return {
        domain: fallbackDomain || (fallbackProject ? (fallbackProject.domain || fallbackProject.app_name || '') : ''),
        project: fallbackProject || null,
        confidence: 'none',
        matchReason: 'Default active project',
        isAutoMatched: false,
    };
}
