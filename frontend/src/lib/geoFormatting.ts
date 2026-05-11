const KEY_TAKEAWAYS_HEADING = /^key takeaways$/i;
const TAKEAWAY_PREFIX = /^key takeaway\s*:\s*/i;
const TAKEAWAY_META_PATTERNS = [
    /^claim extracted from:/i,
    /^outcome:/i,
    /generative ai engines prefer/i,
    /ai answer extraction/i,
    /definition-style openers/i,
    /use structured,\s*scannable sections/i,
    /title & description rewrite authorization/i,
    /original creative intent/i,
    /geo focus area detected/i,
    /primary keyword:/i,
    /secondary keywords/i,
    /\bis a practical topic shaped by\b/i,
    /\bso the best answer depends on your goals, constraints, and timing\b/i,
];

const normalizeTakeawayText = (value: string): string => {
    return value
        .replace(TAKEAWAY_PREFIX, '')
        .replace(/^[\-\u2022]\s*/, '')
        .replace(/\s+/g, ' ')
        .trim();
};

const isUsefulTakeaway = (value: string): boolean => {
    if (!value) return false;
    if (value.length < 40) return false;
    if (TAKEAWAY_META_PATTERNS.some((pattern) => pattern.test(value))) return false;
    if (/(?:\.\.\.|…)\s*(?:because|and adapt)/i.test(value)) return false;
    if (/(?:\.\.\.|…)\s*$/.test(value)) return false;
    return true;
};

const buildTakeawayCandidates = (articleData: any): string[] => {
    const candidates = [
        articleData?.thesis,
        articleData?.excerpt,
        articleData?.hook,
        articleData?.focus_keyword ? `This article is optimized around "${articleData.focus_keyword}".` : '',
    ]
        .map((item: unknown) => normalizeTakeawayText(String(item || '').trim()))
        .filter(isUsefulTakeaway);

    return Array.from(new Set(candidates)).slice(0, 4);
};

const createKeyTakeawaysSection = (doc: Document, takeaways: string[]): HTMLElement | null => {
    if (takeaways.length === 0) return null;

    const section = doc.createElement('section');
    section.className = 'geo-key-takeaways';
    section.setAttribute('data-geo-injected', 'key-takeaways');

    const heading = doc.createElement('h2');
    heading.textContent = 'Key Takeaways';
    section.appendChild(heading);

    const list = doc.createElement('ul');
    takeaways.forEach((item) => {
        const listItem = doc.createElement('li');
        listItem.textContent = item;
        list.appendChild(listItem);
    });
    section.appendChild(list);

    return section;
};

const extractTakeawayTexts = (container: ParentNode): string[] => {
    const contentNodes = Array.from(container.querySelectorAll('p, li'));
    const collected = contentNodes
        .map((node) => normalizeTakeawayText(node.textContent || ''))
        .filter(isUsefulTakeaway);

    return Array.from(new Set(collected)).slice(0, 5);
};

const extractExistingKeyTakeawaysSection = (doc: Document): HTMLElement | null => {
    const existingSection = doc.querySelector('section.geo-key-takeaways');
    if (existingSection instanceof HTMLElement) {
        const takeaways = extractTakeawayTexts(existingSection);
        existingSection.remove();
        if (takeaways.length === 0) return null;
        return createKeyTakeawaysSection(doc, takeaways);
    }

    const heading = Array.from(doc.body.querySelectorAll('h1, h2, h3')).find((node) =>
        KEY_TAKEAWAYS_HEADING.test((node.textContent || '').trim()),
    );

    if (!(heading instanceof HTMLElement) || !heading.parentElement) return null;

    const fragment = doc.createElement('div');
    const nodesToRemove: Element[] = [];
    let current: Element | null = heading;
    while (current) {
        const next: Element | null = current.nextElementSibling;
        fragment.appendChild(current.cloneNode(true));
        nodesToRemove.push(current);

        if (!next) break;
        if (/^H[1-3]$/i.test(next.tagName)) break;
        if (!/^(UL|OL|P)$/i.test(next.tagName)) break;

        current = next;
    }

    const takeaways = extractTakeawayTexts(fragment);
    nodesToRemove.forEach((node) => node.remove());
    if (takeaways.length === 0) return null;

    return createKeyTakeawaysSection(doc, takeaways);
};

export const ensureIntroKeyTakeaways = (html: string, articleData?: any): string => {
    if (!html.trim()) return html;

    const parser = new DOMParser();
    const doc = parser.parseFromString(html, 'text/html');
    const body = doc.body;
    if (!body) return html;

    let takeawaysSection = extractExistingKeyTakeawaysSection(doc);
    if (!takeawaysSection) {
        const fallbackTakeaways = buildTakeawayCandidates(articleData);
        takeawaysSection = createKeyTakeawaysSection(doc, fallbackTakeaways);
    }

    if (!takeawaysSection) return html;

    const h1 = Array.from(body.children).find(
        (node) => node !== takeawaysSection && /^H1$/i.test(node.tagName),
    );
    if (h1?.nextSibling) {
        body.insertBefore(takeawaysSection, h1.nextSibling);
    } else if (h1) {
        body.appendChild(takeawaysSection);
    } else {
        const firstMeaningfulNode = Array.from(body.children).find((node) => node !== takeawaysSection) || null;
        if (firstMeaningfulNode) {
            body.insertBefore(takeawaysSection, firstMeaningfulNode);
        } else {
            body.appendChild(takeawaysSection);
        }
    }

    return body.innerHTML;
};
