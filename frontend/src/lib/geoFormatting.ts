const KEY_TAKEAWAYS_HEADING = /^(?:at a glance|key takeaways|takeaways|tl;?dr)$/i;
const INTRODUCTION_HEADING = /^introduction\b/i;
const TAKEAWAY_PREFIX = /^(?:at a glance|key takeaway|takeaway|tl;?dr)\s*:\s*/i;
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
        .replace(/^[\s\-\u2022\*]+\s*/, '')
        .replace(/\s+/g, ' ')
        .trim();
};

export const formatTakeawayHtml = (value: string): string => {
    let clean = normalizeTakeawayText(value);
    clean = clean.replace(/^\s*<p[^>]*>(.*?)<\/p>\s*$/is, '$1').trim();
    clean = clean.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    clean = clean.replace(/__(.+?)__/g, '<strong>$1</strong>');
    clean = clean.replace(/(?<!\*)\*([^*]+?)\*(?!\*)/g, '<em>$1</em>');
    clean = clean.replace(/(?<!_)_([^_]+?)_(?!_)/g, '<em>$1</em>');
    return clean;
};

const isUsefulTakeaway = (value: string): boolean => {
    const plain = value.replace(/<[^>]+>/g, '').trim();
    if (!plain) return false;
    if (plain.length < 40) return false;
    if (TAKEAWAY_META_PATTERNS.some((pattern) => pattern.test(plain))) return false;
    if (/(?:\.\.\.|…)\s*(?:because|and adapt)/i.test(plain)) return false;
    if (/(?:\.\.\.|…)\s*$/.test(plain)) return false;
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

    const seenPlain = new Set<string>();
    const unique: string[] = [];
    for (const c of candidates) {
        const plain = c.replace(/<[^>]+>/g, '').replace(/\s+/g, ' ').trim().toLowerCase();
        if (plain && !seenPlain.has(plain)) {
            seenPlain.add(plain);
            unique.push(c);
        }
    }
    return unique.slice(0, 4);
};

export const splitTakeawayTextToMaxWords = (text: string, maxWords: number = 60): string[] => {
    const cleanText = (text || '').trim();
    if (!cleanText) return [];

    const plainWords = cleanText.replace(/<[^>]+>/g, ' ').trim().split(/\s+/).filter(Boolean);
    if (plainWords.length <= maxWords) {
        return [cleanText];
    }

    // Split into sentences using punctuation boundaries
    const sentences = cleanText.split(/(?<=[.!?])\s+/).map((s) => s.trim()).filter(Boolean);
    if (sentences.length === 0) return [cleanText];

    const chunks: string[] = [];
    let currentChunk: string[] = [];
    let currentCount = 0;

    for (const sentence of sentences) {
        const sWords = sentence.replace(/<[^>]+>/g, ' ').trim().split(/\s+/).filter(Boolean);
        if (sWords.length > maxWords) {
            if (currentChunk.length > 0) {
                chunks.push(currentChunk.join(' ').trim());
                currentChunk = [];
                currentCount = 0;
            }

            const clauseParts = sentence.split(/(?<=[;:,—–])\s+/).map((c) => c.trim()).filter(Boolean);
            if (clauseParts.length > 1) {
                let subChunk: string[] = [];
                let subCount = 0;
                for (const clause of clauseParts) {
                    const cWords = clause.replace(/<[^>]+>/g, ' ').trim().split(/\s+/).filter(Boolean);
                    if (subCount + cWords.length <= maxWords || subChunk.length === 0) {
                        subChunk.push(clause);
                        subCount += cWords.length;
                    } else {
                        let subText = subChunk.join(' ').trim();
                        if (!/[.!?]$/.test(subText)) subText += '.';
                        chunks.push(subText);
                        subChunk = [clause];
                        subCount = cWords.length;
                    }
                }
                if (subChunk.length > 0) {
                    let subText = subChunk.join(' ').trim();
                    if (!/[.!?]$/.test(subText)) subText += '.';
                    chunks.push(subText);
                }
            } else {
                for (let i = 0; i < sWords.length; i += maxWords) {
                    const chunkSlice = sWords.slice(i, i + maxWords);
                    let chunkText = chunkSlice.join(' ').trim();
                    if (!/[.!?]$/.test(chunkText)) chunkText += '.';
                    chunks.push(chunkText);
                }
            }
        } else if (currentCount + sWords.length <= maxWords) {
            currentChunk.push(sentence);
            currentCount += sWords.length;
        } else {
            chunks.push(currentChunk.join(' ').trim());
            currentChunk = [sentence];
            currentCount = sWords.length;
        }
    }

    if (currentChunk.length > 0) {
        chunks.push(currentChunk.join(' ').trim());
    }

    return chunks
        .map((c) => c.trim())
        .filter(Boolean)
        .map((c) => (/[.!?]$/.test(c) ? c : `${c}.`));
};

const createKeyTakeawaysSection = (doc: Document, takeaways: string[]): HTMLElement | null => {
    const expandedTakeaways: string[] = [];
    takeaways.forEach((item) => {
        const splits = splitTakeawayTextToMaxWords(item, 60);
        if (splits.length > 0) {
            expandedTakeaways.push(...splits);
        } else {
            expandedTakeaways.push(item);
        }
    });

    if (expandedTakeaways.length === 0) return null;

    const section = doc.createElement('section');
    section.className = 'geo-key-takeaways';
    section.setAttribute('data-geo-injected', 'key-takeaways');

    const heading = doc.createElement('h2');
    heading.textContent = 'At a glance';
    section.appendChild(heading);

    const list = doc.createElement('ul');
    expandedTakeaways.forEach((item) => {
        const listItem = doc.createElement('li');
        listItem.innerHTML = formatTakeawayHtml(item);
        list.appendChild(listItem);
    });
    section.appendChild(list);

    return section;
};

const extractTakeawayTexts = (container: ParentNode): string[] => {
    const listItems = Array.from(container.querySelectorAll('li'));
    const contentNodes = listItems.length > 0
        ? listItems
        : Array.from(container.querySelectorAll('p'));

    const seenPlain = new Set<string>();
    const uniqueTakeaways: string[] = [];

    for (const node of contentNodes) {
        let raw = node.innerHTML.includes('<') ? node.innerHTML : (node.textContent || '');
        // Strip any wrapping <p>...</p> tags
        raw = raw.replace(/^\s*<p[^>]*>(.*?)<\/p>\s*$/is, '$1').trim();
        const cleaned = normalizeTakeawayText(raw);
        if (!isUsefulTakeaway(cleaned)) continue;

        const splits = splitTakeawayTextToMaxWords(cleaned, 60);
        for (const splitItem of (splits.length > 0 ? splits : [cleaned])) {
            const plainKey = splitItem.replace(/<[^>]+>/g, '').replace(/\s+/g, ' ').trim().toLowerCase();
            if (plainKey && !seenPlain.has(plainKey)) {
                seenPlain.add(plainKey);
                uniqueTakeaways.push(splitItem);
            }
        }
    }

    return uniqueTakeaways.slice(0, 8);
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
    const nodesToRemove: Element[] = [heading];
    fragment.appendChild(heading.cloneNode(true));

    let next: Element | null = heading.nextElementSibling;
    let consumedList = false;

    while (next) {
        if (/^H[1-6]$/i.test(next.tagName)) {
            // Reached next section heading
            break;
        }

        if (next.tagName === 'UL' || next.tagName === 'OL') {
            fragment.appendChild(next.cloneNode(true));
            nodesToRemove.push(next);
            consumedList = true;
            // The takeaways list has been consumed; any subsequent paragraphs are article body
            break;
        }

        if (next.tagName === 'P') {
            const rawText = (next.textContent || '').trim();
            // Only consume <p> if it appears before a list and has takeaway prefix or bullet markers
            // or matches candidate takeaway patterns. Once a regular body paragraph is reached, stop.
            const isTakeawayPara =
                /^[\s\-\u2022\*]/.test(rawText) ||
                TAKEAWAY_PREFIX.test(rawText) ||
                TAKEAWAY_META_PATTERNS.some((pattern) => pattern.test(rawText));

            if (isTakeawayPara) {
                fragment.appendChild(next.cloneNode(true));
                nodesToRemove.push(next);
                next = next.nextElementSibling;
                continue;
            } else {
                // This is a normal body paragraph - do NOT remove it!
                break;
            }
        }

        // Any other element (img, div, figure, table, etc.) - stop immediately
        break;
    }

    const takeaways = extractTakeawayTexts(fragment);
    nodesToRemove.forEach((node) => node.remove());
    if (takeaways.length === 0) return null;

    return createKeyTakeawaysSection(doc, takeaways);
};

export const ensureIntroKeyTakeaways = (html: string, _articleData?: any): string => {
    if (!html.trim()) return html;

    const parser = new DOMParser();
    const doc = parser.parseFromString(html, 'text/html');
    const body = doc.body;
    if (!body) return html;

    const takeawaysSection = extractExistingKeyTakeawaysSection(doc);
    if (!takeawaysSection) return html;

    const leadingIntroHeading = Array.from(body.children).find((node) =>
        node !== takeawaysSection &&
        /^H[1-3]$/i.test(node.tagName) &&
        INTRODUCTION_HEADING.test((node.textContent || '').trim()),
    );
    if (leadingIntroHeading) {
        body.insertBefore(takeawaysSection, leadingIntroHeading);
        return body.innerHTML;
    }

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
