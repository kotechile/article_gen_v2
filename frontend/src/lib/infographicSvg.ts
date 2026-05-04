const SVG_NS = 'http://www.w3.org/2000/svg';
const XLINK_NS = 'http://www.w3.org/1999/xlink';
const ALLOWED_TAGS = new Set([
    'svg',
    'g',
    'path',
    'rect',
    'circle',
    'ellipse',
    'line',
    'polyline',
    'polygon',
    'text',
    'tspan',
    'defs',
    'filter',
    'linearGradient',
    'radialGradient',
    'stop',
    'feGaussianBlur',
    'feOffset',
    'feFlood',
    'feComposite',
    'feMerge',
    'feMergeNode',
    'feColorMatrix',
    'feBlend',
    'feDropShadow',
    'clipPath',
    'mask',
    'pattern',
    'use',
    'symbol',
    'marker',
    'title',
    'desc',
    'style'
]);

const ALLOWED_ATTRS = new Set([
    'xmlns',
    'xmlns:xlink',
    'viewBox',
    'width',
    'height',
    'x',
    'y',
    'dx',
    'dy',
    'cx',
    'cy',
    'r',
    'rx',
    'ry',
    'x1',
    'x2',
    'y1',
    'y2',
    'd',
    'points',
    'fill',
    'stroke',
    'stroke-width',
    'stroke-linecap',
    'stroke-linejoin',
    'stroke-dasharray',
    'stroke-dashoffset',
    'opacity',
    'fill-opacity',
    'stroke-opacity',
    'filter',
    'transform',
    'font-family',
    'font-size',
    'font-weight',
    'text-anchor',
    'dominant-baseline',
    'letter-spacing',
    'line-height',
    'class',
    'id',
    'role',
    'aria-label',
    'preserveAspectRatio',
    'gradientUnits',
    'gradientTransform',
    'filterUnits',
    'primitiveUnits',
    'offset',
    'stop-color',
    'stop-opacity',
    'stdDeviation',
    'in',
    'in2',
    'result',
    'type',
    'values',
    'flood-color',
    'flood-opacity',
    'color-interpolation-filters',
    'clip-path',
    'clipPathUnits',
    'mask',
    'maskUnits',
    'maskContentUnits',
    'patternUnits',
    'patternContentUnits',
    'patternTransform',
    'href',
    'xlink:href'
]);

const DANGEROUS_ATTR_PREFIXES = ['on'];
const DANGEROUS_VALUE_PATTERNS = [/javascript:/i, /data:text\/html/i, /url\s*\(\s*javascript:/i];

function utf8ToBase64(value: string): string {
    const bytes = new TextEncoder().encode(value);
    let binary = '';
    bytes.forEach((byte) => {
        binary += String.fromCharCode(byte);
    });
    return btoa(binary);
}

function base64ToUtf8(value: string): string {
    const binary = atob(value);
    const bytes = Uint8Array.from(binary, (char) => char.charCodeAt(0));
    return new TextDecoder().decode(bytes);
}

export function encodeSvgMarkup(svg: string): string {
    return utf8ToBase64(svg);
}

export function decodeSvgMarkup(encodedSvg: string): string {
    if (!encodedSvg) return '';
    try {
        return base64ToUtf8(encodedSvg);
    } catch {
        return '';
    }
}

function sanitizeAttributeValue(name: string, value: string): string {
    const trimmed = value.trim();
    if (!trimmed) return trimmed;

    const lowerName = name.toLowerCase();
    if (DANGEROUS_ATTR_PREFIXES.some((prefix) => lowerName.startsWith(prefix))) {
        return '';
    }

    if (DANGEROUS_VALUE_PATTERNS.some((pattern) => pattern.test(trimmed))) {
        return '';
    }

    return trimmed;
}

function sanitizeSvgElement(element: Element): Element | null {
    const tagName = element.tagName;
    if (!ALLOWED_TAGS.has(tagName)) {
        return null;
    }

    const clone = document.createElementNS(SVG_NS, tagName);

    Array.from(element.attributes).forEach((attr) => {
        if (!ALLOWED_ATTRS.has(attr.name)) return;

        const sanitizedValue = sanitizeAttributeValue(attr.name, attr.value);
        if (!sanitizedValue) return;

        if (attr.name === 'href') {
            clone.setAttributeNS(XLINK_NS, 'href', sanitizedValue);
            clone.setAttribute('href', sanitizedValue);
            return;
        }

        if (attr.name === 'xlink:href') {
            clone.setAttributeNS(XLINK_NS, 'xlink:href', sanitizedValue);
            clone.setAttribute('xlink:href', sanitizedValue);
            return;
        }

        clone.setAttribute(attr.name, sanitizedValue);
    });

    if (tagName === 'style') {
        clone.textContent = element.textContent || '';
        return clone;
    }

    Array.from(element.childNodes).forEach((child) => {
        if (child.nodeType === Node.TEXT_NODE) {
            clone.appendChild(document.createTextNode(child.textContent || ''));
            return;
        }

        if (child.nodeType !== Node.ELEMENT_NODE) return;
        const sanitizedChild = sanitizeSvgElement(child as Element);
        if (sanitizedChild) {
            clone.appendChild(sanitizedChild);
        }
    });

    return clone;
}

function extractSvgElement(svgMarkup: string): SVGSVGElement | null {
    const parser = new DOMParser();
    const parsed = parser.parseFromString(svgMarkup, 'image/svg+xml');
    const svgElement = parsed.documentElement as Element | null;

    if (!svgElement || svgElement.tagName !== 'svg') {
        return null;
    }

    return svgElement as unknown as SVGSVGElement;
}

export function sanitizeSvgMarkup(svgMarkup: string): string {
    const normalized = svgMarkup.trim();
    if (!normalized) return '';

    const svgElement = extractSvgElement(normalized);
    if (!svgElement) return '';

    const sanitized = sanitizeSvgElement(svgElement);
    if (!sanitized) return '';

    sanitized.setAttribute('xmlns', SVG_NS);
    if (!sanitized.getAttribute('viewBox')) {
        sanitized.setAttribute('viewBox', '0 0 800 450');
    }

    return sanitized.outerHTML;
}

export function materializeInfographicHtml(html: string): string {
    if (!html) return html;

    const parser = new DOMParser();
    const doc = parser.parseFromString(html, 'text/html');

    doc.querySelectorAll<HTMLElement>('div[data-infographic="true"][data-svg]').forEach((node) => {
        const encodedSvg = node.getAttribute('data-svg') || '';
        const decoded = decodeSvgMarkup(encodedSvg);
        
        // Sanitize the SVG to ensure it's clean before embedding
        const sanitizedSvg = sanitizeSvgMarkup(decoded);
        if (!sanitizedSvg) return;

        // Use Data URI to preserve the exact SVG structure and styles when sending to WordPress
        // This prevents WP sanitization (kses) from stripping SVG elements/styles.
        const finalBase64 = encodeSvgMarkup(sanitizedSvg);
        const dataUri = `data:image/svg+xml;base64,${finalBase64}`;

        node.innerHTML = `<img src="${dataUri}" alt="Infographic" style="width: 100%; height: auto; display: block; margin: 0 auto; border-radius: 16px; box-shadow: 0 20px 50px rgba(0,0,0,0.15);" />`;
        
        node.classList.add('zenith-infographic-container');
        node.style.margin = '2rem auto';
        node.style.maxWidth = '900px';
        node.style.textAlign = 'center';
        node.style.display = 'block';
    });

    return doc.body.innerHTML;
}

export function normalizeInfographicHtmlForEditor(html: string): string {
    if (!html) return html;

    const parser = new DOMParser();
    const doc = parser.parseFromString(html, 'text/html');
    unwrapBlockElementsFromParagraphs(doc);
    repairMalformedTrailingTableContent(doc);

    doc.querySelectorAll<HTMLElement>('div.zenith-infographic-container').forEach((node) => {
        if (node.dataset.infographic === 'true' && node.dataset.svg) return;

        const inlineSvg = node.querySelector('svg');
        if (!inlineSvg) return;

        const sanitizedSvg = sanitizeSvgMarkup(inlineSvg.outerHTML);
        if (!sanitizedSvg) return;

        node.setAttribute('data-infographic', 'true');
        node.setAttribute('data-svg', encodeSvgMarkup(sanitizedSvg));
        node.innerHTML = '';
    });

    return doc.body.innerHTML;
}

function hasMeaningfulNodeContent(node: ChildNode): boolean {
    if (node.nodeType === Node.TEXT_NODE) {
        return Boolean(node.textContent?.trim());
    }

    if (!(node instanceof HTMLElement)) {
        return false;
    }

    if (node.tagName === 'BR') {
        return false;
    }

    return Boolean(node.textContent?.trim() || node.children.length > 0);
}

function hasMeaningfulCellContent(cell: HTMLTableCellElement): boolean {
    return Array.from(cell.childNodes).some(hasMeaningfulNodeContent);
}

function cellContainsHeadingLikeContent(cell: HTMLTableCellElement): boolean {
    return Array.from(cell.childNodes).some((node) => {
        if (!(node instanceof HTMLElement)) return false;
        if (/^H[1-6]$/.test(node.tagName)) return true;
        return Boolean(node.querySelector('h1,h2,h3,h4,h5,h6'));
    });
}

function wrapLooseBlockSiblingsInParagraphs(doc: Document, nodes: ChildNode[]): ChildNode[] {
    const normalized: ChildNode[] = [];
    let inlineBuffer: ChildNode[] = [];

    const flushInlineBuffer = () => {
        if (!inlineBuffer.some(hasMeaningfulNodeContent)) {
            inlineBuffer = [];
            return;
        }

        const paragraph = doc.createElement('p');
        inlineBuffer.forEach((node) => paragraph.appendChild(node));
        normalized.push(paragraph);
        inlineBuffer = [];
    };

    nodes.forEach((node) => {
        if (node instanceof HTMLElement && /^H[1-6]$/.test(node.tagName)) {
            flushInlineBuffer();
            normalized.push(node);
            return;
        }

        inlineBuffer.push(node);
    });

    flushInlineBuffer();
    return normalized;
}

function repairMalformedTrailingTableContent(doc: Document): void {
    doc.querySelectorAll('table').forEach((table) => {
        const rows = Array.from(table.rows);
        if (rows.length === 0) return;

        const lastRow = rows[rows.length - 1];
        const cells = Array.from(lastRow.cells);
        const sourceCellIndex = cells.findIndex(cellContainsHeadingLikeContent);
        if (sourceCellIndex === -1) return;

        const sourceCell = cells[sourceCellIndex];
        const trailingCells = cells.slice(sourceCellIndex + 1);
        if (trailingCells.some(hasMeaningfulCellContent)) return;

        const childNodes = Array.from(sourceCell.childNodes);
        const splitIndex = childNodes.findIndex(
            (node) =>
                node instanceof HTMLElement &&
                (/^H[1-6]$/.test(node.tagName) || Boolean(node.querySelector('h1,h2,h3,h4,h5,h6'))),
        );

        if (splitIndex <= 0) return;

        const trailingNodes = childNodes.slice(splitIndex);
        if (!trailingNodes.some(hasMeaningfulNodeContent)) return;

        trailingNodes.forEach((node) => node.remove());

        const normalizedTrailingNodes = wrapLooseBlockSiblingsInParagraphs(doc, trailingNodes);
        if (normalizedTrailingNodes.length === 0) return;

        const fragment = doc.createDocumentFragment();
        normalizedTrailingNodes.forEach((node) => fragment.appendChild(node));
        table.parentNode?.insertBefore(fragment, table.nextSibling);
    });
}

function unwrapBlockElementsFromParagraphs(doc: Document): void {
    const disallowedBlockTags = new Set([
        'TABLE', 'DIV', 'SECTION', 'ARTICLE', 'ASIDE', 'FIGURE', 'BLOCKQUOTE',
        'UL', 'OL', 'DL', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'PRE',
    ]);

    doc.querySelectorAll('p').forEach((paragraph) => {
        const childNodes = Array.from(paragraph.childNodes);
        const hasDisallowedBlockChild = childNodes.some(
            (node) => node instanceof HTMLElement && disallowedBlockTags.has(node.tagName),
        );
        if (!hasDisallowedBlockChild) return;

        const fragment = doc.createDocumentFragment();
        let inlineBuffer: ChildNode[] = [];

        const flushInlineBuffer = () => {
            const hasMeaningfulContent = inlineBuffer.some((node) => {
                if (node.nodeType === Node.TEXT_NODE) {
                    return Boolean(node.textContent?.trim());
                }
                return true;
            });
            if (!hasMeaningfulContent) {
                inlineBuffer = [];
                return;
            }

            const nextParagraph = doc.createElement('p');
            inlineBuffer.forEach((node) => nextParagraph.appendChild(node));
            fragment.appendChild(nextParagraph);
            inlineBuffer = [];
        };

        childNodes.forEach((node) => {
            if (node instanceof HTMLElement && disallowedBlockTags.has(node.tagName)) {
                flushInlineBuffer();
                fragment.appendChild(node);
                return;
            }
            inlineBuffer.push(node);
        });

        flushInlineBuffer();
        paragraph.replaceWith(fragment);
    });
}

/**
 * Applies premium inline styling to HTML tables to ensure they look excellent 
 * when exported to WordPress, bypassing default plain theme styles.
 */
export function beautifyTablesHtml(html: string): string {
    if (!html) return html;

    const parser = new DOMParser();
    const doc = parser.parseFromString(html, 'text/html');
    unwrapBlockElementsFromParagraphs(doc);
    repairMalformedTrailingTableContent(doc);

    doc.querySelectorAll('table').forEach((table) => {
        // Essential layout
        table.style.borderCollapse = 'separate';
        table.style.borderSpacing = '0';
        table.style.width = '100%';
        table.style.margin = '2.5rem 0';
        table.style.border = '1px solid #e5e7eb';
        table.style.borderRadius = '12px';
        table.style.overflow = 'hidden';
        table.style.boxShadow = '0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.02)';
        
        // Header styling
        table.querySelectorAll('th').forEach((th) => {
            th.style.backgroundColor = '#f8fafc';
            th.style.color = '#1e293b';
            th.style.fontWeight = '700';
            th.style.padding = '16px 20px';
            th.style.borderBottom = '2px solid #e2e8f0';
            th.style.textAlign = 'left';
            th.style.fontSize = '0.875rem';
            th.style.letterSpacing = '0.025em';
        });

        // Cell styling
        table.querySelectorAll('td').forEach((td) => {
            td.style.padding = '14px 20px';
            td.style.borderBottom = '1px solid #f1f5f9';
            td.style.color = '#475569';
            td.style.fontSize = '0.925rem';
            td.style.lineHeight = '1.5';
            td.style.verticalAlign = 'top';
        });

        // Zebra striping and hover-like effect (simulated via background if needed, but keeping it clean)
        const rows = Array.from(table.rows);
        rows.forEach((row, index) => {
            if (index > 0 && index % 2 === 0) {
                row.style.backgroundColor = '#fcfcfd';
            }
            
            // Remove bottom border from last row cells to respect container radius
            if (index === rows.length - 1) {
                Array.from(row.cells).forEach(cell => {
                    cell.style.borderBottom = 'none';
                });
            }
        });
    });

    return doc.body.innerHTML;
}
