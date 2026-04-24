import { Node, mergeAttributes } from '@tiptap/core';
import { decodeSvgMarkup, sanitizeSvgMarkup } from '../../lib/infographicSvg';

function buildNodeDom(svg: string, loading: boolean): HTMLDivElement {
    const dom = document.createElement('div');
    dom.className = 'zenith-infographic-container not-prose';

    if (loading) {
        dom.classList.add('is-loading');
        dom.innerHTML = `
            <div class="zenith-infographic-loading-shell">
                <div class="zenith-infographic-loading-spinner" aria-hidden="true"></div>
                <div class="zenith-infographic-loading-copy">
                    <strong>Generating infographic...</strong>
                    <span>This usually takes a few seconds.</span>
                </div>
            </div>
        `;
        return dom;
    }

    const sanitizedSvg = sanitizeSvgMarkup(decodeSvgMarkup(svg));
    if (sanitizedSvg) {
        dom.innerHTML = sanitizedSvg;
    } else {
        dom.innerHTML = `
            <div class="zenith-infographic-error">
                <strong>Infographic unavailable</strong>
                <span>The SVG could not be rendered.</span>
            </div>
        `;
    }

    return dom;
}

export const InfographicBlock = Node.create({
    name: 'infographicBlock',
    group: 'block',
    atom: true,
    selectable: true,
    draggable: true,

    addAttributes() {
        return {
            svg: {
                default: '',
                parseHTML: (element) => element.getAttribute('data-svg') || '',
                renderHTML: (attributes) => ({ 'data-svg': attributes.svg || '' }),
            },
            requestId: {
                default: null,
                parseHTML: (element) => element.getAttribute('data-request-id'),
                renderHTML: (attributes) => {
                    if (!attributes.requestId) return {};
                    return { 'data-request-id': attributes.requestId };
                },
            },
            loading: {
                default: false,
                parseHTML: (element) => element.getAttribute('data-loading') === 'true',
                renderHTML: (attributes) => {
                    if (!attributes.loading) return {};
                    return { 'data-loading': 'true' };
                },
            },
        };
    },

    parseHTML() {
        return [
            {
                tag: 'div[data-infographic="true"][data-svg]',
            },
        ];
    },

    renderHTML({ HTMLAttributes }) {
        return [
            'div',
            mergeAttributes(HTMLAttributes, {
                'data-infographic': 'true',
                class: 'zenith-infographic-container',
            }),
        ];
    },

    addNodeView() {
        return ({ node }) => {
            const dom = buildNodeDom(node.attrs.svg || '', !!node.attrs.loading);
            return { dom };
        };
    },
});
