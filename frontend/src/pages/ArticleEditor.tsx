
import React, { useEffect, useRef, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { supabase } from '../lib/supabase';
import { useAuth } from '../context/auth-context';

import { useEditor, EditorContent } from '@tiptap/react';
import { Extension, Node, mergeAttributes, InputRule } from '@tiptap/core';
import BulletList from '@tiptap/extension-bullet-list';
import StarterKit from '@tiptap/starter-kit';
import katex from 'katex';
import 'katex/dist/katex.min.css';
import TiptapImage from '@tiptap/extension-image';
import Link from '@tiptap/extension-link';
import { Table } from '@tiptap/extension-table';
import { TableCell } from '@tiptap/extension-table-cell';
import { TableHeader } from '@tiptap/extension-table-header';
import { TableRow } from '@tiptap/extension-table-row';
import CharacterCount from '@tiptap/extension-character-count';
import { ArrowLeft, Save, Bold, Italic, Heading2, Heading3, Link as LinkIcon, Image as ImageIcon, Loader2, Table as TableIcon, Trash2, Plus, RefreshCw, ListOrdered, Globe, List, BarChart3, Link2, Filter, ChartColumn, Sigma, Wand2, Share2 } from 'lucide-react';
import { apiClient } from '../api-client';
import { assembleArticleHtml } from '../lib/contentParser';
import { AddImageModal } from '../components/AddImageModal';
import { ReferenceSelector } from '../components/ReferenceSelector';
import { WordPressExportModal } from '../components/WordPressExportModal';
import { LinkedInPublishModal } from '../components/LinkedInPublishModal';
import { Gauge } from '../components/Gauge';
import { METRIC_EXPLANATIONS } from '../types/metrics';
import { MetricTooltip } from '../components/Tooltip';
import type { ImageMetadata } from '../types/image';
import { rankCitationDomains } from '../lib/citationAuthority';
import { InfographicBlock } from '../components/editor/InfographicBlock';
import { materializeInfographicHtml, normalizeInfographicHtmlForEditor, beautifyTablesHtml } from '../lib/infographicSvg';
import { ensureIntroKeyTakeaways } from '../lib/geoFormatting';


const HeadingIdExtension = Extension.create({
    name: 'headingId',
    addGlobalAttributes() {
        return [
            {
                types: ['heading'],
                attributes: {
                    id: {
                        default: null,
                        parseHTML: element => element.id,
                        renderHTML: attributes => {
                            if (!attributes.id) return {};
                            return { id: attributes.id };
                        },
                    },
                },
            },
        ];
    },
});

const CustomBulletList = BulletList.extend({
    addAttributes() {
        return {
            class: {
                default: 'list-disc ml-4 space-y-1',
                parseHTML: element => element.getAttribute('class'),
                renderHTML: attributes => {
                    return { class: attributes.class || 'list-disc ml-4 space-y-1' }
                },
            }
        }
    }
});

const CustomImage = TiptapImage.extend({
    addAttributes() {
        return {
            ...this.parent?.(),
            width: {
                default: '100%',
                renderHTML: attributes => {
                    if (!attributes.width) return {};
                    const widthStyle = `width: ${attributes.width}; max-width: 100%;`;
                    let floatStyle = '';
                    let marginStyle = 'margin-top: 2rem; margin-bottom: 2rem;';
                    let displayStyle = 'display: block; margin-left: auto; margin-right: auto;';

                    if (attributes.alignment === 'left') {
                        floatStyle = 'float: left;';
                        marginStyle = 'margin-right: 1.5rem; margin-bottom: 1.5rem; margin-top: 0.5rem;';
                        displayStyle = 'display: inline-block;';
                    } else if (attributes.alignment === 'right') {
                        floatStyle = 'float: right;';
                        marginStyle = 'margin-left: 1.5rem; margin-bottom: 1.5rem; margin-top: 0.5rem;';
                        displayStyle = 'display: inline-block;';
                    }

                    return {
                        style: `${widthStyle} ${floatStyle} ${marginStyle} ${displayStyle}`
                    };
                },
                parseHTML: element => {
                    return element.getAttribute('width') || element.style.width || '100%';
                },
            },
            alignment: {
                default: 'center',
                renderHTML: attributes => {
                    if (attributes.alignment === 'left') {
                        return { class: 'align-left' };
                    }
                    if (attributes.alignment === 'right') {
                        return { class: 'align-right' };
                    }
                    return { class: 'align-center' };
                },
                parseHTML: element => {
                    if (element.classList.contains('align-left') || element.style.float === 'left') return 'left';
                    if (element.classList.contains('align-right') || element.style.float === 'right') return 'right';
                    return 'center';
                },
            },
            link: {
                default: null,
                parseHTML: element => {
                    const anchor = element.closest('a');
                    if (anchor) {
                        return anchor.getAttribute('href');
                    }
                    return element.getAttribute('data-link') || null;
                },
                renderHTML: attributes => {
                    if (!attributes.link) return {};
                    return { 'data-link': attributes.link };
                }
            }
        };
    },

    renderHTML({ HTMLAttributes }) {
        const { 'data-link': link, ...rest } = HTMLAttributes;
        const imgElement = ['img', mergeAttributes(this.options.HTMLAttributes, rest)] as any;
        if (link) {
            return ['a', { href: link, target: '_blank', rel: 'noopener noreferrer', class: 'image-link' }, imgElement] as any;
        }
        return imgElement;
    },
});

export const MathNode = Node.create({
    name: 'math',
    group: 'inline',
    inline: true,
    selectable: true,
    atom: true,

    addAttributes() {
        return {
            latex: {
                default: '',
            },
            displayMode: {
                default: false,
            },
        };
    },

    parseHTML() {
        return [
            {
                tag: 'span[data-math]',
                getAttrs: element => {
                    const el = element as HTMLElement;
                    return {
                        latex: el.getAttribute('data-math') || '',
                        displayMode: el.getAttribute('data-display-mode') === 'true',
                    };
                },
            },
            {
                tag: '.katex',
                getAttrs: element => {
                    const el = element as HTMLElement;
                    
                    // 1. Try annotation tag
                    const annotation = el.querySelector('annotation[encoding="application/x-tex"]');
                    if (annotation && annotation.textContent) {
                        const isDisplay = el.querySelector('.katex-display') !== null || el.classList.contains('katex-display');
                        return {
                            latex: annotation.textContent.trim(),
                            displayMode: isDisplay,
                        };
                    }
                    
                    // 2. Try any annotation tag fallback
                    const anyAnnotation = el.querySelector('annotation');
                    if (anyAnnotation && anyAnnotation.textContent) {
                        const isDisplay = el.querySelector('.katex-display') !== null || el.classList.contains('katex-display');
                        return {
                            latex: anyAnnotation.textContent.trim(),
                            displayMode: isDisplay,
                        };
                    }
                    
                    // 3. Try math alttext
                    const mathEl = el.querySelector('math');
                    if (mathEl) {
                        const alt = mathEl.getAttribute('alttext');
                        if (alt) {
                            const isDisplay = el.querySelector('.katex-display') !== null || el.classList.contains('katex-display');
                            return {
                                latex: alt.trim(),
                                displayMode: isDisplay,
                            };
                        }
                    }
                    
                    // 4. Try aria-label
                    const ariaLabel = el.getAttribute('aria-label');
                    if (ariaLabel) {
                        const isDisplay = el.querySelector('.katex-display') !== null || el.classList.contains('katex-display');
                        return {
                            latex: ariaLabel.trim(),
                            displayMode: isDisplay,
                        };
                    }
                    
                    return false;
                },
            },
            {
                tag: 'mjx-container',
                getAttrs: element => {
                    const el = element as HTMLElement;
                    const tex = el.getAttribute('data-tex');
                    if (tex) {
                        const isDisplay = el.getAttribute('display') === 'true' || el.classList.contains('mjx-display') || el.style.display === 'block';
                        return {
                            latex: tex.trim(),
                            displayMode: isDisplay,
                        };
                    }
                    return false;
                },
            },
            {
                tag: 'script[type^="math/tex"]',
                getAttrs: element => {
                    const el = element as HTMLScriptElement;
                    const type = el.getAttribute('type') || '';
                    const isDisplay = type.includes('mode=display');
                    return {
                        latex: el.textContent?.trim() || '',
                        displayMode: isDisplay,
                    };
                },
            },
            {
                tag: '.MathJax',
                getAttrs: element => {
                    const el = element as HTMLElement;
                    
                    // 1. Try data-tex attribute (sometimes present on wrapper)
                    const dataTex = el.getAttribute('data-tex');
                    if (dataTex) {
                        return {
                            latex: dataTex.trim(),
                            displayMode: el.classList.contains('MathJax_Display') || el.style.display === 'block',
                        };
                    }
                    
                    // 2. Try script tag child/sibling
                    const script = el.querySelector('script[type^="math/tex"]');
                    if (script && script.textContent) {
                        const type = script.getAttribute('type') || '';
                        const isDisplay = type.includes('mode=display');
                        return {
                            latex: script.textContent.trim(),
                            displayMode: isDisplay,
                        };
                    }
                    
                    const parent = el.parentElement;
                    if (parent) {
                        const siblingScript = parent.querySelector('script[type^="math/tex"]');
                        if (siblingScript && siblingScript.textContent) {
                            const type = siblingScript.getAttribute('type') || '';
                            const isDisplay = type.includes('mode=display');
                            return {
                                latex: siblingScript.textContent.trim(),
                                displayMode: isDisplay,
                            };
                        }
                    }
                    
                    return false;
                },
            },
            {
                tag: 'img',
                getAttrs: element => {
                    const el = element as HTMLImageElement;
                    const alt = el.getAttribute('alt') || '';
                    const src = el.getAttribute('src') || '';
                    const className = el.className || '';
                    
                    // Check if it's a math image
                    const isMathImage = 
                        className.includes('math') || 
                        className.includes('latex') || 
                        src.includes('latex') || 
                        src.includes('chart.apis.google.com/chart?cht=tx') ||
                        alt.startsWith('$') || 
                        alt.startsWith('\\');
                        
                    if (isMathImage && alt) {
                        let latex = alt.trim();
                        const isDisplay = latex.startsWith('$$') && latex.endsWith('$$');
                        if (isDisplay) {
                            latex = latex.slice(2, -2).trim();
                        } else if (latex.startsWith('$') && latex.endsWith('$')) {
                            latex = latex.slice(1, -1).trim();
                        }
                        return {
                            latex: latex,
                            displayMode: isDisplay || className.includes('block') || src.includes('display=true'),
                        };
                    }
                    return false;
                },
            },
        ];
    },

    renderHTML({ HTMLAttributes }) {
        return [
            'span',
            mergeAttributes(HTMLAttributes, {
                'data-math': HTMLAttributes.latex,
                'data-display-mode': HTMLAttributes.displayMode ? 'true' : 'false',
                class: HTMLAttributes.displayMode ? 'math-block' : 'math-inline',
            }),
            HTMLAttributes.displayMode ? `$$${HTMLAttributes.latex}$$` : `$${HTMLAttributes.latex}$`,
        ];
    },

    addNodeView() {
        return ({ node }) => {
            const dom = document.createElement('span');
            const displayMode = node.attrs.displayMode;
            dom.className = displayMode ? 'math-block-view my-6 py-2 block text-center' : 'math-inline-view inline-block px-1';
            dom.setAttribute('data-math', node.attrs.latex);
            dom.setAttribute('data-display-mode', displayMode ? 'true' : 'false');

            try {
                katex.render(node.attrs.latex, dom, {
                    displayMode: displayMode,
                    throwOnError: false,
                });
            } catch (err) {
                dom.textContent = node.attrs.latex;
            }

            return {
                dom,
            };
        };
    },

    addInputRules() {
        return [
            new InputRule({
                find: /\$\$(.+?)\$\$\s$/,
                handler: ({ state, range, match }) => {
                    const start = range.from;
                    const end = range.to;
                    const latex = match[1].trim();
                    if (!latex) return;
                    state.tr.replaceWith(start, end, this.type.create({ latex, displayMode: true }));
                },
            }),
            new InputRule({
                find: /(?:^|[^$])\$([^$]+?)\$\s$/,
                handler: ({ state, range, match }) => {
                    const latex = match[1].trim();
                    if (!latex) return;
                    
                    // Safety checks to avoid false positive currency matches
                    if (latex.length > 100) return;
                    if (/[.!?]\s+[A-Z]/.test(latex)) return;
                    if (latex.includes('\n')) return;
                    if (latex.includes(' ')) {
                        const hasMathSymbol = /[=+\-*/\\_^{}<>]/.test(latex) || latex.includes('\\times') || latex.includes('\\div');
                        if (!hasMathSymbol) return;
                    }
                    
                    let start = range.from;
                    let end = range.to;
                    if (match[0] && !match[0].startsWith('$')) {
                        start += 1;
                    }
                    state.tr.replaceWith(start, end, this.type.create({ latex, displayMode: false }));
                },
            }),
        ];
    },

    addPasteRules() {
        return [];
    },
});

const convertTextMathToHtmlMath = (html: string): string => {
    return html;
};


interface ToolbarButtonProps {
    onClick: () => void;
    isActive?: boolean;
    disabled?: boolean;
    icon: React.ReactNode;
    title?: string;
    tooltip?: string;
    size?: 'sm' | 'md';
}

const ToolbarButton: React.FC<ToolbarButtonProps> = ({ onClick, isActive, disabled, icon, title, tooltip, size = 'md' }) => {
    const [showTooltip, setShowTooltip] = useState(false);
    const displayTitle = tooltip || title || '';
    const paddingClass = size === 'sm' ? 'p-1' : 'p-2';

    return (
        <div className="relative flex flex-col items-center">
            <button
                onClick={onClick}
                disabled={disabled}
                onMouseEnter={() => setShowTooltip(true)}
                onMouseLeave={() => setShowTooltip(false)}
                className={`${paddingClass} rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${isActive
                    ? 'bg-accent text-accent-foreground'
                    : 'text-muted-foreground hover:bg-muted'
                    }`}
            >
                {icon}
            </button>
            {showTooltip && displayTitle && (
                <div className="absolute top-full mt-2 left-1/2 transform -translate-x-1/2 px-2 py-1 bg-popover text-popover-foreground text-xs rounded whitespace-nowrap z-50 shadow-lg pointer-events-none border border-border">
                    {displayTitle}
                    <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 border-4 border-transparent border-b-popover"></div>
                </div>
            )}
        </div>
    );
};

const getMetricColorClass = (score: number, isInverse: boolean = false) => {
    if (isInverse) {
        // Lower is better (e.g. Difficulty)
        // < 40: Good, 40-79: Medium, >= 80: Bad
        if (score >= 80) return "text-destructive";
        if (score >= 40) return "text-chart-4";
        return "text-chart-2";
    } else {
        // Higher is better (e.g. Readability, SEO)
        // >= 60: Good, 40-59: Medium, < 40: Bad
        if (score >= 60) return "text-chart-2";
        if (score >= 40) return "text-chart-4";
        return "text-destructive";
    }
};


const normalizeResidualMarkdownHeadings = (html: string): string => {
    if (!html) return html;

    const parser = new DOMParser();
    const doc = parser.parseFromString(html, 'text/html');

    doc.querySelectorAll('p').forEach((paragraph) => {
        // Ignore rich blocks where text-only conversion could remove structure.
        if (paragraph.querySelector('img,svg,table,pre,code,blockquote,ul,ol,li')) return;

        const raw = (paragraph.textContent || '').replace(/\u00a0/g, ' ').trim();
        const match = raw.match(/^(#{1,6})\s+(.+)$/);
        if (!match) return;

        const level = Math.min(match[1].length, 6);
        const heading = doc.createElement(`h${level}`);
        heading.textContent = match[2].trim();
        paragraph.replaceWith(heading);
    });

    return doc.body.innerHTML;
};

const EditorStyles = `
  /* Table Styles */
  .ProseMirror table {
    border-collapse: collapse;
    margin: 1.5rem 0;
    overflow: hidden;
    table-layout: fixed;
    width: 100%;
  }
  .ProseMirror td, .ProseMirror th {
    min-width: 1em;
    border: 1px solid hsl(var(--border));
    padding: 0.75rem;
    vertical-align: top;
    box-sizing: border-box;
    position: relative;
    line-height: 1.5;
  }
  .ProseMirror th {
    font-weight: 600;
    text-align: left;
    background-color: hsl(var(--muted));
    color: hsl(var(--foreground));
  }
  .ProseMirror .selectedCell:after {
    z-index: 2;
    position: absolute;
    content: "";
    left: 0; right: 0; top: 0; bottom: 0;
    background: hsl(var(--ring) / 0.2);
    pointer-events: none;
  }

  /* Typography & Spacing - "Professional Look" */
  .ProseMirror {
    padding: 1rem;
  }
  .ProseMirror h1 {
    font-size: 2.25em;
    font-weight: 800;
    margin-top: 2.5rem;
    margin-bottom: 1.25rem;
    line-height: 1.2;
    color: hsl(var(--foreground));
  }
  .ProseMirror h2 {
    font-size: 1.75em;
    font-weight: 700;
    margin-top: 2rem;
    margin-bottom: 1rem;
    line-height: 1.3;
    color: hsl(var(--foreground));
    letter-spacing: -0.025em;
  }
  .ProseMirror h3 {
    font-size: 1.35em;
    font-weight: 600;
    margin-top: 1.75rem;
    margin-bottom: 0.75rem;
    line-height: 1.4;
    color: hsl(var(--foreground));
  }
  .ProseMirror p {
    margin-top: 0.5rem;
    margin-bottom: 1.25rem;
    line-height: 1.75; /* Readable line height */
    color: hsl(var(--muted-foreground));
    font-size: 1.05rem;
  }
  .ProseMirror ul, .ProseMirror ol {
    margin-bottom: 1.25rem;
    padding-left: 1.6em;
  }
  .ProseMirror li {
    margin-bottom: 0.5rem;
  }
  .ProseMirror blockquote {
    border-left: 4px solid hsl(var(--border));
    padding-left: 1rem;
    font-style: italic;
    color: hsl(var(--muted-foreground));
  }
  .ProseMirror .geo-key-takeaways {
    background: #f3f4f6;
    border: 1px solid #e5e7eb;
    border-radius: 1rem;
    padding: 1.25rem 1.5rem;
    margin: 0 0 1.75rem 0;
    color: #111827;
  }
  .ProseMirror .geo-key-takeaways h2,
  .ProseMirror .geo-key-takeaways h3 {
    margin-top: 0;
    margin-bottom: 0.75rem;
    color: #111827;
  }
  .ProseMirror .geo-key-takeaways ul,
  .ProseMirror .geo-key-takeaways ol,
  .ProseMirror .geo-key-takeaways p:last-child {
    margin-bottom: 0;
  }
  .ProseMirror .geo-key-takeaways p,
  .ProseMirror .geo-key-takeaways li {
    color: #1f2937;
  }

  /* Citation Styles */
  .citation-link.hidden-citation {
    display: none;
  }

  .ProseMirror .zenith-infographic-container {
    display: flex;
    justify-content: center;
    margin: 1.75rem auto;
    width: 100%;
  }
  .ProseMirror .zenith-infographic-container svg {
    display: block;
    max-width: 100%;
    height: auto;
    border-radius: 1rem;
    box-shadow: 0 18px 40px hsl(var(--foreground) / 0.08);
    background: white;
  }
  .ProseMirror .zenith-infographic-loading-shell,
  .ProseMirror .zenith-infographic-error {
    width: min(100%, 720px);
    min-height: 180px;
    border: 1px solid hsl(var(--border));
    border-radius: 1rem;
    background: linear-gradient(135deg, hsl(var(--background)), hsl(var(--muted) / 0.55));
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.875rem;
    padding: 1.25rem;
    color: hsl(var(--foreground));
  }
  .ProseMirror .zenith-infographic-loading-copy,
  .ProseMirror .zenith-infographic-error {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
  }
  .ProseMirror .zenith-infographic-loading-copy span,
  .ProseMirror .zenith-infographic-error span {
    color: hsl(var(--muted-foreground));
    font-size: 0.95rem;
  }
  .ProseMirror .zenith-infographic-loading-spinner {
    width: 1.75rem;
    height: 1.75rem;
    border-radius: 999px;
    border: 3px solid hsl(var(--border));
    border-top-color: hsl(var(--primary));
    animation: zenith-spin 0.8s linear infinite;
  }
  @keyframes zenith-spin {
    to { transform: rotate(360deg); }
  }
`;

export const ArticleEditor: React.FC = () => {
    const { id } = useParams<{ id: string }>();
    const { user } = useAuth();

    const navigate = useNavigate();
    const editorContainerRef = useRef<HTMLDivElement | null>(null);
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);
    const [isSuggesting, setIsSuggesting] = useState(false);
    const [isDirty, setIsDirty] = useState(false);

    // Autosave & shortcut states
    const [saveStatus, setSaveStatus] = useState<'saved' | 'unsaved' | 'saving' | 'error'>('saved');
    const [lastSavedTime, setLastSavedTime] = useState<string | null>(null);
    const isMac = React.useMemo(() => typeof window !== 'undefined' && /Mac|iPod|iPhone|iPad/.test(navigator.userAgent), []);

    // Warn on unsaved changes (Browser Navigation)
    useEffect(() => {
        const handleBeforeUnload = (e: BeforeUnloadEvent) => {
            if (isDirty) {
                e.preventDefault();
                e.returnValue = '';
            }
        };
        window.addEventListener('beforeunload', handleBeforeUnload);
        return () => window.removeEventListener('beforeunload', handleBeforeUnload);
    }, [isDirty]);



    // Warn on unsaved changes (In-App Navigation)
    // NOTE: useBlocker requires Data Router (createBrowserRouter). 
    // Since app uses legacy BrowserRouter, we cannot use it easily without refactoring.
    // Removed to prevent crash. Browser-level warning (beforeunload) still active.

    // Form Stats
    const [title, setTitle] = useState('');
    const [hook, setHook] = useState('');
    const [thesis, setThesis] = useState('');
    const [deck, setDeck] = useState(''); // New Deck state
    const [featuredImage, setFeaturedImage] = useState<ImageMetadata | null>(null);
    const [imagePickMode, setImagePickMode] = useState<'content' | 'featured'>('content');
    const [isAddImageModalOpen, setIsAddImageModalOpen] = useState(false);
    const [imageModalInitialTab, setImageModalInitialTab] = useState<'smart' | 'ai' | 'stock' | 'upload' | 'url' | 'infographic' | undefined>(undefined);


    // Reference/Citation state
    const [citations, setCitations] = useState<any[]>([]);
    const [selectedCitations, setSelectedCitations] = useState<Set<number>>(new Set());
    const [showInTextCitations, setShowInTextCitations] = useState(true);
    const [showReferenceSelector, setShowReferenceSelector] = useState(false);

    // WordPress & LinkedIn export state
    const [showWordPressModal, setShowWordPressModal] = useState(false);
    const [showLinkedInModal, setShowLinkedInModal] = useState(false);
    const [articleData, setArticleData] = useState<any>(null);


    // Metrics & Affiliate State
    const [metrics, setMetrics] = useState<any>({});
    const [affiliateOpportunities, setAffiliateOpportunities] = useState<any>(null);
    const citationAuthorityMeta = React.useMemo(() => rankCitationDomains(citations), [citations]);
    const selectedDomainCount = React.useMemo(() => {
        return new Set(
            Array.from(selectedCitations)
                .map((idx) => citationAuthorityMeta[idx]?.domain)
                .filter(Boolean)
        ).size;
    }, [citationAuthorityMeta, selectedCitations]);
    const isCuratedReferenceView = selectedCitations.size > 0 && selectedCitations.size < citations.length;

    const isInitialLoad = useRef(true);

    // Track metadata changes to mark editor as dirty
    useEffect(() => {
        if (loading) return;
        if (isInitialLoad.current) {
            isInitialLoad.current = false;
            setIsDirty(false);
            return;
        }
        setIsDirty(true);
    }, [title, hook, thesis, deck, featuredImage, loading]);

    // Update save status indicator based on dirty state
    useEffect(() => {
        if (isDirty) {
            setSaveStatus('unsaved');
        }
    }, [isDirty]);

    const materializeEditorHtml = React.useCallback((html: string) => {
        let processed = materializeInfographicHtml(html);
        processed = ensureIntroKeyTakeaways(processed, articleData);
        return beautifyTablesHtml(processed);
    }, [articleData]);
    const normalizeEditorHtml = React.useCallback(
        (html: string) => convertTextMathToHtmlMath(ensureIntroKeyTakeaways(normalizeInfographicHtmlForEditor(html), articleData)),
        [articleData],
    );

    const extensions = React.useMemo(() => [
        StarterKit.configure({
            heading: {
                levels: [1, 2, 3],
            },
            bulletList: false, // We use CustomBulletList
        }),
        Link.configure({
            openOnClick: true,
            HTMLAttributes: {
                class: 'text-indigo-600 dark:text-indigo-400 underline hover:text-indigo-800 transition-colors',
            },
        }),
        CustomBulletList,
        CustomImage.configure({
            HTMLAttributes: {
                class: 'rounded-lg max-w-full h-auto my-8 border border-border shadow-sm',
            },
        }),
        Table.configure({
            resizable: true,
        }),
        TableRow,
        TableHeader,
        TableCell,
        CharacterCount,
        HeadingIdExtension,
        InfographicBlock,
        MathNode,
    ], []);

    const editor = useEditor({
        extensions,
        content: '',
        onUpdate: () => {
            setIsDirty(true);
        },
        editorProps: {
            attributes: {
                class: 'prose dark:prose-invert max-w-none focus:outline-none min-h-[500px] prose-table:border-collapse',
            },
            handleDOMEvents: {
                contextmenu: (_, event) => {
                    event.preventDefault();
                    setContextMenu({ x: event.clientX, y: event.clientY });
                    return true;
                },
                paste: (_view, event) => {
                    const html = event.clipboardData?.getData('text/html');
                    const text = event.clipboardData?.getData('text/plain');
                    console.log('%c[CLIPBOARD HTML]', 'color: blue; font-weight: bold;', html);
                    console.log('%c[CLIPBOARD TEXT]', 'color: green; font-weight: bold;', text);
                    return false;
                },
                click: (_view, event) => {
                    const target = event.target as HTMLElement;
                    if (target.tagName === 'IMG') {
                        const anchor = target.closest('a');
                        const link = anchor?.getAttribute('href') || target.getAttribute('data-link');
                        if (link) {
                            window.open(link, '_blank');
                            return true;
                        }
                    }
                    return false;
                }
            }
        },
    });

    const hasTextSelection = !!editor && !editor.state.selection.empty && !!editor.state.doc.textBetween(editor.state.selection.from, editor.state.selection.to).trim();



    const extractCitationsFromHtml = (html: string): any[] => {
        const parser = new DOMParser();
        const doc = parser.parseFromString(html, 'text/html');
        const extracted: any[] = [];

        // Look for paragraphs starting with [n] or [^n]
        const paragraphs = doc.querySelectorAll('p');
        paragraphs.forEach(p => {
            const text = p.textContent || '';
            const match = text.trim().match(/^\[\^?(\d+)\]\s*(.*)$/);
            if (match) {
                // const index = parseInt(match[1]); // Unused
                const content = match[2];

                // Try to find a link
                const link = p.querySelector('a');
                const url = link ? link.getAttribute('href') : '#';

                // Extract title - if there's a link, use its text, otherwise try to parse from content
                let titleStr = link ? link.textContent : '';
                if (!titleStr) {
                    // Remove both [TYPE] and any trailing dots or truncation markers
                    titleStr = content.replace(/\[[A-Z]+\]\.?$/, '').trim();
                    titleStr = titleStr.replace(/\.?\s*\.{2,}$/, '').trim(); // Remove "..." or " .."
                }

                // Try to find source type e.g. [WEB], [JOURNAL]
                const typeMatch = content.match(/\[([A-Z]+)\]/);
                const sourceType = typeMatch ? typeMatch[1].toLowerCase() : 'unknown';

                extracted.push({
                    title: titleStr || 'Unknown Source',
                    url: url || '#',
                    source_type: sourceType,
                    extracted: true // Mark as extracted from HTML
                });
            }
        });

        return extracted;
    };

    useEffect(() => {
        const fetchArticle = async () => {
            if (!user || !id) return;
            try {
                const { data, error } = await supabase
                    .from('Titles')
                    .select('*')
                    .eq('id', id)
                    .single();

                if (error) throw error;

                const d = data as any;

                // Normalize state
                setTitle(d.Title || d.title || '');
                setHook(d.hook || d.Hook || '');
                setThesis(d.thesis || d.Thesis || '');
                // Fetch Deck from root column (priority) or metadata or wp_custom_fields or excerpt
                const metaDeck = d.deck || d.metadata?.deck || d.wp_custom_fields?.deck || d.excerpt || '';
                setDeck(metaDeck);


                if (d.featuredImageURL || d.featuredImageUrl || d.featuredimageurl) {
                    setFeaturedImage({
                        url: d.featuredImageURL || d.featuredImageUrl || d.featuredimageurl || '',
                        author: d.ImageAuthor || d.featuredImageAuthor || d.featuredimageauthor || '',
                        alt: d.mediaAltText || d.MediaAltText || d.mediaalttext || '',
                        title: d.mediaTitle || d.mediatitle || '',
                        caption: d.mediaCaption || d.mediacaption || ''
                    });
                }

                // Extract Metrics & Affiliate Data
                setMetrics({
                    // Basic scores
                    seo_optimization_score: d.seo_optimization_score,
                    readability_score: d.readability_score,
                    viral_potential_score: d.viral_potential_score,
                    humanization_score: d.quality_report?.humanization_score,
                    grounding_score: d.quality_report?.grounding_score,
                    geo_score: d.quality_report?.geo_score,
                    quality_gate_decision: d.quality_gate?.decision,
                    primary_keyword: d.primary_keyword,
                    keyword_selection_source: d.keyword_selection_source || d.keyword_research_source,
                    keyword_research_confidence: d.keyword_research_confidence,
                    selected_keyword_intent: d.selected_keyword_intent,
                    selected_keyword_search_volume: d.selected_keyword_search_volume ?? d.total_search_volume,
                    selected_keyword_difficulty: d.selected_keyword_difficulty ?? d.avg_keyword_difficulty,
                    selected_keyword_metrics_json: d.selected_keyword_metrics_json,
                    secondary_keywords_json: d.secondary_keywords_json,
                    supporting_entities_json: d.supporting_entities_json,
                    priority_questions_json: d.priority_questions_json,

                    // New metrics
                    difficulty_level: d.difficulty_level,
                    estimated_reading_time: d.estimated_reading_time,
                    target_audience: d.target_audience,
                    overall_quality_score: d.overall_quality_score,
                    audience_alignment_score: d.audience_alignment_score,
                    content_feasibility_score: d.content_feasibility_score,
                    business_impact_score: d.business_impact_score,
                    total_search_volume: d.total_search_volume,
                    avg_keyword_difficulty: d.avg_keyword_difficulty,
                    traffic_potential_score: d.traffic_potential_score,
                    competition_score: d.competition_score,

                    // Mapped keys validation (debug)
                    source_idea_id: d.source_idea_id,
                    topic_id: d.topic_id,
                    subtopic: d.subtopic,
                    content_type: d.content_type
                });

                setAffiliateOpportunities(d.affiliate_opportunities);

                // Parse citations from JSON
                let parsedCitations: any[] = [];
                try {
                    const citationsData = d.citations || d.Citations;
                    if (citationsData) {
                        parsedCitations = typeof citationsData === 'string' ? JSON.parse(citationsData) : citationsData;
                    }
                } catch (e) {
                    console.error('Failed to parse citations:', e);
                }
                if (!Array.isArray(parsedCitations)) {
                    parsedCitations = [];
                }

                let content = d.htmlArticle || d.htmlarticle || '';

                // Fallback: extract from HTML if JSON is missing
                if (parsedCitations.length === 0 && content) {
                    const extracted = extractCitationsFromHtml(content);
                    if (extracted.length > 0) {
                        console.log(`Extracted ${extracted.length} citations from HTML`);
                        parsedCitations = extracted;
                    }
                }

                // Fallback: If no htmlArticle, use content_outline (Key Data)
                if (!content && d.content_outline) {
                    console.log("Using content_outline as initial content");

                    // Ensure content_outline is a string
                    let paramsOutline = d.content_outline;
                    if (typeof paramsOutline !== 'string') {
                        paramsOutline = JSON.stringify(paramsOutline, null, 2);
                    }

                    // Simple Markdown-to-HTML conversion for initial display
                    // 1. Headers
                    let outlineHtml = paramsOutline
                        .replace(/^### (.*$)/gim, '<h3>$1</h3>')
                        .replace(/^## (.*$)/gim, '<h2>$1</h2>')
                        .replace(/^# (.*$)/gim, '<h1>$1</h1>')
                        .replace(/^\* (.*$)/gim, '<li>$1</li>')
                        .replace(/^- (.*$)/gim, '<li>$1</li>');

                    // 2. Lists (wrap <li> with <ul>) (simplified)
                    // Note: This is a robust-enough approximation for Tiptap to ingest

                    // 3. Paragraphs (double newlines)
                    outlineHtml = outlineHtml.split('\n\n').map((block: string) => {
                        if (block.trim().startsWith('<h') || block.trim().startsWith('<li')) return block;
                        return `<p>${block.replace(/\n/g, '<br>')}</p>`;
                    }).join('');

                    content = outlineHtml;
                }

                // Clean citations (legacy titles with [WEB] or truncation markers)
                parsedCitations = parsedCitations.map(cit => {
                    let title = cit.title || cit.source_title || 'Unknown Source';
                    // Remove both [TYPE] and any trailing dots or truncation markers
                    title = title.replace(/\[[A-Z]+\]\.?$/, '').trim();
                    title = title.replace(/\.?\s*\.{2,}$/, '').trim(); // Remove "..." or " .."
                    return { ...cit, title: title || 'Unknown Source' };
                });

                setCitations(parsedCitations);

                // Restore selection
                let restoredSelected = new Set<number>(parsedCitations.map((_, i) => i));
                try {
                    const selectedData = d.selected_citations || d.selectedCitations;
                    if (selectedData) {
                        const parsedSelected = typeof selectedData === 'string' ? JSON.parse(selectedData) : selectedData;
                        if (Array.isArray(parsedSelected)) {
                            const validSelected = parsedSelected
                                .map((value) => Number(value))
                                .filter((idx) => Number.isInteger(idx) && idx >= 0 && idx < parsedCitations.length);
                            restoredSelected = new Set(validSelected);
                        }
                    }
                } catch (e) {
                    console.error('Failed to restore selected citations:', e);
                }
                if (parsedCitations.length === 0) {
                    restoredSelected = new Set<number>();
                } else if (restoredSelected.size === 0) {
                    restoredSelected = new Set<number>(parsedCitations.map((_, i) => i));
                }
                setSelectedCitations(restoredSelected);

                setShowInTextCitations(d.include_in_text_citations ?? d.includeInTextCitations ?? true);

                // Ensure content has structure
                const hasStructure = /<h[1-6]/i.test(content);
                if (!content || !hasStructure) {
                    console.log("Assembling content from structured data...");
                    try {
                        const assembled = await assembleArticleHtml({
                            userId: user.id,
                            titleId: id,
                            onlyBody: true,
                            tableOfContentsFlag: true,
                            sectionNumberingFlag: true
                        });
                        if (assembled) content = assembled;
                    } catch (err) {
                        console.warn("Assembly failed:", err);
                    }
                }

                if (editor && content) {
                    // Normalize content to new format if legacy markers detected
                    if (content.includes('[^') || content.includes('<h2>References</h2>')) {
                        console.log("Normalizing content to new format...");
                        // Use a temporary function similar to applyReferenceChanges logic
                        const normalizeCitations = (html: string, currentCitations: any[], currentlySelected: Set<number>, showInText: boolean) => {
                            let tempHtml = html;

                            // Map existing indices
                            const sortedSelected = Array.from(currentlySelected).sort((a, b) => a - b);
                            const indexMap = new Map<number, number>();
                            sortedSelected.forEach((originalIndex, i) => {
                                indexMap.set(originalIndex, i + 1);
                            });

                            // Remove existing References section
                            tempHtml = tempHtml.replace(/\s*<hr>\s*<h2>References<\/h2>[\s\S]*$/i, '');
                            tempHtml = tempHtml.replace(/\s*<hr[^>]*>\s*<h2[^>]*>References<\/h2>[\s\S]*$/i, '');

                            // Update in-text markers (linked markers first)
                            const citationRegex = /<a[^>]*class="citation-link"[^>]*>\[\^?(\d+)\]<\/a>|\[\^?(\d+)\]/g;
                            tempHtml = tempHtml.replace(citationRegex, (_match, p1, p2) => {
                                // Try to get index from data attribute first (stable), fallback to text (unstable)
                                const dataIndexMatch = _match.match(/data-original-index="(\d+)"/);
                                const originalIndex = dataIndexMatch ? parseInt(dataIndexMatch[1]) : (parseInt(p1 || p2) - 1);

                                if (!currentlySelected.has(originalIndex)) return '';

                                const newIndex = indexMap.get(originalIndex) || (originalIndex + 1);
                                const citation = currentCitations[originalIndex];
                                const url = citation?.url || '#';
                                const title = citation?.title || 'Source';

                                const visibilityClass = showInText ? '' : ' hidden-citation';
                                const sourceIndicator = citation?.source_type ? ` (${citation.source_type.toUpperCase()})` : '';
                                const linkTitle = `${title}${sourceIndicator}`;

                                return `<a href="${url}" target="_blank" rel="noopener noreferrer" title="${linkTitle}" data-original-index="${originalIndex}" class="citation-link${visibilityClass}" style="color: hsl(var(--primary)); text-decoration: none; font-weight: 500; border-bottom: 1px dotted hsl(var(--primary));">[${newIndex}]</a>`;
                            });

                            // Handle grouped plain markers like [^1, ^3] or [1, 3]
                            const groupedCitationRegex = /\[(\^?\d+(?:\s*,\s*\^?\d+)*)\]/g;
                            tempHtml = tempHtml.replace(groupedCitationRegex, (_match, group) => {
                                const originalIndices = String(group)
                                    .split(',')
                                    .map(token => parseInt(token.replace('^', '').trim(), 10) - 1)
                                    .filter((idx) => Number.isFinite(idx) && currentlySelected.has(idx));

                                if (originalIndices.length === 0 || !showInText) return '';

                                const links = originalIndices.map((originalIndex) => {
                                    const newIndex = indexMap.get(originalIndex) || (originalIndex + 1);
                                    const citation = currentCitations[originalIndex];
                                    const url = citation?.url || '#';
                                    const title = citation?.title || 'Source';
                                    const sourceIndicator = citation?.source_type ? ` (${citation.source_type.toUpperCase()})` : '';
                                    const linkTitle = `${title}${sourceIndicator}`;
                                    return `<a href="${url}" target="_blank" rel="noopener noreferrer" title="${linkTitle}" data-original-index="${originalIndex}" class="citation-link" style="color: hsl(var(--primary)); text-decoration: none; font-weight: 500; border-bottom: 1px dotted hsl(var(--primary));">[${newIndex}]</a>`;
                                });

                                return links.join(', ');
                            });

                            // Strictly remove all in-text citation markers if disabled.
                            if (!showInText) {
                                tempHtml = tempHtml.replace(/<a[^>]*class="citation-link[^"]*"[^>]*>\[[^\]]+\]<\/a>/g, '');
                                tempHtml = tempHtml.replace(/\[\^?\d+(?:\s*,\s*\^?\d+)*\]/g, '');
                            }

                            // Rebuild References section
                            if (currentlySelected.size > 0) {
                                let referencesHTML = '\n\n<hr>\n\n<h2>References</h2>\n\n';
                                sortedSelected.forEach((originalIndex, i) => {
                                    const citation = currentCitations[originalIndex];
                                    const citationNumber = i + 1;
                                    const titleStr = citation?.title || citation?.source_title || 'Unknown Source';
                                    const url = citation?.url || '#';
                                    const author = citation?.author || '';
                                    const publicationDate = citation?.publication_date || '';

                                    referencesHTML += `<p><strong>[${citationNumber}]</strong> `;
                                    if (author && author !== 'Unknown Author') {
                                        referencesHTML += `${author}`;
                                        if (publicationDate) referencesHTML += ` (${publicationDate})`;
                                        referencesHTML += '. ';
                                    } else if (publicationDate) {
                                        referencesHTML += `(${publicationDate}) `;
                                    }

                                    if (url && url !== '#' && url !== '') {
                                        referencesHTML += `<a href="${url}" target="_blank" rel="noopener noreferrer" style="color: hsl(var(--primary)); text-decoration: underline;">${titleStr}</a>`;
                                    } else {
                                        referencesHTML += `<em>${titleStr}</em>`;
                                    }
                                    referencesHTML += '.</p>\n';
                                });
                                tempHtml += referencesHTML;
                            }
                            return tempHtml;
                        };

                        content = normalizeCitations(content, parsedCitations, restoredSelected, d.include_in_text_citations ?? d.includeInTextCitations ?? true);
                    }
                    const cleanedContent = normalizeResidualMarkdownHeadings(content);
                    editor.commands.setContent(ensureIntroKeyTakeaways(normalizeEditorHtml(cleanedContent), d));
                }

                // Store article data for WordPress export
                setArticleData(d);
            } catch (error) {
                console.error('Error in fetchArticle:', error);
                alert('Failed to load article');
                navigate('/my-articles');
            } finally {
                setLoading(false);
            }
        };

        fetchArticle();
    }, [id, user, editor]);


    const persistArticle = React.useCallback(async (options?: {
        selectedCitations?: Set<number>;
        showInTextCitations?: boolean;
        htmlContent?: string;
    }) => {
        if (!user || !id || !editor) {
            throw new Error('Editor is not ready yet.');
        }

        const htmlContent = options?.htmlContent || materializeEditorHtml(editor.getHTML());
        const selectedCitationSet = options?.selectedCitations || selectedCitations;
        const selectedIndices = Array.from(selectedCitationSet);
        const includeInText = options?.showInTextCitations ?? showInTextCitations;

        const payload = {
            Title: title,
            hook: hook,
            thesis: thesis,
            deck: deck,
            htmlArticle: htmlContent,
            featuredImageURL: featuredImage?.url || null,
            ImageAuthor: featuredImage?.author || null,
            mediaAltText: featuredImage?.alt || null,
            mediaTitle: featuredImage?.title || null,
            mediaCaption: featuredImage?.caption || null,
            include_in_text_citations: includeInText,
            selected_citations: JSON.stringify(selectedIndices)
        };

        const { error } = await supabase
            .from('Titles')
            .update(payload)
            .eq('id', id);

        if (error) throw error;

        setArticleData((current: any) => current ? { ...current, ...payload } : current);
        setIsDirty(false);
        return htmlContent;
    }, [
        user,
        id,
        editor,
        materializeEditorHtml,
        selectedCitations,
        title,
        hook,
        thesis,
        deck,
        featuredImage,
        showInTextCitations
    ]);

    const handleSave = React.useCallback(async () => {
        if (!user || !id || !editor) return;
        setSaving(true);
        setSaveStatus('saving');
        try {
            await persistArticle();
            setSaveStatus('saved');
            const now = new Date();
            setLastSavedTime(now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }));
            alert('Changes saved successfully!');
        } catch (error) {
            console.error('Error saving article:', error);
            setSaveStatus('error');
            alert('Failed to save changes');
        } finally {
            setSaving(false);
        }
    }, [user, id, editor, persistArticle]);

    // Autosave Effect (debounced 3 seconds)
    useEffect(() => {
        if (!isDirty || loading || saving) return;

        const timer = setTimeout(async () => {
            setSaveStatus('saving');
            try {
                await persistArticle();
                setSaveStatus('saved');
                const now = new Date();
                setLastSavedTime(now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }));
            } catch (err) {
                console.error('Autosave failed:', err);
                setSaveStatus('error');
            }
        }, 3000);

        return () => clearTimeout(timer);
    }, [isDirty, loading, saving, persistArticle]);

    // Keyboard shortcut (⌘S / Ctrl+S)
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if ((e.metaKey || e.ctrlKey) && e.key === 's') {
                e.preventDefault();
                handleSave();
            }
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [handleSave]);

    const applyReferenceChanges = async (selectedIndices: Set<number>, showInText: boolean) => {
        if (!editor) return;

        // Update state
        setSelectedCitations(selectedIndices);
        setShowInTextCitations(showInText);

        // Get current content
        let htmlContent = editor.getHTML();

        // Step 1: Sequential Renumbering & Format Change ([^n] -> [n])
        // Create a mapping from original 0-based index to new 1-based sequential index
        const sortedSelected = Array.from(selectedIndices).sort((a, b) => a - b);
        const indexMap = new Map<number, number>();
        sortedSelected.forEach((originalIndex, i) => {
            indexMap.set(originalIndex, i + 1);
        });

        // Regex to find citation markers: [^n] or [n] or links containing them
        // We look for the pattern in the text or within <a> tags
        const citationRegex = /<a[^>]*class="citation-link"[^>]*>\[\^?(\d+)\]<\/a>|\[\^?(\d+)\]/g;

        htmlContent = htmlContent.replace(citationRegex, (_match, p1, p2) => {
            // Try to get index from data attribute first (stable), fallback to text (unstable)
            const dataIndexMatch = _match.match(/data-original-index="(\d+)"/);
            const originalIndex = dataIndexMatch ? parseInt(dataIndexMatch[1]) : (parseInt(p1 || p2) - 1);

            // If this citation is not selected
            if (!selectedIndices.has(originalIndex)) {
                return ''; // Remove from text permanently
            }

            // Get new sequential index
            const newIndex = indexMap.get(originalIndex);
            if (!newIndex) return '';

            // Get citation info for the link
            const citation = citations[originalIndex];
            const url = citation?.url || '#';
            const title = citation?.title || 'Source';

            // Return new format [n] with link
            const sourceIndicator = citation?.source_type ? ` (${citation.source_type.toUpperCase()})` : '';
            const linkTitle = `${title}${sourceIndicator}`;

            // Handle visibility via class
            const visibilityClass = showInText ? '' : ' hidden-citation';

            return `<a href="${url}" target="_blank" rel="noopener noreferrer" title="${linkTitle}" data-original-index="${originalIndex}" class="citation-link${visibilityClass}" style="color: hsl(var(--primary)); text-decoration: none; font-weight: 500; border-bottom: 1px dotted hsl(var(--primary));">[${newIndex}]</a>`;
        });

        // Handle grouped plain markers like [^1, ^3] or [1, 3]
        const groupedCitationRegex = /\[(\^?\d+(?:\s*,\s*\^?\d+)*)\]/g;
        htmlContent = htmlContent.replace(groupedCitationRegex, (_match, group) => {
            const originalIndices = String(group)
                .split(',')
                .map(token => parseInt(token.replace('^', '').trim(), 10) - 1)
                .filter((idx) => Number.isFinite(idx) && selectedIndices.has(idx));

            if (originalIndices.length === 0 || !showInText) return '';

            const links = originalIndices.map((originalIndex) => {
                const newIndex = indexMap.get(originalIndex) || (originalIndex + 1);
                const citation = citations[originalIndex];
                const url = citation?.url || '#';
                const title = citation?.title || 'Source';
                const sourceIndicator = citation?.source_type ? ` (${citation.source_type.toUpperCase()})` : '';
                const linkTitle = `${title}${sourceIndicator}`;
                return `<a href="${url}" target="_blank" rel="noopener noreferrer" title="${linkTitle}" data-original-index="${originalIndex}" class="citation-link" style="color: hsl(var(--primary)); text-decoration: none; font-weight: 500; border-bottom: 1px dotted hsl(var(--primary));">[${newIndex}]</a>`;
            });

            return links.join(', ');
        });

        // Strictly remove all in-text citation markers if disabled.
        if (!showInText) {
            htmlContent = htmlContent.replace(/<a[^>]*class="citation-link[^"]*"[^>]*>\[[^\]]+\]<\/a>/g, '');
            htmlContent = htmlContent.replace(/\[\^?\d+(?:\s*,\s*\^?\d+)*\]/g, '');
        }

        // Step 2: Update References section at the end
        // Remove existing References section (handles various potential formats)
        htmlContent = htmlContent.replace(/\s*<hr>\s*<h2>References<\/h2>[\s\S]*$/i, '');
        htmlContent = htmlContent.replace(/\s*<hr[^>]*>\s*<h2[^>]*>References<\/h2>[\s\S]*$/i, '');

        // Build new References section with only selected citations, sequentially numbered
        if (selectedIndices.size > 0) {
            let referencesHTML = '\n\n<hr>\n\n<h2>References</h2>\n\n';

            sortedSelected.forEach((originalIndex, i) => {
                const citation = citations[originalIndex];
                const citationNumber = i + 1; // New sequential number

                const titleStr = citation.title || citation.source_title || 'Unknown Source';
                const url = citation.url || '#';
                const author = citation.author || '';
                const sourceType = citation.source_type || 'unknown';
                const publicationDate = citation.publication_date || '';

                referencesHTML += `<p><strong>[${citationNumber}]</strong> `;

                if (author && author !== 'Unknown Author') {
                    referencesHTML += `${author}`;
                    if (publicationDate) {
                        referencesHTML += ` (${publicationDate})`;
                    }
                    referencesHTML += '. ';
                } else if (sourceType === 'rag' && author === 'Unknown Author') {
                    referencesHTML += 'Unknown Author';
                    if (publicationDate) {
                        referencesHTML += ` (${publicationDate})`;
                    }
                    referencesHTML += '. ';
                } else if (publicationDate) {
                    referencesHTML += `(${publicationDate}) `;
                }

                if (url && url !== '#' && url !== '') {
                    referencesHTML += `<a href="${url}" target="_blank" rel="noopener noreferrer" style="color: hsl(var(--primary)); text-decoration: underline;">${titleStr}</a>`;
                } else {
                    referencesHTML += `<em>${titleStr}</em>`;
                }

                referencesHTML += '.</p>\n';
            });

            htmlContent += referencesHTML;
        }

        // Update editor content
        editor.commands.setContent(htmlContent);

        // Persist filter changes immediately so refresh and WP export keep the curated view.
        try {
            const persistedHtml = materializeEditorHtml(htmlContent);
            await persistArticle({
                selectedCitations: selectedIndices,
                showInTextCitations: showInText,
                htmlContent: persistedHtml,
            });
            setArticleData((current: any) => current ? {
                ...current,
                htmlArticle: persistedHtml,
                include_in_text_citations: showInText,
                selected_citations: JSON.stringify(Array.from(selectedIndices)),
            } : current);
        } catch (error) {
            console.error('Error persisting reference filter changes:', error);
            alert('Filter applied in editor, but failed to persist. Please click Save Changes.');
        }
    };

    const getSelectedText = (): string => {
        if (!editor) return '';
        const { from, to } = editor.state.selection;
        return editor.state.doc.textBetween(from, to);
    };

    const handleGenerateInfographic = () => {
        if (!editor || !user) return;

        const selectedText = getSelectedText().trim();

        if (!selectedText) {
            alert('Highlight a paragraph or text selection first to generate an infographic.');
            return;
        }

        setImagePickMode('content');
        setImageModalInitialTab('infographic');
        setIsAddImageModalOpen(true);
    };

    const handleGenerateSmartContextImage = () => {
        setImagePickMode('content');
        setImageModalInitialTab('smart');
        setIsAddImageModalOpen(true);
    };

    const setLink = () => {
        if (!editor) return;

        if (editor.isActive('image')) {
            const previousUrl = editor.getAttributes('image').link || '';
            const url = window.prompt('Image Link URL', previousUrl);

            if (url === null) return;
            if (url === '') {
                editor.chain().focus().updateAttributes('image', { link: null }).run();
                return;
            }
            editor.chain().focus().updateAttributes('image', { link: url }).run();
            return;
        }

        const previousUrl = editor.getAttributes('link').href;
        const url = window.prompt('URL', previousUrl);

        if (url === null) return;
        if (url === '') {
            editor.chain().focus().extendMarkRange('link').unsetLink().run();
            return;
        }
        editor.chain().focus().extendMarkRange('link').setLink({ href: url }).run();
    };

    const addMath = () => {
        if (!editor) return;
        
        const isMathActive = editor.isActive('math');
        const attrs = editor.getAttributes('math');
        const currentLatex = isMathActive ? attrs.latex : '';
        const currentDisplayMode = isMathActive ? attrs.displayMode : false;
        
        // Get current selected text if not active
        const { from, to } = editor.state.selection;
        const selectionText = isMathActive ? '' : editor.state.doc.textBetween(from, to).trim();
        
        const latex = window.prompt('Enter LaTeX formula:', currentLatex || selectionText);
        if (latex === null) return;
        
        if (latex.trim() === '') {
            if (isMathActive) {
                editor.chain().focus().deleteSelection().run();
            }
            return;
        }
        
        const mode = window.confirm(`Click OK for Block Equation ($$formula$$) or Cancel for Inline Equation ($formula$).\n\nCurrent is: ${currentDisplayMode ? 'Block' : 'Inline'}`);
        
        editor.chain().focus().insertContent({
            type: 'math',
            attrs: {
                latex: latex.trim(),
                displayMode: mode,
            }
        }).run();
    };

    const addTable = () => {
        editor?.chain().focus().insertTable({ rows: 3, cols: 3, withHeaderRow: true }).run();
    };



    const handleSuggestInternalLinks = async () => {
        if (!editor || !user || !editor.getText()) return;

        try {
            setIsSuggesting(true);
            // Optional: Toast "Analyzing content..."

            const response = await apiClient.post<any>('/internal-links/suggest', {
                content: editor.getText(),
                user_id: user.id
            });

            const matches = response.matches;

            if (!matches || matches.length === 0) {
                alert("No relevant internal links found.");
                return;
            }

            let appliedCount = 0;

            // Apply links
            // We need to match the text in the editor.
            // Since Tiptap/ProseMirror works with nodes and positions, doing a simple global replace on text might be tricky if content has markup.
            // However, the LLM returns the exact text substring. We can search for it.
            // A safer approach for Tiptap is to scan the document for the text and apply marks.

            // NOTE: Simple Find/Replace for demonstration.
            // In a real editor, we'd iterate nodes. Here we'll try to use a slightly robust approach or just regex replace if Tiptap supports it easily.
            // Tiptap doesn't have a "replace all occurrences of string X with link Y" built-in one-liner that is context-aware without plugins.
            // But we can scan.

            // Let's iterate matches and traverse the doc.
            // To avoid complexity of overlapping ranges or multiple matches, we'll implement a simple one-pass or one-by-one.

            // Simplest valid approach for Tiptap:
            // Get JSON, traverse, find text nodes, match, record positions, apply transactions.
            // OR use a library helper.

            // For this task, let's assume we replace the *first* occurrence or *all* occurrences found in text nodes.
            // We'll use a helper to find text positions.

            matches.forEach((match: any) => {
                const { matched_text, link } = match;
                if (!matched_text || !link) return;

                // Very basic implementation: search text in doc
                // Use a custom extension or just manual node traversal?
                // Manual traversal is safer.

                editor.state.doc.descendants((node, pos) => {
                    if (node.isText && node.text) {
                        const index = node.text.indexOf(matched_text);
                        if (index !== -1) {
                            const from = pos + index;
                            const to = from + matched_text.length;

                            // Check if already linked? (Improvement: check marks)
                            const hasLink = node.marks.some(m => m.type.name === 'link');

                            if (!hasLink) {
                                editor.chain()
                                    .setTextSelection({ from, to })
                                    .setLink({ href: link })
                                    .run();
                                appliedCount++;
                            }
                        }
                    }
                });
            });

            if (appliedCount > 0) {
                alert(`Added ${appliedCount} internal links.`);
            } else {
                alert("Matches found but could not be applied (maybe text changed or already linked).");
            }

        } catch (error) {
            console.error("Error suggesting links:", error);
            alert("Failed to suggest links. Check console.");
        } finally {
            setIsSuggesting(false);
        }
    };

    // Context Menu State
    const [contextMenu, setContextMenu] = useState<{ x: number, y: number } | null>(null);

    // Close context menu on click elsewhere
    useEffect(() => {
        const handleClick = () => setContextMenu(null);
        document.addEventListener('click', handleClick);
        return () => document.removeEventListener('click', handleClick);
    }, []);

    const addImage = () => {
        setImagePickMode('content');
        setIsAddImageModalOpen(true);
    };

    const handleImageSelected = (imageUrl: string, metadata: ImageMetadata) => {
        if (imagePickMode === 'content' && editor) {
            // Insert AFTER the selected text, do not replace it
            const { to } = editor.state.selection;
            editor.chain().focus().setTextSelection(to).setImage({
                src: imageUrl,
                alt: metadata.MediaAltText || metadata.mediaTitle || '',
                width: metadata.width || '100%',
                alignment: metadata.alignment || 'center',
                link: metadata.link || null
            } as any).run();
        } else if (imagePickMode === 'featured') {
            setFeaturedImage({
                url: imageUrl,
                author: metadata.ImageAuthor || '',
                alt: metadata.MediaAltText || '',
                title: metadata.mediaTitle || '',
                caption: metadata.mediaCaption || '',
                link: metadata.link || ''
            });
        }
        setIsAddImageModalOpen(false);
    };

    // Table of Contents Logic
    const addTableOfContents = () => {
        if (!editor) return;

        // 1. Remove existing ToC if present
        let found = false;
        editor.state.doc.descendants((node, pos) => {
            if (found) return false;
            if (node.type.name === 'heading' && node.textContent.trim() === 'Table of Contents') {
                // Delete header + valid next sibling list (if any)
                let size = node.nodeSize;
                try {
                    const nextNode = editor.state.doc.nodeAt(pos + size);
                    // Check if next node is a list (bulletList or orderedList)
                    if (nextNode && (nextNode.type.name === 'bulletList' || nextNode.type.name === 'orderedList')) {
                        size += nextNode.nodeSize;
                    }
                } catch (e) {
                    // Ignore error if nodeAt fails
                }

                editor.commands.deleteRange({ from: pos, to: pos + size });
                found = true;
                return false;
            }
        });

        // 2. Generate new ToC
        try {
            const items: { level: number; text: string; id: string }[] = [];
            const tr = editor.state.tr;
            let hasUpdates = false;

            editor.state.doc.descendants((node, pos) => {
                if (node.type.name === 'heading') {
                    const text = node.textContent;
                    // Skip empty headings and the ToC header itself (though it's deleted now, safety check)
                    if (!text.trim() || text.trim() === 'Table of Contents') return;

                    // Generate ID if missing
                    let id = node.attrs.id;
                    if (!id) {
                        id = text.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '');
                        if (!id || id.length === 0) id = `heading-${Math.random().toString(36).substring(7)}`;
                        tr.setNodeMarkup(pos, undefined, { ...node.attrs, id });
                        hasUpdates = true;
                    }
                    items.push({ level: node.attrs.level, text, id });
                }
            });

            if (items.length === 0) {
                alert("No headings found (H1-H3) to generate a Table of Contents.");
                return;
            }

            if (hasUpdates) {
                editor.view.dispatch(tr);
            }

            // Generate Nested HTML List
            const minLevel = Math.min(...items.map(i => i.level));

            // Use 'list-none' to remove bullets for a cleaner look
            let tocHtml = '<h3>Table of Contents</h3><ul class="list-none pl-0 space-y-2">';
            let currentLevel = minLevel;

            items.forEach((item) => {
                if (item.level > currentLevel) {
                    for (let i = currentLevel; i < item.level; i++) {
                        tocHtml += '<ul class="list-none pl-4 mt-1 space-y-1">';
                    }
                }
                else if (item.level < currentLevel) {
                    for (let i = item.level; i < currentLevel; i++) {
                        tocHtml += '</ul>';
                    }
                }

                const content = item.level === 2
                    ? `<strong>${item.text}</strong>`
                    : `<em>${item.text}</em>`;

                const linkClass = "no-underline hover:underline text-primary";
                tocHtml += `<li><a href="#${item.id}" class="${linkClass}">${content}</a></li>`;

                currentLevel = item.level;
            });

            for (let i = minLevel; i < currentLevel; i++) {
                tocHtml += '</ul>';
            }
            tocHtml += '</ul><br/>';

            // Insert at the TOP of the document (index 0)
            editor.commands.insertContentAt(0, tocHtml);
            editor.chain().focus().setTextSelection(0).run();

        } catch (error) {
            console.error("Error generating ToC:", error);
        }
    };

    // Section Numbering Logic
    const addSectionNumbering = () => {
        if (!editor) return;

        // Check if ToC exists before we modify headings
        let tocExists = false;
        editor.state.doc.descendants((node) => {
            if (tocExists) return false;
            if (node.type.name === 'heading' && node.textContent.trim() === 'Table of Contents') {
                tocExists = true;
                return false;
            }
        });

        try {
            const tr = editor.state.tr;
            const updates: { pos: number; text: string; nodeSize: number }[] = [];
            let sectionCounter = 1;

            // 1. Collect all updates first (using original document positions)
            editor.state.doc.descendants((node, pos) => {
                if (node.type.name === 'heading' && node.attrs.level === 2) {
                    const text = node.textContent.trim();
                    // Skip if already numbered
                    if (/^\d+\.\s/.test(text)) {
                        sectionCounter++;
                        return;
                    }

                    const newText = `${sectionCounter}. ${text}`;
                    updates.push({ pos, text: newText, nodeSize: node.nodeSize });
                    sectionCounter++;
                }
            });

            if (updates.length > 0) {
                // 2. Apply updates in REVERSE order (bottom to top) to maintain validity of positions
                updates.reverse().forEach(update => {
                    tr.insertText(update.text, update.pos + 1, update.pos + update.nodeSize - 1);
                });

                editor.view.dispatch(tr);
                setIsDirty(true);

                // 3. Refresh ToC if it existed
                if (tocExists) {
                    // Slight delay to ensure dispatch is processed?
                    // Tiptap/Prosemirror dispatch is synchronous, but React state updates might be async.
                    // addTableOfContents uses editor.state.doc, which should be fresh after dispatch.
                    addTableOfContents();
                }
            } else {
                alert("Sections are already numbered or no H2 headings found.");
            }
        } catch (error) {
            console.error("Error numbering sections:", error);
        }
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center min-h-screen bg-background">
                <Loader2 className="w-8 h-8 animate-spin text-primary" />
            </div>
        );
    }

    const liveWordCount = editor?.storage.characterCount.words() ?? 0;
    const liveCharacterCount = editor?.storage.characterCount.characters() ?? 0;
    const liveReadTimeMinutes = Math.max(1, Math.ceil(liveWordCount / 220));
    const displayReadTime =
        metrics.estimated_reading_time && Number(metrics.estimated_reading_time) > 0
            ? metrics.estimated_reading_time
            : `${liveReadTimeMinutes} min`;

    return (
        <div className="min-h-screen bg-background pb-20">
            <style>{EditorStyles}</style>
            <div className="sticky top-0 z-30 bg-background border-b border-border shadow-sm">
                <div className="max-w-screen-2xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="flex items-center justify-between h-16">
                        <div className="flex items-center gap-4">
                            <button
                                onClick={() => navigate('/my-articles')}
                                className="p-2 -ml-2 text-muted-foreground hover:text-foreground rounded-lg hover:bg-muted transition"
                            >
                                <ArrowLeft className="w-5 h-5" />
                            </button>
                            <h1 className="text-xl font-bold text-foreground truncate max-w-md">
                                Edit: {title}
                            </h1>
                        </div>
                        <div className="flex items-center gap-3">
                            {/* Save Status Indicator */}
                            <div className="flex items-center gap-2 px-3 py-1.5 text-xs rounded-full bg-muted/60 text-muted-foreground border border-border">
                                <span className={`w-2 h-2 rounded-full transition-all duration-300 ${
                                    saveStatus === 'saved' ? 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]' :
                                    saveStatus === 'saving' ? 'bg-amber-500 animate-pulse' :
                                    saveStatus === 'unsaved' ? 'bg-amber-400' :
                                    'bg-destructive'
                                }`} />
                                <span className="font-medium whitespace-nowrap">
                                    {saveStatus === 'saved' && (lastSavedTime ? `Saved at ${lastSavedTime}` : 'Saved')}
                                    {saveStatus === 'saving' && 'Saving...'}
                                    {saveStatus === 'unsaved' && 'Unsaved changes'}
                                    {saveStatus === 'error' && 'Failed to save'}
                                </span>
                            </div>

                            <button
                                onClick={() => {
                                    // Update article data with current editor content before opening modal
                                    if (articleData && editor) {
                                        setArticleData({
                                            ...articleData,
                                            htmlArticle: materializeEditorHtml(editor.getHTML()),
                                            Title: title,
                                            hook: hook,
                                            thesis: thesis,
                                            deck: deck,
                                            selected_citations: JSON.stringify(Array.from(selectedCitations)),
                                            include_in_text_citations: showInTextCitations,
                                            featuredImageUrl: featuredImage?.url,
                                            ImageAuthor: featuredImage?.author,
                                            mediaAltText: featuredImage?.alt,
                                            mediaTitle: featuredImage?.title,
                                            mediaCaption: featuredImage?.caption
                                        });
                                    }
                                    setShowWordPressModal(true);
                                }}
                                className="flex items-center gap-2 px-3 py-2 bg-accent text-accent-foreground rounded-lg hover:bg-accent/80 transition disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                <Globe className="w-4 h-4" />
                                <span className="hidden sm:inline">Export to WP</span>
                            </button>
                            <button
                                onClick={() => {
                                    if (editor && articleData) {
                                        setArticleData({
                                            ...articleData,
                                            htmlArticle: materializeEditorHtml(editor.getHTML()),
                                            Title: title,
                                            hook: hook,
                                            thesis: thesis,
                                            deck: deck,
                                            featuredImageUrl: featuredImage?.url,
                                            ImageAuthor: featuredImage?.author,
                                            mediaAltText: featuredImage?.alt,
                                            mediaTitle: featuredImage?.title,
                                            mediaCaption: featuredImage?.caption
                                        });
                                    }
                                    setShowLinkedInModal(true);
                                }}
                                className="flex items-center gap-2 px-3 py-2 bg-[#0A66C2] text-white rounded-lg hover:bg-[#084e96] transition shadow-sm"
                                title="Publish or repurpose this article for your LinkedIn feed"
                            >
                                <Share2 className="w-4 h-4" />
                                <span className="hidden sm:inline">Publish to LinkedIn</span>
                            </button>
                            <button
                                onClick={handleSave}
                                disabled={saving}
                                className="flex items-center gap-2 px-6 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                {saving ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
                                Save Changes
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            <main className="max-w-screen-2xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* Left Column: Editor */}
                    <div className="lg:col-span-2 space-y-6">
                        <div className="flex items-center justify-between mb-4">
                            <div className="flex items-center gap-2">
                                <span className="text-sm font-medium text-muted-foreground">Words: {liveWordCount}</span>
                                <span className="text-muted-foreground/30">|</span>
                                <span className="text-sm font-medium text-muted-foreground">Characters: {liveCharacterCount}</span>
                                <span className="text-muted-foreground/30">|</span>
                                <span className="text-sm font-medium text-muted-foreground">Read time: {liveReadTimeMinutes} min</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <button
                                    onClick={() => setShowReferenceSelector(true)}
                                    className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm transition ${showReferenceSelector ? 'bg-accent text-accent-foreground' : 'bg-background text-foreground border border-border hover:bg-muted'}`}
                                >
                                    <Filter className="w-4 h-4" />
                                    Reference Filter
                                </button>
                                {isCuratedReferenceView && (
                                    <span className="text-xs px-2 py-1 rounded-full bg-primary/10 text-primary border border-primary/30">
                                        Curated: {selectedDomainCount || 0} domains
                                    </span>
                                )}
                                <button
                                    onClick={() => setIsAddImageModalOpen(true)}
                                    className="flex items-center gap-2 px-3 py-1.5 bg-background border border-border rounded-lg hover:bg-muted transition text-sm text-foreground"
                                >
                                    <ImageIcon className="w-4 h-4" />
                                    Add Image
                                </button>
                            </div>
                        </div>

                        {/* Title, Hook, Thesis Input Fields */}
                        <div className="bg-background rounded-xl shadow-sm border border-border p-6 space-y-4">
                            <div>
                                <label className="block text-sm font-medium mb-1">Article Title</label>
                                <input
                                    type="text"
                                    value={title}
                                    onChange={(e) => setTitle(e.target.value)}
                                    className="w-full px-4 py-2 text-lg font-bold rounded-lg border border-border bg-muted/50 focus:ring-2 focus:ring-ring outline-none transition"
                                    placeholder="Enter article title..."
                                />
                            </div>

                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div>
                                    <label className="block text-sm font-medium mb-1">Hook</label>
                                    <textarea
                                        value={hook}
                                        onChange={(e) => setHook(e.target.value)}
                                        rows={2}
                                        className="w-full px-4 py-2 rounded-lg border border-border bg-muted/50 focus:ring-2 focus:ring-ring outline-none transition resize-none"
                                        placeholder="The hook to grab reader attention..."
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm font-medium mb-1">Thesis</label>
                                    <textarea
                                        value={thesis}
                                        onChange={(e) => setThesis(e.target.value)}
                                        rows={2}
                                        className="w-full px-4 py-2 rounded-lg border border-border bg-muted/50 focus:ring-2 focus:ring-ring outline-none transition resize-none"
                                        placeholder="The main thesis of the article..."
                                    />
                                </div>
                            </div>

                            {/* Deck Input */}
                            <div>
                                <label className="block text-sm font-medium mb-1">Deck (Subtitle/Summary)</label>
                                <textarea
                                    value={deck}
                                    onChange={(e) => setDeck(e.target.value)}
                                    rows={2}
                                    className="w-full px-4 py-2 rounded-lg border border-border bg-muted/50 focus:ring-2 focus:ring-ring outline-none transition resize-none"
                                    placeholder="A brief summary or subtitle for the article (often used below the headline)..."
                                />
                            </div>
                        </div>

                        {/* Editor Toolbar */}
                        <div ref={editorContainerRef} className="bg-background rounded-xl shadow-sm border border-border overflow-hidden">
                            {/* Toolbar Buttons */}
                            <div className="flex flex-wrap items-center gap-1 p-2 border-b border-border bg-muted/50 text-foreground sticky top-0 z-20">
                                <ToolbarButton
                                    onClick={() => editor?.chain().focus().toggleBold().run()}
                                    isActive={editor?.isActive('bold')}
                                    icon={<Bold className="w-4 h-4" />}
                                    tooltip="Bold"
                                />
                                <ToolbarButton
                                    onClick={() => editor?.chain().focus().toggleItalic().run()}
                                    isActive={editor?.isActive('italic')}
                                    icon={<Italic className="w-4 h-4" />}
                                    tooltip="Italic"
                                />
                                <div className="w-px h-6 bg-border mx-1" />
                                <ToolbarButton
                                    onClick={() => editor?.chain().focus().toggleHeading({ level: 2 }).run()}
                                    isActive={editor?.isActive('heading', { level: 2 })}
                                    icon={<Heading2 className="w-4 h-4" />}
                                    tooltip="Heading 2"
                                />
                                <ToolbarButton
                                    onClick={() => editor?.chain().focus().toggleHeading({ level: 3 }).run()}
                                    isActive={editor?.isActive('heading', { level: 3 })}
                                    icon={<Heading3 className="w-4 h-4" />}
                                    tooltip="Heading 3"
                                />
                                <ToolbarButton
                                    onClick={() => editor?.chain().focus().toggleBulletList().run()}
                                    isActive={editor?.isActive('bulletList')}
                                    icon={<List className="w-4 h-4" />}
                                    tooltip="Bullet List"
                                />
                                <div className="w-px h-6 bg-border mx-1" />
                                <ToolbarButton
                                    onClick={setLink}
                                    isActive={editor?.isActive('link')}
                                    icon={<LinkIcon className="w-4 h-4" />}
                                    tooltip="Link"
                                />
                                <ToolbarButton
                                    onClick={addImage}
                                    icon={<ImageIcon className="w-4 h-4" />}
                                    tooltip="Add Image"
                                />
                                <ToolbarButton
                                    onClick={addMath}
                                    isActive={editor?.isActive('math')}
                                    icon={<Sigma className="w-4 h-4" />}
                                    tooltip="Insert / Edit LaTeX Formula"
                                />
                                <ToolbarButton
                                    onClick={handleGenerateInfographic}
                                    disabled={!hasTextSelection}
                                    icon={<ChartColumn className="w-4 h-4" />}
                                    tooltip="Generate Infographic from Selection"
                                />
                                <ToolbarButton
                                    onClick={handleGenerateSmartContextImage}
                                    icon={<Wand2 className="w-4 h-4 text-indigo-500" />}
                                    tooltip="Smart Context Image (Auto Reference & Scene)"
                                />
                                <ToolbarButton
                                    onClick={handleSuggestInternalLinks}
                                    icon={isSuggesting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Link2 className="w-4 h-4" />}
                                    tooltip="Suggest Internal Links from your WP Posts"
                                />
                                <ToolbarButton
                                    onClick={() => setShowReferenceSelector(true)}
                                    icon={<Filter className="w-4 h-4" />}
                                    tooltip="Reference Filter"
                                />
                                <ToolbarButton
                                    onClick={addTable}
                                    icon={<TableIcon className="w-4 h-4" />}
                                    tooltip="Insert Table"
                                />
                                <ToolbarButton
                                    onClick={addTableOfContents}
                                    icon={<ListOrdered className="w-4 h-4" />}
                                    tooltip="Generate ToC"
                                />
                                <ToolbarButton
                                    onClick={addSectionNumbering}
                                    icon={<ListOrdered className="w-4 h-4 text-primary" />}
                                    tooltip="Number Sections (H2)"
                                />

                                {/* Table Controls (Visible only when table selected) */}
                                {editor?.isActive('table') && (
                                    <>
                                        <div className="w-px h-6 bg-border mx-1" />
                                        <div className="flex items-center gap-1 bg-accent rounded px-1">
                                            <span className="text-[10px] uppercase font-bold text-accent-foreground mr-1">Table</span>
                                            <ToolbarButton
                                                onClick={() => editor?.chain().focus().addColumnBefore().run()}
                                                icon={<Plus className="w-3 h-3 rotate-45" />} // Approximate
                                                tooltip="Add Col Before"
                                                size="sm"
                                            />
                                            <ToolbarButton
                                                onClick={() => editor?.chain().focus().addColumnAfter().run()}
                                                icon={<Plus className="w-3 h-3" />}
                                                tooltip="Add Col After"
                                                size="sm"
                                            />
                                            <ToolbarButton
                                                onClick={() => editor?.chain().focus().deleteColumn().run()}
                                                icon={<Trash2 className="w-3 h-3 text-red-400" />}
                                                tooltip="Delete Col"
                                                size="sm"
                                            />
                                            <div className="w-px h-4 bg-border mx-1" />
                                            <ToolbarButton
                                                onClick={() => editor?.chain().focus().addRowBefore().run()}
                                                icon={<Plus className="w-3 h-3" />}
                                                tooltip="Add Row Before"
                                                size="sm"
                                            />
                                            <ToolbarButton
                                                onClick={() => editor?.chain().focus().addRowAfter().run()}
                                                icon={<Plus className="w-3 h-3" />}
                                                tooltip="Add Row After"
                                                size="sm"
                                            />
                                            <ToolbarButton
                                                onClick={() => editor?.chain().focus().deleteRow().run()}
                                                icon={<Trash2 className="w-3 h-3 text-red-400" />}
                                                tooltip="Delete Row"
                                                size="sm"
                                            />
                                            <div className="w-px h-4 bg-border mx-1" />
                                            <ToolbarButton
                                                onClick={() => editor?.chain().focus().deleteTable().run()}
                                                icon={<Trash2 className="w-3 h-3 text-red-500" />}
                                                tooltip="Delete Table"
                                                size="sm"
                                            />
                                        </div>
                                    </>
                                )}
                            </div>

                            {/* Editor Area */}
                            <EditorContent editor={editor} />
                        </div>
                    </div>

                    {/* Right Column: Metrics & Sidebar */}
                    <div className="space-y-6">
                        {/* Featured Image Card */}
                        <div className="bg-card text-card-foreground p-6 rounded-2xl border border-border shadow-sm text-center">
                            <h3 className="font-semibold mb-4 text-foreground">Featured Image</h3>
                            {featuredImage ? (
                                <div className="relative group">
                                    <img
                                        src={featuredImage.url}
                                        alt={featuredImage.alt}
                                        className="w-full h-48 object-cover rounded-xl shadow-md"
                                    />
                                    <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition flex items-center justify-center gap-2 rounded-xl">
                                        <button
                                            onClick={() => {
                                                setImagePickMode('featured');
                                                setIsAddImageModalOpen(true);
                                            }}
                                            className="p-2 bg-background rounded-full text-primary hover:bg-muted"
                                            title="Change Image"
                                        >
                                            <RefreshCw className="w-4 h-4" />
                                        </button>
                                        <button
                                            onClick={() => setFeaturedImage(null)}
                                            className="p-2 bg-background rounded-full text-destructive hover:bg-muted"
                                            title="Remove Image"
                                        >
                                            <Trash2 className="w-4 h-4" />
                                        </button>
                                    </div>
                                    <p className="mt-2 text-xs text-muted-foreground truncate">{featuredImage.title}</p>
                                </div>
                            ) : (
                                <div
                                    onClick={() => {
                                        setImagePickMode('featured');
                                        setIsAddImageModalOpen(true);
                                    }}
                                    className="w-full h-48 border-2 border-dashed border-border rounded-xl flex flex-col items-center justify-center gap-2 text-muted-foreground hover:border-ring hover:text-foreground cursor-pointer transition bg-muted/30"
                                >
                                    <ImageIcon className="w-8 h-8 opacity-50" />
                                    <span className="text-sm">Click to add Featured Image</span>
                                </div>
                            )}
                        </div>

                        {/* Content Metrics Sidebar */}
                        <div className="bg-card text-card-foreground p-6 rounded-2xl border border-border shadow-sm">
                            <div className="flex items-center gap-2 mb-4">
                                <BarChart3 className="w-5 h-5 text-primary" />
                                <h3 className="font-semibold text-foreground">Content Metrics</h3>
                            </div>

                            <div className="space-y-6">
                                {/* Top Section: Main Gauges */}
                                <div className="grid grid-cols-2 gap-4 justify-items-center">
                                    <Gauge
                                        value={metrics.seo_optimization_score || 0}
                                        label="SEO Score"
                                        color={getMetricColorClass(metrics.seo_optimization_score || 0)}
                                        explanation={METRIC_EXPLANATIONS.seo_score}
                                        size={100}
                                    />

                                    <Gauge
                                        value={metrics.humanization_score || 0}
                                        label="Humanization"
                                        color={getMetricColorClass(metrics.humanization_score || 0)}
                                        size={100}
                                    />
                                </div>
                                <div className="grid grid-cols-2 gap-4 justify-items-center">
                                    <Gauge
                                        value={metrics.grounding_score || 0}
                                        label="Grounding"
                                        color={getMetricColorClass(metrics.grounding_score || 0)}
                                        size={100}
                                    />
                                    <Gauge
                                        value={metrics.geo_score || 0}
                                        label="GEO"
                                        color={getMetricColorClass(metrics.geo_score || 0)}
                                        size={100}
                                    />
                                </div>

                                <div className="border-t border-border my-4"></div>

                                {/* Secondary Scores Grid */}
                                <div className="grid grid-cols-2 gap-2">
                                    <div className="p-2 bg-muted/50 rounded-lg text-center">
                                        <div className="text-md font-bold text-foreground">{metrics.overall_quality_score || 0}%</div>
                                        <div className="text-[9px] text-muted-foreground uppercase">Quality</div>
                                    </div>

                                    <div className="p-2 bg-muted/50 rounded-lg text-center">
                                        <div className="text-md font-bold text-foreground">{metrics.quality_gate_decision || '-'}</div>
                                        <div className="text-[9px] text-muted-foreground uppercase">Gate</div>
                                    </div>
                                </div>

                                <div className="border-t border-border my-4"></div>

                                {/* Bottom Section: Text Metrics */}
                                <div className="space-y-2">
                                    <div className="flex items-center justify-between p-2 bg-muted/50 rounded-lg">
                                        <span className="text-xs text-muted-foreground">Difficulty Level</span>
                                        <span className="font-medium text-sm text-foreground">{metrics.difficulty_level || '-'}</span>
                                    </div>
                                    <div className="flex items-center justify-between p-2 bg-muted/50 rounded-lg">
                                        <div className="flex items-center gap-1">
                                            <span className="text-xs text-muted-foreground">Est. Reading Time</span>
                                            <MetricTooltip explanation={METRIC_EXPLANATIONS.reading_time} />
                                        </div>
                                        <span className="font-medium text-sm text-foreground">{displayReadTime}</span>
                                    </div>
                                    <div className="p-2 bg-muted/50 rounded-lg">
                                        <div className="flex items-center gap-1 mb-1">
                                            <span className="text-xs text-muted-foreground block">Target Audience</span>
                                            <MetricTooltip explanation={METRIC_EXPLANATIONS.audience_align} />
                                        </div>
                                        <span className="font-medium text-foreground text-sm">{metrics.target_audience || '-'}</span>
                                    </div>
                                </div>

                                <div className="border-t border-border my-4"></div>

                                <div className="space-y-2">
                                    <div className="flex items-center justify-between p-2 bg-muted/50 rounded-lg">
                                        <span className="text-xs text-muted-foreground">Primary Keyword</span>
                                        <span className="font-medium text-sm text-foreground truncate max-w-[60%] text-right">
                                            {metrics.primary_keyword || '-'}
                                        </span>
                                    </div>
                                    <div className="grid grid-cols-2 gap-2">
                                        <div className="p-2 bg-muted/50 rounded-lg">
                                            <div className="text-[9px] text-muted-foreground uppercase">Source</div>
                                            <div className="font-medium text-xs text-foreground truncate">
                                                {metrics.keyword_selection_source || '-'}
                                            </div>
                                        </div>
                                        <div className="p-2 bg-muted/50 rounded-lg">
                                            <div className="text-[9px] text-muted-foreground uppercase">Confidence</div>
                                            <div className="font-medium text-xs text-foreground">
                                                {typeof metrics.keyword_research_confidence === 'number'
                                                    ? `${Math.round((metrics.keyword_research_confidence <= 1 ? metrics.keyword_research_confidence * 100 : metrics.keyword_research_confidence))}%`
                                                    : '-'}
                                            </div>
                                        </div>
                                    </div>
                                    <div className="grid grid-cols-2 gap-2">
                                        <div className="p-2 bg-muted/50 rounded-lg">
                                            <div className="text-[9px] text-muted-foreground uppercase">Intent</div>
                                            <div className="font-medium text-xs text-foreground">{metrics.selected_keyword_intent || '-'}</div>
                                        </div>
                                        <div className="p-2 bg-muted/50 rounded-lg">
                                            <div className="text-[9px] text-muted-foreground uppercase">Volume</div>
                                            <div className="font-medium text-xs text-foreground">{metrics.selected_keyword_search_volume ?? '-'}</div>
                                        </div>
                                    </div>
                                    <div className="p-2 bg-muted/50 rounded-lg">
                                        <div className="text-[9px] text-muted-foreground uppercase">Difficulty</div>
                                        <div className="font-medium text-xs text-foreground">{metrics.selected_keyword_difficulty ?? '-'}</div>
                                    </div>
                                    <div className="p-2 bg-muted/50 rounded-lg">
                                        <div className="text-[9px] text-muted-foreground uppercase">Metric Provenance</div>
                                        <div className="font-medium text-xs text-foreground">
                                            {metrics.selected_keyword_metrics_json?.primary?.is_estimated
                                                ? 'Estimated aggregate carryover'
                                                : (metrics.selected_keyword_metrics_json?.primary?.metric_source || 'Exact keyword dossier')}
                                        </div>
                                    </div>
                                    <div className="p-2 bg-muted/50 rounded-lg">
                                        <div className="text-[9px] text-muted-foreground uppercase">Secondary Keywords</div>
                                        <div className="font-medium text-xs text-foreground">
                                            {(() => {
                                                const raw = metrics.secondary_keywords_json;
                                                const list = Array.isArray(raw)
                                                    ? raw
                                                    : (typeof raw === 'string'
                                                        ? raw.split(',').map((x: string) => x.trim()).filter(Boolean)
                                                        : []);
                                                return list.length > 0 ? list.slice(0, 6).join(', ') : '-';
                                            })()}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Affiliate Opportunities */}
                        <div className="bg-card text-card-foreground p-6 rounded-2xl border border-border shadow-sm">
                            <h3 className="font-semibold text-foreground mb-4">Affiliate Opportunities</h3>
                            {affiliateOpportunities?.programs?.length > 0 ? (
                                <div className="space-y-3">
                                    {affiliateOpportunities.programs.slice(0, 3).map((prog: any, i: number) => (
                                        <a
                                            key={i}
                                            href={prog.link || prog.url || '#'}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="block p-3 border border-border rounded-xl hover:bg-muted/30 transition"
                                        >
                                            <div className="font-medium text-sm truncate">{prog.name}</div>
                                            <div className="text-xs text-chart-2 mt-1">{prog.commission_rate}% Commission</div>
                                        </a>
                                    ))}
                                </div>
                            ) : (
                                <div className="text-sm text-muted-foreground text-center py-4">No opportunities found</div>
                            )}
                        </div>
                    </div>
                </div>
            </main>

            {contextMenu && (
                <div
                    className="fixed z-50 bg-popover text-popover-foreground border border-border shadow-xl rounded-xl py-2 min-w-[240px] flex flex-col gap-1"
                    style={{ top: contextMenu.y, left: contextMenu.x }}
                >
                    {/* Save Action */}
                    <div className="px-2 pb-1 border-b border-border mb-1">
                        <button
                            onClick={() => {
                                handleSave();
                                setContextMenu(null);
                            }}
                            disabled={saving}
                            className="w-full text-left px-3 py-1.5 hover:bg-primary/10 hover:text-primary text-foreground rounded-lg text-sm flex items-center justify-between font-semibold transition"
                        >
                            <span className="flex items-center gap-3">
                                {saving ? <Loader2 className="w-4 h-4 animate-spin text-primary" /> : <Save className="w-4 h-4 text-primary" />}
                                Save Changes
                            </span>
                            <span className="text-[10px] text-muted-foreground bg-muted px-1.5 py-0.5 rounded font-normal">{isMac ? '⌘S' : 'Ctrl+S'}</span>
                        </button>
                    </div>

                    {/* Basic Formatting */}
                    <div className="px-2 pb-1 border-b border-border mb-1">
                        <button onClick={() => { editor?.chain().focus().toggleBold().run(); setContextMenu(null); }} className="w-full text-left px-3 py-1.5 hover:bg-muted rounded-lg text-sm flex items-center gap-3">
                            <Bold className="w-4 h-4 text-muted-foreground" /> Bold
                        </button>
                        <button onClick={() => { editor?.chain().focus().toggleItalic().run(); setContextMenu(null); }} className="w-full text-left px-3 py-1.5 hover:bg-muted rounded-lg text-sm flex items-center gap-3">
                            <Italic className="w-4 h-4 text-muted-foreground" /> Italic
                        </button>
                        <button onClick={() => { setLink(); setContextMenu(null); }} className="w-full text-left px-3 py-1.5 hover:bg-muted rounded-lg text-sm flex items-center gap-3">
                            <LinkIcon className="w-4 h-4 text-muted-foreground" /> Link
                        </button>
                    </div>

                    {/* Headings & Lists */}
                    <div className="px-2 pb-1 border-b border-border mb-1">
                        <button onClick={() => { editor?.chain().focus().toggleHeading({ level: 2 }).run(); setContextMenu(null); }} className="w-full text-left px-3 py-1.5 hover:bg-muted rounded-lg text-sm flex items-center gap-3">
                            <Heading2 className="w-4 h-4 text-muted-foreground" /> Heading 2
                        </button>
                        <button onClick={() => { editor?.chain().focus().toggleHeading({ level: 3 }).run(); setContextMenu(null); }} className="w-full text-left px-3 py-1.5 hover:bg-muted rounded-lg text-sm flex items-center gap-3">
                            <Heading3 className="w-4 h-4 text-muted-foreground" /> Heading 3
                        </button>
                        <button onClick={() => { editor?.chain().focus().toggleBulletList().run(); setContextMenu(null); }} className="w-full text-left px-3 py-1.5 hover:bg-muted rounded-lg text-sm flex items-center gap-3">
                            <List className="w-4 h-4 text-muted-foreground" /> Bullet List
                        </button>
                    </div>

                    {/* Insert Actions */}
                    <div className="px-2 pb-1 border-b border-border mb-1">
                        <button onClick={() => { addImage(); setContextMenu(null); }} className="w-full text-left px-3 py-1.5 hover:bg-muted rounded-lg text-sm flex items-center gap-3">
                            <ImageIcon className="w-4 h-4 text-muted-foreground" /> Insert Image
                        </button>
                        <button onClick={() => { addMath(); setContextMenu(null); }} className="w-full text-left px-3 py-1.5 hover:bg-muted rounded-lg text-sm flex items-center gap-3">
                            <Sigma className="w-4 h-4 text-muted-foreground" /> Insert / Edit LaTeX Formula
                        </button>
                        <button onClick={() => { addTable(); setContextMenu(null); }} className="w-full text-left px-3 py-1.5 hover:bg-muted rounded-lg text-sm flex items-center gap-3">
                            <TableIcon className="w-4 h-4 text-muted-foreground" /> Insert Table
                        </button>
                    </div>

                    {/* Document Actions */}
                    <div className="px-2 pb-1 border-b border-border mb-1">
                        <button onClick={() => { addTableOfContents(); setContextMenu(null); }} className="w-full text-left px-3 py-1.5 hover:bg-muted rounded-lg text-sm flex items-center gap-3">
                            <ListOrdered className="w-4 h-4 text-muted-foreground" /> Generate ToC
                        </button>
                        <button onClick={() => { addSectionNumbering(); setContextMenu(null); }} className="w-full text-left px-3 py-1.5 hover:bg-muted rounded-lg text-sm flex items-center gap-3">
                            <ListOrdered className="w-4 h-4 text-primary" /> Number Sections
                        </button>
                    </div>

                    {/* Selection Actions */}
                    <div className="px-2">
                        <button
                            onClick={() => {
                                const selection = editor?.state.selection;
                                if (selection && !selection.empty) {
                                    const text = editor?.state.doc.textBetween(selection.from, selection.to);
                                    navigator.clipboard.writeText(text || '');
                                }
                                setContextMenu(null);
                            }}
                            className="w-full text-left px-3 py-1.5 hover:bg-muted rounded-lg text-sm flex items-center gap-3"
                        >
                            <span className="w-4 h-4 flex items-center justify-center font-mono text-xs text-muted-foreground border border-border rounded">C</span> Copy Selection
                        </button>
                        <button
                            onClick={() => {
                                handleGenerateInfographic();
                                setContextMenu(null);
                            }}
                            disabled={!hasTextSelection}
                            className="w-full text-left px-3 py-1.5 hover:bg-accent text-primary rounded-lg text-sm flex items-center gap-3 font-medium"
                        >
                            <ChartColumn className="w-4 h-4" /> Generate Infographic from Selection
                        </button>
                        <button
                            onClick={() => {
                                handleGenerateSmartContextImage();
                                setContextMenu(null);
                            }}
                            disabled={!hasTextSelection}
                            className="w-full text-left px-3 py-1.5 hover:bg-accent text-indigo-600 dark:text-indigo-400 rounded-lg text-sm flex items-center gap-3 font-medium"
                        >
                            <Wand2 className="w-4 h-4" /> Smart Context Image from Selection
                        </button>
                    </div>
                </div>
            )}

            {isAddImageModalOpen && user && (
                <AddImageModal
                    onClose={() => setIsAddImageModalOpen(false)}
                    onImageSelected={handleImageSelected}
                    selectedText={getSelectedText()}
                    userId={user.id}
                    initialTab={imageModalInitialTab}
                />
            )}

            {
                showReferenceSelector && (
                    <ReferenceSelector
                        citations={citations}
                        selectedCitations={selectedCitations}
                        showInTextCitations={showInTextCitations}
                        onClose={() => setShowReferenceSelector(false)}
                        onApply={applyReferenceChanges}
                    />
                )
            }

            {
                showWordPressModal && user && id && articleData && (
                    <WordPressExportModal
                        articleData={articleData}
                        articleId={id}
                        userId={user.id}
                        onClose={() => setShowWordPressModal(false)}
                        onSuccess={(postUrl) => {
                            setShowWordPressModal(false);
                            alert(`Successfully published to WordPress!\n\nView at: ${postUrl}`);
                        }}
                    />
                )
            }

            {
                showLinkedInModal && id && articleData && (
                    <LinkedInPublishModal
                        articleData={articleData}
                        articleId={id}
                        onClose={() => setShowLinkedInModal(false)}
                        onSuccess={(postUrl) => {
                            // Update local articleData with last_linkedin_post_url
                            setArticleData({
                                ...articleData,
                                last_linkedin_post_url: postUrl,
                                last_linkedin_status: 'published'
                            });
                        }}
                    />
                )
            }
        </div >
    );
};

export default ArticleEditor;
