import { supabase } from '../lib/supabase';
import { ensureIntroKeyTakeaways } from './geoFormatting';

// ----------  citation helpers ----------

// Global-ish state for chapter counting (scoped to function execution ideally)
interface ChapterCounts {
    H2: number;
    H3: number;
}

const generateChapterNumber = (headerType: string, chapterCounts: ChapterCounts): string => {
    if (headerType === 'H2') {
        chapterCounts.H2++;
        chapterCounts.H3 = 0;
        return chapterCounts.H2.toString();
    }
    if (headerType === 'H3') {
        chapterCounts.H3++;
        return `${chapterCounts.H2}.${chapterCounts.H3}`;
    }
    return '';
};

export interface ToCEntry {
    TitleID: string;
    headerType: string;
    chapter: string | null;
    sequence: number;
    content: string;
    user_id: string;
    citation_id: string | null;
}

export const createTableOfContents = (htmlContent: string, articleId: string, userId: string): ToCEntry[] => {
    const parser = new DOMParser();
    const doc = parser.parseFromString(htmlContent, 'text/html');
    const tableOfContents: ToCEntry[] = [];
    let sequence = 1000;

    let cumulative = new Set<string>(); // citations seen so far

    // Reset counters for new articles
    const chapterCounts: ChapterCounts = { H2: 0, H3: 0 };

    const addToTableOfContents = (element: Element | null, headerType: string | null = null, content: string | null = null) => {
        const html = content || (element ? element.outerHTML.trim() : '');
        if (!html) return;

        const ownCits = new Set(cumulative);
        let chapterNumber = '';
        if (headerType) chapterNumber = generateChapterNumber(headerType, chapterCounts);

        const citationId = ownCits.size ? Array.from(ownCits)[0] : null;

        tableOfContents.push({
            TitleID: articleId,
            headerType: headerType || (element ? element.tagName : 'P'),
            chapter: chapterNumber || null,
            sequence,
            content: html,
            user_id: userId,
            citation_id: citationId
        });
        sequence += 1000;
    };

    // Process all body elements including text nodes
    const processNodes = (node: Node) => {
        // Skip empty text nodes and whitespace
        if (node.nodeType === Node.TEXT_NODE) {
            const text = node.textContent?.trim();
            if (text) {
                addToTableOfContents(null, 'P', `<p>${text}</p>`);
            }
            return;
        }

        // Process element nodes
        if (node.nodeType === Node.ELEMENT_NODE) {
            const element = node as Element;
            const tagName = element.tagName;

            const isDirectTarget = ['H1', 'H2', 'H3', 'P', 'TABLE', 'UL', 'OL'].includes(tagName);

            if (isDirectTarget) {
                const html = element.outerHTML.trim();

                let m;
                const localRegex = new RegExp(/\[\^(\d+)]/g);
                while ((m = localRegex.exec(html)) !== null) {
                    cumulative.add(m[1]);
                }

                if (['H1', 'H2', 'H3'].includes(tagName)) {
                    addToTableOfContents(element, tagName);
                } else {
                    addToTableOfContents(element);
                }
            } else {
                Array.from(node.childNodes).forEach(child => processNodes(child));
            }
        }
    };

    Array.from(doc.body.childNodes).forEach(node => processNodes(node));

    return tableOfContents;
};

export const updateArticleAfterGeneration = async (articleId: string) => {
    console.log("Starting Post-Generation Update for:", articleId);

    try {
        const { data: titleData, error: fetchError } = await supabase
            .from('Titles')
            .select('htmlArticle, status')
            .eq('id', articleId)
            .single();

        if (fetchError || !titleData) throw fetchError || new Error("Article not found");

        const htmlContent = titleData.htmlArticle;
        if (!htmlContent) {
            console.warn("No HTML content found for article:", articleId);
            return;
        }

        /* 
        // Logic removed: No longer syncing to TableOfContents table as per user request.
        // The Editor and HTML blob are the source of truth.
        const tableOfContents = createTableOfContents(htmlContent, articleId, userId);
        console.log(`Generated ${tableOfContents.length} ToC entries.`);

        const { error: deleteError } = await supabase
            .from('TableOfContents')
            .delete()
            .eq('TitleID', articleId)
            .eq('user_id', userId);

        if (deleteError) throw deleteError;

        if (tableOfContents.length > 0) {
            const { error: insertError } = await supabase
                .from('TableOfContents')
                .insert(tableOfContents);

            if (insertError) throw insertError;
        }

        console.log("Table of Contents updated successfully.");
        */
        console.log("Skipping Table of Contents sync (disabled).");
        return true;

    } catch (error) {
        console.error("Error updating article post-generation:", error);
        throw error;
    }
};

// Helper interfaces for Assembly
export interface AssembleOptions {
    userId: string;
    titleId: string;
    title?: string;
    hook?: string;
    thesis?: string;
    affiliateDisclosure?: boolean;
    featuredImageUrl?: string;
    featuredImageAuthor?: string;
    mediaCaption?: string;
    mediaAltText?: string;
    tableOfContentsFlag?: boolean;
    sectionNumberingFlag?: boolean;
    onlyBody?: boolean;
}

export const assembleArticleHtml = async (options: AssembleOptions) => {
    console.log("Assembling article preview for:", options.titleId);

    // 1. Fetch PostLinks for keywords
    const { data: postLinks } = await supabase
        .from('PostLinks')
        .select('*')
        .eq('user_id', options.userId)
        .eq('titleId', options.titleId)
        .not('linkUrl', 'is', null)
        .neq('linkUrl', '')
        .neq('textKeyWord', '');

    const distinctPostLinks = new Map<string, string>();
    if (postLinks) {
        postLinks.forEach((item: any) => {
            if (!distinctPostLinks.has(item.textKeyWord)) {
                distinctPostLinks.set(item.textKeyWord, item.linkUrl);
            }
        });
    }

    // 2. Fetch TableOfContents
    const { data: tocData } = await supabase
        .from('TableOfContents')
        .select('*')
        .eq('user_id', options.userId)
        .eq('TitleID', options.titleId)
        .order('sequence', { ascending: true });

    if (!tocData || tocData.length === 0) {
        throw new Error("No Table of Contents found. Please regenerate.");
    }

    let postBody = "";

    // 3. Assemble Body from ToC
    for (const item of tocData) {
        // Handle Header Numbering injection
        if (options.sectionNumberingFlag && item.headerType && item.content && item.chapter && item.headerType.startsWith('H')) {
            const headerRegex = new RegExp(`<${item.headerType}(.*?)>`, 'i');
            // Inject chapter number: <H2>... -> <H2>1. ...
            // Logic from Noodl: item.content.replace(headerRegex, `<${headerType}$1>${chapterNumber} `);
            // Note: Noodl logic puts the number inside the tag content
            item.content = item.content.replace(headerRegex, `$&${item.chapter} `);
        }

        // Handle Image Records (if any stored in ToC with ImageURL)
        if (item.ImageURL) { // Assuming 'ImageURL' might exist on generic jsonb or similar if enabled
            const imageWpUrl = item.ImageURL;
            let imgHtml = `<img src="${imageWpUrl}" alt="${item.content || ''}" height="800" style="font-family: 'Roboto', sans-serif !important;" />`;
            if (item.imageLink) {
                imgHtml = `<a href="${item.imageLink}">${imgHtml}</a>`;
            }
            imgHtml += `<p style="font-family: 'Roboto', sans-serif !important;">${item.content || ''}</p>`;

            if (item.mediaAuthor) {
                imgHtml += `<footer style="font-family: 'Roboto', sans-serif !important; font-style: italic; font-size: small;">Image Credit: ${item.mediaAuthor}</footer>`;
            }
            postBody += imgHtml;
        } else {
            // Standard Content
            postBody += item.content || "";
        }
    }

    // Initialize content wrapper with specific styles handling Light and Dark modes
    const spacingStyles = `
        <style>
            .preview-content { font-family: 'Roboto', sans-serif !important; }
            
            /* Light Mode Defaults */
            .preview-content h1 { font-size: 3rem !important; line-height: 1.1 !important; font-weight: 800 !important; margin-bottom: 2rem !important; color: #111827 !important; } /* gray-900 */
            .preview-content h2 { font-size: 2.25rem !important; line-height: 2.5rem !important; font-weight: 700 !important; margin-top: 3rem !important; margin-bottom: 1.5rem !important; color: #1F2937 !important; } /* gray-800 */
            .preview-content h3 { font-size: 1.5rem !important; line-height: 2rem !important; font-weight: 600 !important; margin-top: 2rem !important; margin-bottom: 1rem !important; color: #374151 !important; } /* gray-700 */
            .preview-content p { margin-bottom: 1.5rem !important; line-height: 1.75 !important; color: #374151 !important; } /* gray-700 */
            .preview-content ul, .preview-content ol { margin-bottom: 1.5rem !important; padding-left: 1.5rem !important; color: #374151 !important; }
            .preview-content li { margin-bottom: 0.5rem !important; }
            .preview-content img { margin-top: 2rem !important; margin-bottom: 2rem !important; border-radius: 0.5rem; }
            .preview-content a { color: #4F46E5 !important; text-decoration: underline; }
            .preview-content .geo-key-takeaways {
                background: #f3f4f6 !important;
                border: 1px solid #e5e7eb !important;
                border-radius: 1rem !important;
                padding: 1.25rem 1.5rem !important;
                margin: 0 0 2rem 0 !important;
                color: #111827 !important;
            }
            .preview-content .geo-key-takeaways h2,
            .preview-content .geo-key-takeaways h3 {
                margin-top: 0 !important;
                margin-bottom: 0.75rem !important;
                color: #111827 !important;
            }
            .preview-content .geo-key-takeaways ul,
            .preview-content .geo-key-takeaways ol,
            .preview-content .geo-key-takeaways p:last-child {
                margin-bottom: 0 !important;
            }
            .preview-content .geo-key-takeaways p,
            .preview-content .geo-key-takeaways li {
                color: #1f2937 !important;
            }

            /* Dark Mode Overrides */
            :is(.dark .preview-content) h1 { color: #F9FAFB !important; } /* gray-50 */
            :is(.dark .preview-content) h2 { color: #F3F4F6 !important; } /* gray-100 */
            :is(.dark .preview-content) h3 { color: #E5E7EB !important; } /* gray-200 */
            :is(.dark .preview-content) p { color: #D1D5DB !important; } /* gray-300 */
            :is(.dark .preview-content) ul, :is(.dark .preview-content) ol { color: #D1D5DB !important; }
            :is(.dark .preview-content) a { color: #818CF8 !important; } /* indigo-400 */
        </style>
    `;

    // Wrap body - MOVED TO END
    // postBody = `<div class="preview-content">${postBody}</div>`;

    // 4. Generate Table of Contents (The actual List)
    if (options.tableOfContentsFlag) {
        let tocHtml = "<ul class='toc'>";
        for (const item of tocData) {
            if (item.headerType && item.headerType.startsWith('H')) {
                const level = parseInt(item.headerType.replace('H', '')) || 2;
                const indentation = "&nbsp;".repeat((level - 1) * 4);
                const tag = item.headerType === 'H1' ? 'strong' : 'span';
                const cleanText = item.content ? item.content.replace(/<[^>]*>/g, '').replace(item.chapter || '', '').trim() : '';

                // We assume anchor IDs might need to be generated if not present, but for preview we can just link to top or skip anchors if ID missing
                // Noodl logic used `item.id`. If generic content doesn't have ID, links won't work perfectly in preview without re-parsing body to add IDs.
                // For now, we replicate structure.
                tocHtml += `
                    <li style="font-family: 'Roboto', sans-serif !important;">
                        <a href="#${item.id || ''}" style="text-decoration: none; color: gray; line-height: 1.5; font-family: 'Roboto', sans-serif !important;">
                            <${tag}>${indentation}${item.chapter ? item.chapter + ' ' : ''}${cleanText}</${tag}>
                        </a>
                    </li>`;
            }
        }
        tocHtml += "</ul>";

        const tocSection = `
            <h2 style="margin-top: 3rem !important; margin-bottom: 1.5rem !important; font-family: 'Roboto', sans-serif !important;">Table of Contents</h2>
            <style>
                .toc li { list-style-type: none; }
                .toc ul ul { margin-left: 20px; }
                .toc { font-family: 'Roboto', sans-serif !important; }
                .preview-content .toc a { color: #6B7280 !important; text-decoration: none !important; } /* gray-500 */
                :is(.dark .preview-content .toc) a { color: #9CA3AF !important; } /* gray-400 */
                .preview-content .toc a:hover { color: #4F46E5 !important; }
                :is(.dark .preview-content .toc) a:hover { color: #818CF8 !important; }
            </style>
            <div class="toc" style="font-family: 'Roboto', sans-serif !important;">${tocHtml}</div>
        `;
        postBody = tocSection + postBody;
    }

    // 5. Prepend Metadata, Title, and Styles in Correct Order

    let metadataHtml = "";

    // Hook & Thesis (Smaller font)
    if (options.hook) metadataHtml += `<p style="font-family: 'Roboto', sans-serif !important; font-style: italic; font-size: 0.9em !important; margin-bottom: 1.5em !important;">${options.hook}</p>`;
    if (options.thesis) metadataHtml += `<p style="font-family: 'Roboto', sans-serif !important; font-style: italic; font-size: 0.9em !important; margin-bottom: 1.5em !important;">${options.thesis}</p>`;

    // Affiliate Disclosure
    if (options.affiliateDisclosure) {
        const adText = "This blog post contains affiliate links. This means that if you click on a link and make a purchase, I may receive a small commission at no extra cost to you.";
        metadataHtml += `<p style="font-family: 'Roboto', sans-serif !important; font-style: italic; font-weight: bolder; font-size: 0.85em !important; color: gray;">${adText}</p>`;
    }

    // Featured Image
    if (options.featuredImageUrl) {
        let featImgHtml = `<img src="${options.featuredImageUrl}" alt="${options.mediaAltText || ''}" height="800" />`;
        if (options.mediaCaption) featImgHtml += `<p style="font-family: 'Roboto', sans-serif !important;">${options.mediaCaption}</p>`;
        if (options.featuredImageAuthor) featImgHtml += `<footer style="font-family: 'Roboto', sans-serif !important; font-style: italic; font-size: small;">Image Credit: ${options.featuredImageAuthor}</footer>`;
        metadataHtml += featImgHtml;
    }

    // Combine in specific order: Styles + Title + Metadata + ToC + Content
    let titleHtml = "";
    if (options.title) {
        titleHtml = `<h1>${options.title}</h1>`;
    }

    // If ToC exists, it's already in 'tocSection' but we need to put it AFTER metadata
    // Current logic: postBody = tocSection + postBody (so ToC is at top of body)
    // We want: Title -> Metadata -> ToC -> Body

    // 6. Inject Links (Keyword Replacement) - Run this on postBody before final assembly
    distinctPostLinks.forEach((url, keyword) => {
        const regex = new RegExp(`\\b${keyword}\\b`, 'gi');
        // Simple replace for last occurrence as per Noodl logic (complex)
        // Noodl logic was: matches -> lastMatch -> substring replacement.
        // We will try to match that logic.
        const matches = [...postBody.matchAll(regex)];
        if (matches.length > 0) {
            const lastMatch = matches[matches.length - 1];
            if (lastMatch.index !== undefined) {
                postBody =
                    postBody.substring(0, lastMatch.index) +
                    `<a href="${url}" style="font-family: 'Roboto', sans-serif !important;">${keyword}</a>` +
                    postBody.substring(lastMatch.index + keyword.length);
            }
        }
    });

    // We construct final content BEFORE wrapping
    // Note: 'postBody' currently contains [ToC + Content Loop + Injected Links].

    // We want: [Title] + [Metadata] + [ToC] + [Article Body].
    let finalHtml = "";

    if (options.onlyBody) {
        // For Editor: Just the body content (ToC + Content). No Title, No Metadata blocks.
        // We might want structured ToC, but usually for Editor we just want the text flow.
        // However, keeping ToC in the editor is fine if it's part of the content user can edit.
        finalHtml = postBody;
    } else {
        // For Preview: Full structured page
        finalHtml = ensureIntroKeyTakeaways(
            titleHtml + metadataHtml + postBody,
            { thesis: options.thesis, hook: options.hook },
        );

        // NOW wrap everything in the styled container
        finalHtml = `<div class="preview-content">${finalHtml}</div>`;

        // Prepend the styles
        finalHtml = spacingStyles + finalHtml;
    }

    return finalHtml;
};
