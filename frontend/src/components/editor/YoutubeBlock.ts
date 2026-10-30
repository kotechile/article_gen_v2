import { Node, mergeAttributes } from '@tiptap/core';

/**
 * YouTube embed node for the article editor.
 *
 * Why a node and not a raw iframe: an atom block keeps the embed selectable,
 * movable and deletable as one unit, and gives a single place to normalise every
 * URL shape a writer pastes (watch, youtu.be, shorts, live, /embed, bare id).
 * The serialised HTML is self-contained — inline styles and all — because it is
 * what gets persisted to `Titles.htmlArticle` and pushed to WordPress, neither of
 * which loads this app's stylesheet.
 */

const VIDEO_ID_PATTERN = /^[A-Za-z0-9_-]{11}$/;

/** Hostnames that serve player URLs, with the `www.` prefix stripped. */
const YOUTUBE_HOSTS = new Set([
    'youtube.com',
    'm.youtube.com',
    'music.youtube.com',
    'youtu.be',
    'youtube-nocookie.com',
]);

const IFRAME_ALLOW =
    'accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share';

// Inline so the embed survives outside the app (WordPress, preview exports).
const WRAPPER_STYLE =
    'position:relative;width:100%;max-width:720px;margin:2rem auto;aspect-ratio:16 / 9;' +
    'background:#000;border-radius:12px;overflow:hidden;';
const IFRAME_STYLE = 'position:absolute;inset:0;width:100%;height:100%;border:0;';

export interface YouTubeTarget {
    videoId: string;
    /** Seconds into the video, from `t=` / `start=`; null when absent. */
    start: number | null;
}

/** `t=90`, `t=90s`, `t=1m30s`, `t=1h2m3s` → seconds. */
function parseStartParam(value: string | null): number | null {
    if (!value) return null;
    const trimmed = value.trim();
    if (/^\d+$/.test(trimmed)) {
        const seconds = Number(trimmed);
        return seconds > 0 ? seconds : null;
    }
    const parts = trimmed.match(/^(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$/);
    if (!parts) return null;
    const seconds =
        Number(parts[1] || 0) * 3600 + Number(parts[2] || 0) * 60 + Number(parts[3] || 0);
    return seconds > 0 ? seconds : null;
}

/**
 * Accepts a watch URL, youtu.be short link, /shorts, /live, /embed, a
 * youtube-nocookie player URL, or a bare 11-character id.
 * Returns null for anything else, so callers can tell the writer what happened
 * instead of inserting a broken embed.
 */
export function parseYoutubeUrl(input: string): YouTubeTarget | null {
    const raw = (input || '').trim();
    if (!raw) return null;

    if (VIDEO_ID_PATTERN.test(raw)) {
        return { videoId: raw, start: null };
    }

    let url: URL;
    try {
        url = new URL(/^https?:\/\//i.test(raw) ? raw : `https://${raw}`);
    } catch {
        return null;
    }

    const host = url.hostname.replace(/^www\./i, '').toLowerCase();
    if (!YOUTUBE_HOSTS.has(host)) return null;

    let videoId: string | null = null;
    if (host === 'youtu.be') {
        videoId = url.pathname.split('/').filter(Boolean)[0] || null;
    } else if (url.pathname === '/watch') {
        videoId = url.searchParams.get('v');
    } else {
        const match = url.pathname.match(/^\/(?:embed|shorts|live|v)\/([A-Za-z0-9_-]+)/);
        videoId = match ? match[1] : null;
    }

    if (!videoId || !VIDEO_ID_PATTERN.test(videoId)) return null;

    const start = parseStartParam(url.searchParams.get('t') || url.searchParams.get('start'));
    return { videoId, start };
}

export function buildEmbedUrl(videoId: string, start: number | null): string {
    const base = `https://www.youtube.com/embed/${videoId}`;
    return start ? `${base}?start=${start}` : base;
}

export function buildWatchUrl(videoId: string, start: number | null): string {
    const base = `https://www.youtube.com/watch?v=${videoId}`;
    return start ? `${base}&t=${start}s` : base;
}

/** Recovers the target from an already-rendered embed (our markup or a pasted iframe). */
function targetFromElement(element: HTMLElement): YouTubeTarget | null {
    const iframeSrc =
        element.tagName === 'IFRAME'
            ? element.getAttribute('src') || ''
            : element.querySelector('iframe')?.getAttribute('src') || '';
    const candidate = iframeSrc || element.getAttribute('data-src') || '';
    return parseYoutubeUrl(candidate);
}

export const YoutubeBlock = Node.create({
    name: 'youtube',
    group: 'block',
    atom: true,
    selectable: true,
    draggable: true,

    addAttributes() {
        return {
            videoId: { default: null },
            start: { default: null },
        };
    },

    parseHTML() {
        return [
            {
                tag: 'div[data-youtube-video]',
                getAttrs: (element) => targetFromElement(element as HTMLElement) || false,
            },
            {
                // A bare <iframe> pasted or imported from elsewhere still becomes an editable block.
                tag: 'iframe[src*="youtube"]',
                getAttrs: (element) => targetFromElement(element as HTMLElement) || false,
            },
            {
                tag: 'iframe[src*="youtu.be"]',
                getAttrs: (element) => targetFromElement(element as HTMLElement) || false,
            },
        ];
    },

    renderHTML({ node, HTMLAttributes }) {
        const videoId = String(node.attrs.videoId || '');
        if (!VIDEO_ID_PATTERN.test(videoId)) {
            return ['div', {}, 'Invalid YouTube video'];
        }

        const start = typeof node.attrs.start === 'number' ? node.attrs.start : null;
        const watchUrl = buildWatchUrl(videoId, start);

        return [
            'div',
            mergeAttributes(HTMLAttributes, {
                class: 'video-embed',
                'data-youtube-video': 'true',
                'data-video-id': videoId,
                'data-src': watchUrl,
                style: WRAPPER_STYLE,
            }),
            [
                'iframe',
                {
                    src: buildEmbedUrl(videoId, start),
                    title: 'YouTube video',
                    allow: IFRAME_ALLOW,
                    allowfullscreen: 'true',
                    frameborder: '0',
                    loading: 'lazy',
                    style: IFRAME_STYLE,
                },
            ],
        ];
    },
});
