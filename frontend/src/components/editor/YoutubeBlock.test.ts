import { describe, expect, it } from 'vitest';
import { Editor } from '@tiptap/core';
import StarterKit from '@tiptap/starter-kit';
import { YoutubeBlock, parseYoutubeUrl, buildEmbedUrl, buildWatchUrl } from './YoutubeBlock';

describe('parseYoutubeUrl', () => {
    it('reads the id from a watch URL, with or without extra params', () => {
        expect(parseYoutubeUrl('https://www.youtube.com/watch?v=dQw4w9WgXcQ')).toEqual({
            videoId: 'dQw4w9WgXcQ',
            start: null,
        });
        expect(parseYoutubeUrl('https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PL123&index=4')).toEqual({
            videoId: 'dQw4w9WgXcQ',
            start: null,
        });
    });

    it('reads youtu.be, shorts, live, embed and nocookie forms', () => {
        for (const input of [
            'https://youtu.be/dQw4w9WgXcQ',
            'youtu.be/dQw4w9WgXcQ?si=abc',
            'https://www.youtube.com/shorts/dQw4w9WgXcQ',
            'https://www.youtube.com/live/dQw4w9WgXcQ',
            'https://www.youtube.com/embed/dQw4w9WgXcQ',
            'https://m.youtube.com/watch?v=dQw4w9WgXcQ',
            'https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ',
        ]) {
            expect(parseYoutubeUrl(input)?.videoId, input).toBe('dQw4w9WgXcQ');
        }
    });

    it('accepts a bare video id', () => {
        expect(parseYoutubeUrl('  dQw4w9WgXcQ  ')).toEqual({ videoId: 'dQw4w9WgXcQ', start: null });
    });

    it('normalises the start time from t= and start=', () => {
        expect(parseYoutubeUrl('https://youtu.be/dQw4w9WgXcQ?t=90')?.start).toBe(90);
        expect(parseYoutubeUrl('https://youtu.be/dQw4w9WgXcQ?t=90s')?.start).toBe(90);
        expect(parseYoutubeUrl('https://youtu.be/dQw4w9WgXcQ?t=1m30s')?.start).toBe(90);
        expect(parseYoutubeUrl('https://youtu.be/dQw4w9WgXcQ?t=1h2m3s')?.start).toBe(3723);
        expect(parseYoutubeUrl('https://www.youtube.com/embed/dQw4w9WgXcQ?start=45')?.start).toBe(45);
        expect(parseYoutubeUrl('https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=0')?.start).toBeNull();
    });

    it('rejects anything that is not a YouTube video', () => {
        for (const input of [
            '',
            '   ',
            'https://vimeo.com/123456',
            'https://www.youtube.com/@somechannel',
            'https://www.youtube.com/watch?v',
            'https://www.youtube.com/watch?v=tooshort',
            'not a url',
        ]) {
            expect(parseYoutubeUrl(input), input).toBeNull();
        }
    });

    it('builds embed and watch URLs that agree with each other', () => {
        expect(buildEmbedUrl('dQw4w9WgXcQ', null)).toBe('https://www.youtube.com/embed/dQw4w9WgXcQ');
        expect(buildEmbedUrl('dQw4w9WgXcQ', 90)).toBe('https://www.youtube.com/embed/dQw4w9WgXcQ?start=90');
        expect(buildWatchUrl('dQw4w9WgXcQ', null)).toBe('https://www.youtube.com/watch?v=dQw4w9WgXcQ');
        expect(buildWatchUrl('dQw4w9WgXcQ', 90)).toBe('https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=90s');
        // round trip: a watch URL we render must parse back to the same target
        expect(parseYoutubeUrl(buildWatchUrl('dQw4w9WgXcQ', 90))).toEqual({
            videoId: 'dQw4w9WgXcQ',
            start: 90,
        });
    });
});

describe('YoutubeBlock node', () => {
    const createEditor = (content = '') =>
        new Editor({ extensions: [StarterKit, YoutubeBlock], content });

    const editor = createEditor(
        '<p>before</p><div data-youtube-video data-src="https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=30s">' +
            '<iframe src="https://www.youtube.com/embed/dQw4w9WgXcQ?start=30"></iframe></div><p>after</p>',
    );

    it('parses a stored embed back into a node with its id and start', () => {
        const nodes: any[] = [];
        editor.state.doc.descendants((node) => {
            if (node.type.name === 'youtube') nodes.push(node.attrs);
            return true;
        });
        expect(nodes).toEqual([{ videoId: 'dQw4w9WgXcQ', start: 30 }]);
    });

    it('serialises back to the same embed markup', () => {
        const html = editor.getHTML();
        expect(html).toContain('data-youtube-video="true"');
        expect(html).toContain('data-video-id="dQw4w9WgXcQ"');
        expect(html).toContain('src="https://www.youtube.com/embed/dQw4w9WgXcQ?start=30"');
        expect(html).toContain('allowfullscreen="true"');
        expect(html).toContain('aspect-ratio');
    });

    it('adopts a bare iframe pasted into the document', () => {
        const pasted = createEditor(
            '<p>x</p><iframe src="https://www.youtube.com/embed/dQw4w9WgXcQ"></iframe>',
        );
        const out = pasted.getHTML();
        expect(out).toContain('data-youtube-video="true"');
        expect(out).toContain('data-video-id="dQw4w9WgXcQ"');
    });
});
