import { describe, expect, it, beforeEach, afterEach, vi } from 'vitest';

const { fromMock } = vi.hoisted(() => ({
    fromMock: vi.fn(),
}));

vi.mock('../lib/supabase', () => ({
    supabase: {
        from: fromMock,
    },
}));

import { resolveLinkedWordPressCategoryIds, processAndUploadInlineImages } from './wordpressService';
import type { WordPressSite } from '../types/wordpress';

const buildMaybeSingleQuery = (data: any) => {
    const query = {
        select: vi.fn(() => query),
        eq: vi.fn(() => query),
        maybeSingle: vi.fn().mockResolvedValue({ data }),
    };
    return query;
};

const buildInQuery = (data: any[]) => {
    const query = {
        select: vi.fn(() => query),
        eq: vi.fn(() => query),
        in: vi.fn().mockResolvedValue({ data }),
    };
    return query;
};

describe('resolveLinkedWordPressCategoryIds', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('returns secondary then primary categories from idea metadata context when topic linkage is missing', async () => {
        fromMock.mockImplementation((table: string) => {
            if (table === 'project_categories') {
                return buildInQuery([
                    {
                        id: 'primary-cat',
                        wordpress_category_id: 101,
                        wordpress_site_domain: 'giniloh.com',
                    },
                    {
                        id: 'secondary-cat',
                        wordpress_category_id: 202,
                        wordpress_site_domain: 'giniloh.com',
                    },
                ]);
            }

            throw new Error(`Unexpected table ${table}`);
        });

        const result = await resolveLinkedWordPressCategoryIds(
            {
                idea_metadata: {
                    category_context: {
                        project_id: 'project-1',
                        primary_category_id: 'primary-cat',
                        secondary_category_id: 'secondary-cat',
                    },
                },
            },
            'https://giniloh.com/'
        );

        expect(result).toEqual([202, 101]);
    });

    it('uses source idea metadata fallback before topic lookup when article is missing local category context', async () => {
        fromMock.mockImplementation((table: string) => {
            if (table === 'content_ideas') {
                return buildMaybeSingleQuery({
                    topic_id: null,
                    idea_metadata: {
                        category_context: {
                            project_id: 'project-2',
                            primary_category_id: 'parent-cat',
                            secondary_category_id: 'child-cat',
                        },
                    },
                });
            }

            if (table === 'project_categories') {
                return buildInQuery([
                    {
                        id: 'parent-cat',
                        wordpress_category_id: 11,
                        wordpress_site_domain: 'giniloh.com',
                    },
                    {
                        id: 'child-cat',
                        wordpress_category_id: 22,
                        wordpress_site_domain: 'giniloh.com',
                    },
                ]);
            }

            throw new Error(`Unexpected table ${table}`);
        });

        const result = await resolveLinkedWordPressCategoryIds(
            {
                source_idea_id: 'idea-1',
            },
            'giniloh.com'
        );

        expect(result).toEqual([22, 11]);
    });
});

describe('processAndUploadInlineImages', () => {
    let mockFetch: any;
    let originalImage: any;
    let originalFetch: any;

    const dummySite = {
        id: 1,
        domain: 'kotechile.cl',
        wpUserName: 'admin',
        wordpress_key: 'key123',
        user_id: 'user-1',
        seo_plugin: 'yoast',
    } as any as WordPressSite;

    beforeEach(() => {
        originalImage = globalThis.Image;
        originalFetch = globalThis.fetch;

        mockFetch = vi.fn();
        globalThis.fetch = mockFetch;

        // Mock Image class to trigger onload immediately
        globalThis.Image = class {
            onload: () => void = () => {};
            onerror: () => void = () => {};
            _src: string = '';
            width: number = 100;
            height: number = 100;
            naturalWidth: number = 100;
            naturalHeight: number = 100;
            get src() { return this._src; }
            set src(value: string) {
                this._src = value;
                setTimeout(() => {
                    if (this.onload) this.onload();
                }, 0);
            }
        } as any;

        // Mock canvas.toBlob
        if (!HTMLCanvasElement.prototype.toBlob) {
            HTMLCanvasElement.prototype.toBlob = function(callback: any) {
                callback(new Blob(['mock blob'], { type: 'image/jpeg' }));
            };
        }
    });

    afterEach(() => {
        globalThis.Image = originalImage;
        globalThis.fetch = originalFetch;
    });

    it('skips data URIs and existing WordPress uploads on the same domain, but uploads Supabase and other external images', async () => {
        // Mock download fetch (returning image blob)
        const mockBlob = new Blob(['mock binary content'], { type: 'image/jpeg' });
        
        mockFetch.mockImplementation(async (url: string) => {
            if (url.endsWith('/wp-json/wp/v2/media')) {
                // WordPress media upload response
                return {
                    ok: true,
                    json: async () => ({
                        id: 999,
                        source_url: 'https://kotechile.cl/wp-content/uploads/2026/06/uploaded-image.jpg'
                    })
                };
            }
            // Image download fetch
            return {
                ok: true,
                blob: async () => mockBlob
            };
        });

        const htmlInput = `
            <div>
                <img id="img-data" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==" alt="Data URI" />
                <img id="img-wp-same" src="https://kotechile.cl/wp-content/uploads/2026/05/logo.jpg" alt="WP Same Domain" />
                <img id="img-wp-diff" src="https://otherdomain.com/wp-content/uploads/2026/05/photo.jpg" alt="WP Diff Domain" />
                <img id="img-supabase" src="https://kotechile.supabase.co/storage/v1/object/public/User%20Files/pic.jpg" alt="Supabase Image" />
            </div>
        `;

        const result = await processAndUploadInlineImages(htmlInput, dummySite);

        // Resulting HTML parser
        const parser = new DOMParser();
        const doc = parser.parseFromString(result.html, 'text/html');

        // 1. Data URI should be untouched
        expect(doc.querySelector('#img-data')?.getAttribute('src')).toBe(
            'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='
        );

        // 2. Same-domain WordPress upload should be untouched
        expect(doc.querySelector('#img-wp-same')?.getAttribute('src')).toBe(
            'https://kotechile.cl/wp-content/uploads/2026/05/logo.jpg'
        );

        // 3. Different domain WordPress upload should be replaced with the uploaded url
        expect(doc.querySelector('#img-wp-diff')?.getAttribute('src')).toBe(
            'https://kotechile.cl/wp-content/uploads/2026/06/uploaded-image.jpg'
        );

        // 4. Supabase image should be replaced with the uploaded url
        expect(doc.querySelector('#img-supabase')?.getAttribute('src')).toBe(
            'https://kotechile.cl/wp-content/uploads/2026/06/uploaded-image.jpg'
        );

        // Verify media upload was called twice (once for WP diff, once for Supabase)
        const uploadCalls = mockFetch.mock.calls.filter((call: any) => call[0].endsWith('/wp-json/wp/v2/media'));
        expect(uploadCalls.length).toBe(2);
        expect(result.warnings.length).toBe(0);
    });
});
