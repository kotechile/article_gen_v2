import { describe, expect, it, beforeEach, vi } from 'vitest';

const { fromMock } = vi.hoisted(() => ({
    fromMock: vi.fn(),
}));

vi.mock('../lib/supabase', () => ({
    supabase: {
        from: fromMock,
    },
}));

import { resolveLinkedWordPressCategoryIds } from './wordpressService';

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
