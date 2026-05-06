import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { describe, expect, it, vi, beforeEach } from 'vitest';
import { WordPressExportModal } from './WordPressExportModal';

vi.mock('../utils/seoUtils', () => ({
    computeSEOQualityScore: vi.fn(() => ({
        canPublish: true,
        score: 100,
        grade: 'A',
        checks: [{ label: 'Ready', passed: true }],
    })),
}));

import { computeSEOQualityScore } from '../utils/seoUtils';

vi.mock('../services/wordpressService', () => ({
    fetchWordPressSites: vi.fn(),
    fetchWordPressCategories: vi.fn(),
    publishToWordPress: vi.fn(),
    saveWordPressSettings: vi.fn(),
    loadWordPressSettings: vi.fn(),
    resolveLinkedWordPressCategoryIds: vi.fn(),
}));

import {
    fetchWordPressSites,
    fetchWordPressCategories,
    publishToWordPress,
    saveWordPressSettings,
    loadWordPressSettings,
    resolveLinkedWordPressCategoryIds,
} from '../services/wordpressService';

const fetchWordPressSitesMock = vi.mocked(fetchWordPressSites);
const fetchWordPressCategoriesMock = vi.mocked(fetchWordPressCategories);
const publishToWordPressMock = vi.mocked(publishToWordPress);
const saveWordPressSettingsMock = vi.mocked(saveWordPressSettings);
const loadWordPressSettingsMock = vi.mocked(loadWordPressSettings);
const resolveLinkedWordPressCategoryIdsMock = vi.mocked(resolveLinkedWordPressCategoryIds);
const computeSEOQualityScoreMock = vi.mocked(computeSEOQualityScore);

const baseProps = {
    articleData: {
        Title: 'Sample GEO Article',
        thesis: 'Short and direct answer.',
        primary_keywords: ['sample keyword'],
        wordpress_category_id: 10,
    },
    articleId: 'article-123',
    userId: 'user-123',
    onClose: vi.fn(),
    onSuccess: vi.fn(),
};

const mountModal = () => render(<WordPressExportModal {...baseProps} />);

describe('WordPressExportModal loopback banner', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        computeSEOQualityScoreMock.mockReturnValue({
            canPublish: true,
            score: 100,
            grade: 'A',
            checks: [{ label: 'Ready', passed: true }],
        } as any);
        fetchWordPressSitesMock.mockResolvedValue([
            {
                id: 1,
                user_id: 'user-123',
                domain: 'example.com',
                wpUserName: 'wp-user',
                wordpress_key: 'app-key',
            } as any,
        ]);
        fetchWordPressCategoriesMock.mockResolvedValue([
            { id: 10, name: 'Category A', slug: 'category-a' } as any,
        ]);
        loadWordPressSettingsMock.mockResolvedValue({
            siteId: 1,
            categoryId: 10,
            postStatus: 'publish',
        } as any);
        resolveLinkedWordPressCategoryIdsMock.mockResolvedValue([]);
        saveWordPressSettingsMock.mockResolvedValue(undefined);
    });

    it('shows success banner with saved field count', async () => {
        publishToWordPressMock.mockResolvedValue({
            link: 'https://example.com/post',
            loopback_summary: {
                success: true,
                attemptedFields: ['status', 'published'],
                savedFields: ['status', 'published'],
                removedFields: [],
            },
        } as any);

        mountModal();
        await waitFor(() =>
            expect(screen.getByRole('button', { name: /publish to wordpress/i })).toBeEnabled()
        );

        fireEvent.click(screen.getByRole('button', { name: /publish to wordpress/i }));

        await screen.findByText('Published and synced 2 Titles fields.');
        expect(publishToWordPressMock).toHaveBeenCalledTimes(1);
        await waitFor(
            () => expect(baseProps.onSuccess).toHaveBeenCalledWith('https://example.com/post'),
            { timeout: 2500 }
        );
    });

    it('shows warning banner when loopback fallback removes columns', async () => {
        publishToWordPressMock.mockResolvedValue({
            link: 'https://example.com/post',
            loopback_summary: {
                success: false,
                attemptedFields: ['status', 'published', 'metaTitle'],
                savedFields: ['status', 'published'],
                removedFields: ['metaTitle'],
            },
        } as any);

        mountModal();
        await waitFor(() =>
            expect(screen.getByRole('button', { name: /publish to wordpress/i })).toBeEnabled()
        );

        fireEvent.click(screen.getByRole('button', { name: /publish to wordpress/i }));

        await screen.findByText(
            'Published, but 1 loopback fields were skipped due to schema mismatch.'
        );
        expect(publishToWordPressMock).toHaveBeenCalledTimes(1);
    });

    it('shows generic warning banner when loopback fails without removed fields', async () => {
        publishToWordPressMock.mockResolvedValue({
            link: 'https://example.com/post',
            loopback_summary: {
                success: false,
                attemptedFields: ['status', 'published'],
                savedFields: [],
                removedFields: [],
            },
        } as any);

        mountModal();
        await waitFor(() =>
            expect(screen.getByRole('button', { name: /publish to wordpress/i })).toBeEnabled()
        );

        fireEvent.click(screen.getByRole('button', { name: /publish to wordpress/i }));

        await screen.findByText('Published, but Titles loopback update had issues.');
        expect(publishToWordPressMock).toHaveBeenCalledTimes(1);
    });

    it('publishes both parent and child WordPress categories when both are available', async () => {
        fetchWordPressCategoriesMock.mockResolvedValue([
            { id: 10, name: 'Parent Category', slug: 'parent-category' } as any,
            { id: 20, name: 'Child Category', slug: 'child-category', parent: 10 } as any,
        ]);
        resolveLinkedWordPressCategoryIdsMock.mockResolvedValue([20, 10]);
        publishToWordPressMock.mockResolvedValue({
            link: 'https://example.com/post',
            loopback_summary: {
                success: true,
                attemptedFields: ['status'],
                savedFields: ['status'],
                removedFields: [],
            },
        } as any);

        render(
            <WordPressExportModal
                {...baseProps}
                articleData={{
                    ...baseProps.articleData,
                    wordpress_category_id: 20,
                    wordpress_parent_category_id: 10,
                }}
            />
        );

        await waitFor(() =>
            expect(screen.getByRole('button', { name: /publish to wordpress/i })).toBeEnabled()
        );

        fireEvent.click(screen.getByRole('button', { name: /publish to wordpress/i }));

        await waitFor(() =>
            expect(publishToWordPressMock).toHaveBeenCalledWith(
                expect.anything(),
                expect.objectContaining({
                    wordpress_category_id: 20,
                    wordpress_parent_category_id: 10,
                }),
                expect.objectContaining({
                    categoryIds: [20, 10],
                }),
                expect.anything()
            )
        );
    });

    it('explains the difference between generation readiness and GEO export readiness', async () => {
        computeSEOQualityScoreMock.mockReturnValue({
            canPublish: true,
            score: 55,
            grade: 'D',
            checks: [{ label: 'GEO Readiness', passed: false }],
        } as any);

        mountModal();

        await screen.findByText(/one secondary keyword is enough for generation/i);
        expect(
            screen.getByText(/this GEO export check expects 3\+ secondary keywords or research mode/i)
        ).toBeInTheDocument();
    });

    it('passes the legacy featuredImageURL field through to the publish service', async () => {
        publishToWordPressMock.mockResolvedValue({
            link: 'https://example.com/post',
            loopback_summary: {
                success: true,
                attemptedFields: ['status'],
                savedFields: ['status'],
                removedFields: [],
            },
        } as any);

        render(
            <WordPressExportModal
                {...baseProps}
                articleData={{
                    ...baseProps.articleData,
                    featuredImageURL: 'https://cdn.example.com/featured.jpg',
                    mediaAltText: 'Alt copy',
                    mediaTitle: 'Title copy',
                    mediaCaption: 'Caption copy',
                }}
            />
        );

        await waitFor(() =>
            expect(screen.getByRole('button', { name: /publish to wordpress/i })).toBeEnabled()
        );

        fireEvent.click(screen.getByRole('button', { name: /publish to wordpress/i }));

        await waitFor(() =>
            expect(publishToWordPressMock).toHaveBeenCalledWith(
                expect.anything(),
                expect.anything(),
                expect.objectContaining({
                    featuredImageUrl: 'https://cdn.example.com/featured.jpg',
                    featuredImageMetadata: expect.objectContaining({
                        alt: 'Alt copy',
                        title: 'Title copy',
                        caption: 'Caption copy',
                    }),
                }),
                expect.anything()
            )
        );
    });
});
