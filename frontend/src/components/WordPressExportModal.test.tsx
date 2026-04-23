import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { describe, expect, it, vi, beforeEach } from 'vitest';
import { WordPressExportModal } from './WordPressExportModal';

vi.mock('../utils/seoUtils', () => ({
    computeSEOQualityScore: () => ({
        canPublish: true,
        score: 100,
        grade: 'A',
        checks: [{ label: 'Ready', passed: true }],
    }),
}));

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
});
