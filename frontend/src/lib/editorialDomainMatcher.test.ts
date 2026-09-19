import { describe, it } from 'node:test';
import assert from 'node:assert';
import { findBestMatchingDomain } from './editorialDomainMatcher.ts';
import type { Project } from '../types/index.ts';
import type { EditorialArticle } from '../services/editorial-factory.service.ts';

describe('editorialDomainMatcher', () => {
    const mockProjects: Project[] = [
        {
            id: 'proj-1',
            created_at: '2026-01-01',
            user_id: 'user-1',
            app_name: 'WellRoost',
            domain: 'wellroost.com',
            site_description: 'Home improvement, renovation, DIY, HVAC, energy rebates, heat pumps, roofing',
            categories: 'Home Improvement, Renovation, HVAC',
            target_keywords: ['home renovation', 'heat pump rebate', 'roofing'],
        },
        {
            id: 'proj-2',
            created_at: '2026-01-01',
            user_id: 'user-1',
            app_name: 'Giniloh',
            domain: 'giniloh.com',
            site_description: 'AI technology, developer tools, GPUs, semiconductors, LLMs, neural networks',
            categories: 'Artificial Intelligence, Tech, Hardware',
            target_keywords: ['ai agents', 'gpu hardware', 'llm architecture'],
        },
    ];

    it('should auto-match heat pump home article to wellroost.com based on tag and title', () => {
        const article: EditorialArticle = {
            id: 'art-1',
            title: 'New York heat pump rebate pays double to sealed homes',
            tags: ['home_systems_reno'],
            summary: 'Homeowners can now claim high-efficiency heat pump incentives.',
            content: 'Sealed homes and HVAC upgrades are eligible for expanded state rebates.',
            author: 'Editorial Factory',
            created_at: '2026-09-17',
        };

        const result = findBestMatchingDomain(article, mockProjects, 'giniloh.com');
        assert.strictEqual(result.domain, 'wellroost.com');
        assert.strictEqual(result.isAutoMatched, true);
        assert.strictEqual(result.confidence, 'high');
    });

    it('should auto-match AI hardware article to giniloh.com', () => {
        const article: EditorialArticle = {
            id: 'art-2',
            title: "OpenAI's custom chip beats Nvidia on power",
            tags: ['gpu_hardware'],
            summary: 'New custom AI accelerator outperforms leading GPUs in power efficiency.',
            content: 'AI data centers are seeking lower power alternatives to standard Nvidia chips.',
            author: 'Editorial Factory',
            created_at: '2026-09-17',
        };

        const result = findBestMatchingDomain(article, mockProjects, 'wellroost.com');
        assert.strictEqual(result.domain, 'giniloh.com');
        assert.strictEqual(result.isAutoMatched, true);
        assert.strictEqual(result.confidence, 'high');
    });

    it('should auto-match agentic AI article to giniloh.com', () => {
        const article: EditorialArticle = {
            id: 'art-3',
            title: 'OpenAI just made the agent loop a commodity',
            tags: ['agentic_ai'],
            summary: 'Autonomous agent frameworks are standardizing around common primitives.',
            content: 'Software developers can now build reliable agent loops effortlessly.',
            author: 'Editorial Factory',
            created_at: '2026-09-17',
        };

        const result = findBestMatchingDomain(article, mockProjects, 'wellroost.com');
        assert.strictEqual(result.domain, 'giniloh.com');
        assert.strictEqual(result.isAutoMatched, true);
    });

    it('should fall back to active project if no specific niche matches', () => {
        const article: EditorialArticle = {
            id: 'art-4',
            title: 'Exploring the Philosophy of Ancient Architecture',
            tags: ['philosophy_history'],
            summary: 'A look into ancient structures and thought.',
            content: 'History of early civilizations and philosophical foundations.',
            author: 'Editorial Factory',
            created_at: '2026-09-17',
        };

        const result = findBestMatchingDomain(article, mockProjects, 'giniloh.com');
        assert.strictEqual(result.domain, 'giniloh.com');
        assert.strictEqual(result.confidence, 'none');
        assert.strictEqual(result.isAutoMatched, false);
    });
});
