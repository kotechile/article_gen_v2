import { describe, expect, it } from 'vitest';
import { ensureIntroKeyTakeaways } from './geoFormatting';

describe('ensureIntroKeyTakeaways', () => {
    it('drops malformed GEO fallback paragraphs and keeps only complete takeaways', () => {
        const html = `
            <h1>Real Estate Investment Strategy</h1>
            <h2>Key Takeaways</h2>
            <p>A comprehensive real estate investment strategy requires moving beyond simple mortgage rate comparisons, as a robust housing decision framework reveals that renting often outperforms buying over a five-year net worth… because that is one of the main drivers behind a strong real estate investment decision.</p>
            <p>Real estate investment decisions rarely hinge on mortgage rates alone. The true measure of wealth-building lies in a 5-year net worth framework that strips away conventional wisdom. This guide introduces a robust mental… and adapt that guidance to your budget, timing, and tolerance for trade-offs.</p>
            <ul>
                <li>Only 18% of renters calculate the 5-year net worth impact, yet that single number can redefine your entire real estate investment strategy.</li>
            </ul>
            <p>Real Estate Investment is a practical topic shaped by A comprehensive real estate investment strategy requires moving beyond simple mortgage rate comparisons, as a robust housing decision framework reveals that renting often outperforms buying over a five-year net worth…, so the best answer depends on your goals, constraints, and timing.</p>
            <h2>Body</h2>
            <p>Article body goes here.</p>
        `;

        const normalized = ensureIntroKeyTakeaways(html);
        const doc = new DOMParser().parseFromString(normalized, 'text/html');
        const section = doc.querySelector('section.geo-key-takeaways');

        expect(section).not.toBeNull();
        expect(section?.querySelectorAll('p')).toHaveLength(0);

        const items = Array.from(section?.querySelectorAll('li') || []).map((node) => node.textContent?.trim() || '');
        expect(items).toEqual([
            'Only 18% of renters calculate the 5-year net worth impact, yet that single number can redefine your entire real estate investment strategy.',
        ]);
        expect(section?.textContent || '').not.toContain('because that is one of the main drivers');
        expect(section?.textContent || '').not.toContain('and adapt that guidance');
        expect(section?.textContent || '').not.toContain('is a practical topic shaped by');
    });
});
