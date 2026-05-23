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

    it('keeps key takeaways above a generated introduction heading', () => {
        const html = `
            <h1>Cost Breaking Lease: Termination Fees vs. Subletting in 2026</h1>
            <h2>Introduction: The True Cost of Breaking a Lease in 2026</h2>
            <h2>Key Takeaways</h2>
            <ul>
                <li>Breaking a lease typically costs between one and three months' rent, but total expenses can climb higher when you factor in lost deposits and ongoing rent obligations.</li>
                <li>Some states legally cap what landlords can charge, while others let the lease agreement dictate the penalty.</li>
                <li>You almost always have the right to request a written breakdown of fees, and using a lease termination fee calculator before making any moves can save you from nasty surprises.</li>
            </ul>
            <p>Here's the short answer to a question nobody wants to ask: the cost breaking lease typically lands between two and four months' rent.</p>
        `;

        const normalized = ensureIntroKeyTakeaways(html);
        const doc = new DOMParser().parseFromString(normalized, 'text/html');
        const children = Array.from(doc.body.children);

        expect(children[0]?.tagName).toBe('H1');
        expect(children[1]?.tagName).toBe('SECTION');
        expect(children[1]?.classList.contains('geo-key-takeaways')).toBe(true);
        expect(children[2]?.tagName).toBe('H2');
        expect(children[2]?.textContent?.trim()).toBe('Introduction: The True Cost of Breaking a Lease in 2026');
    });
});
