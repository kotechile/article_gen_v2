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

    it('converts markdown bold **text** in takeaways to HTML strong tags', () => {
        const html = `
            <h1>Site Selection Guide</h1>
            <h2>Key Takeaways</h2>
            <ul>
                <li>• **Built after 2010:** Modern building codes require significantly higher seismic and electrical standards.</li>
                <li>• **Underground utilities:** Always check public easements before breaking ground on expansion plans.</li>
            </ul>
            <p>Here is the full guide to evaluating commercial properties.</p>
        `;

        const normalized = ensureIntroKeyTakeaways(html);
        const doc = new DOMParser().parseFromString(normalized, 'text/html');
        const listItems = Array.from(doc.querySelectorAll('section.geo-key-takeaways li'));

        expect(listItems).toHaveLength(2);
        expect(listItems[0]?.innerHTML).toContain('<strong>Built after 2010:</strong>');
        expect(listItems[0]?.innerHTML).not.toContain('**');
        expect(listItems[0]?.innerHTML).not.toContain('•');
        expect(listItems[1]?.innerHTML).toContain('<strong>Underground utilities:</strong>');
        expect(listItems[1]?.innerHTML).not.toContain('**');
    });

    it('does not duplicate takeaways when li elements contain p tags (TipTap format) across multiple saves', () => {
        const html = `
            <h1>Relocation to other countries</h1>
            <h2>At a glance</h2>
            <ul>
                <li><p>The smartest way to decide if <a href="https://example.com/move">relocating for a job is worth it</a> is to look past cost-of-living salary math and weigh all trade-offs.</p></li>
            </ul>
            <p>Full article introduction and body text goes here.</p>
        `;

        // First pass (e.g. initial load)
        const pass1 = ensureIntroKeyTakeaways(html);
        const doc1 = new DOMParser().parseFromString(pass1, 'text/html');
        const items1 = Array.from(doc1.querySelectorAll('section.geo-key-takeaways li'));
        expect(items1).toHaveLength(1);
        expect(items1[0]?.innerHTML).toContain('relocating for a job is worth it');

        // Second pass (e.g. first save)
        const pass2 = ensureIntroKeyTakeaways(pass1);
        const doc2 = new DOMParser().parseFromString(pass2, 'text/html');
        const items2 = Array.from(doc2.querySelectorAll('section.geo-key-takeaways li'));
        expect(items2).toHaveLength(1);

        // Third pass (e.g. repeated save)
        const pass3 = ensureIntroKeyTakeaways(pass2);
        const doc3 = new DOMParser().parseFromString(pass3, 'text/html');
        const items3 = Array.from(doc3.querySelectorAll('section.geo-key-takeaways li'));
        expect(items3).toHaveLength(1);
        expect(doc3.body.innerHTML).toContain('Full article introduction and body text goes here.');
    });

    it('preserves all introduction and body paragraphs between At a glance and an inserted image across saves', () => {
        const html = `
            <h2>At a glance</h2>
            <ul>
                <li>Adoption hit a record: 40% of billion-dollar companies use AI agents.</li>
                <li>The build-versus-buy line flipped.</li>
                <li>The fix requires workflow redesign.</li>
            </ul>
            <p>Introduction paragraph: AI in the enterprise is experiencing a seismic shift.</p>
            <p>First, consider how executive teams are responding to automation budgets.</p>
            <img src="https://example.com/build-vs-buy.jpg" alt="Build vs Buy" />
            <p>Second, treat the build-versus-buy choice as a daily question rather than a yearly review.</p>
        `;

        // Save pass 1 (e.g. initial load / TipTap normalize)
        const pass1 = ensureIntroKeyTakeaways(html);
        expect(pass1).toContain('Introduction paragraph: AI in the enterprise');
        expect(pass1).toContain('First, consider how executive teams');
        expect(pass1).toContain('Second, treat the build-versus-buy');
        expect(pass1).toContain('https://example.com/build-vs-buy.jpg');

        // Save pass 2 (e.g. autosave after inserting image)
        const pass2 = ensureIntroKeyTakeaways(pass1);
        expect(pass2).toContain('Introduction paragraph: AI in the enterprise');
        expect(pass2).toContain('First, consider how executive teams');
        expect(pass2).toContain('Second, treat the build-versus-buy');
        expect(pass2).toContain('https://example.com/build-vs-buy.jpg');

        // Verify DOM structure
        const doc = new DOMParser().parseFromString(pass2, 'text/html');
        const paragraphs = Array.from(doc.querySelectorAll('p')).map((p) => p.textContent?.trim());
        expect(paragraphs).toContain('Introduction paragraph: AI in the enterprise is experiencing a seismic shift.');
        expect(paragraphs).toContain('First, consider how executive teams are responding to automation budgets.');
        expect(paragraphs).toContain('Second, treat the build-versus-buy choice as a daily question rather than a yearly review.');
    });
});
