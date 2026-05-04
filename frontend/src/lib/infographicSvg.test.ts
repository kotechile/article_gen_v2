import { describe, expect, it } from 'vitest';
import { beautifyTablesHtml, normalizeInfographicHtmlForEditor } from './infographicSvg';

describe('table repair in article HTML', () => {
    const malformedTableHtml = `
        <table>
            <thead>
                <tr>
                    <th>Career Path</th>
                    <th>Typical Salary Growth Pattern</th>
                    <th>Years to Peak</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Healthcare support (e.g.
                    <h2>The Experiment Phase</h2>
                    <p>How to test a career pivot without torching your life.</p>
    `;

    it('moves swallowed trailing content out of the last table cell for the editor', () => {
        const normalized = normalizeInfographicHtmlForEditor(malformedTableHtml);
        const doc = new DOMParser().parseFromString(normalized, 'text/html');
        const bodyParagraphs = Array.from(doc.body.querySelectorAll('p')).map((node) => node.textContent || '');

        expect(doc.querySelector('table h2')).toBeNull();
        expect(doc.querySelector('table + h2')?.textContent).toBe('The Experiment Phase');
        expect(bodyParagraphs.some((text) => text.includes('career pivot'))).toBe(true);
        expect(doc.querySelector('table td')?.textContent).toContain('Healthcare support');
    });

    it('applies the same repair on the export/beautify path', () => {
        const beautified = beautifyTablesHtml(malformedTableHtml);
        const doc = new DOMParser().parseFromString(beautified, 'text/html');

        expect(doc.querySelector('table h2')).toBeNull();
        expect(doc.querySelector('table + h2')?.textContent).toBe('The Experiment Phase');
    });
});
