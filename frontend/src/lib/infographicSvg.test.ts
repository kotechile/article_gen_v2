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

    const partiallyValidMalformedTableHtml = `
        <table>
            <thead>
                <tr>
                    <th>Pillar</th>
                    <th>What It Measures</th>
                    <th>Key Question</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Skills Audit</td>
                    <td>Portability of your core competencies</td>
                    <td>Can I transfer at least 40% of what I do well?</td>
                </tr>
                <tr>
                    <td>Earnings Equation</td>
                    <td>Income trajectory vs.
                    <h2>The Skills Transfer Audit</h2>
                    <p>Here is the body that should not stay inside the cell.</p>
    `;

    it('moves swallowed trailing content out of the last table cell for the editor', () => {
        const normalized = normalizeInfographicHtmlForEditor(malformedTableHtml);
        const doc = new DOMParser().parseFromString(normalized, 'text/html');
        const bodyParagraphs = Array.from(doc.body.querySelectorAll('p')).map((node) => node.textContent || '');
        const tableText = doc.querySelector('table')?.textContent || '';

        expect(doc.querySelector('table h2')).toBeNull();
        expect(doc.querySelector('table + h2')?.textContent).toBe('The Experiment Phase');
        expect(bodyParagraphs.some((text) => text.includes('career pivot'))).toBe(true);
        expect(doc.querySelector('table td')?.textContent).toContain('Healthcare support');
        expect(tableText).not.toContain('The Experiment Phase');
        expect(tableText).not.toContain('career pivot');
    });

    it('applies the same repair on the export/beautify path', () => {
        const beautified = beautifyTablesHtml(malformedTableHtml);
        const doc = new DOMParser().parseFromString(beautified, 'text/html');
        const tableText = doc.querySelector('table')?.textContent || '';

        expect(doc.querySelector('table h2')).toBeNull();
        expect(doc.querySelector('table + h2')?.textContent).toBe('The Experiment Phase');
        expect(tableText).not.toContain('The Experiment Phase');
    });

    it('repairs malformed content when the swallowing cell is not the only populated cell in the row', () => {
        const normalized = normalizeInfographicHtmlForEditor(partiallyValidMalformedTableHtml);
        const doc = new DOMParser().parseFromString(normalized, 'text/html');
        const tableText = doc.querySelector('table')?.textContent || '';

        expect(tableText).toContain('Earnings Equation');
        expect(tableText).toContain('Income trajectory vs.');
        expect(tableText).not.toContain('The Skills Transfer Audit');
        expect(doc.querySelector('table + h2')?.textContent).toBe('The Skills Transfer Audit');
        expect(doc.body.textContent || '').toContain('should not stay inside the cell');
    });
});
