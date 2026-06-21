import { describe, expect, it } from 'vitest';
import { Editor } from '@tiptap/core';
import StarterKit from '@tiptap/starter-kit';
import { MathNode } from './ArticleEditor';

describe('MathNode Tiptap Extension', () => {
    const createEditor = () => {
        return new Editor({
            extensions: [
                StarterKit,
                MathNode,
            ],
        });
    };

    it('parses inline math span with data-math', () => {
        const editor = createEditor();
        editor.commands.setContent('<p>Hello <span data-math="a^2+b^2=c^2" data-display-mode="false">$a^2+b^2=c^2$</span> world</p>');
        
        expect(editor.getHTML()).toContain('data-math="a^2+b^2=c^2"');
        expect(editor.getHTML()).toContain('data-display-mode="false"');
        expect(editor.getHTML()).toContain('class="math-inline"');
    });

    it('parses block math span with data-math', () => {
        const editor = createEditor();
        editor.commands.setContent('<p>Hello <span data-math="x = y" data-display-mode="true">$$x = y$$</span> world</p>');
        
        expect(editor.getHTML()).toContain('data-math="x = y"');
        expect(editor.getHTML()).toContain('data-display-mode="true"');
        expect(editor.getHTML()).toContain('class="math-block"');
    });

    it('parses KaTeX html copied from Gemini/KaTeX pages', () => {
        const editor = createEditor();
        const katexHtml = `
            <p>Here is some KaTeX math:
            <span class="katex">
                <span class="katex-mathml">
                    <math xmlns="http://www.w3.org/1998/Math/MathML">
                        <semantics>
                            <mrow>
                                <msup><mi>e</mi><mrow><mi>i</mi><mi>π</mi></mrow></msup>
                                <mo>+</mo>
                                <mn>1</mn>
                                <mo>=</mo>
                                <mn>0</mn>
                            </mrow>
                            <annotation encoding="application/x-tex">e^{i\\pi} + 1 = 0</annotation>
                        </semantics>
                    </math>
                </span>
            </span>
            </p>
        `;
        editor.commands.setContent(katexHtml);
        expect(editor.getHTML()).toContain('data-math="e^{i\\pi} + 1 = 0"');
    });

    it('parses MathJax mjx-container copied from MathJax pages', () => {
        const editor = createEditor();
        const mjxHtml = `
            <p>Here is MathJax:
            <mjx-container class="MathJax mjx-display" jax="CHTML" data-tex="E = mc^2" display="true">
                <mjx-math class="MJX-TEX" aria-hidden="true">
                    ...
                </mjx-math>
            </mjx-container>
            </p>
        `;
        editor.commands.setContent(mjxHtml);
        expect(editor.getHTML()).toContain('data-math="E = mc^2"');
        expect(editor.getHTML()).toContain('data-display-mode="true"');
    });

    it('parses MathJax script elements', () => {
        const editor = createEditor();
        const scriptHtml = `
            <p>Equation is:
            <script type="math/tex">a \\le b</script>
            </p>
        `;
        editor.commands.setContent(scriptHtml);
        expect(editor.getHTML()).toContain('data-math="a \\le b"');
    });

    it('converts typed block LaTeX via input rules', () => {
        const editor = createEditor();
        editor.commands.setContent('<p>$$a^2+b^2=c^2$$</p>');
        const pos = editor.state.doc.content.size - 1;
        editor.commands.setTextSelection(pos);
        const handled = editor.view.someProp('handleTextInput', f => f(editor.view, pos, pos, ' '));
        expect(editor.getHTML()).toContain('data-math="a^2+b^2=c^2"');
        expect(editor.getHTML()).toContain('data-display-mode="true"');
    });

    it('converts typed inline LaTeX via input rules', () => {
        const editor = createEditor();
        editor.commands.setContent('<p>$x=y$</p>');
        const pos = editor.state.doc.content.size - 1;
        editor.commands.setTextSelection(pos);
        editor.view.someProp('handleTextInput', f => f(editor.view, pos, pos, ' '));
        expect(editor.getHTML()).toContain('data-math="x=y"');
        expect(editor.getHTML()).toContain('data-display-mode="false"');
    });

    it('defines paste rules with correct regex matching', () => {
        // Mock this.type to be passed to call()
        const mockContext = {
            type: {
                create: (attrs: any) => attrs,
            },
        };
        const pasteRules = MathNode.config.addPasteRules?.call(mockContext as any);
        expect(pasteRules).toBeDefined();
        expect(pasteRules?.length).toBe(1);
        
        const rule = pasteRules![0];
        const regex = rule.find;
        
        const text = 'Here is $$a^2+b^2=c^2$$ and $x=y$';
        const matches = [...text.matchAll(regex)];
        
        expect(matches.length).toBe(2);
        expect(matches[0][0]).toBe('$$a^2+b^2=c^2$$');
        expect(matches[0][1]).toBe('$$a^2+b^2=c^2$$');
        expect(matches[1][0]).toBe('$x=y$');
        expect(matches[1][1]).toBe('$x=y$');
    });
});
