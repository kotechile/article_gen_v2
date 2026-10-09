import React, { useState, useEffect } from 'react';
import { Loader2, Sparkles, RefreshCw, Upload, X, Wand2, Palette, Check } from 'lucide-react';
import {
    generateAIImage,
    getImageProviderModels,
    getImageApplicationConfig,
    synthesizeImagePrompt,
    generateOverlayCopy,
    applyImageOverlay
} from '../../services/imageService';
import type { ImageProviderModel, ImageMetadata, OverlayConfig, OverlayDetails } from '../../types/image';
import { EditorialOverlayPanel } from './EditorialOverlayPanel';

export interface StylePreset {
    id: string;
    name: string;
    description: string;
    promptModifier: string;
    isEditorial?: boolean;
}

export const STYLE_PRESETS: StylePreset[] = [
    // Curated Editorial Treatments (Executive Publication & Magazine Cover Standard)
    {
        id: 'cinematic_still',
        name: 'Cinematic Still',
        description: '35mm anamorphic establishing shot, atmospheric depth, practical lighting, teal & amber palette',
        promptModifier: 'cinematic film still, anamorphic 35mm look, wide establishing composition, single strong practical light source, crisp atmospheric depth, restrained teal-and-amber palette, no people facing camera',
        isEditorial: true,
    },
    {
        id: 'editorial_macro',
        name: 'Editorial Macro',
        description: '100mm macro photograph, single razor-sharp plane, physical surface texture & shallow DOF',
        promptModifier: 'extreme close-up macro photograph, 100mm macro lens, one razor-sharp focal plane, shallow depth of field, visible surface texture and dust, soft directional daylight, hero object off-centre on the thirds with background falling away',
        isEditorial: true,
    },
    {
        id: 'technical_isometric',
        name: 'Technical Isometric Cutaway',
        description: 'Axonometric cutaway schematic, clean line weight, physical hardware modules on real surface',
        promptModifier: 'clean isometric cutaway illustration, technical drawing style, axonometric projection, flat muted palette with one accent colour, thin consistent line weight, laid over a real material surface, recognisable hardware with racks, modules, connectors, one directional light, generous empty margin',
        isEditorial: true,
    },
    {
        id: 'clay_render',
        name: 'Matte 3D Render',
        description: 'Matte clay render of mechanical assembly on weathered concrete or brushed steel',
        promptModifier: 'matte clay 3D render of a small mechanical assembly resting on a real textured surface in a real space, engineered parts, modular housing, moulding seams and contact shadows on weathered concrete or brushed steel, directional key light raking across, matte muted palette, no text',
        isEditorial: true,
    },
    {
        id: 'component_assembly',
        name: 'Modular Component Assembly',
        description: 'Minimalist studio assembly on real surface, directional cast shadows, generous negative space',
        promptModifier: 'minimalist studio composition of a modular mechanical assembly on a real surface, one directional light, generous negative space, hard clean edges on weathered concrete or brushed steel, long cast shadow, matte muted palette, no text',
        isEditorial: true,
    },
    {
        id: 'paper_collage',
        name: 'Editorial Paper Collage',
        description: 'Minimalist cut-paper silhouettes, halftone newsprint texture, physical cast shadows',
        promptModifier: 'minimalist editorial cut-paper collage, crisp cut-out object silhouettes, halftone newsprint texture, hand-torn rag paper with visible fibre, physical cast shadows under each layer, one raking light, muted modern editorial palette, crisp clean edges, generous negative space',
        isEditorial: true,
    },
    {
        id: 'long_lens_industry',
        name: 'Compressed Telephoto Industry',
        description: '200mm telephoto compression of industrial scale & infrastructure, crystal distance clarity',
        promptModifier: 'telephoto compression, 200mm long-lens view of industrial infrastructure, stacked overlapping layers of structure with crystal-clear distance clarity, flat compressed perspective, sharp directional lighting and deep industrial contrast, no people in foreground',
        isEditorial: true,
    },
    {
        id: 'document_flatlay',
        name: 'Document Still Life',
        description: 'Overhead 90-degree flat-lay of paper documents on desk, even diffused daylight',
        promptModifier: 'overhead flat-lay photograph of paper documents on a plain desk surface, top-down 90-degree view, even diffused daylight, one object slightly out of alignment to look handled, blank or illegibly cropped paper, muted paper tones',
        isEditorial: true,
    },
    {
        id: 'studio_object',
        name: 'Studio Product Shot',
        description: 'Hero product photography on textured surface, softbox key light & contact shadow',
        promptModifier: 'studio product photograph of one hero object on a real studio surface, single directional softbox key light with visible falloff and a long cast contact shadow across a textured surface, three-quarter angle, catalogue clarity, no text',
        isEditorial: true,
    },
    {
        id: 'architectural_night',
        name: 'Lit Architecture at Dusk',
        description: 'Blue hour architectural photograph, lit windows as warm light, calm atmosphere',
        promptModifier: 'architectural photograph of a modern building or plant at blue hour, lit windows as the only warm light, long-exposure calm with no moving figures, deep blue ambient light, clean geometry',
        isEditorial: true,
    },
    // Creative Presets
    {
        id: 'whiteboard',
        name: 'Whiteboard',
        description: 'Clean markers, sketch notes, clear outlines',
        promptModifier: 'clean whiteboard drawing, markers, clear outlines, hand-drawn sketch illustration, whiteboard notes on crisp white background',
    },
    {
        id: 'watercolor',
        name: 'Watercolor',
        description: 'Artistic brush strokes, soft fluid wash',
        promptModifier: 'vibrant watercolor painting, expressive brushstrokes, soft fluid wash, textured watercolor paper, artistic illustration',
    },
    {
        id: 'oil_painting',
        name: 'Oil Painting',
        description: 'Rich impasto textures, classic fine art',
        promptModifier: 'classical oil painting, visible canvas texture, rich impasto brushwork, fine art masterpiece, museum quality',
    },
    {
        id: 'line_art',
        name: 'Line Art',
        description: 'Crisp vector line art, clean outlines',
        promptModifier: 'crisp vector line art, clean black and white outlines, minimalist illustration, modern elegant contour drawing',
    },
    {
        id: 'cyberpunk',
        name: 'Cyberpunk',
        description: 'Neon lighting, futuristic high-tech glow',
        promptModifier: 'cyberpunk style, vibrant neon glow, futuristic night atmosphere, teal and magenta palette, sci-fi aesthetic',
    },
    {
        id: 'flat_design',
        name: 'Flat Design',
        description: 'Bold geometric shapes, modern 2D vector',
        promptModifier: 'modern flat design illustration, bold geometric shapes, clean vector graphics, vibrant harmonious color palette, 2D minimalist vector',
    },
    {
        id: 'vintage',
        name: 'Vintage Photography',
        description: 'Analog film grain, 70s Kodachrome tones',
        promptModifier: 'vintage analog 35mm film photography, authentic film grain, 1970s Kodachrome color grading, subtle retro light leak',
    },
];

interface AIImageGenerationProps {
    userId: string;
    selectedText?: string;
    onImageGenerated: (imageUrl: string, metadata: Partial<ImageMetadata>) => void;
    articleContext?: {
        title?: string;
        thesis?: string;
        hook?: string;
        deck?: string;
        excerpt?: string;
        vertical?: string;
        topic?: string;
    };
}

export const AIImageGeneration: React.FC<AIImageGenerationProps> = ({
    userId,
    selectedText = '',
    onImageGenerated,
    articleContext
}) => {
    const [contextText, setContextText] = useState(selectedText);
    const [selectedStyleId, setSelectedStyleId] = useState<string>('cinematic_still');
    const [prompt, setPrompt] = useState('');
    const [synthesizingPrompt, setSynthesizingPrompt] = useState(false);

    const [models, setModels] = useState<ImageProviderModel[]>([]);
    const [selectedModel, setSelectedModel] = useState('');
    const [aspectRatio, setAspectRatio] = useState('16:9');
    const [resolution, setResolution] = useState('1K');
    const [referenceImage, setReferenceImage] = useState<File | null>(null);
    const [referenceImagePreview, setReferenceImagePreview] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);
    const [loadingModels, setLoadingModels] = useState(true);
    const [generatedImage, setGeneratedImage] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [artDirection, setArtDirection] = useState<{
        hero_subject?: string;
        core_thesis?: string;
        core_conflict?: string;
        composition?: string;
        alt_text?: string;
        caption?: string;
        title?: string;
        style_label?: string;
    } | null>(null);

    // Editorial Cover Typography Overlay State
    const [overlayConfig, setOverlayConfig] = useState<OverlayConfig>({
        enabled: false,
        kicker: articleContext?.vertical ? `${articleContext.vertical.toUpperCase().slice(0, 30)} // ANALYSIS` : '',
        title: articleContext?.title ? articleContext.title.toUpperCase().slice(0, 34) : '',
        hook: articleContext?.thesis ? articleContext.thesis.slice(0, 58) : (articleContext?.deck ? articleContext.deck.slice(0, 58) : ''),
        corner: 'auto'
    });
    const [overlayDetails, setOverlayDetails] = useState<OverlayDetails | null>(null);
    const [baseImage, setBaseImage] = useState<string | null>(null);
    const [overlayImage, setOverlayImage] = useState<string | null>(null);
    const [showWithOverlay, setShowWithOverlay] = useState<boolean>(true);
    const [draftingCopy, setDraftingCopy] = useState<boolean>(false);
    const [applyingOverlay, setApplyingOverlay] = useState<boolean>(false);

    useEffect(() => {
        loadModels();
    }, []);

    useEffect(() => {
        if (selectedText && selectedText !== contextText) {
            setContextText(selectedText);
        }
    }, [selectedText]);

    // When component mounts with selectedText, synthesize an initial prompt if prompt is empty
    useEffect(() => {
        if (selectedText && !prompt) {
            handleSynthesizePrompt(selectedText, selectedStyleId);
        }
    }, []);

    const loadModels = async () => {
        try {
            const [modelsRes, appConfigRes] = await Promise.allSettled([
                getImageProviderModels(),
                getImageApplicationConfig()
            ]);

            const data = modelsRes.status === 'fulfilled' ? modelsRes.value : [];
            setModels(data);

            const appConfig = appConfigRes.status === 'fulfilled' ? appConfigRes.value : null;
            const configuredModel = appConfig?.applications?.article_image?.model_name;

            if (configuredModel && data.some(m => m.model_technical_name === configuredModel)) {
                setSelectedModel(configuredModel);
            } else if (data.length > 0) {
                setSelectedModel(data[0].model_technical_name);
            }
        } catch (err) {
            setError('Failed to load AI models');
            console.error(err);
        } finally {
            setLoadingModels(false);
        }
    };

    const handleSynthesizePrompt = async (textToUse?: string, styleIdToUse?: string) => {
        const text = (textToUse !== undefined ? textToUse : contextText).trim();
        const styleId = styleIdToUse !== undefined ? styleIdToUse : selectedStyleId;
        const style = STYLE_PRESETS.find(s => s.id === styleId);

        if (!text) {
            if (style) {
                setPrompt(style.promptModifier);
            }
            return;
        }

        setSynthesizingPrompt(true);
        setError(null);

        try {
            const res = await synthesizeImagePrompt({
                text,
                style: style?.name,
                style_id: style?.id,
                style_prompt_modifier: style?.promptModifier,
                article_title: articleContext?.title,
                article_context: articleContext,
            });

            if (res.prompt) {
                setPrompt(res.prompt);
                setArtDirection({
                    hero_subject: res.hero_subject,
                    core_thesis: res.core_thesis,
                    core_conflict: res.core_conflict,
                    composition: res.composition,
                    alt_text: res.alt_text,
                    caption: res.caption,
                    title: res.title,
                    style_label: res.style_label,
                });
            } else {
                // Fallback prompt generation
                const fallback = style
                    ? `${text.slice(0, 150)}, ${style.promptModifier}`
                    : text;
                setPrompt(fallback);
            }
        } catch (err) {
            console.warn('Synthesize prompt failed; using direct style formulation:', err);
            const fallback = style
                ? `${text.slice(0, 150)}, ${style.promptModifier}`
                : text;
            setPrompt(fallback);
        } finally {
            setSynthesizingPrompt(false);
        }
    };

    const handleSelectStyle = (styleId: string) => {
        const newStyleId = selectedStyleId === styleId ? '' : styleId;
        setSelectedStyleId(newStyleId);

        // If we have contextText, update/synthesize the prompt
        if (contextText.trim()) {
            handleSynthesizePrompt(contextText, newStyleId);
        } else if (newStyleId) {
            const style = STYLE_PRESETS.find(s => s.id === newStyleId);
            if (style) {
                setPrompt(style.promptModifier);
            }
        }
    };

    const handleReferenceImageChange = (file: File | null) => {
        setReferenceImage(file);
        if (file) {
            const previewUrl = URL.createObjectURL(file);
            setReferenceImagePreview(previewUrl);
        } else {
            setReferenceImagePreview(null);
        }
    };

    const handleDraftOverlayCopy = async () => {
        setDraftingCopy(true);
        setError(null);
        try {
            const res = await generateOverlayCopy({
                text: contextText || prompt,
                article_title: articleContext?.title,
                article_context: articleContext
            });
            setOverlayConfig(prev => ({
                ...prev,
                kicker: res.kicker,
                title: res.title,
                hook: res.hook
            }));
        } catch (err: any) {
            console.warn('Auto-draft overlay copy failed:', err);
        } finally {
            setDraftingCopy(false);
        }
    };

    const handleApplyOverlay = async () => {
        const imageToUse = baseImage || generatedImage;
        if (!imageToUse) return;

        setApplyingOverlay(true);
        setError(null);
        try {
            const res = await applyImageOverlay({
                image_url: imageToUse.startsWith('data:') ? undefined : imageToUse,
                image_base64: imageToUse.startsWith('data:') ? imageToUse : undefined,
                kicker: overlayConfig.kicker || '',
                title: overlayConfig.title || '',
                hook: overlayConfig.hook || '',
                corner: overlayConfig.corner || 'auto',
                user_id: userId
            });

            setOverlayImage(res.imageUrl);
            setOverlayDetails(res.overlayDetails);
            setShowWithOverlay(true);
            setGeneratedImage(res.imageUrl);
        } catch (err: any) {
            setError(err.message || 'Failed to apply typography overlay');
        } finally {
            setApplyingOverlay(false);
        }
    };

    const handleGenerate = async () => {
        let activePrompt = prompt.trim();

        // If prompt is empty but context exists, synthesize on the fly
        if (!activePrompt && contextText.trim()) {
            const style = STYLE_PRESETS.find(s => s.id === selectedStyleId);
            activePrompt = style
                ? `${contextText.trim().slice(0, 150)}, ${style.promptModifier}`
                : contextText.trim();
            setPrompt(activePrompt);
        }

        if (!activePrompt) {
            setError('Please enter a description or select text from your article to generate an image.');
            return;
        }

        setLoading(true);
        setError(null);
        setGeneratedImage(null);
        setBaseImage(null);
        setOverlayImage(null);

        try {
            let referenceImageBase64: string | undefined;
            if (referenceImage) {
                const reader = new FileReader();
                referenceImageBase64 = await new Promise((resolve) => {
                    reader.onload = () => resolve(reader.result as string);
                    reader.readAsDataURL(referenceImage);
                }).then((result) => (result as string).split(',')[1]);
            }

            const response = await generateAIImage({
                prompt: activePrompt,
                model: selectedModel,
                application: 'article_image',
                aspectRatio,
                resolution,
                referenceImage: referenceImageBase64,
                user_id: userId,
                overlay: overlayConfig.enabled ? overlayConfig : undefined
            });

            const pristineBase = response.baseImageUrl || response.imageUrl;
            setBaseImage(pristineBase);

            if (response.overlayDetails) {
                setOverlayDetails(response.overlayDetails);
                setOverlayImage(response.imageUrl);
                setShowWithOverlay(true);
            } else {
                setOverlayDetails(null);
                setOverlayImage(null);
            }

            setGeneratedImage(response.imageUrl);
        } catch (err: any) {
            setError(err.message || 'Failed to generate image');
        } finally {
            setLoading(false);
        }
    };

    const handleAccept = () => {
        const imageToInsert = (showWithOverlay && overlayImage)
            ? overlayImage
            : (baseImage || generatedImage);

        if (imageToInsert) {
            const modelInfo = models.find(m => m.model_technical_name === selectedModel);
            const styleInfo = STYLE_PRESETS.find(s => s.id === selectedStyleId);
            const styleTag = styleInfo ? ` [Style: ${styleInfo.name}]` : '';

            // Title & Alt Text
            const finalTitle = (overlayDetails?.title || artDirection?.title || prompt.substring(0, 80)).trim();
            const finalAlt = (
                artDirection?.alt_text ||
                (overlayDetails ? `${overlayDetails.title}. ${overlayDetails.hook}` : prompt.substring(0, 125))
            ).trim();
            const finalCaption = (
                overlayDetails ? `${overlayDetails.kicker} — ${overlayDetails.title}` : (artDirection?.caption || '')
            ).trim();

            onImageGenerated(imageToInsert, {
                ImageUrl: imageToInsert,
                ImageAuthor: `AI - ${modelInfo?.model_name || selectedModel}${styleTag}`,
                MediaAltText: finalAlt,
                mediaTitle: finalTitle,
                mediaCaption: finalCaption
            });
        }
    };

    const selectedModelInfo = models.find(m => m.model_technical_name === selectedModel);
    const aspectRatios = selectedModelInfo?.supported_aspect_ratios || ['16:9', '1:1', '4:3', '3:2', '9:16'];
    const rawResolutions = selectedModelInfo?.supported_resolutions || ['1K', '2K', '4K'];
    const resolutions = rawResolutions.includes('1K')
        ? ['1K', ...rawResolutions.filter(r => r !== '1K')]
        : ['1K', ...rawResolutions];

    if (loadingModels) {
        return (
            <div className="flex items-center justify-center py-12">
                <Loader2 className="w-8 h-8 animate-spin text-indigo-600" />
            </div>
        );
    }

    return (
        <div className="space-y-6">
            {/* Header / Intro Banner */}
            <div className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-950/40 dark:to-indigo-950/40 p-4 rounded-xl border border-purple-100 dark:border-purple-900/50">
                <div className="flex items-start gap-3">
                    <Sparkles className="w-5 h-5 text-purple-600 dark:text-purple-400 mt-0.5 flex-shrink-0" />
                    <div>
                        <h3 className="text-sm font-semibold text-purple-950 dark:text-purple-200">
                            AI Image Generation
                        </h3>
                        <p className="text-xs text-purple-700 dark:text-purple-300 mt-0.5">
                            Select or paste text from your article, pick a visual style preset, synthesize an image prompt, and generate high-fidelity AI imagery.
                        </p>
                    </div>
                </div>
            </div>

            {/* Step 1: Context / Article Selection */}
            <div>
                <div className="flex items-center justify-between mb-1.5">
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                        Article Text / Section
                    </label>
                    {contextText && (
                        <button
                            type="button"
                            onClick={() => handleSynthesizePrompt(contextText, selectedStyleId)}
                            disabled={synthesizingPrompt}
                            className="text-xs font-semibold text-indigo-600 dark:text-indigo-400 hover:text-indigo-800 dark:hover:text-indigo-300 flex items-center gap-1.5 transition-colors disabled:opacity-50"
                        >
                            {synthesizingPrompt ? (
                                <>
                                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                                    Synthesizing Prompt...
                                </>
                            ) : (
                                <>
                                    <Wand2 className="w-3.5 h-3.5" />
                                    Synthesize Prompt from Text
                                </>
                            )}
                        </button>
                    )}
                </div>
                <textarea
                    value={contextText}
                    onChange={(e) => setContextText(e.target.value)}
                    rows={3}
                    className="w-full px-4 py-2.5 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-gray-900 dark:text-white text-sm placeholder-gray-400"
                    placeholder="Selected article text will appear here, or paste any paragraph/idea..."
                />
                {selectedText && (
                    <p className="mt-1 text-xs text-indigo-600 dark:text-indigo-400 flex items-center gap-1">
                        <Sparkles className="w-3.5 h-3.5" />
                        Loaded from article selection
                    </p>
                )}
            </div>

            {/* Step 2: Visual Style Presets */}
            <div>
                <div className="flex items-center justify-between mb-2">
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 flex items-center gap-1.5">
                        <Palette className="w-4 h-4 text-indigo-500" />
                        Visual Style Preset
                    </label>
                    {selectedStyleId && (
                        <button
                            type="button"
                            onClick={() => setSelectedStyleId('')}
                            className="text-xs text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
                        >
                            Clear style
                        </button>
                    )}
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2.5">
                    {STYLE_PRESETS.map((style) => {
                        const isSelected = selectedStyleId === style.id;
                        return (
                            <button
                                key={style.id}
                                type="button"
                                onClick={() => handleSelectStyle(style.id)}
                                className={`text-left p-2.5 rounded-xl border transition-all flex flex-col justify-between ${isSelected
                                    ? 'border-indigo-600 bg-indigo-50/80 dark:bg-indigo-950/60 ring-2 ring-indigo-500/20 shadow-sm'
                                    : 'border-gray-200 dark:border-gray-700/80 bg-white dark:bg-gray-800/60 hover:border-gray-300 dark:hover:border-gray-600'
                                    }`}
                            >
                                <div className="flex items-center justify-between w-full mb-1">
                                    <div className="flex items-center gap-1.5 flex-wrap">
                                        <span className={`text-xs font-semibold ${isSelected ? 'text-indigo-950 dark:text-indigo-200' : 'text-gray-900 dark:text-white'}`}>
                                            {style.name}
                                        </span>
                                        {style.isEditorial && (
                                            <span className="text-[9px] px-1.5 py-0.2 bg-purple-100 dark:bg-purple-900/60 text-purple-700 dark:text-purple-300 rounded font-medium">
                                                Editorial
                                            </span>
                                        )}
                                    </div>
                                    {isSelected && (
                                        <div className="w-4 h-4 rounded-full bg-indigo-600 text-white flex items-center justify-center flex-shrink-0">
                                            <Check className="w-2.5 h-2.5" />
                                        </div>
                                    )}
                                </div>
                                <p className="text-[10px] text-gray-500 dark:text-gray-400 line-clamp-2 leading-tight">
                                    {style.description}
                                </p>
                            </button>
                        );
                    })}
                </div>
            </div>

            {/* Art Direction Insights (Magazine Cover Standard) */}
            {artDirection && (artDirection.hero_subject || artDirection.core_thesis) && (
                <div className="p-3.5 bg-gradient-to-r from-indigo-50/80 to-purple-50/80 dark:from-indigo-950/40 dark:to-purple-950/40 rounded-xl border border-indigo-100 dark:border-indigo-900/50 space-y-2">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-1.5 text-xs font-semibold text-indigo-950 dark:text-indigo-200">
                            <Wand2 className="w-3.5 h-3.5 text-indigo-600 dark:text-indigo-400" />
                            Art Direction Insights & Governing Conflict
                        </div>
                        {artDirection.style_label && (
                            <span className="text-[10px] font-medium text-indigo-700 dark:text-indigo-300 bg-white/70 dark:bg-indigo-900/60 px-2 py-0.5 rounded-md border border-indigo-200/50 dark:border-indigo-800/50">
                                {artDirection.style_label}
                            </span>
                        )}
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs">
                        {artDirection.hero_subject && (
                            <div>
                                <span className="font-semibold text-gray-600 dark:text-gray-400">Hero Protagonist:</span>{' '}
                                <span className="text-gray-900 dark:text-gray-100 font-medium">{artDirection.hero_subject}</span>
                            </div>
                        )}
                        {artDirection.core_thesis && (
                            <div>
                                <span className="font-semibold text-gray-600 dark:text-gray-400">Core Thesis:</span>{' '}
                                <span className="text-gray-800 dark:text-gray-200">{artDirection.core_thesis}</span>
                            </div>
                        )}
                        {artDirection.core_conflict && (
                            <div>
                                <span className="font-semibold text-gray-600 dark:text-gray-400">Governing Conflict:</span>{' '}
                                <span className="text-gray-800 dark:text-gray-200">{artDirection.core_conflict}</span>
                            </div>
                        )}
                        {artDirection.composition && (
                            <div>
                                <span className="font-semibold text-gray-600 dark:text-gray-400">Framing Rule:</span>{' '}
                                <span className="text-gray-800 dark:text-gray-200">{artDirection.composition}</span>
                            </div>
                        )}
                    </div>
                    {artDirection.alt_text && (
                        <div className="text-[11px] text-gray-500 dark:text-gray-400 border-t border-indigo-200/60 dark:border-indigo-900/50 pt-1.5 flex items-start gap-1">
                            <span className="font-semibold text-indigo-800 dark:text-indigo-300 flex-shrink-0">Alt Text:</span>
                            <span>{artDirection.alt_text}</span>
                        </div>
                    )}
                </div>
            )}

            {/* Step 3: Editable Synthesized Prompt */}
            <div>
                <div className="flex items-center justify-between mb-1.5">
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                        Generated Image Prompt (Editable)
                    </label>
                    {prompt && (
                        <button
                            type="button"
                            onClick={() => setPrompt('')}
                            className="text-xs text-gray-400 hover:text-red-500"
                        >
                            Clear
                        </button>
                    )}
                </div>
                <textarea
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    rows={3}
                    className="w-full px-4 py-3 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-gray-900 dark:text-white text-sm"
                    placeholder="Synthesized or custom prompt for image generation..."
                />
            </div>

            {/* Step 4: Model, Aspect Ratio, Resolution */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5">
                        AI Model
                    </label>
                    <select
                        value={selectedModel}
                        onChange={(e) => setSelectedModel(e.target.value)}
                        className="w-full px-3.5 py-2.5 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-gray-900 dark:text-white text-sm"
                    >
                        {models.map((model) => (
                            <option key={model.id} value={model.model_technical_name}>
                                {model.model_name}
                            </option>
                        ))}
                    </select>
                </div>

                <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5">
                        Aspect Ratio
                    </label>
                    <select
                        value={aspectRatio}
                        onChange={(e) => setAspectRatio(e.target.value)}
                        className="w-full px-3.5 py-2.5 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-gray-900 dark:text-white text-sm"
                    >
                        {aspectRatios.map((ratio) => (
                            <option key={ratio} value={ratio}>
                                {ratio}
                            </option>
                        ))}
                    </select>
                </div>

                <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5">
                        Resolution
                    </label>
                    <select
                        value={resolution}
                        onChange={(e) => setResolution(e.target.value)}
                        className="w-full px-3.5 py-2.5 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-gray-900 dark:text-white text-sm"
                    >
                        {resolutions.map((res) => (
                            <option key={res} value={res}>
                                {res === '1K' ? '1K (Default)' : res}
                            </option>
                        ))}
                    </select>
                </div>
            </div>

            {/* Step 5: Editorial Cover Typography Overlay */}
            <EditorialOverlayPanel
                config={overlayConfig}
                onChange={setOverlayConfig}
                onDraftCopy={handleDraftOverlayCopy}
                draftingCopy={draftingCopy}
                overlayDetails={overlayDetails}
                onApplyOverlay={handleApplyOverlay}
                applyingOverlay={applyingOverlay}
                hasGeneratedImage={Boolean(generatedImage || baseImage)}
                showWithOverlay={showWithOverlay}
                onToggleOverlayPreview={(show) => setShowWithOverlay(show)}
            />

            {/* Step 6: Optional Reference Image */}
            <div>
                <div className="flex items-center justify-between mb-1.5">
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                        Reference Image (Optional)
                    </label>
                    {referenceImage && (
                        <button
                            type="button"
                            onClick={() => handleReferenceImageChange(null)}
                            className="text-xs font-medium text-red-500 hover:text-red-700 dark:text-red-400 flex items-center gap-1"
                        >
                            <X className="w-3.5 h-3.5" /> Remove reference
                        </button>
                    )}
                </div>
                <p className="text-xs text-gray-500 dark:text-gray-400 mb-2">
                    Upload an image to guide style or object transfer, or leave empty to generate purely from text and style preset.
                </p>

                {referenceImage && referenceImagePreview ? (
                    <div className="flex items-center gap-4 p-3 bg-gray-50 dark:bg-gray-900/60 rounded-xl border border-indigo-200 dark:border-indigo-800/60">
                        <div className="relative w-14 h-14 rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700 flex-shrink-0 bg-black/5">
                            <img
                                src={referenceImagePreview}
                                alt="Reference preview"
                                className="w-full h-full object-cover"
                            />
                        </div>
                        <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2">
                                <span className="text-xs font-semibold text-gray-900 dark:text-white truncate">
                                    {referenceImage.name}
                                </span>
                                <span className="px-2 py-0.5 text-[10px] font-medium bg-indigo-100 dark:bg-indigo-950/60 text-indigo-700 dark:text-indigo-300 rounded-full border border-indigo-200 dark:border-indigo-800">
                                    Reference Attached
                                </span>
                            </div>
                            <p className="text-[11px] text-gray-500 dark:text-gray-400 mt-0.5">
                                {(referenceImage.size / 1024).toFixed(1)} KB • Guiding {selectedModelInfo?.model_name || 'AI model'}
                            </p>
                        </div>
                        <button
                            type="button"
                            onClick={() => handleReferenceImageChange(null)}
                            className="p-1.5 rounded-lg text-gray-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-950/30 transition-colors"
                            title="Remove reference image"
                        >
                            <X className="w-4 h-4" />
                        </button>
                    </div>
                ) : (
                    <label className="flex flex-col items-center justify-center p-4 border-2 border-dashed border-gray-300 dark:border-gray-700 hover:border-indigo-400 dark:hover:border-indigo-500 rounded-xl cursor-pointer bg-gray-50/50 dark:bg-gray-900/30 hover:bg-indigo-50/20 transition-colors group">
                        <div className="flex items-center justify-center gap-2 text-center">
                            <div className="p-2 rounded-lg bg-gray-100 dark:bg-gray-800 group-hover:bg-indigo-100 dark:group-hover:bg-indigo-950/60 text-gray-500 group-hover:text-indigo-600 dark:group-hover:text-indigo-400 transition-colors">
                                <Upload className="w-4 h-4" />
                            </div>
                            <div className="text-left">
                                <p className="text-xs font-medium text-gray-700 dark:text-gray-300">
                                    <span className="text-indigo-600 dark:text-indigo-400 underline">Upload reference image</span> (Optional)
                                </p>
                                <p className="text-[10px] text-gray-400 dark:text-gray-500">
                                    Supports PNG, JPG, or WEBP
                                </p>
                            </div>
                        </div>
                        <input
                            type="file"
                            accept="image/*"
                            onChange={(e) => handleReferenceImageChange(e.target.files?.[0] || null)}
                            className="hidden"
                        />
                    </label>
                )}
            </div>

            {/* Error Message */}
            {error && (
                <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl p-4">
                    <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
                </div>
            )}

            {/* Generated Image Preview */}
            {(generatedImage || baseImage) && (
                <div className="space-y-4">
                    <div className="relative rounded-xl overflow-hidden border border-gray-200 dark:border-gray-700 shadow-sm bg-black/5">
                        <img
                            src={(showWithOverlay && overlayImage) ? overlayImage : (baseImage || generatedImage)!}
                            alt="Generated preview"
                            className="w-full h-auto max-h-[480px] object-contain mx-auto"
                        />
                        {overlayImage && baseImage && (
                            <div className="absolute bottom-3 right-3 flex items-center gap-2 bg-black/60 backdrop-blur-md px-3 py-1.5 rounded-lg border border-white/20 text-white text-xs">
                                <span>Preview:</span>
                                <button
                                    type="button"
                                    onClick={() => setShowWithOverlay(!showWithOverlay)}
                                    className="font-bold underline hover:text-indigo-300 transition-colors"
                                >
                                    {showWithOverlay ? 'With Cover Typography' : 'Clean Base Artwork'}
                                </button>
                            </div>
                        )}
                    </div>
                    <div className="flex gap-3">
                        <button
                            onClick={handleGenerate}
                            className="flex items-center gap-2 px-6 py-3 rounded-xl border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
                        >
                            <RefreshCw className="w-4 h-4" />
                            Regenerate Artwork
                        </button>
                        <button
                            onClick={handleAccept}
                            className="flex-1 bg-indigo-600 hover:bg-indigo-700 text-white px-6 py-3 rounded-xl font-medium transition-colors shadow-sm"
                        >
                            Accept & Insert Image
                        </button>
                    </div>
                </div>
            )}

            {/* Generate Button */}
            {!generatedImage && (
                <button
                    onClick={handleGenerate}
                    disabled={loading || (!prompt.trim() && !contextText.trim())}
                    className="w-full flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white px-6 py-3 rounded-xl font-medium transition-colors shadow-sm"
                >
                    {loading ? (
                        <>
                            <Loader2 className="w-5 h-5 animate-spin" />
                            Generating Image...
                        </>
                    ) : (
                        <>
                            <Sparkles className="w-5 h-5" />
                            Generate Image
                        </>
                    )}
                </button>
            )}
        </div>
    );
};
