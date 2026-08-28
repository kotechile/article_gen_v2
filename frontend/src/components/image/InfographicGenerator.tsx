import React, { useState, useEffect } from 'react';
import {
    Loader2,
    BarChart3,
    Sparkles,
    Cpu,
    ListOrdered,
    GitFork,
    Boxes,
    History,
    Smile,
    Wand2
} from 'lucide-react';
import {
    generateAIInfographic,
    getImageApplicationConfig
} from '../../services/imageService';
import type {
    ImageMetadata,
    InfographicArchetype,
    ImageApplicationConfig
} from '../../types/image';

interface InfographicGeneratorProps {
    userId: string;
    selectedText?: string;
    onInfographicGenerated: (imageUrl: string, metadata: Partial<ImageMetadata>) => void;
}

interface ArchetypeOption {
    id: InfographicArchetype;
    label: string;
    icon: React.FC<{ className?: string }>;
    description: string;
    tag: string;
}

const ARCHETYPES: ArchetypeOption[] = [
    {
        id: 'auto',
        label: 'Auto-Detect Style',
        icon: Sparkles,
        description: 'AI analyzes your text to automatically choose the best diagram archetype.',
        tag: 'Recommended'
    },
    {
        id: 'technical_scientific',
        label: 'Technical & Scientific Diagrams',
        icon: Cpu,
        description: 'Physics concepts, technical systems (e.g. Kubernetes pods), or biology with schematics and callouts.',
        tag: 'Schematic'
    },
    {
        id: 'step_by_step',
        label: 'Step-by-Step Guides & Recipes',
        icon: ListOrdered,
        description: 'Numbered sequential cards showing processes like cooking recipes or DIY workflows.',
        tag: 'Sequential'
    },
    {
        id: 'flowchart_whiteboard',
        label: 'Flowcharts & Whiteboard Sketches',
        icon: GitFork,
        description: 'Hand-drawn dry-erase whiteboard style, notebook flowcharts, and organic brainstorming.',
        tag: 'Hand-drawn'
    },
    {
        id: 'modular_explainer',
        label: 'Modular Explainers',
        icon: Boxes,
        description: 'Central hubs with connected radial components showing how complex systems operate.',
        tag: 'System'
    },
    {
        id: 'timeline_historical',
        label: 'Timelines & Historical Overviews',
        icon: History,
        description: 'Sequential milestone markers tracking events or product evolution with dates.',
        tag: 'Chronological'
    },
    {
        id: 'data_visualization',
        label: 'Data Visualizations',
        icon: BarChart3,
        description: 'Metrics, percentages, KPI cards, comparison columns, and structured financial summaries.',
        tag: 'Analytical'
    },
    {
        id: 'playful_viral',
        label: 'Playful & Viral Listicles',
        icon: Smile,
        description: 'Lighthearted pop-art menus, humorous life steps, or colorful illustrated graphics.',
        tag: 'Pop-Art'
    }
];

export const InfographicGenerator: React.FC<InfographicGeneratorProps> = ({
    userId,
    selectedText = '',
    onInfographicGenerated
}) => {
    const [storyText, setStoryText] = useState(selectedText);
    const [selectedArchetype, setSelectedArchetype] = useState<InfographicArchetype>('auto');
    const [userInstructions, setUserInstructions] = useState('');
    const [aspectRatio, setAspectRatio] = useState('16:9');
    const [resolution, setResolution] = useState('1K');
    const [loading, setLoading] = useState(false);
    const [appConfig, setAppConfig] = useState<ImageApplicationConfig | null>(null);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        loadAppConfig();
    }, []);

    useEffect(() => {
        if (selectedText && selectedText !== storyText) {
            setStoryText(selectedText);
        }
    }, [selectedText]);

    const loadAppConfig = async () => {
        try {
            const res = await getImageApplicationConfig();
            if (res.applications && res.applications.infographics) {
                setAppConfig(res.applications.infographics);
            }
        } catch (err) {
            console.error('Error loading infographics app config:', err);
        }
    };

    const handleGenerate = async () => {
        if (!storyText.trim()) {
            setError('Please enter or highlight some text to generate an infographic.');
            return;
        }

        setLoading(true);
        setError(null);

        try {
            const response = await generateAIInfographic({
                text: storyText.trim(),
                archetype: selectedArchetype,
                user_instructions: userInstructions.trim() || undefined,
                aspectRatio,
                resolution,
                user_id: userId
            });

            onInfographicGenerated(response.imageUrl, response.metadata);
        } catch (err: any) {
            console.error('Infographic generation error:', err);
            setError(err.message || 'Failed to generate AI infographic.');
        } finally {
            setLoading(false);
        }
    };

    const modelDisplayName = appConfig?.display_name || appConfig?.model_name || 'Nano Banana Pro';

    return (
        <div className="space-y-6">
            {/* Header info banner with active model */}
            <div className="bg-gradient-to-r from-emerald-50 via-teal-50 to-indigo-50 dark:from-emerald-950/40 dark:via-teal-950/40 dark:to-indigo-950/40 p-4 rounded-xl border border-emerald-100 dark:border-emerald-900/50">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                    <div className="flex items-start gap-3">
                        <Wand2 className="w-5 h-5 text-emerald-600 dark:text-emerald-400 mt-0.5 flex-shrink-0" />
                        <div>
                            <h3 className="text-sm font-semibold text-emerald-950 dark:text-emerald-200">
                                AI Infographic Generation
                            </h3>
                            <p className="text-xs text-emerald-700 dark:text-emerald-300 mt-0.5">
                                Generates professional diagrams, flowcharts, timelines, and data visuals directly from article text.
                            </p>
                        </div>
                    </div>
                    <div className="flex items-center gap-2">
                        <span className="text-[11px] font-semibold uppercase tracking-wider text-emerald-700 dark:text-emerald-300 bg-white/80 dark:bg-emerald-900/60 px-2.5 py-1 rounded-full border border-emerald-200 dark:border-emerald-700 shadow-sm">
                            Model: {modelDisplayName}
                        </span>
                    </div>
                </div>
            </div>

            {error && (
                <div className="p-4 bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800 rounded-xl text-red-700 dark:text-red-300 text-sm">
                    {error}
                </div>
            )}

            {/* Content to Visualize */}
            <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Text / Content to Visualize
                </label>
                <textarea
                    value={storyText}
                    onChange={(e) => setStoryText(e.target.value)}
                    rows={4}
                    className="w-full px-4 py-3 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 focus:outline-none focus:ring-2 focus:ring-emerald-500 text-gray-900 dark:text-white placeholder-gray-400 text-sm"
                    placeholder="Highlight or paste an article section explaining a process, system architecture, history, or metrics..."
                />
            </div>

            {/* Infographic Archetype Selector */}
            <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Select Infographic Archetype
                </label>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
                    {ARCHETYPES.map((arch) => {
                        const Icon = arch.icon;
                        const isSelected = selectedArchetype === arch.id;
                        return (
                            <div
                                key={arch.id}
                                onClick={() => setSelectedArchetype(arch.id)}
                                className={`p-3.5 rounded-xl border-2 cursor-pointer transition-all flex flex-col justify-between ${
                                    isSelected
                                        ? 'border-emerald-600 bg-emerald-50/50 dark:bg-emerald-950/30 ring-2 ring-emerald-500/20'
                                        : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600 bg-white dark:bg-gray-800/60'
                                }`}
                            >
                                <div>
                                    <div className="flex items-center justify-between mb-2">
                                        <div className={`p-2 rounded-lg ${
                                            isSelected
                                                ? 'bg-emerald-600 text-white'
                                                : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300'
                                        }`}>
                                            <Icon className="w-4 h-4" />
                                        </div>
                                        <span className="text-[10px] font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">
                                            {arch.tag}
                                        </span>
                                    </div>
                                    <h4 className="text-xs font-bold text-gray-900 dark:text-white mb-1">
                                        {arch.label}
                                    </h4>
                                    <p className="text-[11px] text-gray-500 dark:text-gray-400 line-clamp-2 leading-relaxed">
                                        {arch.description}
                                    </p>
                                </div>
                            </div>
                        );
                    })}
                </div>
            </div>

            {/* Optional User Creative Instructions */}
            <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Custom Formatting / Creative Directives (Optional)
                </label>
                <input
                    type="text"
                    value={userInstructions}
                    onChange={(e) => setUserInstructions(e.target.value)}
                    placeholder="e.g. Use a dark cyan blueprint aesthetic, highlight step 3, or emphasize growth percentage"
                    className="w-full px-4 py-2.5 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 focus:outline-none focus:ring-2 focus:ring-emerald-500 text-gray-900 dark:text-white placeholder-gray-400 text-sm"
                />
            </div>

            {/* Aspect Ratio and Resolution */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                        Aspect Ratio
                    </label>
                    <select
                        value={aspectRatio}
                        onChange={(e) => setAspectRatio(e.target.value)}
                        className="w-full px-3 py-2.5 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 text-gray-900 dark:text-white"
                    >
                        <option value="16:9">16:9 (Landscape Diagram / Presentation)</option>
                        <option value="4:3">4:3 (Editorial Standard)</option>
                        <option value="9:16">9:16 (Vertical Mobile / Social Infographic)</option>
                        <option value="1:1">1:1 (Square Infographic)</option>
                    </select>
                </div>

                <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                        Resolution
                    </label>
                    <select
                        value={resolution}
                        onChange={(e) => setResolution(e.target.value)}
                        className="w-full px-3 py-2.5 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 text-gray-900 dark:text-white"
                    >
                        <option value="1K">1K (Default)</option>
                        <option value="2K">2K (High Resolution)</option>
                    </select>
                </div>
            </div>

            {/* Generate Button */}
            <button
                type="button"
                onClick={handleGenerate}
                disabled={loading || !storyText.trim()}
                className="w-full flex items-center justify-center gap-2 px-6 py-3.5 rounded-xl bg-emerald-600 hover:bg-emerald-700 disabled:opacity-50 text-white font-medium shadow-md transition-colors"
            >
                {loading ? (
                    <>
                        <Loader2 className="w-5 h-5 animate-spin" />
                        <span>Synthesizing & Generating Infographic...</span>
                    </>
                ) : (
                    <>
                        <BarChart3 className="w-5 h-5" />
                        <span>Generate Infographic ({modelDisplayName})</span>
                    </>
                )}
            </button>
        </div>
    );
};
