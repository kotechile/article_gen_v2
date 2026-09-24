import React, { useState, useEffect, useMemo } from 'react';
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
    Wand2,
    Search,
    Check,
    RefreshCw,
    Compass,
    Zap,
    Scale,
    Layers,
    Grid,
    Globe,
    Filter,
    Triangle,
    Share2,
    CheckSquare,
    ArrowUpDown,
    Info
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

export type StyleCategory =
    | 'all'
    | 'process_sequential'
    | 'comparison_contrast'
    | 'data_statistics'
    | 'structure_hierarchy'
    | 'lists_summaries'
    | 'classic';

interface StyleOption {
    id: InfographicArchetype;
    label: string;
    icon: React.FC<{ className?: string }>;
    description: string;
    tag: string;
    category: StyleCategory;
}

const CATEGORIES: { id: StyleCategory; label: string }[] = [
    { id: 'all', label: 'All Styles' },
    { id: 'process_sequential', label: 'Process & Sequential' },
    { id: 'comparison_contrast', label: 'Comparison & Contrast' },
    { id: 'data_statistics', label: 'Data & Statistics' },
    { id: 'structure_hierarchy', label: 'Structure & Hierarchy' },
    { id: 'lists_summaries', label: 'Lists & Summaries' },
    { id: 'classic', label: 'Classic Archetypes' }
];

const INFOGRAPHIC_STYLES: StyleOption[] = [
    // Recommended Auto
    {
        id: 'auto',
        label: 'Auto-Detect Style',
        icon: Sparkles,
        description: 'AI analyzes your text to automatically choose the best diagram archetype and visual layout.',
        tag: 'Recommended',
        category: 'classic'
    },

    // Process & Sequential Flow
    {
        id: 'step_by_step_isometric',
        label: 'The Step-by-Step Flowchart',
        icon: Boxes,
        description: 'Clean, colorful 3D blocks and miniature vector elements on light gray in a sequential zigzag path.',
        tag: 'Isometric 3D',
        category: 'process_sequential'
    },
    {
        id: 'timeline_modern',
        label: 'The Modern Milestone Timeline',
        icon: History,
        description: 'Clean, contemporary tech roadmap with distinct milestone nodes without antique parchment textures.',
        tag: 'Sleek Tech',
        category: 'process_sequential'
    },
    {
        id: 'timeline_historical_vintage',
        label: 'The Historical Timeline',
        icon: History,
        description: 'Vintage/retro aesthetic with sepia, mustard, faded teal tones, textured paper, and classic serif typography.',
        tag: 'Vintage / Retro',
        category: 'process_sequential'
    },
    {
        id: 'user_journey_flat',
        label: 'The User Journey Map',
        icon: Compass,
        description: 'Ultra-minimalist 3 solid colors, 4 distinct phases (Awareness, Consideration, Action, Loyalty) with line-art icons.',
        tag: 'Minimalist Flat',
        category: 'process_sequential'
    },
    {
        id: 'lifecycle_loop_watercolor',
        label: 'The Lifecycle Loop',
        icon: RefreshCw,
        description: 'Soft, organic watercolor textures with fluid shapes and smooth gradients in a 4-5 arrow circular loop.',
        tag: 'Watercolor',
        category: 'process_sequential'
    },

    // Comparison & Contrast
    {
        id: 'side_by_side_neon',
        label: 'The Side-by-Side Duel',
        icon: Zap,
        description: 'Dark mode cyberpunk aesthetic with glowing cyan and magenta accents and symmetrical split-screen comparison.',
        tag: 'Dark Neon',
        category: 'comparison_contrast'
    },
    {
        id: 'pros_cons_scandinavian',
        label: 'The Pros and Cons Scales',
        icon: Scale,
        description: 'Clean Scandinavian design with lots of white space, muted pastel colors, and minimalist drop-shadow cards.',
        tag: 'Scandinavian',
        category: 'comparison_contrast'
    },
    {
        id: 'venn_diagram_glassmorphism',
        label: 'The Venn Diagram',
        icon: Layers,
        description: 'Translucent frosted-glass overlapping circles with soft blurs, high-contrast labels, and central core takeaway.',
        tag: 'Glassmorphism',
        category: 'comparison_contrast'
    },
    {
        id: 'quadrant_matrix_bauhaus',
        label: 'The Quadrant Matrix',
        icon: Grid,
        description: 'Bauhaus-inspired bold primary colors (red, blue, yellow), stark black lines, and 2x2 geometric precision grid.',
        tag: 'Bauhaus',
        category: 'comparison_contrast'
    },

    // Data & Statistics
    {
        id: 'corporate_dashboard_ui',
        label: 'The Corporate Dashboard',
        icon: BarChart3,
        description: 'Modern SaaS application UI with clean cards, rounded corners, hero statistic, donut chart, and bar graph.',
        tag: 'UI/UX App',
        category: 'data_statistics'
    },
    {
        id: 'typography_stat_sheet_swiss',
        label: 'The Large Typography Stat Sheet',
        icon: ListOrdered,
        description: 'Strict Swiss grid alignment, high-contrast black-and-white with vibrant accent color, focusing on massive bold numbers.',
        tag: 'Swiss Grid',
        category: 'data_statistics'
    },
    {
        id: 'geographic_map_hologram',
        label: 'The Geographic Map',
        icon: Globe,
        description: 'Futuristic sci-fi hologram with glowing wireframe maps, data nodes, deep blue/teal hues, and callout lines.',
        tag: 'Hologram',
        category: 'data_statistics'
    },
    {
        id: 'funnel_chart_neumorphism',
        label: 'The Funnel/Conversion Chart',
        icon: Filter,
        description: 'Soft extruded UI with subtle highlights and shadows in an inverted 4-layer pyramid with bold percentages.',
        tag: 'Neumorphism',
        category: 'data_statistics'
    },

    // Structure & Hierarchy
    {
        id: 'pyramid_hierarchy_lowpoly',
        label: 'The Pyramid/Hierarchy',
        icon: Triangle,
        description: 'Low-poly 3D art with geometric, faceted surfaces and sharp lighting in a triangle divided into horizontal slices.',
        tag: 'Low-Poly 3D',
        category: 'structure_hierarchy'
    },
    {
        id: 'hub_and_spoke_material',
        label: 'The Hub and Spoke',
        icon: GitFork,
        description: 'Google Material Design flat layers, intentional drop shadows, and playful colors with a core node connected to 6 nodes.',
        tag: 'Material Design',
        category: 'structure_hierarchy'
    },
    {
        id: 'anatomy_exploded_blueprint',
        label: 'The Anatomy/Exploded View',
        icon: Cpu,
        description: 'Technical blueprint with navy blue background, thin white grid lines, and precise drafting aesthetic.',
        tag: 'Blueprint',
        category: 'structure_hierarchy'
    },
    {
        id: 'mind_map_doodle',
        label: 'The Mind Map',
        icon: Share2,
        description: 'Hand-drawn doodle aesthetic with whiteboard marker textures and organic branching lines from a central bubble.',
        tag: 'Hand-Drawn',
        category: 'structure_hierarchy'
    },

    // Lists & Summaries
    {
        id: 'checklist_synthwave',
        label: 'The Checklist/Playbook',
        icon: CheckSquare,
        description: '80s Synthwave/Outrun with chrome text, neon grids, purple/orange sunsets, and large stylized checkboxes.',
        tag: 'Synthwave',
        category: 'lists_summaries'
    },
    {
        id: 'top_10_listicle_popart',
        label: 'The Top 10 Listicle',
        icon: Smile,
        description: 'Vintage comic book/Pop Art with halftone dot patterns, bold black outlines, and cascading numbered sequence (1-10).',
        tag: 'Pop-Art',
        category: 'lists_summaries'
    },
    {
        id: 'cheat_sheet_monochrome',
        label: 'The Cheat Sheet',
        icon: ListOrdered,
        description: 'Dense, highly organized multi-column typographic grid relying on varying font weights, sizes, and spacing.',
        tag: 'Monochrome',
        category: 'lists_summaries'
    },
    {
        id: 'problem_solution_duotone',
        label: 'The Problem/Solution Layout',
        icon: ArrowUpDown,
        description: 'Split-tone duotone with two contrasting colors (magenta and cyan), horizontal split for problem and solution.',
        tag: 'Duotone',
        category: 'lists_summaries'
    },

    // Classic Archetypes
    {
        id: 'technical_scientific',
        label: 'Technical & Scientific Diagrams',
        icon: Cpu,
        description: 'Physics concepts, technical systems (e.g. Kubernetes pods), or biology with schematics and callouts.',
        tag: 'Schematic',
        category: 'classic'
    },
    {
        id: 'step_by_step',
        label: 'Step-by-Step Guides & Recipes',
        icon: ListOrdered,
        description: 'Numbered sequential cards showing processes like cooking recipes or DIY workflows.',
        tag: 'Sequential',
        category: 'classic'
    },
    {
        id: 'flowchart_whiteboard',
        label: 'Flowcharts & Whiteboard Sketches',
        icon: GitFork,
        description: 'Hand-drawn dry-erase whiteboard style, notebook flowcharts, and organic brainstorming.',
        tag: 'Hand-drawn',
        category: 'classic'
    },
    {
        id: 'modular_explainer',
        label: 'Modular Explainers',
        icon: Boxes,
        description: 'Central hubs with connected radial components showing how complex systems operate.',
        tag: 'System',
        category: 'classic'
    },
    {
        id: 'timeline_historical',
        label: 'Timelines & Chronological Overviews',
        icon: History,
        description: 'Clean modern milestone markers tracking events or product evolution (modernized, no antique parchment).',
        tag: 'Chronological',
        category: 'classic'
    },
    {
        id: 'data_visualization',
        label: 'Data Visualizations',
        icon: BarChart3,
        description: 'Metrics, percentages, KPI cards, comparison columns, and structured financial summaries.',
        tag: 'Analytical',
        category: 'classic'
    },
    {
        id: 'playful_viral',
        label: 'Playful & Viral Listicles',
        icon: Smile,
        description: 'Lighthearted pop-art menus, humorous life steps, or colorful illustrated graphics.',
        tag: 'Pop-Art',
        category: 'classic'
    }
];

export const InfographicGenerator: React.FC<InfographicGeneratorProps> = ({
    userId,
    selectedText = '',
    onInfographicGenerated
}) => {
    const [storyText, setStoryText] = useState(selectedText);
    const [selectedArchetype, setSelectedArchetype] = useState<InfographicArchetype>('auto');
    const [selectedCategory, setSelectedCategory] = useState<StyleCategory>('all');
    const [searchQuery, setSearchQuery] = useState('');
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

    const filteredStyles = useMemo(() => {
        const query = searchQuery.trim().toLowerCase();
        return INFOGRAPHIC_STYLES.filter((style) => {
            const matchesCategory =
                selectedCategory === 'all' || style.category === selectedCategory;
            const matchesSearch =
                !query ||
                style.label.toLowerCase().includes(query) ||
                style.tag.toLowerCase().includes(query) ||
                style.description.toLowerCase().includes(query);
            return matchesCategory && matchesSearch;
        });
    }, [selectedCategory, searchQuery]);

    const selectedStyleObj = useMemo(() => {
        return INFOGRAPHIC_STYLES.find((s) => s.id === selectedArchetype);
    }, [selectedArchetype]);

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
                style: selectedArchetype,
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
                                Generates professional diagrams, flowcharts, timelines, and comparison visuals directly from article text.
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

            {/* Infographic Style & Archetype Selector */}
            <div className="space-y-3">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                    <div>
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                            Select Infographic Archetype & Style
                        </label>
                        {selectedStyleObj && (
                            <p className="text-xs text-emerald-600 dark:text-emerald-400 mt-0.5 font-medium">
                                Active: <span className="font-semibold">{selectedStyleObj.label}</span> ({selectedStyleObj.tag})
                            </p>
                        )}
                    </div>

                    {/* Quick search input */}
                    <div className="relative w-full sm:w-60">
                        <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
                        <input
                            type="text"
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            placeholder="Filter styles..."
                            className="w-full pl-8 pr-3 py-1.5 text-xs rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-emerald-500 text-gray-900 dark:text-white placeholder-gray-400"
                        />
                    </div>
                </div>

                {/* Category filter pills */}
                <div className="flex items-center gap-1.5 overflow-x-auto pb-1.5 scrollbar-thin">
                    {CATEGORIES.map((cat) => {
                        const isCatActive = selectedCategory === cat.id;
                        return (
                            <button
                                key={cat.id}
                                type="button"
                                onClick={() => setSelectedCategory(cat.id)}
                                className={`px-3 py-1.5 text-xs rounded-lg font-medium whitespace-nowrap transition-all ${
                                    isCatActive
                                        ? 'bg-emerald-600 text-white shadow-sm'
                                        : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'
                                }`}
                            >
                                {cat.label}
                            </button>
                        );
                    })}
                </div>

                {/* Helpful Chronological Option Guidance Banner */}
                {(selectedCategory === 'all' || selectedCategory === 'process_sequential' || selectedCategory === 'classic') && (
                    <div className="flex items-start gap-2.5 p-3 rounded-lg bg-amber-50/80 dark:bg-amber-950/20 border border-amber-200/70 dark:border-amber-900/50 text-[12px] text-amber-800 dark:text-amber-300 leading-snug">
                        <Info className="w-4 h-4 text-amber-600 dark:text-amber-400 mt-0.5 flex-shrink-0" />
                        <div>
                            <span className="font-semibold">Chronological & Timeline Options:</span> Use{' '}
                            <span className="font-semibold underline cursor-pointer" onClick={() => setSelectedArchetype('timeline_modern')}>
                                The Modern Milestone Timeline
                            </span>{' '}
                            for a clean tech roadmap without antique looks, or choose{' '}
                            <span className="font-semibold underline cursor-pointer" onClick={() => setSelectedArchetype('timeline_historical_vintage')}>
                                The Historical Timeline
                            </span>{' '}
                            if you specifically want vintage sepia/textured paper.
                        </div>
                    </div>
                )}

                {/* Grid of Styles */}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3 max-h-[380px] overflow-y-auto pr-1">
                    {filteredStyles.map((style) => {
                        const Icon = style.icon;
                        const isSelected = selectedArchetype === style.id;
                        return (
                            <div
                                key={style.id}
                                onClick={() => setSelectedArchetype(style.id)}
                                className={`p-3.5 rounded-xl border-2 cursor-pointer transition-all flex flex-col justify-between relative group ${
                                    isSelected
                                        ? 'border-emerald-600 bg-emerald-50/60 dark:bg-emerald-950/30 ring-2 ring-emerald-500/20 shadow-sm'
                                        : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600 bg-white dark:bg-gray-800/60'
                                }`}
                            >
                                <div>
                                    <div className="flex items-center justify-between mb-2">
                                        <div
                                            className={`p-2 rounded-lg transition-colors ${
                                                isSelected
                                                    ? 'bg-emerald-600 text-white'
                                                    : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 group-hover:bg-emerald-100 dark:group-hover:bg-emerald-900/40 group-hover:text-emerald-700 dark:group-hover:text-emerald-300'
                                            }`}
                                        >
                                            <Icon className="w-4 h-4" />
                                        </div>
                                        <span className={`text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-md ${
                                            isSelected
                                                ? 'bg-emerald-200 dark:bg-emerald-900/70 text-emerald-800 dark:text-emerald-200'
                                                : 'bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-400'
                                        }`}>
                                            {style.tag}
                                        </span>
                                    </div>
                                    <h4 className="text-xs font-bold text-gray-900 dark:text-white mb-1 flex items-center justify-between">
                                        <span>{style.label}</span>
                                        {isSelected && (
                                            <Check className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400 ml-1 flex-shrink-0" />
                                        )}
                                    </h4>
                                    <p className="text-[11px] text-gray-500 dark:text-gray-400 line-clamp-2 leading-relaxed">
                                        {style.description}
                                    </p>
                                </div>
                            </div>
                        );
                    })}
                </div>

                {filteredStyles.length === 0 && (
                    <div className="text-center py-8 text-gray-500 text-xs">
                        No styles match your search "{searchQuery}".
                    </div>
                )}
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
                    placeholder="e.g. Highlight step 3 in amber, emphasize growth metrics, or use dark cyan background"
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
