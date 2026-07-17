import * as React from 'react';
import { motion } from 'framer-motion';
import { Play, Sparkles, Sliders, Palette, Video, Download, RefreshCw, Volume2, Eye, ArrowLeft, Upload, Layout, Trash2, PlusCircle, ChevronUp, ChevronDown } from 'lucide-react';
import { apiClient } from '@/api-client';

// Color Presets for easy branding selection
const COLOR_PRESETS = [
    {
        name: 'Neon Magic',
        primary: '#8A2BE2',
        secondary: '#00FFFF',
        background: '#0B0C10',
    },
    {
        name: 'Emerald Slate',
        primary: '#10B981',
        secondary: '#6EE7B7',
        background: '#0F172A',
    },
    {
        name: 'Cyber Gold',
        primary: '#F59E0B',
        secondary: '#FBBF24',
        background: '#111827',
    },
    {
        name: 'Hot Crimson',
        primary: '#EF4444',
        secondary: '#F87171',
        background: '#180202',
    },
];

// Default ElevenLabs & OpenAI Voice presets
const VOICE_PRESETS = [
    { name: 'Onyx (OpenAI - Deep Male)', value: 'onyx', provider: 'openai' },
    { name: 'Nova (OpenAI - Energetic Female)', value: 'nova', provider: 'openai' },
    { name: 'Shimmer (OpenAI - Professional)', value: 'shimmer', provider: 'openai' },
    { name: 'Adam (ElevenLabs - Confident & Clear)', value: 'pNInz6obpgDQGcFmaJgB', provider: 'elevenlabs' },
    { name: 'Mark (ElevenLabs - Everyday Guy)', value: 'v3p1kjzUvro6S76qmYmH', provider: 'elevenlabs' },
    { name: 'Brian (ElevenLabs - Natural Storyteller)', value: 'n8kTUi6dVrplENT9Un56', provider: 'elevenlabs' },
    { name: 'Bill (ElevenLabs - Audiobook & Business)', value: 'pqHfZKP75CvOlQylNhV4', provider: 'elevenlabs' },
    { name: 'Aaden (ElevenLabs - Friendly Tutorial)', value: 'LB3SchLS4lHotJg4GcZW', provider: 'elevenlabs' },
    { name: 'Juniper (ElevenLabs - Conversational Female)', value: 'aMSt68OGf4xUZAnLpTU8', provider: 'elevenlabs' },
    { name: 'Bryan (ElevenLabs - Authoritative Polished)', value: 'n8kTUi6dVrplENT9Un56', provider: 'elevenlabs' },
    { name: 'David (ElevenLabs - Corporate Professional)', value: 'XjLkpWUlnhS8i7gGz3lZ', provider: 'elevenlabs' },
    { name: 'Rachel (ElevenLabs - Conversational)', value: '21m00Tcm4TlvDq8ikWAM', provider: 'elevenlabs' },
    { name: 'Dom (ElevenLabs - Deep Voice)', value: 'AZnzlk1XvdvUeBnXmlld', provider: 'elevenlabs' },
    { name: 'Nicole (ElevenLabs - Narrative)', value: 'piYvJZ4mWaZgI3LLhx4t', provider: 'elevenlabs' },
];

interface Scene {
    sceneId: string;
    type: 'framework_hero' | 'comparison_table' | 'kpi_metric' | 'broll_image' | 'call_to_action' | 'video_clip' | 'outro_logo';
    durationInSeconds: number;
    heading: string;
    subheading?: string;
    voiceoverScript: string;
    imagePrompt?: string;
    visualAssetUrl?: string;
    visualKeyword?: string;
    tableData?: {
        headers: string[];
        rows: string[][];
    };
    kpiData?: {
        value: string;
        label: string;
    };
    imageSizePercent?: number;
    imageValign?: 'top' | 'center' | 'bottom';
    imageHalign?: 'left' | 'center' | 'right';
}

interface Blueprint {
    metadata: {
        title: string;
        format: 'vertical' | 'landscape';
        totalDurationInSeconds: number;
        brandColors: {
            primary: string;
            secondary: string;
            background: string;
        };
        autoSyncTimings?: boolean;
    };
    scenes: Scene[];
}

const shouldDefaultToUpload = (scene: Scene): boolean => {
    const textToSearch = [
        scene.heading,
        scene.subheading,
        scene.voiceoverScript,
        scene.imagePrompt,
        scene.visualKeyword
    ].join(' ').toLowerCase();

    const videoKeywords = [
        'screencast', 'screencapture', 'screen capture', 'screen recording', 
        'types', 'clicks', 'drag', 'hover', 'slider', 'interface', 
        'browser', 'cursor', 'chart', 'plot', 'dashboard', 'donut chart',
        'simulation', 'input', 'types chief', 'tab', 'clicks search', 'storyboard'
    ];

    return videoKeywords.some(keyword => textToSearch.includes(keyword));
};

export function VideoStudio() {
    // Step configuration
    const [step, setStep] = React.useState<'config' | 'editor'>('config');

    const [inputMode, setInputMode] = React.useState<'url' | 'script'>('url');
    const [url, setUrl] = React.useState('');
    const [scriptText, setScriptText] = React.useState('');
    const [provider, setProvider] = React.useState<'openai' | 'elevenlabs'>('openai');
    const [voice, setVoice] = React.useState('onyx');
    const [customVoiceId, setCustomVoiceId] = React.useState('');
    const [captionPosition, setCaptionPosition] = React.useState<'center' | 'bottom' | 'top' | 'none'>('center');
    const [aspectRatio, setAspectRatio] = React.useState<'vertical' | 'landscape'>('landscape');
    const [music, setMusic] = React.useState('background.mp3');
    const [customMusicUrl, setCustomMusicUrl] = React.useState<string | null>(null);
    const [uploadingMusic, setUploadingMusic] = React.useState(false);
    
    // Brand colors state
    const [primaryColor, setPrimaryColor] = React.useState('#8A2BE2');
    const [secondaryColor, setSecondaryColor] = React.useState('#00FFFF');
    const [backgroundColor, setBackgroundColor] = React.useState('#0B0C10');
    const [brandLogoUrl, setBrandLogoUrl] = React.useState('');
    const [uploadingLogo, setUploadingLogo] = React.useState(false);

    // Blueprint blueprint content
    const [blueprint, setBlueprint] = React.useState<Blueprint | null>(null);

    // Status tracking
    const [generating, setGenerating] = React.useState(false);
    const [statusMessage, setStatusMessage] = React.useState('');
    const [videoUrl, setVideoUrl] = React.useState<string | null>(null);
    const [error, setError] = React.useState<string | null>(null);

    // Visual mode for each scene index: 'auto' | 'upload'
    const [sceneModes, setSceneModes] = React.useState<Record<number, 'auto' | 'upload'>>({});
    const [uploadingIndex, setUploadingIndex] = React.useState<number | null>(null);

    // Apply color presets helper
    const handleApplyPreset = (preset: typeof COLOR_PRESETS[0]) => {
        setPrimaryColor(preset.primary);
        setSecondaryColor(preset.secondary);
        setBackgroundColor(preset.background);
    };

    // Auto-switch provider filter based on voice selection
    const handleVoiceChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
        const val = e.target.value;
        setVoice(val);
        const preset = VOICE_PRESETS.find(p => p.value === val);
        if (preset) {
            setProvider(preset.provider as any);
        }
    };

    // Step 1: Ingest URL/Script and generate Blueprint JSON
    const handleGenerateBlueprint = async (e: React.FormEvent) => {
        e.preventDefault();
        if (inputMode === 'url' && !url) return;
        if (inputMode === 'script' && !scriptText) return;

        setGenerating(true);
        setVideoUrl(null);
        setError(null);
        setStatusMessage(inputMode === 'url' ? 'Reading article content & mapping video blueprint...' : 'Analyzing custom script & structuring scenes...');

        try {
            const response = await apiClient.post<any>('/v1/video/blueprint', {
                url: inputMode === 'url' ? url : 'custom',
                script_text: inputMode === 'script' ? scriptText : undefined,
                primary_color: primaryColor,
                secondary_color: secondaryColor,
                background_color: backgroundColor,
            });

            if (response.status === 'success' && response.blueprint) {
                const bp = response.blueprint;
                if (bp.metadata && bp.metadata.autoSyncTimings === undefined) {
                    bp.metadata.autoSyncTimings = true;
                }
                setBlueprint(bp);
                // Initialize visual mode dynamically
                const modes: Record<number, 'auto' | 'upload'> = {};
                bp.scenes.forEach((scene: any, idx: number) => {
                    modes[idx] = shouldDefaultToUpload(scene) ? 'upload' : 'auto';
                });
                setSceneModes(modes);
                setStep('editor');
            } else {
                throw new Error(response.message || 'Failed to construct script blueprint');
            }
        } catch (err: any) {
            console.error('Blueprint generation error:', err);
            const backendError = err.response?.data;
            const errMsg = backendError?.message || err.message || 'Failed to analyze article and plan scenes.';
            setError(errMsg);
        } finally {
            setGenerating(false);
            setStatusMessage('');
        }
    };

    // Handle scene file upload
    const handleSceneFileUpload = async (index: number, file: File) => {
        if (!file) return;

        setUploadingIndex(index);
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await apiClient.post<any>('/v1/video/upload', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });

            if (response.status === 'success') {
                const isVideo = file.name.toLowerCase().endsWith('.mp4') || 
                                file.name.toLowerCase().endsWith('.mov') || 
                                file.name.toLowerCase().endsWith('.webm') ||
                                file.name.toLowerCase().endsWith('.m4v');
                
                updateScene(index, {
                    visualAssetUrl: response.relative_path,
                    type: isVideo ? 'video_clip' : undefined
                });
            } else {
                throw new Error(response.message || 'File upload failed');
            }
        } catch (err: any) {
            alert(`Upload failed: ${err.message}`);
        } finally {
            setUploadingIndex(null);
        }
    };

    const handleMusicFileUpload = async (file: File) => {
        if (!file) return;
        setUploadingMusic(true);
        const formData = new FormData();
        formData.append('file', file);
        try {
            const response = await apiClient.post<any>('/v1/video/upload', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            if (response.status === 'success') {
                setCustomMusicUrl(response.relative_path);
            } else {
                throw new Error(response.message || 'Music upload failed');
            }
        } catch (err: any) {
            alert(`Music upload failed: ${err.message}`);
        } finally {
            setUploadingMusic(false);
        }
    };

    // Update specific scene in the blueprint
    const updateScene = (index: number, updates: Partial<Scene>) => {
        if (!blueprint) return;
        const newScenes = [...blueprint.scenes];
        newScenes[index] = { ...newScenes[index], ...updates };
        setBlueprint({
            ...blueprint,
            scenes: newScenes,
        });
    };

    // Delete scene from blueprint
    const deleteScene = (index: number) => {
        if (!blueprint) return;
        const newScenes = blueprint.scenes.filter((_, idx) => idx !== index);
        setBlueprint({
            ...blueprint,
            scenes: newScenes,
        });
    };

    // Reorder scene (Move Up / Down) in blueprint
    const moveScene = (index: number, direction: 'up' | 'down') => {
        if (!blueprint) return;
        const newScenes = [...blueprint.scenes];
        const targetIndex = direction === 'up' ? index - 1 : index + 1;
        if (targetIndex < 0 || targetIndex >= newScenes.length) return;

        // Swap the elements
        const temp = newScenes[index];
        newScenes[index] = newScenes[targetIndex];
        newScenes[targetIndex] = temp;

        setBlueprint({
            ...blueprint,
            scenes: newScenes,
        });
    };

    // Add a new scene with default structure based on selected layout template
    const addScene = (type: Scene['type']) => {
        if (!blueprint) return;
        const newScene: Scene = {
            sceneId: `scene_${Math.random().toString(36).substring(2, 9)}`,
            type,
            durationInSeconds: 6.0,
            heading: 'New Scene Title',
            subheading: 'Scene description or label',
            voiceoverScript: 'Spoken script details for this scene.',
            imagePrompt: 'Cinematic studio lighting, high tech professional concept, 8K resolution.',
            imageSizePercent: 100,
            imageValign: 'center',
            imageHalign: 'center',
        };

        if (type === 'comparison_table') {
            newScene.tableData = {
                headers: ['Feature', 'Option A', 'Option B'],
                rows: [
                    ['Benefit', 'High', 'Low'],
                    ['Cost', 'Free', 'Paid']
                ]
            };
        } else if (type === 'kpi_metric') {
            newScene.kpiData = {
                value: '95%',
                label: 'Efficiency'
            };
        } else if (type === 'outro_logo') {
            newScene.heading = '';
            newScene.subheading = '';
            newScene.voiceoverScript = '';
            newScene.imagePrompt = 'Gini Loh Outro Logo Display';
            newScene.durationInSeconds = 3.0;
        }

        setBlueprint({
            ...blueprint,
            scenes: [...blueprint.scenes, newScene],
        });
    };

    // Step 2: Render video using compiled blueprint
    const handleRenderVideo = async () => {
        if (!blueprint) return;

        setGenerating(true);
        setVideoUrl(null);
        setError(null);
        setStatusMessage('Synthesizing voiceover track & drawing visual scenes...');

        const finalVoice = provider === 'elevenlabs' && customVoiceId ? customVoiceId : voice;
        const finalMusic = music === 'custom' && customMusicUrl ? customMusicUrl : music;

        // Ensure final blueprint has correct layout colors / format / logo
        const finalBlueprint = {
            ...blueprint,
            metadata: {
                ...blueprint.metadata,
                format: aspectRatio,
                brandColors: {
                    primary: primaryColor,
                    secondary: secondaryColor,
                    background: backgroundColor,
                },
                brandLogoUrl: brandLogoUrl || undefined,
            },
        };

        // Calculate dynamic concurrency based on number of scenes (max 16, min 4)
        const dynamicConcurrency = Math.min(16, Math.max(4, (blueprint?.scenes.length || 5) * 2));

        try {
            const response = await apiClient.post<any>('/v1/generate-video', {
                url,
                voice: finalVoice,
                provider,
                caption_position: captionPosition,
                primary_color: primaryColor,
                secondary_color: secondaryColor,
                background_color: backgroundColor,
                aspect_ratio: aspectRatio,
                music: finalMusic,
                blueprint_payload: finalBlueprint,
                concurrency: dynamicConcurrency,
            }, {
                timeout: 900000,
            });

            if (response.status === 'success') {
                setStatusMessage('Rendering completed! Loading preview...');
                setVideoUrl('/api/v1/video/download');
            } else {
                throw new Error(response.message || 'Video render failed');
            }
        } catch (err: any) {
            console.error('Video generation error:', err);
            const backendError = err.response?.data;
            let errMsg = backendError?.message || err.message || 'An error occurred during video rendering.';
            if (backendError?.stderr || backendError?.stdout) {
                const details = backendError.stderr || backendError.stdout;
                errMsg += ` (Backend Details: ${details.slice(-300)})`;
            }
            setError(errMsg);
        } finally {
            setGenerating(false);
            setStatusMessage('');
        }
    };

    return (
        <div className="min-h-screen bg-background">
            <div className="mx-auto max-w-6xl px-8 py-10 lg:py-14">
                {/* Header */}
                <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mb-8 flex items-center justify-between"
                >
                    <div>
                        <div className="flex items-center gap-2">
                            <Video className="h-6 w-6 text-primary animate-pulse" />
                            <h1 className="text-2xl font-semibold tracking-tight text-foreground">ArtiVids Studio</h1>
                        </div>
                        <p className="mt-1 text-sm text-muted-foreground">
                            Two-step script planning and video compilation studio.
                        </p>
                    </div>

                    {step === 'editor' && (
                        <button
                            onClick={() => setStep('config')}
                            className="inline-flex h-9 px-4 items-center gap-1.5 rounded-lg border border-border bg-muted/30 text-xs font-semibold text-foreground hover:bg-muted cursor-pointer transition"
                        >
                            <ArrowLeft className="h-3.5 w-3.5" />
                            Restart Script
                        </button>
                    )}
                </motion.div>

                {error && (
                    <div className="p-4 rounded-lg bg-destructive/10 border border-destructive/20 text-destructive text-sm text-center mb-6">
                        <strong>Generation Error:</strong> {error}
                    </div>
                )}

                {step === 'config' ? (
                    <div className="grid grid-cols-1 gap-8 lg:grid-cols-12">
                        {/* Configuration Form */}
                        <div className="lg:col-span-7 space-y-6">
                            <form onSubmit={handleGenerateBlueprint} className="space-y-6 rounded-xl border border-border bg-muted/20 p-6">
                                {/* URL Ingestion */}
                                {/* URL / Script Ingestion */}
                                <div className="space-y-3">
                                    <label className="text-sm font-semibold text-foreground flex items-center gap-1.5">
                                        <Sparkles className="h-4 w-4 text-secondary" />
                                        Video Input Content Source
                                    </label>
                                    <div className="flex gap-2 rounded-lg bg-muted/40 p-1 border border-border/50 max-w-sm">
                                        <button
                                            type="button"
                                            onClick={() => setInputMode('url')}
                                            className={`flex-1 py-1.5 text-xs font-bold rounded-md transition cursor-pointer ${
                                                inputMode === 'url'
                                                    ? 'bg-background text-foreground shadow-sm border border-border/10'
                                                    : 'text-muted-foreground hover:text-foreground'
                                            }`}
                                        >
                                            Article URL
                                        </button>
                                        <button
                                            type="button"
                                            onClick={() => setInputMode('script')}
                                            className={`flex-1 py-1.5 text-xs font-bold rounded-md transition cursor-pointer ${
                                                inputMode === 'script'
                                                    ? 'bg-background text-foreground shadow-sm border border-border/10'
                                                    : 'text-muted-foreground hover:text-foreground'
                                            }`}
                                        >
                                            Custom Script Text
                                        </button>
                                    </div>

                                    {inputMode === 'url' ? (
                                        <input
                                            type="url"
                                            required
                                            value={url}
                                            onChange={(e) => setUrl(e.target.value)}
                                            placeholder="https://myblog.com/is-a-3000-espresso-maker-worth-it"
                                            className="h-10 w-full rounded-lg border border-border bg-background px-4 text-sm text-foreground outline-none transition placeholder:text-muted-foreground focus:border-primary/50"
                                        />
                                    ) : (
                                        <textarea
                                            required
                                            rows={8}
                                            value={scriptText}
                                            onChange={(e) => setScriptText(e.target.value)}
                                            placeholder={`Paste your custom script/storyboard. For example:\n\nScene 1: Introduction\nVO: "Is your job AI-proof? Let's check how task risk affects your career radar."\n\nScene 2: Search & Discover\nVO: "Start by searching over one thousand careers."`}
                                            className="w-full rounded-lg border border-border bg-background p-3 text-xs text-foreground outline-none transition placeholder:text-muted-foreground/60 focus:border-primary/50 font-mono resize-y"
                                        />
                                    )}
                                </div>

                                {/* Voice Setup */}
                                <div className="space-y-4 border-t border-border pt-4">
                                    <h3 className="text-sm font-bold text-foreground flex items-center gap-1.5">
                                        <Volume2 className="h-4 w-4 text-primary" />
                                        Audio & Voice Settings
                                    </h3>

                                    <div className="grid grid-cols-2 gap-4">
                                        <div className="space-y-1">
                                            <label className="text-xs text-muted-foreground">Provider</label>
                                            <select
                                                value={provider}
                                                onChange={(e) => setProvider(e.target.value as any)}
                                                className="h-9 w-full rounded-lg border border-border bg-background px-3 text-xs text-foreground outline-none"
                                            >
                                                <option value="openai">OpenAI (TTS-1)</option>
                                                <option value="elevenlabs">ElevenLabs</option>
                                            </select>
                                        </div>

                                        <div className="space-y-1">
                                            <label className="text-xs text-muted-foreground">Select Preset</label>
                                            <select
                                                value={voice}
                                                onChange={handleVoiceChange}
                                                className="h-9 w-full rounded-lg border border-border bg-background px-3 text-xs text-foreground outline-none"
                                            >
                                                {VOICE_PRESETS.map((v) => (
                                                    <option key={v.value} value={v.value}>
                                                        {v.name}
                                                    </option>
                                                ))}
                                            </select>
                                        </div>
                                    </div>

                                    {provider === 'elevenlabs' && (
                                        <div className="space-y-1 animate-scaleIn">
                                            <label className="text-xs text-muted-foreground">Custom ElevenLabs Voice ID</label>
                                            <input
                                                type="text"
                                                value={customVoiceId}
                                                onChange={(e) => setCustomVoiceId(e.target.value)}
                                                placeholder="e.g. pNInz6obpgDQGcFmaJgB"
                                                className="h-9 w-full rounded-lg border border-border bg-background px-3 text-xs text-foreground outline-none focus:border-primary/50"
                                            />
                                        </div>
                                    )}

                                    <div className="grid grid-cols-2 gap-4">
                                        <div className="space-y-1">
                                            <label className="text-xs text-muted-foreground">Caption Alignment</label>
                                            <select
                                                value={captionPosition}
                                                onChange={(e) => setCaptionPosition(e.target.value as any)}
                                                className="h-9 w-full rounded-lg border border-border bg-background px-3 text-xs text-foreground outline-none"
                                            >
                                                <option value="center">Center</option>
                                                <option value="bottom">Bottom (Standard)</option>
                                                <option value="top">Top</option>
                                                <option value="none">No Captions (Hidden)</option>
                                            </select>
                                        </div>

                                        <div className="space-y-1">
                                            <label className="text-xs text-muted-foreground">Aspect Ratio</label>
                                            <select
                                                value={aspectRatio}
                                                onChange={(e) => setAspectRatio(e.target.value as any)}
                                                className="h-9 w-full rounded-lg border border-border bg-background px-3 text-xs text-foreground outline-none"
                                            >
                                                <option value="vertical">Vertical Shorts (9:16)</option>
                                                <option value="landscape">Landscape YouTube (16:9)</option>
                                            </select>
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-2 gap-4">
                                        <div className="space-y-1">
                                            <label className="text-xs text-muted-foreground">Background Music</label>
                                            <select
                                                value={music}
                                                onChange={(e) => setMusic(e.target.value)}
                                                className="h-9 w-full rounded-lg border border-border bg-background px-3 text-xs text-foreground outline-none"
                                            >
                                                <option value="background.mp3">Cyberpunk Grid (Lo-Fi)</option>
                                                <option value="Corporate_Grid_low beat.mp3">Corporate Grid (Low Beat)</option>
                                                <option value="System_Architecture low beat.mp3">System Architecture (Low Beat)</option>
                                                <option value="custom">Upload Custom (.mp3)...</option>
                                                <option value="none">No Music (Voiceover Only)</option>
                                            </select>
                                        </div>

                                        {music === 'custom' && (
                                            <div className="space-y-1 animate-scaleIn flex flex-col justify-end">
                                                <input
                                                    type="file"
                                                    accept="audio/*,video/mp4,video/quicktime,.m4a,.aac,.mp3,.mp4,.mov"
                                                    id="music-upload-input"
                                                    className="hidden"
                                                    onChange={(e) => {
                                                        const file = e.target.files?.[0];
                                                        if (file) handleMusicFileUpload(file);
                                                    }}
                                                />
                                                <label
                                                    htmlFor="music-upload-input"
                                                    className="h-9 px-4 rounded border border-dashed border-border bg-muted/10 flex items-center justify-center gap-1.5 text-xs font-bold text-foreground cursor-pointer hover:bg-muted/20 transition"
                                                >
                                                    <Upload className="h-3.5 w-3.5" />
                                                    {uploadingMusic ? 'Uploading...' : 'Choose MP3 file'}
                                                </label>
                                                {customMusicUrl && (
                                                    <span className="text-[10px] text-primary truncate max-w-xs mt-1">
                                                        Uploaded: {customMusicUrl}
                                                    </span>
                                                )}
                                            </div>
                                        )}
                                    </div>

                                    {/* Auto-Sync Timings toggle */}
                                    <div className="pt-2 border-t border-border/50">
                                        <label className="flex items-center gap-2.5 cursor-pointer select-none">
                                            <input
                                                type="checkbox"
                                                checked={blueprint?.metadata?.autoSyncTimings ?? true}
                                                onChange={(e) => {
                                                    if (blueprint) {
                                                        setBlueprint({
                                                            ...blueprint,
                                                            metadata: {
                                                                ...blueprint.metadata,
                                                                autoSyncTimings: e.target.checked
                                                            }
                                                        });
                                                    }
                                                }}
                                                className="w-4 h-4 rounded border-border text-primary focus:ring-primary bg-background cursor-pointer accent-primary"
                                            />
                                            <div className="flex flex-col">
                                                <span className="text-xs font-bold text-foreground">Auto-Sync Timings</span>
                                                <span className="text-[10px] text-muted-foreground">Automatically detect and calculate scene durations from voiceover scripts and video files upon compilation</span>
                                            </div>
                                        </label>
                                    </div>
                                </div>

                                {/* Branding Presets */}
                                <div className="space-y-4 border-t border-border pt-4">
                                    <h3 className="text-sm font-bold text-foreground flex items-center gap-1.5">
                                        <Palette className="h-4 w-4 text-secondary" />
                                        Brand & Design Customization
                                    </h3>

                                    {/* Presets Grid */}
                                    <div className="grid grid-cols-4 gap-2">
                                        {COLOR_PRESETS.map((preset) => (
                                            <button
                                                key={preset.name}
                                                type="button"
                                                onClick={() => handleApplyPreset(preset)}
                                                className="rounded-lg border border-border bg-muted/40 p-2.5 hover:bg-muted cursor-pointer transition text-left"
                                            >
                                                <span className="block text-[10px] font-bold text-foreground truncate mb-1.5">
                                                    {preset.name}
                                                </span>
                                                <div className="flex gap-1.5">
                                                    <span className="w-3.5 h-3.5 rounded-full border border-black/30" style={{ backgroundColor: preset.primary }} />
                                                    <span className="w-3.5 h-3.5 rounded-full border border-black/30" style={{ backgroundColor: preset.secondary }} />
                                                    <span className="w-3.5 h-3.5 rounded-full border border-black/30" style={{ backgroundColor: preset.background }} />
                                                </div>
                                            </button>
                                        ))}
                                    </div>

                                    {/* Manual Pickers */}
                                    <div className="grid grid-cols-3 gap-4 pt-2">
                                        <div className="space-y-1">
                                            <label className="text-[10px] uppercase font-bold text-muted-foreground">Primary Accent</label>
                                            <div className="flex gap-2">
                                                <input
                                                    type="color"
                                                    value={primaryColor}
                                                    onChange={(e) => setPrimaryColor(e.target.value)}
                                                    className="w-8 h-8 rounded border border-border bg-transparent outline-none cursor-pointer"
                                                />
                                                <input
                                                    type="text"
                                                    value={primaryColor}
                                                    onChange={(e) => setPrimaryColor(e.target.value)}
                                                    className="h-8 w-full rounded border border-border bg-background px-2 text-xs outline-none"
                                                />
                                            </div>
                                        </div>

                                        <div className="space-y-1">
                                            <label className="text-[10px] uppercase font-bold text-muted-foreground">Secondary Accent</label>
                                            <div className="flex gap-2">
                                                <input
                                                    type="color"
                                                    value={secondaryColor}
                                                    onChange={(e) => setSecondaryColor(e.target.value)}
                                                    className="w-8 h-8 rounded border border-border bg-transparent outline-none cursor-pointer"
                                                />
                                                <input
                                                    type="text"
                                                    value={secondaryColor}
                                                    onChange={(e) => setSecondaryColor(e.target.value)}
                                                    className="h-8 w-full rounded border border-border bg-background px-2 text-xs outline-none"
                                                />
                                            </div>
                                        </div>

                                        <div className="space-y-1">
                                            <label className="text-[10px] uppercase font-bold text-muted-foreground">Canvas Background</label>
                                            <div className="flex gap-2">
                                                <input
                                                    type="color"
                                                    value={backgroundColor}
                                                    onChange={(e) => setBackgroundColor(e.target.value)}
                                                    className="w-8 h-8 rounded border border-border bg-transparent outline-none cursor-pointer"
                                                />
                                                <input
                                                    type="text"
                                                    value={backgroundColor}
                                                    onChange={(e) => setBackgroundColor(e.target.value)}
                                                    className="h-8 w-full rounded border border-border bg-background px-2 text-xs outline-none"
                                                />
                                            </div>
                                        </div>
                                    </div>

                                    {/* Brand Logo Upload */}
                                    <div className="space-y-2 border-t border-border pt-4 mt-2">
                                        <label className="text-[10px] uppercase font-bold text-muted-foreground block">Brand Logo Asset</label>
                                        <div className="flex items-center gap-4">
                                            <input
                                                type="file"
                                                accept="image/*"
                                                id="brand-logo-upload"
                                                className="hidden"
                                                onChange={async (e) => {
                                                    const file = e.target.files?.[0];
                                                    if (file) {
                                                        setUploadingLogo(true);
                                                        const formData = new FormData();
                                                        formData.append('file', file);
                                                        try {
                                                            const response = await apiClient.post<any>('/v1/video/upload', formData, {
                                                                headers: {
                                                                    'Content-Type': 'multipart/form-data',
                                                                },
                                                            });
                                                            if (response.status === 'success') {
                                                                setBrandLogoUrl(response.relative_path);
                                                            } else {
                                                                throw new Error(response.message || 'Logo upload failed');
                                                            }
                                                        } catch (err: any) {
                                                            alert(`Logo upload failed: ${err.message}`);
                                                        } finally {
                                                            setUploadingLogo(false);
                                                        }
                                                    }
                                                }}
                                            />
                                            <label
                                                htmlFor="brand-logo-upload"
                                                className="h-8 px-4 rounded border border-dashed border-border bg-muted/10 flex items-center justify-center gap-1.5 text-[10px] font-bold text-foreground cursor-pointer hover:bg-muted/20 transition"
                                            >
                                                <Upload className="h-3.5 w-3.5" />
                                                {uploadingLogo ? 'Uploading...' : 'Choose custom logo image'}
                                            </label>

                                            {brandLogoUrl ? (
                                                <div className="flex items-center gap-2">
                                                    <img 
                                                        src={brandLogoUrl.startsWith('http') ? brandLogoUrl : `/api/static/${brandLogoUrl}`} 
                                                        alt="Logo Thumbnail" 
                                                        className="w-8 h-8 rounded object-cover border border-border bg-muted"
                                                    />
                                                    <button
                                                        type="button"
                                                        onClick={() => setBrandLogoUrl('')}
                                                        className="text-[10px] text-red-500 font-bold hover:underline cursor-pointer"
                                                    >
                                                        Remove Custom
                                                    </button>
                                                </div>
                                            ) : (
                                                <div className="flex items-center gap-2">
                                                    <img 
                                                        src="/api/static/gini_loh_logo.jpg" 
                                                        alt="Default Gini Loh Logo" 
                                                        className="w-8 h-8 rounded object-cover border border-border bg-muted opacity-60"
                                                        onError={(e) => {
                                                            // fallback to public folder relative path if static api server route doesn't match
                                                            (e.target as HTMLImageElement).src = '/gini_loh_logo.jpg';
                                                        }}
                                                    />
                                                    <span className="text-[10px] text-muted-foreground italic">Using default brand outro logo</span>
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </div>

                                <button
                                    type="submit"
                                    disabled={generating || (inputMode === 'url' ? !url : !scriptText)}
                                    className="w-full h-11 rounded-lg text-white font-bold flex items-center justify-center gap-2 shadow-lg disabled:opacity-50 transition cursor-pointer"
                                    style={{ backgroundColor: primaryColor }}
                                >
                                    {generating ? (
                                        <>
                                            <RefreshCw className="h-5 w-5 animate-spin" />
                                            {statusMessage || 'Planning blueprint...'}
                                        </>
                                    ) : (
                                        <>
                                            <Sparkles className="h-5 w-5 fill-white" />
                                            Generate Script & Blueprint
                                        </>
                                    )}
                                </button>
                            </form>
                        </div>

                        {/* Right Preview Intro Column */}
                        <div className="lg:col-span-5 flex flex-col items-center">
                            <div className="w-full max-w-sm rounded-xl border border-border bg-muted/10 p-6 flex flex-col items-center min-h-[500px] justify-center relative shadow-lg text-center">
                                {generating && (
                                    <div className="absolute inset-0 z-10 bg-background/80 backdrop-blur-sm rounded-xl flex flex-col items-center justify-center text-center p-6">
                                        <div className="relative w-16 h-16 mb-4">
                                            <div className="absolute inset-0 rounded-full border-4 border-primary/20" />
                                            <div className="absolute inset-0 rounded-full border-4 border-t-primary animate-spin" />
                                        </div>
                                        <h4 className="text-base font-bold text-foreground mb-2">Analyzing Article</h4>
                                        <p className="text-xs text-muted-foreground max-w-xs">{statusMessage}</p>
                                    </div>
                                )}

                                <div className="text-muted-foreground flex flex-col items-center p-6">
                                    <div className="w-16 h-16 rounded-full bg-muted/50 flex items-center justify-center mb-4 border border-border">
                                        <Sliders className="h-6 w-6 text-muted-foreground animate-pulse" />
                                    </div>
                                    <h3 className="text-sm font-bold text-foreground mb-1">Interactive Planning</h3>
                                    <p className="text-xs max-w-xs">
                                        Paste the article link and click "Generate Script & Blueprint". You can preview, edit, customize images, or choose custom uploads before compilation.
                                    </p>
                                </div>
                            </div>
                        </div>
                    </div>
                ) : (
                    <div className="grid grid-cols-1 gap-8 lg:grid-cols-12">
                        {/* Step 2: Interactive Blueprint Scene Editor */}
                        <div className="lg:col-span-7 space-y-6">
                            <div className="rounded-xl border border-border bg-muted/20 p-6 space-y-6">
                                <div className="flex items-center justify-between border-b border-border pb-4">
                                    <h3 className="text-base font-bold text-foreground flex items-center gap-1.5">
                                        <Layout className="h-5 w-5 text-secondary" />
                                        Interactive Scene Timeline
                                    </h3>
                                    <span className="text-xs bg-primary/10 text-primary px-2.5 py-1 rounded-full font-bold">
                                        {blueprint?.scenes.length || 0} Scenes Planned
                                    </span>
                                </div>

                                <div className="space-y-6 max-h-[60vh] overflow-y-auto pr-2">
                                    {blueprint?.scenes.map((scene, idx) => (
                                        <div key={scene.sceneId} className="p-4 rounded-xl border border-border bg-background shadow-md space-y-4">
                                            <div className="flex items-center justify-between border-b border-border pb-2">
                                                <div className="flex items-center gap-2">
                                                    <span className="w-6 h-6 rounded-full bg-primary text-white text-xs font-bold flex items-center justify-center">
                                                        {idx + 1}
                                                    </span>
                                                    <span className="text-xs font-semibold text-foreground uppercase tracking-wider">
                                                        {scene.type.replace('_', ' ')}
                                                    </span>
                                                </div>
                                                <div className="flex items-center gap-3">
                                                    <div className="flex items-center gap-1" title={blueprint?.metadata?.autoSyncTimings ?? true ? "Auto-Sync is enabled: duration will be calculated from voiceover speech length and custom video files upon compile" : undefined}>
                                                        <input
                                                            type="number"
                                                            min="1"
                                                            max="60"
                                                            step="0.5"
                                                            disabled={blueprint?.metadata?.autoSyncTimings ?? true}
                                                            value={scene.durationInSeconds}
                                                            onChange={(e) => {
                                                                const val = parseFloat(e.target.value);
                                                                if (!isNaN(val) && val > 0) {
                                                                    updateScene(idx, { durationInSeconds: val });
                                                                }
                                                            }}
                                                            className="w-10 h-6 rounded border border-border bg-muted/20 text-center text-[10px] font-bold text-foreground outline-none focus:border-primary/50 disabled:opacity-60 disabled:cursor-not-allowed"
                                                        />
                                                        <span className="text-[10px] text-muted-foreground">{blueprint?.metadata?.autoSyncTimings ?? true ? "Auto" : "Secs"}</span>
                                                    </div>
                                                    <button
                                                        type="button"
                                                        onClick={() => moveScene(idx, 'up')}
                                                        disabled={idx === 0}
                                                        className="text-muted-foreground hover:text-primary transition cursor-pointer disabled:opacity-30 disabled:hover:text-muted-foreground"
                                                        title="Move Up"
                                                    >
                                                        <ChevronUp className="h-4 w-4" />
                                                    </button>
                                                    <button
                                                        type="button"
                                                        onClick={() => moveScene(idx, 'down')}
                                                        disabled={idx === (blueprint?.scenes.length || 0) - 1}
                                                        className="text-muted-foreground hover:text-primary transition cursor-pointer disabled:opacity-30 disabled:hover:text-muted-foreground"
                                                        title="Move Down"
                                                    >
                                                        <ChevronDown className="h-4 w-4" />
                                                    </button>
                                                    <button
                                                        type="button"
                                                        onClick={() => deleteScene(idx)}
                                                        className="text-muted-foreground hover:text-red-500 transition cursor-pointer"
                                                        title="Delete Scene"
                                                    >
                                                        <Trash2 className="h-4 w-4" />
                                                    </button>
                                                </div>
                                            </div>

                                            {/* Text Customization */}
                                            <div className="grid grid-cols-2 gap-4">
                                                <div className="space-y-1">
                                                    <label className="text-[10px] font-bold text-muted-foreground uppercase">Heading Text</label>
                                                    <input
                                                        type="text"
                                                        value={scene.heading}
                                                        onChange={(e) => updateScene(idx, { heading: e.target.value })}
                                                        className="h-8 w-full rounded border border-border bg-muted/20 px-2 text-xs outline-none focus:border-primary/50 text-foreground"
                                                    />
                                                </div>
                                                <div className="space-y-1">
                                                    <label className="text-[10px] font-bold text-muted-foreground uppercase">Subheading / Label</label>
                                                    <input
                                                        type="text"
                                                        value={scene.subheading || ''}
                                                        onChange={(e) => updateScene(idx, { subheading: e.target.value })}
                                                        className="h-8 w-full rounded border border-border bg-muted/20 px-2 text-xs outline-none focus:border-primary/50 text-foreground"
                                                    />
                                                </div>
                                            </div>

                                            {/* Voiceover Edit */}
                                            <div className="space-y-1">
                                                <label className="text-[10px] font-bold text-muted-foreground uppercase">Voiceover Words</label>
                                                <textarea
                                                    rows={2}
                                                    value={scene.voiceoverScript}
                                                    onChange={(e) => updateScene(idx, { voiceoverScript: e.target.value })}
                                                    className="w-full rounded border border-border bg-muted/20 p-2 text-xs outline-none focus:border-primary/50 text-foreground resize-none"
                                                />
                                            </div>

                                            {/* Table / KPI Meta Editing */}
                                            {scene.type === 'comparison_table' && scene.tableData && (
                                                <div className="p-3 rounded-lg border border-dashed border-border bg-muted/5 space-y-3">
                                                    <div>
                                                        <span className="text-[10px] font-bold uppercase text-muted-foreground block mb-1">Edit Table Headers</span>
                                                        <div 
                                                            className="grid gap-2"
                                                            style={{ gridTemplateColumns: `repeat(${scene.tableData?.headers?.length || 2}, minmax(0, 1fr))` }}
                                                        >
                                                            {(scene.tableData.headers || []).map((h, hidx) => (
                                                                <input
                                                                    key={hidx}
                                                                    type="text"
                                                                    value={h}
                                                                    onChange={(e) => {
                                                                        const newHeaders = [...(scene.tableData?.headers || [])];
                                                                        newHeaders[hidx] = e.target.value;
                                                                        updateScene(idx, {
                                                                            tableData: {
                                                                                headers: newHeaders,
                                                                                rows: scene.tableData?.rows || [],
                                                                            },
                                                                        });
                                                                    }}
                                                                    className="h-7 rounded border border-border bg-background px-2 text-[10px] outline-none text-foreground font-bold"
                                                                />
                                                            ))}
                                                        </div>
                                                    </div>

                                                    <div>
                                                        <span className="text-[10px] font-bold uppercase text-muted-foreground block mb-1">Edit Table Rows</span>
                                                        <div className="space-y-2">
                                                            {(scene.tableData.rows || []).map((row, ridx) => (
                                                                <div 
                                                                    key={ridx} 
                                                                    className="grid gap-2"
                                                                    style={{ gridTemplateColumns: `repeat(${scene.tableData?.headers?.length || 2}, minmax(0, 1fr))` }}
                                                                >
                                                                    {row.map((cell, cidx) => (
                                                                        <input
                                                                            key={cidx}
                                                                            type="text"
                                                                            value={cell}
                                                                            onChange={(e) => {
                                                                                const newRows = (scene.tableData?.rows || []).map((r, r_i) => {
                                                                                    if (r_i === ridx) {
                                                                                        const newRow = [...r];
                                                                                        newRow[cidx] = e.target.value;
                                                                                        return newRow;
                                                                                    }
                                                                                    return r;
                                                                                });
                                                                                updateScene(idx, {
                                                                                    tableData: {
                                                                                        headers: scene.tableData?.headers || [],
                                                                                        rows: newRows,
                                                                                    },
                                                                                });
                                                                            }}
                                                                            className="h-7 rounded border border-border bg-background px-2 text-[10px] outline-none text-foreground"
                                                                            placeholder={`Row ${ridx + 1} Col ${cidx + 1}`}
                                                                        />
                                                                    ))}
                                                                </div>
                                                            ))}
                                                        </div>
                                                    </div>
                                                </div>
                                            )}

                                            {scene.type === 'kpi_metric' && scene.kpiData && (
                                                <div className="p-3 rounded-lg border border-dashed border-border bg-muted/5 grid grid-cols-2 gap-4">
                                                    <div className="space-y-1">
                                                        <label className="text-[10px] font-bold uppercase text-muted-foreground">Metric Value</label>
                                                        <input
                                                            type="text"
                                                            value={scene.kpiData.value}
                                                            onChange={(e) => {
                                                                updateScene(idx, {
                                                                    kpiData: {
                                                                        value: e.target.value,
                                                                        label: scene.kpiData?.label || '',
                                                                    },
                                                                });
                                                            }}
                                                            className="h-7 w-full rounded border border-border bg-background px-2 text-[10px] outline-none text-foreground font-bold"
                                                        />
                                                    </div>
                                                    <div className="space-y-1">
                                                        <label className="text-[10px] font-bold uppercase text-muted-foreground">Metric label</label>
                                                        <input
                                                            type="text"
                                                            value={scene.kpiData.label}
                                                            onChange={(e) => {
                                                                updateScene(idx, {
                                                                    kpiData: {
                                                                        value: scene.kpiData?.value || '',
                                                                        label: e.target.value,
                                                                    },
                                                                });
                                                            }}
                                                            className="h-7 w-full rounded border border-border bg-background px-2 text-[10px] outline-none text-foreground"
                                                        />
                                                    </div>
                                                </div>
                                            )}

                                            {/* Visual Asset Control */}
                                            <div className="space-y-2 pt-2 border-t border-border">
                                                <div className="flex items-center justify-between">
                                                    <span className="text-[10px] font-bold text-muted-foreground uppercase">
                                                        {scene.type === 'video_clip' ? 'Background Video Selection' : 'Background Image Selection'}
                                                    </span>
                                                    {scene.type !== 'video_clip' && (
                                                        <div className="flex gap-2">
                                                            <button
                                                                type="button"
                                                                onClick={() => setSceneModes({ ...sceneModes, [idx]: 'auto' })}
                                                                className={`px-2.5 py-1 rounded text-[10px] font-semibold transition cursor-pointer ${
                                                                    sceneModes[idx] === 'auto'
                                                                        ? 'bg-secondary text-background font-bold'
                                                                        : 'bg-muted/40 text-muted-foreground hover:bg-muted'
                                                                }`}
                                                            >
                                                                Auto AI (Flux)
                                                            </button>
                                                            <button
                                                                type="button"
                                                                onClick={() => setSceneModes({ ...sceneModes, [idx]: 'upload' })}
                                                                className={`px-2.5 py-1 rounded text-[10px] font-semibold transition cursor-pointer ${
                                                                    sceneModes[idx] === 'upload'
                                                                        ? 'bg-secondary text-background font-bold'
                                                                        : 'bg-muted/40 text-muted-foreground hover:bg-muted'
                                                                }`}
                                                            >
                                                                Upload Custom
                                                            </button>
                                                        </div>
                                                    )}
                                                </div>

                                                {(sceneModes[idx] === 'auto' && scene.type !== 'video_clip') ? (
                                                    <div className="space-y-1">
                                                        <label className="text-[9px] text-muted-foreground">Flux AI Image Prompt</label>
                                                        <input
                                                            type="text"
                                                            value={scene.imagePrompt || ''}
                                                            onChange={(e) => updateScene(idx, { imagePrompt: e.target.value })}
                                                            placeholder="Flux image description prompt..."
                                                            className="h-8 w-full rounded border border-border bg-muted/20 px-2 text-xs outline-none focus:border-primary/50 text-foreground"
                                                        />
                                                    </div>
                                                ) : (
                                                    <div className="flex items-center gap-3">
                                                        <input
                                                            type="file"
                                                            accept="image/*,video/mp4,video/quicktime,.mp4,.mov,.webm,.m4v"
                                                            id={`upload-${scene.sceneId}`}
                                                            className="hidden"
                                                            onChange={(e) => {
                                                                const file = e.target.files?.[0];
                                                                if (file) handleSceneFileUpload(idx, file);
                                                            }}
                                                        />
                                                        <label
                                                            htmlFor={`upload-${scene.sceneId}`}
                                                            className="h-8 px-4 rounded border border-dashed border-border bg-muted/10 flex items-center justify-center gap-1 text-[10px] font-bold text-foreground cursor-pointer hover:bg-muted/20 transition"
                                                        >
                                                            <Upload className="h-3 w-3" />
                                                            {uploadingIndex === idx 
                                                                ? 'Uploading...' 
                                                                : scene.type === 'video_clip' 
                                                                    ? 'Choose custom video (MP4)' 
                                                                    : 'Choose custom image'}
                                                        </label>
                                                        {scene.visualAssetUrl && (
                                                            <span className="text-[10px] text-primary truncate max-w-xs">
                                                                Uploaded: {scene.visualAssetUrl.split('/').pop()}
                                                            </span>
                                                        )}
                                                    </div>
                                                )}
                                                <div className="space-y-1 mt-2">
                                                    <label className="text-[9px] text-muted-foreground flex justify-between">
                                                        <span>Image Screen Scale (Fit/Centering Size)</span>
                                                        <span className="font-bold text-primary">{scene.imageSizePercent || 100}%</span>
                                                    </label>
                                                    <input
                                                        type="range"
                                                        min="30"
                                                        max="100"
                                                        step="5"
                                                        value={scene.imageSizePercent || 100}
                                                        onChange={(e) => updateScene(idx, { imageSizePercent: parseInt(e.target.value) })}
                                                        className="w-full h-1 bg-muted rounded-lg appearance-none cursor-pointer accent-primary"
                                                    />
                                                </div>
                                                {scene.imageSizePercent && scene.imageSizePercent < 100 && (
                                                    <div className="grid grid-cols-2 gap-2 mt-2">
                                                        <div className="space-y-1">
                                                            <label className="text-[9px] text-muted-foreground">Vertical Align</label>
                                                            <select
                                                                value={scene.imageValign || 'center'}
                                                                onChange={(e) => updateScene(idx, { imageValign: e.target.value as any })}
                                                                className="h-7 w-full rounded border border-border bg-background px-2 text-[10px] text-foreground outline-none"
                                                            >
                                                                <option value="top">Top</option>
                                                                <option value="center">Center</option>
                                                                <option value="bottom">Bottom</option>
                                                            </select>
                                                        </div>
                                                        <div className="space-y-1">
                                                            <label className="text-[9px] text-muted-foreground">Horizontal Align</label>
                                                            <select
                                                                value={scene.imageHalign || 'center'}
                                                                onChange={(e) => updateScene(idx, { imageHalign: e.target.value as any })}
                                                                className="h-7 w-full rounded border border-border bg-background px-2 text-[10px] text-foreground outline-none"
                                                            >
                                                                <option value="left">Left</option>
                                                                <option value="center">Center</option>
                                                                <option value="right">Right</option>
                                                            </select>
                                                        </div>
                                                    </div>
                                                )}
                                            </div>
                                        </div>
                                    ))}
                                </div>

                                {/* Add Scene Section */}
                                <div className="border border-dashed border-border rounded-xl p-4 bg-muted/5 flex flex-col items-center justify-center gap-3">
                                    <span className="text-xs font-bold text-muted-foreground flex items-center gap-1">
                                        <PlusCircle className="h-4 w-4 text-primary" />
                                        Add Custom Scene Layout
                                    </span>
                                    <div className="flex flex-wrap justify-center gap-2">
                                        <button
                                            type="button"
                                            onClick={() => addScene('broll_image')}
                                            className="px-3 py-1.5 rounded-lg border border-border bg-background hover:bg-muted text-[10px] font-bold text-foreground cursor-pointer transition animate-scaleIn"
                                        >
                                            + Detail Image
                                        </button>
                                        <button
                                            type="button"
                                            onClick={() => addScene('framework_hero')}
                                            className="px-3 py-1.5 rounded-lg border border-border bg-background hover:bg-muted text-[10px] font-bold text-foreground cursor-pointer transition animate-scaleIn"
                                        >
                                            + Hero Slide
                                        </button>
                                        <button
                                            type="button"
                                            onClick={() => addScene('kpi_metric')}
                                            className="px-3 py-1.5 rounded-lg border border-border bg-background hover:bg-muted text-[10px] font-bold text-foreground cursor-pointer transition animate-scaleIn"
                                        >
                                            + KPI Metric
                                        </button>
                                        <button
                                            type="button"
                                            onClick={() => addScene('comparison_table')}
                                            className="px-3 py-1.5 rounded-lg border border-border bg-background hover:bg-muted text-[10px] font-bold text-foreground cursor-pointer transition animate-scaleIn"
                                        >
                                            + Table
                                        </button>
                                        <button
                                            type="button"
                                            onClick={() => addScene('call_to_action')}
                                            className="px-3 py-1.5 rounded-lg border border-border bg-background hover:bg-muted text-[10px] font-bold text-foreground cursor-pointer transition animate-scaleIn"
                                        >
                                            + Call To Action
                                        </button>
                                        <button
                                            type="button"
                                            onClick={() => addScene('video_clip')}
                                            className="px-3 py-1.5 rounded-lg border border-border bg-background hover:bg-muted text-[10px] font-bold text-foreground cursor-pointer transition animate-scaleIn"
                                        >
                                            + Video Clip
                                        </button>
                                        <button
                                            type="button"
                                            onClick={() => addScene('outro_logo')}
                                            className="px-3 py-1.5 rounded-lg border border-border bg-background hover:bg-muted text-[10px] font-bold text-foreground cursor-pointer transition animate-scaleIn"
                                        >
                                            + Brand Outro
                                        </button>
                                    </div>
                                </div>

                                <button
                                    type="button"
                                    onClick={handleRenderVideo}
                                    disabled={generating}
                                    className="w-full h-11 rounded-lg text-white font-bold flex items-center justify-center gap-2 shadow-lg disabled:opacity-50 transition cursor-pointer"
                                    style={{ backgroundColor: primaryColor }}
                                >
                                    {generating ? (
                                        <>
                                            <RefreshCw className="h-5 w-5 animate-spin" />
                                            {statusMessage || 'Rendering Final Video...'}
                                        </>
                                    ) : (
                                        <>
                                            <Play className="h-5 w-5 fill-white" />
                                            Compile Final Video
                                        </>
                                    )}
                                </button>
                            </div>
                        </div>

                        {/* Rendering Preview Output Column */}
                        <div className="lg:col-span-5 flex flex-col items-center">
                            <div className="w-full max-w-sm rounded-xl border border-border bg-muted/10 p-6 flex flex-col items-center min-h-[500px] justify-center relative shadow-lg">
                                {generating && (
                                    <div className="absolute inset-0 z-10 bg-background/80 backdrop-blur-sm rounded-xl flex flex-col items-center justify-center text-center p-6">
                                        <div className="relative w-16 h-16 mb-4">
                                            <div className="absolute inset-0 rounded-full border-4 border-primary/20" />
                                            <div className="absolute inset-0 rounded-full border-4 border-t-primary animate-spin" />
                                        </div>
                                        <h4 className="text-base font-bold text-foreground mb-2">Rendering Serverlessly</h4>
                                        <p className="text-xs text-muted-foreground max-w-xs">{statusMessage}</p>
                                    </div>
                                )}

                                {videoUrl ? (
                                    <div className="w-full flex flex-col items-center animate-scaleIn">
                                        {/* Video Preview */}
                                        <div className="w-[280px] h-[500px] rounded-2xl overflow-hidden border border-border bg-black shadow-2xl relative flex items-center justify-center group mb-4">
                                            <video
                                                controls
                                                src={videoUrl}
                                                className="w-full h-full object-cover"
                                                poster="/thumbnail.jpg"
                                            />
                                        </div>

                                        {/* Download button */}
                                        <a
                                            href={videoUrl}
                                            download="output-generated.mp4"
                                            className="inline-flex h-10 px-6 items-center justify-center rounded-lg bg-secondary hover:bg-secondary/95 text-background font-bold gap-2 shadow-lg transition"
                                        >
                                            <Download className="h-4 w-4" />
                                            Download MP4 Video
                                        </a>
                                    </div>
                                ) : (
                                    <div className="text-center text-muted-foreground flex flex-col items-center p-6">
                                        <div className="w-16 h-16 rounded-full bg-muted/50 flex items-center justify-center mb-4 border border-border">
                                            <Eye className="h-6 w-6 text-muted-foreground" />
                                        </div>
                                        <h3 className="text-sm font-bold text-foreground mb-1">Live Video Blueprint</h3>
                                        <p className="text-xs max-w-xs">
                                            Edit your script and scenes on the left, customize background images (Flux prompts or uploads), then compile to render.
                                        </p>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
