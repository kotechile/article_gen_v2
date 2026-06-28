import * as React from 'react';
import { motion } from 'framer-motion';
import { Play, Sparkles, Sliders, Palette, Video, Download, RefreshCw, Volume2, Eye } from 'lucide-react';
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

export function VideoStudio() {
    const [url, setUrl] = React.useState('');
    const [provider, setProvider] = React.useState<'openai' | 'elevenlabs'>('openai');
    const [voice, setVoice] = React.useState('onyx');
    const [customVoiceId, setCustomVoiceId] = React.useState('');
    const [captionPosition, setCaptionPosition] = React.useState<'center' | 'bottom' | 'top'>('center');
    
    // Brand colors state
    const [primaryColor, setPrimaryColor] = React.useState('#8A2BE2');
    const [secondaryColor, setSecondaryColor] = React.useState('#00FFFF');
    const [backgroundColor, setBackgroundColor] = React.useState('#0B0C10');

    // Status tracking
    const [generating, setGenerating] = React.useState(false);
    const [statusMessage, setStatusMessage] = React.useState('');
    const [videoUrl, setVideoUrl] = React.useState<string | null>(null);
    const [error, setError] = React.useState<string | null>(null);

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

    // Trigger video compilation API
    const handleGenerate = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!url) return;

        setGenerating(true);
        setVideoUrl(null);
        setError(null);
        setStatusMessage('Reading article content & generating script...');

        // Determine final voice string (custom vs preset)
        const finalVoice = provider === 'elevenlabs' && customVoiceId ? customVoiceId : voice;

        try {
            // Call Flask backend
            const response = await apiClient.post<any>('/v1/generate-video', {
                url,
                voice: finalVoice,
                provider,
                caption_position: captionPosition,
                primary_color: primaryColor,
                secondary_color: secondaryColor,
                background_color: backgroundColor
            }, {
                timeout: 300000 // 5 minutes client-side timeout
            });

            if (response.status === 'success') {
                setStatusMessage('Rendering completed! Fetching video file...');
                // The URL is served relative to client proxy base
                setVideoUrl('/api/v1/video/download');
            } else {
                throw new Error(response.message || 'Generation failed');
            }
        } catch (err: any) {
            console.error('Video generation error:', err);
            setError(err.response?.data?.message || err.message || 'An error occurred during video rendering.');
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
                    transition={{ duration: 0.25 }}
                    className="mb-8"
                >
                    <div className="flex items-center gap-2">
                        <Video className="h-6 w-6 text-primary animate-pulse" />
                        <h1 className="text-2xl font-semibold tracking-tight text-foreground">ArtiVids Studio</h1>
                    </div>
                    <p className="mt-1 text-sm text-muted-foreground">
                        Programmatic Article-to-Video engine with dynamic Remotion layouts.
                    </p>
                </motion.div>

                <div className="grid grid-cols-1 gap-8 lg:grid-cols-12">
                    {/* Settings Form Column */}
                    <div className="lg:col-span-7 space-y-6">
                        <form onSubmit={handleGenerate} className="space-y-6 rounded-xl border border-border bg-muted/20 p-6">
                            {/* URL Ingestion */}
                            <div className="space-y-2">
                                <label className="text-sm font-semibold text-foreground flex items-center gap-1.5">
                                    <Sparkles className="h-4 w-4 text-secondary" />
                                    Article URL
                                </label>
                                <input
                                    type="url"
                                    required
                                    value={url}
                                    onChange={(e) => setUrl(e.target.value)}
                                    placeholder="https://myblog.com/is-a-3000-espresso-maker-worth-it"
                                    className="h-10 w-full rounded-lg border border-border bg-background px-4 text-sm text-foreground outline-none transition placeholder:text-muted-foreground focus:border-primary/50"
                                />
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
                                            {provider === 'elevenlabs' && (
                                                <option value="custom">Custom Voice ID...</option>
                                            )}
                                        </select>
                                    </div>
                                </div>

                                {provider === 'elevenlabs' && voice === 'custom' && (
                                    <div className="space-y-1 animate-fadeIn">
                                        <label className="text-xs text-muted-foreground">Custom Voice ID</label>
                                        <input
                                            value={customVoiceId}
                                            onChange={(e) => setCustomVoiceId(e.target.value)}
                                            placeholder="Enter ElevenLabs Voice ID hash"
                                            className="h-9 w-full rounded-lg border border-border bg-background px-3 text-xs text-foreground outline-none"
                                        />
                                    </div>
                                )}
                            </div>

                            {/* Caption Positioning & Layout */}
                            <div className="space-y-4 border-t border-border pt-4">
                                <h3 className="text-sm font-bold text-foreground flex items-center gap-1.5">
                                    <Sliders className="h-4 w-4 text-cyan-400" />
                                    Subtitles Positioning
                                </h3>

                                <div className="space-y-1">
                                    <label className="text-xs text-muted-foreground">Subtitle Height Placement</label>
                                    <select
                                        value={captionPosition}
                                        onChange={(e) => setCaptionPosition(e.target.value as any)}
                                        className="h-9 w-full rounded-lg border border-border bg-background px-3 text-xs text-foreground outline-none"
                                    >
                                        <option value="center">Center Safe-Zone (PRD default center 30%)</option>
                                        <option value="bottom">Bottom Overlay (typical subtitles)</option>
                                        <option value="top">Top Header Overlay (leaves center clear)</option>
                                    </select>
                                </div>
                            </div>

                            {/* Colors */}
                            <div className="space-y-4 border-t border-border pt-4">
                                <div className="flex items-center justify-between">
                                    <h3 className="text-sm font-bold text-foreground flex items-center gap-1.5">
                                        <Palette className="h-4 w-4 text-amber-400" />
                                        Brand Colors
                                    </h3>
                                </div>

                                {/* Color Presets */}
                                <div className="flex gap-2">
                                    {COLOR_PRESETS.map((preset) => (
                                        <button
                                            key={preset.name}
                                            type="button"
                                            onClick={() => handleApplyPreset(preset)}
                                            className="flex items-center gap-1 px-2.5 py-1.5 rounded-md border border-border bg-background hover:bg-muted text-xs text-foreground transition"
                                        >
                                            <span
                                                className="w-2.5 h-2.5 rounded-full inline-block"
                                                style={{ backgroundColor: preset.primary }}
                                            />
                                            {preset.name}
                                        </button>
                                    ))}
                                </div>

                                <div className="grid grid-cols-3 gap-4 pt-2">
                                    <div className="space-y-1">
                                        <label className="text-xs text-muted-foreground">Primary Color</label>
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
                                        <label className="text-xs text-muted-foreground">Secondary Color</label>
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
                                        <label className="text-xs text-muted-foreground">Background</label>
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
                            </div>

                            {/* Submit */}
                            <button
                                type="submit"
                                disabled={generating || !url}
                                className="w-full h-11 rounded-lg text-white font-bold flex items-center justify-center gap-2 shadow-lg disabled:opacity-50 transition cursor-pointer"
                                style={{ backgroundColor: primaryColor }}
                            >
                                {generating ? (
                                    <>
                                        <RefreshCw className="h-5 w-5 animate-spin" />
                                        {statusMessage || 'Compiling Video...'}
                                    </>
                                ) : (
                                    <>
                                        <Play className="h-5 w-5 fill-white" />
                                        Compile Video Blueprint
                                    </>
                                )}
                            </button>
                        </form>
                    </div>

                    {/* Render Preview Column */}
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

                            {error && (
                                <div className="p-4 rounded-lg bg-destructive/10 border border-destructive/20 text-destructive text-xs max-w-sm text-center mb-4">
                                    <strong>Generation Error:</strong> {error}
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
                                        Provide an article URL on the left and click "Compile" to generate. Your finished video preview will load here.
                                    </p>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
