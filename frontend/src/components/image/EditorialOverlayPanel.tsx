import React from 'react';
import { Type, Sparkles, Loader2, Compass, CheckCircle2, Sliders, Eye, EyeOff } from 'lucide-react';
import type { OverlayConfig, OverlayDetails } from '../../types/image';

interface EditorialOverlayPanelProps {
    config: OverlayConfig;
    onChange: (config: OverlayConfig) => void;
    onDraftCopy: () => Promise<void>;
    draftingCopy?: boolean;
    overlayDetails?: OverlayDetails | null;
    onApplyOverlay?: () => Promise<void>;
    applyingOverlay?: boolean;
    hasGeneratedImage?: boolean;
    showWithOverlay?: boolean;
    onToggleOverlayPreview?: (show: boolean) => void;
}

export const EditorialOverlayPanel: React.FC<EditorialOverlayPanelProps> = ({
    config,
    onChange,
    onDraftCopy,
    draftingCopy = false,
    overlayDetails = null,
    onApplyOverlay,
    applyingOverlay = false,
    hasGeneratedImage = false,
    showWithOverlay = true,
    onToggleOverlayPreview
}) => {
    const handleToggle = (enabled: boolean) => {
        onChange({ ...config, enabled });
    };

    const handleKickerChange = (val: string) => {
        onChange({ ...config, kicker: val.toUpperCase().slice(0, 46) });
    };

    const handleTitleChange = (val: string) => {
        onChange({ ...config, title: val.toUpperCase().slice(0, 34) });
    };

    const handleHookChange = (val: string) => {
        onChange({ ...config, hook: val.slice(0, 58) });
    };

    const handleCornerChange = (corner: OverlayConfig['corner']) => {
        onChange({ ...config, corner });
    };

    return (
        <div className="rounded-xl border border-indigo-200/80 dark:border-indigo-800/80 bg-white/70 dark:bg-gray-900/60 shadow-sm overflow-hidden transition-all">
            {/* Header with Toggle */}
            <div className="flex items-center justify-between p-4 bg-gradient-to-r from-indigo-50/70 via-purple-50/40 to-transparent dark:from-indigo-950/40 dark:via-purple-950/20 dark:to-transparent border-b border-indigo-100 dark:border-indigo-900/50">
                <div className="flex items-center gap-3">
                    <div className="p-2 rounded-lg bg-indigo-600 text-white shadow-sm">
                        <Type className="w-4 h-4" />
                    </div>
                    <div>
                        <div className="flex items-center gap-2">
                            <h4 className="text-xs font-bold uppercase tracking-wider text-gray-900 dark:text-gray-100">
                                Editorial Cover Typography
                            </h4>
                            <span className="px-2 py-0.5 text-[10px] font-semibold bg-indigo-100 text-indigo-700 dark:bg-indigo-950 dark:text-indigo-300 rounded-full border border-indigo-200 dark:border-indigo-800">
                                PIL Post-Process
                            </span>
                        </div>
                        <p className="text-[11px] text-gray-500 dark:text-gray-400 mt-0.5">
                            Composite deterministic, high-contrast 3-tier headline typography over empty artwork corners with feathered scrims
                        </p>
                    </div>
                </div>

                <label className="relative inline-flex items-center cursor-pointer ml-3 flex-shrink-0">
                    <input
                        type="checkbox"
                        checked={config.enabled}
                        onChange={(e) => handleToggle(e.target.checked)}
                        className="sr-only peer"
                    />
                    <div className="w-9 h-5 bg-gray-200 peer-focus:outline-none rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all dark:border-gray-600 peer-checked:bg-indigo-600"></div>
                </label>
            </div>

            {config.enabled && (
                <div className="p-4 space-y-4">
                    {/* Action Bar: Auto-Draft Copy */}
                    <div className="flex items-center justify-between gap-3 pb-1 border-b border-gray-100 dark:border-gray-800">
                        <span className="text-xs font-medium text-gray-700 dark:text-gray-300">
                            3-Tier Editorial Copy Hierarchy
                        </span>
                        <button
                            type="button"
                            onClick={onDraftCopy}
                            disabled={draftingCopy}
                            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg text-indigo-700 dark:text-indigo-300 bg-indigo-50 dark:bg-indigo-950/60 hover:bg-indigo-100 dark:hover:bg-indigo-900/60 border border-indigo-200 dark:border-indigo-800 transition-colors disabled:opacity-50"
                        >
                            {draftingCopy ? (
                                <>
                                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                                    <span>Drafting Typography...</span>
                                </>
                            ) : (
                                <>
                                    <Sparkles className="w-3.5 h-3.5 text-indigo-600 dark:text-indigo-400" />
                                    <span>Auto-Draft from Article</span>
                                </>
                            )}
                        </button>
                    </div>

                    {/* Inputs */}
                    <div className="space-y-3">
                        {/* 1. Kicker */}
                        <div>
                            <div className="flex justify-between items-center mb-1">
                                <label className="text-[11px] font-semibold uppercase tracking-wider text-gray-600 dark:text-gray-400">
                                    Kicker (Beat & Dynamic • All Caps)
                                </label>
                                <span className={`text-[10px] ${(config.kicker || '').length >= 44 ? 'text-amber-500 font-bold' : 'text-gray-400'}`}>
                                    {(config.kicker || '').length} / 46
                                </span>
                            </div>
                            <input
                                type="text"
                                value={config.kicker || ''}
                                onChange={(e) => handleKickerChange(e.target.value)}
                                placeholder="E.G. LLM HARDWARE // MEMORY BOTTLENECK"
                                className="w-full text-xs px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-gray-50/50 dark:bg-gray-800/50 text-gray-900 dark:text-white placeholder-gray-400 focus:outline-none focus:ring-1 focus:ring-indigo-500 font-mono"
                            />
                        </div>

                        {/* 2. Title */}
                        <div>
                            <div className="flex justify-between items-center mb-1">
                                <label className="text-[11px] font-semibold uppercase tracking-wider text-gray-600 dark:text-gray-400">
                                    Headline Title (Core Story Subject • Max 2 Lines)
                                </label>
                                <span className={`text-[10px] ${(config.title || '').length >= 32 ? 'text-amber-500 font-bold' : 'text-gray-400'}`}>
                                    {(config.title || '').length} / 34
                                </span>
                            </div>
                            <input
                                type="text"
                                value={config.title || ''}
                                onChange={(e) => handleTitleChange(e.target.value)}
                                placeholder="E.G. THE SCALING CEILING"
                                className="w-full text-xs px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-gray-50/50 dark:bg-gray-800/50 text-gray-900 dark:text-white placeholder-gray-400 focus:outline-none focus:ring-1 focus:ring-indigo-500 font-bold"
                            />
                        </div>

                        {/* 3. Hook */}
                        <div>
                            <div className="flex justify-between items-center mb-1">
                                <label className="text-[11px] font-semibold uppercase tracking-wider text-gray-600 dark:text-gray-400">
                                    Hook Fragment (Reader Stake • Sentence Case)
                                </label>
                                <span className={`text-[10px] ${(config.hook || '').length >= 56 ? 'text-amber-500 font-bold' : 'text-gray-400'}`}>
                                    {(config.hook || '').length} / 58
                                </span>
                            </div>
                            <input
                                type="text"
                                value={config.hook || ''}
                                onChange={(e) => handleHookChange(e.target.value)}
                                placeholder="E.G. Bandwidth limits are reshaping inference cluster topologies."
                                className="w-full text-xs px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-gray-50/50 dark:bg-gray-800/50 text-gray-900 dark:text-white placeholder-gray-400 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                            />
                        </div>
                    </div>

                    {/* Spatial Corner Placement */}
                    <div className="pt-2 border-t border-gray-100 dark:border-gray-800">
                        <div className="flex items-center gap-1.5 mb-2">
                            <Compass className="w-3.5 h-3.5 text-indigo-500" />
                            <label className="text-[11px] font-semibold uppercase tracking-wider text-gray-600 dark:text-gray-400">
                                Corner Placement (54% × 50% Safe Box)
                            </label>
                        </div>
                        <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
                            {[
                                { id: 'auto', label: 'Auto (Edge Energy)' },
                                { id: 'top-left', label: 'Top Left' },
                                { id: 'top-right', label: 'Top Right' },
                                { id: 'bottom-left', label: 'Bottom Left' },
                                { id: 'bottom-right', label: 'Bottom Right' }
                            ].map((item) => {
                                const isSelected = (config.corner || 'auto') === item.id;
                                return (
                                    <button
                                        key={item.id}
                                        type="button"
                                        onClick={() => handleCornerChange(item.id as OverlayConfig['corner'])}
                                        className={`px-2.5 py-1.5 text-xs font-medium rounded-lg border transition-all text-center ${
                                            isSelected
                                                ? 'bg-indigo-600 text-white border-indigo-600 shadow-sm'
                                                : 'bg-gray-50 dark:bg-gray-800 border-gray-200 dark:border-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-750'
                                        }`}
                                    >
                                        {item.label}
                                    </button>
                                );
                            })}
                        </div>
                    </div>

                    {/* Live Image Re-apply & Toggle Controls (when image exists) */}
                    {hasGeneratedImage && (
                        <div className="p-3 bg-indigo-50/50 dark:bg-indigo-950/30 rounded-xl border border-indigo-200 dark:border-indigo-800/60 space-y-2">
                            <div className="flex flex-wrap items-center justify-between gap-2">
                                <div className="flex items-center gap-2">
                                    {onToggleOverlayPreview && (
                                        <button
                                            type="button"
                                            onClick={() => onToggleOverlayPreview(!showWithOverlay)}
                                            className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-semibold rounded-lg bg-white dark:bg-gray-800 border border-indigo-200 dark:border-indigo-700 text-indigo-700 dark:text-indigo-300 shadow-sm"
                                        >
                                            {showWithOverlay ? (
                                                <>
                                                    <Eye className="w-3.5 h-3.5 text-indigo-600" />
                                                    <span>Showing With Typography</span>
                                                </>
                                            ) : (
                                                <>
                                                    <EyeOff className="w-3.5 h-3.5 text-gray-500" />
                                                    <span>Showing Clean Base Art</span>
                                                </>
                                            )}
                                        </button>
                                    )}
                                </div>

                                {onApplyOverlay && (
                                    <button
                                        type="button"
                                        onClick={onApplyOverlay}
                                        disabled={applyingOverlay}
                                        className="inline-flex items-center gap-1.5 px-3 py-1 text-xs font-semibold rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white shadow-sm disabled:opacity-50"
                                    >
                                        {applyingOverlay ? (
                                            <>
                                                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                                                <span>Rendering Pillow Overlay...</span>
                                            </>
                                        ) : (
                                            <>
                                                <Sliders className="w-3.5 h-3.5" />
                                                <span>Re-render Typography on Image</span>
                                            </>
                                        )}
                                    </button>
                                )}
                            </div>

                            {/* Overlay Calibration Metadata Pill */}
                            {overlayDetails && (
                                <div className="flex flex-wrap items-center gap-2 pt-1 text-[11px] text-gray-600 dark:text-gray-300">
                                    <span className="inline-flex items-center gap-1 font-semibold text-emerald-700 dark:text-emerald-400">
                                        <CheckCircle2 className="w-3.5 h-3.5" />
                                        WCAG {overlayDetails.contrast_ratio}:1
                                    </span>
                                    <span>•</span>
                                    <span>
                                        Corner: <strong>{overlayDetails.corner}</strong>
                                    </span>
                                    <span>•</span>
                                    <span>
                                        Palette: <strong>{overlayDetails.ink_palette === 'dark' ? 'Warm Gold / Off-White' : 'Dark Amber / Charcoal'}</strong>
                                    </span>
                                    <span>•</span>
                                    <span>
                                        Vignette Scrim: <strong>{Math.round((overlayDetails.scrim_alpha / 255) * 100)}%</strong>
                                    </span>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};
