import React, { useEffect, useState, useRef } from 'react';
import axios from 'axios';
import { supabase } from '../lib/supabase';
import { Loader2, AlertCircle, Sparkles, Wand2 } from 'lucide-react';
import { updateArticleAfterGeneration } from '../lib/contentParser';
import { useAuth } from '../context/auth-context';

function getContentStudioGenerationStorageKey(articleId: string): string {
    return `content_studio_generation_${articleId}`;
}

interface GenerationModalProps {
    articleId: string;
    taskId?: string | null;
    isOpen: boolean;
    onClose: () => void;
    onComplete: (articleId: string) => void;
}

export const GenerationModal: React.FC<GenerationModalProps> = ({ articleId, taskId, isOpen, onClose, onComplete }) => {
    const { user } = useAuth();
    const [status, setStatus] = useState<string>('Initializing...');
    const [progress, setProgress] = useState(0);
    const [error, setError] = useState<string | null>(null);
    const hasUpdatedRef = useRef(false);
    const apiFailureCountRef = useRef(0);
    const transientFailureCountRef = useRef(0);

    // Controversies flow states
    const [modalView, setModalView] = useState<'progress' | 'controversies'>('progress');
    const [controversies, setControversies] = useState<any[]>([]);
    const [selectedControversies, setSelectedControversies] = useState<Record<string, { selected: boolean; selectedTakeId: string }>>({});
    const [originalResearchData, setOriginalResearchData] = useState<any>(null);
    const [activeTaskId, setActiveTaskId] = useState<string | null>(taskId || null);
    const [isResuming, setIsResuming] = useState(false);

    // Reset local states when modal opens or primary taskId changes
    useEffect(() => {
        if (isOpen) {
            setModalView('progress');
            setActiveTaskId(taskId || null);
            setError(null);
            setProgress(0);
            setStatus('Initializing...');
            setControversies([]);
            setSelectedControversies({});
            setOriginalResearchData(null);
            setIsResuming(false);
            hasUpdatedRef.current = false;
        }
    }, [isOpen, taskId]);

    useEffect(() => {
        if (!isOpen) return;

        // Restore progress from storage if available (only if not starting fresh task)
        const storedProgress = localStorage.getItem(`gen_progress_${articleId}`);
        if (storedProgress && !activeTaskId) {
            setProgress(parseFloat(storedProgress));
        }

        let pollInterval: any;

        const checkStatus = async () => {
            // 1. Prefer API Polling if activeTaskId is available
            if (activeTaskId) {
                try {
                    const { data } = await axios.get(`${import.meta.env.VITE_API_URL || 'http://localhost:5001'}/api/v1/research/${activeTaskId}`, {
                        headers: { 'X-API-Key': 'development' }
                    });
                    apiFailureCountRef.current = 0;

                    // Handle different states
                    if (data.status === 'SUCCESS') {
                        // Check if we hit the controversy selection gate
                        if (data.pending_controversies) {
                            setControversies(data.controversy_options || []);
                            
                            // Initialize default selections (select first 3 by default, and pick first take)
                            const initialSelections: Record<string, { selected: boolean; selectedTakeId: string }> = {};
                            (data.controversy_options || []).forEach((c: any, index: number) => {
                                initialSelections[c.id] = {
                                    selected: index < 3,
                                    selectedTakeId: c.takes[0]?.id || ''
                                };
                            });
                            setSelectedControversies(initialSelections);
                            
                            // Store original research data for resuming
                            setOriginalResearchData(data.result?.research_data || null);
                            
                            setModalView('controversies');
                            clearInterval(pollInterval);
                            return;
                        }

                        setProgress(100);
                        setStatus('Completed');
                        localStorage.removeItem(getContentStudioGenerationStorageKey(articleId));
                        localStorage.removeItem(`gen_progress_${articleId}`);

                        // Trigger final updates
                        if (!hasUpdatedRef.current && user?.id) {
                            hasUpdatedRef.current = true;
                            try {
                                await updateArticleAfterGeneration(articleId);
                            } catch (updateErr) {
                                console.error("Failed to update ToC:", updateErr);
                            }
                        }

                        setTimeout(() => onComplete(articleId), 1500);
                        clearInterval(pollInterval);
                        return;
                    }
                    else if (['FAILURE', 'REVOKED'].includes(data.status)) {
                        const failureMessage = typeof data.info === 'string'
                            ? data.info
                            : (data.info?.message || data.message || 'Generation failed');
                        const rawError = String(data.error || '');
                        const transientStatusFailure =
                            /status temporarily unavailable/i.test(failureMessage) ||
                            /persisting result metadata/i.test(failureMessage) ||
                            /exception type/i.test(rawError);

                        if (transientStatusFailure) {
                            transientFailureCountRef.current += 1;
                            setStatus(`Status sync issue (${transientFailureCountRef.current})... retrying`);
                            if (transientFailureCountRef.current >= 12) {
                                setError('Could not read task status after multiple retries. Please refresh and check the article state.');
                                localStorage.removeItem(getContentStudioGenerationStorageKey(articleId));
                                clearInterval(pollInterval);
                            }
                            return;
                        }

                        setError(`${failureMessage}${rawError ? ` (${rawError})` : ''}`);
                        localStorage.removeItem(getContentStudioGenerationStorageKey(articleId));
                        clearInterval(pollInterval);
                        return;
                    }
                    else if (data.status === 'PROGRESS' && data.info) {
                        transientFailureCountRef.current = 0;
                        // Granular updates!
                        if (data.info.message) setStatus(data.info.message);
                        if (data.info.progress) {
                            setProgress(data.info.progress);
                            // Store for resume (optional)
                            localStorage.setItem(`gen_progress_${articleId}`, data.info.progress.toString());
                        }
                    } else {
                        // PENDING or other states
                        setStatus(`Status: ${data.status}`);
                    }

                    // Don't fall through to Supabase polling if we have a valid task
                    return;

                } catch (err) {
                    console.error("API Poll error:", err);
                    apiFailureCountRef.current += 1;
                    setStatus(`Reconnecting to status service (${apiFailureCountRef.current})...`);
                    if (apiFailureCountRef.current >= 10) {
                        setError('Lost connection while tracking generation status. Please refresh and check the article status.');
                        localStorage.removeItem(getContentStudioGenerationStorageKey(articleId));
                        clearInterval(pollInterval);
                    }
                    return;
                }
            }

            // 2. Fallback: Supabase Polling (Legacy / Resume)
            try {
                const { data, error: dbErr } = await supabase
                    .from('Titles')
                    .select('status, htmlArticle')
                    .eq('id', articleId)
                    .single();

                if (dbErr) throw dbErr;

                const statusData = data as any;
                const currentStatus = statusData?.status || 'Processing';

                // Only update status from DB if we don't have a live task running (or API failed)
                if (!activeTaskId) {
                    setStatus(currentStatus);
                }

                // If DB shows pending controversies (e.g. from an offline run or page refresh)
                if (currentStatus === 'Pending Controversies') {
                    // Fetch controversies from Titles.idea_metadata.controversy_options
                    const { data: updatedTitle } = await supabase
                        .from('Titles')
                        .select('idea_metadata')
                        .eq('id', articleId)
                        .single();
                    
                    const meta = updatedTitle?.idea_metadata || {};
                    if (meta.controversy_options) {
                        setControversies(meta.controversy_options || []);
                        const initialSelections: Record<string, { selected: boolean; selectedTakeId: string }> = {};
                        (meta.controversy_options || []).forEach((c: any, index: number) => {
                            initialSelections[c.id] = {
                                selected: index < 3,
                                selectedTakeId: c.takes[0]?.id || ''
                            };
                        });
                        setSelectedControversies(initialSelections);
                        setOriginalResearchData(meta.research_data || meta);
                        setModalView('controversies');
                        clearInterval(pollInterval);
                        return;
                    }
                }

                if (['Created', 'Generated', 'Editing', 'WP Published', 'Scheduled'].includes(currentStatus)) {
                    if (!statusData?.htmlArticle || statusData.htmlArticle.length < 50) {
                        setError('Generation completed but returned empty content. Please try again.');
                        localStorage.removeItem(getContentStudioGenerationStorageKey(articleId));
                        clearInterval(pollInterval);
                        return;
                    }

                    setProgress(100);
                    localStorage.removeItem(getContentStudioGenerationStorageKey(articleId));
                    localStorage.removeItem(`gen_progress_${articleId}`);

                    if (!hasUpdatedRef.current && user?.id) {
                        hasUpdatedRef.current = true;
                        try {
                            await updateArticleAfterGeneration(articleId);
                        } catch (updateErr) {
                            console.error("Failed to update ToC:", updateErr);
                        }
                    }

                    setTimeout(() => { onComplete(articleId); }, 1500);
                    clearInterval(pollInterval);
                } else if (currentStatus.includes('Error') || currentStatus.includes('Failed')) {
                    setError('Generation failed (DB status).');
                    localStorage.removeItem(getContentStudioGenerationStorageKey(articleId));
                    clearInterval(pollInterval);
                } else if (!activeTaskId) {
                    // Only simulate progress if we are strictly using DB polling
                    if (currentStatus === 'New') {
                        setProgress(10);
                    } else if (currentStatus === 'Generating') {
                        setProgress(prev => {
                            const newProgress = prev < 30 ? 30 : Math.min(prev + 0.2, 90);
                            localStorage.setItem(`gen_progress_${articleId}`, newProgress.toString());
                            return newProgress;
                        });
                    }
                }

            } catch (dbErrVal) {
                console.error("Error polling status:", dbErrVal);
            }
        };

        // Initial check
        checkStatus();

        // Poll every 1 second for smoother UI with real data
        pollInterval = setInterval(checkStatus, 1000);

        return () => clearInterval(pollInterval);
    }, [isOpen, articleId, activeTaskId, user]);

    if (!isOpen) return null;

    const selectedCount = Object.values(selectedControversies).filter(v => v.selected).length;
    const isValidSelection = selectedCount >= 1 && selectedCount <= 3;

    const handleToggleControversy = (id: string) => {
        setSelectedControversies(prev => {
            const current = prev[id] || { selected: false, selectedTakeId: '' };
            return {
                ...prev,
                [id]: {
                    ...current,
                    selected: !current.selected
                }
            };
        });
    };

    const handleSelectTake = (controversyId: string, takeId: string) => {
        setSelectedControversies(prev => {
            const current = prev[controversyId] || { selected: false, selectedTakeId: '' };
            return {
                ...prev,
                [controversyId]: {
                    ...current,
                    selectedTakeId: takeId
                }
            };
        });
    };

    const handleContinue = async () => {
        if (!isValidSelection) return;
        setIsResuming(true);
        setError(null);
        try {
            // Build the list of selected controversies
            const selectedList = controversies
                .filter(c => selectedControversies[c.id]?.selected)
                .map(c => {
                    const sel = selectedControversies[c.id];
                    const t = c.takes.find((x: any) => x.id === sel.selectedTakeId);
                    return {
                        id: c.id,
                        title: c.title,
                        summary: c.summary,
                        selected_take_id: sel.selectedTakeId,
                        selected_take_text: t?.text || '',
                        takes: c.takes
                    };
                });

            // Call Backend to resume
            const { data: { session } } = await supabase.auth.getSession();
            const token = session?.access_token;
            if (!token) throw new Error("No session token found");

            // Build request payload using original research data
            const payload = {
                ...(originalResearchData || {}),
                article_id: articleId,
                selected_controversies: selectedList,
                identify_controversies: false, // Don't prompt for controversies again!
            };

            const response = await axios.post(`${import.meta.env.VITE_API_URL || 'http://localhost:5001'}/api/v1/research`, payload, {
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'X-API-Key': 'development'
                }
            });

            if (response.data && response.data.research_id) {
                const nextTaskId = String(response.data.research_id);
                // Reset state to polling
                setModalView('progress');
                setProgress(50);
                setStatus('Structuring article outline with selected takes...');
                setActiveTaskId(nextTaskId);
                localStorage.setItem(
                    getContentStudioGenerationStorageKey(articleId),
                    JSON.stringify({
                        articleId,
                        taskId: nextTaskId,
                        startedAt: new Date().toISOString(),
                    }),
                );
            } else {
                throw new Error("Resumed task but no task ID was returned by API.");
            }
        } catch (err: any) {
            console.error("Failed to resume generation:", err);
            setError(err.response?.data?.message || err.message || "Failed to resume generation");
        } finally {
            setIsResuming(false);
        }
    };

    if (modalView === 'controversies') {
        return (
            <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4 animate-in fade-in duration-200">
                <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-2xl max-w-2xl w-full p-6 border border-gray-100 dark:border-gray-700 flex flex-col max-h-[85vh]">
                    <div className="flex items-center justify-between mb-4">
                        <h3 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center gap-2">
                            <Sparkles className="w-5 h-5 text-yellow-500" />
                            Select Controversies & Takes
                        </h3>
                        <button
                            onClick={onClose}
                            className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 text-2xl font-semibold focus:outline-none"
                        >
                            &times;
                        </button>
                    </div>

                    <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
                        Choose 1 to 3 controversies to address in this article. For each selected topic, pick the take (perspective) you want the article to support and defend.
                    </p>

                    {error && (
                        <div className="bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 p-3 rounded-xl text-xs mb-4">
                            {error}
                        </div>
                    )}

                    <div className="flex-1 overflow-y-auto space-y-4 pr-1">
                        {controversies.map((c: any) => {
                            const isChecked = !!selectedControversies[c.id]?.selected;
                            const currentTakeId = selectedControversies[c.id]?.selectedTakeId || '';
                            
                            return (
                                <div key={c.id} className={`p-4 rounded-xl border transition ${
                                    isChecked 
                                        ? 'border-indigo-500 bg-indigo-500/5 dark:bg-indigo-500/10' 
                                        : 'border-gray-200 dark:border-gray-700 bg-gray-50/50 dark:bg-gray-800/30'
                                }`}>
                                    <div className="flex items-start gap-3">
                                        <input
                                            type="checkbox"
                                            className="mt-1 h-4.5 w-4.5 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500 cursor-pointer"
                                            checked={isChecked}
                                            onChange={() => handleToggleControversy(c.id)}
                                        />
                                        <div className="flex-1">
                                            <h4 className="text-sm font-semibold text-gray-900 dark:text-white">
                                                {c.title}
                                            </h4>
                                            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                                                {c.summary}
                                            </p>

                                            {isChecked && (
                                                <div className="mt-3 pl-2.5 space-y-2 border-l-2 border-indigo-500/30 ml-0.5">
                                                    <p className="text-[10px] uppercase tracking-wider font-semibold text-indigo-600 dark:text-indigo-400 mb-1">
                                                        Choose Writer's Take (Stance to Defend):
                                                    </p>
                                                    {c.takes.map((t: any) => (
                                                        <label key={t.id} className="flex items-start gap-2.5 cursor-pointer hover:bg-black/5 dark:hover:bg-white/5 p-1.5 rounded transition">
                                                            <input
                                                                type="radio"
                                                                name={`take_${c.id}`}
                                                                className="mt-0.5 h-3.5 w-3.5 text-indigo-600 border-gray-300 focus:ring-indigo-500 cursor-pointer"
                                                                checked={currentTakeId === t.id}
                                                                onChange={() => handleSelectTake(c.id, t.id)}
                                                            />
                                                            <span className="text-xs text-gray-700 dark:text-gray-300">
                                                                {t.text}
                                                            </span>
                                                        </label>
                                                    ))}
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </div>
                            );
                        })}
                    </div>

                    <div className="flex items-center justify-between border-t border-gray-100 dark:border-gray-700 pt-4 mt-4">
                        <span className="text-xs font-semibold text-gray-500 dark:text-gray-400">
                            {selectedCount === 0 
                                ? 'Select at least 1 controversy' 
                                : selectedCount > 3 
                                    ? 'Select at most 3 controversies' 
                                    : `${selectedCount} selected (valid)`}
                        </span>
                        <div className="flex gap-3">
                            <button
                                onClick={onClose}
                                className="px-4 py-2 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 rounded-xl text-sm font-medium transition-colors"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={handleContinue}
                                disabled={!isValidSelection || isResuming}
                                className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl text-sm font-semibold transition disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                {isResuming ? (
                                    <>
                                        <Loader2 className="w-4 h-4 animate-spin" />
                                        Starting Stage 5...
                                    </>
                                ) : (
                                    <>
                                        <Wand2 className="w-4 h-4" />
                                        Generate Article Outline & Content
                                    </>
                                )}
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4 animate-in fade-in duration-200">
            <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-2xl max-w-md w-full p-6 border border-gray-100 dark:border-gray-700">
                <div className="flex items-center justify-between mb-6">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Generating Article</h3>
                </div>

                {error ? (
                    <div className="text-center py-4 space-y-3">
                        <div className="mx-auto w-12 h-12 bg-red-100 dark:bg-red-900/30 rounded-full flex items-center justify-center">
                            <AlertCircle className="w-6 h-6 text-red-600 dark:text-red-400" />
                        </div>
                        <p className="text-red-600 dark:text-red-400 font-medium">{error}</p>
                        <button
                            onClick={onClose}
                            className="px-4 py-2 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 rounded-xl text-sm font-medium transition-colors"
                        >
                            Close
                        </button>
                    </div>
                ) : (
                    <div className="space-y-6">
                        {/* Progress Bar */}
                        <div className="relative pt-1">
                            <div className="flex mb-2 items-center justify-between">
                                <span className="text-xs font-semibold inline-block py-1 px-2 uppercase rounded-full text-indigo-600 bg-indigo-200 dark:bg-indigo-900/30 dark:text-indigo-400">
                                    {status}
                                </span>
                                <div className="text-right">
                                    <span className="text-xs font-semibold inline-block text-indigo-600 dark:text-indigo-400">
                                        {Math.round(progress)}%
                                    </span>
                                </div>
                            </div>
                            <div className="overflow-hidden h-2 mb-4 text-xs flex rounded bg-indigo-200 dark:bg-indigo-900/30">
                                <div
                                    style={{ width: `${progress}%` }}
                                    className="shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center bg-indigo-500 transition-all duration-500 ease-out"
                                ></div>
                            </div>
                        </div>

                        {/* Status Message */}
                        <div className="text-center px-4 min-h-[3rem] flex items-center justify-center">
                            <p className="text-sm font-medium text-gray-700 dark:text-gray-300 animate-pulse">
                                {status}
                            </p>
                        </div>

                        {/* Visual Steps (Optional/Decorative) */}
                        <div className="flex justify-between px-2 text-gray-400 text-xs">
                            <div className={`flex flex-col items-center gap-1 ${progress >= 10 ? 'text-indigo-600 dark:text-indigo-400' : ''}`}>
                                <div className={`w-2 h-2 rounded-full ${progress >= 10 ? 'bg-indigo-600' : 'bg-gray-300 dark:bg-gray-700'}`} />
                                <span>Start</span>
                            </div>
                            <div className={`flex flex-col items-center gap-1 ${progress >= 40 ? 'text-indigo-600 dark:text-indigo-400' : ''}`}>
                                <div className={`w-2 h-2 rounded-full ${progress >= 40 ? 'bg-indigo-600' : 'bg-gray-300 dark:bg-gray-700'}`} />
                                <span>Research</span>
                            </div>
                            <div className={`flex flex-col items-center gap-1 ${progress >= 70 ? 'text-indigo-600 dark:text-indigo-400' : ''}`}>
                                <div className={`w-2 h-2 rounded-full ${progress >= 70 ? 'bg-indigo-600' : 'bg-gray-300 dark:bg-gray-700'}`} />
                                <span>Write</span>
                            </div>
                            <div className={`flex flex-col items-center gap-1 ${progress >= 100 ? 'text-green-600 dark:text-green-400' : ''}`}>
                                <div className={`w-2 h-2 rounded-full ${progress >= 100 ? 'bg-green-600' : 'bg-gray-300 dark:bg-gray-700'}`} />
                                <span>Done</span>
                            </div>
                        </div>

                        <div className="flex flex-col items-center gap-3">
                            <Loader2 className="w-8 h-8 animate-spin text-indigo-200 dark:text-indigo-900" />
                            <button
                                onClick={onClose}
                                className="mt-2 px-3 py-1.5 bg-gray-100 hover:bg-gray-200 dark:bg-gray-700 dark:hover:bg-gray-600 rounded-xl text-xs font-semibold text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 transition-colors"
                            >
                                Force Close / Reset
                            </button>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};
