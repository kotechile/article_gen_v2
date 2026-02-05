import React, { useEffect, useState, useRef } from 'react';
import axios from 'axios';
import { supabase } from '../lib/supabase';
import { Loader2, AlertCircle, X } from 'lucide-react';
import { updateArticleAfterGeneration } from '../lib/contentParser';
import { useAuth } from '../context/auth-context';

interface GenerationModalProps {
    articleId: string;
    taskId?: string | null;
    isOpen: boolean;
    onClose: () => void;
    onComplete: () => void;
}

export const GenerationModal: React.FC<GenerationModalProps> = ({ articleId, taskId, isOpen, onClose, onComplete }) => {
    const { user } = useAuth();
    const [status, setStatus] = useState<string>('Initializing...');
    const [progress, setProgress] = useState(0);
    const [error, setError] = useState<string | null>(null);
    const hasUpdatedRef = useRef(false);


    useEffect(() => {
        if (!isOpen) return;

        // Restore progress from storage if available (only if not starting fresh task)
        const storedProgress = localStorage.getItem(`gen_progress_${articleId}`);
        if (storedProgress && !taskId) {
            setProgress(parseFloat(storedProgress));
        }

        let pollInterval: any;

        const checkStatus = async () => {
            // 1. Prefer API Polling if taskId is available
            if (taskId) {
                try {
                    const { data } = await axios.get(`http://localhost:5001/api/v1/research/${taskId}`, {
                        headers: { 'X-API-Key': 'development' }
                    });

                    // Handle different states
                    if (data.status === 'SUCCESS') {
                        setProgress(100);
                        setStatus('Completed');

                        // Trigger final updates
                        if (!hasUpdatedRef.current && user?.id) {
                            hasUpdatedRef.current = true;
                            try {
                                await updateArticleAfterGeneration(articleId);
                            } catch (updateErr) {
                                console.error("Failed to update ToC:", updateErr);
                            }
                        }

                        setTimeout(onComplete, 1500);
                        clearInterval(pollInterval);
                        return;
                    }
                    else if (['FAILURE', 'REVOKED'].includes(data.status)) {
                        setError(typeof data.info === 'string' ? data.info : (data.info?.message || 'Generation failed'));
                        clearInterval(pollInterval);
                        return;
                    }
                    else if (data.status === 'PROGRESS' && data.info) {
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
                    // Fallback to Supabase below if API fails?
                }
            }

            // 2. Fallback: Supabase Polling (Legacy / Resume)
            try {
                const { data, error } = await supabase
                    .from('Titles')
                    .select('status, htmlArticle')
                    .eq('id', articleId)
                    .single();

                if (error) throw error;

                const statusData = data as any;
                const currentStatus = statusData?.status || 'Processing';

                // Only update status from DB if we don't have a live task running (or API failed)
                if (!taskId) {
                    setStatus(currentStatus);
                }

                // ... (rest of existing completion logic) ...
                if (currentStatus === 'Created' || currentStatus === 'Generated') {
                    // ... existing validation/completion logic ...
                    if (!statusData?.htmlArticle || statusData.htmlArticle.length < 50) {
                        setError('Generation completed but returned empty content. Please try again.');
                        clearInterval(pollInterval);
                        return;
                    }

                    setProgress(100);

                    if (!hasUpdatedRef.current && user?.id) {
                        hasUpdatedRef.current = true;
                        await updateArticleAfterGeneration(articleId);
                    }

                    setTimeout(() => { onComplete(); }, 1500);
                    clearInterval(pollInterval);
                } else if (currentStatus.includes('Error') || currentStatus.includes('Failed')) {
                    setError('Generation failed (DB status).');
                    clearInterval(pollInterval);
                } else if (!taskId) {
                    // Only simulate progress if we are strictly using DB polling
                    // ... existing simulation logic ...
                    if (currentStatus === 'New') {
                        setProgress(10);
                    } else if (currentStatus === 'Generating') {
                        // ... simulation ...
                        setProgress(prev => {
                            const newProgress = prev < 30 ? 30 : Math.min(prev + 0.2, 90);
                            localStorage.setItem(`gen_progress_${articleId}`, newProgress.toString());
                            return newProgress;
                        });
                    }
                    // ... other mappings ...
                }

            } catch (err) {
                console.error("Error polling status:", err);
            }
        };

        // Initial check
        checkStatus();

        // Poll every 1 second for smoother UI with real data
        pollInterval = setInterval(checkStatus, 1000);

        return () => clearInterval(pollInterval);
    }, [isOpen, articleId, taskId, user]);

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4 animate-in fade-in duration-200">
            <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-2xl max-w-md w-full p-6 border border-gray-100 dark:border-gray-700">
                <div className="flex items-center justify-between mb-6">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Generating Article</h3>
                    {!error && (
                        <button onClick={onClose} className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300">
                            <X className="w-5 h-5" />
                        </button>
                    )}
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

                        <div className="flex justify-center">
                            <Loader2 className="w-8 h-8 animate-spin text-indigo-200 dark:text-indigo-900" />
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};
