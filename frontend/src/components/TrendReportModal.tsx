import React, { useEffect, useState } from 'react';
import { TrendingUp, X, ExternalLink, ArrowUpRight, FileText, Image as ImageIcon, MessageCircle } from 'lucide-react';
import { apiClient } from '../api-client';
import { Button } from './ui/button';
import { Card, CardContent } from './ui/card';

interface TrendReport {
    generated_at: string;
    report_content: {
        topics: Array<{
            title: string;
            rationale: string;
            suggested_angle: string;
        }>;
    };
    raw_data: {
        keywords: Array<{
            keyword: string;
            growth_formatted?: string;
            growth_pct?: number;
            search_volume?: number;
            competition?: string;
        }>;
        news: Array<{
            title: string;
            url: string;
            source: string;
        }>;
        pinterest: Array<{
            title: string;
            description: string;
            image_url?: string;
            pin_url?: string;
            keyword: string;
        }>;
    };
    pain_points?: Array<{
        source: string;
        question: string;
        hook: string;
    }>;
}

interface TrendReportModalProps {
    siteId: string;
    siteDomain?: string;
    isOpen: boolean;
    onClose: () => void;
}

export const TrendReportModal: React.FC<TrendReportModalProps> = ({ siteId, siteDomain, isOpen, onClose }) => {
    const [loading, setLoading] = useState(true);
    const [report, setReport] = useState<TrendReport | null>(null);
    const [error, setError] = useState<string | null>(null);
    // const [pollCount, setPollCount] = useState(0);

    useEffect(() => {
        if (isOpen && siteId) {
            generateReport();
        } else {
            setReport(null);
            setError(null);
            setLoading(false);
        }
    }, [isOpen, siteId]);

    const generateReport = async () => {
        setLoading(true);
        setError(null);
        setReport(null);
        // setPollCount(0);

        try {
            const startRes = await apiClient.post<any>(`/trends/${siteId}`);
            if (startRes.task_id) {
                pollTask(startRes.task_id);
            } else {
                setError("Failed to start analysis task.");
                setLoading(false);
            }
        } catch (e: any) {
            console.error("Trend Gen Error:", e);
            setError(e.response?.data?.error || e.message || "Failed to start trend analysis");
            setLoading(false);
        }
    };

    const pollTask = async (taskId: string) => {
        console.log("Polling task:", taskId);
        const interval = setInterval(async () => {
            try {
                const statusRes = await apiClient.get<any>(`/trends/task/${taskId}`);
                console.log("Poll Status:", statusRes);

                if (statusRes.status === 'SUCCESS' && statusRes.result) {
                    clearInterval(interval);
                    // result.result contains the full report from engine
                    setReport(statusRes.result.result);
                    setLoading(false);
                } else if (statusRes.status === 'FAILURE') {
                    clearInterval(interval);
                    setError(statusRes.error || "Analysis failed during execution.");
                    setLoading(false);
                } else {
                    // Logic removed
                }
            } catch (e) {
                console.error("Poll Error:", e);
                // Don't stop polling immediately on network transient error
            }
        }, 2000);
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4 animate-in fade-in duration-200 overflow-y-auto">
            <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-2xl max-w-4xl w-full flex flex-col max-h-[90vh]">

                {/* Header */}
                <div className="flex items-center justify-between p-6 border-b border-gray-100 dark:border-gray-700 bg-gray-50/50 dark:bg-gray-900/50 rounded-t-2xl sticky top-0 bg-opacity-95 backdrop-blur z-10">
                    <div className="flex items-center gap-3">
                        <div className="p-2 bg-indigo-100 dark:bg-indigo-900/30 rounded-lg">
                            <TrendingUp className="w-6 h-6 text-indigo-600 dark:text-indigo-400" />
                        </div>
                        <div>
                            <h2 className="text-xl font-bold text-gray-900 dark:text-white">Trend Discovery Report</h2>
                            {siteDomain && <p className="text-sm text-gray-500">Analysis for {siteDomain}</p>}
                        </div>
                    </div>
                    <button onClick={onClose} className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-full transition-colors">
                        <X className="w-5 h-5 text-gray-500" />
                    </button>
                </div>

                {/* Content */}
                <div className="flex-1 overflow-y-auto p-6 min-h-[400px]">
                    {loading ? (
                        <div className="flex flex-col items-center justify-center h-full space-y-6 py-20">
                            <div className="relative">
                                <div className="w-20 h-20 border-4 border-indigo-100 dark:border-indigo-900/30 rounded-full"></div>
                                <div className="absolute top-0 left-0 w-20 h-20 border-4 border-indigo-600 border-t-transparent rounded-full animate-spin"></div>
                                <div className="absolute inset-0 flex items-center justify-center">
                                    <TrendingUp className="w-8 h-8 text-indigo-600 animate-pulse" />
                                </div>
                            </div>
                            <div className="text-center space-y-2">
                                <h3 className="text-lg font-medium text-gray-900 dark:text-white">Analyzing Market Trends</h3>
                                <p className="text-gray-500 max-w-xs mx-auto">
                                    Scanning Search Data, News, and Social Signals to find high-impact opportunities...
                                </p>
                            </div>
                        </div>
                    ) : error ? (
                        <div className="flex flex-col items-center justify-center h-full space-y-4 py-20">
                            <div className="w-16 h-16 bg-red-100 dark:bg-red-900/30 rounded-full flex items-center justify-center">
                                <X className="w-8 h-8 text-red-600" />
                            </div>
                            <h3 className="text-lg font-medium text-red-600">Analysis Failed</h3>
                            <p className="text-gray-500">{error}</p>
                            <Button onClick={generateReport} variant="outline">Retry Analysis</Button>
                        </div>
                    ) : report ? (
                        <div className="space-y-8 animate-in slide-in-from-bottom-4 duration-500">

                            {/* 1. Content Opportunities (Gemini) */}
                            <section>
                                <h3 className="text-lg font-bold flex items-center gap-2 mb-4 text-gray-900 dark:text-white">
                                    <FileText className="w-5 h-5 text-indigo-500" />
                                    AI Content Opportunities
                                </h3>
                                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                    {report.report_content?.topics?.map((topic, i) => (
                                        <Card key={i} className="bg-indigo-50/30 dark:bg-indigo-900/10 border-indigo-100 dark:border-indigo-800 hover:shadow-md transition-shadow">
                                            <CardContent className="p-5 space-y-3">
                                                <h4 className="font-bold text-indigo-900 dark:text-indigo-100 leading-tight">{topic.title}</h4>
                                                <div className="space-y-1">
                                                    <p className="text-xs font-semibold uppercase text-gray-500">Why it's trending</p>
                                                    <p className="text-sm text-gray-600 dark:text-gray-300">{topic.rationale}</p>
                                                </div>
                                                <div className="pt-2 border-t border-indigo-100 dark:border-indigo-800/50">
                                                    <p className="text-xs font-semibold uppercase text-gray-500 mb-1">Suggested Angle</p>
                                                    <p className="text-sm text-gray-600 dark:text-gray-300 italic">"{topic.suggested_angle}"</p>
                                                </div>
                                            </CardContent>
                                        </Card>
                                    ))}
                                </div>
                            </section>

                            {/* SOCIAL PULSE / PAIN POINTS SECTION */}
                            {report.pain_points && report.pain_points.length > 0 && (
                                <section>
                                    <h3 className="text-lg font-bold flex items-center gap-2 mb-4 text-gray-900 dark:text-white">
                                        <MessageCircle className="w-5 h-5 text-blue-500" />
                                        Social Pulse (Pain Points)
                                    </h3>
                                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                                        {report.pain_points.map((pp: any, idx: number) => (
                                            <Card key={idx} className="bg-slate-50 dark:bg-slate-900 border-none">
                                                <CardContent className="p-4 space-y-3">
                                                    <div className="flex items-center justify-between">
                                                        <span className="text-xs font-bold px-2 py-1 rounded bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-100">
                                                            {pp.source}
                                                        </span>
                                                    </div>
                                                    <p className="text-sm font-medium text-slate-800 dark:text-slate-200 line-clamp-3">
                                                        "{pp.question}"
                                                    </p>
                                                    <div className="pt-2 border-t border-slate-200 dark:border-slate-800">
                                                        <p className="text-xs text-slate-500 uppercase tracking-widest font-bold mb-1">Content Hook</p>
                                                        <p className="text-xs text-slate-600 dark:text-slate-400 italic">
                                                            {pp.hook}
                                                        </p>
                                                    </div>
                                                </CardContent>
                                            </Card>
                                        ))}
                                    </div>
                                </section>
                            )}

                            {/* 2. Data Sources Grid */}
                            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">

                                {/* Keywords */}
                                <section>
                                    <h3 className="text-lg font-bold flex items-center gap-2 mb-4 text-gray-900 dark:text-white">
                                        <ArrowUpRight className="w-5 h-5 text-green-500" />
                                        Growing Search Terms
                                    </h3>
                                    <div className="space-y-3">
                                        {report.raw_data?.keywords?.map((kw, i) => (
                                            <div key={i} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800/50 rounded-lg border border-gray-100 dark:border-gray-700">
                                                <div>
                                                    <p className="font-medium text-gray-900 dark:text-gray-100">{kw.keyword}</p>
                                                    <p className="text-xs text-gray-500">Vol: {kw.search_volume}</p>
                                                </div>
                                                <div className="text-right">
                                                    <span className="inline-flex items-center px-2 py-1 rounded-md bg-green-100 text-green-700 text-xs font-bold">
                                                        {kw.growth_formatted || 'High Growth'}
                                                    </span>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </section>

                                {/* News */}
                                <section>
                                    <h3 className="text-lg font-bold flex items-center gap-2 mb-4 text-gray-900 dark:text-white">
                                        <ExternalLink className="w-5 h-5 text-blue-500" />
                                        Recent News Headlines
                                    </h3>
                                    <div className="space-y-3">
                                        {report.raw_data?.news?.map((item, i) => (
                                            <a key={i} href={item.url} target="_blank" rel="noopener noreferrer" className="block p-3 bg-gray-50 dark:bg-gray-800/50 rounded-lg border border-gray-100 dark:border-gray-700 hover:bg-gray-100 transition-colors group">
                                                <p className="text-sm font-medium text-gray-900 dark:text-gray-100 group-hover:text-blue-600 line-clamp-2">{item.title}</p>
                                                <p className="text-xs text-gray-400 mt-1">{item.source}</p>
                                            </a>
                                        ))}
                                    </div>
                                </section>
                            </div>

                            {/* Pinterest */}
                            {report.raw_data?.pinterest && report.raw_data.pinterest.length > 0 && (
                                <section>
                                    <h3 className="text-lg font-bold flex items-center gap-2 mb-4 text-gray-900 dark:text-white">
                                        <ImageIcon className="w-5 h-5 text-red-500" />
                                        Pinterest Visual Trends
                                    </h3>
                                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                        {report.raw_data.pinterest.slice(0, 4).map((pin, i) => (
                                            <a key={i} href={pin.pin_url} target="_blank" rel="noopener noreferrer" className="block group relative aspect-[2/3] rounded-lg overflow-hidden bg-gray-100">
                                                {pin.image_url ? (
                                                    <img src={pin.image_url} alt={pin.title} className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500" />
                                                ) : (
                                                    <div className="w-full h-full flex items-center justify-center bg-gray-200">
                                                        <ImageIcon className="w-8 h-8 text-gray-400" />
                                                    </div>
                                                )}
                                                <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity p-3 flex flex-col justify-end">
                                                    <p className="text-white text-xs font-medium line-clamp-2">{pin.title}</p>
                                                </div>
                                            </a>
                                        ))}
                                    </div>
                                </section>
                            )}

                        </div>
                    ) : null}
                </div>

                <div className="p-4 border-t border-gray-100 dark:border-gray-700 flex justify-end">
                    <Button onClick={onClose} variant="secondary">Close Report</Button>
                </div>
            </div>
        </div>
    );
};
