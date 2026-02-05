import React, { useState, useEffect, useMemo } from 'react';
import { useAuth } from '../context/auth-context';
import { getKnowledgeService } from '../services/knowledgeService';
import { CollectionSelector } from '../components/knowledge/CollectionSelector';
import { DocumentsTable } from '../components/knowledge/DocumentsTable';
import { UploadModal } from '../components/knowledge/UploadModal';
import { ManualEntryModal } from '../components/knowledge/ManualEntryModal';
import { GapAnalysisTable } from '../components/knowledge/GapAnalysisTable';
import { ActionSuggestionsList } from '../components/knowledge/ActionSuggestionsList';
import type { Collection, Document, Title, ManualAction } from '../types/knowledge';
import { Upload, PlusCircle, BookOpen, Search, ClipboardList, Loader2 } from 'lucide-react';

export const KnowledgeGaps: React.FC = () => {
    const { user } = useAuth();
    const service = useMemo(() => getKnowledgeService({ userId: user?.id || '' }), [user?.id]);

    // State
    const [activeTab, setActiveTab] = useState<'documents' | 'gap_analysis' | 'actions'>('gap_analysis');

    const [collections, setCollections] = useState<Collection[]>([]);
    const [selectedCollection, setSelectedCollection] = useState<Collection | null>(null);

    const [documents, setDocuments] = useState<Document[]>([]);
    const [titles, setTitles] = useState<Title[]>([]);
    const [actions, setActions] = useState<ManualAction[]>([]);

    const [loadingDocs, setLoadingDocs] = useState(false);
    const [loadingTitles, setLoadingTitles] = useState(false);
    const [loadingActions, setLoadingActions] = useState(false);

    const [showUploadModal, setShowUploadModal] = useState(false);
    const [showManualModal, setShowManualModal] = useState(false);

    // Initial Load - Collections
    useEffect(() => {
        if (!user) return;
        const loadCollections = async () => {
            try {
                const cols = await service.getCollections();
                setCollections(cols);
                if (cols.length > 0 && !selectedCollection) {
                    setSelectedCollection(cols[0]);
                }
            } catch (error) {
                console.error("Failed to load collections", error);
            }
        };
        loadCollections();
    }, [user, service]);

    // Load Documents when Collection changes
    useEffect(() => {
        if (!selectedCollection) return;

        const loadDocuments = async () => {
            setLoadingDocs(true);
            try {
                const docs = await service.getDocuments(selectedCollection.id);
                setDocuments(docs);
            } catch (e) {
                console.error("Error loading docs", e);
            } finally {
                setLoadingDocs(false);
            }
        };
        loadDocuments();
    }, [selectedCollection, service]);

    // Load other tabs data lazily or on mount
    useEffect(() => {
        const loadAnalysisData = async () => {
            setLoadingTitles(true);
            try {
                const t = await service.getNewTitles();
                setTitles(t);
            } finally {
                setLoadingTitles(false);
            }
        };

        const loadActionsData = async () => {
            setLoadingActions(true);
            try {
                const a = await service.getManualActions();
                setActions(a);
            } finally {
                setLoadingActions(false);
            }
        };

        if (activeTab === 'gap_analysis' && titles.length === 0) loadAnalysisData();
        if (activeTab === 'actions' && actions.length === 0) loadActionsData();
    }, [activeTab, service]);


    // Handlers
    const handleCollectionCreate = async (name: string) => {
        try {
            const newCol = await service.createCollection(name);
            setCollections([...collections, newCol]);
            setSelectedCollection(newCol);
        } catch (e) {
            console.error("Create collection failed", e);
            alert("Failed to create collection");
        }
    };

    const handleUpload = async (file: File) => {
        if (!selectedCollection) return;
        try {
            await service.uploadFile(file, file.name, selectedCollection.id, selectedCollection.name);
            // Refresh
            const docs = await service.getDocuments(selectedCollection.id);
            setDocuments(docs);
        } catch (e: any) {
            throw new Error(e.message || "Upload failed");
        }
    };

    const handleManualEntry = async (title: string, content: string) => {
        if (!selectedCollection) return;
        try {
            await service.createManualDocument(title, content, selectedCollection.id);
            // Refresh
            const docs = await service.getDocuments(selectedCollection.id);
            setDocuments(docs);
        } catch (e: any) {
            throw new Error(e.message || "Entry failed");
        }
    };

    const handleStartGapFill = async (titles: Title[]) => {
        if (!selectedCollection) {
            alert("Please select a collection to store the enhanced knowledge.");
            return;
        }
        try {
            await service.fillKnowledgeGaps(titles, selectedCollection.name);
            alert("Gap filling started! Process running in background.");
        } catch (e: any) {
            console.error(e);
            alert(`Failed: ${e.message}`);
        }
    };

    const handleEnhancePlus = async (titles: Title[]) => {
        if (!selectedCollection) {
            alert("Please select a collection.");
            return;
        }
        try {
            await service.enhanceKnowledge(titles, selectedCollection.name);
            alert("Enhance+ started! Process running in background.");
        } catch (e: any) {
            console.error(e);
            alert(`Failed: ${e.message}`);
        }
    };

    const handleActionComplete = async (id: string | number, currentStatus: string) => {
        // Determine new status
        const isCompleted = currentStatus?.toLowerCase() === 'completed';
        const newStatus = isCompleted ? 'suggested' : 'completed';

        // 1. Optimistic Update (Immediate Feedback)
        setActions(prevActions => prevActions.map(a => a.id === id ? { ...a, status: newStatus } : a));

        try {
            await service.updateManualActionStatus(id, newStatus);
        } catch (e: any) {
            console.error("Failed to update status", e);
            // 2. Revert on failure
            setActions(prevActions => prevActions.map(a => a.id === id ? { ...a, status: currentStatus } : a));
            alert(`Failed to update status: ${e.message || e.error_description || JSON.stringify(e)}`);
        }
    };

    const handleActionDelete = async (id: string | number) => {
        if (!confirm("Are you sure you want to delete this action?")) return;
        try {
            await service.deleteManualAction(id);
            setActions(actions.filter(a => a.id !== id));
        } catch (e) {
            console.error("Failed to delete action", e);
        }
    };



    if (!user) {
        return (
            <div className="flex h-[50vh] items-center justify-center">
                <Loader2 className="w-8 h-8 animate-spin text-indigo-600" />
            </div>
        );
    }

    return (
        <div className="space-y-8">
            {/* Header */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                    <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-600 to-purple-600">
                        Knowledge Gaps
                    </h1>
                    <p className="text-gray-500 dark:text-gray-400 mt-1">
                        Identify and fill knowledge gaps to improve content quality.
                    </p>
                </div>

                <div className="flex items-center gap-3">
                    <span className="text-sm font-medium text-gray-600 dark:text-gray-300">Active Collection:</span>
                    <CollectionSelector
                        collections={collections}
                        selectedCollection={selectedCollection}
                        onSelect={setSelectedCollection}
                        onCreateCollection={handleCollectionCreate}
                    />
                </div>
            </div>

            {/* Navigation Tabs */}
            <div className="flex space-x-1 bg-gray-100 dark:bg-gray-800 p-1 rounded-xl w-fit">
                <button
                    onClick={() => setActiveTab('gap_analysis')}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeTab === 'gap_analysis' ? 'bg-white dark:bg-gray-700 text-indigo-600 dark:text-indigo-400 shadow-sm' : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'}`}
                >
                    <Search className="w-4 h-4" />
                    Identify Gaps
                </button>
                <button
                    onClick={() => setActiveTab('documents')}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeTab === 'documents' ? 'bg-white dark:bg-gray-700 text-indigo-600 dark:text-indigo-400 shadow-sm' : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'}`}
                >
                    <BookOpen className="w-4 h-4" />
                    RAG Collections
                </button>
                <button
                    onClick={() => setActiveTab('actions')}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeTab === 'actions' ? 'bg-white dark:bg-gray-700 text-indigo-600 dark:text-indigo-400 shadow-sm' : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'}`}
                >
                    <ClipboardList className="w-4 h-4" />
                    Actions Needed
                </button>
            </div>

            {/* Tab Content */}
            <div className="bg-white dark:bg-gray-900 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-800 p-6 min-h-[500px] flex flex-col">

                {activeTab === 'documents' && (
                    <div className="space-y-6">
                        <div className="flex justify-between items-center">
                            <h2 className="text-xl font-semibold text-gray-800 dark:text-gray-200">
                                Collection Documents
                            </h2>
                            <div className="flex gap-3">
                                <button
                                    onClick={() => setShowManualModal(true)}
                                    className="flex items-center gap-2 px-4 py-2 text-indigo-600 bg-indigo-50 hover:bg-indigo-100 dark:bg-indigo-900/20 dark:text-indigo-400 dark:hover:bg-indigo-900/30 rounded-lg transition-colors text-sm font-medium"
                                >
                                    <PlusCircle className="w-4 h-4" />
                                    Manual Entry
                                </button>
                                <button
                                    onClick={() => setShowUploadModal(true)}
                                    className="flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white hover:bg-indigo-700 rounded-lg transition-colors text-sm font-medium shadow-md hover:shadow-lg"
                                >
                                    <Upload className="w-4 h-4" />
                                    Upload File
                                </button>
                            </div>
                        </div>
                        <DocumentsTable documents={documents} isLoading={loadingDocs} />
                    </div>
                )}

                {activeTab === 'gap_analysis' && (
                    <div className="h-full flex-1">
                        <GapAnalysisTable
                            titles={titles}
                            isLoading={loadingTitles}
                            onStartGapFill={handleStartGapFill}
                            onEnhancePlus={handleEnhancePlus}
                        />
                    </div>
                )}

                {activeTab === 'actions' && (
                    <div className="space-y-6">
                        <div className="flex justify-between items-center">
                            <h2 className="text-xl font-semibold text-gray-800 dark:text-gray-200">
                                Manually add to your collection
                            </h2>
                        </div>
                        <ActionSuggestionsList
                            actions={actions}
                            isLoading={loadingActions}
                            onComplete={handleActionComplete}
                            onDelete={handleActionDelete}
                        />
                    </div>
                )}

            </div>

            {/* Modals */}
            <UploadModal
                isOpen={showUploadModal}
                onClose={() => setShowUploadModal(false)}
                currentCollection={selectedCollection}
                onUpload={handleUpload}
            />
            <ManualEntryModal
                isOpen={showManualModal}
                onClose={() => setShowManualModal(false)}
                currentCollection={selectedCollection}
                onSubmit={handleManualEntry}
            />
        </div>
    );
};
