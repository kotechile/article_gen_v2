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
import { Upload, PlusCircle, BookOpen, Search, ClipboardList, Loader2, Trash2 } from 'lucide-react';

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
    const [researchMethod, setResearchMethod] = useState<'standard' | 'deep'>('standard');

    const [loadingDocs, setLoadingDocs] = useState(false);
    const [loadingTitles, setLoadingTitles] = useState(false);
    const [loadingActions, setLoadingActions] = useState(false);
    const [deletingCollection, setDeletingCollection] = useState(false);
    const [selectedDocumentIds, setSelectedDocumentIds] = useState<Set<string>>(new Set());
    const [deletingDocumentIds, setDeletingDocumentIds] = useState<Set<string>>(new Set());

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
                setSelectedDocumentIds(new Set());
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
            await service.createManualDocument(title, content, selectedCollection.id, selectedCollection.name);
            // Refresh
            const docs = await service.getDocuments(selectedCollection.id);
            setDocuments(docs);
        } catch (e: any) {
            throw new Error(e.message || "Entry failed");
        }
    };

    const refreshDocuments = async () => {
        if (!selectedCollection) return;
        const docs = await service.getDocuments(selectedCollection.id);
        setDocuments(docs);
        setSelectedDocumentIds((current) => {
            const existingIds = new Set(docs.map((doc) => String(doc.id)));
            return new Set([...current].filter((id) => existingIds.has(id)));
        });
    };

    const handleToggleDocumentSelected = (documentId: string, checked: boolean) => {
        setSelectedDocumentIds((current) => {
            const next = new Set(current);
            if (checked) {
                next.add(documentId);
            } else {
                next.delete(documentId);
            }
            return next;
        });
    };

    const handleToggleAllDocuments = (checked: boolean) => {
        if (checked) {
            setSelectedDocumentIds(new Set(documents.map((doc) => String(doc.id))));
            return;
        }
        setSelectedDocumentIds(new Set());
    };

    const handleDeleteDocuments = async (documentIds: string[]) => {
        if (!selectedCollection || documentIds.length === 0) return;

        const warning = documentIds.length === 1
            ? 'Delete this document from the collection and RAG backend? This cannot be undone.'
            : `Delete ${documentIds.length} selected documents from the collection and RAG backend? This cannot be undone.`;

        if (!window.confirm(warning)) {
            return;
        }

        setDeletingDocumentIds((current) => new Set([...current, ...documentIds]));
        try {
            const result = await service.deleteDocuments(selectedCollection.name, documentIds);
            await refreshDocuments();
            const failed = result.failed_documents?.length || 0;
            const missing = result.missing_document_ids?.length || 0;
            if (failed > 0 || missing > 0) {
                alert(`Delete completed with issues. Failed: ${failed}, Missing: ${missing}.`);
            }
        } catch (error: any) {
            console.error('Delete documents failed', error);
            alert(error?.message || 'Failed to delete documents');
        } finally {
            setDeletingDocumentIds((current) => {
                const next = new Set(current);
                documentIds.forEach((id) => next.delete(id));
                return next;
            });
        }
    };

    const handleDeleteSingleDocument = async (documentId: string) => {
        await handleDeleteDocuments([documentId]);
    };

    const handleDeleteSelectedDocuments = async () => {
        await handleDeleteDocuments([...selectedDocumentIds]);
    };

    const handleStartGapFill = async (titlesToProcess: Title[]) => {
        if (!selectedCollection) {
            alert("Please select a collection to store the enhanced knowledge.");
            return;
        }
        try {
            if (researchMethod === 'deep') {
                await service.fillKnowledgeGapsDeep(titlesToProcess, selectedCollection.name);
                alert("Deep Research started! Agents are working in the background.");
            } else {
                await service.fillKnowledgeGaps(titlesToProcess, selectedCollection.name);
                alert("Gap filling started! Process running in background.");
            }
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

    const handleCollectionDelete = async () => {
        if (!selectedCollection) return;

        const targetName = selectedCollection.name;
        const warningMessage = [
            `Delete collection "${targetName}"?`,
            '',
            'This will:',
            '- remove the collection from lindex_collections',
            '- delete linked lindex_documents rows',
            '- delete indexed vectors reachable through the current Supabase RPC helpers',
            '',
            'This cannot be undone.',
        ].join('\n');

        if (!window.confirm(warningMessage)) {
            return;
        }

        const confirmation = window.prompt(
            `Type the collection name to confirm deletion:\n\n${targetName}`,
            ''
        );

        if (confirmation !== targetName) {
            alert('Collection name did not match. Delete cancelled.');
            return;
        }

        setDeletingCollection(true);
        try {
            await service.deleteCollection(String(selectedCollection.id), targetName);
            const updatedCollections = collections.filter((collection) => collection.id !== selectedCollection.id);
            setCollections(updatedCollections);
            setSelectedCollection(updatedCollections[0] || null);
            setDocuments([]);
        } catch (error: any) {
            console.error('Delete collection failed', error);
            alert(error?.message || 'Failed to delete collection');
        } finally {
            setDeletingCollection(false);
        }
    };



    if (!user) {
        return (
            <div className="flex h-[50vh] items-center justify-center">
                <Loader2 className="w-8 h-8 animate-spin text-primary" />
            </div>
        );
    }

    return (
            <div className="space-y-8">
            {/* Header */}
            <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
                <div>
                    <h1 className="text-3xl font-bold text-foreground">
                        Knowledge Gaps
                    </h1>
                    <p className="text-muted-foreground mt-1">
                        Identify and fill knowledge gaps to improve content quality.
                    </p>
                </div>

                <div className="w-full max-w-full space-y-3 md:w-auto md:min-w-[18rem] md:max-w-[24rem]">
                    <div className="space-y-2">
                        <span className="block text-sm font-medium text-foreground">Active Collection</span>
                        <CollectionSelector
                            collections={collections}
                            selectedCollection={selectedCollection}
                            onSelect={setSelectedCollection}
                            onCreateCollection={handleCollectionCreate}
                            disabled={deletingCollection}
                        />
                    </div>

                    <button
                        type="button"
                        onClick={handleCollectionDelete}
                        disabled={!selectedCollection || deletingCollection}
                        className="inline-flex w-full items-center justify-center gap-2 rounded-lg border border-red-500/20 bg-red-500/10 px-4 py-2 text-sm font-medium text-red-400 transition hover:bg-red-500/15 disabled:cursor-not-allowed disabled:opacity-60"
                    >
                        <Trash2 className="h-4 w-4" />
                        <span>{deletingCollection ? 'Deleting Collection…' : 'Delete Collection'}</span>
                    </button>
                </div>
            </div>

            {/* Navigation Tabs */}
            <div className="flex space-x-1 bg-muted/50 p-1 rounded-xl w-fit">
                <button
                    onClick={() => setActiveTab('gap_analysis')}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeTab === 'gap_analysis' ? 'bg-background text-primary shadow-sm' : 'text-muted-foreground hover:text-foreground'}`}
                >
                    <Search className="w-4 h-4" />
                    Identify Gaps
                </button>
                <button
                    onClick={() => setActiveTab('documents')}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeTab === 'documents' ? 'bg-background text-primary shadow-sm' : 'text-muted-foreground hover:text-foreground'}`}
                >
                    <BookOpen className="w-4 h-4" />
                    Documents
                </button>
                <button
                    onClick={() => setActiveTab('actions')}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeTab === 'actions' ? 'bg-background text-primary shadow-sm' : 'text-muted-foreground hover:text-foreground'}`}
                >
                    <ClipboardList className="w-4 h-4" />
                    Actions Needed
                </button>
            </div>

            {/* Tab Content */}
            <div className="bg-card text-card-foreground rounded-2xl shadow-sm border border-border p-6 min-h-[500px] flex flex-col">

                {activeTab === 'documents' && (
                    <div className="space-y-6">
                        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                            <h2 className="text-xl font-semibold text-foreground">
                                Collection Documents
                            </h2>
                            <div className="flex flex-wrap gap-3">
                                {selectedDocumentIds.size > 0 && (
                                    <button
                                        onClick={handleDeleteSelectedDocuments}
                                        className="flex items-center gap-2 rounded-lg border border-red-500/20 bg-red-500/10 px-4 py-2 text-sm font-medium text-red-400 transition hover:bg-red-500/15"
                                    >
                                        <Trash2 className="h-4 w-4" />
                                        Delete Selected ({selectedDocumentIds.size})
                                    </button>
                                )}
                                <button
                                    onClick={() => setShowManualModal(true)}
                                    className="flex items-center gap-2 px-4 py-2 text-primary bg-primary/10 hover:bg-primary/15 rounded-lg transition-colors text-sm font-medium"
                                >
                                    <PlusCircle className="w-4 h-4" />
                                    Manual Entry
                                </button>
                                <button
                                    onClick={() => setShowUploadModal(true)}
                                    className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground hover:bg-primary/90 rounded-lg transition-colors text-sm font-medium shadow-md hover:shadow-lg"
                                >
                                    <Upload className="w-4 h-4" />
                                    Upload File
                                </button>
                            </div>
                        </div>
                        <DocumentsTable
                            documents={documents}
                            isLoading={loadingDocs}
                            selectedIds={selectedDocumentIds}
                            onToggleSelected={handleToggleDocumentSelected}
                            onToggleAll={handleToggleAllDocuments}
                            onDeleteDocument={handleDeleteSingleDocument}
                            deletingIds={deletingDocumentIds}
                        />
                    </div>
                )}

                {activeTab === 'gap_analysis' && (
                    <div className="h-full flex-1">
                        <GapAnalysisTable
                            titles={titles}
                            isLoading={loadingTitles}
                            onStartGapFill={handleStartGapFill}
                            onEnhancePlus={handleEnhancePlus}
                            researchMethod={researchMethod}
                            onResearchMethodChange={setResearchMethod}
                        />
                    </div>
                )}

                {activeTab === 'actions' && (
                    <div className="space-y-6">
                        <div className="flex justify-between items-center">
                            <h2 className="text-xl font-semibold text-foreground">
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
