import { supabase } from '../lib/supabase';
import type { Collection, Document, Title, ManualAction, RagQueryResponse } from '../types/knowledge';

interface KnowledgeServiceDeps {
    userId: string;
    ragUrl?: string; // Optional, can be fetched from settings if not provided
}

const DEFAULT_RAG_URL = import.meta.env.VITE_RAG_API_URL || 'https://rag.buildomain.com';

function getRagBaseUrl(ragUrl?: string): string {
    return (ragUrl || DEFAULT_RAG_URL).replace(/\/+$/, '');
}

export const getKnowledgeService = ({ userId, ragUrl }: KnowledgeServiceDeps) => {
    const getAvailableLlmModels = async (): Promise<string[]> => {
        const queryAttempts: Array<{
            select: string;
            activeOnly: boolean;
        }> = [
            {
                select: 'model_name, is_active, is_default',
                activeOnly: true,
            },
            {
                select: 'model_name, is_active, is_default',
                activeOnly: false,
            },
            {
                select: 'model_name, is_default',
                activeOnly: false,
            },
            {
                select: 'model_name',
                activeOnly: false,
            },
        ];

        for (const attempt of queryAttempts) {
            let query = supabase.from('llm_providers').select(attempt.select);
            if (attempt.activeOnly) {
                query = query.eq('is_active', true);
            }

            const { data, error } = await query;
            if (error) {
                continue;
            }

            const models = (Array.isArray(data) ? data : [])
                .map((row: any) => String(row?.model_name || '').trim())
                .filter(Boolean);

            if (models.length > 0) {
                return Array.from(new Set(models));
            }
        }

        return [];
    };

    // --- Collections ---

    const getCollections = async (): Promise<Collection[]> => {
        const { data, error } = await supabase
            .from('lindex_collections')
            .select('id, name, user_id, created_at')
            .eq('user_id', userId);

        if (error) throw error;
        return data || [];
    };

    const createCollection = async (name: string): Promise<Collection> => {
        const { data, error } = await supabase
            .from('lindex_collections')
            .insert([{ name, user_id: userId }])
            .select()
            .single();

        if (error) throw error;
        return data;
    };

    const deleteCollection = async (
        collectionName: string
    ): Promise<{
        status: string;
        collection_name: string;
        collection_id?: number | string;
        collection_table?: string;
        deleted_documents_count?: number;
        documents_found_in_collection?: number;
        vector_collection_deleted?: boolean;
        engine_cache_cleared?: boolean;
    }> => {
        const baseUrl = getRagBaseUrl(ragUrl);
        const response = await fetch(`${baseUrl}/collections/delete`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
            },
            body: JSON.stringify({
                collection_name: collectionName.trim(),
            }),
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Collection delete failed: ${errorText || response.statusText}`);
        }

        return await response.json();
    };

    // --- Documents ---

    const getDocuments = async (collectionId: string): Promise<Document[]> => {
        const { data, error } = await supabase
            .from('lindex_documents')
            .select('*')
            .eq('user_id', userId)
            .eq('collectionId', collectionId)
            // .order('source_type', { ascending: true }); // Ordering might need index
            .order('created_at', { ascending: false });

        if (error) throw error;
        return data || [];
    };

    const deleteDocuments = async (
        collectionName: string,
        documentIds: Array<string | number>
    ): Promise<{
        deleted_documents?: Array<string | number>;
        missing_document_ids?: Array<string | number>;
        failed_documents?: Array<string | number>;
        total_vector_chunks_deleted?: number;
    }> => {
        const normalizedIds = documentIds.map((id) => String(id));
        if (normalizedIds.length === 0) {
            return {
                deleted_documents: [],
                missing_document_ids: [],
                failed_documents: [],
                total_vector_chunks_deleted: 0,
            };
        }

        const baseUrl = getRagBaseUrl(ragUrl);
        const payload = normalizedIds.length === 1
            ? { document_id: normalizedIds[0], user_id: userId }
            : { document_ids: normalizedIds, user_id: userId };

        const response = await fetch(
            `${baseUrl}/collections/${encodeURIComponent(collectionName)}/documents/delete`,
            {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json',
                },
                body: JSON.stringify(payload),
            }
        );

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Document delete failed: ${errorText || response.statusText}`);
        }

        return await response.json();
    };

    const createManualDocument = async (
        title: string,
        content: string,
        collectionId: string,
        collectionName: string
    ): Promise<Document> => {
        const { data, error } = await supabase
            .from('lindex_documents')
            .insert([{
                title,
                parsedText: content,
                user_id: userId,
                collectionId,
                source_type: 'manual',
                in_vector_store: false,
                processing_status: 'pending',
            }])
            .select()
            .single();

        if (error) throw error;

        const baseUrl = getRagBaseUrl(ragUrl);
        const docId = String(data.id);
        const response = await fetch(`${baseUrl}/upload_text`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
            },
            body: JSON.stringify({
                docid: docId,
                collection_name: collectionName,
                text: content,
                source_type: 'text',
                metadata: {
                    user_id: userId,
                    title,
                },
            }),
        });

        if (!response.ok) {
            const errorText = await response.text();
            await supabase
                .from('lindex_documents')
                .delete()
                .eq('id', data.id)
                .eq('user_id', userId);
            throw new Error(`Manual text upload failed: ${errorText || response.statusText}`);
        }

        return data;
    };

    const uploadFile = async (file: File, title: string, collectionId: string, collectionName: string): Promise<{ docId: string, startResult: any }> => {
        // 1. Create Data record
        const { data: docData, error: docError } = await supabase
            .from('lindex_documents')
            .insert([{
                title,
                source_type: 'file', // or specific extension
                in_vector_store: false,
                user_id: userId,
                collectionId
            }])
            .select('id')
            .single();

        if (docError) throw docError;
        const docId = docData.id;

        // 2. Upload to Backend
        const formData = new FormData();
        formData.append('file', file);
        formData.append('docid', docId);
        formData.append('collection_name', collectionName);

        const baseUrl = getRagBaseUrl(ragUrl);

        const response = await fetch(`${baseUrl}/upload`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Upload failed: ${response.statusText}`);
        }

        const result = await response.json();
        return { docId, startResult: result };
    };

    const getDocumentStatus = async (docId: string): Promise<Document> => {
        const { data, error } = await supabase
            .from('lindex_documents')
            .select('*')
            .eq('id', docId)
            .single();

        if (error) throw error;
        return data;
    };


    // --- Gap Analysis ---

    const getNewTitles = async (): Promise<Title[]> => {
        const { data, error } = await supabase
            .from('Titles')
            .select('*')
            .eq('user_id', userId)
            .in('status', ['New', 'NEW'])
            .order('id', { ascending: false }); // Ensure stable ordering

        if (error) {
            console.error("Error fetching titles:", error);
            throw error;
        }
        return data || [];
    };

    // "Start" behavior based on Reference 3/4: Bulk Enhance Knowledge (Gap Filling)
    const fillKnowledgeGaps = async (titles: Title[], collectionName: string): Promise<any> => {
        const baseUrl = getRagBaseUrl(ragUrl);
        const titleIds = titles.map(t => t.id);

        // Extract content outlines map
        const contentOutlines = titles.reduce((acc, t) => {
            if (t.content_outline) acc[t.id] = t.content_outline;
            return acc;
        }, {} as Record<string, string>);

        const response = await fetch(`${baseUrl}/bulk_enhance_knowledge`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({
                title_ids: titleIds,
                content_outlines: contentOutlines, // Send outlines to backend
                user_id: userId,
                collection_name: collectionName,
                merge_with_existing: true
            })
        });

        if (!response.ok) throw new Error(`Gap filling failed: ${response.statusText}`);
        return await response.json();
    };

    // "Enhance+" behavior based on Reference 5: Additional Enhancement
    const enhanceKnowledge = async (titles: Title[], collectionName: string): Promise<any> => {
        const baseUrl = getRagBaseUrl(ragUrl);
        const titleIds = titles.map(t => t.id);

        // Extract content outlines map
        const contentOutlines = titles.reduce((acc, t) => {
            if (t.content_outline) acc[t.id] = t.content_outline;
            return acc;
        }, {} as Record<string, string>);

        const response = await fetch(`${baseUrl}/enhance_additional_knowledge_bulk`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({
                title_ids: titleIds,
                content_outlines: contentOutlines, // Send outlines to backend
                user_id: userId,
                collection_name: collectionName,
                research_depth: 'standard', // Default
                source_types: ['all'],     // Default
                exclude_existing_urls: true // Default
            })
        });

        if (!response.ok) throw new Error(`Enhancement failed: ${response.statusText}`);
        return await response.json();
    };

    // Deep Research (Agentic / Tavily)
    // Calls the Content Generator Backend, not the RAG service directly
    const fillKnowledgeGapsDeep = async (titles: Title[], collectionName: string): Promise<any> => {
        // Use the VITE_API_URL if available via env, otherwise fallback. 
        // In knowledgeService we only have ragUrl dependency. 
        // We'll assume the main backend is serving this.
        const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
        const titleIds = titles.map(t => t.id);

        const response = await fetch(`${apiUrl}/api/research/deep-gap-fill`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({
                title_ids: titleIds,
                user_id: userId,
                collection_name: collectionName
            })
        });

        if (!response.ok) throw new Error(`Deep Research failed: ${response.statusText}`);
        return await response.json();
    };

    const queryCollection = async (params: {
        endpoint: '/query_simple' | '/query_hybrid_enhanced' | '/query_agentic_iterative' | '/query_truly_agentic' | '/query_agentic_fixed';
        payload: Record<string, unknown>;
    }): Promise<RagQueryResponse> => {
        const baseUrl = getRagBaseUrl(ragUrl);
        const response = await fetch(`${baseUrl}${params.endpoint}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
            },
            body: JSON.stringify(params.payload),
        });

        let result: RagQueryResponse | null = null;
        try {
            result = await response.json();
        } catch {
            result = null;
        }

        if (!response.ok) {
            const message = result?.error || `${response.status} ${response.statusText}`.trim();
            throw new Error(`Collection query failed: ${message}`);
        }

        if (!result) {
            throw new Error('Collection query failed: empty response from RAG service');
        }

        return result;
    };


    // --- Manual Actions ---

    const getManualActions = async (): Promise<ManualAction[]> => {
        const { data, error } = await supabase
            .from('manual_action_suggestions')
            .select('*')
            .eq('user_id', userId)
            .order('created_at', { ascending: false });

        if (error) throw error;
        return data || [];
    };

    const updateManualActionStatus = async (id: string | number, status: string): Promise<void> => {
        const lowerCaseStatus = status.toLowerCase();
        const updates: any = { status: lowerCaseStatus };

        if (lowerCaseStatus === 'completed') {
            updates.completed_at = new Date().toISOString();
        } else {
            updates.completed_at = null;
        }

        const { error } = await supabase
            .from('manual_action_suggestions')
            .update(updates)
            .eq('id', id);

        if (error) throw error;
    };

    const deleteManualAction = async (id: string | number): Promise<void> => {
        const { error } = await supabase
            .from('manual_action_suggestions')
            .delete()
            .eq('id', id);

        if (error) throw error;
    };

    return {
        getCollections,
        createCollection,
        deleteCollection,
        getDocuments,
        deleteDocuments,
        createManualDocument,
        uploadFile,
        getDocumentStatus,
        getNewTitles,
        fillKnowledgeGaps,
        enhanceKnowledge,
        fillKnowledgeGapsDeep,
        getAvailableLlmModels,
        queryCollection,
        getManualActions,
        updateManualActionStatus,
        deleteManualAction
    };
};
