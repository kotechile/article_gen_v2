import { supabase } from '../lib/supabase';
import type { Collection, Document, Title, ManualAction } from '../types/knowledge';

interface KnowledgeServiceDeps {
    userId: string;
    ragUrl?: string; // Optional, can be fetched from settings if not provided
}

const DEFAULT_RAG_URL = import.meta.env.VITE_RAG_API_URL || 'https://rag.buildomain.com';

function getRagBaseUrl(ragUrl?: string): string {
    return (ragUrl || DEFAULT_RAG_URL).replace(/\/+$/, '');
}

export const getKnowledgeService = ({ userId, ragUrl }: KnowledgeServiceDeps) => {

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

    const deleteCollection = async (collectionId: string, collectionName: string): Promise<void> => {
        const normalizedCollectionName = collectionName.trim();

        let vectorDocIds: string[] = [];
        try {
            const { data, error } = await supabase.rpc('get_collection_docids', {
                target_collection: normalizedCollectionName,
            });

            if (error) {
                throw error;
            }

            vectorDocIds = (data || [])
                .map((row: { docid?: string | null }) => row?.docid?.trim())
                .filter((docid: string | undefined): docid is string => Boolean(docid));
        } catch (error) {
            console.warn(`Unable to enumerate vector docids for collection '${normalizedCollectionName}'.`, error);
        }

        for (const docId of vectorDocIds) {
            const { error } = await supabase.rpc('delete_vector_by_docid', {
                collection_name_param: normalizedCollectionName,
                doc_id_param: docId,
            });

            if (error) {
                throw error;
            }
        }

        const { error: docsError } = await supabase
            .from('lindex_documents')
            .delete()
            .eq('user_id', userId)
            .eq('collectionId', collectionId);

        if (docsError) throw docsError;

        const { error: collectionError } = await supabase
            .from('lindex_collections')
            .delete()
            .eq('id', collectionId)
            .eq('user_id', userId);

        if (collectionError) throw collectionError;
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
        createManualDocument,
        uploadFile,
        getDocumentStatus,
        getNewTitles,
        fillKnowledgeGaps,
        enhanceKnowledge,
        fillKnowledgeGapsDeep,
        getManualActions,
        updateManualActionStatus,
        deleteManualAction
    };
};
