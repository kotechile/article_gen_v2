import React from 'react';
import type { Document } from '../../types/knowledge';
import { FileText, Loader, AlertTriangle, CheckCircle, File } from 'lucide-react';

interface DocumentsTableProps {
    documents: Document[];
    isLoading: boolean;
}

export const DocumentsTable: React.FC<DocumentsTableProps> = ({ documents, isLoading }) => {

    const getStatusIcon = (status?: string) => {
        switch (status) {
            case 'completed': return <CheckCircle className="w-4 h-4 text-green-500" />;
            case 'processing': return <Loader className="w-4 h-4 text-indigo-500 animate-spin" />;
            case 'error': return <AlertTriangle className="w-4 h-4 text-red-500" />;
            default: return <div className="w-2 h-2 rounded-full bg-gray-300" />;
        }
    };

    if (isLoading) {
        return <div className="p-8 text-center text-gray-500">Loading documents...</div>;
    }

    if (documents.length === 0) {
        return (
            <div className="p-12 border-2 border-dashed border-gray-200 dark:border-gray-700 rounded-xl flex flex-col items-center justify-center text-gray-400">
                <FileText className="w-12 h-12 mb-3 opacity-50" />
                <p>No documents in this collection yet.</p>
            </div>
        );
    }

    return (
        <div className="overflow-x-auto border border-gray-200 dark:border-gray-700 rounded-xl">
            <table className="w-full text-left text-sm">
                <thead>
                    <tr className="bg-gray-50 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
                        <th className="px-6 py-4 font-medium text-gray-500 dark:text-gray-400">ID</th>
                        <th className="px-6 py-4 font-medium text-gray-500 dark:text-gray-400">Title</th>
                        <th className="px-6 py-4 font-medium text-gray-500 dark:text-gray-400">Created At</th>
                        <th className="px-6 py-4 font-medium text-gray-500 dark:text-gray-400">Status</th>
                        <th className="px-6 py-4 font-medium text-gray-500 dark:text-gray-400 text-right">Type</th>
                    </tr>
                </thead>
                <tbody className="divide-y divide-gray-100 dark:divide-gray-800 bg-white dark:bg-gray-900">
                    {documents.map((doc) => (
                        <tr key={doc.id} className="hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors">
                            <td className="px-6 py-4 text-gray-400 font-mono text-xs">{String(doc.id).slice(0, 8)}...</td>
                            <td className="px-6 py-4 font-medium text-gray-900 dark:text-gray-200">
                                {doc.title}
                            </td>
                            <td className="px-6 py-4 text-gray-500 dark:text-gray-400">
                                {doc.created_at ? new Date(doc.created_at).toLocaleDateString() : '-'}
                            </td>
                            <td className="px-6 py-4">
                                <div className="flex items-center gap-2">
                                    {getStatusIcon(doc.processing_status || (doc.in_vector_store ? 'completed' : 'pending'))}
                                    <span className="capitalize text-gray-600 dark:text-gray-300">
                                        {doc.processing_status || (doc.in_vector_store ? 'Ready' : 'Pending')}
                                    </span>
                                </div>
                            </td>
                            <td className="px-6 py-4 text-right">
                                <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400">
                                    {doc.source_type === 'file' ? <File className="w-3 h-3" /> : <FileText className="w-3 h-3" />}
                                    {doc.source_type}
                                </span>
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
};
