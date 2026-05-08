import React from 'react';
import { X } from 'lucide-react';

interface PreviewModalProps {
    isOpen: boolean;
    onClose: () => void;
    htmlContent: string;
    title: string;
}

export const PreviewModal: React.FC<PreviewModalProps> = ({ isOpen, onClose, htmlContent, title }) => {
    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4 animate-in fade-in duration-200">
            <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-2xl max-w-4xl w-full max-h-[90vh] flex flex-col border border-gray-100 dark:border-gray-700">

                {/* Header */}
                <div className="flex items-center justify-between p-4 border-b border-gray-100 dark:border-gray-700">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white truncate pr-4">
                        Preview: {title}
                    </h3>
                    <button
                        onClick={onClose}
                        className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                    >
                        <X className="w-5 h-5" />
                    </button>
                </div>

                {/* Content */}
                <div className="flex-1 overflow-y-auto p-8 bg-gray-50/50 dark:bg-gray-900/50">
                    <div
                        className="prose prose-indigo dark:prose-invert max-w-none mx-auto bg-white dark:bg-gray-800 p-8 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700 
                        prose-p:my-6 prose-headings:my-6 prose-li:my-2 prose-img:my-8 prose-h2:mt-10 prose-h2:mb-6
                        [&_.geo-key-takeaways]:rounded-2xl [&_.geo-key-takeaways]:border [&_.geo-key-takeaways]:border-gray-200
                        [&_.geo-key-takeaways]:bg-gray-100 [&_.geo-key-takeaways]:px-6 [&_.geo-key-takeaways]:py-5
                        [&_.geo-key-takeaways]:text-gray-900 [&_.geo-key-takeaways_h2]:mt-0 [&_.geo-key-takeaways_h2]:mb-3
                        [&_.geo-key-takeaways_h2]:text-gray-900 [&_.geo-key-takeaways_p]:text-gray-800
                        [&_.geo-key-takeaways_li]:text-gray-800 [&_.geo-key-takeaways_ul]:mb-0"
                        dangerouslySetInnerHTML={{ __html: htmlContent }}
                    />
                </div>

                {/* Footer */}
                <div className="p-4 border-t border-gray-100 dark:border-gray-700 flex justify-end gap-3 bg-white dark:bg-gray-800 rounded-b-2xl">
                    <button
                        onClick={onClose}
                        className="px-4 py-2 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl hover:bg-gray-50 dark:hover:bg-gray-700 transition"
                    >
                        Close Preview
                    </button>
                    {/* Placeholder for Copy/Export actions if needed */}
                </div>
            </div>
        </div>
    );
};
