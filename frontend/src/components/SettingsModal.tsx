
import React, { useState, useEffect } from 'react';
import { X, Plus, Trash2, Edit2, Save, Loader2, RefreshCw } from 'lucide-react';
import { supabase } from '../lib/supabase';
import { useAuth } from '../context/auth-context';
import type { WordPressDetail } from '../types';

interface SettingsModalProps {
    isOpen: boolean;
    onClose: () => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({ isOpen, onClose }) => {
    const { user } = useAuth();
    const [sites, setSites] = useState<WordPressDetail[]>([]);
    const [loading, setLoading] = useState(true);
    const [editingId, setEditingId] = useState<string | null>(null);
    const [formData, setFormData] = useState<Partial<WordPressDetail>>({});
    const [isSaving, setIsSaving] = useState(false);
    const [isSyncing, setIsSyncing] = useState(false);
    const [syncResult, setSyncResult] = useState<{ count: number, message: string } | null>(null);

    // New state for key/posts management
    const [activeTab, setActiveTab] = useState<'settings' | 'posts'>('settings');
    const [importedPosts, setImportedPosts] = useState<any[]>([]);
    const [postsLoading, setPostsLoading] = useState(false);
    const [showAddPost, setShowAddPost] = useState(false);
    const [newPost, setNewPost] = useState({ title: '', link: '', excerpt: '' });

    useEffect(() => {
        if (isOpen && user) {
            fetchSites();
        }
    }, [isOpen, user]);

    useEffect(() => {
        if (activeTab === 'posts' && user) {
            fetchImportedPosts();
        }
    }, [activeTab, user]);

    const fetchImportedPosts = async () => {
        try {
            setPostsLoading(true);
            const { data, error } = await supabase
                .from('wordpress_imported_posts')
                .select('*')
                .eq('user_id', user!.id)
                .order('created_at', { ascending: false });

            if (error) throw error;
            setImportedPosts(data || []);
        } catch (error) {
            console.error('Error fetching posts:', error);
        } finally {
            setPostsLoading(false);
        }
    };

    const handleAddManualPost = async () => {
        if (!user || !newPost.title || !newPost.link) return;

        try {
            setPostsLoading(true);
            const { error } = await supabase
                .from('wordpress_imported_posts')
                .insert([{
                    user_id: user.id,
                    title: newPost.title,
                    link: newPost.link,
                    excerpt: newPost.excerpt,
                    source_site: 'manual',
                    status: 'publish'
                }]);

            if (error) throw error;

            setNewPost({ title: '', link: '', excerpt: '' });
            setShowAddPost(false);
            fetchImportedPosts();
        } catch (error) {
            console.error('Error adding post:', error);
            alert('Failed to add post');
        } finally {
            setPostsLoading(false);
        }
    };

    const fetchSites = async () => {
        try {
            setLoading(true);
            const { data, error } = await supabase
                .from('wordPress_details')
                .select('*')
                .eq('user_id', user!.id);

            if (error) throw error;
            setSites(data || []);
        } catch (error) {
            console.error('Error fetching sites:', error);
        } finally {
            setLoading(false);
        }
    };

    const handleSave = async () => {
        if (!user || !formData.domain || !formData.wpUserName || !formData.wordpress_key) return;

        try {
            setIsSaving(true);
            const payload = {
                ...formData,
                user_id: user.id
            };

            let error;
            if (editingId && editingId !== 'new') {
                const { error: updateError } = await supabase
                    .from('wordPress_details')
                    .update(payload)
                    .eq('id', editingId);
                error = updateError;
            } else {
                const { error: insertError } = await supabase
                    .from('wordPress_details')
                    .insert([payload]);
                error = insertError;
            }

            if (error) throw error;

            setEditingId(null);
            setFormData({});
            fetchSites();
        } catch (error) {
            console.error('Error saving site:', error);
            alert('Failed to save site details');
        } finally {
            setIsSaving(false);
        }
    };

    const handleDelete = async (id: string) => {
        if (!confirm('Are you sure you want to delete this site?')) return;

        try {
            const { error } = await supabase
                .from('wordPress_details')
                .delete()
                .eq('id', id);

            if (error) throw error;
            setSites(sites.filter(s => s.id !== id));
        } catch (error) {
            console.error('Error deleting site:', error);
            alert('Failed to delete site');
        }
    };

    const handleSync = async () => {
        if (!user) return;
        setIsSyncing(true);
        setSyncResult(null);
        try {
            // Call the backend API
            const response = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:5001'}/api/wordpress/sync-posts?user_id=${user.id}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) throw new Error('Failed to sync posts');

            const result = await response.json();
            setSyncResult({ count: result.total_synced, message: result.details });

            // Clear success message after 5 seconds
            setTimeout(() => setSyncResult(null), 5000);
        } catch (error) {
            console.error('Error syncing posts:', error);
            setSyncResult({ count: 0, message: 'Sync failed. Check console for details.' });
        } finally {
            setIsSyncing(false);
        }
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
            <div className="bg-white dark:bg-gray-800 rounded-2xl w-full max-w-4xl max-h-[85vh] overflow-hidden flex flex-col shadow-2xl">
                <div className="p-6 border-b border-gray-100 dark:border-gray-700 flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <h2 className="text-xl font-bold text-gray-900 dark:text-white">Settings</h2>
                        <div className="flex bg-gray-100 dark:bg-gray-700 rounded-lg p-1">
                            <button
                                onClick={() => setActiveTab('settings')}
                                className={`px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${activeTab === 'settings'
                                    ? 'bg-white dark:bg-gray-600 text-gray-900 dark:text-white shadow-sm'
                                    : 'text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
                                    }`}
                            >
                                WordPress Sites
                            </button>
                            <button
                                onClick={() => setActiveTab('posts')}
                                className={`px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${activeTab === 'posts'
                                    ? 'bg-white dark:bg-gray-600 text-gray-900 dark:text-white shadow-sm'
                                    : 'text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
                                    }`}
                            >
                                Imported Posts
                            </button>
                        </div>
                    </div>

                    <div className="flex items-center gap-2">
                        {activeTab === 'settings' && (
                            <>
                                {syncResult && (
                                    <span className={`text-sm font-medium ${syncResult.message.includes('failed') ? 'text-red-500' : 'text-green-500'}`}>
                                        {syncResult.message} {syncResult.count > 0 && `(${syncResult.count} posts)`}
                                    </span>
                                )}
                                <button
                                    onClick={handleSync}
                                    disabled={isSyncing || sites.length === 0}
                                    className="flex items-center gap-2 px-3 py-1.5 text-sm font-medium text-indigo-600 bg-indigo-50 hover:bg-indigo-100 dark:text-indigo-400 dark:bg-indigo-900/30 dark:hover:bg-indigo-900/50 rounded-lg transition-colors disabled:opacity-50"
                                    title="Sync posts from all sites"
                                >
                                    <RefreshCw className={`w-4 h-4 ${isSyncing ? 'animate-spin' : ''}`} />
                                    {isSyncing ? 'Syncing...' : 'Sync Posts'}
                                </button>
                            </>
                        )}
                        <button onClick={onClose} className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-full transition-colors">
                            <X className="w-5 h-5 text-gray-500" />
                        </button>
                    </div>
                </div>

                <div className="flex-1 overflow-y-auto p-6">
                    {activeTab === 'settings' ? (
                        <>
                            {loading ? (
                                <div className="flex justify-center p-8">
                                    <Loader2 className="w-8 h-8 animate-spin text-indigo-600" />
                                </div>
                            ) : (
                                <div className="space-y-4">
                                    {sites.map(site => (
                                        <div key={site.id} className="p-4 border border-gray-200 dark:border-gray-700 rounded-xl bg-gray-50 dark:bg-gray-900/50">
                                            {editingId === site.id ? (
                                                <div className="space-y-4">
                                                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                                                        <div>
                                                            <label className="block text-xs font-medium text-gray-500 mb-1">Domain</label>
                                                            <input
                                                                type="text"
                                                                value={formData.domain || ''}
                                                                onChange={e => setFormData({ ...formData, domain: e.target.value })}
                                                                className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-sm"
                                                                placeholder="example.com"
                                                            />
                                                        </div>
                                                        <div>
                                                            <label className="block text-xs font-medium text-gray-500 mb-1">Username</label>
                                                            <input
                                                                type="text"
                                                                value={formData.wpUserName || ''}
                                                                onChange={e => setFormData({ ...formData, wpUserName: e.target.value })}
                                                                className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-sm"
                                                                placeholder="admin"
                                                            />
                                                        </div>
                                                        <div className="sm:col-span-2">
                                                            <label className="block text-xs font-medium text-gray-500 mb-1">Application Password</label>
                                                            <input
                                                                type="password"
                                                                value={formData.wordpress_key || ''}
                                                                onChange={e => setFormData({ ...formData, wordpress_key: e.target.value })}
                                                                className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-sm"
                                                                placeholder="xxxx xxxx xxxx xxxx"
                                                            />
                                                        </div>
                                                    </div>
                                                    <div className="flex justify-end gap-2">
                                                        <button
                                                            onClick={() => { setEditingId(null); setFormData({}); }}
                                                            className="px-3 py-1.5 text-sm text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700 rounded-lg"
                                                        >
                                                            Cancel
                                                        </button>
                                                        <button
                                                            onClick={handleSave}
                                                            disabled={isSaving}
                                                            className="px-3 py-1.5 text-sm bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 flex items-center gap-2"
                                                        >
                                                            {isSaving ? <Loader2 className="w-3 h-3 animate-spin" /> : <Save className="w-3 h-3" />}
                                                            Save Changes
                                                        </button>
                                                    </div>
                                                </div>
                                            ) : (
                                                <div className="flex items-center justify-between">
                                                    <div>
                                                        <h3 className="font-medium text-gray-900 dark:text-white">{site.domain}</h3>
                                                        <p className="text-sm text-gray-500">{site.wpUserName}</p>
                                                    </div>
                                                    <div className="flex items-center gap-2">
                                                        <button
                                                            onClick={() => { setEditingId(site.id); setFormData(site); }}
                                                            className="p-2 text-gray-400 hover:text-indigo-600 hover:bg-indigo-50 dark:hover:bg-indigo-900/30 rounded-lg"
                                                        >
                                                            <Edit2 className="w-4 h-4" />
                                                        </button>
                                                        <button
                                                            onClick={() => handleDelete(site.id)}
                                                            className="p-2 text-gray-400 hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-900/30 rounded-lg"
                                                        >
                                                            <Trash2 className="w-4 h-4" />
                                                        </button>
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    ))}

                                    {editingId === 'new' && (
                                        <div className="p-4 border border-indigo-200 dark:border-indigo-800 rounded-xl bg-indigo-50/50 dark:bg-indigo-900/20">
                                            <div className="space-y-4">
                                                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                                                    <div>
                                                        <label className="block text-xs font-medium text-gray-500 mb-1">Domain</label>
                                                        <input
                                                            type="text"
                                                            value={formData.domain || ''}
                                                            onChange={e => setFormData({ ...formData, domain: e.target.value })}
                                                            className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-sm"
                                                            placeholder="example.com"
                                                        />
                                                    </div>
                                                    <div>
                                                        <label className="block text-xs font-medium text-gray-500 mb-1">Username</label>
                                                        <input
                                                            type="text"
                                                            value={formData.wpUserName || ''}
                                                            onChange={e => setFormData({ ...formData, wpUserName: e.target.value })}
                                                            className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-sm"
                                                            placeholder="admin"
                                                        />
                                                    </div>
                                                    <div className="sm:col-span-2">
                                                        <label className="block text-xs font-medium text-gray-500 mb-1">Application Password</label>
                                                        <input
                                                            type="password"
                                                            value={formData.wordpress_key || ''}
                                                            onChange={e => setFormData({ ...formData, wordpress_key: e.target.value })}
                                                            className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-sm"
                                                            placeholder="xxxx xxxx xxxx xxxx"
                                                        />
                                                    </div>
                                                </div>
                                                <div className="flex justify-end gap-2">
                                                    <button
                                                        onClick={() => { setEditingId(null); setFormData({}); }}
                                                        className="px-3 py-1.5 text-sm text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700 rounded-lg"
                                                    >
                                                        Cancel
                                                    </button>
                                                    <button
                                                        onClick={handleSave}
                                                        disabled={isSaving}
                                                        className="px-3 py-1.5 text-sm bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 flex items-center gap-2"
                                                    >
                                                        {isSaving ? <Loader2 className="w-3 h-3 animate-spin" /> : <Save className="w-3 h-3" />}
                                                        Save Site
                                                    </button>
                                                </div>
                                            </div>
                                        </div>
                                    )}

                                    {!editingId && (
                                        <button
                                            onClick={() => { setEditingId('new'); setFormData({}); }}
                                            className="w-full py-3 border-2 border-dashed border-gray-300 dark:border-gray-700 rounded-xl text-gray-500 hover:border-indigo-500 hover:text-indigo-500 transition-colors flex items-center justify-center gap-2"
                                        >
                                            <Plus className="w-5 h-5" />
                                            Add New WordPress Site
                                        </button>
                                    )}
                                </div>
                            )}
                        </>
                    ) : (
                        <div className="space-y-6">
                            <div className="flex justify-between items-center">
                                <h3 className="text-lg font-medium text-gray-900 dark:text-white">External Links & Posts</h3>
                                <button
                                    onClick={() => setShowAddPost(!showAddPost)}
                                    className="flex items-center gap-2 px-3 py-1.5 text-sm font-medium text-indigo-600 bg-indigo-50 hover:bg-indigo-100 dark:text-indigo-400 dark:bg-indigo-900/30 dark:hover:bg-indigo-900/50 rounded-lg transition-colors"
                                >
                                    <Plus className="w-4 h-4" />
                                    Add Manually
                                </button>
                            </div>

                            {showAddPost && (
                                <div className="p-4 bg-gray-50 dark:bg-gray-700/50 rounded-xl border border-gray-100 dark:border-gray-700 space-y-3">
                                    <input
                                        placeholder="Article Title"
                                        value={newPost.title}
                                        onChange={(e) => setNewPost({ ...newPost, title: e.target.value })}
                                        className="w-full px-3 py-2 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg"
                                    />
                                    <input
                                        placeholder="Link / URL"
                                        value={newPost.link}
                                        onChange={(e) => setNewPost({ ...newPost, link: e.target.value })}
                                        className="w-full px-3 py-2 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg"
                                    />
                                    <textarea
                                        placeholder="Excerpt / Short Description (Optional)"
                                        value={newPost.excerpt}
                                        onChange={(e) => setNewPost({ ...newPost, excerpt: e.target.value })}
                                        className="w-full px-3 py-2 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg"
                                        rows={2}
                                    />
                                    <div className="flex justify-end gap-2">
                                        <button onClick={() => setShowAddPost(false)} className="px-3 py-1.5 text-sm text-gray-500">Cancel</button>
                                        <button
                                            onClick={handleAddManualPost}
                                            className="px-3 py-1.5 text-sm font-medium text-white bg-indigo-600 rounded-lg hover:bg-indigo-700"
                                        >
                                            Add Post
                                        </button>
                                    </div>
                                </div>
                            )}

                            {postsLoading ? (
                                <div className="flex justify-center p-8"><Loader2 className="w-6 h-6 animate-spin text-gray-400" /></div>
                            ) : (
                                <div className="space-y-3">
                                    {importedPosts.length === 0 ? (
                                        <div className="text-center py-8 text-gray-500">No posts imported yet. Sync from Settings or add manually.</div>
                                    ) : (
                                        importedPosts.map((post) => (
                                            <div key={post.id} className="p-3 bg-white dark:bg-gray-800 border border-gray-100 dark:border-gray-700 rounded-lg hover:shadow-sm transition-shadow">
                                                <div className="flex justify-between items-start">
                                                    <div>
                                                        <h4 className="font-medium text-gray-900 dark:text-white line-clamp-1">{post.title}</h4>
                                                        <a href={post.link} target="_blank" rel="noopener noreferrer" className="text-xs text-indigo-500 hover:underline line-clamp-1">{post.link}</a>
                                                        {post.excerpt && <p className="text-xs text-gray-500 mt-1 line-clamp-2" dangerouslySetInnerHTML={{ __html: post.excerpt }} />}
                                                    </div>
                                                </div>
                                            </div>
                                        ))
                                    )}
                                </div>
                            )}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};
