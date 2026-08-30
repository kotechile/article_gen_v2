import React, { useState, useEffect } from 'react';
import {
    X,
    Loader2,
    CheckCircle2,
    AlertCircle,
    Share2,
    Sparkles,
    ExternalLink,
    Image as ImageIcon,
    Link as LinkIcon,
    AlertTriangle,
    RefreshCw
} from 'lucide-react';
import {
    getLinkedInAccount,
    getLinkedInAuthUrl,
    publishToLinkedIn,
    repurposeForLinkedIn,
    type LinkedInAccountStatus,
    type RepurposedLinkedInContent
} from '../services/linkedinService';

interface LinkedInPublishModalProps {
    articleData: any;
    articleId: string;
    onClose: () => void;
    onSuccess?: (postUrl: string) => void;
}

const LINKEDIN_MAX_CHARS = 3000;

export const LinkedInPublishModal: React.FC<LinkedInPublishModalProps> = ({
    articleData,
    articleId,
    onClose,
    onSuccess
}) => {
    const [accountStatus, setAccountStatus] = useState<LinkedInAccountStatus | null>(null);
    const [loadingAccount, setLoadingAccount] = useState(true);
    const [repurposing, setRepurposing] = useState(false);
    const [publishing, setPublishing] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [publishedUrl, setPublishedUrl] = useState<string | null>(null);

    // Form inputs
    const [commentary, setCommentary] = useState<string>('');
    const [includeImage, setIncludeImage] = useState<boolean>(true);
    const [includeLink, setIncludeLink] = useState<boolean>(false);
    const [articleUrl, setArticleUrl] = useState<string>('');

    // Pre-populate article URL from WordPress post link if available
    useEffect(() => {
        const wpUrl = articleData?.last_wp_post_url || articleData?.canonical_url || '';
        setArticleUrl(wpUrl);
        if (wpUrl) {
            setIncludeLink(true);
        }
    }, [articleData]);

    // Initial content setup
    useEffect(() => {
        // If article already has saved linkedin_post_content, use that
        if (articleData?.linkedin_post_content) {
            setCommentary(articleData.linkedin_post_content);
            return;
        }

        // Otherwise, construct a clean initial post draft from hook and excerpt
        const hook = articleData?.hook || articleData?.deck || articleData?.title || articleData?.Title || '';
        const excerpt = articleData?.excerpt || articleData?.thesis || '';
        const keywords = Array.isArray(articleData?.keywords)
            ? articleData.keywords
            : typeof articleData?.Keywords === 'string'
            ? articleData.Keywords.split(',').map((k: string) => k.trim()).filter(Boolean)
            : [];

        const hashtags = keywords
            .slice(0, 4)
            .map((k: string) => `#${k.replace(/[^a-zA-Z0-9]/g, '')}`)
            .join(' ');

        const initialDraft = [
            hook,
            '',
            excerpt,
            '',
            "What's your take on this? How do you approach it in your work? Drop your thoughts below 👇",
            '',
            hashtags || '#Leadership #Innovation'
        ].filter(Boolean).join('\n');

        setCommentary(initialDraft);
    }, [articleData]);

    // Load account on mount
    useEffect(() => {
        loadAccount();
    }, []);

    const loadAccount = async () => {
        try {
            setLoadingAccount(true);
            setError(null);
            const status = await getLinkedInAccount();
            setAccountStatus(status);
        } catch (err: any) {
            setError(err?.message || 'Failed to load LinkedIn account status');
        } finally {
            setLoadingAccount(false);
        }
    };

    const handleConnect = async () => {
        try {
            const url = await getLinkedInAuthUrl();
            window.location.href = url;
        } catch (err: any) {
            setError(err?.message || 'Failed to initiate LinkedIn connection');
        }
    };

    const handleRepurposeWithAI = async () => {
        const title = articleData?.Title || articleData?.title || 'Article';
        const content = articleData?.htmlArticle || articleData?.articleText || commentary;

        if (!content || content.trim().length < 20) {
            setError('Article content is too short to repurpose.');
            return;
        }

        try {
            setRepurposing(true);
            setError(null);
            const result: RepurposedLinkedInContent = await repurposeForLinkedIn(
                title,
                content,
                articleData?.tone || 'thought_leadership'
            );
            if (result?.full_post) {
                setCommentary(result.full_post);
            }
        } catch (err: any) {
            setError(err?.message || 'Failed to repurpose with AI');
        } finally {
            setRepurposing(false);
        }
    };

    const handlePublish = async () => {
        if (!accountStatus?.connected || !accountStatus?.account) {
            setError('Please connect your LinkedIn account first.');
            return;
        }

        if (!commentary.trim()) {
            setError('Post commentary cannot be empty.');
            return;
        }

        try {
            setPublishing(true);
            setError(null);

            const imageUrl = includeImage ? (articleData?.featuredImageUrl || articleData?.imageurl || null) : null;
            const link = includeLink && articleUrl.trim() ? articleUrl.trim() : null;

            const res = await publishToLinkedIn({
                article_id: articleId,
                commentary: commentary.trim(),
                image_url: imageUrl || undefined,
                image_alt_text: articleData?.mediaAltText || articleData?.title || undefined,
                article_url: link || undefined,
                article_title: articleData?.Title || articleData?.title || undefined,
                article_description: articleData?.excerpt || undefined,
            });

            if (res.success && res.post_url) {
                setPublishedUrl(res.post_url);
                if (onSuccess) {
                    onSuccess(res.post_url);
                }
            } else {
                setError(res.message || 'Publishing failed. Please try again.');
            }
        } catch (err: any) {
            setError(err?.response?.data?.message || err?.message || 'Failed to publish to LinkedIn.');
        } finally {
            setPublishing(false);
        }
    };

    const charCount = commentary.length;
    const isOverLimit = charCount > LINKEDIN_MAX_CHARS;
    const featuredImg = articleData?.featuredImageUrl || articleData?.imageurl;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-background/80 backdrop-blur-sm animate-in fade-in duration-200">
            <div className="bg-card border border-border w-full max-w-4xl max-h-[90vh] rounded-2xl shadow-2xl flex flex-col overflow-hidden">
                {/* Modal Header */}
                <div className="px-6 py-4 border-b border-border flex items-center justify-between bg-muted/40">
                    <div className="flex items-center gap-3">
                        <div className="w-9 h-9 rounded-xl bg-[#0A66C2] text-white flex items-center justify-center font-bold text-lg shadow-sm">
                            in
                        </div>
                        <div>
                            <h2 className="text-lg font-bold text-foreground">Publish to LinkedIn</h2>
                            <p className="text-xs text-muted-foreground">Tailor and share directly to your personal LinkedIn feed</p>
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="text-muted-foreground hover:text-foreground p-1.5 rounded-lg hover:bg-accent transition"
                    >
                        <X className="w-5 h-5" />
                    </button>
                </div>

                {/* Modal Body */}
                <div className="flex-1 overflow-y-auto p-6 space-y-6">
                    {/* Error message */}
                    {error && (
                        <div className="p-3 bg-destructive/10 border border-destructive/20 rounded-xl text-destructive text-sm flex items-start gap-2">
                            <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                            <span className="flex-1">{error}</span>
                        </div>
                    )}

                    {/* Success Screen */}
                    {publishedUrl ? (
                        <div className="py-12 px-6 text-center space-y-4">
                            <div className="w-16 h-16 bg-emerald-500/10 text-emerald-500 rounded-full flex items-center justify-center mx-auto">
                                <CheckCircle2 className="w-10 h-10" />
                            </div>
                            <h3 className="text-xl font-bold text-foreground">Post Published to LinkedIn!</h3>
                            <p className="text-sm text-muted-foreground max-w-md mx-auto">
                                Your post is live on your LinkedIn feed. You can now view it, interact with comments, or share it.
                            </p>
                            <div className="pt-2 flex justify-center gap-3">
                                <a
                                    href={publishedUrl}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="flex items-center gap-2 px-5 py-2.5 bg-[#0A66C2] hover:bg-[#084e96] text-white font-medium rounded-xl transition shadow-md"
                                >
                                    <ExternalLink className="w-4 h-4" />
                                    View Post on LinkedIn
                                </a>
                                <button
                                    onClick={onClose}
                                    className="px-5 py-2.5 border border-border bg-background hover:bg-muted font-medium rounded-xl transition"
                                >
                                    Done
                                </button>
                            </div>
                        </div>
                    ) : (
                        <>
                            {/* Account Status Card */}
                            <div className="bg-muted/30 border border-border rounded-xl p-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
                                {loadingAccount ? (
                                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                                        <Loader2 className="w-4 h-4 animate-spin" />
                                        Checking LinkedIn connection...
                                    </div>
                                ) : accountStatus?.connected && accountStatus?.account ? (
                                    <div className="flex items-center gap-3">
                                        {accountStatus.account.profile_picture_url ? (
                                            <img
                                                src={accountStatus.account.profile_picture_url}
                                                alt={accountStatus.account.account_name}
                                                className="w-10 h-10 rounded-full border border-border object-cover"
                                            />
                                        ) : (
                                            <div className="w-10 h-10 rounded-full bg-primary/10 text-primary flex items-center justify-center font-bold">
                                                {accountStatus.account.account_name.charAt(0)}
                                            </div>
                                        )}
                                        <div>
                                            <div className="flex items-center gap-2">
                                                <span className="font-semibold text-foreground text-sm">
                                                    {accountStatus.account.account_name}
                                                </span>
                                                <span className="px-2 py-0.5 text-xs bg-emerald-500/10 text-emerald-500 border border-emerald-500/20 rounded-full font-medium">
                                                    Connected
                                                </span>
                                            </div>
                                            <p className="text-xs text-muted-foreground">Publishing to Personal Feed</p>
                                        </div>
                                    </div>
                                ) : (
                                    <div className="flex items-center gap-3">
                                        <AlertTriangle className="w-5 h-5 text-amber-500 flex-shrink-0" />
                                        <div>
                                            <p className="text-sm font-semibold text-foreground">LinkedIn account not connected</p>
                                            <p className="text-xs text-muted-foreground">Connect your personal account via OAuth to publish directly.</p>
                                        </div>
                                    </div>
                                )}

                                {!accountStatus?.connected && (
                                    <button
                                        onClick={handleConnect}
                                        className="flex items-center gap-2 px-4 py-2 bg-[#0A66C2] hover:bg-[#084e96] text-white text-xs font-semibold rounded-xl transition shadow-sm"
                                    >
                                        Connect LinkedIn
                                    </button>
                                )}
                            </div>

                            {/* Two-column layout: Editor & Live Preview */}
                            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                                {/* Left Column: Post Editor */}
                                <div className="space-y-4">
                                    <div className="flex items-center justify-between">
                                        <label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                                            Post Commentary
                                        </label>
                                        <button
                                            type="button"
                                            onClick={handleRepurposeWithAI}
                                            disabled={repurposing}
                                            className="flex items-center gap-1.5 text-xs font-medium text-primary hover:text-primary/80 transition disabled:opacity-50"
                                            title="Use AI to condense and optimize this article into a high-engagement LinkedIn post"
                                        >
                                            {repurposing ? (
                                                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                                            ) : (
                                                <Sparkles className="w-3.5 h-3.5 text-amber-500" />
                                            )}
                                            Repurpose with AI
                                        </button>
                                    </div>

                                    <div className="relative">
                                        <textarea
                                            rows={12}
                                            value={commentary}
                                            onChange={(e) => setCommentary(e.target.value)}
                                            placeholder="Write your LinkedIn thought leadership post here..."
                                            className={`w-full text-sm font-sans rounded-xl border p-3.5 bg-background text-foreground focus:outline-none focus:ring-2 resize-y transition ${
                                                isOverLimit
                                                    ? 'border-destructive focus:ring-destructive/30'
                                                    : 'border-border focus:ring-primary/20 focus:border-primary'
                                            }`}
                                        />
                                        <div className="flex justify-between items-center mt-1.5 px-1">
                                            <span className="text-[11px] text-muted-foreground">
                                                Tip: The "...see more" cutoff occurs at ~210 characters.
                                            </span>
                                            <span
                                                className={`text-xs font-medium ${
                                                    isOverLimit
                                                        ? 'text-destructive font-bold'
                                                        : charCount > 2500
                                                        ? 'text-amber-500'
                                                        : 'text-muted-foreground'
                                                }`}
                                            >
                                                {charCount.toLocaleString()} / {LINKEDIN_MAX_CHARS.toLocaleString()}
                                            </span>
                                        </div>
                                    </div>

                                    {/* Attachment controls */}
                                    <div className="space-y-3 pt-2 border-t border-border">
                                        <label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                                            Attachments
                                        </label>

                                        {/* Image Attachment Toggle */}
                                        <div className="flex items-center justify-between p-3 rounded-xl border border-border bg-muted/20">
                                            <div className="flex items-center gap-2.5">
                                                <ImageIcon className="w-4 h-4 text-primary" />
                                                <div>
                                                    <p className="text-xs font-medium text-foreground">Attach Featured Image</p>
                                                    <p className="text-[11px] text-muted-foreground">
                                                        {featuredImg ? 'Image detected and ready to upload' : 'No featured image found'}
                                                    </p>
                                                </div>
                                            </div>
                                            <input
                                                type="checkbox"
                                                checked={includeImage && Boolean(featuredImg)}
                                                disabled={!featuredImg}
                                                onChange={(e) => setIncludeImage(e.target.checked)}
                                                className="w-4 h-4 rounded text-primary focus:ring-primary/30"
                                            />
                                        </div>

                                        {/* Article Link Attachment Toggle */}
                                        <div className="space-y-2 p-3 rounded-xl border border-border bg-muted/20">
                                            <div className="flex items-center justify-between">
                                                <div className="flex items-center gap-2.5">
                                                    <LinkIcon className="w-4 h-4 text-primary" />
                                                    <div>
                                                        <p className="text-xs font-medium text-foreground">Attach Article Link Card</p>
                                                        <p className="text-[11px] text-muted-foreground">Display clickable link preview card</p>
                                                    </div>
                                                </div>
                                                <input
                                                    type="checkbox"
                                                    checked={includeLink}
                                                    onChange={(e) => setIncludeLink(e.target.checked)}
                                                    className="w-4 h-4 rounded text-primary focus:ring-primary/30"
                                                />
                                            </div>
                                            {includeLink && (
                                                <input
                                                    type="url"
                                                    placeholder="https://yourblog.com/article-url"
                                                    value={articleUrl}
                                                    onChange={(e) => setArticleUrl(e.target.value)}
                                                    className="w-full text-xs rounded-lg border border-border p-2 bg-background text-foreground focus:outline-none focus:border-primary"
                                                />
                                            )}
                                        </div>
                                    </div>
                                </div>

                                {/* Right Column: Live Feed Mockup */}
                                <div className="space-y-2">
                                    <label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
                                        <span>LinkedIn Feed Preview</span>
                                        <span className="text-[10px] text-muted-foreground/80 font-normal">(as it will appear in feed)</span>
                                    </label>

                                    <div className="bg-card border border-border/80 rounded-xl p-4 shadow-sm space-y-3 font-sans">
                                        {/* Feed Post Header */}
                                        <div className="flex items-center gap-3">
                                            {accountStatus?.account?.profile_picture_url ? (
                                                <img
                                                    src={accountStatus.account.profile_picture_url}
                                                    alt="Author"
                                                    className="w-11 h-11 rounded-full object-cover border border-border"
                                                />
                                            ) : (
                                                <div className="w-11 h-11 rounded-full bg-[#0A66C2] text-white font-bold flex items-center justify-center">
                                                    {accountStatus?.account?.account_name?.charAt(0) || 'U'}
                                                </div>
                                            )}
                                            <div className="flex-1 min-w-0">
                                                <p className="text-sm font-semibold text-foreground truncate">
                                                    {accountStatus?.account?.account_name || 'Your Name'}
                                                </p>
                                                <p className="text-xs text-muted-foreground truncate">Professional & Creator</p>
                                                <div className="flex items-center gap-1 text-[11px] text-muted-foreground">
                                                    <span>Just now</span>
                                                    <span>•</span>
                                                    <span>🌐</span>
                                                </div>
                                            </div>
                                        </div>

                                        {/* Post Content */}
                                        <div className="text-xs text-foreground/90 whitespace-pre-line leading-relaxed max-h-60 overflow-y-auto">
                                            {commentary || 'Your LinkedIn post text will appear here...'}
                                        </div>

                                        {/* Attached Image or Link Card Preview */}
                                        {includeImage && featuredImg && !includeLink && (
                                            <div className="rounded-lg overflow-hidden border border-border">
                                                <img
                                                    src={featuredImg}
                                                    alt="Post Preview"
                                                    className="w-full max-h-48 object-cover"
                                                />
                                            </div>
                                        )}

                                        {includeLink && articleUrl && (
                                            <div className="border border-border rounded-lg overflow-hidden bg-muted/20">
                                                {featuredImg && (
                                                    <img
                                                        src={featuredImg}
                                                        alt="Article thumbnail"
                                                        className="w-full h-32 object-cover"
                                                    />
                                                )}
                                                <div className="p-2.5 space-y-1">
                                                    <p className="text-[11px] uppercase tracking-wide text-muted-foreground truncate">
                                                        {new URL(articleUrl.startsWith('http') ? articleUrl : `https://${articleUrl}`).hostname}
                                                    </p>
                                                    <p className="text-xs font-semibold text-foreground line-clamp-1">
                                                        {articleData?.Title || articleData?.title || 'Article Title'}
                                                    </p>
                                                    <p className="text-[11px] text-muted-foreground line-clamp-1">
                                                        {articleData?.excerpt || 'Click to read the full article.'}
                                                    </p>
                                                </div>
                                            </div>
                                        )}

                                        {/* LinkedIn Action Bar Mockup */}
                                        <div className="pt-2 border-t border-border flex items-center justify-between text-muted-foreground text-xs">
                                            <div className="flex items-center gap-1 hover:text-foreground transition cursor-default">
                                                <span>👍 Like</span>
                                            </div>
                                            <div className="flex items-center gap-1 hover:text-foreground transition cursor-default">
                                                <span>💬 Comment</span>
                                            </div>
                                            <div className="flex items-center gap-1 hover:text-foreground transition cursor-default">
                                                <span>🔁 Repost</span>
                                            </div>
                                            <div className="flex items-center gap-1 hover:text-foreground transition cursor-default">
                                                <span>✈️ Send</span>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </>
                    )}
                </div>

                {/* Modal Footer */}
                {!publishedUrl && (
                    <div className="px-6 py-4 border-t border-border bg-muted/20 flex items-center justify-between">
                        <button
                            type="button"
                            onClick={onClose}
                            className="px-4 py-2 text-sm font-medium rounded-xl hover:bg-muted transition text-foreground"
                        >
                            Cancel
                        </button>

                        <button
                            type="button"
                            onClick={handlePublish}
                            disabled={publishing || isOverLimit || !accountStatus?.connected}
                            className="flex items-center gap-2 px-6 py-2.5 bg-[#0A66C2] hover:bg-[#084e96] text-white text-sm font-medium rounded-xl transition shadow-md disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            {publishing ? (
                                <>
                                    <Loader2 className="w-4 h-4 animate-spin" />
                                    Publishing to LinkedIn...
                                </>
                            ) : (
                                <>
                                    <Share2 className="w-4 h-4" />
                                    Publish Now
                                </>
                            )}
                        </button>
                    </div>
                )}
            </div>
        </div>
    );
};
