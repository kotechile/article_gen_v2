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
    RefreshCw,
    Copy,
    Check,
    Download,
    BookOpen,
    FileText
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

    // Modal view mode: Automated Feed Post vs Long-form Copy Kit
    const [activeModalTab, setActiveModalTab] = useState<'feed_post' | 'pulse_article'>('feed_post');
    const [copiedField, setCopiedField] = useState<string | null>(null);

    // Form inputs
    const [commentary, setCommentary] = useState<string>('');
    const [includeImage, setIncludeImage] = useState<boolean>(true);
    const [includeLink, setIncludeLink] = useState<boolean>(false);
    const [articleUrl, setArticleUrl] = useState<string>('');
    const [targetType, setTargetType] = useState<'organization' | 'personal'>('personal');
    const [organizationId, setOrganizationId] = useState<string>('144076511');

    const handleCopy = (text: string, fieldName: string) => {
        navigator.clipboard.writeText(text);
        setCopiedField(fieldName);
        setTimeout(() => setCopiedField(null), 2000);
    };

    const handleCopyRichText = async (htmlContent: string) => {
        try {
            if (typeof ClipboardItem !== 'undefined' && navigator.clipboard?.write) {
                const blobHtml = new Blob([htmlContent], { type: 'text/html' });
                const blobText = new Blob([htmlContent.replace(/<[^>]+>/g, '')], { type: 'text/plain' });
                await navigator.clipboard.write([
                    new ClipboardItem({
                        'text/html': blobHtml,
                        'text/plain': blobText,
                    })
                ]);
            } else {
                await navigator.clipboard.writeText(htmlContent.replace(/<[^>]+>/g, ''));
            }
            setCopiedField('rich_body');
            setTimeout(() => setCopiedField(null), 2000);
        } catch {
            await navigator.clipboard.writeText(htmlContent.replace(/<[^>]+>/g, ''));
            setCopiedField('rich_body');
            setTimeout(() => setCopiedField(null), 2000);
        }
    };

    const getDomainFromUrl = (url: string) => {
        try {
            const u = url.startsWith('http') ? url : `https://${url}`;
            return new URL(u).hostname.replace(/^www\./, '');
        } catch {
            return 'aichieve.net';
        }
    };

    const handleInsertLinkIntoPost = () => {
        const clean = articleUrl.trim();
        if (!clean) return;
        if (commentary.includes(clean)) return;
        const linkWithProtocol = clean.startsWith('http') ? clean : `https://${clean}`;
        setCommentary((prev) => {
            const trimmed = prev.trim();
            return `${trimmed}\n\n🔗 Read the full analysis on Aichieve:\n${linkWithProtocol}`;
        });
    };

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
            if (status?.account?.organization_id) {
                setOrganizationId(status.account.organization_id);
            }
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
            setError(err?.response?.data?.message || err?.message || 'Failed to repurpose with AI');
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
            const cleanUrl = articleUrl.trim();
            const rawLink = (includeLink && cleanUrl) ? cleanUrl : (cleanUrl || null);
            const link = rawLink ? (rawLink.startsWith('http') ? rawLink : `https://${rawLink}`) : undefined;

            const res = await publishToLinkedIn({
                article_id: articleId,
                commentary: commentary.trim(),
                image_url: imageUrl || undefined,
                image_alt_text: articleData?.mediaAltText || articleData?.title || undefined,
                article_url: link,
                article_title: articleData?.Title || articleData?.title || undefined,
                article_description: articleData?.excerpt || undefined,
                target_type: targetType,
                organization_id: organizationId || '144076511',
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
    const featuredImg = articleData?.featuredImageUrl || articleData?.imageurl || '';

    // Extract inline images & infographics
    const inlineImages: string[] = [];
    if (featuredImg && !inlineImages.includes(featuredImg)) {
        inlineImages.push(featuredImg);
    }
    if (articleData?.infographicUrl && !inlineImages.includes(articleData.infographicUrl)) {
        inlineImages.push(articleData.infographicUrl);
    }
    if (articleData?.htmlArticle) {
        const matches = articleData.htmlArticle.match(/<img[^>]+src=["']([^"']+)["']/g);
        if (matches) {
            matches.forEach((imgTag: string) => {
                const srcMatch = imgTag.match(/src=["']([^"']+)["']/);
                if (srcMatch && srcMatch[1] && !inlineImages.includes(srcMatch[1])) {
                    inlineImages.push(srcMatch[1]);
                }
            });
        }
    }

    const ctaSnippet = `Enjoyed this article? Follow Aichieve.net (https://www.linkedin.com/company/144076511) for more research, data, and technical deep dives.\n\nSubscribe to our newsletter on LinkedIn to get our next analysis delivered directly to your inbox.`;

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
                            <h2 className="text-lg font-bold text-foreground">LinkedIn Publisher</h2>
                            <p className="text-xs text-muted-foreground">Automated Feed Post & Long-Form Article Package</p>
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="text-muted-foreground hover:text-foreground p-1.5 rounded-lg hover:bg-accent transition"
                    >
                        <X className="w-5 h-5" />
                    </button>
                </div>

                {/* Mode Selector Tabs */}
                <div className="px-6 pt-3 border-b border-border bg-muted/20 flex gap-2">
                    <button
                        type="button"
                        onClick={() => setActiveModalTab('feed_post')}
                        className={`pb-2.5 px-3 text-xs font-semibold border-b-2 transition flex items-center gap-1.5 ${
                            activeModalTab === 'feed_post'
                                ? 'border-[#0A66C2] text-[#0A66C2]'
                                : 'border-transparent text-muted-foreground hover:text-foreground'
                        }`}
                    >
                        <Share2 className="w-3.5 h-3.5" />
                        Automated Feed Post
                    </button>
                    <button
                        type="button"
                        onClick={() => setActiveModalTab('pulse_article')}
                        className={`pb-2.5 px-3 text-xs font-semibold border-b-2 transition flex items-center gap-1.5 ${
                            activeModalTab === 'pulse_article'
                                ? 'border-[#0A66C2] text-[#0A66C2]'
                                : 'border-transparent text-muted-foreground hover:text-foreground'
                        }`}
                    >
                        <BookOpen className="w-3.5 h-3.5" />
                        Long-Form Article Package (Copy to LinkedIn)
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
                    ) : activeModalTab === 'feed_post' ? (
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
                                            <p className="text-xs text-muted-foreground">
                                                Authenticated User ({accountStatus.account.account_name})
                                            </p>
                                        </div>
                                    </div>
                                ) : (
                                    <div className="flex items-center gap-3">
                                        <AlertTriangle className="w-5 h-5 text-amber-500 flex-shrink-0" />
                                        <div>
                                            <p className="text-sm font-semibold text-foreground">LinkedIn account not connected</p>
                                            <p className="text-xs text-muted-foreground">Connect your LinkedIn account via OAuth to publish directly.</p>
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

                            {/* Destination Selector */}
                            {accountStatus?.connected && (
                                <div className="p-3.5 bg-muted/40 border border-border rounded-xl flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
                                    <div>
                                        <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground block">
                                            Publishing Destination
                                        </span>
                                        <span className="text-xs text-muted-foreground">
                                            Choose where to publish this article post
                                        </span>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <button
                                            type="button"
                                            onClick={() => setTargetType('organization')}
                                            className={`flex items-center gap-1.5 px-3.5 py-2 rounded-lg text-xs font-semibold transition border ${
                                                targetType === 'organization'
                                                    ? 'bg-[#0A66C2] text-white border-[#0A66C2] shadow-sm'
                                                    : 'bg-background text-muted-foreground border-border hover:bg-muted'
                                            }`}
                                        >
                                            🏢 Aichieve.net Page ({organizationId})
                                        </button>
                                        <button
                                            type="button"
                                            onClick={() => setTargetType('personal')}
                                            className={`flex items-center gap-1.5 px-3.5 py-2 rounded-lg text-xs font-semibold transition border ${
                                                targetType === 'personal'
                                                    ? 'bg-[#0A66C2] text-white border-[#0A66C2] shadow-sm'
                                                    : 'bg-background text-muted-foreground border-border hover:bg-muted'
                                            }`}
                                        >
                                            👤 Personal Feed
                                        </button>
                                    </div>
                                </div>
                            )}

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

                                    {/* Article Link Card (Target URL on your website) */}
                                    <div className="p-3.5 rounded-xl border border-border bg-card shadow-sm space-y-2.5">
                                        <div className="flex items-center justify-between">
                                            <label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
                                                <LinkIcon className="w-3.5 h-3.5 text-[#0A66C2]" />
                                                Article URL (Your Website)
                                            </label>
                                            {articleUrl.trim() && (
                                                <button
                                                    type="button"
                                                    onClick={handleInsertLinkIntoPost}
                                                    disabled={commentary.includes(articleUrl.trim().replace(/^https?:\/\//, ''))}
                                                    className="text-[11px] font-medium flex items-center gap-1 transition disabled:opacity-50 disabled:cursor-default text-[#0A66C2] hover:underline"
                                                    title="Append URL directly into your post body text"
                                                >
                                                    {commentary.includes(articleUrl.trim().replace(/^https?:\/\//, '')) ? (
                                                        <span className="text-emerald-600 font-semibold flex items-center gap-1">
                                                            <Check className="w-3 h-3" /> Link already in post text
                                                        </span>
                                                    ) : (
                                                        <span>+ Insert link into post text</span>
                                                    )}
                                                </button>
                                            )}
                                        </div>
                                        <div className="relative">
                                            <input
                                                type="url"
                                                placeholder="https://aichieve.net/stop-blaming-demand-planning-for-stockouts/"
                                                value={articleUrl}
                                                onChange={(e) => {
                                                    const val = e.target.value;
                                                    setArticleUrl(val);
                                                    if (val.trim()) {
                                                        setIncludeLink(true);
                                                    }
                                                }}
                                                className="w-full text-xs font-mono rounded-lg border border-border p-2.5 bg-background text-foreground focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary transition"
                                            />
                                        </div>
                                        <div className="flex items-center justify-between text-[11px] text-muted-foreground">
                                            <span>LinkedIn will create a clickable rich preview card to this live article.</span>
                                            <label className="flex items-center gap-1.5 cursor-pointer text-xs font-medium text-foreground">
                                                <input
                                                    type="checkbox"
                                                    checked={includeLink && Boolean(articleUrl.trim())}
                                                    onChange={(e) => setIncludeLink(e.target.checked)}
                                                    className="w-3.5 h-3.5 rounded text-primary focus:ring-primary/30 accent-[#0A66C2]"
                                                />
                                                Attach Link Preview
                                            </label>
                                        </div>
                                    </div>

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
                                            {targetType === 'organization' ? (
                                                <div className="w-11 h-11 rounded-lg bg-[#0A66C2] text-white font-bold flex items-center justify-center text-sm shadow-sm">
                                                    AI
                                                </div>
                                            ) : accountStatus?.account?.profile_picture_url ? (
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
                                                    {targetType === 'organization'
                                                        ? 'Aichieve.net'
                                                        : (accountStatus?.account?.account_name || 'Your Name')}
                                                </p>
                                                <p className="text-xs text-muted-foreground truncate">
                                                    {targetType === 'organization'
                                                        ? 'Technology, Information and Internet • Company'
                                                        : 'Professional'}
                                                </p>
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

                                        {includeLink && articleUrl.trim() && (
                                            <div className="border border-border rounded-lg overflow-hidden bg-muted/20 shadow-sm transition">
                                                {featuredImg && (
                                                    <div className="relative">
                                                        <img
                                                            src={featuredImg}
                                                            alt="Article thumbnail"
                                                            className="w-full h-36 object-cover"
                                                        />
                                                        <div className="absolute bottom-2 left-2 px-2 py-0.5 rounded bg-black/80 text-white text-[11px] font-mono tracking-tight flex items-center gap-1 backdrop-blur-sm shadow">
                                                            <LinkIcon className="w-3 h-3 text-sky-400" />
                                                            <span>{getDomainFromUrl(articleUrl)}</span>
                                                        </div>
                                                    </div>
                                                )}
                                                <div className="p-2.5 space-y-1 bg-card">
                                                    <p className="text-[11px] uppercase tracking-wide text-muted-foreground truncate font-semibold flex items-center gap-1">
                                                        <span>{getDomainFromUrl(articleUrl)}</span>
                                                        <span>•</span>
                                                        <span className="text-[10px] text-[#0A66C2] font-medium lowercase">article link card</span>
                                                    </p>
                                                    <p className="text-xs font-semibold text-foreground line-clamp-1">
                                                        {articleData?.Title || articleData?.title || 'Article Title'}
                                                    </p>
                                                    <p className="text-[11px] text-muted-foreground line-clamp-1">
                                                        {articleData?.excerpt || 'Click anywhere on this card to read the full article.'}
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
                    ) : (
                        /* Mode 2: Long-Form Article Package (Copy & Paste to LinkedIn Pulse) */
                        <div className="space-y-6 animate-in fade-in duration-200">
                            {/* Info Banner */}
                            <div className="p-4 rounded-xl border border-blue-500/20 bg-blue-500/5 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                                <div className="space-y-1">
                                    <h4 className="text-sm font-semibold text-foreground flex items-center gap-2">
                                        <BookOpen className="w-4 h-4 text-[#0A66C2]" />
                                        LinkedIn Long-Form Article & Newsletter Kit
                                    </h4>
                                    <p className="text-xs text-muted-foreground max-w-xl">
                                        LinkedIn's API does not support publishing native Pulse articles directly. Use this ready-to-paste package to publish as a native Article or Newsletter on your <strong>Aichieve.net Company Page</strong>.
                                    </p>
                                </div>
                                <a
                                    href="https://www.linkedin.com/post/new/"
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="flex items-center gap-1.5 px-4 py-2 bg-[#0A66C2] hover:bg-[#084e96] text-white text-xs font-semibold rounded-xl shadow-sm whitespace-nowrap transition"
                                >
                                    Open LinkedIn Editor
                                    <ExternalLink className="w-3.5 h-3.5" />
                                </a>
                            </div>

                            {/* Section 1: Cover Image (Top of Article) */}
                            <div className="border border-border rounded-xl p-4 bg-card space-y-3">
                                <div className="flex items-center justify-between">
                                    <div>
                                        <h5 className="text-sm font-semibold text-foreground flex items-center gap-2">
                                            <ImageIcon className="w-4 h-4 text-[#0A66C2]" />
                                            1. Cover Image (Banner)
                                        </h5>
                                        <p className="text-xs text-muted-foreground">
                                            Include at the top of your LinkedIn article (16:9 ratio). Increases click-through rates by ~3x.
                                        </p>
                                    </div>
                                    {featuredImg && (
                                        <div className="flex items-center gap-2">
                                            <button
                                                type="button"
                                                onClick={() => handleCopy(featuredImg, 'cover_url')}
                                                className="flex items-center gap-1.5 px-3 py-1.5 border border-border bg-background hover:bg-muted text-xs font-medium rounded-lg transition"
                                            >
                                                {copiedField === 'cover_url' ? <Check className="w-3.5 h-3.5 text-emerald-500" /> : <Copy className="w-3.5 h-3.5" />}
                                                {copiedField === 'cover_url' ? 'Copied URL!' : 'Copy URL'}
                                            </button>
                                            <a
                                                href={featuredImg}
                                                download="linkedin-cover-image.png"
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                className="flex items-center gap-1.5 px-3 py-1.5 bg-[#0A66C2] hover:bg-[#084e96] text-white text-xs font-medium rounded-lg transition shadow-sm"
                                            >
                                                <Download className="w-3.5 h-3.5" />
                                                Download
                                            </a>
                                        </div>
                                    )}
                                </div>
                                {featuredImg ? (
                                    <div className="relative rounded-lg overflow-hidden border border-border max-h-48 bg-muted/20">
                                        <img src={featuredImg} alt="Cover" className="w-full h-48 object-cover" />
                                    </div>
                                ) : (
                                    <div className="p-4 border border-dashed border-border rounded-lg text-center text-xs text-muted-foreground">
                                        No featured image set for this article yet.
                                    </div>
                                )}
                            </div>

                            {/* Section 2: Headline & Subtitle */}
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div className="border border-border rounded-xl p-4 bg-card space-y-2">
                                    <div className="flex items-center justify-between">
                                        <h5 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                                            2. Article Headline (H1)
                                        </h5>
                                        <button
                                            type="button"
                                            onClick={() => handleCopy(articleData?.Title || articleData?.title || '', 'headline')}
                                            className="flex items-center gap-1 text-xs text-primary font-medium hover:underline"
                                        >
                                            {copiedField === 'headline' ? <Check className="w-3.5 h-3.5 text-emerald-500" /> : <Copy className="w-3.5 h-3.5" />}
                                            {copiedField === 'headline' ? 'Copied' : 'Copy'}
                                        </button>
                                    </div>
                                    <p className="text-sm font-semibold text-foreground bg-muted/30 p-2.5 rounded-lg border border-border">
                                        {articleData?.Title || articleData?.title || 'Untitled Article'}
                                    </p>
                                </div>

                                <div className="border border-border rounded-xl p-4 bg-card space-y-2">
                                    <div className="flex items-center justify-between">
                                        <h5 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                                            3. Subtitle / Thesis
                                        </h5>
                                        <button
                                            type="button"
                                            onClick={() => handleCopy(articleData?.deck || articleData?.excerpt || articleData?.thesis || '', 'subtitle')}
                                            className="flex items-center gap-1 text-xs text-primary font-medium hover:underline"
                                        >
                                            {copiedField === 'subtitle' ? <Check className="w-3.5 h-3.5 text-emerald-500" /> : <Copy className="w-3.5 h-3.5" />}
                                            {copiedField === 'subtitle' ? 'Copied' : 'Copy'}
                                        </button>
                                    </div>
                                    <p className="text-xs text-muted-foreground bg-muted/30 p-2.5 rounded-lg border border-border line-clamp-3">
                                        {articleData?.deck || articleData?.excerpt || articleData?.thesis || 'No subtitle provided.'}
                                    </p>
                                </div>
                            </div>

                            {/* Section 3: Full Article Body */}
                            <div className="border border-border rounded-xl p-4 bg-card space-y-3">
                                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                                    <div>
                                        <h5 className="text-sm font-semibold text-foreground flex items-center gap-2">
                                            <FileText className="w-4 h-4 text-[#0A66C2]" />
                                            4. Full Article Body
                                        </h5>
                                        <p className="text-xs text-muted-foreground">
                                            Copy as formatted text to preserve headings, bolding, and bullet lists in LinkedIn.
                                        </p>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <button
                                            type="button"
                                            onClick={() => handleCopyRichText(articleData?.htmlArticle || articleData?.content || articleData?.articleText || '')}
                                            className="flex items-center gap-1.5 px-3 py-1.5 bg-[#0A66C2] hover:bg-[#084e96] text-white text-xs font-semibold rounded-lg shadow-sm transition"
                                        >
                                            {copiedField === 'rich_body' ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
                                            {copiedField === 'rich_body' ? 'Copied Formatted Text!' : 'Copy Formatted Text'}
                                        </button>
                                        <button
                                            type="button"
                                            onClick={() => handleCopy((articleData?.htmlArticle || articleData?.content || '').replace(/<[^>]+>/g, ''), 'plain_body')}
                                            className="flex items-center gap-1.5 px-3 py-1.5 border border-border bg-background hover:bg-muted text-xs font-medium rounded-lg transition"
                                        >
                                            {copiedField === 'plain_body' ? <Check className="w-3.5 h-3.5 text-emerald-500" /> : <Copy className="w-3.5 h-3.5" />}
                                            {copiedField === 'plain_body' ? 'Copied Plain Text!' : 'Copy Plain Text'}
                                        </button>
                                    </div>
                                </div>
                                <div className="p-3 bg-muted/20 border border-border rounded-lg max-h-60 overflow-y-auto text-xs font-mono text-muted-foreground whitespace-pre-wrap">
                                    {(articleData?.htmlArticle || articleData?.content || articleData?.articleText || '').slice(0, 1500)}...
                                </div>
                            </div>

                            {/* Section 4: Inline Images & Infographics */}
                            <div className="border border-border rounded-xl p-4 bg-card space-y-3">
                                <div>
                                    <h5 className="text-sm font-semibold text-foreground flex items-center gap-2">
                                        <ImageIcon className="w-4 h-4 text-[#0A66C2]" />
                                        5. Inline Images & Infographics
                                    </h5>
                                    <p className="text-xs text-muted-foreground">
                                        <strong>Should you include images? YES!</strong> Visual breaks (infographics, diagrams, flowcharts) every 300–400 words improve reader completion rate significantly on LinkedIn. Click below to download and upload them inside the article body:
                                    </p>
                                </div>
                                {inlineImages.length > 0 ? (
                                    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
                                        {inlineImages.map((imgUrl, idx) => (
                                            <div key={idx} className="border border-border rounded-lg p-2 bg-muted/20 space-y-2">
                                                <img src={imgUrl} alt={`Inline ${idx + 1}`} className="w-full h-28 object-cover rounded" />
                                                <div className="flex items-center justify-between text-[11px]">
                                                    <span className="text-muted-foreground truncate max-w-[120px]">Image {idx + 1}</span>
                                                    <a
                                                        href={imgUrl}
                                                        download={`article-inline-${idx + 1}.png`}
                                                        target="_blank"
                                                        rel="noopener noreferrer"
                                                        className="text-primary hover:underline flex items-center gap-1 font-medium"
                                                    >
                                                        <Download className="w-3 h-3" /> Download
                                                    </a>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                ) : (
                                    <div className="p-3 bg-muted/20 border border-border rounded-lg text-xs text-muted-foreground">
                                        No inline infographics attached yet. You can insert your charts or diagrams directly into the LinkedIn editor.
                                    </div>
                                )}
                            </div>

                            {/* Section 5: Closing Call-to-Action */}
                            <div className="border border-border rounded-xl p-4 bg-card space-y-2">
                                <div className="flex items-center justify-between">
                                    <h5 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                                        6. Recommended Footer CTA (Aichieve.net)
                                    </h5>
                                    <button
                                        type="button"
                                        onClick={() => handleCopy(ctaSnippet, 'cta')}
                                        className="flex items-center gap-1 text-xs text-primary font-medium hover:underline"
                                    >
                                        {copiedField === 'cta' ? <Check className="w-3.5 h-3.5 text-emerald-500" /> : <Copy className="w-3.5 h-3.5" />}
                                        {copiedField === 'cta' ? 'Copied CTA' : 'Copy CTA'}
                                    </button>
                                </div>
                                <div className="p-2.5 bg-muted/30 border border-border rounded-lg text-xs text-foreground whitespace-pre-line">
                                    {ctaSnippet}
                                </div>
                            </div>
                        </div>
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

                        {activeModalTab === 'feed_post' ? (
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
                        ) : (
                            <a
                                href="https://www.linkedin.com/post/new/"
                                target="_blank"
                                rel="noopener noreferrer"
                                className="flex items-center gap-2 px-6 py-2.5 bg-[#0A66C2] hover:bg-[#084e96] text-white text-sm font-medium rounded-xl transition shadow-md"
                            >
                                Open LinkedIn Article Editor
                                <ExternalLink className="w-4 h-4" />
                            </a>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};
