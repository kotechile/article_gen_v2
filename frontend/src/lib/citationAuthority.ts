export interface CitationDomainMeta {
  domain: string;
  domainFrequency: number;
  domainRank: number;
  authorityScore: number;
}

const MULTIPART_SUFFIXES = new Set([
  'co.uk',
  'org.uk',
  'gov.uk',
  'ac.uk',
  'co.jp',
  'com.au',
  'net.au',
  'org.au',
  'co.nz',
  'com.br',
]);

export const extractRootDomain = (value?: string | null): string => {
  if (!value) return 'unknown';

  let hostname = '';
  try {
    const normalized = value.match(/^https?:\/\//i) ? value : `https://${value}`;
    hostname = new URL(normalized).hostname.toLowerCase();
  } catch {
    return 'unknown';
  }

  if (!hostname) return 'unknown';
  if (hostname.startsWith('www.')) hostname = hostname.slice(4);

  const parts = hostname.split('.').filter(Boolean);
  if (parts.length <= 2) return hostname;

  const lastTwo = `${parts[parts.length - 2]}.${parts[parts.length - 1]}`;
  if (MULTIPART_SUFFIXES.has(lastTwo) && parts.length >= 3) {
    return `${parts[parts.length - 3]}.${lastTwo}`;
  }

  return lastTwo;
};

export const rankCitationDomains = (citations: any[]): CitationDomainMeta[] => {
  const counts = new Map<string, number>();
  const domains = citations.map((citation) => {
    const explicitDomain = citation?.domain || citation?.root_domain || citation?.source_domain;
    const domain = explicitDomain ? String(explicitDomain).toLowerCase() : extractRootDomain(citation?.url);
    counts.set(domain, (counts.get(domain) || 0) + 1);
    return domain;
  });

  const rankedDomains = Array.from(counts.entries())
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .map(([domain], index) => ({ domain, rank: index + 1 }));

  const rankMap = new Map(rankedDomains.map((item) => [item.domain, item.rank]));
  const maxFrequency = Math.max(...Array.from(counts.values()), 1);

  return domains.map((domain) => {
    const frequency = counts.get(domain) || 0;
    const rank = rankMap.get(domain) || rankedDomains.length;
    return {
      domain,
      domainFrequency: frequency,
      domainRank: rank,
      authorityScore: Number((frequency / maxFrequency).toFixed(4)),
    };
  });
};

export const topDomains = (citations: any[], limit: number): string[] => {
  if (!citations.length || limit <= 0) return [];
  const domainMeta = rankCitationDomains(citations);
  const counts = new Map<string, number>();
  domainMeta.forEach((meta) => {
    counts.set(meta.domain, (counts.get(meta.domain) || 0) + 1);
  });

  return Array.from(counts.entries())
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, limit)
    .map(([domain]) => domain);
};
