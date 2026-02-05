export interface MetricExplanation {
    title: string;
    origin_type: 'AI Estimate' | 'Real Data' | 'Hybrid (AI + Data)' | 'Real Data (Via DataForSEO)' | 'Hybrid (Real Data Priority)' | 'Real Data (Priority)';
    meaning: string;
    calculation: string;
    unit: string;
    range: string;
}

export const METRIC_EXPLANATIONS: Record<string, MetricExplanation> = {
    search_volume: {
        title: "Search Volume",
        origin_type: "Real Data (Via DataForSEO)",
        meaning: "The average number of monthly searches for the primary keyword.",
        calculation: "Directly fetched from the DataForSEO API (e.g., 5400). If real data is missing, the AI looks at the topic popularity and estimates a numeric value.",
        unit: "Count (Monthly Searches)",
        range: "0 to ∞ (e.g., 5,400)"
    },
    competition: {
        title: "Competition",
        origin_type: "Hybrid (Real Data Priority)",
        meaning: "A measure of how many other websites are targeting this keyword (0-100).",
        calculation: "Directly mapped from the Average Difficulty score found in DataForSEO. If real data is missing, the AI estimates it based on niche saturation.",
        unit: "Difficulty Index",
        range: "0 to 100 (Lower is easier)"
    },
    seo_score: {
        title: "SEO Score",
        origin_type: "AI Estimate",
        meaning: "How well the content structure and keywords match search user intent.",
        calculation: "The AI analyzes your title + keywords against SEO best practices. It scores high (80+) if the topic is highly relevant to the provided high-volume keywords.",
        unit: "Quality Score",
        range: "0 to 100 (Higher is better)"
    },
    traffic_potential: {
        title: "Traffic Potential",
        origin_type: "Hybrid (AI + Data)",
        meaning: "Estimate of capturing monthly visitors.",
        calculation: "The AI looks at the actual Search Volume. It then predicts your \"Capture Rate\" based on the topic's virality. High volume + viral topic = High Score.",
        unit: "Potential Score",
        range: "0 to 100 (Higher is better)"
    },
    difficulty: {
        title: "Key. Difficulty",
        origin_type: "Real Data (Priority)",
        meaning: "How hard it is to rank on Google (Synonymous with Competition).",
        calculation: "1. If DataForSEO data exists: Uses the numeric value (0-100) directly.\n2. If no data: AI estimates based on competitiveness.",
        unit: "Difficulty Index",
        range: "0 to 100 (Lower is easier)"
    },
    viral_score: {
        title: "Viral Score",
        origin_type: "AI Estimate",
        meaning: "The likelihood of this content being shared on social media.",
        calculation: "Mapped directly from the traffic_potential_score (since high viral potential leads to high traffic).",
        unit: "Probability Score",
        range: "0 to 100 (Higher is more likely)"
    },
    audience_align: {
        title: "Audience Alignment",
        origin_type: "AI Estimate",
        meaning: "Fit for your specific target audience (e.g., \"Beginners\" vs \"Experts\").",
        calculation: "The AI compares the complexity of the generated idea against your defined target_audience. Example: A \"Basic Guide\" aligns perfectly (100%) with \"Beginners\".",
        unit: "Relevance Score",
        range: "0 to 100 (100% = Perfect fit)"
    },
    feasibility: {
        title: "Feasibility",
        origin_type: "AI Estimate",
        meaning: "Ease of creation.",
        calculation: "The AI assesses the resources needed.\nHigh Score (80+): Easy to write (e.g., \"Opinion piece\").\nLow Score (<50): Hard to write (e.g., \"Original Research Study\" needing data collection).",
        unit: "Ease Score",
        range: "0 to 100 (100 = Very Easy to create)"
    },
    impact: {
        title: "Business Impact",
        origin_type: "AI Estimate",
        meaning: "Potential for business revenue/conversions.",
        calculation: "AI evaluates commercial intent (CPC).\nHigh Score: Topics with high CPC or transactional intent (e.g., \"Best tools for X\").",
        unit: "Value Score",
        range: "0 to 100 (100 = High Revenue Impact)"
    },
    reading_time: {
        title: "Reading Time",
        origin_type: "AI Estimate",
        meaning: "Calculated: Based on the estimated word count (~200 words per minute).",
        calculation: "Based on the estimated word count (~200 words per minute).",
        unit: "Time",
        range: "Minutes (e.g., 5 min)"
    }
};
