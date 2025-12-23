// User types
export interface User {
  user_id: string;
  email: string;
  name: string;
  organization?: string;
  research_fields?: string[];
  interests?: string[];
  subscription_tier: string;
  created_at: string;
  last_login?: string;
  is_active: boolean;
}

export interface Token {
  access_token: string;
  refresh_token: string;
  token_type: string;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface RegisterRequest {
  email: string;
  password: string;
  name: string;
  organization?: string;
  research_fields?: string[];
  interests?: string[];
}

// Search types
export interface SearchFilters {
  year_from?: number;
  year_to?: number;
  journals?: string[];
  authors?: string[];
}

export interface SearchQuery {
  query: string;
  filters?: SearchFilters;
  limit?: number;
  offset?: number;
}

export interface PaperSummary {
  pmid: string;
  title: string;
  authors: string[];
  journal: string;
  publication_date?: string;
  abstract?: string;
  relevance_score: number;
  pdf_url?: string;
}

export interface SearchResult {
  results: PaperSummary[];
  total: number;
  query_time_ms: number;
}

// Chat types
export interface Citation {
  pmid: string;
  title: string;
  relevance_score: number;
  snippet: string;
}

export interface ChatQuery {
  session_id?: string;
  query: string;
}

export interface ChatResponse {
  session_id: string;
  message_id: string;
  answer: string;
  citations: Citation[];
  confidence_score?: number;
  latency_ms: number;
}

export interface ChatMessage {
  message_id: string;
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
  timestamp: string;
}

export interface ChatSession {
  session_id: string;
  title?: string;
  message_count: number;
  created_at: string;
  updated_at: string;
}

export interface ChatSessionDetail {
  session_id: string;
  title?: string;
  messages: ChatMessage[];
  created_at: string;
  updated_at: string;
}

// ==================== ArXiv Types ====================
export interface ArXivPaper {
  arxiv_id: string;
  title: string;
  abstract: string;
  authors: string[];
  categories: string[];
  published: string;
  updated: string;
  pdf_url: string;
  comment?: string;
  journal_ref?: string;
}

export interface ArXivSearchQuery {
  query: string;
  max_results?: number;
  sort_by?: 'relevance' | 'lastUpdatedDate' | 'submittedDate';
  sort_order?: 'ascending' | 'descending';
}

export interface ArXivSearchResult {
  results: ArXivPaper[];
  total: number;
  query: string;
}

// ==================== Analytics Types ====================
export interface TrendAnalysisRequest {
  keyword: string;
  max_papers?: number;
  source?: 'pubmed' | 'arxiv' | 'both';
  include_ai_summary?: boolean;
}

export interface TrendAnalysisResponse {
  keyword: string;
  translated_keyword?: string;
  original_language: string;
  total_papers: number;
  year_trend?: {
    years: number[];
    counts: number[];
  };
  key_terms?: {
    terms: string[];
    frequencies: number[];
  };
  emerging_topics?: Array<{
    topic: string;
    growth_rate: number;
    recent_count: number;
  }>;
  content_summary?: {
    main_themes: string[];
    key_findings: string[];
  };
  report?: string;
}

export interface QuickTrendResponse {
  keyword: string;
  total_papers: number;
  years: number[];
  counts: number[];
  top_terms: string[];
}

// ==================== Documents Types ====================
export interface DownloadRequest {
  papers: Array<{
    id: string;
    title: string;
    pdf_url?: string;
    abstract?: string;
  }>;
}

export interface DownloadResult {
  paper_id: string;
  title: string;
  filepath?: string;
  success: boolean;
  error?: string;
}

export interface DownloadResponse {
  downloaded: DownloadResult[];
  total: number;
  success_count: number;
}

export interface ExtractRequest {
  filepaths: string[];
}

export interface ExtractedDocument {
  source: string;
  filepath: string;
  text: string;
  text_length: number;
  success: boolean;
  error?: string;
}

export interface ExtractResponse {
  documents: ExtractedDocument[];
  total: number;
  success_count: number;
}

export interface SummarizeRequest {
  paper: {
    id: string;
    title: string;
    abstract?: string;
  };
  content?: string;
  language?: 'en' | 'ko';
}

export interface SummaryResponse {
  paper_id: string;
  title: string;
  summary: string;
  language: string;
  success: boolean;
  error?: string;
}

// ==================== i18n Types ====================
export interface DetectLanguageResponse {
  text: string;
  language: 'ko' | 'en';
  language_name: string;
}

export interface TranslateResponse {
  original: string;
  translated: string;
  source_language: string;
  target_language: string;
  method: 'dictionary' | 'ai' | 'none';
}

export interface MedicalTerm {
  korean: string;
  english: string;
}

export interface QueryTranslationResponse {
  original: string;
  translated: string;
  language: string;
  is_translated: boolean;
}

export interface SupportedLanguage {
  code: string;
  name: string;
  name_native: string;
}

export interface SupportedLanguagesResponse {
  languages: SupportedLanguage[];
  translation_direction: string;
  medical_terms_count: number;
}
