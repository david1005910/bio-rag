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
