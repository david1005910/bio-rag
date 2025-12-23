import api from './client';
import type {
  DetectLanguageResponse,
  TranslateResponse,
  MedicalTerm,
  QueryTranslationResponse,
  SupportedLanguagesResponse,
} from '../types';

export const i18nApi = {
  /**
   * 텍스트 언어 감지
   */
  async detectLanguage(text: string): Promise<DetectLanguageResponse> {
    const response = await api.get<DetectLanguageResponse>('/i18n/detect', {
      params: { text },
    });
    return response.data;
  },

  /**
   * 한국어 → 영어 번역
   */
  async translate(text: string, useAi: boolean = false): Promise<TranslateResponse> {
    const response = await api.get<TranslateResponse>('/i18n/translate', {
      params: { text, use_ai: useAi },
    });
    return response.data;
  },

  /**
   * 검색 쿼리 번역
   */
  async translateQuery(query: string): Promise<QueryTranslationResponse> {
    const response = await api.post<QueryTranslationResponse>('/i18n/translate-query', {
      query,
    });
    return response.data;
  },

  /**
   * 의학 용어 사전 조회
   */
  async getMedicalTerms(search?: string, limit: number = 50): Promise<MedicalTerm[]> {
    const response = await api.get<MedicalTerm[]>('/i18n/medical-terms', {
      params: { search, limit },
    });
    return response.data;
  },

  /**
   * 지원 언어 목록
   */
  async getSupportedLanguages(): Promise<SupportedLanguagesResponse> {
    const response = await api.get<SupportedLanguagesResponse>('/i18n/supported-languages');
    return response.data;
  },
};
