import { useQuery, useMutation } from '@tanstack/react-query';
import { i18nApi } from '../api/i18n';

export const i18nKeys = {
  all: ['i18n'] as const,
  detect: (text: string) => [...i18nKeys.all, 'detect', text] as const,
  translate: (text: string) => [...i18nKeys.all, 'translate', text] as const,
  medicalTerms: (search?: string) => [...i18nKeys.all, 'medical-terms', search] as const,
  supportedLanguages: () => [...i18nKeys.all, 'supported-languages'] as const,
};

/**
 * 언어 감지 훅
 */
export function useDetectLanguage(text: string, options?: { enabled?: boolean }) {
  return useQuery({
    queryKey: i18nKeys.detect(text),
    queryFn: () => i18nApi.detectLanguage(text),
    enabled: options?.enabled !== false && text.length > 0,
  });
}

/**
 * 번역 훅
 */
export function useTranslate(text: string, options?: { enabled?: boolean; useAi?: boolean }) {
  return useQuery({
    queryKey: i18nKeys.translate(text),
    queryFn: () => i18nApi.translate(text, options?.useAi ?? false),
    enabled: options?.enabled !== false && text.length > 0,
  });
}

/**
 * 검색 쿼리 번역 (mutation)
 */
export function useTranslateQuery() {
  return useMutation({
    mutationFn: (query: string) => i18nApi.translateQuery(query),
  });
}

/**
 * 의학 용어 사전 조회 훅
 */
export function useMedicalTerms(search?: string, options?: { enabled?: boolean; limit?: number }) {
  return useQuery({
    queryKey: i18nKeys.medicalTerms(search),
    queryFn: () => i18nApi.getMedicalTerms(search, options?.limit ?? 50),
    enabled: options?.enabled !== false,
  });
}

/**
 * 지원 언어 목록 훅
 */
export function useSupportedLanguages() {
  return useQuery({
    queryKey: i18nKeys.supportedLanguages(),
    queryFn: () => i18nApi.getSupportedLanguages(),
    staleTime: Infinity, // 언어 목록은 변하지 않음
  });
}
