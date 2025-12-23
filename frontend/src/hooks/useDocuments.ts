import { useMutation } from '@tanstack/react-query';
import { documentsApi } from '../api/documents';
import type { DownloadRequest, ExtractRequest, SummarizeRequest } from '../types';

/**
 * 논문 PDF 다운로드 (mutation)
 */
export function useDownloadPapers() {
  return useMutation({
    mutationFn: (request: DownloadRequest) => documentsApi.downloadPapers(request),
  });
}

/**
 * 파일에서 텍스트 추출 (mutation)
 */
export function useExtractText() {
  return useMutation({
    mutationFn: (request: ExtractRequest) => documentsApi.extractText(request),
  });
}

/**
 * 업로드 파일에서 텍스트 추출 (mutation)
 */
export function useExtractFromUpload() {
  return useMutation({
    mutationFn: (file: File) => documentsApi.extractFromUpload(file),
  });
}

/**
 * 논문 요약 (mutation)
 */
export function useSummarizePaper() {
  return useMutation({
    mutationFn: (request: SummarizeRequest) => documentsApi.summarizePaper(request),
  });
}

/**
 * 여러 논문 일괄 요약 (mutation)
 */
export function useSummarizeBatch() {
  return useMutation({
    mutationFn: ({
      papers,
      language = 'en',
    }: {
      papers: Array<{ id: string; title: string; abstract?: string }>;
      language?: 'en' | 'ko';
    }) => documentsApi.summarizeBatch(papers, language),
  });
}
