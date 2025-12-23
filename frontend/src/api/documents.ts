import api from './client';
import type {
  DownloadRequest,
  DownloadResponse,
  ExtractRequest,
  ExtractResponse,
  ExtractedDocument,
  SummarizeRequest,
  SummaryResponse,
} from '../types';

export const documentsApi = {
  /**
   * 논문 PDF 다운로드
   */
  async downloadPapers(request: DownloadRequest): Promise<DownloadResponse> {
    const response = await api.post<DownloadResponse>('/documents/download', request);
    return response.data;
  },

  /**
   * 파일에서 텍스트 추출
   */
  async extractText(request: ExtractRequest): Promise<ExtractResponse> {
    const response = await api.post<ExtractResponse>('/documents/extract', request);
    return response.data;
  },

  /**
   * 업로드된 파일에서 텍스트 추출
   */
  async extractFromUpload(file: File): Promise<ExtractedDocument> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await api.post<ExtractedDocument>('/documents/extract-upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  /**
   * 논문 요약
   */
  async summarizePaper(request: SummarizeRequest): Promise<SummaryResponse> {
    const response = await api.post<SummaryResponse>('/documents/summarize', request);
    return response.data;
  },

  /**
   * 여러 논문 일괄 요약
   */
  async summarizeBatch(
    papers: Array<{ id: string; title: string; abstract?: string }>,
    language: 'en' | 'ko' = 'en'
  ): Promise<SummaryResponse[]> {
    const response = await api.post<SummaryResponse[]>(
      '/documents/summarize-batch',
      papers,
      { params: { language } }
    );
    return response.data;
  },
};
