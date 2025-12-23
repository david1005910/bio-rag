// ArXiv hooks
export {
  arxivKeys,
  useArxivSearch,
  useArxivSearchMutation,
  useArxivPaper,
} from './useArxiv';

// PubMed hooks
export {
  pubmedKeys,
  usePubMedSearch,
  usePubMedSearchMutation,
  usePubMedPaper,
} from './usePubMed';

// Analytics hooks
export {
  analyticsKeys,
  useQuickTrend,
  useAnalyzeTrends,
  useAnalyzeTrendsFromPapers,
} from './useAnalytics';

// Documents hooks
export {
  useDownloadPapers,
  useExtractText,
  useExtractFromUpload,
  useSummarizePaper,
  useSummarizeBatch,
} from './useDocuments';

// i18n hooks
export {
  i18nKeys,
  useDetectLanguage,
  useTranslate,
  useTranslateQuery,
  useMedicalTerms,
  useSupportedLanguages,
} from './useI18n';

// Hybrid Search hooks
export {
  hybridKeys,
  useHybridStats,
  useIndexPapers,
  useHybridSearch,
  useHybridSearchMutation,
  useClearIndex,
} from './useHybridSearch';
