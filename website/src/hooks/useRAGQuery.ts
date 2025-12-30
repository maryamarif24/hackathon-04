import { useState, useCallback } from 'react';

interface Source {
  chunk_id: string;
  chapter_id: number;
  section_id: string;
  section_title: string;
  preview_text: string;
  relevance_score: number;
}

interface EducationalMetadata {
  questionType: 'definition' | 'explanation' | 'general';
  complexity: 'simple' | 'moderate' | 'complex';
  estimatedWordCount: string;
  needsStructure: boolean;
}

interface RAGResponse {
  answer: string;
  sources: Source[];
  chapter_id: number | null;
  query_time_ms: number;
  educational_metadata?: EducationalMetadata;
}

interface UseRAGQueryReturn {
  loading: boolean;
  answer: string;
  sources: Source[];
  error: string | null;
  query: (question: string, context?: string, useContextOnly?: boolean, chapterId?: number | null) => Promise<void>;
  clear: () => void;
}

// API endpoint configuration
// - Development: localhost:8001 (for local testing)
// - Production: VERCEL_URL or fallback to /api/query (Vercel serverless)
// - Or use VITE_RAG_API_URL for Hugging Face Space
const getApiUrl = () => {
  // If custom API URL is set (e.g., Hugging Face Space)
  const customUrl = process.env.VITE_RAG_API_URL || process.env.REACT_APP_RAG_API_URL;
  if (customUrl) {
    return customUrl;
  }

  // Development: local backend
  if (process.env.NODE_ENV === 'development') {
    return 'http://localhost:8001/api/query';
  }

  // Production: use /api/query (Vercel) or external API
  return process.env.VERCEL_URL ? `https://${process.env.VERCEL_URL}/api/query` : '/api/query';
};

const API_URL = getApiUrl();

export function useRAGQuery(): UseRAGQueryReturn {
  const [loading, setLoading] = useState(false);
  const [answer, setAnswer] = useState('');
  const [sources, setSources] = useState<Source[]>([]);
  const [error, setError] = useState<string | null>(null);

  const query = useCallback(async (
    question: string,
    context?: string,
    useContextOnly?: boolean,
    chapterId?: number | null
  ): Promise<void> => {
    if (!question.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question,
          context: context || '',
          use_context_only: useContextOnly || false,
          chapter_id: chapterId ?? null,
        }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data: RAGResponse = await response.json();
      setAnswer(data.answer);
      setSources(data.sources || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to query');
      setAnswer('');
      setSources([]);
    } finally {
      setLoading(false);
    }
  }, []);

  const clear = useCallback(() => {
    setAnswer('');
    setSources([]);
    setError(null);
  }, []);

  return {
    loading,
    answer,
    sources,
    error,
    query,
    clear,
  };
}
