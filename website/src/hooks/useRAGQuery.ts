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
// - Local development: http://localhost:8000/api/query (external backend)
// - Production: /api/query (Vercel serverless function)
const getApiUrl = () => {
  // For local development, use the external backend
  // For production, use the Vercel API route
  const isLocalhost = typeof window !== 'undefined'
    ? (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
    : false;

<<<<<<< HEAD
  return isLocalhost ? 'http://localhost:8000/api/query' : '/api/query';
=======
  return isLocalhost ? 'http://localhost:8000/query' : '/api/query';
>>>>>>> 001-rag-chatbot
};

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
      const apiUrl = getApiUrl();
      const response = await fetch(apiUrl, {
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
