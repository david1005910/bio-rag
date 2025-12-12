import api from './client';
import type { ChatQuery, ChatResponse, ChatSession, ChatSessionDetail } from '../types';

export const chatApi = {
  async createSession(title?: string): Promise<ChatSession> {
    const response = await api.post<ChatSession>('/chat/sessions', { title });
    return response.data;
  },

  async listSessions(limit: number = 20, offset: number = 0): Promise<ChatSession[]> {
    const response = await api.get<ChatSession[]>('/chat/sessions', {
      params: { limit, offset },
    });
    return response.data;
  },

  async getSession(sessionId: string): Promise<ChatSessionDetail> {
    const response = await api.get<ChatSessionDetail>(`/chat/sessions/${sessionId}`);
    return response.data;
  },

  async deleteSession(sessionId: string): Promise<void> {
    await api.delete(`/chat/sessions/${sessionId}`);
  },

  async query(chatQuery: ChatQuery): Promise<ChatResponse> {
    const response = await api.post<ChatResponse>('/chat/query', chatQuery);
    return response.data;
  },

  // Streaming query using EventSource
  queryStream(chatQuery: ChatQuery, onChunk: (chunk: string) => void, onDone: () => void): () => void {
    const controller = new AbortController();

    fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1'}/chat/query/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
      },
      body: JSON.stringify(chatQuery),
      signal: controller.signal,
    })
      .then(async (response) => {
        const reader = response.body?.getReader();
        if (!reader) return;

        const decoder = new TextDecoder();

        let done = false;
        while (!done) {
          const result = await reader.read();
          done = result.done;
          if (done) break;
          const value = result.value;

          const text = decoder.decode(value);
          const lines = text.split('\n');

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6);
              if (data === '[DONE]') {
                onDone();
              } else {
                onChunk(data);
              }
            }
          }
        }
      })
      .catch((error) => {
        if (error.name !== 'AbortError') {
          console.error('Stream error:', error);
        }
      });

    return () => controller.abort();
  },
};
