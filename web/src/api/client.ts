import axios from 'axios';

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';

export const apiClient = axios.create({
  baseURL: apiBaseUrl,
  timeout: 20000
});

export interface ComparePayload {
  message: string;
  max_new?: number;
  k?: number;
  temp?: number;
  ema_alpha?: number;
}

export interface CompareResponse {
  input: string;
  emo_output: string;
  emo_alignment: number;
  chatgpt_output: string | null;
  chatgpt_model: string | null;
  warnings: string[];
}

export async function fetchComparison(payload: ComparePayload): Promise<CompareResponse> {
  const { data } = await apiClient.post<CompareResponse>('/compare', payload);
  return data;
}
