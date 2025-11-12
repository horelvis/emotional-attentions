export interface ConversationTurn {
  id: string;
  prompt: string;
  emoResponse: string;
  alignment: number;
  chatgptResponse: string | null;
  chatgptModel: string | null;
  warnings: string[];
  createdAt: string;
}
