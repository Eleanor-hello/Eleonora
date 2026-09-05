export interface Message {
  id: number;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  sessionId?: number;
}

export interface ChatSession {
  id: number;
  title: string;
  createdAt: string;
  updatedAt: string;
}

export interface LogEntry {
  id: string;
  timestamp: string;
  level: 'INFO' | 'WARNING' | 'ERROR';
  component: string;
  message: string;
}

export interface PersonalFact {
  id: string;
  fact: string;
  category?: string;
  createdAt: string;
}

export interface StressOverride {
  bare: string;
  marked: string;
  addedAt?: string;
}
