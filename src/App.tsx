import React, { useState, useEffect, useRef } from 'react';
import { Message, ChatSession, LogEntry, PersonalFact, StressOverride } from './types';
import { ChatWidget } from './components/ChatWidget';
import { LogPanel } from './components/LogPanel';
import { SessionsModal } from './components/SessionsModal';
import { MemoriesModal } from './components/MemoriesModal';
import { StressModal } from './components/StressModal';
import { sanitizeForTts, yoficate, applyStress } from './lib/tts_preprocessor';
import { Brain, BookOpen, Send, Plus, History, FileText, Volume2, VolumeX } from 'lucide-react';

export const App: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<number | null>(null);
  const [inputText, setInputText] = useState('');
  const [isThinking, setIsThinking] = useState(false);
  const [statusText, setStatusText] = useState('');

  // Toggles & Modals
  const [isTtsEnabled, setIsTtsEnabled] = useState(true);
  const [isLogsOpen, setIsLogsOpen] = useState(false);
  const [isSessionsModalOpen, setIsSessionsModalOpen] = useState(false);
  const [isMemoriesModalOpen, setIsMemoriesModalOpen] = useState(false);
  const [isStressModalOpen, setIsStressModalOpen] = useState(false);

  // Aux state
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [memories, setMemories] = useState<PersonalFact[]>([]);
  const [stressOverrides, setStressOverrides] = useState<StressOverride[]>([]);

  const inputRef = useRef<HTMLInputElement>(null);

  // Load initial data
  useEffect(() => {
    loadSessions();
    loadLogs();
    loadMemories();
    loadStressOverrides();
  }, []);

  // Sync logs periodically
  useEffect(() => {
    const interval = setInterval(() => {
      loadLogs();
    }, 4000);
    return () => clearInterval(interval);
  }, []);

  // When currentSessionId changes, load messages
  useEffect(() => {
    if (currentSessionId) {
      loadSessionMessages(currentSessionId);
    }
  }, [currentSessionId]);

  const loadSessions = async () => {
    try {
      const res = await fetch('/api/sessions');
      const data: ChatSession[] = await res.json();
      setSessions(data);
      if (data.length > 0 && currentSessionId === null) {
        setCurrentSessionId(data[0].id);
      }
    } catch (err) {
      console.error('Failed to load sessions', err);
    }
  };

  const loadSessionMessages = async (sid: number) => {
    try {
      const res = await fetch(`/api/sessions/${sid}/messages`);
      const data: Message[] = await res.json();
      setMessages(data);
      setStatusText(`История: ${data.length} сообщений`);
    } catch (err) {
      console.error('Failed to load messages', err);
    }
  };

  const loadLogs = async () => {
    try {
      const res = await fetch('/api/logs');
      const data: LogEntry[] = await res.json();
      setLogs(data);
    } catch (err) {
      console.error('Failed to load logs', err);
    }
  };

  const loadMemories = async () => {
    try {
      const res = await fetch('/api/memories');
      const data: PersonalFact[] = await res.json();
      setMemories(data);
    } catch (err) {
      console.error('Failed to load memories', err);
    }
  };

  const loadStressOverrides = async () => {
    try {
      const res = await fetch('/api/stress');
      const data: StressOverride[] = await res.json();
      setStressOverrides(data);
    } catch (err) {
      console.error('Failed to load stress overrides', err);
    }
  };

  // TTS Speech Synthesis in Browser
  const speakText = (text: string) => {
    if (!isTtsEnabled || typeof window === 'undefined' || !('speechSynthesis' in window)) {
      return;
    }

    try {
      window.speechSynthesis.cancel(); // Stop any ongoing speech

      // Preprocess text
      let processed = sanitizeForTts(text);
      processed = yoficate(processed);

      // Apply stress map
      const stressMap: Record<string, string> = {};
      stressOverrides.forEach((s) => {
        stressMap[s.bare] = s.marked;
      });
      processed = applyStress(processed, stressMap);

      // Clean '+' for actual SpeechSynthesis voice engine
      const voiceSpeakable = processed.replace(/\+/g, '');

      const utterance = new SpeechSynthesisUtterance(voiceSpeakable);
      utterance.lang = 'ru-RU';
      utterance.rate = 1.0;
      utterance.pitch = 1.05; // Slightly higher pitch matching Eleonora

      // Find best Russian female voice if available
      const voices = window.speechSynthesis.getVoices();
      const ruVoice = voices.find(
        (v) => v.lang.startsWith('ru') && (v.name.includes('Female') || v.name.includes('xenia') || v.name.includes('Google') || v.name.includes('Milena'))
      ) || voices.find((v) => v.lang.startsWith('ru'));

      if (ruVoice) {
        utterance.voice = ruVoice;
      }

      window.speechSynthesis.speak(utterance);
    } catch (e) {
      console.warn('Speech synthesis error', e);
    }
  };

  const handleSend = async () => {
    const text = inputText.trim();
    if (!text || isThinking) return;

    setInputText('');
    const tempUserMsg: Message = {
      id: Date.now(),
      role: 'user',
      content: text,
      timestamp: new Date().toISOString(),
      sessionId: currentSessionId || undefined,
    };

    setMessages((prev) => [...prev, tempUserMsg]);
    setIsThinking(true);
    setStatusText('Элеонора думает...');

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text,
          sessionId: currentSessionId,
        }),
      });

      const data = await res.json();
      if (data.message) {
        setMessages((prev) => [...prev, data.message]);
        speakText(data.message.content);
      }

      if (data.learnedStress) {
        loadStressOverrides();
      }

      setStatusText('');
      loadSessions();
      loadLogs();
      loadMemories();
    } catch (err: any) {
      console.error('Chat error', err);
      setStatusText(`Ошибка: ${err.message || 'Сбой соединения'}`);
    } finally {
      setIsThinking(false);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  };

  const handleNewChat = async () => {
    try {
      const res = await fetch('/api/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: `Диалог ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}` }),
      });
      const newSession: ChatSession = await res.json();
      setSessions((prev) => [newSession, ...prev]);
      setCurrentSessionId(newSession.id);
      setMessages([]);
      setStatusText('Новый диалог создан');
      loadLogs();
    } catch (err) {
      console.error('Failed to create new session', err);
    }
  };

  const handleDeleteSession = async (id: number) => {
    try {
      await fetch(`/api/sessions/${id}`, { method: 'DELETE' });
      setSessions((prev) => prev.filter((s) => s.id !== id));
      if (currentSessionId === id) {
        const remaining = sessions.filter((s) => s.id !== id);
        if (remaining.length > 0) {
          setCurrentSessionId(remaining[0].id);
        } else {
          handleNewChat();
        }
      }
      setStatusText('Диалог удалён');
      loadLogs();
    } catch (err) {
      console.error('Failed to delete session', err);
    }
  };

  const handleAddStressOverride = async (marked: string) => {
    const res = await fetch('/api/stress', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ marked }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || 'Ошибка добавления');
    }
    await loadStressOverrides();
    await loadLogs();
  };

  const handleDeleteStressOverride = async (bare: string) => {
    await fetch(`/api/stress/${bare}`, { method: 'DELETE' });
    await loadStressOverrides();
    await loadLogs();
  };

  const handleAddMemory = async (fact: string, category?: string) => {
    const res = await fetch('/api/memories', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ fact, category }),
    });
    if (!res.ok) throw new Error('Ошибка добавления');
    await loadMemories();
    await loadLogs();
  };

  const handleDeleteMemory = async (id: string) => {
    await fetch(`/api/memories/${id}`, { method: 'DELETE' });
    await loadMemories();
    await loadLogs();
  };

  const activeSessionTitle =
    sessions.find((s) => s.id === currentSessionId)?.title || 'Eleonora v3';

  return (
    <div
      id="eleonora-main-window"
      className="flex flex-col h-screen w-screen bg-[#0B0B16] text-[#E2E8F0] select-none overflow-hidden font-sans"
    >
      {/* Top Header */}
      <header className="h-12 bg-[#131324] border-b border-[#1E1E4A] px-4 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-7 h-7 rounded-full bg-[#1A1A3E] border border-[#8B5CF6] flex items-center justify-center text-sm shadow-sm shadow-[#8B5CF6]/20">
            💜
          </div>
          <div>
            <h1 className="text-sm font-semibold text-[#E2E8F0] tracking-wide flex items-center gap-2">
              <span>Eleonora v3</span>
              <span className="text-[10px] text-[#06B6D4] font-normal px-1.5 py-0.2 rounded bg-[#06B6D4]/10 border border-[#06B6D4]/30">
                AI Companion
              </span>
            </h1>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            id="open-memories-btn"
            type="button"
            onClick={() => setIsMemoriesModalOpen(true)}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-[#1A1A3E] hover:bg-[#1E1E4A] text-[#E2E8F0] text-xs border border-[#1E1E4A] transition-colors"
            title="Просмотр памяти и личных фактов"
          >
            <Brain className="w-3.5 h-3.5 text-[#06B6D4]" />
            <span className="hidden sm:inline">Память ({memories.length})</span>
          </button>

          <button
            id="open-stress-btn"
            type="button"
            onClick={() => setIsStressModalOpen(true)}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-[#1A1A3E] hover:bg-[#1E1E4A] text-[#E2E8F0] text-xs border border-[#1E1E4A] transition-colors"
            title="База ударений"
          >
            <BookOpen className="w-3.5 h-3.5 text-[#8B5CF6]" />
            <span className="hidden sm:inline">Ударения ({stressOverrides.length})</span>
          </button>
        </div>
      </header>

      {/* Main Body */}
      <main className="flex-1 flex flex-col p-3 md:p-4 gap-2.5 min-h-0 relative">
        {/* Chat Widget */}
        <ChatWidget
          messages={messages}
          isThinking={isThinking}
          onSpeak={speakText}
        />

        {/* Input Row: matching PySide6 MainWindow input_row */}
        <div id="input-row" className="flex items-center gap-2 shrink-0">
          {/* ➕ New Chat */}
          <button
            id="new-chat-btn"
            type="button"
            onClick={handleNewChat}
            className="w-11 h-11 rounded-full bg-[#131324] hover:bg-[#1E1E4A] text-white border border-[#8B5CF6] flex items-center justify-center shrink-0 transition-all shadow-sm hover:scale-105"
            title="Новый диалог"
          >
            <Plus className="w-5 h-5 text-[#E2E8F0]" />
          </button>

          {/* 📜 Dialog History */}
          <button
            id="history-btn"
            type="button"
            onClick={() => setIsSessionsModalOpen(true)}
            className="w-11 h-11 rounded-full bg-[#131324] hover:bg-[#1E1E4A] text-white border border-[#8B5CF6] flex items-center justify-center shrink-0 transition-all shadow-sm hover:scale-105"
            title="История диалогов"
          >
            <History className="w-4 h-4 text-[#E2E8F0]" />
          </button>

          {/* 📋 Logs Toggle */}
          <button
            id="toggle-logs-btn"
            type="button"
            onClick={() => setIsLogsOpen(!isLogsOpen)}
            className={`w-11 h-11 rounded-full border flex items-center justify-center shrink-0 transition-all shadow-sm hover:scale-105 ${
              isLogsOpen
                ? 'bg-[#8B5CF6] text-white border-[#8B5CF6]'
                : 'bg-[#131324] hover:bg-[#1E1E4A] text-[#E2E8F0] border-[#8B5CF6]'
            }`}
            title="Показать/скрыть логи агентов"
          >
            <FileText className="w-4 h-4" />
          </button>

          {/* 🔊/🔇 Voice / TTS Toggle */}
          <button
            id="toggle-mute-btn"
            type="button"
            onClick={() => {
              if (isTtsEnabled) {
                window.speechSynthesis?.cancel();
              }
              setIsTtsEnabled(!isTtsEnabled);
            }}
            className="w-11 h-11 rounded-full bg-[#131324] hover:bg-[#1E1E4A] text-[#E2E8F0] border border-[#8B5CF6] flex items-center justify-center shrink-0 transition-all shadow-sm hover:scale-105"
            title={isTtsEnabled ? 'Выключить озвучку' : 'Включить озвучку'}
          >
            {isTtsEnabled ? (
              <Volume2 className="w-4 h-4 text-[#06B6D4]" />
            ) : (
              <VolumeX className="w-4 h-4 text-[#8B8FA3]" />
            )}
          </button>

          {/* Input field */}
          <div className="flex-1 relative">
            <input
              id="chat-input"
              ref={inputRef}
              type="text"
              placeholder="Напиши сообщение..."
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleSend();
              }}
              disabled={isThinking}
              className="w-full h-11 bg-[#131324] border border-[#8B5CF6] rounded-[20px] px-4 text-sm text-[#E2E8F0] placeholder-[#8B8FA3] focus:outline-none focus:ring-1 focus:ring-[#8B5CF6] transition-all disabled:opacity-50"
            />
          </div>

          {/* ✈ Send button */}
          <button
            id="send-btn"
            type="button"
            onClick={handleSend}
            disabled={isThinking || !inputText.trim()}
            className="w-11 h-11 rounded-full bg-[#8B5CF6] hover:bg-[#7C3AED] disabled:bg-[#1E1E4A] text-white flex items-center justify-center shrink-0 transition-all shadow-md shadow-[#8B5CF6]/20 disabled:shadow-none hover:scale-105 disabled:hover:scale-100"
            title="Отправить"
          >
            <Send className="w-4 h-4 translate-x-[-1px] translate-y-[1px]" />
          </button>
        </div>

        {/* Status bar */}
        <div id="status-bar" className="h-4 flex items-center justify-between text-[11px] font-bold text-[#06B6D4] px-1 shrink-0">
          <span>{statusText}</span>
          <span className="text-[10px] text-[#8B8FA3] font-normal">
            Сессия: {activeSessionTitle}
          </span>
        </div>

        {/* Log Panel (Dockable bottom area) */}
        <LogPanel
          logs={logs}
          isOpen={isLogsOpen}
          onClose={() => setIsLogsOpen(false)}
          onClear={() => setLogs([])}
        />
      </main>

      {/* Modals */}
      <SessionsModal
        isOpen={isSessionsModalOpen}
        sessions={sessions}
        currentSessionId={currentSessionId}
        onSelectSession={(id) => setCurrentSessionId(id)}
        onDeleteSession={handleDeleteSession}
        onNewSession={handleNewChat}
        onClose={() => setIsSessionsModalOpen(false)}
      />

      <StressModal
        isOpen={isStressModalOpen}
        overrides={stressOverrides}
        onAddOverride={handleAddStressOverride}
        onDeleteOverride={handleDeleteStressOverride}
        onClose={() => setIsStressModalOpen(false)}
      />

      <MemoriesModal
        isOpen={isMemoriesModalOpen}
        memories={memories}
        onAddMemory={handleAddMemory}
        onDeleteMemory={handleDeleteMemory}
        onClose={() => setIsMemoriesModalOpen(false)}
      />
    </div>
  );
};
