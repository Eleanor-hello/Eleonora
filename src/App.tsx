import React, { useState, useEffect, useRef } from 'react';
import { Message, ChatSession, LogEntry, PersonalFact, StressOverride, EngineConfig } from './types';
import { ChatWidget } from './components/ChatWidget';
import { LogPanel } from './components/LogPanel';
import { SessionsModal } from './components/SessionsModal';
import { MemoriesModal } from './components/MemoriesModal';
import { StressModal } from './components/StressModal';
import { GemmaAgentModal } from './components/GemmaAgentModal';
import { CosmicBackground } from './components/CosmicBackground';
import { sanitizeForTts, yoficate, applyStress } from './lib/tts_preprocessor';
import {
  Brain,
  BookOpen,
  Send,
  Plus,
  History,
  FileText,
  Volume2,
  VolumeX,
  Sparkles,
  Zap,
  Radio,
} from 'lucide-react';

export const App: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<number | null>(null);
  const [inputText, setInputText] = useState('');
  const [isThinking, setIsThinking] = useState(false);
  const [statusText, setStatusText] = useState('');

  // Engine & llama.cpp configuration
  const [engineConfig, setEngineConfig] = useState<EngineConfig>({
    provider: 'gemini',
    llamacppUrl: 'http://127.0.0.1:8080/v1',
    modelName: 'gemma-2-4b-it',
    parallelSlots: 4,
  });

  // Toggles & Modals
  const [isTtsEnabled, setIsTtsEnabled] = useState(true);
  const [isLogsOpen, setIsLogsOpen] = useState(false);
  const [isSessionsModalOpen, setIsSessionsModalOpen] = useState(false);
  const [isMemoriesModalOpen, setIsMemoriesModalOpen] = useState(false);
  const [isStressModalOpen, setIsStressModalOpen] = useState(false);
  const [isGemmaModalOpen, setIsGemmaModalOpen] = useState(false);

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
    loadEngineConfig();
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

  const loadEngineConfig = async () => {
    try {
      const res = await fetch('/api/engine-config');
      if (res.ok) {
        const data: EngineConfig = await res.json();
        setEngineConfig(data);
      }
    } catch (err) {
      console.warn('Failed to load engine config', err);
    }
  };

  const handleUpdateEngineConfig = async (newCfg: Partial<EngineConfig>) => {
    try {
      const res = await fetch('/api/engine-config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newCfg),
      });
      if (res.ok) {
        const data: EngineConfig = await res.json();
        setEngineConfig(data);
        setStatusText(
          data.provider === 'llamacpp'
            ? 'Активен локальный llama.cpp (Windows)'
            : 'Активен облачный Gemini API'
        );
        loadLogs();
      }
    } catch (err) {
      console.error('Failed to update engine config', err);
    }
  };

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

      let processed = sanitizeForTts(text);
      processed = yoficate(processed);

      const stressMap: Record<string, string> = {};
      stressOverrides.forEach((s) => {
        stressMap[s.bare] = s.marked;
      });
      processed = applyStress(processed, stressMap);

      const voiceSpeakable = processed.replace(/\+/g, '');

      const utterance = new SpeechSynthesisUtterance(voiceSpeakable);
      utterance.lang = 'ru-RU';
      utterance.rate = 1.0;
      utterance.pitch = 1.05;

      const voices = window.speechSynthesis.getVoices();
      const ruVoice =
        voices.find(
          (v) =>
            v.lang.startsWith('ru') &&
            (v.name.includes('Female') ||
              v.name.includes('xenia') ||
              v.name.includes('Google') ||
              v.name.includes('Milena'))
        ) || voices.find((v) => v.lang.startsWith('ru'));

      if (ruVoice) {
        utterance.voice = ruVoice;
      }

      window.speechSynthesis.speak(utterance);
    } catch (e) {
      console.warn('Speech synthesis error', e);
    }
  };

  const handleSend = async (customPrompt?: string) => {
    const text = (customPrompt || inputText).trim();
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
    setStatusText('3 параллельных агента анализируют задачу...');

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

      if (data.parallelDuration) {
        setStatusText(
          `⚡ Параллельный анализ завершен за ${data.parallelDuration}ms (${data.provider || 'gemini'})`
        );
      } else {
        setStatusText('');
      }

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
        body: JSON.stringify({
          title: `Диалог ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`,
        }),
      });
      const newSession: ChatSession = await res.json();
      setSessions((prev) => [newSession, ...prev]);
      setCurrentSessionId(newSession.id);
      setMessages([]);
      setStatusText('Создана новая космическая сессия');
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
    sessions.find((s) => s.id === currentSessionId)?.title || 'Космический диалог';

  return (
    <div
      id="eleonora-main-window"
      className="flex flex-col h-screen w-screen bg-[#060511] text-[#F1F5F9] select-none overflow-hidden font-sans relative"
    >
      {/* Dynamic Cosmic Universe Background (Canvas + Nebula) */}
      <CosmicBackground />

      {/* Top Cosmic Glass Header */}
      <header className="h-14 bg-[#0A081E]/80 backdrop-blur-xl border-b border-[#8B5CF6]/25 px-4 md:px-6 flex items-center justify-between shrink-0 relative z-20 shadow-[0_4px_24px_rgba(0,0,0,0.5)]">
        <div className="flex items-center gap-3">
          {/* Pulsing galaxy core avatar */}
          <div className="relative">
            <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-[#8B5CF6] via-[#06B6D4] to-[#EC4899] blur-md opacity-50 animate-pulse" />
            <div className="relative w-8 h-8 rounded-xl bg-gradient-to-br from-[#1E1B4B] to-[#0F0D2E] border border-[#8B5CF6]/60 flex items-center justify-center text-sm shadow-md">
              💜
            </div>
          </div>

          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-sm font-bold tracking-tight text-white flex items-center gap-1.5">
                <span className="bg-clip-text text-transparent bg-gradient-to-r from-white via-[#E0E7FF] to-[#C4B5FD]">
                  Элеонора
                </span>
                <span className="text-[11px] font-medium text-cyan-300">v3 Cosmic</span>
              </h1>
              <span className="hidden sm:inline-flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full bg-cyan-500/15 text-cyan-300 font-mono border border-cyan-400/30">
                <Radio className="w-2.5 h-2.5 text-cyan-400 animate-pulse" />
                {engineConfig.provider === 'llamacpp'
                  ? 'Windows llama.cpp (-np 4)'
                  : 'Gemma 4 (Gemini API)'}
              </span>
            </div>
            <p className="text-[10.5px] text-[#94A3B8] hidden md:block">
              Мульти-агентная параллельная инъекция цепочек рассуждений ($reasoning)
            </p>
          </div>
        </div>

        {/* Action buttons with cosmic gradient styling */}
        <div className="flex items-center gap-2">
          {/* Gemma 4 & Parallel Agent Settings */}
          <button
            id="open-gemma-agent-btn"
            type="button"
            onClick={() => setIsGemmaModalOpen(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-gradient-to-r from-[#7C3AED]/25 to-[#06B6D4]/25 hover:from-[#7C3AED]/40 hover:to-[#06B6D4]/40 text-[#E0E7FF] text-xs border border-[#8B5CF6]/40 hover:border-cyan-400/50 transition-all duration-200 shadow-sm"
            title="Архитектура параллельных агентов и настройки llama.cpp"
          >
            <Zap className="w-3.5 h-3.5 text-cyan-400" />
            <span className="font-medium hidden sm:inline">Gemma 4 & llama.cpp</span>
          </button>

          {/* Memories */}
          <button
            id="open-memories-btn"
            type="button"
            onClick={() => setIsMemoriesModalOpen(true)}
            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl bg-[#120F2E]/80 hover:bg-[#1E1B4B] text-[#E2E8F0] text-xs border border-[#8B5CF6]/30 hover:border-[#8B5CF6]/50 transition-all"
            title="Просмотр памяти и фактов"
          >
            <Brain className="w-3.5 h-3.5 text-cyan-400" />
            <span className="hidden sm:inline">Память ({memories.length})</span>
          </button>

          {/* Stress overrides */}
          <button
            id="open-stress-btn"
            type="button"
            onClick={() => setIsStressModalOpen(true)}
            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl bg-[#120F2E]/80 hover:bg-[#1E1B4B] text-[#E2E8F0] text-xs border border-[#8B5CF6]/30 hover:border-[#8B5CF6]/50 transition-all"
            title="База ударений"
          >
            <BookOpen className="w-3.5 h-3.5 text-[#C4B5FD]" />
            <span className="hidden sm:inline">Ударения ({stressOverrides.length})</span>
          </button>
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 flex flex-col p-3 md:p-4 gap-3 min-h-0 relative z-10">
        {/* Chat Widget with modern cosmic bubbles */}
        <ChatWidget
          messages={messages}
          isThinking={isThinking}
          onSpeak={speakText}
          onSelectPrompt={(text) => handleSend(text)}
        />

        {/* Quick prompt launcher bar */}
        <div className="flex items-center gap-1.5 overflow-x-auto pb-1 scrollbar-none text-[11px] text-[#94A3B8]">
          <span className="shrink-0 flex items-center gap-1 text-[#C4B5FD] font-medium pr-1">
            <Sparkles className="w-3 h-3 text-cyan-400" />
            Запустить тест:
          </span>
          {[
            {
              label: 'Шарик в коробке',
              cmd: '$spatial_analysis',
              text: 'Внутри закрытой картонной коробки ко дну привязан на ниточке воздушный шарик, наполненный гелием. Коробку наклоняют на 90 градусов вправо и фиксируют. Куда направлен шарик относительно того места, куда он привязан?',
              color: 'text-cyan-300 border-cyan-500/30 hover:bg-cyan-500/10',
            },
            {
              label: 'Свиные крылья',
              cmd: '$object_check',
              text: 'Напиши рецепт свиных крыльев',
              color: 'text-amber-300 border-amber-500/30 hover:bg-amber-500/10',
            },
            {
              label: 'Робот и контейнеры',
              cmd: '$logic_conflicts',
              text: 'Роботу нужно перевезти со склада в лабораторию три контейнера: с Альфа-сырьем, Бета-сырьем и Гамма-сырьем. Грузовая платформа робота вмещает сам дрон и ровно два любых контейнера. По правилам лаборатории: Альфа и Бета химически инертны друг к другу (их можно оставлять вместе). Бета и Гамма также инертны. За сколько минимальных рейсов (один рейс — путь в одну сторону) робот перевезет всё?',
              color: 'text-purple-300 border-purple-500/30 hover:bg-purple-500/10',
            },
            {
              label: 'Theory of Mind (Сейф/Ваза)',
              cmd: '$theory_of_mind',
              text: 'Дима кладет ключи в непрозрачный стальной сейф и уходит. Катя перекладывает их в полностью прозрачную стеклянную вазу на столе. Дима возвращается и внимательно смотрит на стол. Где, по мнению Кати, Дима будет искать ключи?',
              color: 'text-pink-300 border-pink-500/30 hover:bg-pink-500/10',
            },
            {
              label: '340 × 12',
              cmd: '$mathematics',
              text: 'Сколько будет 340 умножить на 12?',
              color: 'text-emerald-300 border-emerald-500/30 hover:bg-emerald-500/10',
            },
          ].map((item, idx) => (
            <button
              key={idx}
              type="button"
              onClick={() => handleSend(item.text)}
              className={`shrink-0 px-2.5 py-1 rounded-lg bg-[#0C0A22]/80 border transition-all text-[11px] ${item.color}`}
            >
              {item.label}
            </button>
          ))}
        </div>

        {/* Input Row: Cosmic capsule layout */}
        <div id="input-row" className="flex items-center gap-2 shrink-0">
          {/* ➕ New Chat */}
          <button
            id="new-chat-btn"
            type="button"
            onClick={handleNewChat}
            className="w-11 h-11 rounded-2xl bg-[#120F2E]/85 hover:bg-[#1A163E] text-white border border-[#8B5CF6]/40 hover:border-[#8B5CF6] flex items-center justify-center shrink-0 transition-all shadow-md shadow-[#8B5CF6]/10 hover:scale-105 active:scale-95"
            title="Новый диалог"
          >
            <Plus className="w-5 h-5 text-cyan-300" />
          </button>

          {/* 📜 Dialog History */}
          <button
            id="history-btn"
            type="button"
            onClick={() => setIsSessionsModalOpen(true)}
            className="w-11 h-11 rounded-2xl bg-[#120F2E]/85 hover:bg-[#1A163E] text-white border border-[#8B5CF6]/40 hover:border-[#8B5CF6] flex items-center justify-center shrink-0 transition-all shadow-md shadow-[#8B5CF6]/10 hover:scale-105 active:scale-95"
            title="История диалогов"
          >
            <History className="w-4 h-4 text-[#E2E8F0]" />
          </button>

          {/* 📋 Logs Toggle */}
          <button
            id="toggle-logs-btn"
            type="button"
            onClick={() => setIsLogsOpen(!isLogsOpen)}
            className={`w-11 h-11 rounded-2xl border flex items-center justify-center shrink-0 transition-all shadow-md hover:scale-105 active:scale-95 ${
              isLogsOpen
                ? 'bg-gradient-to-br from-[#7C3AED] to-[#06B6D4] text-white border-transparent shadow-[0_0_16px_rgba(6,182,212,0.4)]'
                : 'bg-[#120F2E]/85 hover:bg-[#1A163E] text-[#E2E8F0] border-[#8B5CF6]/40'
            }`}
            title="Телеметрия и логи параллельных агентов"
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
            className="w-11 h-11 rounded-2xl bg-[#120F2E]/85 hover:bg-[#1A163E] text-[#E2E8F0] border border-[#8B5CF6]/40 hover:border-[#8B5CF6] flex items-center justify-center shrink-0 transition-all shadow-md hover:scale-105 active:scale-95"
            title={isTtsEnabled ? 'Выключить озвучку' : 'Включить озвучку'}
          >
            {isTtsEnabled ? (
              <Volume2 className="w-4 h-4 text-cyan-400" />
            ) : (
              <VolumeX className="w-4 h-4 text-[#94A3B8]" />
            )}
          </button>

          {/* Input field with cosmic glow */}
          <div className="flex-1 relative">
            <input
              id="chat-input"
              ref={inputRef}
              type="text"
              placeholder="Спроси о физике, объектах, логике или просто поговори..."
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleSend();
              }}
              disabled={isThinking}
              className="w-full h-11 bg-[#0E0C26]/90 backdrop-blur-xl border border-[#8B5CF6]/40 rounded-2xl px-4 text-sm text-white placeholder-[#94A3B8]/60 focus:outline-none focus:border-cyan-400 focus:ring-2 focus:ring-cyan-500/20 transition-all shadow-inner shadow-black/40 disabled:opacity-50"
            />
          </div>

          {/* 🚀 Cosmic Rocket / Send button */}
          <button
            id="send-btn"
            type="button"
            onClick={() => handleSend()}
            disabled={isThinking || !inputText.trim()}
            className="w-11 h-11 rounded-2xl bg-gradient-to-tr from-[#7C3AED] via-[#6366F1] to-[#06B6D4] hover:shadow-[0_0_20px_rgba(124,58,237,0.55)] disabled:opacity-40 disabled:hover:shadow-none text-white flex items-center justify-center shrink-0 transition-all shadow-lg hover:scale-105 active:scale-95"
            title="Отправить"
          >
            <Send className="w-4 h-4 translate-x-[-1px] translate-y-[1px]" />
          </button>
        </div>

        {/* Status bar */}
        <div
          id="status-bar"
          className="h-4 flex items-center justify-between text-[11px] font-medium text-cyan-300 px-1 shrink-0"
        >
          <span className="truncate">{statusText}</span>
          <span className="text-[10.5px] text-[#94A3B8] shrink-0">
            {activeSessionTitle}
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

      <GemmaAgentModal
        isOpen={isGemmaModalOpen}
        onClose={() => setIsGemmaModalOpen(false)}
        onSelectPrompt={(prompt) => {
          setInputText(prompt);
          setTimeout(() => inputRef.current?.focus(), 80);
        }}
        engineConfig={engineConfig}
        onUpdateEngineConfig={handleUpdateEngineConfig}
      />
    </div>
  );
};
