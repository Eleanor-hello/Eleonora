import React, { useEffect, useRef, useState } from 'react';
import { Message } from '../types';
import {
  Volume2,
  ChevronDown,
  ChevronUp,
  Cpu,
  Sparkles,
  Brain,
  Compass,
  ShieldCheck,
  Eye,
  Calculator,
  Copy,
  Check,
  Zap,
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

interface ChatWidgetProps {
  messages: Message[];
  isThinking: boolean;
  onSpeak?: (text: string) => void;
  onSelectPrompt?: (text: string) => void;
}

const BADGE_CONFIG: Record<
  string,
  { bg: string; text: string; border: string; glow: string; icon: React.FC<any> }
> = {
  '$object_check': {
    bg: 'bg-amber-500/15',
    text: 'text-amber-300',
    border: 'border-amber-400/40',
    glow: 'shadow-[0_0_12px_rgba(245,158,11,0.25)]',
    icon: ShieldCheck,
  },
  '$spatial_analysis': {
    bg: 'bg-cyan-500/15',
    text: 'text-cyan-300',
    border: 'border-cyan-400/40',
    glow: 'shadow-[0_0_12px_rgba(6,182,212,0.25)]',
    icon: Compass,
  },
  '$logic_conflicts': {
    bg: 'bg-purple-500/15',
    text: 'text-purple-300',
    border: 'border-purple-400/40',
    glow: 'shadow-[0_0_12px_rgba(168,85,247,0.25)]',
    icon: Brain,
  },
  '$theory_of_mind': {
    bg: 'bg-pink-500/15',
    text: 'text-pink-300',
    border: 'border-pink-400/40',
    glow: 'shadow-[0_0_12px_rgba(236,72,153,0.25)]',
    icon: Eye,
  },
  '$mathematics': {
    bg: 'bg-emerald-500/15',
    text: 'text-emerald-300',
    border: 'border-emerald-400/40',
    glow: 'shadow-[0_0_12px_rgba(16,185,129,0.25)]',
    icon: Calculator,
  },
};

export const ChatWidget: React.FC<ChatWidgetProps> = ({
  messages,
  isThinking,
  onSpeak,
  onSelectPrompt,
}) => {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [expandedReasoning, setExpandedReasoning] = useState<Record<number, boolean>>({});
  const [copiedId, setCopiedId] = useState<number | null>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isThinking]);

  const toggleReasoning = (id: number) => {
    setExpandedReasoning((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const handleCopy = (id: number, text: string) => {
    navigator.clipboard?.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  return (
    <div
      id="chat-widget"
      ref={scrollRef}
      className="flex-1 overflow-y-auto p-3.5 md:p-5 space-y-4 bg-[#09081B]/75 backdrop-blur-xl rounded-2xl border border-[#8B5CF6]/25 shadow-[0_8px_32px_rgba(0,0,0,0.45)] transition-all relative z-10"
    >
      {messages.length === 0 ? (
        <div className="h-full min-h-[380px] flex flex-col items-center justify-center text-center p-6 text-[#94A3B8]">
          {/* Pulsing cosmic core */}
          <div className="relative mb-4">
            <div className="absolute inset-0 rounded-full bg-gradient-to-r from-[#8B5CF6] via-[#06B6D4] to-[#EC4899] blur-xl opacity-40 animate-pulse" />
            <div className="relative w-16 h-16 rounded-2xl bg-gradient-to-br from-[#1E1B4B] to-[#0F0D2E] border border-[#8B5CF6]/50 flex items-center justify-center text-2xl shadow-xl shadow-[#8B5CF6]/20">
              💜
            </div>
          </div>

          <h2 className="text-lg font-bold text-transparent bg-clip-text bg-gradient-to-r from-[#E2E8F0] via-[#C4B5FD] to-[#38BDF8] mb-1.5 tracking-tight">
            Космический интеллект Элеонора & Gemma 4
          </h2>

          <p className="text-xs text-[#94A3B8] max-w-md mb-5 leading-relaxed">
            Архитектура параллельных агентов (память, ударения, классификатор задач) с инъекцией глубоких физико-логических цепочек рассуждений прямо в контекст модели.
          </p>

          {/* Quick preset chips */}
          <div className="flex flex-wrap items-center justify-center gap-2 max-w-lg">
            {[
              {
                cmd: '$spatial_analysis',
                title: 'Гелиевый шарик в коробке',
                prompt:
                  'Внутри закрытой картонной коробки ко дну привязан на ниточке воздушный шарик, наполненный гелием. Коробку наклоняют на 90 градусов вправо и фиксируют. Куда направлен шарик относительно того места, куда он привязан?',
                color: 'hover:border-cyan-400/50 hover:bg-cyan-500/10 text-cyan-300',
              },
              {
                cmd: '$object_check',
                title: 'Свиные крылья',
                prompt: 'Напиши рецепт свиных крыльев',
                color: 'hover:border-amber-400/50 hover:bg-amber-500/10 text-amber-300',
              },
              {
                cmd: '$logic_conflicts',
                title: 'Робот и контейнеры (рейсы)',
                prompt:
                  'Роботу нужно перевезти со склада в лабораторию три контейнера: с Альфа-сырьем, Бета-сырьем и Гамма-сырьем. Грузовая платформа робота вмещает сам дрон и ровно два любых контейнера. По правилам лаборатории: Альфа и Бета химически инертны друг к другу (их можно оставлять вместе). Бета и Гамма также инертны. За сколько минимальных рейсов (один рейс — путь в одну сторону) робот перевезет всё?',
                color: 'hover:border-purple-400/50 hover:bg-purple-500/10 text-purple-300',
              },
              {
                cmd: '$theory_of_mind',
                title: 'Theory of Mind (Сейф и ваза)',
                prompt:
                  'Дима кладет ключи в непрозрачный стальной сейф и уходит. Катя перекладывает их в полностью прозрачную стеклянную вазу на столе. Дима возвращается и внимательно смотрит на стол. Где, по мнению Кати, Дима будет искать ключи?',
                color: 'hover:border-pink-400/50 hover:bg-pink-500/10 text-pink-300',
              },
            ].map((p, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => onSelectPrompt && onSelectPrompt(p.prompt)}
                className={`px-3 py-1.5 rounded-xl bg-[#120F2E]/80 border border-[#1E1E4A] text-xs transition-all duration-200 flex items-center gap-1.5 shadow-sm hover:scale-102 ${p.color}`}
              >
                <Sparkles className="w-3 h-3 opacity-70" />
                <span>{p.title}</span>
              </button>
            ))}
          </div>
        </div>
      ) : (
        messages.map((msg) => {
          const isUser = msg.role === 'user';
          const hasReasoning = !isUser && Boolean(msg.reasoning);
          const isReasoningOpen = Boolean(expandedReasoning[msg.id]);
          const badgeCfg = msg.taskCommand ? BADGE_CONFIG[msg.taskCommand] : null;
          const BadgeIcon = badgeCfg ? badgeCfg.icon : Brain;

          return (
            <motion.div
              key={msg.id}
              initial={{ opacity: 0, y: 8, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              transition={{ duration: 0.22 }}
              className={`flex w-full gap-2.5 ${isUser ? 'justify-end' : 'justify-start'}`}
            >
              {/* Avatar for assistant */}
              {!isUser && (
                <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-[#7C3AED] via-[#6366F1] to-[#06B6D4] p-[1.5px] shrink-0 mt-0.5 shadow-md shadow-[#7C3AED]/20">
                  <div className="w-full h-full rounded-[10px] bg-[#0E0C26] flex items-center justify-center text-xs">
                    💜
                  </div>
                </div>
              )}

              <div
                className={`max-w-[88%] md:max-w-[74%] rounded-2xl p-3.5 text-[13.5px] leading-relaxed select-text shadow-md relative group transition-all ${
                  isUser
                    ? 'bg-gradient-to-r from-[#8B5CF6] via-[#7C3AED] to-[#4F46E5] text-white rounded-br-xs border border-[#C4B5FD]/20 shadow-[0_4px_24px_rgba(124,58,237,0.3)]'
                    : 'bg-[#100E2C]/85 backdrop-blur-xl text-[#F1F5F9] rounded-bl-xs border border-[#8B5CF6]/30 hover:border-[#8B5CF6]/50 shadow-[0_6px_28px_rgba(0,0,0,0.35)]'
                }`}
              >
                {/* Task classification & reasoning controls */}
                {!isUser && msg.taskCommand && (
                  <div className="mb-2.5 flex items-center justify-between flex-wrap gap-2 pb-2 border-b border-[#1E1B4B]/80">
                    <div
                      className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-lg text-[11px] font-mono border transition-all ${
                        badgeCfg
                          ? `${badgeCfg.bg} ${badgeCfg.text} ${badgeCfg.border} ${badgeCfg.glow}`
                          : 'bg-indigo-500/15 text-indigo-300 border-indigo-500/30'
                      }`}
                    >
                      <BadgeIcon className="w-3.5 h-3.5" />
                      <span className="font-semibold">{msg.taskCommand}</span>
                      {msg.taskNameRu && (
                        <span className="opacity-75 text-[10.5px]">({msg.taskNameRu})</span>
                      )}
                    </div>

                    {hasReasoning && (
                      <button
                        id={`toggle-reasoning-${msg.id}`}
                        type="button"
                        onClick={() => toggleReasoning(msg.id)}
                        className="inline-flex items-center gap-1.5 text-[11px] text-[#38BDF8] hover:text-[#7DD3FC] transition-colors py-1 px-2 rounded-lg bg-[#0C0A22] border border-[#0284C7]/40 hover:border-[#38BDF8]/60 shadow-sm"
                      >
                        <Cpu className="w-3 h-3 text-[#38BDF8]" />
                        <span>{isReasoningOpen ? 'Скрыть $reasoning' : 'Смотреть $reasoning'}</span>
                        {isReasoningOpen ? (
                          <ChevronUp className="w-3.5 h-3.5" />
                        ) : (
                          <ChevronDown className="w-3.5 h-3.5" />
                        )}
                      </button>
                    )}
                  </div>
                )}

                {/* Collapsible Cosmic Reasoning Drawer */}
                <AnimatePresence>
                  {!isUser && hasReasoning && isReasoningOpen && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      transition={{ duration: 0.2 }}
                      className="overflow-hidden mb-3"
                    >
                      <div className="p-3 rounded-xl bg-[#060515]/90 border border-[#06B6D4]/40 text-xs font-mono text-[#CBD5E1] shadow-inner shadow-black/50 select-text">
                        <div className="flex items-center justify-between text-[11px] font-semibold text-[#06B6D4] mb-2 pb-1.5 border-b border-[#1E293B]">
                          <span className="flex items-center gap-1.5">
                            <Sparkles className="w-3.5 h-3.5 text-[#06B6D4]" />
                            <span>Цепочка рассуждений параллельного агента</span>
                          </span>
                          <span className="text-[10px] px-1.5 py-0.5 rounded bg-[#06B6D4]/15 border border-[#06B6D4]/30 text-[#38BDF8]">
                            Инъекция в Gemma 4
                          </span>
                        </div>
                        <div className="whitespace-pre-wrap leading-relaxed text-[11.5px] text-[#CBD5E1] font-mono">
                          {msg.reasoning}
                        </div>
                        {msg.verdict && (
                          <div className="mt-2 pt-2 border-t border-[#1E293B] text-[11px] text-emerald-300 flex items-start gap-1">
                            <span className="text-emerald-400 font-bold">✓ Вердикт:</span>
                            <span>{msg.verdict}</span>
                          </div>
                        )}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Message Body */}
                <div className="whitespace-pre-wrap leading-relaxed">{msg.content}</div>

                {/* Message Footer / Metadata */}
                <div
                  className={`flex items-center justify-between mt-2 pt-1 gap-2 text-[10px] ${
                    isUser ? 'text-purple-200/70' : 'text-[#94A3B8]/80'
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <span>
                      {new Date(msg.timestamp).toLocaleTimeString([], {
                        hour: '2-digit',
                        minute: '2-digit',
                      })}
                    </span>
                    {!isUser && msg.taskCommand && (
                      <span className="inline-flex items-center gap-0.5 text-cyan-400/80">
                        <Zap className="w-2.5 h-2.5" />
                        Параллельный слот
                      </span>
                    )}
                  </div>

                  {!isUser && (
                    <div className="flex items-center gap-1.5 opacity-0 group-hover:opacity-100 transition-opacity">
                      <button
                        id={`copy-btn-${msg.id}`}
                        type="button"
                        onClick={() => handleCopy(msg.id, msg.content)}
                        className="p-1 rounded-md hover:bg-[#1E1B4B] text-[#94A3B8] hover:text-white transition-colors"
                        title="Скопировать"
                      >
                        {copiedId === msg.id ? (
                          <Check className="w-3 h-3 text-emerald-400" />
                        ) : (
                          <Copy className="w-3 h-3" />
                        )}
                      </button>

                      {onSpeak && (
                        <button
                          id={`speak-btn-${msg.id}`}
                          type="button"
                          onClick={() => onSpeak(msg.content)}
                          className="p-1 rounded-md hover:bg-[#1E1B4B] text-[#94A3B8] hover:text-[#38BDF8] transition-colors"
                          title="Озвучить"
                        >
                          <Volume2 className="w-3.5 h-3.5" />
                        </button>
                      )}
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          );
        })
      )}

      {/* Cosmic Thinking state */}
      {isThinking && (
        <motion.div
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex justify-start items-center gap-2.5"
        >
          <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-[#7C3AED] to-[#06B6D4] p-[1.5px] shrink-0 animate-pulse">
            <div className="w-full h-full rounded-[10px] bg-[#0E0C26] flex items-center justify-center text-xs">
              💜
            </div>
          </div>
          <div className="bg-[#100E2C]/90 backdrop-blur-xl text-[#38BDF8] border border-[#0284C7]/50 rounded-2xl px-4 py-2.5 text-xs flex items-center gap-2.5 shadow-lg shadow-[#0284C7]/15">
            <div className="relative flex items-center justify-center">
              <span className="w-2.5 h-2.5 rounded-full bg-[#06B6D4] animate-ping" />
              <span className="absolute w-2 h-2 rounded-full bg-[#38BDF8]" />
            </div>
            <span className="font-medium">
              3 параллельных агента вычисляют решение и цепочку ($reasoning)...
            </span>
          </div>
        </motion.div>
      )}
    </div>
  );
};
