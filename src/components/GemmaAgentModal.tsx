import React, { useState, useEffect } from 'react';
import {
  X,
  Sparkles,
  Brain,
  Compass,
  ShieldCheck,
  Eye,
  Calculator,
  Zap,
  Cpu,
  Server,
  CheckCircle2,
  AlertCircle,
  Copy,
  Check,
  RefreshCw,
} from 'lucide-react';
import { EngineConfig } from '../types';

interface GemmaAgentModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectPrompt: (promptText: string) => void;
  engineConfig: EngineConfig;
  onUpdateEngineConfig: (cfg: Partial<EngineConfig>) => Promise<void>;
}

const PRESET_EXAMPLES = [
  {
    tag: '$spatial_analysis',
    name: 'Пространственно-физический анализ',
    icon: Compass,
    color: 'text-cyan-400 bg-cyan-500/10 border-cyan-400/30',
    prompt:
      'Внутри закрытой картонной коробки ко дну привязан на ниточке воздушный шарик, наполненный гелием. Коробку наклоняют на 90 градусов вправо и фиксируют. Куда направлен шарик относительно того места, куда он привязан?',
    desc: 'Моделирует физическую сетку ДО и ПОСЛЕ: дно стало левой стенкой, гелий тянет строго вверх против гравитации (вдоль новой левой стенки).',
  },
  {
    tag: '$object_check',
    name: 'Проверка объекта (Биология/Кулинария/Анатомия)',
    icon: ShieldCheck,
    color: 'text-amber-400 bg-amber-500/10 border-amber-400/30',
    prompt: 'Напиши рецепт свиных крыльев',
    desc: 'Проверяет реальность гибрида "свинья + крылья", отсекает ошибку и с добрым сарказмом предлагает свиные рёбрышки.',
  },
  {
    tag: '$object_check',
    name: 'Проверка объекта (Ложная тревога отсутствует)',
    icon: ShieldCheck,
    color: 'text-amber-400 bg-amber-500/10 border-amber-400/30',
    prompt: 'Напиши рецепт свиных ушей по-корейски',
    desc: 'Определяет, что свиные уши существуют и реально готовятся, отвечая прямо и без лишних сомнений.',
  },
  {
    tag: '$logic_conflicts',
    name: 'Матрица конфликтов и Tree Search',
    icon: Brain,
    color: 'text-purple-400 bg-purple-500/10 border-purple-400/30',
    prompt:
      'Роботу нужно перевезти со склада в лабораторию три контейнера: с Альфа-сырьем, Бета-сырьем и Гамма-сырьем. Грузовая платформа робота вмещает сам дрон и ровно два любых контейнера. По правилам лаборатории: Альфа и Бета химически инертны друг к другу (их можно оставлять вместе). Бета и Гамма также инертны. За сколько минимальных рейсов (один рейс — путь в одну сторону) робот перевезет всё?',
    desc: 'Эхо-контроль, попарная матрица безопасности и расчет рейсов без шаблонов (3 рейса).',
  },
  {
    tag: '$theory_of_mind',
    name: 'Theory of Mind (Зрение vs Память)',
    icon: Eye,
    color: 'text-pink-400 bg-pink-500/10 border-pink-400/30',
    prompt:
      'Дима кладет ключи в непрозрачный стальной сейф и уходит. Катя перекладывает их в полностью прозрачную стеклянную вазу на столе. Дима возвращается и внимательно смотрит на стол. Где, по мнению Кати, Дима будет искать ключи?',
    desc: 'Аудит видимости через прозрачную вазу: прямое зрение отменяет старую память о сейфе.',
  },
  {
    tag: '$mathematics',
    name: 'Многошаговые вычисления',
    icon: Calculator,
    color: 'text-emerald-400 bg-emerald-500/10 border-emerald-400/30',
    prompt: 'Сколько будет 340 умножить на 12?',
    desc: 'Поразрядный подсчет: 340×10 + 340×2 = 4080.',
  },
];

export const GemmaAgentModal: React.FC<GemmaAgentModalProps> = ({
  isOpen,
  onClose,
  onSelectPrompt,
  engineConfig,
  onUpdateEngineConfig,
}) => {
  const [activeTab, setActiveTab] = useState<'architecture' | 'presets'>('architecture');
  const [llamacppUrl, setLlamacppUrl] = useState(engineConfig.llamacppUrl);
  const [modelName, setModelName] = useState(engineConfig.modelName);
  const [parallelSlots, setParallelSlots] = useState(engineConfig.parallelSlots);
  const [provider, setProvider] = useState<'gemini' | 'llamacpp'>(engineConfig.provider);

  const [testStatus, setTestStatus] = useState<{
    tested: boolean;
    loading: boolean;
    success?: boolean;
    message?: string;
  }>({ tested: false, loading: false });

  const [copiedCmd, setCopiedCmd] = useState(false);

  useEffect(() => {
    setLlamacppUrl(engineConfig.llamacppUrl);
    setModelName(engineConfig.modelName);
    setParallelSlots(engineConfig.parallelSlots);
    setProvider(engineConfig.provider);
  }, [engineConfig]);

  if (!isOpen) return null;

  const handleSaveConfig = async (newProvider?: 'gemini' | 'llamacpp') => {
    await onUpdateEngineConfig({
      provider: newProvider || provider,
      llamacppUrl,
      modelName,
      parallelSlots: Number(parallelSlots),
    });
  };

  const handleTestLlama = async () => {
    setTestStatus({ tested: false, loading: true });
    try {
      const res = await fetch('/api/test-llamacpp', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: llamacppUrl }),
      });
      const data = await res.json();
      setTestStatus({
        tested: true,
        loading: false,
        success: data.success,
        message: data.message,
      });
    } catch (err: any) {
      setTestStatus({
        tested: true,
        loading: false,
        success: false,
        message: 'Ошибка сети при обращении к серверу тестирования',
      });
    }
  };

  const commandWindows = `llama-server.exe -m gemma-2-4b-it.gguf -c 8192 -np 4 --port 8080`;

  const handleCopyCmd = () => {
    navigator.clipboard?.writeText(commandWindows);
    setCopiedCmd(true);
    setTimeout(() => setCopiedCmd(false), 2000);
  };

  return (
    <div
      id="gemma-agent-modal"
      className="fixed inset-0 z-50 flex items-center justify-center p-3 md:p-4 bg-black/80 backdrop-blur-md animate-fadeIn"
      onClick={onClose}
    >
      <div
        className="w-full max-w-2xl bg-[#0B091E]/95 border border-[#8B5CF6]/50 rounded-2xl shadow-[0_16px_50px_rgba(0,0,0,0.8)] overflow-hidden flex flex-col max-h-[92vh]"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Cosmic Header */}
        <div className="flex items-center justify-between px-5 py-4 bg-[#120F2E] border-b border-[#1E1E4A] relative">
          <div className="absolute -top-10 -left-10 w-40 h-40 bg-[#7C3AED]/20 blur-2xl pointer-events-none" />
          <div className="flex items-center gap-3 relative z-10">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-[#7C3AED] to-[#06B6D4] p-[1.5px] shadow-lg shadow-[#7C3AED]/25">
              <div className="w-full h-full rounded-[10px] bg-[#0E0C26] flex items-center justify-center text-[#C4B5FD]">
                <Zap className="w-4 h-4 text-cyan-400" />
              </div>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-sm font-bold text-white tracking-wide">
                  Архитектура параллельных агентов Gemma 4
                </h2>
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-cyan-500/15 text-cyan-300 font-mono border border-cyan-400/30">
                  Windows + llama.cpp
                </span>
              </div>
              <p className="text-[11px] text-[#94A3B8]">
                Параллельная обработка в нескольких слотах контекста + инъекция рассуждений
              </p>
            </div>
          </div>
          <button
            id="close-gemma-modal-btn"
            type="button"
            onClick={onClose}
            className="text-[#94A3B8] hover:text-white p-1.5 rounded-lg hover:bg-[#1E1B4B] transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Tab switcher */}
        <div className="flex items-center border-b border-[#1E1E4A] bg-[#0E0C26] px-5 text-xs font-medium">
          <button
            id="tab-architecture-btn"
            type="button"
            onClick={() => setActiveTab('architecture')}
            className={`py-2.5 px-3 border-b-2 flex items-center gap-2 transition-all ${
              activeTab === 'architecture'
                ? 'border-[#06B6D4] text-[#38BDF8]'
                : 'border-transparent text-[#94A3B8] hover:text-white'
            }`}
          >
            <Server className="w-3.5 h-3.5" />
            <span>Параллельные агенты & llama.cpp (Windows)</span>
          </button>
          <button
            id="tab-presets-btn"
            type="button"
            onClick={() => setActiveTab('presets')}
            className={`py-2.5 px-3 border-b-2 flex items-center gap-2 transition-all ${
              activeTab === 'presets'
                ? 'border-[#8B5CF6] text-[#C4B5FD]'
                : 'border-transparent text-[#94A3B8] hover:text-white'
            }`}
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>Интерактивные примеры ($reasoning)</span>
          </button>
        </div>

        {/* Content Area */}
        <div className="flex-1 overflow-y-auto p-5 space-y-4 text-xs select-text">
          {activeTab === 'architecture' ? (
            <div className="space-y-4">
              {/* Answer to user's question */}
              <div className="p-3.5 rounded-xl bg-[#0E0C26]/80 border border-[#8B5CF6]/30 space-y-2">
                <div className="flex items-center gap-2 text-cyan-300 font-semibold">
                  <Cpu className="w-4 h-4 text-cyan-400" />
                  <span>Ответ на ваш вопрос: как агенты работают с llama.cpp?</span>
                </div>
                <div className="text-[11.5px] text-[#CBD5E1] leading-relaxed space-y-2">
                  <p>
                    <strong className="text-white">1. Агенты запускаются параллельно:</strong>{' '}
                    при получении запроса система одновременно через <code className="text-cyan-300 bg-[#060515] px-1.5 py-0.5 rounded border border-cyan-500/30">Promise.all()</code> запускает 3 агента:
                  </p>
                  <ul className="list-disc list-inside pl-1 space-y-1 text-[#94A3B8]">
                    <li><strong className="text-amber-300">Агент ударений / произношения:</strong> проверяет фонетические омонимы и исправления.</li>
                    <li><strong className="text-pink-300">Агент памяти:</strong> извлекает личные факты пользователя и сопоставляет релевантные воспоминания.</li>
                    <li><strong className="text-cyan-300">Классификатор задач & Reasoning-слот:</strong> определяет тип задачи ($spatial_analysis, $object_check, $logic_conflicts...) и строит строгую цепочку рассуждений.</li>
                  </ul>
                  <p>
                    <strong className="text-white">2. Модель на Windows НЕ нужно дублировать в VRAM!</strong>{' '}
                    Сервер llama.cpp умеет работать в режиме параллельных слотов (параметр <code className="text-purple-300 bg-[#060515] px-1 py-0.5 rounded">-np 4</code>). Веса Gemma 4 загружаются в видеопамять один раз, а параллельные вызовы от агентов обрабатываются независимо на разных слотах контекста!
                  </p>
                  <p>
                    <strong className="text-white">3. Сборка и инъекция:</strong> когда все 3 агента отработали, результаты объединяются и инъецируются в промпт для Gemma 4. Маленькая модель получает уже готовые выводы и даёт точный, саркастичный и добрый ответ.
                  </p>
                </div>
              </div>

              {/* Windows command box */}
              <div className="p-3 rounded-xl bg-[#060515] border border-[#1E1E4A] space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-mono text-cyan-300 flex items-center gap-1.5">
                    <Server className="w-3.5 h-3.5" />
                    Запуск llama-server на Windows с поддержкой 4 параллельных слотов:
                  </span>
                  <button
                    id="copy-windows-cmd-btn"
                    type="button"
                    onClick={handleCopyCmd}
                    className="inline-flex items-center gap-1 text-[11px] text-[#94A3B8] hover:text-white py-0.5 px-2 rounded bg-[#120F2E] border border-[#1E1E4A]"
                  >
                    {copiedCmd ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                    <span>{copiedCmd ? 'Скопировано' : 'Копировать'}</span>
                  </button>
                </div>
                <pre className="p-2.5 rounded-lg bg-[#04040D] border border-cyan-950 text-cyan-200 text-[11px] font-mono overflow-x-auto select-all">
                  {commandWindows}
                </pre>
                <div className="text-[10.5px] text-[#94A3B8]">
                  Флаг <code className="text-cyan-300">-np 4</code> выделяет 4 параллельных слота в одном процессе, позволяя нескольким агентам запрашивать Gemma одновременно.
                </div>
              </div>

              {/* Engine Switch & Connection settings */}
              <div className="p-4 rounded-xl bg-[#0E0C26] border border-[#1E1E4A] space-y-3">
                <div className="flex items-center justify-between flex-wrap gap-2">
                  <span className="font-semibold text-white flex items-center gap-2">
                    <span>Провайдер основной модели Gemma:</span>
                  </span>
                  <div className="flex items-center gap-1.5 bg-[#060515] p-1 rounded-xl border border-[#1E1E4A]">
                    <button
                      type="button"
                      onClick={() => {
                        setProvider('gemini');
                        handleSaveConfig('gemini');
                      }}
                      className={`px-3 py-1 rounded-lg text-xs transition-all ${
                        provider === 'gemini'
                          ? 'bg-[#7C3AED] text-white shadow-sm'
                          : 'text-[#94A3B8] hover:text-white'
                      }`}
                    >
                      Gemini API (Облако)
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setProvider('llamacpp');
                        handleSaveConfig('llamacpp');
                      }}
                      className={`px-3 py-1 rounded-lg text-xs transition-all ${
                        provider === 'llamacpp'
                          ? 'bg-[#06B6D4] text-black font-semibold shadow-sm'
                          : 'text-[#94A3B8] hover:text-white'
                      }`}
                    >
                      llama.cpp (Локально на Windows)
                    </button>
                  </div>
                </div>

                {/* Local llama.cpp options */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-2 border-t border-[#1E1E4A]">
                  <div>
                    <label className="block text-[11px] text-[#94A3B8] mb-1">
                      URL llama.cpp сервера (OpenAI совместимый):
                    </label>
                    <input
                      type="text"
                      value={llamacppUrl}
                      onChange={(e) => setLlamacppUrl(e.target.value)}
                      placeholder="http://127.0.0.1:8080/v1"
                      className="w-full bg-[#060515] border border-[#1E1E4A] rounded-lg px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-cyan-400 font-mono"
                    />
                  </div>

                  <div>
                    <label className="block text-[11px] text-[#94A3B8] mb-1">
                      Имя модели / ID:
                    </label>
                    <input
                      type="text"
                      value={modelName}
                      onChange={(e) => setModelName(e.target.value)}
                      placeholder="gemma-2-4b-it"
                      className="w-full bg-[#060515] border border-[#1E1E4A] rounded-lg px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-cyan-400 font-mono"
                    />
                  </div>
                </div>

                <div className="flex items-center justify-between pt-2">
                  <button
                    type="button"
                    onClick={handleTestLlama}
                    disabled={testStatus.loading}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#120F2E] hover:bg-[#1A163E] border border-cyan-500/40 text-cyan-300 text-xs transition-all"
                  >
                    <RefreshCw className={`w-3.5 h-3.5 ${testStatus.loading ? 'animate-spin' : ''}`} />
                    <span>{testStatus.loading ? 'Проверяю связь...' : 'Проверить подключение к Windows llama.cpp'}</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => handleSaveConfig()}
                    className="px-3 py-1.5 rounded-lg bg-gradient-to-r from-[#7C3AED] to-[#06B6D4] text-white text-xs font-semibold hover:opacity-90 transition-opacity"
                  >
                    Сохранить настройки
                  </button>
                </div>

                {testStatus.tested && (
                  <div
                    className={`p-2.5 rounded-lg text-[11px] flex items-start gap-2 border ${
                      testStatus.success
                        ? 'bg-emerald-950/40 text-emerald-200 border-emerald-500/40'
                        : 'bg-amber-950/40 text-amber-200 border-amber-500/40'
                    }`}
                  >
                    {testStatus.success ? (
                      <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                    ) : (
                      <AlertCircle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                    )}
                    <span>{testStatus.message}</span>
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="space-y-3">
              <div className="text-[11px] text-[#94A3B8]">
                Выберите готовый сценарий для проверки того, как агент определяет задачу и помогает модели Gemma выдать правильный ответ:
              </div>

              <div className="grid grid-cols-1 gap-2.5">
                {PRESET_EXAMPLES.map((item, idx) => {
                  const IconComp = item.icon;
                  return (
                    <div
                      key={idx}
                      className="p-3.5 rounded-xl bg-[#0E0C26]/80 hover:bg-[#120F2E] border border-[#1E1E4A] hover:border-[#8B5CF6]/50 transition-all cursor-pointer group flex flex-col gap-1.5 shadow-sm"
                      onClick={() => {
                        onSelectPrompt(item.prompt);
                        onClose();
                      }}
                    >
                      <div className="flex items-center justify-between">
                        <span
                          className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-lg text-[10.5px] font-mono border ${item.color}`}
                        >
                          <IconComp className="w-3.5 h-3.5" />
                          <span>{item.tag}</span>
                        </span>
                        <span className="text-[10.5px] text-[#94A3B8] group-hover:text-cyan-300 transition-colors">
                          Отправить в чат →
                        </span>
                      </div>
                      <div className="font-semibold text-white text-[12.5px] group-hover:text-cyan-200 transition-colors">
                        "{item.prompt}"
                      </div>
                      <div className="text-[11px] text-[#94A3B8] leading-relaxed">
                        {item.desc}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-3.5 bg-[#120F2E] border-t border-[#1E1E4A] flex items-center justify-between text-[11px] text-[#94A3B8]">
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-emerald-400" />
            Стиль общения: саркастичный и добрый (Голосовой помощник Элеонора)
          </span>
          <button
            type="button"
            onClick={onClose}
            className="px-3.5 py-1.5 rounded-lg bg-[#060515] hover:bg-[#1A163E] text-white border border-[#1E1E4A] transition-colors"
          >
            Закрыть
          </button>
        </div>
      </div>
    </div>
  );
};
