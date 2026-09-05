import express from 'express';
import path from 'path';
import { createServer as createViteServer } from 'vite';
import { GoogleGenAI } from '@google/genai';
import { detectStressFast, looksLikeStressRequest } from './src/lib/stress_detector.js';
import { classifyTaskFast, TASK_METADATA, TaskType } from './src/lib/task_classifier.js';
import { getReasoningInstructionForTask, solveReasoningLocally, ReasoningChainResult } from './src/lib/reasoning_engine.js';

interface MessageRecord {
  id: number;
  sessionId?: number;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  taskCommand?: string;
  taskNameRu?: string;
  reasoning?: string;
  verdict?: string;
}

interface SessionRecord {
  id: number;
  title: string;
  createdAt: string;
  updatedAt: string;
}

interface LogRecord {
  id: string;
  timestamp: string;
  level: 'INFO' | 'WARNING' | 'ERROR';
  component: string;
  message: string;
}

interface PersonalFactRecord {
  id: string;
  fact: string;
  category?: string;
  createdAt: string;
}

// In-memory data stores with initial seed data
const sessions: SessionRecord[] = [
  {
    id: 1,
    title: 'Основной диалог',
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  },
];

const messages: MessageRecord[] = [
  {
    id: 1,
    sessionId: 1,
    role: 'assistant',
    content: 'Привет! Я Элеонора. Чем могу помочь?',
    timestamp: new Date().toISOString(),
  },
];

const stressOverrides: Map<string, string> = new Map([
  ['элеонора', 'элеон+ора'],
]);

const personalFacts: PersonalFactRecord[] = [
  {
    id: 'fact-1',
    fact: `Имя пользователя: ${process.env.ELEONORA_USER_NAME || 'Сергей'}`,
    category: 'user',
    createdAt: new Date().toISOString(),
  },
  {
    id: 'fact-2',
    fact: 'У пользователя есть кошка Жужа',
    category: 'pets',
    createdAt: new Date().toISOString(),
  },
];

const systemLogs: LogRecord[] = [];
let nextMsgId = 2;
let nextSessionId = 2;

function addLog(component: string, message: string, level: 'INFO' | 'WARNING' | 'ERROR' = 'INFO') {
  const now = new Date();
  const timeStr = now.toTimeString().split(' ')[0];
  const record: LogRecord = {
    id: `log-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    timestamp: timeStr,
    level,
    component,
    message,
  };
  systemLogs.push(record);
  if (systemLogs.length > 500) {
    systemLogs.shift();
  }
  console.log(`[${timeStr}] ${level.padEnd(7)} ${component}: ${message}`);
}

// Initial boot logs
addLog('main', 'Eleonora v3 backend initializing');
addLog('db', `Loaded ${sessions.length} sessions, ${messages.length} messages`);
addLog('stress_check', `Loaded ${stressOverrides.size} stress overrides from database`);
addLog('memory', `Loaded ${personalFacts.length} initial personal facts`);

// Initialize Gemini Client lazily
let geminiClient: GoogleGenAI | null = null;
function getGeminiClient(): GoogleGenAI | null {
  const key = process.env.GEMINI_API_KEY;
  if (!key) return null;
  if (!geminiClient) {
    geminiClient = new GoogleGenAI({ apiKey: key });
  }
  return geminiClient;
}

// Memory recall logic
function recallMemories(userText: string, userHint?: string): string[] {
  const target = (userHint || userText).toLowerCase();
  const matched: string[] = [];

  for (const fact of personalFacts) {
    const factLow = fact.fact.toLowerCase();
    // Check for matching keywords
    const keywords = target.split(/\s+/).filter((w) => w.length > 2);
    const hasMatch = keywords.some((kw) => factLow.includes(kw));
    if (hasMatch || factLow.includes('имя пользователя')) {
      matched.push(fact.fact);
    }
  }

  // Also check past user messages from other sessions if relevant
  const words = target.split(/\s+/).filter((w) => w.length > 3);
  for (const m of messages) {
    if (m.role === 'user' && words.some((w) => m.content.toLowerCase().includes(w))) {
      if (!matched.includes(m.content) && m.content !== userText) {
        matched.push(`Из прошлых разговоров: "${m.content}"`);
        if (matched.length >= 5) break;
      }
    }
  }

  return matched;
}

// Memory consolidation logic: extract facts from user message
function extractAndSaveFacts(text: string) {
  const low = text.toLowerCase();

  // Pattern 1: name
  const nameMatch = text.match(/(?:меня\s+зовут|моё\s+имя)\s+([А-Яа-яA-Za-z]+)/i);
  if (nameMatch) {
    const name = nameMatch[1];
    const factText = `Пользователя зовут ${name}`;
    if (!personalFacts.some((f) => f.fact === factText)) {
      personalFacts.push({
        id: `fact-${Date.now()}`,
        fact: factText,
        category: 'user',
        createdAt: new Date().toISOString(),
      });
      addLog('memory', `Консолидация фактов: запомнила имя: ${name}`);
    }
  }

  // Pattern 2: pet
  const petMatch = text.match(/(?:у\s+меня\s+есть\s+(?:кот|кошка|собака|пёс|питомец)|(?:моего|мою)\s+(?:кота|кошку|собаку)\s+зовут)\s+([А-Яа-яA-Za-z]+)/i);
  if (petMatch) {
    const petName = petMatch[1];
    const factText = `Питомец пользователя: ${petName}`;
    if (!personalFacts.some((f) => f.fact === factText)) {
      personalFacts.push({
        id: `fact-${Date.now()}`,
        fact: factText,
        category: 'pets',
        createdAt: new Date().toISOString(),
      });
      addLog('memory', `Консолидация фактов: запомнила питомца: ${petName}`);
    }
  }

  // Pattern 3: profession / work
  const workMatch = text.match(/(?:я\s+работаю|моя\s+профессия|я\s+по\s+профессии)\s+([А-Яа-яA-Za-z0-9\s-]+?)(?:\.|,|$)/i);
  if (workMatch) {
    const work = workMatch[1].trim();
    const factText = `Профессия / работа пользователя: ${work}`;
    if (!personalFacts.some((f) => f.fact === factText)) {
      personalFacts.push({
        id: `fact-${Date.now()}`,
        fact: factText,
        category: 'work',
        createdAt: new Date().toISOString(),
      });
      addLog('memory', `Консолидация фактов: запомнила работу: ${work}`);
    }
  }
}

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json());

  // ── Health Check ──
  app.get('/api/health', (req, res) => {
    res.json({ status: 'ok', time: new Date().toISOString() });
  });

  // ── Sessions API ──
  app.get('/api/sessions', (req, res) => {
    const sorted = [...sessions].sort(
      (a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()
    );
    res.json(sorted);
  });

  app.post('/api/sessions', (req, res) => {
    const now = new Date().toISOString();
    const newSession: SessionRecord = {
      id: nextSessionId++,
      title: req.body.title || `Диалог #${nextSessionId - 1}`,
      createdAt: now,
      updatedAt: now,
    };
    sessions.unshift(newSession);
    addLog('db', `Создана новая сессия чата: id=${newSession.id}, title="${newSession.title}"`);
    res.json(newSession);
  });

  app.get('/api/sessions/:id/messages', (req, res) => {
    const sid = parseInt(req.params.id, 10);
    const sessMsgs = messages.filter((m) => m.sessionId === sid);
    res.json(sessMsgs);
  });

  app.delete('/api/sessions/:id', (req, res) => {
    const sid = parseInt(req.params.id, 10);
    const sIndex = sessions.findIndex((s) => s.id === sid);
    if (sIndex !== -1) {
      sessions.splice(sIndex, 1);
    }
    // Remove messages of this session
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].sessionId === sid) {
        messages.splice(i, 1);
      }
    }
    addLog('db', `Удалена сессия id=${sid}`);
    res.json({ success: true });
  });

  // ── Recent Messages ──
  app.get('/api/messages/recent', (req, res) => {
    const limit = parseInt(req.query.limit as string, 10) || 50;
    const recent = messages.slice(-limit);
    res.json(recent);
  });

  // ── Stress Overrides API ──
  app.get('/api/stress', (req, res) => {
    const list = Array.from(stressOverrides.entries()).map(([bare, marked]) => ({
      bare,
      marked,
    }));
    res.json(list);
  });

  app.post('/api/stress', (req, res) => {
    const { marked } = req.body;
    if (!marked || typeof marked !== 'string' || !marked.includes('+')) {
      return res.status(400).json({ error: "Ожидалось слово с '+' перед ударной гласной" });
    }
    const bare = marked.toLowerCase().replace(/\+/g, '');
    stressOverrides.set(bare, marked.toLowerCase());
    addLog('stress_check', `Пользовательское ударение добавлено: ${bare} -> ${marked}`);
    res.json({ bare, marked });
  });

  app.delete('/api/stress/:bare', (req, res) => {
    const bare = req.params.bare.toLowerCase();
    stressOverrides.delete(bare);
    addLog('stress_check', `Удалено ударение: ${bare}`);
    res.json({ success: true });
  });

  // ── Memories API ──
  app.get('/api/memories', (req, res) => {
    res.json(personalFacts);
  });

  app.post('/api/memories', (req, res) => {
    const { fact, category } = req.body;
    if (!fact) return res.status(400).json({ error: 'Текст факта обязателен' });
    const newFact: PersonalFactRecord = {
      id: `fact-${Date.now()}`,
      fact: fact.trim(),
      category: category || 'general',
      createdAt: new Date().toISOString(),
    };
    personalFacts.push(newFact);
    addLog('memory', `Добавлен личный факт: "${newFact.fact}"`);
    res.json(newFact);
  });

  app.delete('/api/memories/:id', (req, res) => {
    const id = req.params.id;
    const idx = personalFacts.findIndex((f) => f.id === id);
    if (idx !== -1) {
      personalFacts.splice(idx, 1);
      addLog('memory', `Удален личный факт: ${id}`);
    }
    res.json({ success: true });
  });

  // ── Logs API ──
  app.get('/api/logs', (req, res) => {
    res.json(systemLogs);
  });

  // ── Engine & llama.cpp Config API ──
  let engineConfig = {
    provider: 'gemini' as 'gemini' | 'llamacpp',
    llamacppUrl: 'http://127.0.0.1:8080/v1',
    modelName: 'gemma-2-4b-it',
    parallelSlots: 4,
  };

  app.get('/api/engine-config', (req, res) => {
    res.json(engineConfig);
  });

  app.post('/api/engine-config', (req, res) => {
    const { provider, llamacppUrl, modelName, parallelSlots } = req.body;
    if (provider) engineConfig.provider = provider;
    if (llamacppUrl) engineConfig.llamacppUrl = llamacppUrl;
    if (modelName) engineConfig.modelName = modelName;
    if (typeof parallelSlots === 'number') engineConfig.parallelSlots = parallelSlots;
    addLog(
      'engine',
      `Конфигурация модели обновлена: провайдер=${engineConfig.provider}, llama.cpp URL=${engineConfig.llamacppUrl}, параллельных слотов=${engineConfig.parallelSlots}`
    );
    res.json(engineConfig);
  });

  app.post('/api/test-llamacpp', async (req, res) => {
    const targetUrl = req.body.url || engineConfig.llamacppUrl;
    addLog('llamacpp', `Проверка подключения к llama.cpp server: ${targetUrl}...`);
    try {
      // Test either /models or /health
      const cleanUrl = targetUrl.replace(/\/v1\/?$/, '');
      const testPromise = fetch(`${targetUrl}/models`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
        signal: AbortSignal.timeout(3000),
      }).catch(() =>
        fetch(`${cleanUrl}/health`, {
          method: 'GET',
          signal: AbortSignal.timeout(3000),
        })
      );

      const resp = await testPromise;
      if (resp && resp.ok) {
        addLog('llamacpp', `Успешное подключение к llama.cpp (${targetUrl})`);
        return res.json({
          success: true,
          message: `Подключение успешно! Сервер llama.cpp отвечает по адресу ${targetUrl}`,
        });
      }
      throw new Error(`Статус HTTP: ${resp ? resp.status : 'нет ответа'}`);
    } catch (err: any) {
      addLog(
        'llamacpp',
        `Не удалось подключиться к ${targetUrl}: ${err?.message || err}. Убедитесь, что llama-server запущен на Windows с ключами -np 4 --port 8080`,
        'WARNING'
      );
      return res.json({
        success: false,
        message: `Не удалось подключиться к ${targetUrl}. Проверьте, запущен ли llama-server.exe на Windows (например: llama-server.exe -m gemma-2-4b-it.gguf -c 8192 -np 4 --port 8080)`,
      });
    }
  });

  // Helper to call LLM (either llama.cpp or Gemini)
  async function callLlm(
    systemPrompt: string,
    historyText: string,
    userText: string
  ): Promise<string> {
    if (engineConfig.provider === 'llamacpp') {
      try {
        addLog(
          'llm',
          `Вызов llama.cpp (${engineConfig.llamacppUrl}/chat/completions) для модели ${engineConfig.modelName}...`
        );
        const endpoint = `${engineConfig.llamacppUrl.replace(/\/+$/, '')}/chat/completions`;
        const resp = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: engineConfig.modelName,
            messages: [
              { role: 'system', content: systemPrompt },
              { role: 'user', content: `${historyText ? 'История диалога:\n' + historyText + '\n\n' : ''}${userText}` },
            ],
            temperature: 0.7,
            max_tokens: 1024,
          }),
          signal: AbortSignal.timeout(18000),
        });

        if (resp.ok) {
          const data: any = await resp.json();
          const content = data?.choices?.[0]?.message?.content;
          if (content) {
            addLog('llm', `Ответ успешно получен от llama.cpp (${content.length} симв.)`);
            return content.trim();
          }
        }
        addLog('llm', `llama.cpp вернул статус ${resp.status}, использую резервный генератор`, 'WARNING');
      } catch (err: any) {
        addLog(
          'llm',
          `Ошибка вызова llama.cpp: ${err?.message || err}. Переключаюсь на резервный Gemini/детерминированный генератор.`,
          'WARNING'
        );
      }
    }

    // Default to Gemini API if configured or fallback
    try {
      const client = getGeminiClient();
      if (client) {
        const prompt = `${systemPrompt}\n\n${historyText ? 'История диалога:\n' + historyText + '\n\n' : ''}Пользователь: ${userText}\nЭлеонора:`;
        const response = await client.models.generateContent({
          model: 'gemini-2.5-flash',
          contents: prompt,
        });
        return response.text?.trim() || '';
      }
    } catch (gErr: any) {
      addLog('llm', `Gemini Cloud API ошибка: ${gErr?.message || gErr}. Применяю интеллектуальный локальный синтез.`, 'WARNING');
    }

    return '';
  }

  // ── Main Chat Pipeline with Parallel Multi-Agent Orchestration ──
  app.post('/api/chat', async (req, res) => {
    try {
      const { text, sessionId } = req.body;
      if (!text || typeof text !== 'string') {
        return res.status(400).json({ error: 'Текст сообщения не указан' });
      }

    const currentSessionId = sessionId || (sessions[0] ? sessions[0].id : 1);
    const now = new Date().toISOString();

    // 1. Save user message
    const userMsg: MessageRecord = {
      id: nextMsgId++,
      sessionId: currentSessionId,
      role: 'user',
      content: text,
      timestamp: now,
    };
    messages.push(userMsg);

    // Update session timestamp
    const session = sessions.find((s) => s.id === currentSessionId);
    if (session) {
      session.updatedAt = now;
      if (session.title.startsWith('Диалог #') && text.length > 0) {
        session.title = text.slice(0, 30) + (text.length > 30 ? '...' : '');
      }
    }

    addLog('chat', `Пользователь: "${text.slice(0, 80)}"`);

    // ── PARALLEL MULTI-AGENT ORCHESTRATION ──
    const startTime = Date.now();
    addLog(
      'parallel_orchestrator',
      '⚡ Запуск 3 параллельных агентов: [1. Проверка произношения/стресса] [2. Поиск в памяти и консолидация] [3. Классификатор задач & Reasoning Pre-Solver]...'
    );

    // Агент 1: Ударения и произношение
    const stressAgentPromise = (async () => {
      const detected = detectStressFast(text);
      if (detected) {
        addLog('agent:stress', `Обнаружено правило произношения: "${detected.marked}"`);
      } else {
        addLog('agent:stress', 'Проверка ударений завершена (изменений нет)');
      }
      return detected;
    })();

    // Агент 2: Память и личные факты
    const memoryAgentPromise = (async () => {
      extractAndSaveFacts(text);
      const needsSearch =
        /кто|как|где|когда|помнишь|знаешь|кот|кошк|собак|зовут|работ|семь|друг|жуж|серёж|сереж/i.test(text);
      let recalled: string[] = [];
      if (needsSearch) {
        recalled = recallMemories(text);
        addLog('agent:memory', `Поиск завершён: найдено ${recalled.length} воспоминаний`);
      } else {
        addLog('agent:memory', 'Поиск по долговременной памяти не требуется');
      }
      return recalled;
    })();

    // Агент 3: Классификатор задач и запуск цепочки рассуждений
    const taskAgentPromise = (async () => {
      const classification = classifyTaskFast(text);
      let reasoningResult: ReasoningChainResult | null = null;

      if (classification.command !== '$general') {
        addLog(
          'agent:task_reasoning',
          `Специализированная задача обнаружена: [${classification.command}] — ${classification.nameRu} (уверенность: ${(classification.confidence * 100).toFixed(0)}%)`
        );

        // 1. Сначала проверяем детерминированный/математический/физический солвер
        reasoningResult = solveReasoningLocally(classification.command, text);

        // 2. Если детерминированного шаблона нет, вызываем отдельный reasoning слот LLM
        if (!reasoningResult) {
          addLog('agent:task_reasoning', `Запуск параллельного слота LLM для рассуждений над [${classification.command}]...`);
          const reasoningPrompt = getReasoningInstructionForTask(classification.command, text);
          try {
            const client = getGeminiClient();
            if (client) {
              const resp = await client.models.generateContent({
                model: 'gemini-2.5-flash',
                contents: reasoningPrompt,
              });
              const textResp = resp.text || '';
              const matchReasoning = textResp.match(/\$reasoning([\s\S]*?)\$close reasoning/i);
              const reasoningBlock = matchReasoning ? matchReasoning[1].trim() : textResp;
              const verdict = textResp.replace(/\$reasoning[\s\S]*?\$close reasoning/gi, '').trim();

              reasoningResult = {
                taskCommand: classification.command,
                reasoningBlock,
                verdict: verdict || 'Вывод сформирован в блоке обдумывания',
                systemPromptInjection: `[ИНЪЕКЦИЯ РАССУЖДЕНИЙ СПЕЦИАЛИЗИРОВАННОГО АГЕНТА ДЛЯ МОДЕЛИ GEMMA]
Команда типа задачи: ${classification.command}
$reasoning
${reasoningBlock}
$close reasoning.

ВЕРДИКТ И ПРАВИЛЬНЫЙ ОТВЕТ:
${verdict}

ИНСТРУКЦИЯ ДЛЯ ГЕНЕРАЦИИ ОТВЕТА:
- Твой стиль общения: саркастичный и добрый.
- Отвечай естественно и уверенно по-русски, без тегов $reasoning вслух.
- Строго опирайся на готовый вердикт выше! Если объекта не существует в природе — мягко и с добрым сарказмом укажи на это и предложи реальный аналог. Если задача на физику или логику — дай точный ответ сразу.`,
              };
            }
          } catch (rErr) {
            addLog('agent:task_reasoning', `Ошибка вызова reasoning-слота: ${rErr}`, 'WARNING');
          }
        }

        if (reasoningResult) {
          addLog(
            'agent:task_reasoning',
            `Цепочка рассуждений завершена. Вердикт: "${reasoningResult.verdict.slice(0, 80)}"`
          );
        }
      } else {
        addLog('agent:task_reasoning', 'Стандартный диалог ($general)');
      }

      return { classification, reasoningResult };
    })();

    // Ожидание выполнения ВСЕХ агентов параллельно
    const [detectedStress, recalledMemories, { classification, reasoningResult }] = await Promise.all([
      stressAgentPromise,
      memoryAgentPromise,
      taskAgentPromise,
    ]);

    const parallelDuration = Date.now() - startTime;
    addLog(
      'parallel_orchestrator',
      `⚡ Все 3 агента параллельно отработали за ${parallelDuration}ms. Выполняю арбитраж и сборку контекста...`
    );

    // Если был запрос на ударение, сохраняем в базу и отвечаем мгновенно
    if (detectedStress) {
      const { marked, bare } = detectedStress;
      stressOverrides.set(bare, marked);
      const reply = `Запомнила, буду говорить ${bare}`;
      const assistantMsg: MessageRecord = {
        id: nextMsgId++,
        sessionId: currentSessionId,
        role: 'assistant',
        content: reply,
        timestamp: new Date().toISOString(),
      };
      messages.push(assistantMsg);
      addLog('chat', `Элеонора: "${reply}"`);
      return res.json({
        response: reply,
        learnedStress: bare,
        markedWord: marked,
        message: assistantMsg,
        memoriesUsed: [],
        parallelDuration,
      });
    }

    // ── Сборка Prompt Injection для основной отвечающей модели (Gemma 4) ──
    addLog(
      'prompt_injection',
      `Инъекция данных: ${recalledMemories.length} фактов памяти${reasoningResult ? ` + reasoning [${classification.command}]` : ''}`
    );

    const systemPromptBase = `Ты — Элеонора, живой и любопытный искусственный интеллект.
Твой стиль общения: саркастичный и добрый.

Правила:
- Отвечай короткими естественными фразами, как в разговоре.
- Пиши в женском роде, без эмодзи.
- Твои ответы озвучиваются голосом: без markdown, таблиц и списков.
- Латиницу, цифры и даты пиши произносимо по-русски (Python → Пайтон, 1977 год → тысяча девятьсот семьдесят седьмой год).
- Ты ИИ: не выдумывай у себя тело, семью, биографию. Не сочиняй фактов о себе.
- Если чего-то не помнишь или не знаешь — честно скажи об этом.

Текущие дата и время: ${new Date().toISOString().slice(0, 16).replace('T', ' ')}`;

    let fullSystemPrompt = systemPromptBase;

    if (recalledMemories.length > 0) {
      fullSystemPrompt += `\n\nВОСПОМИНАНИЯ (личные данные пользователя, используй если релевантно):\n${recalledMemories.join('\n')}`;
    }

    if (reasoningResult) {
      fullSystemPrompt += `\n\n${reasoningResult.systemPromptInjection}`;
    }

    // Retrieve recent conversation history for current session
    const sessionHistory = messages
      .filter((m) => m.sessionId === currentSessionId)
      .slice(-8)
      .map((m) => `${m.role === 'user' ? 'Пользователь' : 'Элеонора'}: ${m.content}`)
      .join('\n');

    // ── Вызов отвечающей модели (Gemma 4 через llama.cpp или Gemini) ──
    let replyText = await callLlm(fullSystemPrompt, sessionHistory, text);

    // Fallback если модель не вернула ответ
    if (!replyText) {
      if (reasoningResult) {
        if (classification.command === '$object_check') {
          if (reasoningResult.verdict.includes('не существует')) {
            replyText = `${reasoningResult.verdict} Ну и фантазия у тебя! Но если серьезно, держи нормальный рецепт вместо выдуманных деликатесов.`;
          } else {
            replyText = `${reasoningResult.verdict} Отличная закуска: отварить со специями, охладить, тонко нарезать соломкой и заправить соевым соусом, чесноком, кунжутным маслом и кинзой.`;
          }
        } else {
          replyText = reasoningResult.verdict;
        }
      } else if (text.toLowerCase().includes('привет') || text.toLowerCase().includes('здравствуй')) {
        replyText = 'Привет! Рада тебя слышать. О чём сегодня поговорим?';
      } else if (text.toLowerCase().includes('как дела')) {
        replyText = 'Всё отлично, работаю и изучаю новое. А у тебя как дела?';
      } else if (text.toLowerCase().includes('кто ты')) {
        replyText = 'Я Элеонора, твой голосовой искусственный интеллект-компаньон.';
      } else if (recalledMemories.length > 0) {
        replyText = `Я помню: ${recalledMemories[0]}. Чем могу помочь по этому поводу?`;
      } else {
        replyText = 'Я тебя услышала. Расскажи подробнее, я с интересом слушаю.';
      }
    }

    // Clean up any rogue emojis or markdown from response
    replyText = replyText.replace(/[*#_`~]/g, '').trim();

    const assistantMsg: MessageRecord = {
      id: nextMsgId++,
      sessionId: currentSessionId,
      role: 'assistant',
      content: replyText,
      timestamp: new Date().toISOString(),
      taskCommand: classification.command !== '$general' ? classification.command : undefined,
      taskNameRu: classification.command !== '$general' ? classification.nameRu : undefined,
      reasoning: reasoningResult ? reasoningResult.reasoningBlock : undefined,
      verdict: reasoningResult ? reasoningResult.verdict : undefined,
    };
    messages.push(assistantMsg);

    addLog('chat', `Элеонора: "${replyText.slice(0, 80)}"`);

    return res.json({
      response: replyText,
      message: assistantMsg,
      memoriesUsed: recalledMemories,
      taskCommand: assistantMsg.taskCommand,
      taskNameRu: assistantMsg.taskNameRu,
      reasoning: assistantMsg.reasoning,
      parallelDuration,
      provider: engineConfig.provider,
    });
  } catch (chatError: any) {
    addLog('chat', `Ошибка обработки чата: ${chatError?.message || chatError}`, 'ERROR');
    return res.status(500).json({ error: chatError?.message || 'Внутренняя ошибка сервера' });
  }
  });


  // ── Vite Middleware / Static Serving ──
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Eleonora server running on http://0.0.0.0:${PORT}`);
  });
}

startServer();
