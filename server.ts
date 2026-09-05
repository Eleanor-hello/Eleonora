import express from 'express';
import path from 'path';
import { createServer as createViteServer } from 'vite';
import { GoogleGenAI } from '@google/genai';
import { detectStressFast, looksLikeStressRequest } from './src/lib/stress_detector.js';

interface MessageRecord {
  id: number;
  sessionId?: number;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
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

  // ── Main Chat Pipeline ──
  app.post('/api/chat', async (req, res) => {
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

    // ── 1. Агент ударений (Stress Check Agent) ──
    addLog('stress_check', 'Проверяю сообщение на правила произношения...');
    const detectedStress = detectStressFast(text);

    if (detectedStress) {
      const { marked, bare } = detectedStress;
      stressOverrides.set(bare, marked);
      addLog('stress_check', `Найдено исправление ударения: ${marked} (сохранено в базу)`);

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
      });
    }

    addLog('stress_check', 'Исправлений ударения нет');

    // ── 2. Агент поиска в памяти (Memory Search Agent) ──
    addLog('search_check', 'Проверяю, нужен ли поиск в памяти...');
    extractAndSaveFacts(text);

    let recalledMemories: string[] = [];
    const needsSearch =
      /кто|как|где|когда|помнишь|знаешь|кот|кошк|собак|зовут|работ|семь|друг|жуж|серёж|сереж/i.test(text);

    if (needsSearch) {
      addLog('search_check', 'Нужен поиск в личных фактах');
      recalledMemories = recallMemories(text);
      if (recalledMemories.length > 0) {
        addLog('search', `Найдено воспоминаний: ${recalledMemories.length}`);
      } else {
        addLog('search', 'Релевантных воспоминаний не найдено');
      }
    } else {
      addLog('search_check', 'Поиск не требуется');
    }

    // ── 3. Генерация ответа моделью (LLM) ──
    addLog('llm', 'Генерирую ответ...');

    const systemPromptBase = `Ты — Элеонора, живой и любопытный искусственный интеллект.

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

    // Retrieve recent conversation history for current session
    const sessionHistory = messages
      .filter((m) => m.sessionId === currentSessionId)
      .slice(-10)
      .map((m) => `${m.role === 'user' ? 'Пользователь' : 'Элеонора'}: ${m.content}`)
      .join('\n');

    let replyText = '';

    try {
      const client = getGeminiClient();
      if (client) {
        const prompt = `${fullSystemPrompt}\n\nИстория диалога:\n${sessionHistory}\n\nЭлеонора:`;
        const response = await client.models.generateContent({
          model: 'gemini-2.5-flash',
          contents: prompt,
        });
        replyText = response.text?.trim() || '';
      }
    } catch (llmErr) {
      addLog('llm', `Gemini API вызов завершился с ошибкой: ${llmErr}`, 'WARNING');
    }

    // Fallback if LLM is unavailable or offline
    if (!replyText) {
      if (text.toLowerCase().includes('привет') || text.toLowerCase().includes('здравствуй')) {
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
    };
    messages.push(assistantMsg);

    addLog('llm', `Ответ готов (${replyText.length} символов)`);
    addLog('chat', `Элеонора: "${replyText.slice(0, 80)}"`);

    return res.json({
      response: replyText,
      message: assistantMsg,
      memoriesUsed: recalledMemories,
    });
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
