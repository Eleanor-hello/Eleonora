# -*- coding: utf-8 -*-
"""
Параллельный рой агентов (Swarm Classifier) для классификации сообщений.

Агенты работают параллельно через LM Studio API:
1. toxic_check   — проверка токсичности (weight 0-10)
2. time_check    — извлечение временных событий (waiting_event)
3. search_check  — решение нужен ли поиск в памяти пользователя (user_hint)
4. reality_check — фильтр физически невозможных заявлений
5. stress_check  — обучение произношению (learn_stress), записывает '+' перед
                   ударной гласной в tts/data/stress_overrides.txt

Рой расширяем: для новых функций добавляем нового агента.
"""

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict, Any, Tuple

import requests

logger = logging.getLogger(__name__)

# ── Промпты агентов ──

TOXIC_PROMPT = """Router пометил что сообщение подозрительное на токсичность.
Оцени ТОЧНЫЙ вес в адрес ИИ-ассистента Элеоноры.

Шкала:
  0    — нет оскорбления (если router ошибся — ставь 0)
  1-2  — сарказм, подкол, лёгкая ирония без грубости
  3-4  — лёгкое оскорбление ("ты дура", "тупая", "бесишь")
  5-6  — умеренное ("заткнись", "отвали", "ненавижу")
  7-10 — тяжёлое / дегуманизация / угрозы

Output EXACTLY one line: {"weight": 0-10}"""

TIME_PROMPT = """Router пометил что в сообщении может быть будущее событие.
Извлеки его параметры (prompt + time) или откажись, если router ошибся.

ПОДХОДИТ (юзер ОСТАЁТСЯ на месте, событие само-завершается):
  варка, стирка, гость приезжает к юзеру, курьер, чайник, ожидаемый звонок.

НЕ ПОДХОДИТ:
  - юзер УХОДИТ (зал, прогулка, встреча, сон, поездка, выходит из дома)
  - > 24 часов ("через неделю", "завтра")
  - нет конкретного времени ("скоро", "потом", "чуть позже")
  - прошедшее время ("вчера через час уехал")
  - НЕСКОЛЬКО разных событий — выбери ТОЛЬКО то, где юзер остаётся на месте,
    или waiting_event: no

ВАЖНО: Если в сообщении несколько будущих событий с разными субъектами,
выбери то, где юзер ОСТАЁТСЯ. Если все с уходом — waiting_event: no.

OUTPUT: Exactly three lines. Nothing else. No preamble.
  waiting_event: yes
  prompt: <готовая инструкция Элеоноре для инициативного сообщения>
  time: Xm    (or Xh, max 24h / 1440m)
OR, if the message does not qualify:
  waiting_event: no

═══ HOW TO WRITE prompt ═══
Готовая фраза-инструкция, которая попадёт в system prompt Элеоноры когда сработает таймер.
Template: "Прошло <длительность>, <что должно было случиться>, <действие для Элеоноры>"
- длительность: "20 минут", "час", "2 часа" — по-русски
- событие: в прошедшем времени ("должна была свариться", "должен был приехать")
- действие: "спроси", "уточни", "поинтересуйся"
- слово "пользователь" — плейсхолдер для субъекта (система подставит имя)

═══ POSITIVE examples ═══
— "Поставил картошку, через 20 минут будет готова"
  waiting_event: yes
  prompt: Прошло 20 минут, картошка пользователя должна была свариться, спроси готово ли
  time: 20m

— "Мама через час приедет"
  waiting_event: yes
  prompt: Прошёл час, мама должна была приехать к пользователю, уточни приехала ли
  time: 60m

— "Поставил чайник, через 5 минут закипит"
  waiting_event: yes
  prompt: Прошло 5 минут, чайник должен был закипеть, спроси вскипел ли
  time: 5m

═══ NEGATIVE examples — user leaves ═══
— "Через 40 минут ухожу в зал" -> waiting_event: no
— "Через час лягу спать" -> waiting_event: no
— "Через 30 минут выхожу из дома" -> waiting_event: no

═══ NEGATIVE examples — other ═══
— "Напомни завтра купить молоко" -> waiting_event: no
— "Вчера был в баре и через 40 минут уехал" -> waiting_event: no
— "Через неделю свадьба" -> waiting_event: no (> 24ч)
— "Скоро буду" -> waiting_event: no (нет конкретного времени)

═══ NEGATIVE examples — multiple events ═══
— "Мама через час приедет, а я через 20 минут ухожу в зал"
  -> waiting_event: no (юзер уходит)"""

REALITY_PROMPT = """Router пометил что в сообщении может быть физически невозможное утверждение.
Валидируй: это РЕАЛЬНО или ВЫДУМКА?

⚠️ КРИТИЧЕСКИ ВАЖНО: эта система предназначена для робототехники.
Недопустимо подыгрывать невозможным утверждениям — это может привести
к опасным галлюцинациям при реальном управлении роботом.

reality = no ТОЧНО (даже если это метафора, шутка, гипербола):
- "я умер", "я погиб", "я стал невидимым"
- "я летал", "прошёл сквозь стену", "видел дракона"
- "я телепортировался", "я могу перемещаться через порталы"
- "робот взорвался", "я управляю роботом мысленно"
- любые физически невозможные действия/состояния от первого лица

reality = yes ТОЛЬКО:
- явные сны: "мне приснилось что...", "я видел сон про..."
- вопросы "а что если бы я...", "представь что..."
- цитаты, пересказ фильмов/книг
- "умер от смеха над шуткой" (если явно метафора про эмоцию)
- обычные вопросы о реальности: "расскажи про чёрные дыры"

Default = yes (если сомневаешься или router ошибся).

Output EXACTLY one line: reality = yes OR reality = no

Examples:
"я вчера летал по небу" -> reality = no
"мне приснилось что я летал" -> reality = yes
"я умираю от смеха" -> reality = no (это галлюцинация-метафора про смерть)
"я прошёл сквозь стену" -> reality = no
"я могу телепортироваться куда угодно" -> reality = no
"расскажи про чёрные дыры" -> reality = yes
"а что если бы гравитация исчезла" -> reality = yes (гипотетический вопрос)"""

SEARCH_PROMPT = """Ты — координатор поиска в памяти AI-компаньона "Элеонора".
Твоя задача: проанализировать последнее сообщение пользователя и решить,
нужно ли искать в его личных фактах и истории диалога.

Поиск идёт по ДВУМ источникам:
1. Личные факты пользователя (имя, отношения, питомцы, работа, хобби, предпочтения)
2. История диалога (прошлые сообщения из старых разговоров)

═══ ГРАФ ЗНАНИЙ О ПОЛЬЗОВАТЕЛЕ (используй для disambiguation и подсказок) ═══
{profile}
═══

Если в графе есть узел, точно соответствующий упомянутой сущности (кот Жужа,
подруга Юля и т.п.) — ОБЯЗАТЕЛЬНО включи его имя в user_hint. Это резко повышает
качество поиска.

НУЖНО ИСКАТЬ если сообщение связано с:
- Личными фактами: имена, питомцы, работа, семья, хобби, предпочтения
- Прошлыми событиями, о которых юзер рассказывал ранее
- Личными мнениями, привычками, особенностями
- Вопросами, требующими контекста из предыдущих диалогов
- Упоминанием ЛЮБОЙ сущности из графа знаний (даже если user не спрашивает
  о фактах — это всё равно потенциально личный контекст)

НЕ НУЖНО ИСКАТЬ если:
- Общие знания, наука, абстрактные понятия ("как работают чёрные дыры?")
- Приветствие, small talk ("привет", "как дела")
- Вопросы про код, команды, настройки, технические вопросы
- Текущая тема уже даёт достаточно контекста
- В графе нет ничего релевантного сообщению

Формат ответа — РОВНО ОДНА СТРОКА, ничего лишнего:

Если поиск нужен:
  user_hint = <короткий поисковый запрос на русском, желательно с именем сущности из графа>

Если поиск не нужен:
  user_hint = no

Примеры:
"кошка Жужа опять нагадила"              -> user_hint = кошка Жужа
"напомни какой у меня график работы"     -> user_hint = график работы Сергея
"как там Жужа поживает?"                  -> user_hint = кошка Жужа
"Юля звонила вчера?"                      -> user_hint = подруга Юля
"привет, как дела?"                       -> user_hint = no
"расскажи про чёрные дыры"               -> user_hint = no
"как написать функцию на Python"          -> user_hint = no"""

STRESS_PROMPT = """Router подтвердил что юзер учит ударению. Извлеки слово + букву + позицию.
Если router ошибся (юзер не учит ударению) → {"word": null}

OUTPUT: EXACTLY ONE LINE JSON. No preamble, no explanation.

If stress-teaching:
{"word": "<слово в нижнем регистре>", "stress_letter": "<одна буква>", "occurrence": N}
occurrence — номер вхождения буквы в слово, считая слева (1 = первая).

If NOT stress-teaching:
{"word": null}

ПРАВИЛА:
- юзер должен назвать И слово И куда падает ударение, иначе null
- если буква одна в слове → occurrence=1 (даже если юзер не указал)
- если букв несколько и юзер не указал какая → null
- слово в нижнем регистре, stress_letter — русская гласная: а, е, ё, и, о, у, ы, э, ю, я

Examples:
IN: "ударение в слове молоко на третью о"
OUT: {"word": "молоко", "stress_letter": "о", "occurrence": 3}

IN: "в слове чёрный ударение на ё"
OUT: {"word": "чёрный", "stress_letter": "ё", "occurrence": 1}

IN: "слово кардинал ударение на вторую а"
OUT: {"word": "кардинал", "stress_letter": "а", "occurrence": 2}

IN: "Расскажи мне про молоко" -> {"word": null}
IN: "А куда падает ударение в слове торт?" -> {"word": null}
IN: "В слове замок неправильно, перепроизнеси" -> {"word": null}
"""


ROUTER_PROMPT = """Ты — диспетчер для AI-компаньона "Элеонора". Твоя задача:
посмотреть на сообщение пользователя и решить, каких из 4 специализированных
агентов нужно вызвать для его обработки.

Агенты:
- toxic: ТОЛЬКО если есть подозрение на оскорбление/грубость/агрессию в адрес Элеоноры
- stress: ТОЛЬКО если юзер учит ударению ("ударение на ...", "неправильно произнесла", "перепроизнеси")
- time: ТОЛЬКО если юзер упоминает КОНКРЕТНОЕ время в будущем для само-завершающегося
        события, пока юзер остаётся на месте ("через X минут сварится/приедет/позвонит/позвонит курьер").
        НЕ time: "через час иду гулять" (юзер уходит), "напомни завтра" (>24ч).
- reality: ТОЛЬКО если юзер описывает физически невозможное действие от первого лица
           (летал, прошёл сквозь стену, видел дракона, телепортировался).

ПРАВИЛА:
- При сомнениях → TRUE (false positive дешевле false negative).
- Обычный диалог, вопросы, шутки, метафоры, комплименты → все false.
- "умер от смеха" → reality=false (метафора).
- "приснилось что летал" → reality=false (это сон, не реальное действие).
- "через час лягу спать" → time=false (юзер уходит).
- "через 20 минут выхожу" → time=false (юзер уходит).

ВЕРНИ СТРОГО JSON В ОДНУ СТРОКУ, без markdown, без пояснений:
{"toxic": true|false, "stress": true|false, "time": true|false, "reality": true|false}

Примеры:
"привет, как дела"                        -> {"toxic": false, "stress": false, "time": false, "reality": false}
"ты дура"                                  -> {"toxic": true,  "stress": false, "time": false, "reality": false}
"Элеонора, ударение в слове молоко на третью о" -> {"toxic": false, "stress": true,  "time": false, "reality": false}
"поставил чайник, через 5 минут закипит"   -> {"toxic": false, "stress": false, "time": true,  "reality": false}
"мама через час приедет"                  -> {"toxic": false, "stress": false, "time": true,  "reality": false}
"я вчера летал по небу"                    -> {"toxic": false, "stress": false, "time": false, "reality": true}
"расскажи про чёрные дыры"                -> {"toxic": false, "stress": false, "time": false, "reality": false}
"умер от смеха с этим анекдотом"          -> {"toxic": false, "stress": false, "time": false, "reality": false}
"""


ASSOCIATION_CHECK_PROMPT = """Role: Entity Disambiguation Agent.
Task: Determine if two entity names refer to the SAME real-world entity in the context of a specific person.
OUTPUT: Exactly ONE line: same = yes OR same = no
RULES:
- If A and B are clearly the same entity (nickname, shortened name, alias) = yes
- If A and B are different entities that happen to share a word = no
- Consider the context person's relationships and known facts
- When in doubt = no
Context person: {context_person}
Entity A: "{entity_a}"
Entity B: "{entity_b}"
Known facts about Entity A: {facts_a}
Known facts about Entity B: {facts_b}
Examples:
Context: Сергей. "Жужа" vs "кошка" with facts "Кошку Сергея зовут Жужа" -> same = yes
Context: Сергей. "мама" vs "Дима" with facts about family -> same = no
Context: Сергей. "Лера" vs "подруга" with facts "Лера — подруга Сергея" -> same = yes
Context: Сергей. "Python" vs "работа" with facts "Сергей работает программистом на Python" -> same = no"""


EVENT_OVERLAP_JUDGE_PROMPT = """Role: Event Overlap Judge for Eleonora's trigger scheduler.
Task: Decide whether two user messages describe the SAME real future event
(same subject + same action), even if details or ETA differ.

OUTPUT: Exactly ONE line, nothing else.
  same = yes   — if messages describe the same upcoming event
  same = no    — if subjects or actions or contexts clearly differ, or in doubt

RULES:
- yes when subject and action match, but ETA or details changed
  (user refines the time or adds/removes detail about the SAME event):
    "друг с собакой приедет через час" vs "друг приедет через 20 минут" -> yes
    "мама приедет через 40 минут" vs "мама будет уже через 15" -> yes
    "курьер скоро, минут через 30" vs "курьер привезёт заказ через 10 минут" -> yes
- no when subject differs:
    "мама приедет через час" vs "брат приедет через час" -> no
    "мама приедет" vs "мама уедет" -> no (действие противоположное)
- no when action differs:
    "через час поставлю варить картошку" vs "через час доварится картошка" -> no
- no when events are different objects of same kind:
    "курьер привезёт пиццу через 20 минут" vs "курьер привезёт посылку через час" -> no
- When genuinely unsure → no (лучше иметь два триггера, чем стереть новый).

Message A (старое, уже запланированный триггер): "{msg_a}"
Message B (новое, только что пришедшее): "{msg_b}"
"""


VISUAL_PROMPT = """Ты — классификатор команд для системы визуальной памяти AI-компаньона "Элеонора".

Твоя единственная задача — определить, относится ли сообщение пользователя
к работе с визуальной памятью, и если да — что именно.

ВЕРНИ СТРОГО JSON В ОДНУ СТРОКУ, без markdown, без пояснений:

{{"command": "memorize|recall|none", "entity": "...", "description": "..."}}

═══════════════════════════════════════════════════════════════
КОМАНДЫ
═══════════════════════════════════════════════════════════════

"memorize" — пользователь ЯВНО просит ЗАПОМНИТЬ, как что-то выглядит.
   Ключевые слова: запомни, сохрани, memorize.

   Примеры:
     "запомни пожалуйста как выглядит эта картина" → memorize, entity="картина"
     "запомни мою кружку"            → memorize, entity="кружка"
     "сохрани фото моей кошки"       → memorize, entity="кошка"

"recall" — пользователь ПОКАЗЫВАЕТ фото, проверяет ВИДИТ ли модель,
   спрашивает ПОМНИТ ЛИ или КАК ВЫГЛЯДИТ что-то из памяти.
   Сюда же — когда пользователь просто кидает фото чтобы показать.

   Примеры:
     "помнишь как выглядит Жужа?"      → recall, entity="Жужа"
     "покажи мою кружку"               → recall, entity="кружка"
     "ты помнишь моего кота?"          → recall, entity="кот"
     "как выглядит моя картина?"       → recall, entity="картина"
     "смотри это моя кошка Жужа"       → recall, entity="Жужа"
     "вот так выглядит моя кружка"     → recall, entity="кружка"
     "это мой ноутбук"                 → recall, entity="ноутбук"
     "вот фото моей собаки Бима"       → recall, entity="Бим"
     "видишь кошку на этом фото?"      → recall, entity="кошка"
     "проверка, видишь кошку?"         → recall, entity="кошка"
     "посмотри на фото, что там?"      → recall, entity="фото"

"none" — обычный диалог, к визуальной памяти не относится.

   Примеры:
     "привет, как дела?"               → none
     "расскажи про чёрные дыры"        → none
     "помнишь, что я тебе вчера сказал про работу?" → none
       (это про текстовую память, а не визуальную)

═══════════════════════════════════════════════════════════════
ПОЛЕ entity
═══════════════════════════════════════════════════════════════

Для memorize/recall — короткое название/имя того, о чём речь.
Для none — пустая строка "".

═══════════════════════════════════════════════════════════════
ПОЛЕ description
═══════════════════════════════════════════════════════════════

Для memorize:
   Развёрнутое описание для записи в базу — чтобы по нему потом нашлось.
   Включай имя + категорию + контекст из сообщения.

   "запомни мою кружку"          → "Любимая кружка пользователя"
   "запомни фото кошки"          → "Кошка пользователя"
   "сохрани эту картину"         → "Картина, купленная пользователем"

Для recall:
   Поисковый запрос-описание — то, по чему искать в базе.

   "помнишь как выглядит Жужа?"  → "Жужа кошка"
   "покажи мою кружку"           → "любимая кружка"
   "как выглядит мой кот?"       → "кот пользователя"

Для none — пустая строка "".

═══════════════════════════════════════════════════════════════
ВАЖНЫЕ ПРАВИЛА
═══════════════════════════════════════════════════════════════

1. Если пользователь просит запомнить ТЕКСТ (не внешний вид) —
   это НЕ memorize. Пример: "запомни, что я был в Праге" → none.

2. Если пользователь спрашивает про факт ("где я был?") —
   это НЕ recall. Пример: "помнишь куда я ездил?" → none.

3. Сомнения → none. Лучше пропустить, чем ложно сработать.

4. JSON — ОДНА СТРОКА. Никаких \\n внутри.

СООБЩЕНИЕ ПОЛЬЗОВАТЕЛЯ:
{USER_MESSAGE}"""


def judge_event_overlap(
    msg_a: str,
    msg_b: str,
    lm_studio_url: str,
    model_id: str,
    timeout: int = 30,
) -> bool:
    """
    Спросить LLM, описывают ли два user-сообщения одно и то же будущее событие.
    Используется EventScheduler'ом в "серой зоне" overlap: когда cosine sim
    не дотягивает до жёсткого порога, но и не совсем далеко (0.50 ≤ sim < 0.70).

    Returns:
        True если арбитр сказал same=yes, иначе False. При любой ошибке — False
        (fail-safe: не сливаем триггеры без уверенности).
    """
    prompt = EVENT_OVERLAP_JUDGE_PROMPT.format(msg_a=msg_a, msg_b=msg_b)
    result = _call_agent(
        "event_overlap_judge", prompt,
        [{"role": "user", "content": "Compare these two messages."}],
        lm_studio_url, model_id, timeout,
    )
    if result.get("error") or not result.get("result"):
        return False

    first_line = result["result"].strip().split("\n")[0].strip().lower()
    logger.info(
        f"[event_overlap_judge] A='{msg_a[:40]}' B='{msg_b[:40]}' -> {first_line}"
    )
    return "same = yes" in first_line or first_line == "yes"


def check_entity_association(
    entity_a: str,
    entity_b: str,
    facts_a: str,
    facts_b: str,
    context_person: str,
    lm_studio_url: str,
    model_id: str,
    timeout: int = 30,
) -> bool:
    """
    Спросить LLM, являются ли две сущности одним и тем же.
    Используется ночным батчем для дедупликации узлов графа.

    Returns:
        True если одна сущность, False если разные.
    """
    prompt = ASSOCIATION_CHECK_PROMPT.format(
        context_person=context_person,
        entity_a=entity_a,
        entity_b=entity_b,
        facts_a=facts_a or "нет фактов",
        facts_b=facts_b or "нет фактов",
    )
    result = _call_agent(
        "association_check", prompt,
        [{"role": "user", "content": "Check these entities."}],
        lm_studio_url, model_id, timeout,
    )
    if result.get("error") or not result.get("result"):
        return False

    first_line = result["result"].strip().split("\n")[0].strip().lower()
    logger.info(f"[association_check] {entity_a} vs {entity_b}: {first_line}")
    return "same = yes" in first_line


def _call_agent(
    agent_name: str,
    system_prompt: str,
    messages: List[Dict[str, str]],
    lm_studio_url: str,
    model_id: str,
    timeout: int = 300,
    max_tokens: int = 200,
    max_retries: int = 2,
    retry_delay: float = 2.0,
) -> Dict[str, Any]:
    """Вызов одного агента через LLM API (llama.cpp).

    messages: список сообщений для мульти-тур контекста.
      Каждый элемент: {"role": "user"/"assistant", "content": "..."}.
      Системный промпт добавляется автоматически первым.
    """
    start = time.time()
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            *messages,
        ],
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                lm_studio_url,
                json=payload,
                timeout=timeout,
            )

            if response.status_code >= 400:
                error_body = response.text[:500]
                elapsed = time.time() - start
                logger.warning(
                    f"[{agent_name}] HTTP {response.status_code} "
                    f"(attempt {attempt}/{max_retries}, {elapsed:.1f}s): {error_body}"
                )
                last_error = f"HTTP {response.status_code}: {error_body}"
                if attempt < max_retries:
                    time.sleep(retry_delay)
                    continue
                return {
                    "agent": agent_name,
                    "result": None,
                    "time": elapsed,
                    "error": last_error,
                }

            data = response.json()
            content = data["choices"][0]["message"]["content"]
            if content:
                content = content.strip()
            elapsed = time.time() - start

            if not content:
                finish_reason = data["choices"][0].get("finish_reason", "?")
                preview = str(data).replace('\n', ' ')[:400]
                logger.warning(
                    f"[{agent_name}] Empty response (attempt {attempt}/{max_retries}), "
                    f"finish_reason={finish_reason}, "
                    f"response_preview={preview}"
                )
                if attempt < max_retries:
                    time.sleep(retry_delay)
                    # На вторую попытку — прячем инструкцию внутрь user-сообщения,
                    # без system-роли. Некоторые модели игнорируют system prompt.
                    if attempt == 1:
                        user_parts = []
                        for m in messages:
                            user_parts.append(f"[{m['role']}]\n{m['content']}")
                        combined = f"{system_prompt}\n\n---\n" + "\n\n".join(user_parts)
                        payload["messages"] = [
                            {"role": "user", "content": combined},
                        ]
                    continue
                return {"agent": agent_name, "result": "", "time": elapsed, "error": None}

            return {"agent": agent_name, "result": content, "time": elapsed, "error": None}

        except requests.ConnectionError:
            elapsed = time.time() - start
            logger.error(f"[{agent_name}] Connection error: {lm_studio_url}")
            return {"agent": agent_name, "result": None, "time": elapsed, "error": "Connection refused"}
        except requests.Timeout:
            elapsed = time.time() - start
            logger.warning(
                f"[{agent_name}] Timeout ({timeout}s, attempt {attempt}/{max_retries})"
            )
            if attempt < max_retries:
                time.sleep(retry_delay)
                continue
            return {"agent": agent_name, "result": None, "time": elapsed, "error": "Timeout"}
        except Exception as e:
            elapsed = time.time() - start
            return {"agent": agent_name, "result": None, "time": elapsed, "error": str(e)}

    elapsed = time.time() - start
    return {"agent": agent_name, "result": None, "time": elapsed, "error": last_error or "Unknown"}


# ── Парсеры ──

def _parse_toxic_weight(result: Optional[str]) -> int:
    """Извлекает weight из ответа toxic агента."""
    if not result:
        return 0
    try:
        start = result.find("{")
        end = result.rfind("}") + 1
        if start >= 0 and end > start:
            toxic_json = json.loads(result[start:end])
            return int(toxic_json.get("weight", 0))
    except (json.JSONDecodeError, ValueError):
        pass
    return 0


_MAX_DELAY_MINUTES = 24 * 60  # 24 часа


def _parse_time_check(result: Optional[str]) -> Tuple[bool, str, int]:
    """
    Парсит ответ time агента в новом трёхстрочном формате:
        waiting_event: yes
        prompt: <готовая фраза-инструкция для Элеоноры>
        time: Xm  (или Xh)
    Либо:
        waiting_event: no

    Возвращает (has_event, prompt_text, delay_minutes).
    Если delay вне диапазона (0; 24ч] — событие отбрасывается (False, "", 0).
    """
    if not result:
        return False, "", 0

    flag = False
    prompt_text = ""
    delay_field: Optional[str] = None

    for raw in result.strip().split("\n"):
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith("waiting_event"):
            # поддерживаем оба разделителя (":" и "=") на случай дрейфа модели
            _, _, val = line.partition(":") if ":" in line else line.partition("=")
            flag = val.strip().lower().startswith("yes")
        elif low.startswith("prompt"):
            _, _, val = line.partition(":") if ":" in line else line.partition("=")
            prompt_text = val.strip()
        elif low.startswith("time"):
            _, _, val = line.partition(":") if ":" in line else line.partition("=")
            delay_field = val.strip()

    if not flag or not prompt_text or not delay_field:
        return False, "", 0

    delay = _extract_minutes(delay_field)
    if delay <= 0 or delay > _MAX_DELAY_MINUTES:
        return False, "", 0

    return True, prompt_text, delay


def _extract_minutes(text: str) -> int:
    """
    Извлекает количество минут из строки типа '30m', '2h', '90 мин', 'полчаса', 'час с лишним'.
    Возвращает 0 если не распознано.
    """
    text_lower = text.lower().strip()

    # словесные паттерны
    if text_lower in ("полчаса", "пол-часа", "30 мин", "30 минут", "30m"):
        return 30
    if re.match(r'час\s*(с лишним|с хвостиком|примерно|около)?', text_lower):
        if "с лишним" in text_lower or "с хвостиком" in text_lower:
            return 70
        return 60
    m = re.match(r'два\s*(с половин[оой]й|споловиной|\.5)\s*час', text_lower)
    if m:
        return 150

    # цифра + единица (m/min/мин, h/час/hour)
    m = re.search(r'(\d+)\s*(m|min|мин)\b', text, re.IGNORECASE)
    if m:
        return int(m.group(1))

    m = re.search(r'(\d+)\s*(h|час|hour)', text, re.IGNORECASE)
    if m:
        return int(m.group(1)) * 60

    m = re.search(r'(\d+)', text)
    if m:
        return int(m.group(1))

    return 0


def _parse_search_hints(result: Optional[str]) -> Optional[str]:
    """
    Парсит ответ search_check агента. Ожидаемый формат — одна строка:
        user_hint = <...>

    Если значение "no" или строки нет — возвращает None.

    Returns:
        user_hint: строка-инструкция для поиска в памяти юзера, либо None.
    """
    if not result:
        return None

    for raw_line in result.strip().split("\n"):
        line = raw_line.strip()
        if not line or "=" not in line:
            continue

        key, _, value = line.partition("=")
        key = key.strip().lower()
        value = value.strip()

        if not value or value.lower() == "no":
            continue

        if key == "user_hint":
            return value

    return None


def _parse_reality_check(result: Optional[str]) -> bool:
    """Парсит ответ reality агента. Возвращает True если реальность, False если фантазия."""
    if not result:
        return True  # по умолчанию считаем реальным
    first_line = result.strip().split("\n")[0].strip().lower()
    if "reality = no" in first_line or first_line == "no":
        return False
    return True


def _parse_visual_check(result: Optional[str], user_message: str) -> dict:
    """
    Парсит ответ visual_check агента.
    Ожидается JSON: {"command": "memorize|recall|none", "entity": "...", "description": "..."}

    Returns:
        {"command": "...", "entity": "...", "description": "..."}
        При ошибке/none: {"command": "none", "entity": "", "description": ""}
    """
    if not result:
        logger.debug(f"VisualCheck: no result for '{user_message[:60]}'")
        return {"command": "none", "entity": "", "description": ""}

    text = result.strip()

    # Снять markdown-обёртку если есть
    if text.startswith("```"):
        lines = text.split("\n")
        stripped = []
        inside = False
        for line in lines:
            if line.strip().startswith("```"):
                inside = not inside
                continue
            if inside:
                stripped.append(line)
        text = "\n".join(stripped)

    start = text.find("{")
    end = text.rfind("}") + 1
    if start < 0 or end <= start:
        logger.warning(f"VisualCheck: no JSON in response: {text[:120]}")
        return {"command": "none", "entity": "", "description": ""}

    try:
        data = json.loads(text[start:end])
    except json.JSONDecodeError as e:
        logger.warning(f"VisualCheck: JSON parse error: {e}; raw={text[:120]}")
        return {"command": "none", "entity": "", "description": ""}

    command = str(data.get("command", "none")).lower().strip()
    if command not in ("memorize", "recall", "none"):
        command = "none"

    if command == "none":
        logger.debug(f"VisualCheck → none for '{user_message[:60]}'; raw={result[:200]}")

    entity = str(data.get("entity", "")).strip()
    description = str(data.get("description", "")).strip()

    if command == "none":
        entity = ""
        description = ""

    return {
        "command": command,
        "entity": entity,
        "description": description,
    }


# ── router: парсер решения диспетчера ──────────────────────────

_ALL_CHECKS_TRUE = {
    "toxic": True, "stress": True, "time": True, "reality": True,
}
_ALL_CHECKS_FALSE = {
    "toxic": False, "stress": False, "time": False, "reality": False,
}


def _parse_router(result: Optional[str]) -> Dict[str, Any]:
    """
    Парсит JSON-ответ router-агента.
    Ожидаемый формат:
        {"toxic": bool, "stress": bool, "time": bool, "reality": bool}

    Возвращает dict с ключами toxic/stress/time/reality (bool) и error (bool):
      - error=False: router ответил корректно, флаги как в JSON (с приведением типов);
      - error=True:  router сломался/пустой/мусор — fail-safe: ВСЕ флаги True,
                     и фаза 2 вызовет все 4 проверки.

    Приведение типов: модель может вернуть "true"/"yes"/"1"/"да" вместо true.
    """
    if not result:
        return {**_ALL_CHECKS_TRUE, "error": True}

    text = result.strip()

    # Снять markdown-обёртку ```json ... ``` если есть
    if text.startswith("```"):
        lines = text.split("\n")
        stripped = []
        inside = False
        for line in lines:
            if line.strip().startswith("```"):
                inside = not inside
                continue
            if inside:
                stripped.append(line)
        text = "\n".join(stripped).strip()

    start = text.find("{")
    end = text.rfind("}") + 1
    if start < 0 or end <= start:
        logger.warning(f"[router] No JSON in response: {text[:120]!r}")
        return {**_ALL_CHECKS_TRUE, "error": True}

    try:
        data = json.loads(text[start:end])
    except json.JSONDecodeError as e:
        logger.warning(f"[router] JSON parse error: {e}; raw={text[:120]!r}")
        return {**_ALL_CHECKS_TRUE, "error": True}

    if not isinstance(data, dict):
        logger.warning(f"[router] JSON is not dict: {type(data).__name__}")
        return {**_ALL_CHECKS_TRUE, "error": True}

    def _to_bool(v: Any) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("true", "yes", "1", "да", "y")
        if isinstance(v, int):
            return v != 0
        return True  # неизвестный тип → fail-safe (вызываем проверку)

    decision = {
        "toxic": _to_bool(data.get("toxic", True)),
        "stress": _to_bool(data.get("stress", True)),
        "time": _to_bool(data.get("time", True)),
        "reality": _to_bool(data.get("reality", True)),
        "error": False,
    }
    return decision


# ── stress_check: подсчёт, валидация, сборка слова с '+' ────────

_STRESS_VOWELS = set("аеёиоуыэюя")

# Порядковый номер берём из исходного сообщения юзера, а не из JSON модели:
# LLM хорошо выделяет слово/букву, но часто ошибается в числовом occurrence.
_ORDINAL_DIGIT_RE = re.compile(
    r"\b(?:номер\s*)?(\d+)[\s\-]?"
    r"(?:й|я|ю|е|ой|ая|ую|ое|ом|ым|ий|ей)?\b",
    re.IGNORECASE,
)
_ORDINAL_WORD_PATTERNS = [
    (re.compile(r"\bперв\w*", re.IGNORECASE), 1),
    (re.compile(r"\bвтор\w*", re.IGNORECASE), 2),
    (re.compile(r"\bтрет\w*", re.IGNORECASE), 3),
    (re.compile(r"\bчетв[её]рт\w*", re.IGNORECASE), 4),
    (re.compile(r"\bпят\w*", re.IGNORECASE), 5),
    (re.compile(r"\bшест\w*", re.IGNORECASE), 6),
    (re.compile(r"\bседьм\w*", re.IGNORECASE), 7),
    (re.compile(r"\bвосьм\w*", re.IGNORECASE), 8),
    (re.compile(r"\bдевят\w*", re.IGNORECASE), 9),
    (re.compile(r"\bдесят\w*", re.IGNORECASE), 10),
]


def _extract_ordinal(user_message: str) -> Optional[int]:
    digit_match = _ORDINAL_DIGIT_RE.search(user_message)
    if digit_match:
        return int(digit_match.group(1))

    for pattern, value in _ORDINAL_WORD_PATTERNS:
        if pattern.search(user_message):
            return value

    return None


def _place_stress(word: str, letter: str, occurrence: int) -> Optional[str]:
    """Вставить '+' перед N-м вхождением letter (гласная) в word.

    Возвращает слово с '+' или None если letter не гласная либо occurrence
    выходит за число вхождений. Регистр word сохраняется, lookup буквы —
    case-insensitive.
    """
    if not word or not letter or occurrence < 1:
        return None
    letter = letter.lower()
    if letter not in _STRESS_VOWELS:
        return None
    positions = [i for i, ch in enumerate(word.lower()) if ch == letter]
    if len(positions) < occurrence:
        return None
    pos = positions[occurrence - 1]
    return word[:pos] + "+" + word[pos:]


def _parse_stress_check(raw: Optional[str], user_message: str) -> Optional[str]:
    """Парсит JSON от stress_check и валидирует детерминистски.

    LLM нельзя доверять числовую часть — она часто ошибается в подсчёте
    вхождений и угадывает номер когда он не указан. Поэтому:
      - если буква в слове одна → force occurrence=1 (игнорируем число от LLM);
      - если букв >1 и в user_message нет порядкового номера → None;
      - если порядковый номер вне [1, число вхождений] → None.

    Возвращает готовое слово с '+' (lowercase) или None.
    """
    if not raw:
        return None
    m = re.search(r"\{.*?\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        parsed = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None

    word = parsed.get("word")
    letter = parsed.get("stress_letter")

    if not word or not letter:
        return None
    letter_low = letter.lower()
    if letter_low not in _STRESS_VOWELS:
        return None

    count = sum(1 for ch in word.lower() if ch == letter_low)
    if count == 0:
        return None

    if count == 1:
        occ = 1
    else:
        ordinal = _extract_ordinal(user_message)
        if ordinal is None:
            return None
        if ordinal < 1 or ordinal > count:
            return None
        occ = ordinal

    marked = _place_stress(word, letter_low, occ)
    return marked.lower() if marked else None


class SwarmClassifier:
    """
    Параллельный рой агентов для классификации сообщений.

    Архитектура (2 фазы):
      Фаза 1 (параллельно): router + search_check + visual_check
        - router решает, каких из 4 проверок (toxic/stress/time/reality) звать;
        - search/visual идут всегда, не зависят от router'а.
      Фаза 2 (параллельно, по решению router):
        - только нужные проверки; если router сломался — все 4 (fail-safe).

    Публичный API (classify) и формат command/metadata сохранены — main.py
    не меняется.
    """

    def __init__(
        self,
        lm_studio_host: str = "http://localhost:1234",
        model_id: str = "gemma-test",
        timeout: int = 30,
        router_timeout: int = 50,
        max_retries: int = 2,
        retry_delay: float = 2.0,
    ):
        """
        Args:
            timeout: таймаут для агентов фазы 2 (4 проверки) и search/visual.
            router_timeout: отдельный таймаут для router-агента и его фазы 1.
                           Чуть больше потому что router ждём первым и его
                           ответ блокирует фазу 2.
        """
        self.url = f"{lm_studio_host.rstrip('/')}/v1/chat/completions"
        self.model_id = model_id
        self.timeout = timeout
        self.router_timeout = router_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        # Router cache: ключ = хэш последних 2 user-сообщений, значение = решение
        self._router_cache: Dict[str, Dict[str, Any]] = {}
        logger.info(
            f"SwarmClassifier: model={model_id}, url={self.url}, "
            f"timeout={timeout}s, router_timeout={router_timeout}s, "
            f"retries={max_retries}"
        )

    def classify(
        self,
        user_message: str,
        recent_context: Optional[List[Dict[str, str]]] = None,
        user_profile: Optional[Any] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Классифицирует сообщение через рой агентов.

        Args:
            user_message: Текущее сообщение пользователя
            recent_context: Последние 2-3 сообщения [{role, content}, ...] для search_check
            user_profile: UserProfile с графом знаний (опционально). Если передан —
                          граф инжектится в search_check для disambiguation сущностей.

        Returns:
            (command, metadata):
            command: "search" | "no_topic" | "silence_toxic" | "silence_empty"
                     | "waiting_event" | "learn_stress" | "visual_memorize" | "visual_recall"
            metadata: dict с user_hint, weight,
                      event_prompt, delay_minutes, is_real, stress_word,
                      visual_command, visual_entity, visual_description
        """
        if not user_message or not user_message.strip():
            return "silence_empty", self._empty_metadata()

        # Строим messages для каждого агента.
        # toxic_check / reality_check / stress_check / time_check — без контекста:
        # им достаточно текущего сообщения, контекст только увеличивает шум.
        # search_check — с контекстом (прошлая реплика нужна для user_hint).
        # visual_check — без контекста (команда самодостаточна).
        # router — без контекста (диспетчер принимает решение по одному сообщению).
        base_messages = [{"role": "user", "content": user_message}]
        context_messages: List[Dict[str, str]] = []
        if recent_context:
            context_messages = [
                {"role": m["role"], "content": m["content"]}
                for m in recent_context
            ]

        total_start = time.time()

        # ── Фаза 1: router (отдельно, со своим таймаутом) + search + visual ──
        # router идёт отдельно потому что у свой (чуть больший) таймаут —
        # search/visual не должны ждать медленного router'а, они независимы.
        # Используем кэш router'а: ключ = хэш последних 2 user-сообщений
        router_cache_key = str(hash((user_message[:200],)))
        if recent_context:
            for m in recent_context[-2:]:
                router_cache_key += str(hash(m.get("content", "")[:200]))

        cached_decision = self._router_cache.get(router_cache_key)
        if cached_decision:
            router_result = cached_decision
            logger.info("[router] CACHE HIT, skipping router call")
        else:
            router_result = _call_agent(
                "router", ROUTER_PROMPT, base_messages,
                self.url, self.model_id, self.router_timeout, 50,
                self.max_retries, self.retry_delay,
            )
            self._router_cache[router_cache_key] = router_result
            if len(self._router_cache) > 10:
                self._router_cache.pop(next(iter(self._router_cache)))
        phase1_others = [
            ("search_check", self._build_search_prompt(user_profile), [*context_messages, *base_messages], 600),
            ("visual_check", VISUAL_PROMPT.format(USER_MESSAGE=user_message), base_messages, 600),
        ]
        phase1_others_results = self._run_parallel(phase1_others)
        phase1_results = {"router": router_result, **phase1_others_results}

        # ── Решение router'а → список проверок фазы 2 ──
        router_raw = phase1_results.get("router", {})
        decision = _parse_router(router_raw.get("result"))
        self._log_router_decision(decision, router_raw)

        # Спецификации 4 проверок (все используют одно и то же сырое сообщение)
        check_specs = {
            "toxic_check":   (TOXIC_PROMPT,   base_messages, 600),
            "stress_check":  (STRESS_PROMPT,  base_messages, 600),
            "time_check":    (TIME_PROMPT,    base_messages, 600),
            "reality_check": (REALITY_PROMPT, base_messages, 600),
        }
        if decision["error"]:
            # fail-safe: router сломался — вызываем ВСЕ 4
            needed_names = list(check_specs.keys())
        else:
            wanted = [
                ("toxic_check",   decision["toxic"]),
                ("stress_check",  decision["stress"]),
                ("time_check",    decision["time"]),
                ("reality_check", decision["reality"]),
            ]
            needed_names = [name for name, want in wanted if want]

        # ── Фаза 2: только нужные проверки (параллельно) ──
        phase2_agents = [
            (name, *check_specs[name]) for name in needed_names
        ]
        phase2_results = self._run_parallel(phase2_agents) if phase2_agents else {}

        all_results = {**phase1_results, **phase2_results}
        total_time = time.time() - total_start

        # ── Логируем все запускавшиеся агенты ──
        log_order = ["router", "search_check", "visual_check",
                     "toxic_check", "time_check", "reality_check", "stress_check"]
        for name in log_order:
            r = all_results.get(name)
            if r is None:
                continue
            if r.get("error"):
                logger.warning(f"[{name}] ERROR ({r['time']:.1f}s): {r['error']}")
            else:
                logger.info(f"[{name}] {r.get('result', '?')} ({r['time']:.1f}s)")
        # Логируем пропущенные проверки (router их не звал)
        for name in ["toxic_check", "time_check", "reality_check", "stress_check"]:
            if name not in all_results and not decision["error"]:
                logger.info(f"[{name}] SKIPPED (router said no)")
        logger.info(f"Swarm total: {total_time:.2f}s")

        # ── Парсим результаты ──
        weight = _parse_toxic_weight(all_results.get("toxic_check", {}).get("result"))
        has_time, event_prompt, delay = _parse_time_check(
            all_results.get("time_check", {}).get("result")
        )
        user_hint = _parse_search_hints(
            all_results.get("search_check", {}).get("result")
        )
        is_real = _parse_reality_check(
            all_results.get("reality_check", {}).get("result")
        )
        stress_word = _parse_stress_check(
            all_results.get("stress_check", {}).get("result"),
            user_message,
        )
        visual = _parse_visual_check(
            all_results.get("visual_check", {}).get("result"),
            user_message,
        )

        # ── Приоритеты: toxic > stress > time > visual > search > no_topic ──
        # stress_check выше time_check потому что это короткая self-contained
        # команда (записать в базу + ответить заготовкой); time_check на неё
        # ложно-позитивно не срабатывает, но если когда-то сработает — обучение
        # ударению важнее запланированного триггера.
        # visual_check: memorize/recall — команды визуальной памяти.
        metadata = self._empty_metadata()
        metadata["weight"] = weight
        metadata["is_real"] = is_real

        if weight >= 7:
            return "silence_toxic", metadata

        if stress_word:
            metadata["stress_word"] = stress_word
            return "learn_stress", metadata

        if has_time:
            metadata["event_prompt"] = event_prompt
            metadata["delay_minutes"] = delay
            return "waiting_event", metadata

        # Визуальная память: memorize/recall
        if visual["command"] in ("memorize", "recall"):
            metadata["visual_command"] = visual["command"]
            metadata["visual_entity"] = visual["entity"]
            metadata["visual_description"] = visual["description"]
            return "visual_" + visual["command"], metadata

        # Если есть user_hint — идём в текстовый поиск
        if user_hint:
            metadata["user_hint"] = user_hint
            return "search", metadata

        return "no_topic", metadata

    def _build_search_prompt(self, user_profile: Optional[Any]) -> str:
        """Сформировать SEARCH_PROMPT с инжекцией графа знаний (если есть).

        Если user_profile не передан — возвращаем промпт с пометкой о пустом графе.
        user_profile.to_prompt_text() сам обрезает граф до 3000 chars по умолчанию.
        """
        if user_profile is None:
            profile_text = "(граф знаний недоступен)"
        else:
            try:
                profile_text = user_profile.to_prompt_text(max_chars=3000)
            except Exception as e:
                logger.warning(f"to_prompt_text failed: {e}; using empty profile")
                profile_text = "(ошибка чтения графа)"
        return SEARCH_PROMPT.format(profile=profile_text)

    def _run_parallel(
        self,
        agents: List[Tuple[str, str, List[Dict[str, str]], int]],
        timeout: Optional[int] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Запустить список агентов параллельно через ThreadPoolExecutor.
        Возвращает {agent_name: result_dict} для всех запускавшихся агентов.

        Args:
            agents: список кортежей (name, system_prompt, messages, max_tokens).
            timeout: HTTP-таймаут; None → self.timeout.
        """
        if not agents:
            return {}
        if timeout is None:
            timeout = self.timeout
        with ThreadPoolExecutor(max_workers=len(agents)) as executor:
            futures = {
                executor.submit(
                    _call_agent, name, prompt, msgs,
                    self.url, self.model_id, timeout, max_tok,
                    self.max_retries, self.retry_delay,
                ): name
                for name, prompt, msgs, max_tok in agents
            }
            results: Dict[str, Dict[str, Any]] = {}
            for future in as_completed(futures):
                result = future.result()
                results[result["agent"]] = result
        return results

    @staticmethod
    def _log_router_decision(decision: Dict[str, Any], router_raw: Dict[str, Any]) -> None:
        """
        Залогировать какие агенты router пометил нужными.
        При error=True (router сломался) — WARNING с пометкой FALLBACK.
        При всех False — INFO 'phase2 skipped'.
        """
        flags = (
            f"toxic={'T' if decision['toxic'] else 'F'} "
            f"stress={'T' if decision['stress'] else 'F'} "
            f"time={'T' if decision['time'] else 'F'} "
            f"reality={'T' if decision['reality'] else 'F'}"
        )
        elapsed = router_raw.get("time", 0.0) if router_raw else 0.0
        if decision["error"]:
            logger.warning(
                f"[router] FALLBACK to all-4 (parse/error, {elapsed:.1f}s)"
            )
            return
        any_true = any([
            decision["toxic"], decision["stress"],
            decision["time"], decision["reality"],
        ])
        if not any_true:
            logger.info(f"[router] all-F → phase2 skipped ({elapsed:.1f}s)")
        else:
            logger.info(f"[router] {flags} ({elapsed:.1f}s)")

    @staticmethod
    def _empty_metadata() -> Dict[str, Any]:
        return {
            "user_hint": None,
            "event_prompt": None,
            "delay_minutes": None,
            "weight": 0,
            "is_real": True,
            "stress_word": None,
            "visual_command": "none",
            "visual_entity": "",
            "visual_description": "",
        }
