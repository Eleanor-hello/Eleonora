// Text-To-Speech Preprocessor ported from tts/sanitizer.py & tts/preprocessor.py

const CODE_BLOCK = /```[\s\S]*?```/g;
const CODE_INLINE = /`([^`]+?)`/g;
const BOLD = /\*\*(.+?)\*\*/g;
const BOLD_UNDERSCORE = /__(.+?)__/g;
const ITALIC = /(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g;
const ITALIC_UNDERSCORE = /(?<![\w_])_(?!_)(.+?)(?<!_)_(?![\w_])/g;
const HEADER = /^\s*#{1,6}\s+/gm;
const BULLET = /^\s*[-*]\s+/gm;
const URL_RE = /https?:\/\/\S+/g;
const EMOJI = /[\u{1F000}-\u{1FAFF}\u{2600}-\u{27BF}\u{FE0F}\u{200D}]+/gu;
const WS = /\s+/g;

export function sanitizeForTts(text: string): string {
  if (!text) return '';

  let res = text;
  res = res.replace(CODE_BLOCK, ' ');
  res = res.replace(CODE_INLINE, '$1');
  res = res.replace(BOLD, '$1');
  res = res.replace(BOLD_UNDERSCORE, '$1');
  res = res.replace(ITALIC, '$1');
  res = res.replace(ITALIC_UNDERSCORE, '$1');
  res = res.replace(HEADER, '');
  res = res.replace(BULLET, '');
  res = res.replace(URL_RE, 'ссылка');
  res = res.replace(EMOJI, ' ');
  return res.replace(WS, ' ').trim();
}

// Common project yofication rules (from yo_overrides.txt and key dictionary entries)
const YO_RULES: Record<string, string> = {
  'сережа': 'серёжа',
  'сережи': 'серёжи',
  'сереже': 'серёже',
  'сережу': 'серёжу',
  'сережей': 'серёжей',
  'полет': 'полёт',
  'полета': 'полёта',
  'полету': 'полёту',
  'полетом': 'полётом',
  'полете': 'полёте',
  'ее': 'её',
  'все': 'всё',
  'еще': 'ещё',
  'моем': 'моём',
  'твоем': 'твоём',
  'своем': 'своём',
  'звезды': 'звёзды',
  'звезда': 'звезда',
  'черный': 'чёрный',
  'черные': 'чёрные',
  'желтый': 'жёлтый',
  'самолет': 'самолёт',
  'вертолет': 'вертолёт',
  'котенок': 'котёнок',
  'ребенок': 'ребёнок',
};

export function yoficate(text: string): string {
  return text.replace(/[А-Яа-яЁё]+/g, (word) => {
    const low = word.toLowerCase().replace(/ё/g, 'е');
    const replacement = YO_RULES[low];
    if (!replacement) return word;

    // Match case
    if (word === word.toUpperCase()) return replacement.toUpperCase();
    if (word[0] === word[0].toUpperCase()) {
      return replacement[0].toUpperCase() + replacement.slice(1);
    }
    return replacement;
  });
}

export function applyStress(text: string, stressMap: Record<string, string>): string {
  if (!stressMap || Object.keys(stressMap).length === 0) return text;

  return text.replace(/[А-Яа-яЁё]+/g, (word) => {
    const low = word.toLowerCase();
    const marked = stressMap[low];
    if (!marked) return word;

    // Return the marked word or stress-emphasized version
    return marked;
  });
}
