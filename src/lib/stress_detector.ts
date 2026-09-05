// Stress check agent logic ported from agents/stress_check.py

const VOWELS = "аеёиоуыэюя";
const VOWELS_UPPER = "АЕЁИОУЫЭЮЯ";

// Fast path 1: Marked word with + ("молок+о")
const MARKED_RE = /\b([а-яёА-ЯЁ]+\+[а-яёА-ЯЁ]+)\b/;

// Fast path 2: "не СЛОВО1, а слОво2"
const NOT_A_RE = /[Нн]е\s+([а-яёА-ЯЁ]+)\s*,?\s*а\s+(?:правильно\s+)?([а-яёА-ЯЁ]+)/;

// Keywords indicating pronunciation teaching
const STRESS_KEYWORDS = ["ударени", "произнос", "произнес", "перепроизнос", "говор", "озвуч"];

export interface StressDetectionResult {
  marked: string;
  bare: string;
}

export function detectStressFast(text: string): StressDetectionResult | null {
  // Check fast path 1: "элеон+ора", "молок+о"
  const m1 = text.match(MARKED_RE);
  if (m1) {
    const marked = m1[1].toLowerCase();
    const bare = marked.replace(/\+/g, "");
    return { marked, bare };
  }

  // Check fast path 2: "не звонИт, а звОнит"
  const m2 = text.match(NOT_A_RE);
  if (m2) {
    const right = m2[2];
    const accents: number[] = [];
    for (let i = 1; i < right.length; i++) {
      if (VOWELS_UPPER.includes(right[i])) {
        accents.push(i);
      }
    }
    if (right.length >= 3 && accents.length === 1) {
      const idx = accents[0];
      const low = right.toLowerCase();
      const marked = low.slice(0, idx) + "+" + low.slice(idx);
      return { marked, bare: low };
    }
  }

  // Natural language pattern: "ударение в слове [слово] на [букву/слог/позицию]"
  const nlMatch = text.match(/ударени[ея]?\s+(?:в\s+слове\s+)?([а-яёА-ЯЁ]+)\s+(?:падает\s+)?на\s+(?:букву\s+)?([а-яёА-ЯЁ])/i);
  if (nlMatch) {
    const word = nlMatch[1].toLowerCase();
    const letter = nlMatch[2].toLowerCase();
    if (VOWELS.includes(letter)) {
      const idx = word.lastIndexOf(letter);
      if (idx !== -1) {
        const marked = word.slice(0, idx) + "+" + word.slice(idx);
        return { marked, bare: word };
      }
    }
  }

  return null;
}

export function looksLikeStressRequest(text: string): boolean {
  const low = text.toLowerCase();
  return STRESS_KEYWORDS.some((k) => low.includes(k));
}
