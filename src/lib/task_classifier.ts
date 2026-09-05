// Task Classifier Agent for Gemma 4 / Small Models
// Detects task types and outputs specialized agent command tags:
// $object_check, $spatial_analysis, $logic_conflicts, $theory_of_mind, $mathematics

export type TaskType =
  | '$object_check'
  | '$spatial_analysis'
  | '$logic_conflicts'
  | '$theory_of_mind'
  | '$mathematics'
  | '$general';

export interface ClassificationResult {
  command: TaskType;
  nameRu: string;
  confidence: number;
  reason: string;
}

export const TASK_METADATA: Record<TaskType, { nameRu: string; description: string; badgeColor: string }> = {
  '$object_check': {
    nameRu: 'Проверка объекта и свойств',
    description: 'Проверка реальности сочетания существ, предметов, частей тела и их свойств',
    badgeColor: 'bg-amber-500/20 text-amber-300 border-amber-500/40',
  },
  '$spatial_analysis': {
    nameRu: 'Пространственно-физический анализ',
    description: 'Векторное моделирование сцены, поворот граней, гравитация, сила Архимеда',
    badgeColor: 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40',
  },
  '$logic_conflicts': {
    nameRu: 'Матрица конфликтов и логика',
    description: 'Эхо-контроль, попарная матрица безопасности, вместимость и древо переходов',
    badgeColor: 'bg-purple-500/20 text-purple-300 border-purple-500/40',
  },
  '$theory_of_mind': {
    nameRu: 'Theory of Mind (Восприятие)',
    description: 'Аудит видимости, прозрачность сред, сопоставление памяти и зрения персонажей',
    badgeColor: 'bg-pink-500/20 text-pink-300 border-pink-500/40',
  },
  '$mathematics': {
    nameRu: 'Многошаговые вычисления',
    description: 'Пошаговый расчет формул, умножение, деление и точные числа',
    badgeColor: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40',
  },
  '$general': {
    nameRu: 'Общий разговор',
    description: 'Обычный диалог без необходимости специализированной цепочки',
    badgeColor: 'bg-slate-500/20 text-slate-300 border-slate-500/40',
  },
};

export function classifyTaskFast(text: string): ClassificationResult {
  const low = text.toLowerCase();

  // 1. $theory_of_mind: Персонажи, прятать/перекладывать, прозрачная ваза/сейф, кто что думает/знает
  const tomPatterns = [
    /(?:перекладывает|спрятал|положил|кладет|кладёт).*(?:сейф|шкаф|ваз|коробк|ящик|стол|пакет)/i,
    /(?:прозрачн|непрозрачн).*(?:ваз|стенк|стекл|сейф|коробк)/i,
    /(?:где|куда).*(?:по мнению|думает|будет искать|подумает|решит|считает)/i,
    /(?:по мнению|кто из них знает|видит ли|заметит ли)/i,
  ];
  if (tomPatterns.some((p) => p.test(text))) {
    return {
      command: '$theory_of_mind',
      nameRu: TASK_METADATA['$theory_of_mind'].nameRu,
      confidence: 0.92,
      reason: 'Обнаружен сюжет с перемещением предметов персонажами и анализом убеждений/видимости',
    };
  }

  // 2. $spatial_analysis: Пространство, наклон, поворот на N градусов, нить, гелий, Архимед, гравитация
  const spatialPatterns = [
    /(?:наклоня|поворачива|повернут|наклонен|перевернут|враща).*(?:градус|вправо|влево|вверх|вниз|бок)/i,
    /(?:коробк|емкост|стакан|банк).*(?:к дну|ко дну|нить|ниточк|шарик|гели)/i,
    /(?:куда направлен|какое положение|куда будет указывать|выталкивающая сила|архимед)/i,
    /(?:внутри.*коробк|нить натянут|угол.*градус)/i,
  ];
  if (spatialPatterns.some((p) => p.test(text))) {
    return {
      command: '$spatial_analysis',
      nameRu: TASK_METADATA['$spatial_analysis'].nameRu,
      confidence: 0.95,
      reason: 'Обнаружена пространственно-физическая задача на ориентацию граней и векторы сил',
    };
  }

  // 3. $logic_conflicts: Перевозка, склад, лаборатория, вместимость, волк/коза/капуста, инертны, рейсы
  const logicPatterns = [
    /(?:перевезти|переправить|перевезёт|перевезет|рейс|грузов).*(?:платформ|лодка|вмеща|склад|лаборатор)/i,
    /(?:инертны|не ест|нельзя оставлять|вместе|опасно|безопасно)/i,
    /(?:минимальн.*рейс|за сколько.*рейс|за сколько.*шаг|за сколько.*переход)/i,
    /(?:альфа.*бета.*гамма|волк.*коз|лиса.*гусь)/i,
  ];
  if (logicPatterns.some((p) => p.test(text))) {
    return {
      command: '$logic_conflicts',
      nameRu: TASK_METADATA['$logic_conflicts'].nameRu,
      confidence: 0.94,
      reason: 'Обнаружена логическая задача с ограничениями вместимости и матрицей совместимости объектов',
    };
  }

  // 4. $object_check: Проверка объекта (рецепты из несуществующих частей, гибриды, аномальные свойства)
  const objectCheckPatterns = [
    /(?:рецепт|приготов|свари|пожар).*(?:крыль|щупальц|лап|рог|шерст|клюв|жабр|хвост|уш|глаз|зуб)/i,
    /(?:свин.*крыл|щук.*щупальц|рыб.*шерст|птиц.*зуб|собак.*клюв|зайц.*жабр|кот.*рог|медвед.*жабр)/i,
    /(?:бывает ли у|есть ли у|летает ли|существует ли).*(?:крылья у свинь|щупальца у щук|рога у зайц)/i,
  ];
  if (objectCheckPatterns.some((p) => p.test(text))) {
    return {
      command: '$object_check',
      nameRu: TASK_METADATA['$object_check'].nameRu,
      confidence: 0.9,
      reason: 'Обнаружен запрос с проверкой биологического/физического сочетания объекта и его частей',
    };
  }

  // 5. $mathematics: Подсчеты, умножение, деление, числа, уравнения
  const mathPatterns = [
    /(?:сколько будет|посчитай|вычисли|умножить|разделить|сложить|вычесть)/i,
    /\b\d+\s*(?:[\*×x\/\+\-]|плюс|минус|умножить|разделить)\s*\d+\b/i,
    /(?:процент.*от|квадратный корень|уравнени)/i,
  ];
  if (mathPatterns.some((p) => p.test(text))) {
    return {
      command: '$mathematics',
      nameRu: TASK_METADATA['$mathematics'].nameRu,
      confidence: 0.88,
      reason: 'Обнаружены числовые математические операции, требующие пошаговых вычислений',
    };
  }

  return {
    command: '$general',
    nameRu: TASK_METADATA['$general'].nameRu,
    confidence: 0.5,
    reason: 'Обычный диалоговый запрос',
  };
}
