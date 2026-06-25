/**
 * LLM-редактор для Google Docs
 *
 * Надёжная версия (исправлен рассинхрон контракта с backend):
 * - подменю в Google Docs
 * - единый MODE_CONFIG
 * - явные handler-функции
 * - настройки температуры по режимам
 * - intent отправляется ТОЛЬКО если он входит в ALLOWED_INTENTS backend
 * - domain соответствует реальным файлам config/domains/*.json
 * - жанровая специфика дополнительно усиливается через overlays
 *
 * ВАЖНО про синхронизацию с backend (после расширения shared_contracts.py):
 *   POST /api/edit принимает:
 *     domain   : marketing | blog | deai | basic_edit | logic_edit |
 *                nora_gal | nora_gal_soft | cutnoise | makeclear |
 *                restructure | readerfirst | genre | fiction | composition |
 *                balanced_edit
 *     intent   : analytical | marketingpush | storytelling | engagement | (отсутствует)
 *                (все четыре имеют файлы config/intents/*.json с плоским
 *                 списком инструкций; noragal/deai как intent всё ещё дают 500 —
 *                 их специфика выражается через domain)
 *     overlays : base | infostyle | landing | coldemail | pressrelease | workdoc | ...
 *                base    — нейтральный, не навязывает стиль
 *                infostyle — информационный деловой стиль
 *     output_mode : 'text_only' | 'text_and_report'
 */




/* ============================================================================
 * SECTION: BACKEND
 * ============================================================================
 */




const BACKEND_CONFIG = {
  url: 'https://google-docs-editor-backend.onrender.com/api/edit',
  provider: 'openrouter',
  temperature: 0.3
};




const DEFAULT_AUDIENCE = {
  kind: 'b2b',
  expertise: 'pro',
  formality: 'neutral',
  description: 'Редактор текста в Google Docs'
};




/**
 * intent-значения, которые backend реально принимает БЕЗ ошибок.
 * Любой intent вне этого набора в payload НЕ попадёт (см. buildPayload_).
 *
 * Все четыре имеют файлы config/intents/*.json с плоским списком инструкций.
 * 'neutral' эквивалентен отсутствию intent (backend трактует его как None).
 * ТРЕБОВАНИЕ: файлы интентов должны быть задеплоены на сервер ДО расширения
 * ALLOWED_INTENTS в backend, иначе отсутствие файла даёт HTTP 500.
 */
const ALLOWED_INTENTS = [
  'analytical',
  'marketingpush',
  'storytelling',
  'engagement',
  'neutral'
];




/* ============================================================================
 * SECTION: MENU
 * ============================================================================
 */




const MENU_GROUP_ORDER = [
  'marketing',
  'blog',
  'editing',
  'cleanup',
  'genre',
  'creative'
];




const MENU_GROUP_TITLES = {
  marketing: 'Маркетинг',
  blog: 'Блог и соцсети',
  editing: 'Правка и стиль',
  cleanup: 'Чистка и структура',
  genre: 'Жанры',
  creative: 'Творческие режимы'
};




/* ============================================================================
 * SECTION: MODES
 *
 * Поля:
 *   domain      — ТОЛЬКО 'marketing' | 'blog' | 'deai'
 *   intent      — ТОЛЬКО 'storytelling' или null
 *   overlays    — массив overlay-тегов (несут жанровую/стилевую специфику)
 *                 'base'      — нейтральный, без стилевого уклона
 *                 'infostyle' — информационный деловой стиль
 *   output_mode — 'text_only' | 'text_and_report'
 * ============================================================================
 */




const MODE_CONFIG = {
  marketing_analytical: {
    menu: 'Аналитично',
    group: 'marketing',
    order: 10,
    domain: 'marketing',
    intent: 'analytical',    // файл config/intents/analytical.json
    overlays: ['infostyle'],
    temperature: 0.3,
    handler: 'editSelection_marketing_analytical'
  },




  marketing_push: {
    menu: 'Продающий',
    group: 'marketing',
    order: 20,
    domain: 'marketing',
    intent: 'marketingpush', // файл config/intents/marketingpush.json (было 'marketing_push' — источник 422)
    overlays: ['infostyle'],
    temperature: 0.6,
    handler: 'editSelection_marketing_push'
  },




  marketing_analysis: {
    menu: 'Правка с объяснением',
    group: 'marketing',
    order: 30,
    domain: 'marketing',
    intent: 'analytical',    // файл config/intents/analytical.json
    overlays: ['infostyle'],
    output_mode: 'text_and_report',
    temperature: 0.35,
    handler: 'analyzeSelection_marketing'
  },




  blog_opinion: {
    menu: 'Обычный режим',
    group: 'blog',
    order: 10,
    domain: 'blog',
    intent: 'analytical',    // файл config/intents/analytical.json
    overlays: ['infostyle'],
    temperature: 0.4,
    handler: 'editSelection_blog_opinion'
  },




  blog_engagement: {
    menu: 'Повысить вовлечённость',
    group: 'blog',
    order: 20,
    domain: 'blog',
    intent: 'engagement',    // файл config/intents/engagement.json (профильный интент для этого режима)
    overlays: ['base'],      // base: вовлечённость — живой стиль, деловой уклон мешает
    temperature: 0.65,
    handler: 'editSelection_blog_engagement'
  },




  basic_edit: {
    menu: 'Базовая правка',
    group: 'editing',
    order: 10,
    domain: 'basic_edit',    // реальный файл config/domains/basic_edit.json
    intent: null,
    overlays: ['base'],      // base: базовая правка не должна навязывать деловой стиль
    temperature: 0.4,
    handler: 'editSelection_basic_edit'
  },




  logic_edit: {
    menu: 'Проверка логики',
    group: 'editing',
    order: 20,
    domain: 'logic_edit',    // реальный файл config/domains/logic_edit.json
    intent: null,
    overlays: ['base'],      // base: логическая правка стиленейтральна
    temperature: 0.3,
    handler: 'editSelection_logic_edit'
  },




  nora_gal: {
    menu: 'Правка по Норе Галь',
    group: 'editing',
    order: 30,
    domain: 'nora_gal',      // реальный файл config/domains/nora_gal.json
    intent: null,            // intent 'noragal' даёт 500; принципы Норы Галь несёт domain
    overlays: ['base'],      // base вместо infostyle: у Норы Галь своя философия стиля
    temperature: 0.5,
    handler: 'editSelection_nora_gal'
  },




  nora_gal_soft: {
    menu: 'Правка по Норе Галь — бережно',
    group: 'editing',
    order: 40,
    domain: 'nora_gal_soft', // реальный файл config/domains/nora_gal_soft.json
    intent: null,
    overlays: ['base'],      // base вместо infostyle: та же логика, что у nora_gal
    temperature: 0.45,
    handler: 'editSelection_nora_gal_soft'
  },




  balanced_edit: {
    menu: 'Взвешенная правка',
    group: 'editing',
    order: 45,
    domain: 'balanced_edit', // реальный файл config/domains/balanced_edit.json
    intent: null,            // без интента — максимальная предсказуемость
    overlays: ['base'],      // base: взвешенная правка универсальна, не привязана к деловому стилю
    temperature: 0.35,
    handler: 'editSelection_balanced_edit'
  },




  deai: {
    menu: 'Убрать признаки ИИ',
    group: 'editing',
    order: 50,
    domain: 'deai',          // валидный domain — оставляем
    intent: null,            // intent 'deai' даёт HTTP 500, поэтому не отправляем
    overlays: ['infostyle'],
    output_mode: 'text_and_report',
    temperature: 0.2,
    handler: 'editSelection_deai'
  },




  // --- режимы «Пиши, сокращай 2025» в группе cleanup ---




  readerfirst: {
    menu: 'Фокус на читателе',
    group: 'cleanup',
    order: 10,
    domain: 'readerfirst',   // реальный файл config/domains/readerfirst.json
    intent: null,
    overlays: ['infostyle'], // FIX: убран несуществующий overlay 'readerfocus' (давал 422)
    temperature: 0.25,
    handler: 'editSelection_readerfirst'
  },




  cutnoise: {
    menu: 'Убрать мусор',
    group: 'cleanup',
    order: 20,
    domain: 'cutnoise',      // реальный файл config/domains/cutnoise.json
    intent: null,
    overlays: ['infostyle'],
    temperature: 0.15,
    handler: 'editSelection_cutnoise'
  },




  makeclear: {
    menu: 'Упростить предложения',
    group: 'cleanup',
    order: 30,
    domain: 'makeclear',     // реальный файл config/domains/makeclear.json
    intent: null,
    overlays: ['infostyle'],
    temperature: 0.2,
    handler: 'editSelection_makeclear'
  },




  restructure: {
    menu: 'Перестроить структуру',
    group: 'cleanup',
    order: 40,
    domain: 'restructure',   // реальный файл config/domains/restructure.json
    intent: null,
    overlays: ['base'],      // FIX: убран несуществующий overlay 'structurefirst' (давал 422)
    temperature: 0.2,
    handler: 'editSelection_restructure'
  },




  // --- жанровые режимы: domain приведён к допустимому, жанр несут overlays ---




  genre_coldemail: {
    menu: 'Холодное письмо',
    group: 'genre',
    order: 10,
    domain: 'genre',         // реальный файл config/domains/genre.json
    intent: 'analytical',    // файл config/intents/analytical.json; жанр усилен overlay coldemail
    overlays: ['infostyle', 'coldemail'],
    temperature: 0.3,
    handler: 'editSelection_genre_coldemail'
  },




  genre_pressrelease: {
    menu: 'Пресс-релиз / новость',
    group: 'genre',
    order: 20,
    domain: 'genre',         // реальный файл config/domains/genre.json
    intent: 'analytical',    // файл config/intents/analytical.json; жанр усилен overlay pressrelease
    overlays: ['infostyle', 'pressrelease'],
    temperature: 0.2,
    handler: 'editSelection_genre_pressrelease'
  },




  genre_landing: {
    menu: 'Лендинг / промостраница',
    group: 'genre',
    order: 30,
    domain: 'genre',         // реальный файл config/domains/genre.json
    intent: 'marketingpush', // файл config/intents/marketingpush.json (было 'marketing_push' — 422)
    overlays: ['infostyle', 'landing'],
    temperature: 0.35,
    handler: 'editSelection_genre_landing'
  },




  genre_workdoc: {
    menu: 'Рабочий документ',
    group: 'genre',
    order: 40,
    domain: 'genre',         // реальный файл config/domains/genre.json
    intent: null,            // жанр усилен overlay workdoc
    overlays: ['infostyle', 'workdoc'],
    temperature: 0.15,
    handler: 'editSelection_genre_workdoc'
  },




  fiction_story: {
    menu: 'Художественный текст',
    group: 'creative',
    order: 10,
    domain: 'fiction',       // реальный файл config/domains/fiction.json
    intent: 'storytelling',  // валидный intent — оставляем
    overlays: ['base'],      // base вместо infostyle: художественному тексту не нужен деловой стиль
    temperature: 0.75,
    handler: 'editSelection_fiction_story'
  },




  composition_analysis: {
    menu: 'Анализ композиции',
    group: 'creative',
    order: 20,
    domain: 'composition',   // реальный файл config/domains/composition.json
    intent: null,
    overlays: ['base'],      // base вместо infostyle: анализ композиции — интерпретация, не деловой стиль
    temperature: 0.45,
    handler: 'editSelection_composition_analysis'
  }
};




/**
 * domain-значения, которые backend реально принимает.
 * Используется в validateModeConfig_ для раннего отлова ошибок.
 */
const ALLOWED_DOMAINS = [
  'marketing',
  'blog',
  'deai',
  'basic_edit',
  'logic_edit',
  'nora_gal',
  'nora_gal_soft',
  'cutnoise',
  'makeclear',
  'restructure',
  'readerfirst',
  'genre',
  'fiction',
  'composition',
  'balanced_edit'
];




/* ============================================================================
 * SECTION: CONFIG VALIDATION
 * ============================================================================
 */




function validateModeConfig_() {
  const knownGroups = new Set(MENU_GROUP_ORDER);
  const knownDomains = new Set(ALLOWED_DOMAINS);
  const knownIntents = new Set(ALLOWED_INTENTS);
  const modeIds = Object.keys(MODE_CONFIG);




  if (!modeIds.length) {
    throw new Error('MODE_CONFIG пустой');
  }




  modeIds.forEach((modeId) => {
    const mode = MODE_CONFIG[modeId];




    if (!mode.menu || typeof mode.menu !== 'string') {
      throw new Error('У режима "' + modeId + '" отсутствует menu');
    }




    if (!mode.group || !knownGroups.has(mode.group)) {
      throw new Error('У режима "' + modeId + '" неверная group');
    }




    if (typeof mode.order !== 'number') {
      throw new Error('У режима "' + modeId + '" отсутствует числовой order');
    }




    if (!mode.domain || typeof mode.domain !== 'string') {
      throw new Error('У режима "' + modeId + '" отсутствует domain');
    }




    if (!knownDomains.has(mode.domain)) {
      throw new Error(
        'У режима "' + modeId + '" domain="' + mode.domain +
        '" не входит в ALLOWED_DOMAINS backend (' + ALLOWED_DOMAINS.join(', ') + ')'
      );
    }




    if (
      mode.intent !== null &&
      mode.intent !== undefined &&
      typeof mode.intent !== 'string'
    ) {
      throw new Error('У режима "' + modeId + '" intent должен быть строкой или null');
    }




    if (
      mode.intent !== null &&
      mode.intent !== undefined &&
      !knownIntents.has(mode.intent)
    ) {
      throw new Error(
        'У режима "' + modeId + '" intent="' + mode.intent +
        '" не входит в ALLOWED_INTENTS backend (' + ALLOWED_INTENTS.join(', ') + ')'
      );
    }




    if (!Array.isArray(mode.overlays) || !mode.overlays.length) {
      throw new Error('У режима "' + modeId + '" overlays должны быть непустым массивом');
    }




    if (
      mode.temperature !== undefined &&
      typeof mode.temperature !== 'number'
    ) {
      throw new Error('У режима "' + modeId + '" temperature должна быть числом');
    }




    if (!mode.handler || typeof mode.handler !== 'string') {
      throw new Error('У режима "' + modeId + '" отсутствует handler');
    }
  });
}




/* ============================================================================
 * SECTION: MENU BUILDING
 * ============================================================================
 */




function onOpen(e) {
  validateModeConfig_();




  const ui = DocumentApp.getUi();
  const rootMenu = ui.createMenu('LLM редактор');




  MENU_GROUP_ORDER.forEach((groupId) => {
    const submenu = buildSubMenu_(ui, groupId);




    if (submenu) {
      rootMenu.addSubMenu(submenu);
    }
  });




  rootMenu.addToUi();
}




function buildSubMenu_(ui, groupId) {
  const items = getModesByGroup_(groupId);




  if (!items.length) {
    return null;
  }




  const submenu = ui.createMenu(MENU_GROUP_TITLES[groupId] || groupId);




  items.forEach((item) => {
    submenu.addItem(item.menu, item.handler);
  });




  return submenu;
}




function getModesByGroup_(groupId) {
  return Object.keys(MODE_CONFIG)
    .filter((modeId) => MODE_CONFIG[modeId].group === groupId)
    .map((modeId) => ({
      modeId: modeId,
      ...MODE_CONFIG[modeId]
    }))
    .sort((a, b) => {
      if (a.order !== b.order) {
        return a.order - b.order;
      }




      return a.menu.localeCompare(b.menu, 'ru');
    });
}




/* ============================================================================
 * SECTION: EXPLICIT HANDLERS
 * ============================================================================
 */




function editSelection_marketing_analytical() {
  runModeById_('marketing_analytical');
}




function editSelection_marketing_push() {
  runModeById_('marketing_push');
}




function analyzeSelection_marketing() {
  runModeById_('marketing_analysis');
}




function editSelection_blog_opinion() {
  runModeById_('blog_opinion');
}




function editSelection_blog_engagement() {
  runModeById_('blog_engagement');
}




function editSelection_basic_edit() {
  runModeById_('basic_edit');
}




function editSelection_logic_edit() {
  runModeById_('logic_edit');
}




function editSelection_nora_gal() {
  runModeById_('nora_gal');
}




function editSelection_nora_gal_soft() {
  runModeById_('nora_gal_soft');
}




function editSelection_balanced_edit() {
  runModeById_('balanced_edit');
}




// режимы «Пиши, сокращай»




function editSelection_readerfirst() {
  runModeById_('readerfirst');
}




function editSelection_cutnoise() {
  runModeById_('cutnoise');
}




function editSelection_makeclear() {
  runModeById_('makeclear');
}




function editSelection_restructure() {
  runModeById_('restructure');
}




// режим «Убрать признаки ИИ»




function editSelection_deai() {
  runModeById_('deai');
}




// жанровые режимы




function editSelection_genre_coldemail() {
  runModeById_('genre_coldemail');
}




function editSelection_genre_pressrelease() {
  runModeById_('genre_pressrelease');
}




function editSelection_genre_landing() {
  runModeById_('genre_landing');
}




function editSelection_genre_workdoc() {
  runModeById_('genre_workdoc');
}




function editSelection_fiction_story() {
  runModeById_('fiction_story');
}




function editSelection_composition_analysis() {
  runModeById_('composition_analysis');
}




/* ============================================================================
 * SECTION: MODE EXECUTION
 * ============================================================================
 */




function runModeById_(modeId) {
  const mode = MODE_CONFIG[modeId];




  if (!mode) {
    throw new Error('Неизвестный режим: ' + modeId);
  }




  editSelection_withMode_(mode);
}




function editSelection_withMode_(mode) {
  const doc = DocumentApp.getActiveDocument();
  const selection = doc.getSelection();




  if (!selection) {
    DocumentApp.getUi().alert('Нет выделенного текста');
    return;
  }




  const originalText = getSelectedText_(selection);




  if (!originalText || originalText.replace(/\s/g, '') === '') {
    DocumentApp.getUi().alert('Выделение пустое (не удалось прочитать текст)');
    return;
  }




  try {
    const editedText = callBackend_(originalText, mode);
    replaceSelection_(selection, editedText);
  } catch (err) {
    Logger.log('Ошибка при вызове backend: ' + err);
    DocumentApp.getUi().alert('Ошибка при обращении к серверу: ' + err);
  }
}




/* ============================================================================
 * SECTION: SELECTION
 * ============================================================================
 */




function getSelectedText_(selection) {
  const rangeElements = selection.getRangeElements();
  const parts = [];




  rangeElements.forEach((rangeElement) => {
    const element = rangeElement.getElement();




    if (!element.editAsText) {
      return;
    }




    const textElement = element.editAsText();
    const fullText = textElement.getText();




    let start = rangeElement.getStartOffset();
    let end = rangeElement.getEndOffsetInclusive();




    if (start === -1 && end === -1) {
      parts.push(fullText);
      return;
    }




    if (start < 0 || end < 0) {
      parts.push(fullText);
      return;
    }




    if (end >= fullText.length) {
      end = fullText.length - 1;
    }




    parts.push(fullText.substring(start, end + 1));
  });




  return parts.join('\n');
}




/**
 * Заменяет только выделенный фрагмент, не трогая соседние абзацы и текст.
 */
function replaceSelection_(selection, newText) {
  const doc = DocumentApp.getActiveDocument();
  const rangeElements = selection.getRangeElements();




  if (!rangeElements || !rangeElements.length) {
    return;
  }




  let insertTarget = null;
  let insertOffset = 0;




  // Идём с конца, чтобы удаление не ломало оффсеты следующих элементов.
  for (let i = rangeElements.length - 1; i >= 0; i -= 1) {
    const rangeElement = rangeElements[i];
    const element = rangeElement.getElement();




    if (!element.editAsText) {
      continue;
    }




    const text = element.editAsText();
    const start = rangeElement.getStartOffset();
    const end = rangeElement.getEndOffsetInclusive();




    if (start === -1 && end === -1) {
      const fullText = text.getText();




      if (insertTarget === null) {
        insertTarget = text;
        insertOffset = 0;
      }




      if (fullText.length > 0) {
        text.deleteText(0, fullText.length - 1);
      }
      continue;
    }




    if (start < 0 || end < 0) {
      continue;
    }




    if (insertTarget === null) {
      insertTarget = text;
      insertOffset = start;
    }




    text.deleteText(start, end);
  }




  if (insertTarget !== null) {
    insertTarget.insertText(insertOffset, newText);
  }
}




/* ============================================================================
 * SECTION: PAYLOAD
 * ============================================================================
 */




function buildPayload_(text, mode) {
  const payload = {
    text: text,
    domain: mode.domain,
    audience: mode.audience || DEFAULT_AUDIENCE,
    overlays: mode.overlays || ['base'],
    output_mode: mode.output_mode || 'text_only',
    provider: BACKEND_CONFIG.provider,
    temperature:
      typeof mode.temperature === 'number'
        ? mode.temperature
        : BACKEND_CONFIG.temperature
  };




  // intent уходит в тело запроса ТОЛЬКО если он валиден для backend.
  // 'neutral' трактуется backend как отсутствие intent, поэтому его не шлём.
  if (
    mode.intent &&
    mode.intent !== 'neutral' &&
    ALLOWED_INTENTS.indexOf(mode.intent) !== -1
  ) {
    payload.intent = mode.intent;
  } else if (mode.intent && mode.intent !== 'neutral') {
    Logger.log(
      'Пропущен невалидный intent "' + mode.intent +
      '" для режима "' + (mode.menu || '') + '"'
    );
  }




  return payload;
}




/* ============================================================================
 * SECTION: HTTP (исправлен – ретраи при 502/503/504)
 * ============================================================================
 */




/**
 * Вызывает бэкенд с ретраями при временных ошибках (502, 503, 504).
 * @param {string} text - исходный текст
 * @param {object} mode - режим редактирования
 * @param {number} maxRetries - максимальное количество попыток (включая первую)
 * @returns {string} отредактированный текст
 * @throws {Error} понятное пользователю сообщение об ошибке
 */
function callBackend_(text, mode, maxRetries = 2) {
  const payload = buildPayload_(text, mode);
  Logger.log('PAYLOAD -> ' + JSON.stringify(payload));

  const options = {
    method: 'post',
    contentType: 'application/json',
    payload: JSON.stringify(payload),
    muteHttpExceptions: true
  };

  let attempt = 0;
  let lastError = null;

  while (attempt < maxRetries) {
    attempt++;
    Logger.log(`Backend call attempt ${attempt} of ${maxRetries}`);

    try {
      const response = UrlFetchApp.fetch(BACKEND_CONFIG.url, options);
      const statusCode = response.getResponseCode();
      const responseBody = response.getContentText();

      // Успешный ответ
      if (statusCode >= 200 && statusCode < 300) {
        const data = JSON.parse(responseBody);
        return data.edited_text || text;
      }

      // Повторяемые статусы (502 Bad Gateway, 503 Service Unavailable, 504 Gateway Timeout)
      if ([502, 503, 504].indexOf(statusCode) !== -1 && attempt < maxRetries) {
        Logger.log(`Retriable error ${statusCode}, retrying in 2.5 sec`);
        Utilities.sleep(2500);
        continue;
      }

      // Неповторяемая ошибка — показываем тело ответа для диагностики (до 500 символов).
      // FIX: увеличен порог с 200 до 500, чтобы FastAPI detail-сообщения читались целиком.
      let errorMsg = `HTTP ${statusCode}`;
      if (responseBody && responseBody.length < 500) {
        errorMsg += `: ${responseBody}`;
      } else {
        errorMsg += ' (сервер вернул некорректный ответ)';
      }
      throw new Error(errorMsg);
    } catch (err) {
      lastError = err;
      // Сетевая ошибка (не HTTP) – тоже пробуем повторить
      if (
        err.message &&
        err.message.indexOf('HTTP') === -1 &&
        attempt < maxRetries
      ) {
        Logger.log(`Network error, retrying: ${err.message}`);
        Utilities.sleep(2500);
        continue;
      }
      // Если повторы кончились или ошибка не повторимая – выбрасываем
      throw err;
    }
  }

  // Если вышли из цикла (все попытки исчерпаны) – дружественное сообщение
  throw new Error(
    'Сервер временно недоступен (перезагрузка). Попробуйте снова через минуту.'
  );
}
