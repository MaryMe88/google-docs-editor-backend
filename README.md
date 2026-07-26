# LLM Editor for Google Docs (RU)

Расширение для Google Docs и backend‑сервис, который помогает редактировать русскоязычные тексты с опорой на структурированную базу знаний: грамматика, стилистика, логика, композиция, сторителлинг и риторика.

Правки применяются к выделенному фрагменту — автор сразу видит отредактированный текст, без объяснительных комментариев.

## Что умеет

В Google Docs появляется меню **«LLM редактор»** с шестью подменю:

**Маркетинг**
- Аналитично — деловой, информационный стиль
- Продающий — эмоциональный, конверсионный стиль
- Правка с объяснением — аналитичная правка с подробным отчётом

**Блог и соцсети**
- Обычный режим — авторский голос с аналитическим уклоном
- Повысить вовлечённость — живой стиль, ориентированный на реакцию

**Правка и стиль**
- Базовая правка — орфография, пунктуация, грамматика
- Проверка логики — противоречия, дырки в аргументации, рваные переходы
- Правка по Норе Галь — против канцелярита и штампов
- Правка по Норе Галь — бережно — мягкая версия того же подхода
- Взвешенная правка — универсальная, без стилевого уклона
- Убрать признаки ИИ — деайизация с отчётом о правках

**Чистка и структура**
- Фокус на читателе — убрать всё, что не работает на читателя
- Убрать мусор — вода, повторы, пустые вводные
- Упростить предложения — короче, яснее, без потери смысла
- Перестроить структуру — переставить блоки в логичный порядок

**Жанры**
- Холодное письмо
- Пресс-релиз / новость
- Лендинг / промостраница
- Рабочий документ

**Творческие режимы**
- Художественный текст — нарратив, сторителлинг
- Анализ композиции — структура, ритм, акценты

## Стек

- **Backend:** FastAPI, задеплоен на [Render](https://render.com)
- **LLM‑провайдер:** [OpenRouter](https://openrouter.ai) (модель задаётся в конфиге или env)
- **Клиент:** Google Apps Script (файл `New Script.js`)
- **Python:** 3.11+

## Архитектура

### Файловая структура

```
google-docs-editor-backend/
├── src/
│   ├── main.py                  # FastAPI-приложение, эндпоинт POST /api/edit
│   ├── auth.py                  # Проверка X-API-Key (мягкий / строгий режим)
│   ├── contracts.py             # Pydantic-схемы запросов и ответов API
│   ├── shared_contracts.py      # Общие Pydantic-типы, используемые в нескольких модулях
│   ├── config_types.py          # TypedDict-типы для конфигов (core, domain, intent, overlay)
│   ├── prompt_builder.py        # Оркестрация сборки промпта
│   ├── knowledge_retrieval.py   # Retrieval-логика: скоринг, селекторы, fallback
│   ├── kb_manifest_loader.py    # Загрузка и валидация kb_manifest.json
│   ├── semantic_index.py        # Лёгкий семантический индекс KB по тегам
│   ├── tag_registry.py          # Реестр допустимых тегов KB
│   ├── scoring_weights.py       # Веса скоринга для ранжирования правил KB
│   ├── reason_codes.py          # Коды причин (reason codes) в отчёте правки
│   ├── output_guard.py          # Постобработка и валидация ответа LLM
│   ├── llm_client.py            # HTTP-клиент OpenRouter
│   ├── registry.py              # Реестр провайдеров и моделей
│   ├── provider_registry.py     # Маппинг имён провайдеров на реализации клиентов
│   └── startup_checks.py        # Проверки конфигурации при старте сервиса
├── config/
│   ├── core.json                # Базовая роль редактора, приоритеты, запреты
│   ├── domains/                 # Режимы: basic_edit, logic_edit, marketing, blog, fiction,
│   │                            #   composition, nora_gal, nora_gal_soft, balanced_edit,
│   │                            #   deai, genre, cutnoise, makeclear, restructure, readerfirst
│   ├── intents/                 # Цели: analytical, marketingpush, storytelling, engagement
│   ├── overlays/                # Надстройки: base, infostyle, coldemail, pressrelease,
│   │                            #   landing, workdoc
│   └── output_format.json       # Форматы ответа: text_only, text_and_report
├── knowledge_base/
│   ├── kb_manifest.json                    # Реестр всех файлов KB с тегами и метаданными
│   ├── grammar_errors.json                 # Грамматические и орфографические ошибки
│   ├── stop_words.json                     # Стоп-слова и нежелательные конструкции
│   ├── logic_issues.json                   # Логические ошибки и проблемы связности
│   ├── local_cohesion.json                 # Приёмы локальной связности
│   ├── composition_principles.json         # Принципы композиции
│   ├── composition_errors.json             # Типичные композиционные ошибки
│   ├── nkrj_structure_patterns.json        # Структурные паттерны (по корпусу НКРЯ)
│   ├── stylistic_issues/                   # Стилистические ошибки (папка, несколько файлов)
│   ├── editorial_techniques/               # Редакторские приёмы (папка, несколько файлов)
│   ├── storytelling_macrostructures.json   # Макроструктуры нарратива (AIDA, трёхактная и др.)
│   ├── storytelling_microtechniques.json   # Микротехники сторителлинга
│   ├── rhetoric_figures.json               # Риторические фигуры
│   ├── rhetoric_topoi.json                 # Топосы и аргументативные схемы
│   ├── rhetoric_tropes_and_strategies.json # Тропы и риторические стратегии
│   ├── marketing_email.json                # Шаблоны писем
│   ├── marketing_social.json               # Шаблоны постов
│   ├── marketing_web.json                  # Шаблоны лендингов и веб-текстов
│   └── marketing_other.json                # Прочие маркетинговые форматы
├── New Script.js            # Apps Script для Google Docs
├── generate_kb_manifest.py  # Скрипт генерации kb_manifest.json
└── requirements.txt
```

### Поток данных (request lifecycle)

```
Google Docs (Apps Script)
        │
        │  POST /api/edit
        │  { text, domain, intent, audience, overlays }
        ▼
┌─────────────────────────────────────────────┐
│              main.py  (FastAPI)             │
│  1. auth.py — проверка X-API-Key            │
│  2. contracts.py — валидация запроса        │
│  3. вызов PromptBuilder                     │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│           prompt_builder.py                 │
│  · Загружает core.json + domain + intent    │
│    + overlays (config_types.py)             │
│  · Вызывает KnowledgeRetrieval для отбора   │
│    правил из KB                             │
│  · Собирает финальный промпт:               │
│    роль + инструкции + KB-блок + текст      │
└──────────────────┬──────────────────────────┘
                   │
          ┌────────┴────────┐
          ▼                 ▼
┌──────────────────┐  ┌─────────────────────────┐
│  knowledge_      │  │  config/                │
│  retrieval.py    │  │  core.json              │
│                  │  │  domains/<domain>.json  │
│  · kb_manifest_  │  │  intents/<intent>.json  │
│    loader.py     │  │  overlays/*.json        │
│  · semantic_     │  │  output_format.json     │
│    index.py      │  └─────────────────────────┘
│  · tag_registry  │
│  · scoring_      │
│    weights.py    │
└──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│              llm_client.py                  │
│  · Отправляет промпт в OpenRouter           │
│  · registry.py / provider_registry.py —     │
│    выбор модели и провайдера                │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│              output_guard.py                │
│  · Валидирует и санирует ответ LLM          │
│  · reason_codes.py — коды правок в отчёте   │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
        Ответ → Google Docs
```

### Слои и ответственности модулей

| Слой | Модули | Роль |
|---|---|---|
| **API / Transport** | `main.py`, `auth.py`, `contracts.py`, `shared_contracts.py` | Приём запроса, аутентификация, валидация схемы |
| **Оркестрация** | `prompt_builder.py` | Сборка финального промпта из конфигов и KB |
| **Конфигурация** | `config_types.py`, `startup_checks.py` | Типизированные конфиги, проверка окружения при старте |
| **Knowledge Retrieval** | `knowledge_retrieval.py`, `kb_manifest_loader.py`, `semantic_index.py`, `tag_registry.py`, `scoring_weights.py` | Загрузка KB, скоринг и отбор релевантных правил |
| **LLM Client** | `llm_client.py`, `registry.py`, `provider_registry.py` | HTTP-обращение к OpenRouter, маршрутизация по провайдерам |
| **Output** | `output_guard.py`, `reason_codes.py` | Постобработка ответа, формирование отчёта о правках |

### Knowledge Retrieval: как выбираются правила

Модуль `knowledge_retrieval.py` реализует многоступенчатый отбор правил из базы знаний:

1. **Манифест** (`kb_manifest_loader.py`) — загружает `kb_manifest.json` со списком всех файлов KB, их тегами и режимом загрузки (`load_mode`: `always` / `on_demand` / `tagged`).
2. **Тег-реестр** (`tag_registry.py`) — хранит допустимые теги и их синонимы; предотвращает опечатки и дрейф тегов.
3. **Семантический индекс** (`semantic_index.py`) — лёгкий индекс KB по тегам; позволяет быстро сузить кандидатов без эмбеддингов.
4. **Скоринг** (`scoring_weights.py`) — взвешивает совпадение тегов запроса (domain + intent + overlays) с тегами файлов KB; чем выше совпадение — тем выше ранг.
5. **Fallback** — если ни один файл не прошёл порог релевантности, активируются файлы с `load_mode: always`.
6. **Финальный набор** правил передаётся в `prompt_builder.py` для включения в KB-блок промпта.

### Как работает сборка промпта

1. Google Docs отправляет JSON с текстом, `domain`, `intent`, `audience` и `overlays`
2. `PromptBuilder` загружает `config/core.json` + нужный домен + интент + оверлеи
3. `KnowledgeRetrieval` выбирает релевантные файлы KB по тегам и `load_mode`
4. Финальный промпт = роль + инструкции + KB-блок + текст пользователя
5. Запрос уходит в OpenRouter, ответ проходит через `output_guard.py` и возвращается в документ

## Локальный запуск

```bash
git clone https://github.com/MaryMe88/google-docs-editor-backend.git
cd google-docs-editor-backend
pip install -r requirements.txt
```

Создайте `.env`:

```env
OPENROUTER_API_KEY=...
OPENROUTER_SITE_URL=https://docs.google.com
OPENROUTER_APP_NAME=GoogleDocs LLM Editor
# Секрет для аутентификации запросов из Apps Script (заголовок X-API-Key).
# Если не задан — эндпоинты работают без проверки (soft-mode).
API_SECRET_KEY=
```

Запуск:

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

Проверка:

```bash
curl http://localhost:8000/health
```

## Подключение к Google Docs

1. Откройте документ Google Docs
2. `Extensions → Apps Script`
3. Вставьте код из `New Script.js`
4. Сохраните проект и обновите документ — появится меню **«LLM редактор»**

### Настройка подключения (URL и API-ключ)

URL бэкенда и секретный API-ключ **больше не хардкодятся** — они хранятся
в Script Properties и не попадают в Git.

1. В меню выберите **«LLM редактор → ⚙ Настройка подключения»**
2. Введите URL бэкенда (endpoint `/api/edit`) — или оставьте пустым для fallback
3. Введите API-ключ (значение `API_SECRET_KEY` на сервере). Чтобы удалить ключ
   и работать в soft-mode, введите `CLEAR`

Альтернатива: задайте свойства `BACKEND_URL` и `API_SECRET_KEY` вручную в
`Project Settings → Script Properties` редактора Apps Script.

### Безопасность и ограничения

- **Аутентификация**: запросы подписываются заголовком `X-API-Key`, если ключ задан
- **Лимит текста**: выделение > 10000 символов отклоняется до отправки на сервер
- **Откат**: перед заменой сохраняется оригинал; пункт **«↩ Отменить последнюю правку»**
  возвращает его на место

## Расширение базы знаний

База знаний управляется через `knowledge_base/kb_manifest.json` — реестр всех файлов с тегами, `load_mode` и метаданными. Добавить новый файл:

1. Создайте JSON в нужной папке `knowledge_base/`
2. Запустите `python generate_kb_manifest.py` — он обновит манифест
3. При необходимости отредактируйте теги и `load_mode` в `kb_manifest.json` вручную

Добавить новый режим (домен):

1. Создайте `config/domains/<name>.json`
2. При необходимости добавьте intent в `config/intents/`
3. Добавьте пункт меню в `New Script.js`

## Ограничения

- Модель не должна придумывать факты и менять позицию автора — это задаётся в `config/core.json`
- Режим `basic_edit` ограничен лёгкой правкой, не затрагивает композицию
- Язык работы — русский; другие языки требуют отдельной конфигурации

## Лицензия

MIT
