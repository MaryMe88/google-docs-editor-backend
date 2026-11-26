# LLM Editor for Google Docs (RU)

Backend-сервис (FastAPI) и клиент для Google Docs (Google Apps Script), которые помогают редактировать русскоязычные тексты с опорой на структурированную базу знаний: грамматика, стилистика, логика, композиция, сторителлинг и риторика.

Правки применяются к выделенному фрагменту — автор сразу видит отредактированный текст, без объяснительных комментариев (отчёт включается опционально).

## Что умеет

В Google Docs появляется меню **«LLM редактор»** с шестью подменю:

**Маркетинг**

* Аналитично — деловой, информационный стиль
* Продающий — эмоциональный, конверсионный стиль
* Правка с объяснением — аналитичная правка с подробным отчётом

**Блог и соцсети**

* Обычный режим — авторский голос с аналитическим уклоном
* Повысить вовлечённость — живой стиль, ориентированный на реакцию

**Правка и стиль**

* Базовая правка — орфография, пунктуация, грамматика
* Проверка логики — противоречия, дырки в аргументации, рваные переходы
* Правка по Норе Галь — против канцелярита и штампов
* Правка по Норе Галь — бережно — мягкая версия того же подхода
* Взвешенная правка — универсальная, без стилевого уклона
* Убрать признаки ИИ — деайизация с отчётом о правках

**Чистка и структура**

* Фокус на читателе — убрать всё, что не работает на читателя
* Убрать мусор — вода, повторы, пустые вводные
* Упростить предложения — короче, яснее, без потери смысла
* Перестроить структуру — переставить блоки в логичный порядок

**Жанры**

* Холодное письмо
* Пресс-релиз / новость
* Лендинг / промостраница
* Рабочий документ
* Кейс-стади (`statya` + overlay `casestudy`)

**Творческие режимы**

* Художественный текст — нарратив, сторителлинг
* Анализ композиции — структура, ритм, акценты

### 🔬 Глубокая проверка (опционально)

Стандартный режим подбирает правила из базы знаний по точному совпадению ключевых слов и тегов — это быстро и надёжно для большинства текстов.

При `deep_semantic_search: true` в запросе дополнительно применяется семантический re-rankинг:

* Используется лёгкая модель `sentence-transformers` (`cointegrated/rubert-tiny2`), которая вычисляет векторные представления записей базы знаний и текста пользователя.
* Правила, уже отобранные по тегам, пересортировываются с учётом смысловой близости к тексту.
* Это позволяет точнее подбирать релевантные правила для творческих, композиционных и сложных стилистических задач, где ключевые слова могут не совпадать дословно.

**Цена включения:**

* Первый запрос строит индекс для всей базы знаний — занимает несколько секунд и немного памяти.
* Последующие запросы добавляют ≈ 100–300 мс к времени ответа (эмбеддинг текста + сравнение).
* Модель не загружается, если ни один запрос не использовал эту опцию.

**По умолчанию опция выключена** (`deep_semantic_search: false`).

## Стек

* **Backend:** FastAPI + Uvicorn, задеплоен на [Render](https://render.com) (`render.yaml`)
* **LLM-провайдер:** [OpenRouter](https://openrouter.ai) по умолчанию; модель задаётся через allowlist в `src/model_policy.py`
* **Семантический индекс:** `sentence-transformers` (`cointegrated/rubert-tiny2`), опционально
* **Клиент:** Google Apps Script (`google_apps_script/New Script.js`)
* **Python:** 3.11+ (CI проверяет 3.11 и 3.12)
* **Качество:** Ruff (lint + format), mypy, pytest; CI и security-воркфлоу в `.github/workflows/`

## Архитектура

### Файловая структура

```
├── src/
│   ├── main.py                     # FastAPI-приложение, lifespan, middleware, роутеры
│   ├── auth.py                     # Проверка X-API-Key (soft-mode / строгий режим)
│   ├── contracts.py                # Pydantic-схемы запросов и ответов API
│   ├── shared_contracts.py         # Общие Pydantic-типы для нескольких модулей
│   ├── config_types.py             # TypedDict-типы конфигов (core, domain, intent, overlay)
│   ├── startup_checks.py           # Проверки конфигурации и окружения при старте
│   │
│   ├── routers/
│   │   ├── edit.py                 # POST /api/edit
│   │   └── health.py               # GET /, /health, /livez
│   ├── services/
│   │   ├── edit_service.py         # Оркестрация: retrieval → prompt → LLM → guard
│   │   └── provider_health.py      # Проверка доступности провайдеров
│   ├── middleware/
│   │   └── body_size.py            # Ограничение размера тела запроса
│   │
│   ├── knowledge_retrieval.py      # Retrieval-логика: скоринг, селекторы, fallback
│   ├── kb_manifest_loader.py       # Загрузка и валидация kb_manifest.json
│   ├── semantic_index.py           # Семантический индекс KB (rubert-tiny2)
│   ├── tag_registry.py             # Реестр допустимых тегов KB
│   ├── scoring_weights.py          # Веса скоринга для ранжирования правил KB
│   ├── reason_codes.py             # Коды причин правок в отчёте
│   ├── output_guard.py             # Постобработка и валидация ответа LLM
│   ├── context_budget.py           # Бюджет контекста модели, оценка токенов
│   ├── model_policy.py             # Allowlist моделей и дефолты по провайдерам
│   ├── error_mapping.py            # Маппинг внутренних ошибок в ответы API
│   ├── rate_limit.py               # Rate limiting (slowapi)
│   ├── llm_client.py               # HTTP-клиент OpenRouter (+ прочие провайдеры)
│   ├── registry.py                 # Реестр провайдеров и моделей
│   ├── provider_registry.py        # Маппинг имён провайдеров на реализации клиентов
│   │
│   ├── prompt_builder/
│   │   ├── __init__.py             # Публичный API пакета (реэкспорт)
│   │   ├── builder.py              # Класс PromptBuilder (сборка финального промпта)
│   │   ├── contracts.py            # Внутренние контракты пакета
│   │   ├── defaults.py             # Константы и дефолтные конфиги
│   │   ├── normalization.py        # Нормализация intent/overlays
│   │   ├── config_loaders.py       # Загрузка конфигов core/domain/intent/overlays
│   │   ├── kb_loading.py           # Загрузка KB-блоков, KBBlockConfig, реестр блоков
│   │   ├── kb_rendering.py         # Рендеринг KB-блоков, few-shot, confidence notes
│   │   ├── feature_resolution.py   # resolve_prompt_features и объяснимость
│   │   └── _patchable.py           # Точки для тестовых моков
│   │
│   └── config_models/
│       ├── domain.py               # Типы доменных конфигов
│       ├── tags.py                 # Теги и их синонимы
│       ├── tag_constants.py        # Константы тегов
│       ├── limits.py               # Лимиты на количество записей KB по блокам
│       ├── knowledge_budget.py     # Бюджет знаний для промпта
│       ├── explainability.py       # Модели объяснимости выбора правил
│       └── cache.py                # Кэш конфигов и KB
│
├── config/
│   ├── core.json                   # Базовая роль редактора, приоритеты, запреты
│   ├── output_format.json          # Форматы ответа: text_only, text_and_report
│   ├── scoring_weights.json        # Веса retrieval-скоринга
│   ├── tag_map.json                # Карта тегов (запросы ↔ теги KB)
│   ├── domains/                    # Режимы: basic_edit, logic_edit, marketing, blog,
│   │                               # fiction, composition, nora_gal, nora_gal_soft,
│   │                               # balanced_edit, deai, genre, cutnoise, makeclear,
│   │                               # restructure, readerfirst, statya
│   ├── intents/                    # Цели: analytical, engagement, marketingpush, storytelling
│   └── overlays/                   # Надстройки: base, infostyle, post, coldemail,
│                                   # pressrelease, landing, workdoc, casestudy, cohesion,
│                                   # factcheck, finalcheck_full, finalcheck_light,
│                                   # no_overwrite, readerfocus, recommendations,
│                                   # structurefirst, audiencesegment
│
├── knowledge_base/
│   ├── kb_manifest.json            # Реестр всех файлов KB: теги, load_mode, бюджет
│   ├── grammar_errors.json         # Грамматические и орфографические ошибки
│   ├── stop_words.json             # Стоп-слова и нежелательные конструкции
│   ├── logic_issues.json           # Логические ошибки и проблемы связности
│   ├── local_cohesion.json         # Приёмы локальной связности
│   ├── composition_principles.json # Принципы композиции
│   ├── composition_errors.json     # Типичные композиционные ошибки
│   ├── nkrj_structure_patterns.json# Структурные паттерны (по корпусу НКРЯ)
│   ├── otsenki.json                # Техники оценочного анализа текста
│   ├── stylistic_issues/           # Стилистические ошибки (грамматика, лексика, фразеология)
│   ├── editorial_techniques/       # Редакторские приёмы (Нора Галь, анти-ИИ, жанры и др.)
│   ├── genres/business/            # Жанровые шаблоны (кейс-стади и др.)
│   ├── storytelling_macrostructures.json
│   ├── storytelling_microtechniques.json
│   ├── rhetoric_figures.json       # Риторические фигуры
│   ├── rhetoric_topoi.json         # Топосы и аргументативные схемы
│   ├── rhetoric_tropes_and_strategies.json
│   ├── marketing_email.json        # Шаблоны писем
│   ├── marketing_social.json       # Шаблоны постов
│   ├── marketing_web.json          # Шаблоны лендингов и веб-текстов
│   └── marketing_other.json        # Прочие маркетинговые форматы
│
├── google_apps_script/
│   └── New Script.js               # Клиент для Google Docs (Apps Script)
├── .github/
│   ├── workflows/                  # ci.yml (lint+mypy+tests), security.yml
│   └── scripts/verify_torch.py     # Проверка установки torch в CI
├── scripts/
│   ├── generate_kb_manifest.py     # Генерация kb_manifest.json
│   ├── validate_kb_tags.py         # Валидация тегов KB (запускается и при деплое)
│   ├── audit_tags.py               # Аудит покрытия тегов
│   ├── fix_tag_map.py              # Синхронизация tag_map.json
│   └── llm_smoke.py                # Смоук-тест LLM-цепочки
├── tests/                          # Юнит-, интеграционные и golden-тесты (pytest)
├── docs/security/                  # Отчёты по безопасности и аудитам зависимостей
├── render.yaml                     # Конфиг деплоя на Render
├── pyproject.toml                  # Настройки Ruff, mypy, pytest
├── requirements.txt / requirements-dev.txt
└── .env.example                    # Шаблон переменных окружения
```

### Поток данных (request lifecycle)

```
Google Docs (Apps Script)
        │  POST /api/edit  (X-API-Key)
        ▼
main.py (FastAPI)
  1. middleware/body_size.py — лимит размера тела; rate_limit.py — лимит запросов
  2. auth.py — проверка X-API-Key (strict-mode отклоняет запросы без ключа)
  3. contracts.py — валидация схемы (text ≤ 10 000 симв., domain/intent/overlays/model)
        ▼
services/edit_service.py — оркестрация
  ├─ Knowledge Retrieval:
  │    kb_manifest_loader.py → tag_registry.py → semantic_index.py (опц.)
  │    knowledge_retrieval.py + scoring_weights.py — скоринг и отбор правил
  ├─ PromptBuilder (prompt_builder/):
  │    core.json + domain + intent + overlays + KB-блоки + few-shot
  │    context_budget.py — контроль бюджета токенов модели
  ├─ llm_client.py + provider_registry.py — запрос к OpenRouter
  └─ output_guard.py + reason_codes.py — валидация ответа, формирование отчёта
        ▼
EditResponse { edited_text, report?, usage, retrieval_meta? }
```

### Слои и ответственности модулей

| Слой | Модули | Ответственность |
|---|---|---|
| API / Transport | `main.py`, `routers/`, `auth.py`, `contracts.py`, `shared_contracts.py`, `middleware/`, `rate_limit.py` | Приём запроса, аутентификация, валидация схемы, лимиты |
| Сервис | `services/edit_service.py`, `error_mapping.py` | Оркестрация пайплайна, маппинг ошибок |
| Конфигурация | `config_types.py`, `config_models/`, `startup_checks.py`, `model_policy.py` | Типизированные конфиги, allowlist моделей, проверки при старте |
| Knowledge Retrieval | `knowledge_retrieval.py`, `kb_manifest_loader.py`, `semantic_index.py`, `tag_registry.py`, `scoring_weights.py` | Загрузка KB, скоринг и отбор релевантных правил |
| Prompt Assembly | `prompt_builder/`, `context_budget.py` | Сборка промпта, бюджет контекста, объяснимость |
| LLM Client | `llm_client.py`, `registry.py`, `provider_registry.py`, `services/provider_health.py` | HTTP-обращение к провайдерам, маршрутизация, fallback |
| Output | `output_guard.py`, `reason_codes.py` | Постобработка ответа, защита от порчи текста, отчёт |

### Knowledge Retrieval: как выбираются правила

Модуль `knowledge_retrieval.py` реализует многоступенчатый отбор:

1. **Манифест** (`kb_manifest_loader.py`) — `kb_manifest.json` содержит все файлы KB с полями `tags`, `intents`, `load_mode`, `budget_weight`, `stage`, `block_name`.
2. **Режимы загрузки** (`load_mode`): `always` — грузится при каждом запросе; `by_tags` — при совпадении тегов; `by_intent` — только при явном интенте; `never` — отключён.
3. **Тег-реестр** (`tag_registry.py`) — допустимые теги и синонимы; предотвращает опечатки и дрейф тегов.
4. **Семантический индекс** (`semantic_index.py`) — эмбеддинги `rubert-tiny2` для re-rankинга (только при `deep_semantic_search=true`).
5. **Скоринг** (`scoring_weights.py` + `config/scoring_weights.json`) — взвешивает совпадение тегов запроса (domain + intent + overlays) с тегами файлов KB.
6. **Лимиты и бюджет** (`config_models/limits.py`, `knowledge_budget.py`, `context_budget.py`) — ограничивают число записей на блок и общий размер промпта; блоки с меньшим `budget_weight` урезаются первыми.
7. Финальный набор правил передаётся в `prompt_builder/` для включения в KB-блок промпта.

### Как работает сборка промпта

1. Клиент присылает `domain`, `intent`, `overlays`, `audience`, `output_mode`.
2. `PromptBuilder` загружает `config/core.json` + домен + интент + оверлеи (`config_loaders.py`), нормализует их (`normalization.py`).
3. Отбранные правила KB рендерятся в блоки с few-shot примерами (`kb_loading.py`, `kb_rendering.py`).
4. `context_budget.py` контролирует, чтобы промпт + ожидаемый ответ помещались в окно контекста модели.
5. Запрос уходит в OpenRouter (`llm_client.py`); ответ проходит через `output_guard.py` и возвращается в документ.

## Локальный запуск

```bash
git clone https://github.com/MaryMe88/google-docs-editor-backend.git
cd google-docs-editor-backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # заполните значения
```

Минимально необходимые переменные окружения (полный список — в `.env.example`):

```env
OPENROUTER_API_KEY=...
OPENROUTER_SITE_URL=https://docs.google.com
OPENROUTER_APP_NAME=GoogleDocs LLM Editor
# Секрет для аутентификации запросов из Apps Script (заголовок X-API-Key).
# Если не задан — эндпоинты работают в soft-mode (без проверки).
API_SECRET_KEY=
ENV=development
```

Запуск:

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

Проверка:

```bash
curl http://localhost:8000/health   # readiness
curl http://localhost:8000/livez    # liveness
```

Тесты и линтеры:

```bash
pip install -r requirements-dev.txt
pytest
ruff check src tests scripts && ruff format --check src tests scripts
mypy src
```

## API

### POST /api/edit

Запрос (`src/contracts.py`, contract version 1.1.0):

```json
{
  "text": "выделенный фрагмент, ≤ 10000 символов",
  "domain": "marketing",
  "intent": "analytical",
  "overlays": ["infostyle"],
  "audience": {"kind": "b2b", "expertise": "pro", "formality": "neutral"},
  "output_mode": "text_only",
  "provider": "openrouter",
  "model": null,
  "temperature": 0.3,
  "include_knowledge": true,
  "include_few_shot": true,
  "include_retrieval_meta": false,
  "deep_semantic_search": false,
  "dry_run": false
}
```

Ответ:

```json
{
  "edited_text": "...",
  "report": null,
  "provider": "openrouter",
  "model": "deepseek/deepseek-v4-flash",
  "usage": {},
  "retrieval_meta": null
}
```

* `output_mode: "text_and_report"` — добавляет отчёт о правках с кодами причин (`reason_codes.py`).
* `dry_run: true` — возвращает собранный промпт без обращения к LLM.
* Явный выбор `model` разрешён только для OpenRouter и только из allowlist (`src/model_policy.py`).

### Служебные эндпоинты

* `GET /` и `GET /health` — состояние сервиса и провайдеров
* `GET /livez` — liveness-проба для Render

## Подключение к Google Docs

1. Откройте документ Google Docs
2. `Extensions → Apps Script`
3. Вставьте код из `google_apps_script/New Script.js`
4. Сохраните проект и обновите документ — появится меню **«LLM редактор»**

### Настройка подключения (URL и API-ключ)

URL бэкенда и секретный API-ключ **не хардкодятся** — они хранятся в Script Properties и не попадают в Git.

1. В меню выберите **«LLM редактор → ⚙ Настройка подключения»**
2. Введите URL бэкенда (endpoint `/api/edit`)
3. Введите API-ключ (значение `API_SECRET_KEY` на сервере). Чтобы удалить ключ
   и работать в soft-mode, введите `CLEAR`

Альтернатива: задайте свойства `BACKEND_URL` и `API_SECRET_KEY` вручную в
`Project Settings → Script Properties` редактора Apps Script.

### Безопасность и ограничения

* **Аутентификация**: запросы подписываются заголовком `X-API-Key`; в strict-mode
  (`ENV=production` + заданный ключ) запросы без ключа отклоняются
* **Rate limiting**: `/api/edit` — 10 запросов/минуту на IP, health-эндпоинты — 60/минуту
* **Лимит текста**: выделение > 10 000 символов отклоняется до отправки на сервер;
  дополнительно ограничивается размер тела запроса (`middleware/body_size.py`)
* **Allowlist моделей**: произвольные имена моделей отклоняются (`model_policy.py`)
* **Output guard**: ответ LLM проверяется перед вставкой в документ — правки не должны
  сокращать или полностью переписывать текст сверх разрешённого режима
* **Откат**: перед заменой сохраняется оригинал; пункт **«↩ Отменить последнюю правку»**
  возвращает его на место
* Отчёты по аудиту безопасности — в `docs/security/`

## Расширение базы знаний

База знаний управляется через `knowledge_base/kb_manifest.json` — реестр всех файлов
с тегами, `load_mode`, `budget_weight` и метаданными. Добавить новый файл:

1. Создайте JSON в нужной папке `knowledge_base/`
2. Запустите `python scripts/generate_kb_manifest.py` — он обновит манифест
3. При необходимости отредактируйте теги, `load_mode` и `budget_weight` в манифесте
4. Проверьте теги: `python scripts/validate_kb_tags.py`

Добавить новый режим (домен):

1. Создайте `config/domains/<name>.json`
2. При необходимости добавьте intent в `config/intents/` или overlay в `config/overlays/`
3. Добавьте пункт меню в `google_apps_script/New Script.js`

## Деплой

`render.yaml` описывает сервис на Render: build `pip install -r requirements.txt`,
pre-deploy `python3 scripts/validate_kb_tags.py`, start `uvicorn src.main:app --workers 2`,
health check `/livez`. Секреты (`OPENROUTER_API_KEY`, `API_SECRET_KEY`) задаются
в Dashboard Environment Variables и не хранятся в репозитории.

CI (`.github/workflows/ci.yml`): ruff lint/format, mypy, gate сложности,
валидация тегов KB, pytest на Python 3.11 и 3.12.

## Ограничения

* Модель не должна придумывать факты и менять позицию автора — это задаётся в `config/core.json`
* Режим `basic_edit` ограничен лёгкой правкой, не затрагивает композицию
* Язык работы — русский; другие языки требуют отдельной конфигурации

## Лицензия

MIT (см. `LICENSE.txt` и `NOTICE.md`).
