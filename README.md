# LLM Editor for Google Docs (RU)

Расширение для Google Docs и backend‑сервис, который помогает редактировать русскоязычные тексты с опорой на структурированную базу знаний: грамматика, стилистика, логика, композиция, сторителлинг и риторика.

Правки применяются к выделенному фрагменту — автор сразу видит отредактированный текст, без объяснительных комментариев.

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

**Творческие режимы**

* Художественный текст — нарратив, сторителлинг
* Анализ композиции — структура, ритм, акценты

## Стек

* **Backend:** FastAPI, задеплоен на [Render](https://render.com)
* **LLM‑провайдер:** [OpenRouter](https://openrouter.ai) (модель задаётся в конфиге или env)
* **Клиент:** Google Apps Script (файл `New Script.js`)
* **Python:** 3.11+

## Архитектура

### Файловая структура

```
google-docs-editor-backend/

├── src/

│ ├── main.py # FastAPI-приложение, эндпоинт POST /api/edit

│ ├── auth.py # Проверка X-API-Key (мягкий / строгий режим)

│ ├── contracts.py # Pydantic-схемы запросов и ответов API

│ ├── shared\_contracts.py # Общие Pydantic-типы, используемые в нескольких модулях

│ ├── config\_types.py # TypedDict-типы для конфигов (core, domain, intent, overlay)

│ ├── prompt\_builder/ # Пакет сборки промпта

│ │ ├── init.py # Публичный API (реэкспорт)

│ │ ├── builder.py # Класс PromptBuilder

│ │ ├── normalization.py # Нормализация intent/overlays

│ │ ├── defaults.py # Константы и дефолтные конфиги

│ │ ├── config\_loaders.py # Загрузка конфигов (core, domain, intent, overlays)

│ │ ├── kb\_loading.py # Загрузка KB, KBBlockConfig, KB\_BLOCK\_REGISTRY

│ │ ├── kb\_rendering.py # Рендеринг KB-блоков, few-shot, confidence notes

│ │ └── feature\_resolution.py# resolve\_prompt\_features и объяснимость

│ ├── knowledge\_retrieval.py # Retrieval-логика: скоринг, селекторы, fallback

│ ├── kb\_manifest\_loader.py # Загрузка и валидация kb\_manifest.json

│ ├── semantic\_index.py # Лёгкий семантический индекс KB по тегам

│ ├── tag\_registry.py # Реестр допустимых тегов KB

│ ├── scoring\_weights.py # Веса скоринга для ранжирования правил KB

│ ├── reason\_codes.py # Коды причин (reason codes) в отчёте правки

│ ├── output\_guard.py # Постобработка и валидация ответа LLM

│ ├── llm\_client.py # HTTP-клиент OpenRouter

│ ├── registry.py # Реестр провайдеров и моделей

│ ├── provider\_registry.py # Маппинг имён провайдеров на реализации клиентов

│ └── startup\_checks.py # Проверки конфигурации при старте сервиса

├── config/

│ ├── core.json # Базовая роль редактора, приоритеты, запреты

│ ├── domains/ # Режимы: basic\_edit, logic\_edit, marketing, blog, fiction,

│ │ # composition, nora\_gal, nora\_gal\_soft, balanced\_edit,

│ │ # deai, genre, cutnoise, makeclear, restructure, readerfirst

│ ├── intents/ # Цели: analytical, marketingpush, storytelling, engagement

│ ├── overlays/ # Надстройки: base, infostyle, coldemail, pressrelease,

│ │ # landing, workdoc

│ └── output\_format.json # Форматы ответа: text\_only, text\_and\_report

├── knowledge\_base/

│ ├── kb\_manifest.json # Реестр всех файлов KB с тегами и метаданными

│ ├── grammar\_errors.json # Грамматические и орфографические ошибки

│ ├── stop\_words.json # Стоп-слова и нежелательные конструкции

│ ├── logic\_issues.json # Логические ошибки и проблемы связности

│ ├── local\_cohesion.json # Приёмы локальной связности

│ ├── composition\_principles.json # Принципы композиции

│ ├── composition\_errors.json # Типичные композиционные ошибки

│ ├── nkrj\_structure\_patterns.json # Структурные паттерны (по корпусу НКРЯ)

│ ├── stylistic\_issues/ # Стилистические ошибки (папка, несколько файлов)

│ ├── editorial\_techniques/ # Редакторские приёмы (папка, несколько файлов)

│ ├── storytelling\_macrostructures.json # Макроструктуры нарратива (AIDA, трёхактная и др.)

│ ├── storytelling\_microtechniques.json # Микротехники сторителлинга

│ ├── rhetoric\_figures.json # Риторические фигуры

│ ├── rhetoric\_topoi.json # Топосы и аргументативные схемы

│ ├── rhetoric\_tropes\_and\_strategies.json # Тропы и риторические стратегии

│ ├── marketing\_email.json # Шаблоны писем

│ ├── marketing\_social.json # Шаблоны постов

│ ├── marketing\_web.json # Шаблоны лендингов и веб-текстов

│ └── marketing\_other.json # Прочие маркетинговые форматы

├── New Script.js # Apps Script для Google Docs

├── generate\_kb\_manifest.py # Скрипт генерации kb\_manifest.json

└── requirements.txt

```

### Поток данных (request lifecycle)

```
Google Docs (Apps Script)

│

│ POST /api/edit

│ { text, domain, intent, audience, overlays }

▼

┌─────────────────────────────────────────────┐

│ main.py (FastAPI) │

│ 1. auth.py — проверка X-API-Key │

│ 2. contracts.py — валидация запроса │

│ 3. вызов PromptBuilder │

└──────────────────┬──────────────────────────┘

│

▼

┌─────────────────────────────────────────────┐

│ prompt\_builder/ (пакет) │

│ · Загружает core.json + domain + intent │

│ + overlays (config\_loaders.py) │

│ · Вызывает KnowledgeRetrieval для отбора │

│ правил из KB │

│ · Собирает финальный промпт: │

│ роль + инструкции + KB-блок + текст │

└──────────────────┬──────────────────────────┘

│

┌────────┴────────┐

▼ ▼

┌──────────────────┐ ┌─────────────────────────┐

│ knowledge\_ │ │ config/ │

│ retrieval.py │ │ core.json │

│ │ │ domains/<domain>.json │

│ · kb\_manifest\_ │ │ intents/<intent>.json │

│ loader.py │ │ overlays/\*.json │

│ · semantic\_ │ │ output\_format.json │

│ index.py │ └─────────────────────────┘

│ · tag\_registry │

│ · scoring\_ │

│ weights.py │

└──────────────────┘

│

▼

┌─────────────────────────────────────────────┐

│ llm\_client.py │

│ · Отправляет промпт в OpenRouter │

│ · registry.py / provider\_registry.py — │

│ выбор модели и провайдера │

└──────────────────┬──────────────────────────┘

│

▼

┌─────────────────────────────────────────────┐

│ output\_guard.py │

│ · Валидирует и санирует ответ LLM │

│ · reason\_codes.py — коды правок в отчёте │

└──────────────────┬──────────────────────────┘

│

▼

Ответ → Google Docs

```

### Слои и ответственности модулей

|Слой|Модули|Роль|
|-|-|-|
|**API / Transport**|`main.py`, `auth.py`, `contracts.py`, `shared\_contracts.py`|Приём запроса, аутентификация, валидация схемы|
|**Оркестрация**|`prompt\_builder/`|Сборка финального промпта из конфигов и KB|
|**Конфигурация**|`config\_types.py`, `startup\_checks.py`|Типизированные конфиги, проверка окружения при старте|
|**Knowledge Retrieval**|`knowledge\_retrieval.py`, `kb\_manifest\_loader.py`, `semantic\_index.py`, `tag\_registry.py`, `scoring\_weights.py`|Загрузка KB, скоринг и отбор релевантных правил|
|**LLM Client**|`llm\_client.py`, `registry.py`, `provider\_registry.py`|HTTP-обращение к OpenRouter, маршрутизация по провайдерам|
|**Output**|`output\_guard.py`, `reason\_codes.py`|Постобработка ответа, формирование отчёта о правках|

### Knowledge Retrieval: как выбираются правила

Модуль `knowledge\_retrieval.py` реализует многоступенчатый отбор правил из базы знаний:

1. **Манифест** (`kb\_manifest\_loader.py`) — загружает `kb\_manifest.json` со списком всех файлов KB, их тегами и режимом загрузки (`load\_mode`: `always` / `on\_demand` / `tagged`).
2. **Тег-реестр** (`tag\_registry.py`) — хранит допустимые теги и их синонимы; предотвращает опечатки и дрейф тегов.
3. **Семантический индекс** (`semantic\_index.py`) — лёгкий индекс KB по тегам; позволяет быстро сузить кандидатов без эмбеддингов.
4. **Скоринг** (`scoring\_weights.py`) — взвешивает совпадение тегов запроса (domain + intent + overlays) с тегами файлов KB; чем выше совпадение — тем выше ранг.
5. **Fallback** — если ни один файл не прошёл порог релевантности, активируются файлы с `load\_mode: always`.
6. **Финальный набор** правил передаётся в `prompt\_builder.py/` для включения в KB-блок промпта.

### Как работает сборка промпта

1. Google Docs отправляет JSON с текстом, `domain`, `intent`, `audience` и `overlays`
2. `PromptBuilder` (в пакете `prompt\_builder/`) загружает `config/core.json` + нужный домен + интент + оверлеи
3. `KnowledgeRetrieval` выбирает релевантные файлы KB по тегам и `load\_mode`
4. Финальный промпт = роль + инструкции + KB-блок + текст пользователя
5. Запрос уходит в OpenRouter, ответ проходит через `output\_guard.py` и возвращается в документ

## Локальный запуск

```bash
git clone https://github.com/MaryMe88/google-docs-editor-backend.git
cd google-docs-editor-backend
pip install -r requirements.txt
```

Создайте `.env`:

```env
OPENROUTER\_API\_KEY=...
OPENROUTER\_SITE\_URL=https://docs.google.com
OPENROUTER\_APP\_NAME=GoogleDocs LLM Editor
# Секрет для аутентификации запросов из Apps Script (заголовок X-API-Key).
# Если не задан — эндпоинты работают без проверки (soft-mode).
API\_SECRET\_KEY=
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
3. Введите API-ключ (значение `API\_SECRET\_KEY` на сервере). Чтобы удалить ключ
и работать в soft-mode, введите `CLEAR`

Альтернатива: задайте свойства `BACKEND\_URL` и `API\_SECRET\_KEY` вручную в
`Project Settings → Script Properties` редактора Apps Script.

### Безопасность и ограничения

* **Аутентификация**: запросы подписываются заголовком `X-API-Key`, если ключ задан
* **Лимит текста**: выделение > 10000 символов отклоняется до отправки на сервер
* **Откат**: перед заменой сохраняется оригинал; пункт **«↩ Отменить последнюю правку»**
возвращает его на место

## Расширение базы знаний

База знаний управляется через `knowledge\_base/kb\_manifest.json` — реестр всех файлов с тегами, `load\_mode` и метаданными. Добавить новый файл:

1. Создайте JSON в нужной папке `knowledge\_base/`
2. Запустите `python generate\_kb\_manifest.py` — он обновит манифест
3. При необходимости отредактируйте теги и `load\_mode` в `kb\_manifest.json` вручную

Добавить новый режим (домен):

1. Создайте `config/domains/<name>.json`
2. При необходимости добавьте intent в `config/intents/`
3. Добавьте пункт меню в `New Script.js`

## Ограничения

* Модель не должна придумывать факты и менять позицию автора — это задаётся в `config/core.json`
* Режим `basic\_edit` ограничен лёгкой правкой, не затрагивает композицию
* Язык работы — русский; другие языки требуют отдельной конфигурации

## Лицензия

MIT

