# LLM Editor for Google Docs (RU)

Расширение для Google Docs и backend‑сервис, который помогает редактировать русскоязычные тексты с опорой на структурированную базу знаний: грамматика, стилистика, логика, композиция, сторителлинг и риторика.

Правки применяются к выделенному фрагменту — автор сразу видит отредактированный текст, без объяснительных комментариев.

## Что умеет

В Google Docs появляется меню **«LLM редактор»** с режимами:

| Режим | Что делает |
|---|---|
| ✏️ Базовая правка | Орфография, пунктуация, грамматика, лёгкая стилистика |
| 🧠 Проверка логики | Противоречия, дырки в аргументации, рваные переходы |
| 🧱 Анализ композиции | Структура, ритм, акценты |
| 📣 Маркетинг — аналитично | Деловой, информационный стиль |
| 📣 Маркетинг — продающий | Эмоциональный, конверсионный стиль |
| 💬 Блог / соцсети | Живой авторский голос |
| 📖 Художественный текст | Нарратив, сторителлинг |

## Стек

- **Backend:** FastAPI, задеплоен на [Render](https://render.com)
- **LLM‑провайдер:** [OpenRouter](https://openrouter.ai) (модель задаётся в конфиге или env)
- **Клиент:** Google Apps Script (файл `New Script.js`)
- **Python:** 3.11+

## Архитектура

```
google-docs-editor-backend/
├── src/
│   ├── main.py              # FastAPI-приложение, эндпоинт POST /api/edit
│   ├── prompt_builder.py    # Сборка промпта из конфигов и базы знаний
│   ├── knowledge_base.py    # Загрузка и хранение KB
│   └── llm_client.py        # Клиент OpenRouter
├── config/
│   ├── core.json            # Базовая роль редактора, приоритеты, запреты
│   ├── domains/             # Режимы: basic_edit, logic_edit, marketing, blog, fiction, composition
│   ├── intents/             # Цели: analytical, marketing_push, storytelling и др.
│   ├── overlays/            # Надстройки: infostyle, factcheck, recommendations, finalcheck
│   └── output_format.json   # Форматы ответа: text_only, text_and_report
├── knowledge_base/
│   ├── kb_manifest.json              # Реестр всех файлов KB с тегами и метаданными
│   ├── grammar_errors.json           # Грамматические и орфографические ошибки
│   ├── stop_words.json               # Стоп-слова и нежелательные конструкции
│   ├── logic_issues.json             # Логические ошибки и проблемы связности
│   ├── local_cohesion.json           # Приёмы локальной связности
│   ├── composition_principles.json   # Принципы композиции
│   ├── composition_errors.json       # Типичные композиционные ошибки
│   ├── nkrj_structure_patterns.json  # Структурные паттерны (по корпусу НКРЯ)
│   ├── stylistic_issues/             # Стилистические ошибки (папка, несколько файлов)
│   ├── editorial_techniques/         # Редакторские приёмы (папка, несколько файлов)
│   ├── storytelling_macrostructures.json   # Макроструктуры нарратива (AIDA, трёхактная и др.)
│   ├── storytelling_microtechniques.json   # Микротехники сторителлинга
│   ├── rhetoric_figures.json         # Риторические фигуры
│   ├── rhetoric_topoi.json           # Топосы и аргументативные схемы
│   ├── rhetoric_tropes_and_strategies.json # Тропы и риторические стратегии
│   ├── marketing_email.json          # Шаблоны писем
│   ├── marketing_social.json         # Шаблоны постов
│   ├── marketing_web.json            # Шаблоны лендингов и веб-текстов
│   └── marketing_other.json          # Прочие маркетинговые форматы
├── New Script.js            # Apps Script для Google Docs
├── generate_kb_manifest.py  # Скрипт генерации kb_manifest.json
└── requirements.txt
```

### Как работает сборка промпта

1. Google Docs отправляет JSON с текстом, `domain`, `intent`, `audience` и `overlays`
2. `PromptBuilder` загружает `config/core.json` + нужный домен + интент + оверлеи
3. Из `kb_manifest.json` выбираются релевантные файлы базы знаний по тегам и `load_mode`
4. Финальный промпт = роль + инструкции + KB-блок + текст пользователя
5. Запрос уходит в OpenRouter, ответ возвращается в документ

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
4. Если нужен собственный backend, обновите URL:
   ```js
   const url = 'https://google-docs-editor-backend.onrender.com/api/edit';
   ```
5. Сохраните проект и обновите документ — появится меню **«LLM редактор»**

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
