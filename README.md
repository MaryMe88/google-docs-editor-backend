\# LLM Editor for Google Docs (RU)



Расширение для Google Docs и backend‑сервис, который помогает редактировать

русскоязычные тексты с опорой на базу знаний: грамматика, стилистика,

логика, композиция, сторителлинг и риторика.



\- Backend: FastAPI (`src/main.py`, `src/prompt\_builder.py`, `src/llm\_client.py`)

\- Клиент: Google Docs Apps Script (`google\_docs\_editor.gs`)

\- Хранилище правил: JSON‑файлы в `knowledge\_base/`

\- Хостинг backend’а: \[Render](https://render.com)

\- LLM‑провайдер: \[OpenRouter](https://openrouter.ai)



\## Что делает расширение



В Google Docs появляется меню \*\*«LLM редактор»\*\*, из которого можно:



\- ✏️ сделать базовую правку (орфография, пунктуация, грамматика, лёгкая стилистика);

\- 🧠 провести проверку логики — противоречия, дырки в аргументации, рваные переходы;

\- 🧱 выполнить анализ композиции — структура текста, ритм, акценты;

\- 📣 работать с текстами в режимах:

&#x20; - «Маркетинг — аналитично» / «Маркетинг — продающий»,

&#x20; - «Блог / соцсети»,

&#x20; - «Художественный текст».



Правки применяются к выделенному фрагменту без объяснительных комментариев:

автор просто видит уже отредактированный текст.



\## Архитектура



\### Google Docs (Apps Script)



Скрипт `google\_docs\_editor.gs`:



\- добавляет меню «LLM редактор» с командами:



&#x20; - `editSelection\_basic\_edit`

&#x20; - `editSelection\_logic\_edit`

&#x20; - `editSelection\_marketing\_analytical`

&#x20; - `editSelection\_marketing\_push`

&#x20; - `editSelection\_blog\_opinion`

&#x20; - `editSelection\_fiction\_story`

&#x20; - `editSelection\_composition\_analysis`



\- собирает выделенный текст из документа;

\- формирует JSON‑запрос:



```jsonc

{

&#x20; "text": "...",

&#x20; "domain": "marketing" | "blog" | "fiction" | "basic\_edit" | "logic\_edit" | "composition",

&#x20; "intent": "analytical" | "marketing\_push" | "storytelling" | null,

&#x20; "audience": {

&#x20;   "kind": "b2b" | "b2c" | "mixed",

&#x20;   "expertise": "novice" | "pro" | "expert",

&#x20;   "formality": "casual" | "neutral" | "formal",

&#x20;   "description": "Редактор текста в Google Docs"

&#x20; },

&#x20; "overlays": \["infostyle"],

&#x20; "output\_mode": "text\_only",

&#x20; "provider": "openrouter",

&#x20; "temperature": 0.3

}

```



\- отправляет запрос на backend `https://google-docs-editor-backend.onrender.com/api/edit`;

\- заменяет выделенный текст ответом модели.



\### Backend (FastAPI на Render)



Файл `src/main.py`:



\- поднимает сервис Text Editor API на FastAPI;

\- при старте инициализирует `PromptBuilder`;

\- предоставляет эндпоинты:

&#x20; - `GET /` и `GET /health` — health‑check;

&#x20; - `POST /api/edit` — основной метод правки текста.



Запуск на Render:



\- репозиторий развёрнут как Web Service;

\- переменные окружения задаются в настройках сервиса.



\### Взаимодействие с LLM (OpenRouter)



Файл `src/llm\_client.py`:



\- реализует общий интерфейс `BaseLLMClient`;

\- используется провайдер `LLMProvider.OPENROUTER`;

\- по умолчанию модель задаётся как `openrouter/auto` (или другая, указанная в конфиге/env);

\- запросы отправляются в OpenRouter с заголовками `Authorization`, `HTTP-Referer` и `X-Title`.



Переменные окружения:



```env

OPENROUTER\_API\_KEY=...

OPENROUTER\_SITE\_URL=https://docs.google.com

OPENROUTER\_APP\_NAME=GoogleDocs LLM Editor

```



\## PromptBuilder и база знаний



Файл `src/prompt\_builder.py` отвечает за сборку финального промпта.



\### Конфиги



Загружаются из `config/`:



\- `core.json` — базовая роль редактора, приоритеты и запреты;

\- `domains/\*.json` — режимы:

&#x20; - `marketing.json`

&#x20; - `blog.json`

&#x20; - `fiction.json`

&#x20; - `basic\_edit.json`

&#x20; - `logic\_edit.json`

&#x20; - `composition.json` (анализ композиции);

\- `intents/\*.json` — цели (`analytical`, `storytelling`, `marketing\_push` и др.);

\- `overlays/\*.json` — надстройки (`infostyle`, `factcheck`, `recommendations`, `finalcheck`);

\- `output\_format.json` — форматы ответа (`text\_only`, `text\_and\_report`).



\### База знаний



Загружается из `knowledge\_base/`:



\- `stop\_words.json` — стоп‑слова и нежелательные конструкции;

\- `grammar\_errors.json` — типичные грамматические / орфографические ошибки;

\- `stylistic\_issues.json` — стилистические ошибки, штампы, канцелярит, повторы;

\- `logic\_issues.json` — логические ошибки и проблемы связности;

\- `composition\_principles.json` — принципы композиции;

\- `local\_cohesion.json` — приёмы локальной связности;

\- `composition\_errors.json` — типичные композиционные ошибки;

\- `storytelling\_frameworks.json` — сторителлинговые фреймворки (AIDA, PAS, трёхактная структура и др.);

\- `marketing\_templates.json` — шаблоны лендингов, писем, постов;

\- `rithoric.json` — риторические топосы и приёмы аргументации.



В промпте база знаний показывается как большой справочный блок:



\- примеры грамматических и стилистических правок;

\- типовые логические и композиционные ошибки;

\- принципы композиции и локальной связности;

\- сторителлинговые и риторические схемы;

\- маркетинговые шаблоны (для домена `marketing`).



\## Локальный запуск backend



```bash

git clone https://github.com/<your-username>/<your-repo>.git

cd <your-repo>

pip install -r requirements.txt

```



Создайте `.env`:



```env

OPENROUTER\_API\_KEY=...

OPENROUTER\_SITE\_URL=https://docs.google.com

OPENROUTER\_APP\_NAME=GoogleDocs LLM Editor

```



Запуск:



```bash

uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

```



Проверка:



```bash

curl http://localhost:8000/health

```



\## Подключение к Google Docs



1\. Откройте документ Google Docs.

2\. `Extensions → Apps Script`.

3\. Вставьте код из `google\_docs\_editor.gs`.

4\. Обновите URL backend’а (если нужно свой):



&#x20;  ```js

&#x20;  const url = 'https://google-docs-editor-backend.onrender.com/api/edit';

&#x20;  ```



5\. Сохраните проект и обновите документ — появится меню \*\*«LLM редактор»\*\*.



\## Использование



1\. Выделите текст в документе.

2\. Выберите режим в меню «LLM редактор»:

&#x20;  - «Базовая правка»

&#x20;  - «Проверка логики»

&#x20;  - «Блог / соцсети»

&#x20;  - «Маркетинг — аналитично / продающий»

&#x20;  - «Анализ композиции»

3\. Скрипт отправит текст на backend, получит отредактированный вариант

&#x20;  и заменит им выделение.



\## Как расширять базу знаний



\- Добавить новое правило — дописать соответствующий JSON в `knowledge\_base/`:

&#x20; - грамматика — `grammar\_errors.json`,

&#x20; - стилистика — `stylistic\_issues.json`,

&#x20; - логика — `logic\_issues.json`,

&#x20; - композиция — `composition\_principles.json` / `composition\_errors.json`,

&#x20; - локальная связность — `local\_cohesion.json`,

&#x20; - сторителлинг — `storytelling\_frameworks.json`,

&#x20; - маркетинг — `marketing\_templates.json`,

&#x20; - риторика — `rithoric.json`.

\- Добавить новый режим:

&#x20; - создать домен в `config/domains/<name>.json`,

&#x20; - при необходимости добавить intent в `config/intents/`,

&#x20; - добавить пункт меню в `google\_docs\_editor.gs`.



Ограничения и этика



\- Модель не должна придумывать факты и менять позицию автора —

&#x20; это задаётся в `config/core.json` и доменных настройках.

\- Режим basic\_edit ограничен лёгкой правкой и не трогает композицию.

\- Режим composition может перестраивать порядок абзацев и ритм,

&#x20; но не добавляет новых событий и фактов.

\- Язык работы — русский; другие языки требуют отдельной конфигурации.



Лицензия



MIT

