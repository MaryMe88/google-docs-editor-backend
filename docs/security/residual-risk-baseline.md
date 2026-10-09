\# Residual security baseline — 2026-10-09



\## Назначение



Документ фиксирует фактическое состояние остаточных рисков

в проекте `google-docs-editor-backend` после завершения первой

дорожной карты устранения уязвимостей (issue #35).



\*\*Это фиксация, не исправление.\*\* Никаких изменений кода,

зависимостей или конфигурации в этой итерации не делается.



Связанная задача: https://github.com/MaryMe88/google-docs-editor-backend/issues/35



\*\*Дата аудита:\*\* 2026-10-09.

\*\*Следующий пересмотр:\*\* не позднее 2026-04-08.



\## Окружение



| Параметр | Значение |

|---|---|

| Python | 3.12.2 |

| Ветка | `security/residual-risk-baseline` (от `main` `9d98975`) |

| Тесты | \*\*539 passed, 9 skipped\*\* |

| `pip check` | OK |

| `pip-audit` | 9 advisory (все в `transformers 4.57.6`) + skip `torch` |



\## Фактические версии security-зависимостей



| Пакет | Версия | Источник | Проверен pip-audit |

|---|---|---|---|

| fastapi | 0.142.2 | PyPI | Да |

| starlette | 1.3.1 | PyPI | Да |

| python-dotenv | 1.2.4 | PyPI | Да |

| pytest | 9.1.1 | PyPI | Да |

| anyio | 4.15.1 | PyPI | Да |

| sentence-transformers | 5.6.1 | PyPI | Да |

| transformers | 4.57.6 | PyPI | Да (9 advisory, формально приняты) |

| torch | 2.5.0+cpu | CPU-индекс PyTorch | \*\*Нет\*\* |



\## Закрытые advisory (первая дорожная карта)



| Пакет | Закрыто итерацией | PR |

|---|---|---|

| python-dotenv | 1 | #38 |

| fastapi + starlette | 2 | #39 |

| pytest + pytest-asyncio | 4 | #46 |

| anyio | 4 | #46 |

| sentence-transformers | 5 | #47 |

| starlette 0.x → 1.x | 6 | #48 |



Всего: \*\*12 уникальных advisory\*\*.



\## Таблица остаточных рисков



\### Область 1. FastAPI / Starlette



| Поле | Значение |

|---|---|

| Подтверждено | Да |

| Доказательство | `fastapi==0.142.2`, `starlette==1.3.1`; `pip-audit` по environment — 0 записей |

| Остаточный риск | Нет |

| Следующий шаг | — |



\### Область 2. Torch



| Поле | Значение |

|---|---|

| Подтверждено | Частично |

| Доказательство | `torch==2.5.0+cpu` из `--extra-index-url https://download.pytorch.org/whl/cpu` (`requirements.txt:1,11`); `pip-audit` пишет `Skip Reason: torch not found on PyPI` |

| Остаточный риск | (1) Torch вне охвата pip-audit. (2) \*\*Ссылки на `docs/security/2026-10-torch-audit.md` из `.github/workflows/security.yml` и `.github/scripts/verify\_torch.py` битые\*\* — документа нет в main. (3) Версия Torch 2.5.0 устарела относительно 2.7+ (для transformers 5.x). |

| Следующий шаг | \*\*Итерация 4\*\* — сделать установку Torch воспроизводимой. \*\*Итерация 5\*\* — создать `docs/security/torch-risk-assessment.md` (актуальный документ), обновить ссылки в workflow и скрипте. |



\#### Обнаруженный дефект: битая ссылка на Torch-документ



Ссылки на `docs/security/2026-10-torch-audit.md` существуют в:



\- `.github/workflows/security.yml` (комментарии к job `torch-audit`).

\- `.github/scripts/verify\_torch.py` (докстринг и сообщение об ошибке).



Файл в \*\*main отсутствует\*\*. Он был создан в коммите `27452b5` на ветке

`security/audit-torch-runtime`, которая \*\*никогда не была влита\*\*.



Содержимое файла на той ветке:

\- Соответствует версии torch 2.5.0+cpu.

\- Содержит устаревшие данные: `sentence-transformers 3.2.1` (сейчас 5.6.1).

\- Не упоминает `\_SEMANTIC\_MODEL\_REVISION` (добавлен в итерации 5).



\*\*Решение:\*\* не восстанавливать сейчас (итерация 0 — фиксация, не исправление).

Восстановить + актуализировать в \*\*итерации 5\*\* под именем

`docs/security/torch-risk-assessment.md` (по новой дорожной карте),

с одновременным обновлением ссылок в `.github/workflows/security.yml`

и `.github/scripts/verify\_torch.py`.



\### Область 3. Production auth (API\_SECRET\_KEY)



| Поле | Значение |

|---|---|

| Подтверждено | Да |

| Доказательство | `src/main.py:116-120` — при `ENV != development` и `PYTEST\_RUNNING != true` отсутствие `API\_SECRET\_KEY` → `RuntimeError` (приложение не стартует). `src/auth.py:16-53` — soft-mode в dev, 401 при неверном ключе. `src/routers/edit.py:55` — `/api/edit` защищён `Depends(verify\_api\_key)`. `src/routers/health.py` — `/health` защищён; `/livez` — нет (для Render health check). |

| Остаточный риск | Логика уже реализована. \*\*Не хватает тестов\*\*, формально закрепляющих поведение (production без ключа не стартует; dev работает; неверный ключ → 401). |

| Следующий шаг | \*\*Итерация 1\*\* — добавить тесты и документацию. Реализацию не дублировать. |



\### Область 4. Лимиты запроса на backend



| Поле | Значение |

|---|---|

| Подтверждено | Частично |

| Доказательство | `src/contracts.py:63` — `text: str = Field(..., min\_length=1, max\_length=10000)`. Лимит текста действует. |

| Остаточный риск | \*\*Размер HTTP-тела не ограничен.\*\* В `src/main.py` есть только `CORSMiddleware`, `log\_requests`, `add\_security\_headers`. Нет проверки `Content-Length`, нет ограничения на чтение body. Клиент может отправить произвольно большое тело, и Pydantic будет парсить его целиком. |

| Следующий шаг | \*\*Итерация 2\*\* — добавить ограничение размера HTTP-тела (отдельным middleware) + тесты. |



\### Область 5. Semantic model



| Поле | Значение |

|---|---|

| Подтверждено | Да |

| Доказательство | `src/semantic\_index.py:30` — `\_SEMANTIC\_MODEL\_REVISION = "e8ed3b0c8bbf..."` (закреплённый commit SHA модели `cointegrated/rubert-tiny2`). `semantic\_index.py:203` — `revision=\_SEMANTIC\_MODEL\_REVISION` передаётся в `SentenceTransformer`. `trust\_remote\_code=True` не используется (проверено тестом `test\_no\_trust\_remote\_code\_true`). Пользователь не может выбрать модель через API (тест `test\_no\_http\_endpoint\_accepts\_model\_name`). |

| Остаточный риск | Нет |

| Следующий шаг | — |



\### Область 6. Rate limiting



| Поле | Значение |

|---|---|

| Подтверждено | Да, но с ограничениями |

| Доказательство | `src/rate\_limit.py` — slowapi, `Limiter(key\_func=\_client\_ip\_key)`, лимит `10/minute` (1000 в тестах через `PYTEST\_RUNNING=true`). Подключено в `src/main.py:169-170`. |

| Остаточный риск | \*\*In-memory\*\*. При нескольких инстансах Render (`--workers 2` в `render.yaml`) лимит считается независимо в каждом процессе. Нет внешнего хранилища (Redis). Ключ — IP-адрес клиента; за reverse proxy может быть адрес прокси. |

| Следующий шаг | \*\*Итерация 3\*\* — оценить, нужны ли распределённые лимиты или это допустимо для текущего масштаба. |



\### Область 7. Dev-зависимости в production



| Поле | Значение |

|---|---|

| Подтверждено | Да |

| Доказательство | `render.yaml:7` — `buildCommand: pip install -r requirements.txt` (только runtime). `pytest`, `mypy`, `ruff`, `pip-audit` в `requirements-dev.txt`, в production не ставятся. |

| Остаточный риск | Нет |

| Следующий шаг | — |



\### Область 8. Security CI



| Поле | Значение |

|---|---|

| Подтверждено | Да |

| Доказательство | `.github/workflows/security.yml` — 3 job'а: `Audit runtime dependencies`, `Audit dev dependencies`, `Verify torch (outside pip-audit coverage)`. Все блокирующие, без `continue-on-error`. Явные `--ignore-vuln` для 6 остаточных advisory в `transformers`. Branch protection на `main` требует все три checks. |

| Остаточный риск | Битая ссылка на Torch-документ (см. область 2). |

| Следующий шаг | — |



\## Устранённые риски



\- 12 уникальных advisory из первой дорожной карты.

\- Защита загрузки semantic-модели (закреплённая ревизия).

\- Блокирующий security-контроль в CI.



\## Принятые риски



| Риск | Обоснование | Дата пересмотра |

|---|---|---|

| `transformers 4.57.6` — 9 advisory | Все требуют `trust\_remote\_code=True` или загрузки недоверенной модели; в конфигурации проекта недостижимы. Обновление до 5.x заблокировано несовместимостью с `torch 2.5.0`. | 2026-04-08 |

| `torch 2.5.0+cpu` вне pip-audit | Проверено вручную в итерации 3 первой дорожной карты. Опасные векторы (`torch.load`, `pickle`) в коде отсутствуют. | 2026-04-08 |



\## Непроверенные пакеты



\- \*\*`torch`\*\* — не покрыт `pip-audit` (CPU-индекс PyTorch).

\- Возможные транзитивные зависимости `torch`, поставляемые из того же индекса, тоже вне охвата pip-audit.



\## Открытые дефекты (не security)



\- 4 deprecation warnings в тестах: `httpx` → `httpx2`, `HTTP\_422\_UNPROCESSABLE\_ENTITY`, pytest class-scoped fixtures, pydantic `\_\_fields\_\_`. Задокументированы в `docs/security/2026-10-final-audit.md`.

\- Битая ссылка на Torch-документ (см. область 2).



\## Ссылки



\- Issue #35: https://github.com/MaryMe88/google-docs-editor-backend/issues/35

\- Первая дорожная карта (завершена): `docs/security/2026-10-final-audit.md`

\- Документы security: `docs/security/`

\- Ветка со старым Torch-документом: `security/audit-torch-runtime` (коммит `27452b5`)

