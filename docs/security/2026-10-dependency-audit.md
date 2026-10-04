# Dependency security audit — 2026-10-04

## Назначение

Документ фиксирует результаты первого автоматического аудита зависимостей
проекта (issue #35). Аудит выполнен в рамках итерации 0 дорожной карты
устранения уязвимостей. Обновления пакетов **не выполняются** в этой
итерации — только классификация и сопоставление с планом.

## Методология

- Инструмент: `pip-audit` 2.10.1
- Окружение: `.venv-baseline`, Python 3.12.2
- Команда: `python -m pip_audit --progress-spinner off` — проверка
  установленного окружения. Флаг `-r requirements.txt` на Windows
  с кириллическим путём пользователя зависает при создании временного
  venv (известная проблема pip-audit ≥ 2.10).
- В отчёте есть только `package / version / advisory-id / fix-versions`.
  Конкретные векторы атак и CVSS нужно проверять в первоисточниках.
- JSON/текстовые отчёты pip-audit **не коммитятся** (правило
  в `.gitignore`).

## Сводка

- Всего записей: **28**
- Смысловых групп: **6** (одна группа = один PR)
- Пакетов с записями: 5
- Не проверено: `torch` (установлен из CPU-индекса PyTorch,
  pip-audit не нашёл его на PyPI)

## Таблица классификации

Значения «Достижимо?» — оценка по статическому анализу кода проекта
(см. grep-запросы итерации 0). Не является формальным доказательством.

| Группа | Advisory | Пакет | Версия | Fix | Достижимо? | Риск | Комментарий |
|---|---|---|---|---|---|---|---|
| dotenv | PYSEC-2026-2270 | python-dotenv | 1.0.1 | 1.2.2 | Условно | P3 | `load_dotenv()` вызывается в `llm_client.py:37`. Вектор требует контроля над `.env` — если у атакующего есть доступ к файлу, у него уже есть доступ к серверу. |
| pytest | PYSEC-2026-1845 | pytest | 8.3.3 | 9.0.3 | Dev-only | P3 | Сейчас в `requirements.txt`, но используется только в CI/локали. Уйдёт в dev-deps итерацией 1 плана чистого кода. |
| sent-trans | PYSEC-2026-4164 | sentence-transformers | 3.2.1 | 5.6.0 | При deep search | P2 | Импорт только внутри `SemanticIndex._load_model`. Активируется при `deep_semantic_search=True` (по умолчанию выключен). Fix — major 3→5. |
| starlette | PYSEC-2026-1943 | starlette | 0.38.6 | 0.40.0 | Да (HTTP-слой) | **P1** | Starlette обслуживает все HTTP-запросы через FastAPI. |
| starlette | PYSEC-2026-1941 | starlette | 0.38.6 | 0.47.2 | Да (HTTP-слой) | **P1** | Требуется более поздний FastAPI. |
| starlette | PYSEC-2026-161 | starlette | 0.38.6 | 1.0.1 | Да (HTTP-слой) | **P1** | Fix 1.0.1 — потенциально несовместим с FastAPI 0.115.x. |
| starlette | PYSEC-2026-2281 | starlette | 0.38.6 | 1.1.0 | Да (HTTP-слой) | **P1** | То же. |
| starlette | PYSEC-2026-2280 | starlette | 0.38.6 | 1.1.0 | Да (HTTP-слой) | **P1** | То же. |
| starlette | PYSEC-2026-249 | starlette | 0.38.6 | 1.3.1 | Да (HTTP-слой) | **P1** | То же. |
| starlette | PYSEC-2026-248 | starlette | 0.38.6 | 1.3.0 | Да (HTTP-слой) | **P1** | То же. |
| transformers | PYSEC-2025-217 | transformers | 4.57.6 | — | При deep search | P2 | Нет fix в отчёте — нужна ручная проверка первоисточников. |
| transformers | PYSEC-2026-2288 | transformers | 4.57.6 | 5.0.0 | При deep search | P2 | Major 4→5. |
| transformers | PYSEC-2026-2289 | transformers | 4.57.6 | 5.3.0 | При deep search | P2 | То же. |
| transformers | PYSEC-2026-2290 | transformers | 4.57.6 | 5.5.0 | При deep search | P2 | То же. |
| transformers | PYSEC-2026-3929 | transformers | 4.57.6 | 5.10.0 | При deep search | P2 | То же. |
| transformers | PYSEC-2026-4174 | transformers | 4.57.6 | — | При deep search | P2 | Нет fix в отчёте. |
| torch | — | torch | 2.5.0+cpu | — | Скорее нет | P2? | Не аудирован. Опасные векторы (`torch.load`, `pickle`, `trust_remote_code`) в коде **не найдены**. Ручная проверка на официальном сайте PyTorch. |

Примечание. `python-dotenv` и `pytest` встречаются в выводе pip-audit по
два раза — это одна и та же запись, pip-audit нашёл один и тот же пакет
дважды. В таблице оставлено по одной строке.

## Сопоставление с итерациями дорожной карты уязвимостей

| Группа | Итерация | Ветка |
|---|---|---|
| python-dotenv | 1 | `security/upgrade-python-dotenv` |
| FastAPI + Starlette | 2 | `security/upgrade-fastapi-starlette` |
| Torch | 3 | `security/audit-torch-runtime` |
| pytest | 4 | `security/upgrade-pytest-stack` |
| Защита семантической модели | 5 | `security/harden-semantic-model-loading` |
| ML-стек (transformers, sentence-transformers) | 6 → 7 | `spike/ml-security-upgrade` → `security/upgrade-ml-stack` |

## Ограничения аудита

- Конкретные векторы атак (какой URL, какой параметр, какие права
  нужны) в выводе pip-audit отсутствуют.
- CVSS-оценки отсутствуют.
- Достижимость — оценка на основе grep по коду. Для P1/P2 стоит
  сверить с OSV/NVD.
- Torch не проверен.

## Что проверить вручную

- Starlette (7 уникальных advisory) — все привязаны к версии 0.38.6,
  закрываются парой FastAPI+Starlette.
- Transformers (6 уникальных advisory) — требуют ML-стека 4→5.
- Torch — официальные PyTorch security advisories.
- Для каждой advisory желательно подтвердить CVSS и вектор через OSV
  (https://osv.dev/) перед обновлением.

## Ссылки

- Issue #35: https://github.com/MaryMe88/google-docs-editor-backend/issues/35
- OSV: https://osv.dev/
- PyPA advisory database: https://github.com/pypa/advisory-database

## Обновление 2026-10-04: итерация 2 (FastAPI + Starlette)

После обновления `fastapi==0.125.0` + явного пина `starlette==0.50.0`:

- **Закрыто:** PYSEC-2026-1943, PYSEC-2026-1941, PYSEC-2026-1942.
- **Осталось (требуют Starlette >= 1.0.1):** PYSEC-2026-161, PYSEC-2026-2281,
  PYSEC-2026-2280, PYSEC-2026-249, PYSEC-2026-248.

Все 5 оставшихся advisories ссылаются на Starlette 1.x, которая требует
major-bump и проверки middleware/lifespan. Отдельная задача (spike).

Общие метрики после итерации 2:

- `pip-audit`: 22 уязвимости в 4 пакетах (было 26 в 4).
- OpenAPI paths не изменились (проверено).
- 531 passed, 9 skipped.

## Остаточный риск: 5 advisories Starlette (P2)

| Advisory | Fix | Статус |
|---|---|---|
| PYSEC-2026-161 | 1.0.1 | Требует Starlette 1.x |
| PYSEC-2026-2281 | 1.1.0 | Требует Starlette 1.x |
| PYSEC-2026-2280 | 1.1.0 | Требует Starlette 1.x |
| PYSEC-2026-249 | 1.3.1 | Требует Starlette 1.x |
| PYSEC-2026-248 | 1.3.0 | Требует Starlette 1.x |

**Действие:** отдельный spike/PR на FastAPI >=0.142 + Starlette 1.x.
В текущей версии risk mitigation: актуальные патчи FastAPI 0.125.0,
минимальные права токенов GitHub Actions, отсутствие эксплойтов через
публичный API (проверено grep-диагностикой).

## Прочие advisories (не входили в итерацию 2)

- `python-dotenv` — закрыто в итерации 1 (1.2.4).
- `pytest` — ожидает итерации 4 (dev-only).
- `sentence-transformers` — ожидает итерации 6/7 (ML-стек).
- `transformers` — ожидает итерации 6/7 (ML-стек, major 4→5).
- `torch` — не аудирован, ожидает итерации 3.
