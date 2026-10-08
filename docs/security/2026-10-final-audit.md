\# Final dependency security audit — 2026-10



\## Назначение



Итоговый документ дорожной карты устранения уязвимостей

(issue #35). Фиксирует состояние зависимостей проекта после

всех выполненных итераций, остаточные риски и план будущей работы.



\*\*Дата аудита:\*\* 2026-10-08.

\*\*Следующий пересмотр:\*\* не позднее 2026-04-08, либо раньше при

появлении новых critical advisory в `transformers` или `torch`.



\## Обзор выполненных итераций



| Итерация | Тема | Результат |

|---|---|---|

| 0 | Проверка отчёта pip-audit | Классификация advisory, PR #37 |

| 1 | python-dotenv | 1 advisory закрыто, PR #38 |

| 2 | FastAPI + Starlette | 2 advisory закрыто, PR #39 |

| 3 | Torch audit | Документация, PR #40 |

| 4 | pytest и плагины | 2 advisory закрыто, PR #46 |

| 5 | sentence-transformers + hardening | 1 advisory закрыто + защита загрузки модели, PR #47 |

| 6 | FastAPI + Starlette (v2, starlette 1.x) | 6 advisory закрыто, PR #48 |

| 7 | transformers | Заблокировано (несовместимость с torch 2.5), PR #49 |

| — | Пин transformers | Критический фикс production-стабильности, PR #50 |

| 8 | Финальный аудит | Этот документ |



\*\*Всего закрыто:\*\* 12 уникальных advisory.



\## Финальное состояние стека



\### Runtime (`requirements.txt`)

fastapi==0.142.2
starlette==1.3.1
uvicorn[standard]==0.30.1
pydantic==2.9.2
python-dotenv==1.2.4
httpx==0.27.2
anyio==4.15.1
slowapi==0.1.9
torch==2.5.0,<3.0
sentence-transformers==5.6.1
transformers==4.57.6
numpy==2.1.3


### Dev (`requirements-dev.txt`)

ruff==0.8.4
pytest==9.1.1
pytest-asyncio==1.4.0
pytest-cov==7.1.0
mypy==2.3.1
pip-audit==2.10.1


### Python

`3.12.2` (Windows CPU-индекс PyTorch)

## Методика аудита

Аудит выполнен в **чистом виртуальном окружении**:

```powershell
python -m venv .venv-audit
.\.venv-audit\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pip check
python -m pip_audit --progress-spinner off
python -m pytest tests/ -q

Ограничение: pip-audit -r requirements.txt не используется,
потому что на Windows с кириллическим путём он падает с
KeyboardInterrupt при создании временного venv. Аудит по
environment даёт эквивалентный результат.

Результаты аудита
Baseline в чистом venv
text
539 passed, 9 skipped
pip-audit (runtime + dev)
text
Found 9 known vulnerabilities in 1 package

Name          Version  ID               Fix Versions
transformers  4.57.6   PYSEC-2025-217   (не указан)
transformers  4.57.6   PYSEC-2026-2288  5.0.0rc3 / 5.0.0
transformers  4.57.6   PYSEC-2026-2289  5.3.0
transformers  4.57.6   PYSEC-2026-2290  (не указан) / 5.5.0
transformers  4.57.6   PYSEC-2026-3929  5.10.0
transformers  4.57.6   PYSEC-2026-4174  (не указан)

Name  Skip Reason
torch Dependency not found on PyPI and could not be audited: torch (2.5.0+cpu)
Dev-зависимости не добавили новых advisory.

Закрытые advisory (12 уникальных)
Пакет	Advisory	Закрыто в итерации
python-dotenv	1 запись	1
FastAPI + Starlette (0.x)	2 записи	2
pytest	PYSEC-2026-1845 / GHSA-6w46-j5rx-g56g	4
anyio	GHSA-82r6-8w77-94w6 (critical)	4
sentence-transformers	PYSEC-2026-4164 (CVSS 9.8)	5
Starlette (1.x фиксы)	6 уникальных	6
Остаточные риски
transformers 4.57.6 — 9 advisory
Причина незакрытия: обновление до 5.x невозможно без
предварительного обновления torch до 2.7+. См. итерацию 7
(docs/security/2026-10-transformers-audit.md).

Классификация риска: принят формально.

Все 9 advisory объединяет одно — они требуют одного из:

загрузки недоверенной модели / чекпоинта;

включения trust_remote_code=True;

использования Trainer, save_pretrained(), X-CLIP, LightGlue,
RNG state persistence.

В нашей конфигурации недостижимы:

Загружается ровно одна модель — cointegrated/rubert-tiny2,
заданная константой в src/semantic_index.py.

Ревизия модели закреплена commit SHA
(e8ed3b0c8bbf..., итерация 5).

trust_remote_code=True не передаётся — проверено тестом
tests/test_semantic_model_security.py::test_no_trust_remote_code_true.

Пользователь не может выбрать модель — проверено тестом
test_no_http_endpoint_accepts_model_name.

Пользователь не может передать путь к модели — имя
зафиксировано дефолтом SemanticIndex.__init__.

Trainer / save_pretrained() не используются — только
SentenceTransformer.encode() для inference.

Mitigation: строгий security-контроль в CI после итерации 9
заблокирует появление новых critical advisory без обсуждения.

torch 2.5.0+cpu — не покрыт pip-audit
Причина: torch установлен из отдельного CPU-индекса
download.pytorch.org/whl/cpu, которого нет в PyPI. pip-audit
не может его проанализировать.

Классификация риска: принят формально после ручной проверки.

См. docs/security/2026-10-torch-audit.md (итерация 3):

Опасные векторы (torch.load, pickle, trust_remote_code,
from_pretrained) в коде отсутствуют.

Модель публичная, из доверенного HF-пространства.

Deep semantic search выключен по умолчанию.

Заблокированные обновления
Пакет	Текущая	Целевая	Блокер
transformers	4.57.6	5.10.1+	требует torch >= 2.7
torch	2.5.0+cpu	2.7+	отдельная итерация
План будущей работы
Итерация 7a — Torch upgrade (подготовительная)
Проверить совместимость torch 2.7+ с sentence-transformers 5.6.1.

Установить CPU-сборку из download.pytorch.org/whl/cpu.

Прогнать полный baseline.

Форсированный rebuild эмбеддингов и сравнение с
golden-эталоном (методика итерации 5).

При Max abs diff > 1e-3 — исследовать изменения.

Итерация 7b — transformers 5.x
Только после успешной 7a:

Обновить transformers до последней стабильной 5.x.

Синхронизировать huggingface-hub (major).

Прогнать baseline + rebuild + golden-тесты.

Итерация 9 — Блокирующий security-контроль в CI
Сделать security-проверки обязательными. См. дорожную карту.

Deprecation warnings (не блокеры)
Зафиксированы в текущем стеке, требуют отдельной работы:

Starlette 1.3.1: Using httpx with starlette.testclient is deprecated; install httpx2 instead.
Требует миграции тестового клиента на httpx2.

Starlette 1.3.1: HTTP_422_UNPROCESSABLE_ENTITY переименован в
HTTP_422_UNPROCESSABLE_CONTENT. Влияет на tests/test_main.py:244.

pytest 9: PytestRemovedIn10Warning про class-scoped fixture
как instance method в tests/test_contracts.py.

Pydantic 2.9.2: PydanticDeprecatedSince20 про __fields__.
Исправление — в тестах.

Ни один не влияет на production-поведение. Будут исправлены
отдельными PR по мере обновления соответствующих пакетов.

Итог
12 уникальных advisory закрыто за 8 итераций.

9 остаточных advisory в transformers 4.57.6 — формально
приняты, недостижимы в текущей конфигурации.

torch 2.5.0+cpu — вне охвата pip-audit, проверен вручную.

transformers 5.x — ждёт обновления torch.

Производственная конфигурация стабильна, тесты зелёные
(539 passed, 9 skipped), следующий деплой на Render не сломается.

Ссылки
Issue #35: https://github.com/MaryMe88/google-docs-editor-backend/issues/35

Audit (итерация 0): docs/security/2026-10-dependency-audit.md

Torch audit (итерация 3): docs/security/2026-10-torch-audit.md

Transformers audit (итерация 7): docs/security/2026-10-transformers-audit.md
