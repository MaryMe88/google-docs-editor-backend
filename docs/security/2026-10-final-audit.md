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


