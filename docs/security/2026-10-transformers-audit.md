\# Transformers security audit — 2026-10-06



\## Назначение



Документ фиксирует результат итерации 7 дорожной карты устранения

уязвимостей (issue #35): попытку обновить `transformers` до 5.x для

закрытия оставшихся advisory.



\*\*Итог: обновление заблокировано.\*\* Новая версия `transformers` не

может быть установлена без предварительного обновления `torch` до

2.7+, что выходит за рамки текущей итерации. Остаёмся на

`transformers 4.57.6`, остаточный риск документирован ниже.



\## Что пытались сделать



\- Обновить `transformers 4.57.6` → `5.10.1`.

\- Закрыть 9 advisory в `transformers 4.57.6`, оставшихся после

&#x20; итераций 4–6 (см. issue #35).



\## Что произошло



Dry-run показал, что `transformers 5.10.1` тянет `huggingface-hub`

1.33.0 (major) плюс новые транзитивные зависимости `hf-xet`,

`typer`, `shellingham`. `torch` и `sentence-transformers` dry-run

не трогал, поэтому установка казалась безопасной.



После установки тесты прошли (\*\*539 passed, 9 skipped\*\*), потому что

`tests/test\_semantic\_index.py` использует моки модели и не загружает

реальную `cointegrated/rubert-tiny2`.



При попытке форсированного rebuild с настоящей моделью

(`force\_rebuild=True`) получена ошибка:

AttributeError: module 'torch' has no attribute 'float8_e8m0fnu'


Файл `transformers/integrations/finegrained_fp8.py:45` безусловно
обращается к `torch.float8_e8m0fnu` при импорте. Эта константа
появилась только в **torch 2.7.0**. У нас `torch 2.5.0+cpu`.

Из-за механизма `transformers.utils.import_utils` реальная причина
маскируется под `ModuleNotFoundError: Could not import module
'PreTrainedModel'`. Наш `src/semantic_index.py` интерпретирует это
как `ImportError` и сообщает о ненайденной `sentence-transformers`,
хотя фактическая проблема — несовместимость `torch`.

## Решение

**Откат к `transformers 4.57.6` и `huggingface-hub 0.36.2`.** Стек
вернулся в рабочее состояние:

transformers 4.57.6
huggingface-hub 0.36.2
torch 2.5.0+cpu
sentence-transformers 5.6.1


Baseline: **539 passed, 9 skipped**.

## Остаточные advisory в transformers 4.57.6

По данным pip-audit (2026-10):

| ID | Суть | Fixed in |
|---|---|---|
| PYSEC-2025-217 | X-CLIP checkpoint conversion → RCE | не указан |
| PYSEC-2026-2288 | Trainer `_load_rng_state()` → RCE через `torch.load` | 5.0.0 |
| PYSEC-2026-2289 | Config injection через `_attn_implementation_internal` | 5.3.0 |
| PYSEC-2026-2290 | LightGlue model loading bypass `trust_remote_code` | 5.5.0 |
| PYSEC-2026-3929 | Path traversal в `save_pretrained()` | 5.10.0 |
| PYSEC-2026-4174 | Запись remote Python-файлов без consent | 5.10.0 |

## Анализ достижимости

Все перечисленные advisory объединяет одно: **они требуют загрузки
недоверенной модели/чекпоинта или включения `trust_remote_code=True`**.
В нашем проекте:

- **Загружается ровно одна модель** — `cointegrated/rubert-tiny2`,
  заданная константой в `src/semantic_index.py`.
- **Ревизия модели закреплена** конкретным commit SHA
  (`_SEMANTIC_MODEL_REVISION = "e8ed3b0c8bbf..."`, итерация 5).
- **`trust_remote_code=True` не передаётся** — проверено тестом
  `tests/test_semantic_model_security.py::test_no_trust_remote_code_true`.
- **Пользователь API не может выбрать модель** — проверено тестом
  `test_no_http_endpoint_accepts_model_name`.
- **Пользователь не может передать путь к модели** — имя модели
  зафиксировано дефолтом `SemanticIndex.__init__`.
- **`save_pretrained()` не вызывается** — мы не сохраняем модели.
- **Trainer не используется** — мы только inference через
  `SentenceTransformer.encode()`.
- **X-CLIP / LightGlue не используются** — это модели для vision.

**Вывод:** ни один из advisory не достижим через HTTP-API проекта
и не достижим через локальные действия без прав на запись в кэш
Hugging Face.

## Что делать в будущем

Обновление `transformers` до 5.x должно быть **отдельной итерацией
после обновления torch**:

### Итерация 7a — Torch upgrade (подготовительная)

1. Проверить совместимость `torch 2.7+` с `sentence-transformers 5.6.1`.
2. Установить CPU-сборку `torch 2.7.x` из `download.pytorch.org/whl/cpu`.
3. Прогнать полный baseline.
4. **Форсированный rebuild эмбеддингов** (`force_rebuild=True`) и
   сравнение с golden-эталоном (см. методику в итерации 5).
5. Если `Max abs diff` эмбеддингов > 1e-3 — исследовать, что
   изменилось, и решать, допустимо ли это.

### Итерация 7b — transformers 5.x

Только после успешной 7a:

1. Обновить `transformers` до последней стабильной 5.x.
2. Синхронизировать `huggingface-hub` до 1.x.
3. Заново прогнать baseline + rebuild + golden-тесты.

## Ссылки

- Issue #35: https://github.com/MaryMe88/google-docs-editor-backend/issues/35
- Дорожная карта: см. `docs/security/2026-10-dependency-audit.md`
- Torch audit: см. `docs/security/2026-10-torch-audit.md`
- Модель: https://huggingface.co/cointegrated/rubert-tiny2
