# Torch Risk Assessment

**Version:** 2.5.0+cpu  
**Build type:** CPU-only (no CUDA)  
**Installation source:** `download.pytorch.org/whl/cpu`, pinned as `torch==2.5.0+cpu`  
**Supported Python versions:** 3.11, 3.12  
**Review date:** 2026-10-10  
**Next review date:** 2026-04-10  
**Tracking issue:** #35

## Related ML packages

| Package | Version |
|---|---|
| transformers | 4.57.6 |
| sentence-transformers | 5.6.1 |
| tokenizers | 0.22.2 |
| safetensors | 0.8.0 |
| huggingface-hub | 0.36.2 |

## Reviewed advisories

| ID | Description | Fixed in | Applicable? |
|---|---|---|---|
| CVE-2025-32434 / GHSA-53q9-r3pm-6pq6 | RCE via `torch.load(weights_only=True)` | 2.6.0 | **No** — `torch.load` не используется |
| CVE-2025-46148 | Некорректные результаты `nn.PairwiseDistance(p=2)` | 2.7.0 | **No** — не используется |

## Applicable advisories

Нет. Ни одна из проверенных advisory не затрагивает фактический код проекта.

## Non-applicable advisories

Перечисленные advisory требуют вызовов, которых нет в кодовой базе:

- `torch.load` / `pickle` / десериализация — не используются.
- `trust_remote_code=True` — не используется.
- CUDA-специфичные операции — сборка CPU-only.
- Профайлер — не используется.

## Mitigations

- `trust_remote_code=False` при загрузке модели.
- Ревизия модели закреплена проверенным commit SHA (`cointegrated/rubert-tiny2`).
- `torch.load` не вызывается нигде в коде.
- CPU-only сборка исключает CUDA-специфичную поверхность атаки.
- `numpy.load(..., allow_pickle=False)` при загрузке кеша embeddings (`src/semantic_index.py`).

## Accepted risks

**Torch не покрывается `pip-audit`.** Пакет установлен из внешнего индекса `download.pytorch.org/whl/cpu`, поэтому `pip-audit` не может получить данные об уязвимостях для него. Это известное ограничение. Вместо этого выполняется отдельная ручная проверка, задокументированная в этом файле.

## Verification

Скрипт `verify_torch.py` проверяет:
- точную версию `2.5.0`;
- CPU-only сборку (суффикс `+cpu`);
- выводит явный статус: `Torch is not covered by pip-audit. Separate Torch assessment: passed (2.5.0+cpu, CPU-only).`