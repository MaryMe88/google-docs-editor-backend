# Torch Risk Assessment

**Version:** 2.12.1+cpu  
**Build type:** CPU-only (no CUDA)  
**Installation source:** `download.pytorch.org/whl/cpu`, pinned as `torch==2.12.1+cpu`  
**Supported Python versions:** 3.11, 3.12  
**Review date:** 2026-11-07  
**Next review date:** 2027-05-07  
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
| CVE-2026-24747 / GHSA-63cw-57p8-fm3p | RCE via `weights_only` unpickler (SETITEM/SETITEMS on non-dict, storage size mismatch) | 2.10.0 | **No** — version patched (2.12.1+cpu); model loads via `.safetensors`, no `.bin` in cache |
| CVE-2025-46148 | Некорректные результаты `nn.PairwiseDistance(p=2)` | 2.7.0 | **No** — не используется |
| CVE-2025-2999 | Memory corruption в `torch.nn.utils.rnn.unpack_sequence` | 2.7.0 | **No** — RNN-операция, не используется |
| CVE-2025-2998 | Memory corruption в `torch.nn.utils.rnn.pad_packed_sequence` | 2.7.0 | **No** — RNN-операция, не используется |
| CVE-2025-3000 | Memory corruption в `torch.jit.script` | 2.7.0 | **No** — JIT-компиляция не используется |
| CVE-2025-2149 | Improper initialization в Quantized Sigmoid (`nnq_Sigmoid`) | 2.7.0 | **No** — квантование не используется |
| CVE-2025-2148 | Memory corruption в `torch.ops.profiler._call_end_callbacks_on_jit_fut` | 2.7.0 | **No** — профайлер не используется |
| CVE-2025-3730 | DoS в `torch.nn.functional.ctc_loss` | 2.8.0 | **No** — CTC loss не используется |
| CVE-2025-63396 | DoS при отсутствии `profiler.stop()` | 2.7.1 | **No** — профайлер не используется |
| CVE-2025-3001 / GHSA-qfhq-4f3w-5fph | Memory corruption в `torch.lstm_cell` | 2.10.0 | **No** — LSTM не используется |

## Applicable advisories

Нет. Ни одна из проверенных advisory не затрагивает фактический код проекта.

## Non-applicable advisories

Перечисленные advisory требуют вызовов, которых нет в кодовой базе:

- `torch.load` / `pickle` / десериализация — не вызываются; модель кеширована как `.safetensors` (117 MB), `pytorch_model.bin` отсутствует в snapshot.
- `trust_remote_code=True` — не используется.
- CUDA-специфичные операции — сборка CPU-only.
- Профайлер — не используется.
- RNN-операции (`unpack_sequence`, `pad_packed_sequence`, `lstm_cell`) — не используются.
- JIT-компиляция (`torch.jit.script`) — не используется.
- Квантование (`nnq_Sigmoid`) — не используется.
- CTC loss — не используется (это для speech recognition).

## Mitigations

- `trust_remote_code=False` при загрузке модели.
- Ревизия модели закреплена проверенным commit SHA (`cointegrated/rubert-tiny2`).
- `torch.load` не вызывается нигде в коде.
- CPU-only сборка исключает CUDA-специфичную поверхность атаки.
- `numpy.load(..., allow_pickle=False)` при загрузке кеша embeddings (`src/semantic_index.py`).

## Model loading evidence (2026-11-07)

Facts collected before and after the CVE-2026-24747 update, on `main` and
`cointegrated/rubert-tiny2` at revision `e8ed3b0c8bbf4fb6984c3de043bf7d2f4e5969ae`:

| Check | Result |
|---|---|
| `pytorch_model.bin` in HF snapshot | **absent** |
| `model.safetensors` in HF snapshot | **present** (117,529,600 bytes) |
| `trust_remote_code` in `config.json` | not set (defaults to False) |
| `trust_remote_code` passed anywhere in `src/` | **not mentioned** |
| `torch.load` called directly in `src/` | **not called** |
| `architectures` in `config.json` | `['BertForPreTraining']` (stock BertForPreTraining) |
| `revision` argument in `SentenceTransformer(...)` | pinned to commit SHA above |

**Reachability:** the CVE-2026-24747 attack path requires loading a
malicious `.pth` / `.bin` through `torch.load(weights_only=True)`. This
project never calls `torch.load` directly, and the cached snapshot ships
only `model.safetensors`. With `transformers` defaulting to safetensors
when available, the pickle path is **not reachable** for this model.

## Accepted risks

**Torch не покрывается `pip-audit`.** Пакет установлен из внешнего индекса `download.pytorch.org/whl/cpu`, поэтому `pip-audit` не может получить данные об уязвимостях для него. Это известное ограничение. Вместо этого выполняется отдельная ручная проверка, задокументированная в этом файле.

## Verification

Скрипт `verify_torch.py` проверяет:
- ожидаемую версию по prefix `2.12.1` (не exact-match — механизм является
  version guard, а не vulnerability scanner);
- CPU-only сборку (суффикс `+cpu`);
- выводит явный статус: `Torch is not covered by pip-audit. Separate Torch assessment: passed (2.12.1+cpu, CPU-only).`