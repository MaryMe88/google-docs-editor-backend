"""
Контракт установки Torch.

Итерация 4 новой дорожной карты остаточных security-работ (issue #35).

Проверяем, что torch установлен именно в CPU-сборке и что
requirements.txt явно требует CPU-вариант.

Это критично:
- CPU-сборка работает на Render (free tier, без видеокарты);
- CUDA-сборка весит ~2 ГБ и не запустится на Render;
- запись `torch==2.5.0` без `+cpu` в requirements.txt матчит ОБА
  варианта (2.5.0 из PyPI и 2.5.0+cpu из CPU-индекса). Если
  CPU-индекс временно недоступен, pip молча возьмёт CUDA-сборку.

Явный пин `torch==2.5.0+cpu` гарантирует воспроизводимость.

Запуск:
    pytest tests/test_torch_install_contract.py -v
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"


# ============================================================================
# Установленная версия torch
# ============================================================================


def test_torch_version_has_cpu_suffix() -> None:
    """
    torch.__version__ должен заканчиваться на +cpu.

    Формат: '2.5.0+cpu'. Если сюда прилетит просто '2.5.0'
    или '2.5.0+cu121' — мы случайно поставили CUDA-сборку.
    """
    version = torch.__version__
    assert version.endswith("+cpu"), (
        f"Ожидалась CPU-сборка torch (суффикс +cpu), "
        f"установлена: {version!r}. "
        "Это CUDA-сборка — она не запустится на Render."
    )


def test_torch_cuda_version_is_none() -> None:
    """torch.version.cuda — None для CPU-сборки."""
    assert torch.version.cuda is None, (
        f"torch.version.cuda должен быть None для CPU-сборки, " f"получено: {torch.version.cuda!r}"
    )


def test_torch_cuda_is_not_available() -> None:
    """torch.cuda.is_available() — False для CPU-сборки."""
    assert torch.cuda.is_available() is False, (
        "torch.cuda.is_available() должен быть False. " "Если True — установлена CUDA-сборка."
    )


def test_torch_importable() -> None:
    """torch импортируется без ошибок (защита от битой сборки)."""
    # Сам факт успешного `import torch` в начале файла — уже проверка.
    # Здесь добавим минимальное assert, чтобы тест не был пустым.
    assert torch is not None


# ============================================================================
# Запись в requirements.txt
# ============================================================================


def _read_requirements() -> str:
    if not REQUIREMENTS.is_file():
        pytest.fail(f"requirements.txt не найден: {REQUIREMENTS}")
    return REQUIREMENTS.read_text(encoding="utf-8")


def _find_torch_line() -> str:
    for raw_line in _read_requirements().splitlines():
        line = raw_line.strip()
        if line.startswith("torch=="):
            return line
    pytest.fail("В requirements.txt не найдена строка torch==")


def test_requirements_has_torch_pin() -> None:
    """В requirements.txt есть строка torch==<версия>."""
    line = _find_torch_line()
    assert line.startswith("torch=="), line


def test_requirements_torch_pin_has_cpu_suffix() -> None:
    """
    Строка torch== должна содержать +cpu.
    Без этого пина pip может выбрать CUDA-сборку из PyPI.
    """
    line = _find_torch_line()
    assert "+cpu" in line, (
        f"Ожидалось 'torch==X.Y.Z+cpu' в requirements.txt, "
        f"найдено: {line!r}. "
        "Без явного +cpu возможно случайное получение CUDA-сборки."
    )


def test_requirements_torch_pin_has_no_version_range() -> None:
    """
    Строка torch== не должна содержать запятую (диапазон версий).
    Диапазоны вроде '2.5.0,<3.0' — источник недетерминированности.
    """
    line = _find_torch_line()
    assert "," not in line, (
        f"В записи torch не должно быть диапазона версий "
        f"(запятой), найдено: {line!r}. "
        "Используйте точный пин 'torch==X.Y.Z+cpu'."
    )


def test_requirements_torch_pin_is_exact_semver() -> None:
    """
    Строка torch== должна соответствовать формату 'torch==X.Y.Z+cpu'.
    """
    line = _find_torch_line()
    pattern = re.compile(r"^torch==\d+\.\d+\.\d+\+cpu$")
    assert pattern.match(line), (
        f"Формат записи torch не соответствует 'torch==X.Y.Z+cpu': " f"{line!r}"
    )


# ============================================================================
# Источник установки
# ============================================================================


def test_requirements_has_cpu_index_url() -> None:
    """
    В requirements.txt должна быть строка --extra-index-url
    с адресом CPU-индекса PyTorch.
    """
    text = _read_requirements()
    assert "download.pytorch.org/whl/cpu" in text, (
        "В requirements.txt не найден CPU-индекс PyTorch. "
        "Без него pip не сможет скачать CPU-сборку."
    )


def test_sentence_transformers_requires_torch() -> None:
    """
    sanity-check: sentence-transformers действительно требует torch.
    Если это перестанет быть правдой — надо пересмотреть всю логику.
    """
    import importlib.metadata as m

    requires = m.requires("sentence-transformers") or []
    torch_requires = [r for r in requires if "torch" in r.lower()]
    assert torch_requires, (
        "sentence-transformers не требует torch — "
        "возможно, зависимость изменилась, надо пересмотреть"
    )


def test_verify_torch_script_does_not_reference_missing_doc() -> None:
    script_path = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "verify_torch.py"
    script = script_path.read_text(encoding="utf-8")
    assert "2026-10-torch-audit.md" not in script, "verify_torch.py ссылается на удалённый документ"
    assert "torch-risk-assessment.md" in script
