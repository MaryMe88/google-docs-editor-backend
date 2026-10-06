"""Security tests for semantic model loading.

Итерация 5 дорожной карты устранения уязвимостей (issue #35).

Эти тесты закрепляют инварианты безопасности загрузки семантической
модели:

- ревизия модели зафиксирована конкретным commit SHA;
- trust_remote_code=True не используется;
- имя модели не приходит из HTTP-запроса;
- модель не загружается при импорте модуля;
- модель не загружается при создании SemanticIndex (lazy loading).

Тесты написаны на уровне анализа исходников и публичных инвариантов,
чтобы быть устойчивыми к внутренним рефакторингам.
"""

from __future__ import annotations

import inspect

import pytest

import src.semantic_index as si

# ---------------------------------------------------------------------------
# 1. Ревизия модели закреплена
# ---------------------------------------------------------------------------


def test_revision_constant_exists() -> None:
    """Модуль должен определять _SEMANTIC_MODEL_REVISION."""
    assert hasattr(si, "_SEMANTIC_MODEL_REVISION"), (
        "semantic_index.py должен определять _SEMANTIC_MODEL_REVISION "
        "для защиты от подмены модели на стороне Hugging Face"
    )


def test_revision_is_valid_sha() -> None:
    """Ревизия — это корректный 40-символьный git SHA."""
    revision = si._SEMANTIC_MODEL_REVISION
    assert isinstance(revision, str), "ревизия должна быть строкой"
    assert len(revision) == 40, (
        f"Hugging Face revision — 40-символьный git SHA, "
        f"получено {len(revision)} символов: {revision!r}"
    )
    assert all(
        c in "0123456789abcdef" for c in revision.lower()
    ), f"ревизия должна состоять только из hex-символов: {revision!r}"


def test_sentence_transformer_passes_revision() -> None:
    """Вызов SentenceTransformer должен передавать revision=..."""
    source = inspect.getsource(si)
    assert "SentenceTransformer(" in source, "в модуле не найден вызов SentenceTransformer"
    assert "revision=" in source, (
        "вызов SentenceTransformer должен передавать revision= "
        "для закрепления модели конкретным commit SHA"
    )


# ---------------------------------------------------------------------------
# 2. trust_remote_code не используется
# ---------------------------------------------------------------------------


def test_no_trust_remote_code_true() -> None:
    """trust_remote_code=True не должен использоваться в модуле."""
    source = inspect.getsource(si)
    # Ловим разные формы записи: True, истина, пробелы вокруг =
    forbidden = [
        "trust_remote_code=True",
        "trust_remote_code = True",
        'trust_remote_code="True"',
    ]
    for pattern in forbidden:
        assert pattern not in source, (
            f"найдена опасная конструкция {pattern!r}: "
            "выполнение кода модели на стороне сервера запрещено"
        )


# ---------------------------------------------------------------------------
# 3. Имя модели — фиксированное, не из запроса
# ---------------------------------------------------------------------------


def test_default_model_name_is_fixed() -> None:
    """Дефолтное имя модели — фиксированная строка, известная на сервере."""
    sig = inspect.signature(si.SemanticIndex.__init__)
    param = sig.parameters.get("model_name")
    assert param is not None, "SemanticIndex.__init__ должен принимать model_name"
    assert param.default == "cointegrated/rubert-tiny2", (
        f"дефолтное имя модели должно быть 'cointegrated/rubert-tiny2', "
        f"получено {param.default!r}"
    )


def test_no_http_endpoint_accepts_model_name() -> None:
    """Ни один HTTP-эндпоинт не должен принимать model_name.

    Это гарантирует, что API-клиент не может выбрать модель.
    """
    try:
        from src.main import app
    except ImportError as exc:
        pytest.skip(f"src.main недоступен: {exc}")

    for route in app.routes:
        endpoint = getattr(route, "endpoint", None)
        if endpoint is None:
            continue
        try:
            sig = inspect.signature(endpoint)
        except (TypeError, ValueError):
            continue
        assert "model_name" not in sig.parameters, (
            f"эндпоинт {getattr(route, 'path', '?')} принимает model_name — "
            "это позволит клиенту выбрать модель"
        )


# ---------------------------------------------------------------------------
# 4. Lazy loading: модель не загружается при импорте и создании индекса
# ---------------------------------------------------------------------------


def test_model_not_loaded_on_semantic_index_creation() -> None:
    """Создание SemanticIndex не должно загружать модель."""
    index = si.SemanticIndex()
    assert index._model is None, (
        "модель должна загружаться лениво (при первом semantic-запросе), "
        "а не при создании SemanticIndex"
    )


def test_no_model_instance_at_module_level() -> None:
    """На уровне модуля не должно быть загруженной модели."""
    # SentenceTransformer имеет методы encode() и get_sentence_embedding_dimension().
    # Если такой объект есть в глобалах модуля — значит модель загрузили при импорте.
    for name, value in vars(si).items():
        if name.startswith("__"):
            continue
        if isinstance(value, type):
            continue
        has_encode = hasattr(value, "encode")
        has_dim = hasattr(value, "get_sentence_embedding_dimension")
        assert not (has_encode and has_dim), (
            f"на уровне модуля обнаружен загруженный объект модели: {name!r} — "
            "модель должна загружаться лениво"
        )
