"""Проверка синхронизации Google Apps Script и backend-контрактов.

Тест читает реальный Apps Script-файл, извлекает MODE_CONFIG без выполнения
JavaScript и проверяет, что domain, intent, overlays и output_mode
поддерживаются backend.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pytest

from src.shared_contracts import (
    ALLOWED_DOMAINS,
    ALLOWED_INTENTS,
    ALLOWED_OUTPUT_MODES,
    ALLOWED_OVERLAYS,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CLIENT_SCRIPT_PATH = (
    REPOSITORY_ROOT / "google_apps_script" / "New Script.js"
)

CLIENT_SCRIPT_PATH = Path(
    os.environ.get(
        "GOOGLE_APPS_SCRIPT_PATH",
        str(DEFAULT_CLIENT_SCRIPT_PATH),
    )
)


class ModeConfigParseError(ValueError):
    """Ошибка разбора MODE_CONFIG в Apps Script."""


def _find_matching_brace(source: str, opening_index: int) -> int:
    """Находит закрывающую скобку объекта с учётом строк и комментариев."""
    if opening_index >= len(source) or source[opening_index] != "{":
        raise ModeConfigParseError(
            "Ожидалась открывающая фигурная скобка."
        )

    depth = 0
    index = opening_index
    quote: Optional[str] = None
    in_line_comment = False
    in_block_comment = False

    while index < len(source):
        current = source[index]
        next_char = source[index + 1] if index + 1 < len(source) else ""

        if in_line_comment:
            if current == "\n":
                in_line_comment = False
            index += 1
            continue

        if in_block_comment:
            if current == "*" and next_char == "/":
                in_block_comment = False
                index += 2
                continue
            index += 1
            continue

        if quote is not None:
            if current == "\\":
                index += 2
                continue

            if current == quote:
                quote = None

            index += 1
            continue

        if current in {"'", '"', "`"}:
            quote = current
            index += 1
            continue

        if current == "/" and next_char == "/":
            in_line_comment = True
            index += 2
            continue

        if current == "/" and next_char == "*":
            in_block_comment = True
            index += 2
            continue

        if current == "{":
            depth += 1
        elif current == "}":
            depth -= 1

            if depth == 0:
                return index

            if depth < 0:
                raise ModeConfigParseError(
                    "Обнаружена лишняя закрывающая фигурная скобка."
                )

        index += 1

    raise ModeConfigParseError(
        "Не удалось найти закрывающую фигурную скобку MODE_CONFIG."
    )


def _extract_mode_config_object(source: str) -> str:
    """Возвращает содержимое объекта MODE_CONFIG без внешних скобок."""
    match = re.search(
        r"\bconst\s+MODE_CONFIG\s*=\s*\{",
        source,
    )

    if match is None:
        raise ModeConfigParseError(
            "В Apps Script не найдена декларация "
            "const MODE_CONFIG = {...}."
        )

    opening_index = source.find("{", match.start())
    closing_index = _find_matching_brace(source, opening_index)

    return source[opening_index + 1:closing_index]


def _iter_top_level_modes(
    mode_config: str,
) -> Iterator[Tuple[str, str]]:
    """Итерирует mode_id и содержимое объекта режима верхнего уровня."""
    pattern = re.compile(
        r"""
        (?P<mode_id>[A-Za-z_][A-Za-z0-9_]*)
        \s*:\s*
        \{
        """,
        re.VERBOSE,
    )

    position = 0

    while position < len(mode_config):
        match = pattern.search(mode_config, position)

        if match is None:
            return

        opening_index = mode_config.find("{", match.start())
        closing_index = _find_matching_brace(mode_config, opening_index)

        yield (
            match.group("mode_id"),
            mode_config[opening_index + 1:closing_index],
        )

        position = closing_index + 1


def _extract_required_string(
    mode_id: str,
    body: str,
    field_name: str,
) -> str:
    """Извлекает обязательное строковое поле режима."""
    match = re.search(
        rf"\b{re.escape(field_name)}\s*:\s*['\"]([^'\"]+)['\"]",
        body,
    )

    if match is None:
        raise ModeConfigParseError(
            f'У режима "{mode_id}" отсутствует строковое поле '
            f'"{field_name}".'
        )

    value = match.group(1).strip()

    if not value:
        raise ModeConfigParseError(
            f'У режима "{mode_id}" поле "{field_name}" пустое.'
        )

    return value


def _extract_optional_string(
    mode_id: str,
    body: str,
    field_name: str,
) -> Optional[str]:
    """Извлекает необязательное строковое поле или null."""
    match = re.search(
        rf"\b{re.escape(field_name)}\s*:\s*"
        r"(null|['\"]([^'\"]+)['\"])",
        body,
    )

    if match is None:
        return None

    if match.group(1) == "null":
        return None

    value = match.group(2)

    if value is None or not value.strip():
        raise ModeConfigParseError(
            f'Не удалось разобрать поле "{field_name}" '
            f'режима "{mode_id}".'
        )

    return value.strip()


def _extract_overlays(mode_id: str, body: str) -> List[str]:
    """Извлекает массив строк overlays."""
    match = re.search(
        r"\boverlays\s*:\s*\[([^\]]*)\]",
        body,
        flags=re.DOTALL,
    )

    if match is None:
        raise ModeConfigParseError(
            f'У режима "{mode_id}" отсутствует массив "overlays".'
        )

    raw_items = match.group(1)
    overlays = re.findall(r"['\"]([^'\"]+)['\"]", raw_items)

    if not overlays:
        raise ModeConfigParseError(
            f'У режима "{mode_id}" массив "overlays" пуст '
            "или содержит нестроковые значения."
        )

    return [overlay.strip() for overlay in overlays]


def load_client_modes(script_path: Path) -> Dict[str, Dict[str, Any]]:
    """Читает и разбирает MODE_CONFIG из реального Apps Script-файла."""
    if not script_path.is_file():
        pytest.fail(
            "Не найден Apps Script-файл для проверки синхронизации:\n"
            f"  {script_path}\n\n"
            "Добавьте файл в Git, например:\n"
            "  google_apps_script/New Script.js\n\n"
            "Или для локального запуска задайте переменную окружения:\n"
            "  GOOGLE_APPS_SCRIPT_PATH"
        )

    source = script_path.read_text(encoding="utf-8")
    mode_config = _extract_mode_config_object(source)

    modes: Dict[str, Dict[str, Any]] = {}

    for mode_id, body in _iter_top_level_modes(mode_config):
        if mode_id in modes:
            raise ModeConfigParseError(
                f'Режим "{mode_id}" объявлен в MODE_CONFIG более одного раза.'
            )

        modes[mode_id] = {
            "domain": _extract_required_string(mode_id, body, "domain"),
            "intent": _extract_optional_string(mode_id, body, "intent"),
            "overlays": _extract_overlays(mode_id, body),
            "output_mode": _extract_optional_string(
                mode_id,
                body,
                "output_mode",
            )
            or "text_only",
        }

    if not modes:
        raise ModeConfigParseError(
            "MODE_CONFIG найден, но в нём не обнаружено ни одного режима."
        )

    return modes


@pytest.fixture(scope="module")
def client_modes() -> Dict[str, Dict[str, Any]]:
    """Возвращает режимы из реального Google Apps Script-клиента."""
    return load_client_modes(CLIENT_SCRIPT_PATH)


def test_client_script_exists() -> None:
    """Apps Script должен быть доступен для проверки."""
    assert CLIENT_SCRIPT_PATH.is_file(), (
        "Apps Script не найден: "
        f"{CLIENT_SCRIPT_PATH}"
    )


def test_all_client_domains_are_allowed(
    client_modes: Dict[str, Dict[str, Any]],
) -> None:
    """Каждый domain из MODE_CONFIG должен существовать на backend."""
    client_domains = {
        str(mode["domain"])
        for mode in client_modes.values()
    }

    unsupported = client_domains - ALLOWED_DOMAINS

    assert not unsupported, (
        "В реальном Google Apps Script найдены domain, "
        "которых нет в ALLOWED_DOMAINS: "
        f"{sorted(unsupported)}"
    )


def test_all_client_intents_are_allowed(
    client_modes: Dict[str, Dict[str, Any]],
) -> None:
    """Каждый непустой intent из MODE_CONFIG должен поддерживаться backend."""
    client_intents = {
        str(mode["intent"])
        for mode in client_modes.values()
        if mode["intent"] is not None
    }

    unsupported = client_intents - ALLOWED_INTENTS

    assert not unsupported, (
        "В реальном Google Apps Script найдены intent, "
        "которых нет в ALLOWED_INTENTS: "
        f"{sorted(unsupported)}"
    )


def test_all_client_overlays_are_allowed(
    client_modes: Dict[str, Dict[str, Any]],
) -> None:
    """Каждый overlay из MODE_CONFIG должен поддерживаться backend."""
    client_overlays = {
        overlay
        for mode in client_modes.values()
        for overlay in mode["overlays"]
    }

    unsupported = client_overlays - ALLOWED_OVERLAYS

    assert not unsupported, (
        "В реальном Google Apps Script найдены overlays, "
        "которых нет в ALLOWED_OVERLAYS: "
        f"{sorted(unsupported)}"
    )


def test_all_client_output_modes_are_allowed(
    client_modes: Dict[str, Dict[str, Any]],
) -> None:
    """Каждый output_mode из MODE_CONFIG должен поддерживаться backend."""
    client_output_modes = {
        str(mode["output_mode"])
        for mode in client_modes.values()
    }

    unsupported = client_output_modes - ALLOWED_OUTPUT_MODES

    assert not unsupported, (
        "В реальном Google Apps Script найдены output_mode, "
        "которых нет в ALLOWED_OUTPUT_MODES: "
        f"{sorted(unsupported)}"
    )


def test_every_client_mode_has_minimum_contract(
    client_modes: Dict[str, Dict[str, Any]],
) -> None:
    """Каждый режим обязан содержать непустой domain и overlays."""
    for mode_id, mode in client_modes.items():
        assert isinstance(mode["domain"], str) and mode["domain"], (
            f'Режим "{mode_id}" не содержит корректный domain.'
        )

        assert mode["intent"] is None or isinstance(mode["intent"], str), (
            f'Режим "{mode_id}" содержит некорректный intent.'
        )

        assert isinstance(mode["overlays"], list) and mode["overlays"], (
            f'Режим "{mode_id}" не содержит непустой массив overlays.'
        )

        assert all(
            isinstance(overlay, str) and overlay
            for overlay in mode["overlays"]
        ), (
            f'Режим "{mode_id}" содержит некорректное значение overlays.'
        )

        assert isinstance(mode["output_mode"], str) and mode["output_mode"], (
            f'Режим "{mode_id}" не содержит корректный output_mode.'
        )