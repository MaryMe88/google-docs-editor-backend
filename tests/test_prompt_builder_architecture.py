from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional, Set, Tuple


PROMPT_BUILDER_PATH = Path("src/prompt_builder/builder.py")


def _read_source() -> str:
    return PROMPT_BUILDER_PATH.read_text(encoding="utf-8")


def _read_module() -> ast.Module:
    return ast.parse(_read_source())


def _find_class(module: ast.Module, class_name: str) -> ast.ClassDef:
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError(f"Class {class_name!r} not found in {PROMPT_BUILDER_PATH}")


def _find_method(class_node: ast.ClassDef, method_name: str) -> ast.FunctionDef:
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return node
    raise AssertionError(
        f"Method {method_name!r} not found in class {class_node.name!r}"
    )


def _function_length(node: ast.FunctionDef) -> int:
    end_lineno: Optional[int] = getattr(node, "end_lineno", None)
    if end_lineno is None:
        raise AssertionError("Python 3.8+ is required for end_lineno support")
    return end_lineno - node.lineno + 1


class _BuildMetrics(ast.NodeVisitor):
    def __init__(self) -> None:
        self.branch_nodes = 0
        self.self_helper_calls: Set[str] = set()
        self.external_calls: Set[str] = set()
        self.return_count = 0

    def visit_If(self, node: ast.If) -> None:
        self.branch_nodes += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.branch_nodes += 1
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.branch_nodes += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self.branch_nodes += 1
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        self.branch_nodes += 1 + len(node.handlers)
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.branch_nodes += 1
        self.generic_visit(node)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self.branch_nodes += len(node.generators)
        self.generic_visit(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self.branch_nodes += len(node.generators)
        self.generic_visit(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self.branch_nodes += len(node.generators)
        self.generic_visit(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self.branch_nodes += len(node.generators)
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        if len(node.values) > 1:
            self.branch_nodes += len(node.values) - 1
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        self.return_count += 1
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func

        if isinstance(func, ast.Attribute):
            if isinstance(func.value, ast.Name) and func.value.id == "self":
                self.self_helper_calls.add(func.attr)
            elif isinstance(func.value, ast.Name):
                self.external_calls.add(f"{func.value.id}.{func.attr}")
            else:
                self.external_calls.add(func.attr)
        elif isinstance(func, ast.Name):
            self.external_calls.add(func.id)

        self.generic_visit(node)


def _collect_build_metrics() -> Tuple[ast.FunctionDef, _BuildMetrics]:
    module = _read_module()
    prompt_builder = _find_class(module, "PromptBuilder")
    build_method = _find_method(prompt_builder, "build")
    visitor = _BuildMetrics()
    visitor.visit(build_method)
    return build_method, visitor


class TestPromptBuilderArchitecture:
    """
    Архитектурные предохранители.

    Эти тесты не говорят, что бизнес-логика неверна.
    Они показывают, что PromptBuilder начинает разрастаться
    и становится слишком сложным для безопасных точечных патчей.
    """

    def test_build_method_length_is_under_control(self) -> None:
        build_method, _ = _collect_build_metrics()
        max_allowed_lines = 160
        actual_lines = _function_length(build_method)

        assert actual_lines <= max_allowed_lines, (
            "PromptBuilder.build() стал слишком длинным: "
            f"{actual_lines} строк при лимите {max_allowed_lines}. "
            "Это сигнал, что orchestration-логика накопилась и "
            "её пора выделять в более мелкие компоненты."
        )

    def test_build_method_branching_is_under_control(self) -> None:
        _, metrics = _collect_build_metrics()
        max_allowed_branch_nodes = 30 

        assert metrics.branch_nodes <= max_allowed_branch_nodes, (
            "PromptBuilder.build() стал слишком ветвистым: "
            f"{metrics.branch_nodes} узлов ветвления при лимите "
            f"{max_allowed_branch_nodes}. "
            "Это признак растущей когнитивной сложности и того, что "
            "следующие изменения лучше делать через рефакторинг, а не патчи."
        )

    def test_build_method_does_not_depend_on_too_many_self_helpers(self) -> None:
        _, metrics = _collect_build_metrics()
        max_allowed_self_helper_calls = 10

        assert len(metrics.self_helper_calls) <= max_allowed_self_helper_calls, (
            "PromptBuilder.build() дёргает слишком много внутренних helper-методов: "
            f"{sorted(metrics.self_helper_calls)} "
            f"(всего {len(metrics.self_helper_calls)}, "
            f"лимит {max_allowed_self_helper_calls}). "
            "Это означает, что фасад перегружен координацией и "
            "логика уже просится в отдельные сущности."
        )

    def test_build_method_external_call_surface_is_under_control(self) -> None:
        _, metrics = _collect_build_metrics()

        ignored_exact_calls = {
            "str",
            "set",
            "list",
            "dict",
            "sorted",
            "len",
            "isinstance",
            "TypeError",
            "ValueError",
        }

        ignored_method_suffixes = {
            "append",
            "extend",
            "strip",
            "join",
            "pop",
            "get",
            "items",
            "lower",
            "upper",
        }

        effective_external_calls = set()
        for call_name in metrics.external_calls:
            if call_name in ignored_exact_calls:
                continue

            suffix = call_name.split(".")[-1]
            if suffix in ignored_method_suffixes:
                continue

            effective_external_calls.add(call_name)

        max_allowed_external_calls = 10

        assert len(effective_external_calls) <= max_allowed_external_calls, (
            "PromptBuilder.build() зависит от слишком большого числа "
            "осмысленных внешних вызовов: "
            f"{sorted(effective_external_calls)} "
            f"(всего {len(effective_external_calls)}, "
            f"лимит {max_allowed_external_calls}). "
            "Это признак того, что orchestration размазана и build() "
            "стал слишком хрупкой точкой изменений."
        )

    def test_build_method_has_single_facade_style_return(self) -> None:
        _, metrics = _collect_build_metrics()
        max_allowed_returns = 3

        assert metrics.return_count <= max_allowed_returns, (
            "У PromptBuilder.build() слишком много точек выхода: "
            f"{metrics.return_count} при лимите {max_allowed_returns}. "
            "Для фасада это часто означает накопление частных случаев и "
            "затрудняет безопасный рефакторинг."
        )