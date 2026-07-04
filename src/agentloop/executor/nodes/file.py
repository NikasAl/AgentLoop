"""
FileNode — first-class файловые операции.

Поддерживает: read, write, append, list, move, copy, delete, exists, write_batch
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from ..state import PipelineState, VariableResolver
from .base import BaseNode, NodeResult


class FileNode(BaseNode):
    """Узел файловой операции."""

    def __init__(
        self,
        node_id: str,
        operation: str,  # read|write|append|list|move|copy|delete|exists|write_batch
        path: str | None = None,
        content: str | None = None,
        content_from: Any = None,
        path_template: str | None = None,
        from_collection: str | None = None,
        content_field: str = "text",
        format: str = "text",  # text|json|jsonl
        pattern: str | None = None,
        timeout_sec: int = 30,
        condition: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(node_id, timeout_sec, 0, condition, **kwargs)
        self.operation = operation
        self.path = path
        self.content = content
        self.content_from = content_from
        self.path_template = path_template
        self.from_collection = from_collection
        self.content_field = content_field
        self.format = format
        self.pattern = pattern

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FileNode":
        return cls(
            node_id=d["id"],
            operation=d["operation"],
            path=d.get("path"),
            content=d.get("content"),
            content_from=d.get("content_from"),
            path_template=d.get("path_template"),
            from_collection=d.get("from_collection"),
            content_field=d.get("content_field", "text"),
            format=d.get("format", "text"),
            pattern=d.get("pattern"),
            timeout_sec=d.get("timeout_sec", 30),
            condition=d.get("condition"),
            fallback_on_failure=d.get("fallback_on_failure", "error"),
        )

    def _execute(self, state: PipelineState, resolver: VariableResolver) -> NodeResult:
        op = self.operation
        if op == "read":
            return self._op_read(state, resolver)
        elif op == "write":
            return self._op_write(state, resolver)
        elif op == "append":
            return self._op_append(state, resolver)
        elif op == "list":
            return self._op_list(state, resolver)
        elif op == "move":
            return self._op_move_copy(state, resolver, move=True)
        elif op == "copy":
            return self._op_move_copy(state, resolver, move=False)
        elif op == "delete":
            return self._op_delete(state, resolver)
        elif op == "exists":
            return self._op_exists(state, resolver)
        elif op == "write_batch":
            return self._op_write_batch(state, resolver)
        else:
            return NodeResult(
                node_id=self.node_id,
                success=False,
                error=f"Unknown operation: {op}",
            )

    def _op_read(self, state: PipelineState, resolver: VariableResolver) -> NodeResult:
        path = Path(resolver.resolve(self.path))
        if not path.exists():
            return NodeResult(
                node_id=self.node_id,
                success=False,
                error=f"File not found: {path}",
            )
        content = path.read_text(encoding="utf-8")
        output: dict[str, Any] = {"content": content, "path": str(path)}

        if self.format == "json":
            try:
                output["parsed"] = json.loads(content)
            except json.JSONDecodeError as e:
                return NodeResult(
                    node_id=self.node_id,
                    success=False,
                    error=f"JSON parse error: {e}",
                )

        return NodeResult(node_id=self.node_id, success=True, output=output, files=[str(path)])

    def _op_write(self, state: PipelineState, resolver: VariableResolver) -> NodeResult:
        path = Path(resolver.resolve(self.path))

        # Определяем content
        if self.content is not None:
            content = resolver.resolve(self.content)
        elif self.content_from is not None:
            content = resolver.resolve(self.content_from)
            if isinstance(content, (dict, list)):
                if self.format == "json":
                    content = json.dumps(content, ensure_ascii=False, indent=2, default=str)
                else:
                    content = str(content)
        else:
            return NodeResult(
                node_id=self.node_id,
                success=False,
                error="Neither 'content' nor 'content_from' provided",
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

        return NodeResult(
            node_id=self.node_id,
            success=True,
            output={
                "path": str(path),
                "size_bytes": len(content.encode("utf-8")),
                # Контент в output, чтобы при использовании FileNode как exit-узла
                # evaluator видел реальные данные, а не только путь/размер.
                "content": content,
            },
            files=[str(path)],
        )

    def _op_append(self, state: PipelineState, resolver: VariableResolver) -> NodeResult:
        path = Path(resolver.resolve(self.path))

        if self.content_from is not None:
            content = resolver.resolve(self.content_from)
            if isinstance(content, (dict, list)):
                if self.format == "jsonl":
                    content = json.dumps(content, ensure_ascii=False, default=str) + "\n"
                elif self.format == "json":
                    content = json.dumps(content, ensure_ascii=False, indent=2, default=str)
                else:
                    content = str(content) + "\n"
        elif self.content is not None:
            content = resolver.resolve(self.content) + "\n"
        else:
            return NodeResult(
                node_id=self.node_id,
                success=False,
                error="Neither 'content' nor 'content_from' provided",
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(content)

        return NodeResult(
            node_id=self.node_id,
            success=True,
            output={
                "path": str(path),
                "appended_bytes": len(content.encode("utf-8")),
                # Контент добавленной порции — для видимости при exit-узле FileNode.
                "content": content,
            },
            files=[str(path)],
        )

    def _op_list(self, state: PipelineState, resolver: VariableResolver) -> NodeResult:
        import glob

        pattern = resolver.resolve(self.pattern or self.path or "*")
        matches = sorted(glob.glob(pattern))
        return NodeResult(
            node_id=self.node_id,
            success=True,
            output={"files": matches, "count": len(matches)},
            files=matches,
        )

    def _op_move_copy(self, state: PipelineState, resolver: VariableResolver, move: bool) -> NodeResult:
        src = Path(resolver.resolve(self.path))
        # Если path_template, заменяем
        dst_str = resolver.resolve(self.path_template or self.path)
        dst = Path(dst_str)
        dst.parent.mkdir(parents=True, exist_ok=True)

        if not src.exists():
            return NodeResult(
                node_id=self.node_id,
                success=False,
                error=f"Source not found: {src}",
            )

        if move:
            shutil.move(str(src), str(dst))
        else:
            shutil.copy2(str(src), str(dst))

        return NodeResult(
            node_id=self.node_id,
            success=True,
            output={"src": str(src), "dst": str(dst), "operation": "move" if move else "copy"},
            files=[str(dst)],
        )

    def _op_delete(self, state: PipelineState, resolver: VariableResolver) -> NodeResult:
        path = Path(resolver.resolve(self.path))
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
        else:
            return NodeResult(
                node_id=self.node_id,
                success=True,
                output={"path": str(path), "existed": False},
            )
        return NodeResult(
            node_id=self.node_id,
            success=True,
            output={"path": str(path), "deleted": True},
        )

    def _op_exists(self, state: PipelineState, resolver: VariableResolver) -> NodeResult:
        path = Path(resolver.resolve(self.path))
        exists = path.exists()
        return NodeResult(
            node_id=self.node_id,
            success=True,
            output={"path": str(path), "exists": exists},
        )

    def _op_write_batch(self, state: PipelineState, resolver: VariableResolver) -> NodeResult:
        """Записывает несколько файлов из коллекции."""
        collection_ref = resolver.resolve(self.from_collection or "")
        if not isinstance(collection_ref, list):
            return NodeResult(
                node_id=self.node_id,
                success=False,
                error=f"from_collection must resolve to list, got {type(collection_ref)}",
            )

        # Path template: резолвим $WORKDIR и {node.field}, но оставляем {item.field} для подстановки
        path_tmpl = self.path_template or ""
        path_tmpl = resolver.resolve(path_tmpl)

        written: list[str] = []

        for item in collection_ref:
            if not isinstance(item, dict):
                continue
            # Подставляем поля item в path_template
            item_path_str = path_tmpl
            for key, val in item.items():
                item_path_str = item_path_str.replace("{" + key + "}", str(val))

            item_path = Path(item_path_str)
            item_path.parent.mkdir(parents=True, exist_ok=True)

            content = item.get(self.content_field, "")
            if isinstance(content, (dict, list)):
                content = json.dumps(content, ensure_ascii=False, indent=2, default=str)

            item_path.write_text(content, encoding="utf-8")
            written.append(str(item_path))

        return NodeResult(
            node_id=self.node_id,
            success=True,
            output={"files_written": written, "count": len(written)},
            files=written,
        )
