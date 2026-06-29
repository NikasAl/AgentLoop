"""
Scanner — сканирует систему для Layer 2.

Проверяет:
- Какие утилиты доступны в PATH (через `which`)
- Какие Python-пакеты установлены (через `pip list`)

Результат кешируется в ~/.agentloop/tool_cache.yaml.
При установке нового пакета — кеш обновляется через steward или вручную (--refresh).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

from .base import (
    InstallSpec,
    ToolCategory,
    ToolDescriptor,
    ToolLayer,
)

# Где хранить кеш
CACHE_DIR = Path(os.getenv("AGENTLOOP_CACHE_DIR", Path.home() / ".agentloop"))
CACHE_FILE = CACHE_DIR / "tool_cache.yaml"

# Список утилит, которые сканируем в PATH
# Полный список с описаниями — в known_tools.yaml
KNOWN_UTILITIES_DEFAULT = [
    # PDF
    "pdftoppm", "pdftotext", "pdfimages", "pdfinfo", "pdftk",
    "mutool", "qpdf", "ghostscript", "gs",
    # OCR
    "tesseract",
    # LaTeX
    "pdflatex", "latexmk", "xelatex",
    # Images
    "convert", "magick", "identify", "mogrify",
    # Audio/Video
    "ffmpeg", "ffprobe", "sox",
    # Text/Markdown
    "pandoc", "jq", "yq", "xq",
    # Office
    "libreoffice", "soffice",
    # Network
    "curl", "wget", "ssh",
    # System
    "python3", "python", "pip", "pip3", "pipx",
    # Arch Linux specific
    "pacman", "yay", "paru",
    # Editors
    "subl", "vim", "nano", "code",
    # Version control
    "git",
]


class SystemScanner:
    """Сканирует систему для Layer 2."""

    def __init__(self, cache_file: Path | None = None):
        self.cache_file = cache_file or CACHE_FILE
        self._known_tools: dict[str, dict] = {}
        self._load_known_tools()

    def _load_known_tools(self) -> None:
        """Загружает описания известных утилит из known_tools.yaml."""
        yaml_path = Path(__file__).parent / "known_tools.yaml"
        if yaml_path.exists():
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            self._known_tools = {t["name"]: t for t in data.get("tools", [])}

    def scan_all(self, use_cache: bool = True) -> list[ToolDescriptor]:
        """
        Полное сканирование: PATH + pip list.
        Возвращает список ToolDescriptor для Layer 2.
        """
        if use_cache:
            cached = self._load_cache()
            if cached:
                return self._deserialize(cached)

        tools: list[ToolDescriptor] = []
        tools.extend(self.scan_path())
        tools.extend(self.scan_pip())

        self._save_cache(tools)
        return tools

    def scan_path(self) -> list[ToolDescriptor]:
        """Сканирует PATH на наличие известных утилит."""
        tools: list[ToolDescriptor] = []
        for util_name in KNOWN_UTILITIES_DEFAULT:
            path = shutil.which(util_name)
            known = self._known_tools.get(util_name, {})

            desc = ToolDescriptor(
                name=util_name,
                layer=ToolLayer.DISCOVERED,
                category=ToolCategory(known.get("category", "other")),
                description=known.get("description", f"Системная утилита {util_name}"),
                available=path is not None,
                input_schema=known.get("input_schema", {}),
                output_schema=known.get("output_schema", {}),
                example_usage=known.get("example_usage", ""),
                keywords=known.get("keywords", [util_name]),
                aliases=known.get("aliases", []),
            )

            if not desc.available and known.get("install"):
                desc.install_spec = InstallSpec(
                    managers=known["install"],
                    notes=known.get("install_notes", ""),
                )

            tools.append(desc)
        return tools

    def scan_pip(self) -> list[ToolDescriptor]:
        """Сканирует pip list для поиска Python-библиотек."""
        tools: list[ToolDescriptor] = []
        try:
            r = subprocess.run(
                ["pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if r.returncode != 0:
                return tools
            packages = json.loads(r.stdout)
        except Exception:
            return tools

        # Описания известных Python-библиотек
        known_py = {
            "pylatexenc": ("math", "LaTeX parser and encoder"),
            "sympy": ("math", "Symbolic mathematics"),
            "numpy": ("math", "Numerical computing"),
            "pandas": ("text", "Data analysis"),
            "Pillow": ("image", "Image processing"),
            "pydub": ("audio", "Audio processing"),
            "librosa": ("audio", "Music and audio analysis"),
            "stdversh": ("text", "Russian verse metre analyzer"),
            "pymorphy3": ("text", "Russian morphological analyzer"),
            "pymorphy2": ("text", "Russian morphological analyzer (legacy)"),
            "NLTK": ("text", "Natural language toolkit"),
            "spacy": ("text", "Industrial NLP"),
            "transformers": ("llm", "HuggingFace transformers"),
            "torch": ("llm", "PyTorch deep learning"),
            "tensorflow": ("llm", "TensorFlow deep learning"),
            "requests": ("network", "HTTP library"),
            "httpx": ("network", "HTTP client"),
            "beautifulsoup4": ("network", "HTML parser"),
            "lxml": ("text", "XML/HTML parser"),
            "markdown": ("text", "Markdown to HTML"),
            "pdfplumber": ("pdf", "PDF text extraction (Python)"),
            "PyPDF2": ("pdf", "PDF manipulation (Python)"),
            "pypdf": ("pdf", "PDF manipulation (Python, modern)"),
            "reportlab": ("pdf", "PDF generation"),
        }

        for pkg in packages:
            name = pkg["name"]
            if name in known_py:
                cat_str, desc_text = known_py[name]
                tools.append(
                    ToolDescriptor(
                        name=f"pip:{name}",
                        layer=ToolLayer.DISCOVERED,
                        category=ToolCategory(cat_str),
                        description=f"{desc_text} (v{pkg['version']})",
                        available=True,
                        keywords=[name.lower(), "python", "pip"],
                        aliases=[name.lower()],
                    )
                )

        return tools

    def refresh(self) -> list[ToolDescriptor]:
        """Принудительное пересканирование без кеша."""
        return self.scan_all(use_cache=False)

    def _load_cache(self) -> list[dict] | None:
        if not self.cache_file.exists():
            return None
        try:
            data = yaml.safe_load(self.cache_file.read_text(encoding="utf-8"))
            return data.get("tools") if data else None
        except Exception:
            return None

    def _save_cache(self, tools: list[ToolDescriptor]) -> None:
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "tools": self._serialize(tools),
            "last_scan": str(Path.cwd()),
        }
        self.cache_file.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")

    def _serialize(self, tools: list[ToolDescriptor]) -> list[dict]:
        result = []
        for t in tools:
            d = t.to_dict()
            if t.install_spec:
                d["install_spec"] = {
                    "managers": t.install_spec.managers,
                    "notes": t.install_spec.notes,
                }
            result.append(d)
        return result

    def _deserialize(self, raw: list[dict]) -> list[ToolDescriptor]:
        tools = []
        for d in raw:
            install_spec = None
            if "install_spec" in d:
                install_spec = InstallSpec(
                    managers=d["install_spec"]["managers"],
                    notes=d["install_spec"].get("notes", ""),
                )
            tools.append(
                ToolDescriptor(
                    name=d["name"],
                    layer=ToolLayer(d["layer"]),
                    category=ToolCategory(d["category"]),
                    description=d["description"],
                    available=d["available"],
                    install_spec=install_spec,
                    input_schema=d.get("input_schema", {}),
                    output_schema=d.get("output_schema", {}),
                    example_usage=d.get("example_usage", ""),
                    keywords=d.get("keywords", []),
                    aliases=d.get("aliases", []),
                )
            )
        return tools

    def detect_package_manager(self) -> str | None:
        """Определяет пакетный менеджер системы."""
        for mgr in ["pacman", "apt", "dnf", "yum", "zypper", "brew"]:
            if shutil.which(mgr):
                return mgr
        return None

    def detect_os(self) -> dict[str, str]:
        """Определяет OS (для Arch Linux — особый кейс)."""
        info = {"family": "unknown", "distro": "unknown"}
        try:
            if Path("/etc/arch-release").exists():
                info = {"family": "arch", "distro": "arch"}
            elif Path("/etc/os-release").exists():
                content = Path("/etc/os-release").read_text()
                for line in content.splitlines():
                    if line.startswith("ID="):
                        info["distro"] = line.split("=", 1)[1].strip('"').strip("'")
                        info["family"] = "debian" if info["distro"] in (
                            "debian", "ubuntu", "linuxmint"
                        ) else "rhel" if info["distro"] in (
                            "fedora", "centos", "rhel"
                        ) else info["distro"]
                        break
        except Exception:
            pass
        return info
