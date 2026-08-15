from typing import Protocol

import tree_sitter_c_sharp as tscsharp
import tree_sitter_javascript as tsjavascript
import tree_sitter_python as tspython
import tree_sitter_typescript as tstypescript
from tree_sitter import Language, Parser


class GrammarRegistry(Protocol):
    """Interface for looking up a tree-sitter `Parser` by canonical language name.

    The seam that lets `code_parser.py` and its tests depend on an interface
    instead of the concrete grammar packages.
    """

    def get_parser(self, language: str) -> Parser:
        """Return the `Parser` configured for `language`."""
        ...


class LanguageRegistry:
    """Eagerly builds and caches a `Parser` per supported language."""

    def __init__(self) -> None:
        """Build every supported language's parser now, so a broken/missing
        grammar fails immediately here rather than on first use."""
        self._parsers: dict[str, Parser] = {
            "python": Parser(Language(tspython.language())),
            "javascript": Parser(Language(tsjavascript.language())),
            "typescript": Parser(Language(tstypescript.language_typescript())),
            "tsx": Parser(Language(tstypescript.language_tsx())),
            "csharp": Parser(Language(tscsharp.language())),
        }

    def get_parser(self, language: str) -> Parser:
        """Return the cached `Parser` for `language`."""
        try:
            return self._parsers[language]
        except KeyError:
            raise ValueError(
                f"unsupported language {language!r}; supported: {sorted(self._parsers)}"
            ) from None
