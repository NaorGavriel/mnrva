from typing import Protocol

import tree_sitter_c_sharp as tscsharp
import tree_sitter_javascript as tsjavascript
import tree_sitter_python as tspython
import tree_sitter_typescript as tstypescript
from tree_sitter import Language, Parser


class GrammarRegistry(Protocol):
    def get_parser(self, language: str) -> Parser: ...


class LanguageRegistry:
    def __init__(self) -> None:
        self._parsers: dict[str, Parser] = {
            "python": Parser(Language(tspython.language())),
            "javascript": Parser(Language(tsjavascript.language())),
            "typescript": Parser(Language(tstypescript.language_typescript())),
            "tsx": Parser(Language(tstypescript.language_tsx())),
            "csharp": Parser(Language(tscsharp.language())),
        }

    def get_parser(self, language: str) -> Parser:
        return self._parsers[language]
