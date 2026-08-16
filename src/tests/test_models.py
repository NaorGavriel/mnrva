from pathlib import PurePosixPath, PureWindowsPath

from models import make_chunk_id


def test_make_chunk_id_is_the_same_regardless_of_path_separator_style() -> None:
    """A WindowsPath and a PosixPath for the same logical file must hash to
    the same id - otherwise ingesting the same repo from Windows vs
    Linux/Mac would produce different, non-matching chunk ids."""
    windows_id = make_chunk_id(
        PureWindowsPath("client\\src\\AuthCard.tsx"), "function", "", "AuthCard"
    )
    posix_id = make_chunk_id(
        PurePosixPath("client/src/AuthCard.tsx"), "function", "", "AuthCard"
    )

    assert windows_id == posix_id


def test_make_chunk_id_is_deterministic() -> None:
    """Calling make_chunk_id twice with the same inputs returns the same id."""
    first = make_chunk_id(PurePosixPath("main.py"), "function", "", "greet")
    second = make_chunk_id(PurePosixPath("main.py"), "function", "", "greet")

    assert first == second


def test_make_chunk_id_differs_for_different_symbols() -> None:
    """Different symbol_name -> different id, same file/kind/class_name otherwise."""
    greet_id = make_chunk_id(PurePosixPath("main.py"), "function", "", "greet")
    farewell_id = make_chunk_id(PurePosixPath("main.py"), "function", "", "farewell")

    assert greet_id != farewell_id
