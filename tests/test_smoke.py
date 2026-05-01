"""Smoke test: verify the package is importable and the environment is sane."""

import sys


def test_python_version_is_supported() -> None:
    """We pin >=3.11,<3.13 in pyproject.toml; assert the runtime matches."""
    assert (3, 11) <= sys.version_info[:2] < (3, 13), (
        f"Unsupported Python version: {sys.version_info[:3]}"
    )


def test_package_is_importable() -> None:
    """The package must be importable in editable-install mode."""
    import llm_playground

    assert llm_playground is not None


def test_package_has_version() -> None:
    """__version__ must be a non-empty string (PEP 396 convention)."""
    from llm_playground import __version__

    assert isinstance(__version__, str)
    assert len(__version__) > 0
