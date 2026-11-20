"""mkdocs hooks: render the README as the home page.

README links are relative to the repository root, so they are rewritten to be
relative to ``docs/``.
"""

from pathlib import Path

README = Path(__file__).parent.parent / "README.md"


def on_page_markdown(markdown: str, page, config, files) -> str:
    """Replace the home page with the README, with links fixed up."""
    if page.file.src_path == "index.md":
        return README.read_text().replace("](docs/", "](")
    return markdown
