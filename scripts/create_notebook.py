# /// script
# requires-python = ">=3.11"
# ///

"""
CLI generator that creates new notebooks from the template.

Usage:
    python3 scripts/create_notebook.py 02 "Fine-Tuning" "LoRA on Apple Silicon"
"""

import argparse
import json
import re
from pathlib import Path


def slugify(text: str) -> str:
    """Convert text to lowercase-hyphenated slug."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def make_tag_pills(title: str) -> str:
    """Generate tag pill HTML spans from title words."""
    words = title.split()
    pills = []
    for word in words:
        # Strip any punctuation from the word for cleaner tags
        clean = re.sub(r"[^\w\-]", "", word)
        if clean:
            pill = (
                f'<span style="background:rgba(255,255,255,0.1); color:#e2e8f0; '
                f'padding:6px 14px; border-radius:20px; font-size:0.85em;">{clean}</span>'
            )
            pills.append(pill)
    return "\n    ".join(pills)


def replace_placeholders(source: str, section_num: str, title: str, subtitle: str, tags_html: str) -> str:
    """Replace all template placeholders in a string."""
    source = source.replace("{{SECTION_NUM}}", section_num)
    source = source.replace("{{TITLE}}", title)
    source = source.replace("{{SUBTITLE}}", subtitle)
    source = source.replace("{{TAGS}}", tags_html)
    return source


def process_cell_source(source, section_num: str, title: str, subtitle: str, tags_html: str):
    """Process a notebook cell's source field (str or list of str)."""
    if isinstance(source, list):
        return [
            replace_placeholders(line, section_num, title, subtitle, tags_html)
            for line in source
        ]
    else:
        return replace_placeholders(source, section_num, title, subtitle, tags_html)


def main():
    parser = argparse.ArgumentParser(
        description="Create a new notebook from the template."
    )
    parser.add_argument("section_number", help='Section number, e.g. "02"')
    parser.add_argument("title", help='Section title, e.g. "Fine-Tuning"')
    parser.add_argument("subtitle", help='Section subtitle, e.g. "LoRA on Apple Silicon"')
    args = parser.parse_args()

    section_num = args.section_number
    title = args.title
    subtitle = args.subtitle

    # Compute paths
    repo_root = Path(__file__).parent.parent
    template_path = Path(__file__).parent / "notebook_template.ipynb"

    slug = slugify(title)
    dir_name = f"{section_num}-{slug}"
    section_dir = repo_root / "sections" / dir_name
    notebook_filename = f"{dir_name}.ipynb"
    notebook_path = section_dir / notebook_filename

    # Read template
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    with open(template_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    # Generate tag pills HTML
    tags_html = make_tag_pills(title)

    # Replace placeholders in all cell sources
    for cell in notebook.get("cells", []):
        if "source" in cell:
            cell["source"] = process_cell_source(
                cell["source"], section_num, title, subtitle, tags_html
            )

    # Create directory and write notebook
    section_dir.mkdir(parents=True, exist_ok=True)
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
        f.write("\n")

    # --- Output ---

    rel_path = notebook_path.relative_to(repo_root)
    print(f"\n✅ Created: {rel_path}")

    # Config block for test_notebooks.py
    config_key = notebook_filename
    print(f"""
📋 Add to NOTEBOOK_CONFIGS in scripts/test_notebooks.py:

    "{config_key}": {{
        "cell-cover":       {{"type": "static"}},
        "cell-setup-check": {{"type": "static"}},
        "cell-setup":       {{"type": "live", "needs_ports": "any"}},
        "cell-helpers":     {{"type": "static"}},
        "cell-example":     {{"type": "live", "needs_ports": "any"}},
    }},
""")

    print("""📝 Next steps:
  1. Add content cells to the notebook
  2. Register the config block above in scripts/test_notebooks.py
  3. Run tests: uv run scripts/test_notebooks.py
""")


if __name__ == "__main__":
    main()
