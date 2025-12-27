"""
Scalable HTML to Markdown converter for medical service files.

Converts all HTML files in a directory to clean Markdown format.
Preserves Hebrew text and table structures for better LLM embeddings.

Usage:
    python scripts/parse_html.py
"""

import os
from pathlib import Path
from markdownify import markdownify as md


def convert_html_to_markdown(html_content: str) -> str:
    """
    Convert HTML content to clean Markdown format.

    Args:
        html_content: Raw HTML string

    Returns:
        Formatted Markdown string
    """
    # Convert HTML to Markdown with table support
    markdown_content = md(
        html_content,
        heading_style="ATX",  # Use # for headings
        bullets="-",  # Use - for bullet points
        strip=['script', 'style'],  # Remove script/style tags
    )

    return markdown_content.strip()


def convert_directory(
    input_dir: str = "phase2_data",
    output_dir: str = "data/knowledge_base_markdown"
) -> None:
    """
    Convert all HTML files in input directory to Markdown in output directory.

    Args:
        input_dir: Directory containing HTML files
        output_dir: Directory to save Markdown files
    """
    # Create output directory if it doesn't exist
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all HTML files
    html_files = list(input_path.glob("*.html"))

    if not html_files:
        print(f"ERROR: No HTML files found in {input_dir}")
        return

    print(f"Found {len(html_files)} HTML files in {input_dir}")
    print(f"Converting to Markdown in {output_dir}\n")

    converted_count = 0

    for html_file in html_files:
        try:
            # Read HTML file with UTF-8 encoding (for Hebrew text)
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()

            # Convert to Markdown
            markdown_content = convert_html_to_markdown(html_content)

            # Generate output filename (replace .html with .md)
            output_filename = html_file.stem + ".md"
            output_filepath = output_path / output_filename

            # Write Markdown file
            with open(output_filepath, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            print(f"SUCCESS: {html_file.name} -> {output_filename}")
            converted_count += 1

        except Exception as e:
            print(f"ERROR: Failed to convert {html_file.name}: {e}")

    print(f"\nSuccessfully converted {converted_count}/{len(html_files)} files")


def main():
    """Main entry point for the script."""
    print("=" * 60)
    print("HTML to Markdown Converter for Medical Services")
    print("=" * 60)
    print()

    convert_directory()

    print()
    print("Conversion complete!")
    print(f"Markdown files saved in: data/knowledge_base_markdown/")


if __name__ == "__main__":
    main()
