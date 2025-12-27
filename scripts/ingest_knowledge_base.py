"""
Knowledge Base Ingestion Script with 3-Type Chunking Strategy.

Parses Markdown files and creates three types of chunks:
1. Context chunks (6) - Category descriptions and service overviews
2. Benefit chunks (324) - Service x HMO x Tier specific pricing
3. Contact chunks (18) - HMO contact information per category

Embeds chunks with Azure OpenAI ADA-002 and stores in ChromaDB.

Usage:
    python scripts/ingest_knowledge_base.py
"""

import sys
from pathlib import Path
import re
from typing import List, Dict, Any, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings
from openai import AzureOpenAI

# Add parent directory to path to import settings
sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import get_settings


class KnowledgeBaseIngestion:
    """Handles ingestion of medical service knowledge base into ChromaDB."""

    def __init__(self):
        """Initialize Azure OpenAI client and ChromaDB."""
        # Load application settings
        self.settings = get_settings()

        # Azure OpenAI setup
        self.openai_client = AzureOpenAI(
            api_key=self.settings.AZURE_OPENAI_KEY,
            api_version=self.settings.AZURE_OPENAI_API_VERSION,
            azure_endpoint=self.settings.AZURE_OPENAI_ENDPOINT
        )

        # ChromaDB setup
        self.chroma_client = chromadb.PersistentClient(
            path="./vector_db",
            settings=ChromaSettings(anonymized_telemetry=False)
        )

        # Delete existing collection if it exists
        try:
            self.chroma_client.delete_collection(name="medical_services")
        except:
            pass

        # Create fresh collection
        self.collection = self.chroma_client.create_collection(
            name="medical_services",
            metadata={"description": "Israeli health fund medical services knowledge base"}
        )

        # Infer category from directory structure (robust to filename changes)
        self.category_keywords = {
            "alternative": ["alternative", "רפואה משלימה"],
            "dental": ["dental", "dentel", "שיניים"],
            "optometry": ["optometry", "אופטומטרי"],
            "communication": ["communication", "תקשורת"],
            "pregnancy": ["pregnancy", "pragrency", "הריון"],
            "workshops": ["workshops", "סדנאות"]
        }

        # HMO names in Hebrew and English (for detection)
        self.hmo_he_to_en = {
            "מכבי": "maccabi",
            "מאוחדת": "meuhedet",
            "כללית": "clalit"
        }

        # Tier names in Hebrew and English (for detection)
        self.tier_he_to_en = {
            "זהב": "gold",
            "כסף": "silver",
            "ארד": "bronze"
        }

    def embed_text_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts using Azure OpenAI ADA-002 in a single API call.

        Args:
            texts: List of texts to embed (max 100 per batch)

        Returns:
            List of embedding vectors
        """
        response = self.openai_client.embeddings.create(
            input=texts,
            model="text-embedding-ada-002"
        )
        return [item.embedding for item in response.data]

    def infer_category(self, file_path: Path) -> str:
        """
        Infer category from filename using keyword matching.

        Args:
            file_path: Path to Markdown file

        Returns:
            Category name (dental, optometry, etc.)
        """
        filename_lower = file_path.stem.lower()

        for category, keywords in self.category_keywords.items():
            if any(keyword.lower() in filename_lower for keyword in keywords):
                return category

        # Fallback to filename stem
        return file_path.stem

    def extract_context_chunk(self, markdown_content: str, category: str) -> Dict[str, Any]:
        """
        Extract context chunk (title + description + service list).

        Args:
            markdown_content: Full Markdown file content
            category: Category name (dental, optometry, etc.)

        Returns:
            Dictionary with chunk text and metadata
        """
        # Extract title (first H2 heading)
        title_match = re.search(r'^##\s+(.+)$', markdown_content, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else ""

        # Find table start (first line with |)
        lines = markdown_content.split('\n')
        table_start_idx = None
        for i, line in enumerate(lines):
            if '|' in line:
                table_start_idx = i
                break

        # Extract everything before the table
        if table_start_idx is not None:
            pre_table_lines = lines[:table_start_idx]
        else:
            pre_table_lines = lines

        # Remove title line and empty lines
        description_lines = []
        for line in pre_table_lines:
            line = line.strip()
            if line and not line.startswith('##'):
                description_lines.append(line)

        description = '\n'.join(description_lines)

        # Build context text
        context_text = f"{title}\n\n{description}"

        return {
            "text": context_text,
            "metadata": {
                "type": "context",
                "category": category,
                "hmo": "all",
                "tier": "all"
            }
        }

    def parse_table_headers(self, header_row: str) -> List[str]:
        """
        Parse table header row to detect HMO column order.

        Args:
            header_row: Header row from Markdown table

        Returns:
            List of column names in order
        """
        cells = [cell.strip() for cell in header_row.split('|')]
        # Remove first and last empty cells from split
        cells = [cell for cell in cells if cell]
        return cells

    def parse_tier_benefits(self, cell_content: str) -> List[Tuple[str, str]]:
        """
        Parse tier benefits from a table cell using flexible regex.

        Args:
            cell_content: Cell content with tier benefits

        Returns:
            List of (tier_en, benefits) tuples
        """
        results = []

        # Flexible regex that handles variations:
        # - Matches tier name with or without bold markers
        # - Uses .+? instead of [^*]+? to handle bold text within benefits
        # - Lookahead finds next tier or end of string
        tier_names = '|'.join(self.tier_he_to_en.keys())
        tier_pattern = rf'(?:\*\*)?({tier_names})(?:\*\*)?\s*:\s*(.+?)(?=(?:\*\*)?(?:{tier_names})(?:\*\*)?\s*:|$)'

        matches = re.finditer(tier_pattern, cell_content, re.DOTALL)

        for match in matches:
            tier_he = match.group(1).strip()
            benefits = match.group(2).strip()

            # Map to English
            tier_en = self.tier_he_to_en.get(tier_he, tier_he)

            results.append((tier_en, benefits))

        return results

    def extract_benefit_chunks(self, markdown_content: str, category: str) -> List[Dict[str, Any]]:
        """
        Extract benefit chunks from Markdown table (service x HMO x tier).

        Args:
            markdown_content: Full Markdown file content
            category: Category name (dental, optometry, etc.)

        Returns:
            List of dictionaries with chunk text and metadata
        """
        chunks = []
        lines = markdown_content.split('\n')

        # Find table boundaries
        table_start = None
        table_end = None

        for i, line in enumerate(lines):
            if '|' in line:
                if table_start is None:
                    table_start = i
                table_end = i

        if table_start is None:
            return chunks

        # Extract table lines
        table_lines = lines[table_start:table_end + 1]

        # First line is header
        if len(table_lines) < 3:
            return chunks

        header_row = table_lines[0]
        headers = self.parse_table_headers(header_row)

        # Find which columns are HMOs (skip first column which is service name)
        hmo_columns = {}
        for i, header in enumerate(headers[1:], start=1):
            for hmo_he, hmo_en in self.hmo_he_to_en.items():
                if hmo_he in header or hmo_en in header:
                    hmo_columns[i] = hmo_en
                    break

        # Process data rows (skip header and separator)
        data_rows = table_lines[2:]

        for row in data_rows:
            if not row.strip() or '---' in row:
                continue

            # Split cells (handle empty cells properly)
            raw_cells = row.split('|')
            # First and last are empty from split
            cells = [cell.strip() for cell in raw_cells[1:-1]]

            if len(cells) < len(headers):
                # Row has fewer cells than headers (missing data)
                # Pad with empty strings
                cells.extend([''] * (len(headers) - len(cells)))

            # First cell is service name
            service_name = cells[0]
            if not service_name:
                continue

            # Normalize service name for metadata
            service_id = re.sub(r'[^\w\s]', '', service_name)
            service_id = service_id.replace(' ', '_').lower()

            # Process each HMO column
            for col_idx, hmo_en in hmo_columns.items():
                if col_idx >= len(cells):
                    continue

                cell_content = cells[col_idx]

                if not cell_content or cell_content == '':
                    # Empty cell - create a single chunk indicating no coverage
                    chunk_text = f"שירות: {service_name}\nקופת חולים: {hmo_en}\nסטטוס: אין כיסוי"

                    chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            "type": "benefit",
                            "category": category,
                            "service": service_id,
                            "hmo": hmo_en,
                            "tier": "none"
                        }
                    })
                else:
                    # Parse tier benefits
                    tier_benefits = self.parse_tier_benefits(cell_content)

                    for tier_en, benefits in tier_benefits:
                        chunk_text = f"שירות: {service_name}\nקופת חולים: {hmo_en}\nמסלול: {tier_en}\nהטבות: {benefits}"

                        chunks.append({
                            "text": chunk_text,
                            "metadata": {
                                "type": "benefit",
                                "category": category,
                                "service": service_id,
                                "hmo": hmo_en,
                                "tier": tier_en
                            }
                        })

        return chunks

    def extract_contact_chunks(self, markdown_content: str, category: str) -> List[Dict[str, Any]]:
        """
        Extract contact chunks (phone numbers and URLs per HMO).

        Args:
            markdown_content: Full Markdown file content
            category: Category name (dental, optometry, etc.)

        Returns:
            List of dictionaries with chunk text and metadata
        """
        chunks = []

        # Find sections with contact information
        # Look for H3 headings that mention phone or contact
        contact_sections = re.findall(
            r'###[^\n]*(?:טלפון|פרטים|contact|phone)[^\n]*\n+(.*?)(?=\n###|\n##|$)',
            markdown_content,
            re.DOTALL | re.IGNORECASE
        )

        # Combine all contact sections
        contact_text = '\n'.join(contact_sections)

        # Extract contact info for each HMO
        for hmo_he, hmo_en in self.hmo_he_to_en.items():
            # Find all lines mentioning this HMO
            hmo_info = []

            for line in contact_text.split('\n'):
                line = line.strip()
                if hmo_he in line or hmo_en.lower() in line.lower():
                    # Clean up list markers
                    line = re.sub(r'^[-*]\s*', '', line)
                    hmo_info.append(line)

            if hmo_info:
                chunk_text = f"קופת חולים: {hmo_en}\nקטגוריה: {category}\nפרטי התקשרות:\n" + "\n".join(hmo_info)

                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "type": "contact",
                        "category": category,
                        "hmo": hmo_en,
                        "tier": "all"
                    }
                })

        return chunks

    def process_markdown_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Process a single Markdown file and extract all chunks.

        Args:
            file_path: Path to Markdown file

        Returns:
            List of all chunks (context + benefits + contacts)
        """
        # Infer category from filename
        category = self.infer_category(file_path)

        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        all_chunks = []

        # Extract context chunk
        context_chunk = self.extract_context_chunk(content, category)
        all_chunks.append(context_chunk)

        # Extract benefit chunks
        benefit_chunks = self.extract_benefit_chunks(content, category)
        all_chunks.extend(benefit_chunks)

        # Extract contact chunks
        contact_chunks = self.extract_contact_chunks(content, category)
        all_chunks.extend(contact_chunks)

        return all_chunks

    def ingest_directory(self, input_dir: str = "data/knowledge_base_markdown") -> None:
        """
        Process all Markdown files and ingest into ChromaDB.

        Args:
            input_dir: Directory containing Markdown files
        """
        input_path = Path(input_dir)

        if not input_path.exists():
            print(f"ERROR: Directory {input_dir} does not exist")
            return

        # Find all Markdown files
        md_files = sorted(list(input_path.glob("*.md")))

        if not md_files:
            print(f"ERROR: No Markdown files found in {input_dir}")
            return

        print(f"Found {len(md_files)} Markdown files")
        print(f"Processing chunks...\n")

        all_chunks = []
        chunk_counts = {"context": 0, "benefit": 0, "contact": 0}

        # Process each file
        for md_file in md_files:
            print(f"Processing: {md_file.name}")
            chunks = self.process_markdown_file(md_file)

            # Count by type
            for chunk in chunks:
                chunk_type = chunk["metadata"]["type"]
                chunk_counts[chunk_type] += 1

            all_chunks.extend(chunks)
            print(f"  Extracted {len(chunks)} chunks")

        print(f"\nTotal chunks: {len(all_chunks)}")
        print(f"  Context chunks: {chunk_counts['context']}")
        print(f"  Benefit chunks: {chunk_counts['benefit']}")
        print(f"  Contact chunks: {chunk_counts['contact']}")

        # Embed chunks in batches
        print(f"\nEmbedding chunks with ADA-002 (batched)...")

        documents = []
        embeddings = []
        metadatas = []
        ids = []

        batch_size = 20
        total_embedded = 0

        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            batch_texts = [chunk["text"] for chunk in batch]

            # Embed batch
            batch_embeddings = self.embed_text_batch(batch_texts)

            # Collect results
            for j, chunk in enumerate(batch):
                documents.append(chunk["text"])
                embeddings.append(batch_embeddings[j])
                metadatas.append(chunk["metadata"])
                ids.append(f"chunk_{total_embedded + j}")

            total_embedded += len(batch)
            print(f"  Embedded {total_embedded}/{len(all_chunks)} chunks")

        # Store in ChromaDB
        print(f"\nStoring in ChromaDB...")
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        print(f"SUCCESS: Ingested {len(all_chunks)} chunks into ChromaDB")
        print(f"Vector database saved to: ./vector_db/")

        # Verify
        print(f"\nVerification:")
        print(f"  Collection count: {self.collection.count()}")


def main():
    """Main entry point for the script."""
    print("=" * 60)
    print("Medical Services Knowledge Base Ingestion")
    print("=" * 60)
    print()

    ingestion = KnowledgeBaseIngestion()
    ingestion.ingest_directory()

    print()
    print("Ingestion complete!")


if __name__ == "__main__":
    main()
