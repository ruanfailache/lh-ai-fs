from pathlib import Path

DOCUMENTS_DIR = Path(__file__).parent / "documents"


def load_documents() -> dict[str, str]:
    """Load all documents from the documents directory."""
    documents = {}
    for file_path in DOCUMENTS_DIR.glob("*.txt"):
        documents[file_path.stem] = file_path.read_text()
    return documents
