from pathlib import Path
import subprocess


SUPPORTED_DIRECT = {".txt", ".md"}
SUPPORTED_LITEPARSE = {
    ".pdf",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx",
    ".csv",
    ".png",
    ".jpg",
    ".jpeg",
}


def read_text_file(file_path: Path) -> dict:
    return {
        "source": file_path.name,
        "text": file_path.read_text(encoding="utf-8"),
        "metadata": {
            "source": file_path.name,
            "file_type": file_path.suffix.lower(),
            "parser": "direct_text",
        },
    }


def parse_with_liteparse(file_path: Path, parsed_dir: Path = Path("parsed")) -> dict:
    parsed_dir.mkdir(exist_ok=True)

    output_path = parsed_dir / f"{file_path.stem}.txt"

    subprocess.run(
        [
            "lit",
            "parse",
            str(file_path),
            "--format",
            "text",
            "-o",
            str(output_path),
        ],
        check=True,
    )

    return {
        "source": file_path.name,
        "text": output_path.read_text(encoding="utf-8"),
        "metadata": {
            "source": file_path.name,
            "parsed_file": output_path.name,
            "file_type": file_path.suffix.lower(),
            "parser": "liteparse",
        },
    }


def load_documents(docs_dir: str | Path) -> list[dict]:
    docs_path = Path(docs_dir)
    documents = []

    if not docs_path.exists():
        return documents

    for file_path in docs_path.iterdir():
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()

        if suffix in SUPPORTED_DIRECT:
            documents.append(read_text_file(file_path))
        elif suffix in SUPPORTED_LITEPARSE:
            documents.append(parse_with_liteparse(file_path))

    return documents
