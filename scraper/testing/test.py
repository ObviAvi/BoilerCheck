import json
import sys
from pathlib import Path

# ── Load documents from JSON files ───────────────────────────────────────────

DEFAULT_DOC1 = Path(__file__).parent / "../data/test.json"
DEFAULT_DOC2 = Path(__file__).parent / "../data/purdue_policies.json"


def load_json(filepath: Path) -> dict:
    path = Path(filepath)
    if not path.exists():
        print(f"ERROR: File not found: {path.resolve()}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if len(sys.argv) == 3:
    doc1 = load_json(sys.argv[1])
    doc2 = load_json(sys.argv[2])
elif len(sys.argv) == 1:
    print(f"No args provided — using defaults:")
    print(f"  doc1 (to test)     : {DEFAULT_DOC1.resolve()}")
    print(f"  doc2 (answer key)  : {DEFAULT_DOC2.resolve()}")
    doc1 = load_json(DEFAULT_DOC1)
    doc2 = load_json(DEFAULT_DOC2)
else:
    print("Usage:")
    print("  python test.py                          # uses default paths")
    print("  python test.py <doc1.json> <doc2.json>  # custom paths")
    sys.exit(1)

# ── Helper ───────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Strip extra whitespace/newlines for fair text comparison."""
    return " ".join(text.split())


def run_tests():
    passed = 0
    failed = 0

    def check(test_name: str, actual, expected, note: str = ""):
        nonlocal passed, failed
        if actual == expected:
            print(f"  [PASS] {test_name}")
            passed += 1
        else:
            print(f"  [FAIL] {test_name}")
            if note:
                print(f"         Note    : {note}")
            print(f"         Expected: {repr(expected)}")
            print(f"         Got     : {repr(actual)}")
            failed += 1

    # ── Top-level field tests ────────────────────────────────────────────────
    print("\n=== Top-Level Fields ===")

    for field in ["document_id", "title", "domain", "url", "effective_date"]:
        check(
            f"{field} matches answer key",
            doc1.get(field, ""),
            doc2.get(field, "")
        )

    # ── Section presence tests ───────────────────────────────────────────────
    print("\n=== Section Presence ===")

    doc1_titles       = {s["section_title"] for s in doc1.get("sections", [])}
    doc2_titles       = {s["section_title"] for s in doc2.get("sections", [])}
    extra_in_doc1     = doc1_titles - doc2_titles
    missing_from_doc1 = doc2_titles - doc1_titles

    check("No extra sections in doc1 vs answer key",
        extra_in_doc1, set(),
        note=f"doc1 has sections not in answer key: {extra_in_doc1}")

    check("No sections missing from doc1 vs answer key",
        missing_from_doc1, set(),
        note=f"doc1 is missing sections: {missing_from_doc1}")

    # ── Section content tests ────────────────────────────────────────────────
    print("\n=== Section Content (normalized) ===")

    doc2_sections = {s["section_title"]: s["text"] for s in doc2.get("sections", [])}

    for section in doc1.get("sections", []):
        title = section["section_title"]
        if title not in doc2_sections:
            print(f"  [SKIP] '{title}' — not present in answer key, skipping content check")
            continue
        check(
            f"Section '{title}' text matches",
            normalize(section["text"]),
            normalize(doc2_sections[title])
        )

    # ── Summary ──────────────────────────────────────────────────────────────
    total = passed + failed
    print(f"\n{'='*45}")
    print(f"Results: {passed}/{total} passed, {failed}/{total} failed")
    print('='*45)


if __name__ == "__main__":
    run_tests()