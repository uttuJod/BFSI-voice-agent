from __future__ import annotations

from pathlib import Path

from business import (
    BusinessDatabase,
    VerificationRegistry,
)


def main() -> None:
    db_path = Path(
        "results/verification_speech_parser_test.sqlite3"
    )

    if db_path.exists():
        db_path.unlink()

    db = BusinessDatabase(db_path)
    db.initialize()
    db.seed_demo_data()

    registry = VerificationRegistry(db)

    cases = [
        ("0641", "0641"),
        ("0 6 4 1", "0641"),
        ("Zero six four one", "0641"),
        ("zero, six, four, one.", "0641"),
        ("oh six four one", "0641"),
        ("Zero six", None),
    ]

    for text, expected in cases:
        parsed = registry._extract_four_digits(text)
        print(repr(text), "->", parsed)
        assert parsed == expected

    print()
    print("VERIFICATION SPEECH PARSER: PASS")


if __name__ == "__main__":
    main()
