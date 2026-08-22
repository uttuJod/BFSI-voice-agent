from __future__ import annotations

from pathlib import Path

from business import BusinessDatabase, VerificationRegistry


def main() -> None:
    db_path = Path("results/verification_currency_artifact_test.sqlite3")

    if db_path.exists():
        db_path.unlink()

    db = BusinessDatabase(db_path)
    db.initialize()
    db.seed_demo_data()

    registry = VerificationRegistry(db)

    cases = [
        ("0641", "0641"),
        ("Zero six four one", "0641"),
        ("$64.01", "0641"),
        ("64.01", "0641"),
        ("$64.01.", "0641"),
        ("Zero six", None),
    ]

    for source, expected in cases:
        parsed = registry._extract_four_digits(source)
        print(repr(source), "->", parsed)
        assert parsed == expected

    print()
    print("VERIFICATION CURRENCY ARTIFACT NORMALIZATION: PASS")


if __name__ == "__main__":
    main()
