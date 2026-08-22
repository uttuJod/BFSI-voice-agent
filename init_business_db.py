from pathlib import Path

from business import BusinessDatabase


ROOT = Path(
    __file__
).resolve().parent


DB_PATH = (
    ROOT
    / "data"
    / "business.db"
)


def main():

    db = BusinessDatabase(
        DB_PATH
    )

    db.initialize()

    db.seed_demo_data()

    print(
        "Business database initialized."
    )

    print(
        f"Path: {DB_PATH}"
    )

    print()
    print(
        "Demo customer:"
    )

    print(
        "  customer_id: CUST-1001"
    )

    print(
        "  account_id:  ACC-1001"
    )


if __name__ == "__main__":
    main()