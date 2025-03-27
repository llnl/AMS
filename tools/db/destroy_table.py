from sqlalchemy import create_engine, text
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="Receive from exchange")
    parser.add_argument("--url", "-u", help="url to connect to sql database", required=True)
    args = parser.parse_args()

    engine = create_engine(args.url)
    with engine.connect() as conn:
        conn.execute(text("DROP DATABASE IF EXISTS amsTEST"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
