from sqlalchemy import create_engine, text
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="Receive from exchange")
    parser.add_argument("--url", "-u", help="url to connect to sql database", required=True)
    args = parser.parse_args()
    print(args.url)

    engine = create_engine(args.url)
    with engine.connect() as conn:
        result = conn.execute(text("SHOW DATABASES;"))
        print("Available databases:")
        for row in result:
            print("-", row[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
