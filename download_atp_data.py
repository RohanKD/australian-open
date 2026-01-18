import argparse
from pathlib import Path

import pandas as pd
import requests


BASE_URL = (
    "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/"
    "atp_matches_{year}.csv"
)


def download_file(url: str, destination: Path) -> None:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    destination.write_bytes(response.content)


def download_atp_data(start_year: int, end_year: int, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded_files = []

    for year in range(start_year, end_year + 1):
        url = BASE_URL.format(year=year)
        destination = output_dir / f"atp_matches_{year}.csv"
        if destination.exists():
            downloaded_files.append(destination)
            continue
        download_file(url, destination)
        downloaded_files.append(destination)

    return downloaded_files


def combine_csvs(files: list[Path], output_path: Path) -> None:
    frames = []
    for file_path in files:
        frames.append(pd.read_csv(file_path))
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download ATP match data and create australian_open_history.csv."
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2005,
        help="First year of ATP data to download.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2025,
        help="Last year of ATP data to download.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Directory to store downloaded ATP files.",
    )
    parser.add_argument(
        "--output-csv",
        default="australian_open_history.csv",
        help="Combined CSV output path.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_csv = Path(args.output_csv)

    files = download_atp_data(args.start_year, args.end_year, output_dir)
    combine_csvs(files, output_csv)
    print(f"Downloaded {len(files)} files into {output_dir}")
    print(f"Wrote combined CSV to {output_csv}")


if __name__ == "__main__":
    main()
