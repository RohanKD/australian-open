import argparse
import random

import pandas as pd

from australian_open_model import (
    RANDOM_SEED,
    compute_head_to_head,
    evaluate_model,
    load_and_clean_data,
    train_model,
)


def build_symmetric_dataset_with_head_to_head(
    data: pd.DataFrame, head_to_head: dict
) -> pd.DataFrame:
    rng = random.Random(RANDOM_SEED)
    rows = []

    for _, row in data.iterrows():
        winner = row["winner_name"]
        loser = row["loser_name"]
        winner_rank = row["winner_rank"]
        loser_rank = row["loser_rank"]
        winner_age = row["winner_age"]
        loser_age = row["loser_age"]
        year = row["year"]

        if rng.random() < 0.5:
            player1 = winner
            player2 = loser
            p1_rank = winner_rank
            p2_rank = loser_rank
            p1_age = winner_age
            p2_age = loser_age
            target = 1
        else:
            player1 = loser
            player2 = winner
            p1_rank = loser_rank
            p2_rank = winner_rank
            p1_age = loser_age
            p2_age = winner_age
            target = 0

        results = head_to_head.get((player1, player2), [])
        head_to_head_pct = 0.5 if not results else sum(results) / len(results)

        rows.append(
            {
                "player1": player1,
                "player2": player2,
                "p1_rank": p1_rank,
                "p2_rank": p2_rank,
                "p1_age": p1_age,
                "p2_age": p2_age,
                "year": year,
                "rank_diff": p1_rank - p2_rank,
                "age_diff": p1_age - p2_age,
                "head_to_head": head_to_head_pct,
                "target": target,
            }
        )

    return pd.DataFrame(rows)


def backtest_years(
    data: pd.DataFrame, train_end_year: int, test_years: list[int]
) -> None:
    train_data = data[data["year"] <= train_end_year].copy()
    test_data = data[data["year"].isin(test_years)].copy()

    head_to_head = compute_head_to_head(train_data)
    train_df = build_symmetric_dataset_with_head_to_head(train_data, head_to_head)
    test_df = build_symmetric_dataset_with_head_to_head(test_data, head_to_head)

    model = train_model(train_df)

    if test_df.empty:
        print("No test data available for the specified years.")
        return

    overall_accuracy = evaluate_model(model, test_df)
    print(
        f"Backtest accuracy ({min(test_years)}-{max(test_years)}): "
        f"{overall_accuracy:.3f}"
    )

    for year in test_years:
        year_df = test_df[test_df["year"] == year]
        if year_df.empty:
            print(f"No matches found for {year}.")
            continue
        year_accuracy = evaluate_model(model, year_df)
        print(f"Backtest accuracy ({year}): {year_accuracy:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest an Australian Open model on 2024 and 2025."
    )
    parser.add_argument(
        "--csv",
        default="australian_open_history.csv",
        help="Path to the historical Australian Open CSV file.",
    )
    parser.add_argument(
        "--train_end_year",
        type=int,
        default=2023,
        help="Last year to include in the training set.",
    )
    args = parser.parse_args()

    raw_data = load_and_clean_data(args.csv)
    backtest_years(raw_data, args.train_end_year, [2024, 2025])


if __name__ == "__main__":
    main()
