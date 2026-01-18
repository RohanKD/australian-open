import argparse
import random

import matplotlib.pyplot as plt
import pandas as pd

from australian_open_model import (
    RANDOM_SEED,
    compute_head_to_head,
    evaluate_model,
    load_and_clean_data,
    train_model,
)


def build_symmetric_dataset_with_head_to_head(
    data: pd.DataFrame, head_to_head: dict, round_col: str
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
        round_value = row[round_col] if round_col in row else "All"

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
                "winner_name": winner,
                "loser_name": loser,
                "round": round_value,
            }
        )

    return pd.DataFrame(rows)


def visualize_results(results: pd.DataFrame, output_path: str) -> None:
    accuracy_by_round = (
        results.groupby("round")["correct_prediction"].mean().sort_index()
    )

    plt.figure(figsize=(10, 6))
    accuracy_by_round.plot(kind="bar", color="#1f77b4")
    plt.title("Prediction Accuracy by Round (Current Year)")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def predict_current_year(
    csv_path: str, output_csv: str, output_plot: str
) -> None:
    raw_data = load_and_clean_data(csv_path)
    current_year = int(raw_data["year"].max())

    train_data = raw_data[raw_data["year"] < current_year].copy()
    test_data = raw_data[raw_data["year"] == current_year].copy()

    if test_data.empty:
        print("No matches found for the current year.")
        return

    round_col = "round"
    if round_col not in raw_data.columns:
        round_col = "match_num" if "match_num" in raw_data.columns else "round"

    head_to_head = compute_head_to_head(train_data)
    train_df = build_symmetric_dataset_with_head_to_head(
        train_data, head_to_head, round_col
    )
    test_df = build_symmetric_dataset_with_head_to_head(
        test_data, head_to_head, round_col
    )

    model = train_model(train_df)
    features = test_df[["rank_diff", "age_diff", "head_to_head"]]
    probabilities = model.predict_proba(features)[:, 1]
    predictions = model.predict(features)

    results = test_df.copy()
    results["p1_win_probability"] = probabilities
    results["predicted_winner"] = results.apply(
        lambda row: row["player1"] if row["p1_win_probability"] >= 0.5 else row["player2"],
        axis=1,
    )
    results["actual_winner"] = results.apply(
        lambda row: row["player1"] if row["target"] == 1 else row["player2"], axis=1
    )
    results["correct_prediction"] = predictions == results["target"]

    accuracy = evaluate_model(model, test_df)
    print(f"Current year ({current_year}) accuracy: {accuracy:.3f}")

    results.to_csv(output_csv, index=False)
    if results["round"].nunique() > 1:
        visualize_results(results, output_plot)
        print(f"Saved visualization to {output_plot}")
    else:
        print("Skipping visualization; round data is not available.")
    print(f"Saved predictions to {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict and visualize results for the current year's Australian Open."
    )
    parser.add_argument(
        "--csv",
        default="australian_open_history.csv",
        help="Path to the historical Australian Open CSV file.",
    )
    parser.add_argument(
        "--output_csv",
        default="current_year_predictions.csv",
        help="Path to write predictions for the current year.",
    )
    parser.add_argument(
        "--output_plot",
        default="current_year_accuracy_by_round.png",
        help="Path to save the accuracy by round plot.",
    )
    args = parser.parse_args()

    predict_current_year(args.csv, args.output_csv, args.output_plot)


if __name__ == "__main__":
    main()
