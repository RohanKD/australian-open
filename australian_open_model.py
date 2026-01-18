import argparse
import random
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


RANDOM_SEED = 42


def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    data = pd.read_csv(csv_path)
    data = data[data["tourney_name"] == "Australian Open"].copy()

    required_cols = [
        "winner_name",
        "loser_name",
        "winner_rank",
        "loser_rank",
        "winner_age",
        "loser_age",
        "year",
    ]
    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = data.dropna(
        subset=["winner_rank", "loser_rank", "winner_age", "loser_age", "year"]
    )
    data["winner_rank"] = data["winner_rank"].astype(float)
    data["loser_rank"] = data["loser_rank"].astype(float)
    data["winner_age"] = data["winner_age"].astype(float)
    data["loser_age"] = data["loser_age"].astype(float)
    data["year"] = data["year"].astype(int)
    return data


def compute_head_to_head(records: pd.DataFrame) -> dict:
    head_to_head = defaultdict(list)
    for _, row in records.iterrows():
        winner = row["winner_name"]
        loser = row["loser_name"]
        head_to_head[(winner, loser)].append(1)
        head_to_head[(loser, winner)].append(0)
    return head_to_head


def head_to_head_pct(head_to_head: dict, player1: str, player2: str) -> float:
    results = head_to_head.get((player1, player2), [])
    if not results:
        return 0.5
    return float(np.mean(results))


def build_symmetric_dataset(data: pd.DataFrame) -> pd.DataFrame:
    random.seed(RANDOM_SEED)
    head_to_head = compute_head_to_head(data)

    rows = []
    for _, row in data.iterrows():
        winner = row["winner_name"]
        loser = row["loser_name"]
        winner_rank = row["winner_rank"]
        loser_rank = row["loser_rank"]
        winner_age = row["winner_age"]
        loser_age = row["loser_age"]
        year = row["year"]

        if random.random() < 0.5:
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
                "head_to_head": head_to_head_pct(head_to_head, player1, player2),
                "target": target,
            }
        )

    return pd.DataFrame(rows)


def split_train_test(data: pd.DataFrame, train_end_year: int = 2023) -> tuple:
    train = data[data["year"] <= train_end_year].copy()
    test = data[data["year"] > train_end_year].copy()
    return train, test


def train_model(train_df: pd.DataFrame) -> XGBClassifier:
    features = train_df[["rank_diff", "age_diff", "head_to_head"]]
    target = train_df["target"]
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        eval_metric="logloss",
        n_jobs=4,
    )
    model.fit(features, target)
    return model


def evaluate_model(model: XGBClassifier, test_df: pd.DataFrame) -> float:
    features = test_df[["rank_diff", "age_diff", "head_to_head"]]
    target = test_df["target"]
    predictions = model.predict(features)
    return accuracy_score(target, predictions)


def predict_match(
    model: XGBClassifier,
    head_to_head: dict,
    player1_name: str,
    player2_name: str,
    player1_rank: float,
    player2_rank: float,
    player1_age: float | None = None,
    player2_age: float | None = None,
) -> None:
    if player1_age is None or player2_age is None:
        player1_age = 0.0
        player2_age = 0.0

    features = pd.DataFrame(
        [
            {
                "rank_diff": player1_rank - player2_rank,
                "age_diff": player1_age - player2_age,
                "head_to_head": head_to_head_pct(
                    head_to_head, player1_name, player2_name
                ),
            }
        ]
    )
    proba = model.predict_proba(features)[0][1]
    winner = player1_name if proba >= 0.5 else player2_name
    print(
        f"Predicted winner: {winner} (P1 win probability: {proba * 100:.2f}%)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an XGBoost model for Australian Open winner prediction."
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
    symmetric_data = build_symmetric_dataset(raw_data)
    train_df, test_df = split_train_test(symmetric_data, args.train_end_year)

    model = train_model(train_df)

    if test_df.empty:
        print("No test data available for the specified split.")
    else:
        accuracy = evaluate_model(model, test_df)
        print(f"Test accuracy: {accuracy:.3f}")

    head_to_head = compute_head_to_head(raw_data)
    predict_match(
        model,
        head_to_head,
        "Novak Djokovic",
        "Jannik Sinner",
        1,
        4,
        player1_age=36,
        player2_age=22,
    )


if __name__ == "__main__":
    main()
