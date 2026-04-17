import pandas as pd


def season_train_test_split(df: pd.DataFrame, test_season: int):
    train_df = df[df["season"] < test_season].copy()
    test_df = df[df["season"] == test_season].copy()
    return train_df, test_df


def walk_forward_splits(df: pd.DataFrame, min_train_matches: int = 100):
    df = df.sort_values("date").reset_index(drop=True)
    splits = []

    for i in range(min_train_matches, len(df)):
        train_idx = list(range(i))
        test_idx = [i]
        splits.append((train_idx, test_idx))

    return splits