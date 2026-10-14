import pandas as pd

df = pd.read_csv("output/evaluation_dataset.csv")

print("Total test cases:", len(df))

print("\nTest cases by category:")
print(df["test_type"].value_counts())

print("\nTest cases by intent:")
print(df.groupby("intent").size().sort_values(ascending=False))

print("\nTest cases by risk level:")
print(df["risk_level"].value_counts())

print("\nCases awaiting review:")
print(
    df[df["review_status"] == "pending_review"][
        ["case_id", "input", "test_type", "risk_level"]
    ].to_string(index=False)
)
