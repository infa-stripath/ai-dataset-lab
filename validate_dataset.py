import json
from pathlib import Path

import pandas as pd

DATASET = Path("output/evaluation_dataset.csv")
REPORT = Path("output/validation_report.json")

REQUIRED_COLUMNS = [
	"case_id",
	"input",
	"intent",
	"expected_behavior",
	"test_type",
	"risk_level",
	"source",
	"review_status",
	"generation_model",
	"dataset_version",
]


def main():
	if not DATASET.exists():
		raise FileNotFoundError("Run generate_dataset.py first.")

	df = pd.read_csv(DATASET)
	missing_columns = [
		col for col in REQUIRED_COLUMNS
		if col not in df.columns
	]

	if missing_columns:
		raise ValueError(f"Missing columns: {missing_columns}")

	required_data = [
		"case_id",
		"input",
		"intent",
		"expected_behavior",
		"test_type",
	]
	blank_fields = {}

	for col in required_data:
		values = df[col].fillna("").astype(str).str.strip()
		blank_fields[col] = int(values.eq("").sum())

	report = {
		"total_cases": int(len(df)),
		"unique_case_ids": int(df["case_id"].nunique()),
		"duplicate_case_ids": int(
			df["case_id"].duplicated().sum()
		),
		"duplicate_inputs": int(
			df["input"].fillna("").str.strip()
			.str.casefold().duplicated().sum()
		),
		"blank_fields": blank_fields,
		"test_type_distribution": {
			str(key): int(value)
			for key, value in df["test_type"].value_counts().items()
		},
		"risk_distribution": {
			str(key): int(value)
			for key, value in df["risk_level"].value_counts().items()
		},
		"review_status_distribution": {
			str(key): int(value)
			for key, value in df["review_status"].value_counts().items()
		},
		"minimum_100_cases_met": bool(len(df) >= 100),
	}

	REPORT.parent.mkdir(exist_ok=True)
	REPORT.write_text(
		json.dumps(report, indent=2),
		encoding="utf-8",
	)

	print(json.dumps(report, indent=2))

	has_errors = (
		report["duplicate_case_ids"] > 0
		or any(blank_fields.values())
		or not report["minimum_100_cases_met"]
	)

	if has_errors:
		raise SystemExit(
			"Validation failed. Fix the errors before publishing."
		)

	print("\nStructural validation passed.")
	print("Review semantic correctness before approval.")


if __name__ == "__main__":
	main()
