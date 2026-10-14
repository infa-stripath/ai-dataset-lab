import os
import json
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

TEST_TYPES = [
	"paraphrase",
	"ambiguous",
	"incomplete",
	"edge_case",
	"negative",
	"adversarial",
	"typo",
	"context_conflict",
]


def generate_cases(seed):
	prompt = f"""
You are an enterprise AI quality engineer.

Generate exactly 10 synthetic test cases for this seed:

{json.dumps(seed)}

Create diverse cases covering:

{json.dumps(TEST_TYPES)}

Requirements:

- Preserve the business intent where appropriate.
- Generate realistic and distinct inputs.
- Include incomplete, ambiguous, negative, and adversarial cases.
- Never provide actual secrets or personal data.
- Define safe and testable expected behavior.
- Do not invent company policies.
- Set risk_level to low, medium, or high.
- Return a JSON object containing a "cases" array.

Each case must contain:

input, intent, expected_behavior, test_type, risk_level.

Return only valid JSON.
"""

	response = client.responses.create(
		model=MODEL,
		input=prompt,
		text={
			"format": {
				"type": "json_object"
			}
		},
	)

	result = json.loads(response.output_text)

	if not isinstance(result.get("cases"), list):
		raise ValueError("Invalid response: missing cases array")

	return result["cases"]


def main():
	seeds = pd.read_csv("seed_cases.csv").fillna("")
	records = []

	# Preserve original seed cases.
	for seed in seeds.to_dict(orient="records"):
		records.append({
			"case_id": seed["case_id"],
			"input": seed["input"],
			"intent": seed["intent"],
			"expected_behavior": seed["expected_behavior"],
			"test_type": "seed",
			"risk_level": "medium",
			"source": "human_authored",
			"review_status": "approved_seed",
			"generation_model": "not_applicable",
			"dataset_version": "1.0.0",
		})

	# Generate 10 cases for each seed.
	for index, seed in enumerate(
		seeds.to_dict(orient="records"), start=1
	):
		print(f"Processing seed {index}/{len(seeds)}")

		try:
			generated_cases = generate_cases(seed)
		except Exception as exc:
			print(f"Generation failed for {seed['case_id']}: {exc}")
			continue

		for case in generated_cases:
			if not isinstance(case, dict):
				continue

			required = {
				"input",
				"intent",
				"expected_behavior",
				"test_type",
				"risk_level",
			}

			if not required.issubset(case):
				continue

			test_type = str(case["test_type"]).lower().strip()
			risk_level = str(case["risk_level"]).lower().strip()
			input_text = str(case["input"]).strip()
			expected = str(case["expected_behavior"]).strip()

			if test_type not in TEST_TYPES:
				continue

			if risk_level not in {"low", "medium", "high"}:
				continue

			if not input_text or not expected:
				continue

			records.append({
				"case_id": "",
				"input": input_text,
				"intent": str(case["intent"]).strip(),
				"expected_behavior": expected,
				"test_type": test_type,
				"risk_level": risk_level,
				"source": f"openai_generated_from_{seed['case_id']}",
				"review_status": "pending_review",
				"generation_model": MODEL,
				"dataset_version": "1.0.0",
			})

		time.sleep(0.5)

	df = pd.DataFrame(records)

	# Remove exact duplicate inputs.
	df["_normalized"] = (
		df["input"].astype(str)
		.str.strip()
		.str.casefold()
		.str.replace(r"\s+", " ", regex=True)
	)

	df = df.drop_duplicates("_normalized").drop(
		columns=["_normalized"]
	)

	# Assign unique identifiers.
	df["case_id"] = [
		f"TC-{i:04d}" for i in range(1, len(df) + 1)
	]

	df.to_csv(
		OUTPUT_DIR / "evaluation_dataset.csv",
		index=False,
	)

	df.to_json(
		OUTPUT_DIR / "evaluation_dataset.jsonl",
		orient="records",
		lines=True,
		force_ascii=False,
	)

	print("\nGeneration completed.")
	print("Total cases:", len(df))
	print("\nCategory distribution:")
	print(df["test_type"].value_counts())

	if len(df) < 100:
		print("WARNING: Fewer than 100 cases generated.")

	print("\nDataset saved in the output directory.")


if __name__ == "__main__":
	main()
