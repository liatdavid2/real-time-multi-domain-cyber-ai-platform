import json
from pathlib import Path

RULES_DIR = Path(__file__).parent / "rules"


def load_all_rules():
    rules = []

    for file in RULES_DIR.glob("*.json"):
        with open(file, "r") as f:
            rule = json.load(f)

            if rule.get("enabled", True):
                rules.append(rule)

    return rules