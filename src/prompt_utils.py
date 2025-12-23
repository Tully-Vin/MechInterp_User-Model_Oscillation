from pathlib import Path
import yaml


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_system_prompt(prompts: dict) -> str:
    return prompts.get("system", "")


def build_base_prompt(prompts: dict, domain: str, condition: str) -> str:
    base = prompts["domains"][domain]["base"]
    if condition == "anchored":
        anchor = prompts["conditions"][condition].get("anchor_prefix", "")
        if anchor:
            return anchor.strip() + "\n\n" + base
    return base
