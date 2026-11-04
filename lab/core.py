"""Shared benchmark core: protocol planning, statistics, schema validation.

Frozen protocol (identical to the parent project so numbers stay
comparable): 3 prompts (zh scene / en scene / zh typography poster) x
seeds [42, 1337, 20261030], euler/simple, cfg 1.0, cold/warmup/measured
discipline with cache-defeating seeds.
"""

import json
from pathlib import Path

from jsonschema import validate as _validate

REPO = Path(__file__).resolve().parents[1]
PROMPTS_PATH = Path(__file__).resolve().parent / "prompts.json"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema" / "result.schema.json"


def plan_runs(
    prompt_ids: list[str],
    resolutions: list[tuple[int, int]],
    seeds: list[int],
    warmup: int = 1,
    measured: int = 3,
) -> list[dict]:
    """Return ordered run specs: one cold first, then per config warmup+measures.

    Seed discipline (input-hash cache defeat):
    - cold uses seeds[0] + 1_000_000
    - warmup uses seeds[0] + 2_000_000 (reusing the cold seed would make the
      warmup an input-identical node-cache hit instead of an executing run)
    - measured runs cycle the frozen seeds
    """
    cold_seed = seeds[0] + 1_000_000
    warmup_seed = seeds[0] + 2_000_000
    specs: list[dict] = []
    first = True
    for prompt_id in prompt_ids:
        for (w, h) in resolutions:
            if first:
                specs.append(
                    {"kind": "cold", "prompt_id": prompt_id, "width": w,
                     "height": h, "seed": cold_seed, "run_idx": 0}
                )
                first = False
            for _ in range(warmup):
                specs.append(
                    {"kind": "warmup", "prompt_id": prompt_id, "width": w,
                     "height": h, "seed": warmup_seed, "run_idx": 0}
                )
            for i in range(measured):
                specs.append(
                    {"kind": "measure", "prompt_id": prompt_id, "width": w,
                     "height": h, "seed": seeds[i % len(seeds)], "run_idx": i}
                )
    return specs


def steady_step_rate(step_times: list[float]) -> float | None:
    """Median per-step wall time over steady steps (index 1+).

    step_times_s[0] carries text-encode + weight-restore costs on top of
    step 1's compute, so it is excluded.
    """
    steady = step_times[1:]
    if not steady:
        return None
    import statistics

    return statistics.median(steady)


def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def validate_record(record: dict) -> None:
    """Validate a result record against the schema; raises ValidationError."""
    _validate(record, load_schema())
