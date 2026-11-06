# Bilingual Coverage

**Contract (CI-enforced):** for each of the 9 modules, the English and
zh-CN notebooks carry **byte-identical executable code cells**
(`scripts/course-quality/check_notebooks.py`), identical numerical claims and
experiment-status enums, and localized learning prose (objectives, cautions,
explanations, exercises). Markdown lessons are separate bilingual pairs
(`check_bilingual.py`: commands and experiment IDs must appear verbatim in
both languages).

| Module | EN notebook | zh-CN notebook | Code parity | Localized prose |
|---|---|---|---|---|
| M0 | 00_experimental_discipline.ipynb | 00_experimental_discipline.zh-CN.ipynb | identical | yes |
| M1 | 01_baseline_and_uncertainty.ipynb | 01_baseline_and_uncertainty.zh-CN.ipynb | identical | yes |
| M2 | 02_profile_and_bottlenecks.ipynb | 02_profile_and_bottlenecks.zh-CN.ipynb | identical | yes |
| M3 | 03_less_computation.ipynb | 03_less_computation.zh-CN.ipynb | identical | yes |
| M4 | 04_faster_computation.ipynb | 04_faster_computation.zh-CN.ipynb | identical | yes |
| M5.0 | 05_cpu_offload_negative.ipynb | 05_cpu_offload_negative.zh-CN.ipynb | identical | yes |
| M5.1 | 06_openvino_and_fair_ab.ipynb | 06_openvino_and_fair_ab.zh-CN.ipynb | identical | yes |
| M6 | 07_combined_and_quality.ipynb | 07_combined_and_quality.zh-CN.ipynb | identical | yes |
| M7 | 08_capstone_and_pr.ipynb | 08_capstone_and_pr.zh-CN.ipynb | identical | yes |

Notebooks are generated from one spec
(`scripts/notebook-qualification/build_notebooks.py --check` fails CI if a
committed notebook is stale), which guarantees the parity by construction.

Both language tracks were **independently executed** in REFERENCE and
GPU_CLASSROOM modes (see FULL_18_NOTEBOOK_VALIDATION.md). Where a language
variant analyzes the same shared campaign record, the run id is displayed and
the record is never counted as two independent physical measurements.
