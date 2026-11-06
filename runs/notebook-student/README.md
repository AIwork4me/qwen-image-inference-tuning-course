# Student notebook analysis records / 学生 Notebook 记录

This directory is the explicit opt-in destination for exercises such as M7.
The notebooks NEVER write here during their default Run All path.

When a student enables `WRITE_STUDENT_RECORD=True`, the generated JSON
will be stored as `runs/notebook-student/<student-tag>.json`. It is an analysis
derived from the **already committed reference** unless a separate verified
physical benchmark is attached. It must not be presented as new GPU evidence.

Never overwrite the frozen `evidence/` tree.

中文：这里仅用于**显式授权**的学生 Notebook 分析记录。默认 Run All 不会写入文件。
在 M7 启用 `WRITE_STUDENT_RECORD=True` 后，记录保存在此处；引用数据的重算
**不等于**新完成的 W7900 真机评测。不要修改 `evidence/`。
