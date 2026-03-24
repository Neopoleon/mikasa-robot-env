"""Check success ratio for all NPZ files across the three unbatched tasks."""
import os
import numpy as np

base_dir = "data/MIKASA-Robo/unbatched"
tasks = ["RememberColor3-v0", "InterceptMedium-v0", "ShellGameTouch-v0"]

for task in tasks:
    task_dir = os.path.join(base_dir, task)
    npz_files = sorted([f for f in os.listdir(task_dir) if f.endswith(".npz")])
    total = len(npz_files)
    successes = 0
    for fname in npz_files:
        d = np.load(os.path.join(task_dir, fname))
        if d["success"].any():
            successes += 1
    ratio = successes / total if total > 0 else 0
    print(f"{task}: {successes}/{total} success ({ratio:.2%})")
