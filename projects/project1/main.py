# main.py

import sys
from config import load_config
from tests import run_cv_verification, run_grad_verification

from experiments import (
    run_ols,
    run_ridge,
    run_bootstrap,
    run_cv,
    run_gd,
    run_adaptive_gd,
    run_lasso,
    run_sgd,
    run_model_selection,
)

tasks = {
    "ols": run_ols,
    "ridge": run_ridge,
    "bootstrap": run_bootstrap,
    "cv": run_cv,
    "gd": run_gd,
    "adaptive_gd": run_adaptive_gd,
    "verify_cv": run_cv_verification,
    "verify_gradients": run_grad_verification,
    "lasso": run_lasso,
    "sgd": run_sgd,
    "model_selection": run_model_selection,
}

if __name__ == "__main__":
    cfg = load_config()  # Load configuration from config.toml

    selected_tasks = sys.argv[
        1:
    ]  # get task from cmd line, ie. python main.py ols ridge

    if not selected_tasks:
        print("No tasks specified. Running all tasks.")
        selected_tasks = tasks.keys()  # run all tasks

    for task in selected_tasks:
        if task not in tasks:
            raise ValueError(
                f"Unknown task '{task}'. Available tasks: {', '.join(tasks)}"
            )

        tasks[task](cfg)
