#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
# main.py
import argparse
import importlib
import os
import sys
import argcomplete

TASKS = [
    "regression",
    "activation_study",
    "regularization_study",
    "classification",
    "cross_validation",
    "torch_comparison",
]
PACKAGES = {"torch_comparison": "external_model"}  # tasks that are not in experiments/
RUN_ALL = {"machine.learn"}  # aliases that run every task


def build_parser():
    parser = argparse.ArgumentParser(description="Run experiments.")
    tasks_arg = parser.add_argument(
        "tasks",
        nargs="*",
        metavar="TASK",
        help=f"tasks to run (default: all). Choices: {', '.join(TASKS)}. "
        "Add 'plot' to only plot from saved results. "
        "'machine.learn' runs everything.",
    )
    tasks_arg.completer = argcomplete.ChoicesCompleter(
        [*TASKS, "plot", "machine.learn"]
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    argcomplete.autocomplete(parser)  # must come before the heavy imports
    args = parser.parse_args()

    plot_only = "plot" in args.tasks
    run_all = any(t in RUN_ALL for t in args.tasks)
    selected = [t for t in args.tasks if t != "plot" and t not in RUN_ALL]

    unknown = [t for t in selected if t not in TASKS]
    if unknown:
        parser.error(
            f"unknown task(s): {', '.join(unknown)}. Available: {', '.join(TASKS)}"
        )

    if run_all:
        print("Machine is learning...")
        selected = TASKS
    elif not selected:
        print("No tasks specified. Running all tasks.")
        selected = TASKS

    # heavy stuff only after parsing
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    import jax

    jax.config.update("jax_enable_x64", True)
    from config import load_config

    cfg = load_config()

    for task in selected:
        module = importlib.import_module(f"{PACKAGES.get(task, 'experiments')}.{task}")
        (module.plot if plot_only else module.run)(cfg)
