# tests/test_cross_validation.py

from experiments.cross_validation import get_cv_data
from analysis import print_cv_implementations_comparison

# ------------------------------------------------
# Task d:
# Verify our own k-fold cv implementation against
# Scikit-Learn implementation.
# -------------------------------------------------


def verify_cv(cfg: dict):
    """
    Verify our k-fold cross-validation implementation against Scikit-Learn.
    The resulting MSE values are printed for comparison.

    Parameters
    ----------
    cfg : dict      Configuration dictionary containing the verification,
                    regression, and data-generation settings.
    """
    cv_cfg = cfg["cv"]
    x, y = get_cv_data(cfg)
    seed = cfg["data"]["seed"]

    print_cv_implementations_comparison(
        x,
        y,
        degree=cv_cfg["verify_degree"],
        method="ols",
        k=cv_cfg["verify_k"],
        random_state=seed,
    )
    print_cv_implementations_comparison(
        x,
        y,
        degree=cv_cfg["verify_degree"],
        method="ridge",
        k=cv_cfg["verify_k"],
        random_state=seed,
        lmbda=cv_cfg["ridge_lambda"],
    )


# usikker på denne her:
def run_cv_verification(cfg: dict):
    """
    Run the verification of the CV implemnetations (part of task d).

    Parameters
    ----------
    cfg : dict      Configuration dictionary containing the Ridge regression,
                    data-generation, and plotting settings.
    """
    verify_cv(cfg)
