# Contributing

This project uses the [Conventional Branch](https://conventionalbranch.org/) naming. `main` is protected, so all changes must be made in a branch and merged. 

## Project structure & naming

- `src/` — reusable Python functions (data loading, models, plotting). Logic lives
  here, not in notebooks.
- `notebooks/` — one notebook per exercise/experiment, imports from `src/`.
  Name as `weekNN_topic.ipynb`, e.g. `week34_gradient_descent.ipynb`.
- `report/` — report source files (Markdown/LaTeX), one file per section:
  `01_introduction.md`, `02_methods.md`, etc.
- `data/` — datasets. Not committed if large; add a `.gitignore` entry and a short
  `data/README.md` explaining how to obtain it.
- File and folder names: lowercase, hyphens or underscores instead of spaces
  (`ridge_regression.py`, not `Ridge Regression.py`).

## Code style

Follow [PEP 8](https://peps.python.org/pep-0008/) naming:

| Kind                        | Style             | Example                          |
|-----------------------------|-------------------|----------------------------------|
| Variables, functions        | `snake_case`      | `learning_rate`, `make_output_dirs` |
| Classes                     | `PascalCase`      | `BinaryCrossEntropy`, `LeakyReLU` |
| Module-level constants      | `UPPER_CASE`      | `OUTPUT_DIR`, `PROJECT_ROOT`     |
| Internal (module-only) names | leading `_`       | `_clip`, `_OUTPUT_DIR`           |

- **Math notation is allowed** where it matches the report: `X` for the design
  matrix, `y` for targets, `z` for pre-activations, `W`/`b` for weights and biases.
  Use the same symbol in code and report.
- **Descriptive names everywhere else**: `n_samples`, not `ns`; `y_pred`, not `yp`.
- **Docstrings** follow the [NumPy style](https://numpydoc.readthedocs.io/en/latest/format.html)
  for every public module, class and function. Write formulas in plain text
  (`sigma(z) = 1 / (1 + exp(-z))`), not `.. math::`, so they show on hover in the editor.
- **Base classes stay public** (`Cost`, `Activation`), since other code type-hints
  and subclasses them. Use a leading `_` only for helpers used inside one module.

## Branch naming

```
<type>/<description>
```

- Lowercase only, words separated by hyphens (no spaces, no underscores).
- No consecutive/leading/trailing hyphens.
- Keep the description short but specific.

| Type       | Use for                                              | Example                          |
|------------|-------------------------------------------------------|-----------------------------------|
| `feature/` | New functionality / analysis / exercises              | `feature/ridge-regression`       |
| `fix/`     | Bug fixes                                              | `fix/train-test-split-bug`       |
| `chore/`   | Non-code tasks: docs, dependencies, repo setup         | `chore/add-contributing-guide`   |


`main` itself never takes a prefix.

## Daily workflow

1. **Sync before branching:**
   ```bash
   git checkout main
   git pull
   ```
2. **Create a branch:**
   ```bash
   git checkout -b feature/your-thing
   ```
3. **Commit in small steps** 
   ```bash
   git add -p
   git commit -m "add ridge regression fit function"
   ```
4. **Push the branch** (never `main`):
   ```bash
   git push -u origin feature/your-thing
   ```
5. **Open a pull request** on GitHub as soon as there's something reviewable.
6. **One teammate gives it a quick look**
7. **Squash and merge**, then delete the branch.
8. **Everyone pulls `main`** before starting their next branch.

### If Git gets stuck because of Jupyter notebooks / `nbstripout`

If `git pull` or `git rebase` fails because of Jupyter notebook or `nbstripout` errors, and **all work you want to keep has already been pushed/merged**, you can reset your local `main` branch to exactly match the remote:

```bash
git checkout main
git fetch origin
git reset --hard origin/main
```

> **Warning:** `git reset --hard` permanently discards uncommitted changes and local commits that are not present on `origin/main`. Only use this if you are sure there is nothing local you need to keep.


## Avoiding merge conflicts

- **Split ownership by file.** 
- **`nbstripout` strips notebook outputs automatically on `git add`/`commit`.**
  Set it up once per clone. This works regardless of whether a project venv
  is active, so you won't need to remember to activate anything before, but installs globally.
  When committing:
```bash
  # Install pipx once (if you don't have it)
  sudo apt install pipx
  pipx ensurepath

  # open a NEW terminal window after this step
  
  # Install nbstripout globally via pipx
  pipx install nbstripout

  # Wire up the git filter for this repo
  nbstripout --install --attributes .gitattributes --python python3
```


- **Rebase early if `main` has moved on:**
  ```bash
  git fetch origin
  git rebase origin/main
  ```
- **Give a heads-up in the group chat**
- **Merge often, in small pieces** 

# Issues
We will not use GitHub issues, but rather a group chat.
