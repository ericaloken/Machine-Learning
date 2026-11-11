# FYS-STK4155 - Applied Data Analysis and Machine Learning

*This repository contains the work of Victor Berge Johansen, Erica Løken, Casper Johnsen and Adrian Gjersvik Gundersen for the course FYS-STK4155 at the University of Oslo.*

## Contents

- [Projects](./projects)
  - [Project 1](./projects/project1): Regression analysis and resampling on the Runge function (OLS, Ridge and Lasso, gradient descent and SGD, bootstrap, cross-validation)
  
- [Weekly Exercises](./weekly_Exercises)

---

## How to Use

Each project has it's own `README.md` for usage.

## Dependencies
Each project has dependencies listed in their respective `README.md` files.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
## Nvidia GPU suport for Jax
```bash
python -m pip install --upgrade "jax[cuda13]"
```

## Adding a new package

```bash
pip install <package-name>
pip freeze > requirements.txt
```
