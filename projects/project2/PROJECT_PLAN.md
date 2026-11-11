# Project Plan
A description of the project, goals, timeline, and task assignments. Only meant to be used internally.

## Project Overview

**jk** 

**Seed**: 67 (HOOLY)

### Milestones and Primary Goals
- **Analytical calculations**
- **Core code** structure and config input
- **Optimization** with OpenMP
- **Method & Theory** Combined
- **Analysis and plotting** 
- **Report writing** and review
- **Final delivery** of code and report
  
### File Structure
```text
project2/
├── code/
│   ├── experiments/
│   │   └── common.py
│   ├── main.py
│   ├── plotting.py
│   └── test/
├── supplementary_figs.ipynb
├── project2.pdf
└── output/
    ├── task1/
    │   ├── figs/
    │   └── results/
    └── task2/
```
## Timeline
Project Start Date: 06.10.2026
Project End Date: 06.11.2026

**Week 41** *(Tuesday 06.10.2026 - Sunday 11.10.2026)*
- Finish draft of report plan
- Do weekly exercise
- Make draft of report in overleaf
- Begin week 42 weekly exercise

**Week 42** *(Monday 12.10.2026 - Sunday 18.10.2026)*
-
- Danish boat 17.10, very fun. Going to chill in Commodore lugar with Amund, looking forwards to it!

**Week 43** *(Monday 19.10.2026 - Sunday 25.10.2026)*
- Finished with all main code so only plotting tweaks and result tweaks should be changed. Changes should however be avoided.

**Week 44** *(Monday 26.10.2026 - Sunday 01.11.2026)*
- 

**Week 45** *(Monday 02.11.2026 - Friday 06.11.2026)*
- 
  
## Tasks and Assignments

### Main responsibilities
*Note*: This is just a preliminary distribution of tasks. Everyone is expected to keep track of all parts.

Things to pick:

- **Adrian:**
  - Plot responsible (along with Claude)
  - GD methods
  
- **Casper:** 
  - Analytical calculations
  - Write main body draft of report based on the spewing of Erica.

- **Erica:**  
  - Grammar police
  - Write draft and report
  - NN


- **Victor:**  
  - GitHub management
  - Code structure and distribution of tasks (objectify the code and tweak algorithms)
  - Optimization of code
  
- **TBD**
  - Optimization
---------------------------------------------------------------------------------

### Simulation Plan

### Task List
Put in *italics* if done and fill in name for main responsibles.

**Overview**
- **Weekly exercises**
  - *Week 41* (Casper & Erica)
  - Week 42
- Derivations of analytical cost functions (part a)
- Make skeleton code with input parameters
- Update requirements (seaborn)
- **Methods:**
  - Reuse regression code from p1
  - Implement NN with flexible num of layers and nodes in layers and optional L1- L2- norm.
  - Implement back-propagation algorithm to compute the gradient of NN
  - Reuse of implement of plain and stochastic GD. 
  - RMSprop and Adam
  - Log regression as base model
  - Scaling & train_test_split
  - MSE, accuracy, confusion matrix
- **Analysis**
  - Pros and cons of methods in p1
  - Explore impact of nodes per layer, activation func, including L1, L2-norm
  - Linear regression vs NN (pros and cons for NN)
  - Compute & interpret confusion matrix of best classification model
  - Pros and cons of different gradient methods and learning rates
  - Compare our NN with Sklearn
- **Report**
  - Make draft
  - Abstract
  - **Results**
    - NN
  - **Method & Theory**
    - NN
    - Tools
    - 
  - **Conclusion**
- **Optional**
  - Bias--Variance
  - Check against 2D runke function
  


## Notes
- Units:

## Important
- Use same data for OLS and NN comparison
- Cite where we found image data
- General plotting functions in a common plotting.py along helper functions. But, individual one-time use plotting functions respective experiments.
- Make seperate folder for each experiments output
- All data for plotting should be stored as txt or equivalent
- Difference between run code and plot code ```python3 main.py task plot/run_plot/run```
- Make classes and nice code
- make a ```/code```-folder for python-code used for reproducing code
- Solve the daily krillion.io