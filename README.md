# SSOAP-RTA-Optimization-Tool
This is a repo where I try to apply optimization/regression techniques to the RDII analysis tool for SSOAP and SWMM 5.2
# Setting Up Optimization

## 1. Installation

### 1.1 Requirements

To run the calibration program, make sure the following are installed:

- **Python**: Version 3.12.6 (used for development)
- **Packages** (listed in `requirements.txt`):
  - `pyswmm`
  - `numpy`
  - `sklearn`
  - `matplotlib`
  - `pandas`
  - `deap`

- **SWMM 5.2**: Download from [EPA's website](https://www.epa.gov/water-research/storm-water-management-model-swmm)
- An SWMM `.net` or `.inp` file you want to optimize
- **SSOAP**: Download from [EPA's website](https://www.epa.gov/water-research/sanitary-sewer-overflow-analysis-and-planning-ssoap-toolbox)
- Two `.csv` files with EVENTS and RDII flow exported from SSOAP
  
- ### 1.2 Demo Setup (7/1/25)
The demo has already performed the toolbox processing after installation. All you should need to do is press Run and wait. Please feel free to contact me with any issues. 

