# Employment Rate Analysis Project

This project involves analyzing and modeling employment rates using machine learning models, including Random Forest and Gradient Boosting, with datasets from nationwide employment data.

## Folder Structure

- **`Data/`**
  - **`processed/`**: Contains preprocessed datasets for analysis and model training.
    - Example: `data_binary_clean.csv`
  - **`raw/`**: Contains raw datasets with original data.
    - Example: `อัตราการมีงานทำต่อประชากรวัยแรงงาน.csv`

- **`Results/`**
  - **`figures/`**: Stores CSV files with evaluation metrics for trained models.
    - Example: `Gradient_Boosting_Regression_model_metrics.csv`
  - **`output/`**: Stores visualization outputs, such as plots of predicted vs actual values.
    - Example: `Gradient_Boosting_Regression_expected_vs_predicted.png`

- **`src/`**
  - **`analysis/`**: Scripts for data analysis and model evaluation.
  - **`main/`**: Main scripts for running models and generating outputs.
  - **`preprocess/`**: Scripts for data cleaning and preprocessing.

- **`requirements.txt`**
  - A file listing all the required Python libraries to run the project.

## Installation

To run this project, ensure you have Python 3.7 or higher installed. Install the required libraries by running:

```bash
pip install -r requirements.txt
```

## Datasets

1. **Raw Data**:
   - File: `อัตราการมีงานทำต่อประชากรวัยแรงงาน.csv`
   - Contains detailed employment data by region, area, and education level.

2. **Processed Data**:
   - File: `data_binary_clean.csv`
   - Encoded dataset ready for machine learning.

## Outputs

1. **Metrics**:
   - Metrics for each model are saved as CSV files in `results/figures/`.

2. **Graphs**:
   - Plots of "Expected vs Predicted" values are saved as PNG files in `results/output/`.
