# Project Workflow with GitHub Actions

This project is set up to automatically run a series of tasks using GitHub Actions whenever changes are pushed to the repository. The workflow includes testing CSV columns, data cleaning, model training, and selecting the best model based on mean absolute error.

## Workflow Overview

After every push to the repository, the following steps are executed as part of the workflow:

1. **Testing CSV Columns** (`pytest_csv_columns.py`):
   - The `pytest_csv_columns.py` script is executed to test whether a list of required columns is available in a CSV file.
   - The script checks if the required columns are present and raises an error if any column is missing.

2. **Data Cleaning** (`CleanData.py`):
   - The `CleanData.py` script is run to perform data processing tasks such as handling missing values and performing data transformations on the input data.
   - The cleaned data is saved as `train.csv` and `test.csv` files.

3. **Model Training and Selection** (`training.py`):
   - The `training.py` script trains three different models on the cleaned data.
   - For each model, hyperparameter tuning is performed, and the model with the lowest mean absolute error (MAE) is selected as the best model.
   - The results of the model comparison, including the MAE scores, are saved in a `results.txt` file.

4. **Output**:
   - After the workflow completes, the following outputs are available:
     - `train.csv` and `test.csv`: Cleaned data files.
     - `results.txt`: Results of the model comparison and MAE scores.
     - The best model trained using the data.

## Running the Workflow Manually

You can also manually trigger the workflow to run:

1. Open the "Actions" tab in your GitHub repository.
2. Click on the "Workflows" on the left side.
3. Select the workflow named "Main Workflow" or similar.
4. Click the "Run workflow" button to manually trigger the workflow.

## Configuring Workflow Behavior

The workflow behavior can be customized by modifying the `.github/workflows/main.yml` file. You can adjust the triggers, steps, and environment as needed for your project.

## Dependencies

Make sure you have the required dependencies installed for running the scripts. You might need to set up a virtual environment and install the necessary packages using `pip` or a similar tool.

## Notes

- Make sure to replace placeholders like `pytest_csv_columns.py`, `CleanData.py`, and `training.py` with the actual filenames of your scripts.
- Customize the script behavior, column names, and data processing steps according to your project requirements.
- This README provides a high-level overview of the workflow. Refer to the script files and workflow configuration for more details.

For any questions or issues related to the workflow or project, feel free to open an issue or contact the project maintainers.
`diallomous@gmail.com`