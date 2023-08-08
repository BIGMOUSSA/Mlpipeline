import csv
import os

REQUIRED_COLUMNS = ["course_type",
                    "year", 
                    "enrollment_count", 
                    "pre_score",
                    "post_score",
                    "pre_requirement",
                    "department"
                    ]  # Replace with your required column names

def test_csv_columns():
    csv_path = "data/raw/university_enrollment_2306.csv"
    with open(csv_path, "r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        csv_columns = reader.fieldnames

        for column in REQUIRED_COLUMNS:
            assert column in csv_columns, f"Required column '{column}' not found in CSV file."
