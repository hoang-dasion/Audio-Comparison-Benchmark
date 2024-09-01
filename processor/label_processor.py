import os
import pandas as pd
import numpy as np
from const import DISEASE_COLUMNS

class LabelProcessor:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.label_df = pd.DataFrame()
        self.columns = pd.Index(DISEASE_COLUMNS)

    def make_labels_complete(self, file_name):
        if not file_name.endswith('.csv'):
            file_name += ".csv"

        input_path = os.path.join(self.input_dir, file_name)
        output_path = os.path.join(self.output_dir, f"complete_{file_name}")

        try:
            self.label_df = pd.read_csv(input_path)
            self.label_df['PHQ_class'] = self.label_df.apply(self._classify_phq_score, axis=1)
            self.label_df['PTSD_class'] = self.label_df.apply(self._classify_ptsd_score, axis=1)
            os.makedirs(self.output_dir, exist_ok=True)
            self.label_df.to_csv(output_path, index=False)
            print(f"Complete labels saved to {output_path}")
        except Exception as e:
            print(f"An error occurred while processing the file: {e}")

    def _classify_phq_score(self, row):
        phq_score = row['PHQ_Score']
        if pd.isna(phq_score):
            return self._handle_missing_value(row, 'PHQ_Score')
        elif phq_score <= 4:
            return 0  # Minimal depression
        elif phq_score <= 9:
            return 1  # Mild depression
        elif phq_score <= 14:
            return 2  # Moderate depression
        elif phq_score <= 19:
            return 3  # Moderately severe depression
        else:
            return 4  # Severe depression

    def _classify_ptsd_score(self, row):
        pcl_5_score = row['PTSD Severity']
        if pd.isna(pcl_5_score):
            return self._handle_missing_value(row, 'PTSD Severity')
        elif pcl_5_score < 20:
            return 0
        elif 20 <= pcl_5_score <= 30:
            return 1
        elif 31 <= pcl_5_score <= 40:
            return 2
        else:  # pcl_5_score > 40
            return 3

    def _handle_missing_value(self, row, missing_col):
        missing_count = row[self.columns].isna().sum()
        if missing_count == 1:
            return self._fill_single_missing(row, missing_col)
        else:
            return self._fill_multiple_missing(missing_col)

    def _fill_single_missing(self, row, missing_col):
        gender = row['Gender']
        similar = self.label_df[
            (self.label_df['Gender'] == gender) & 
            (self.label_df[self.columns].notna().all(axis=1)) &
            (self.label_df[self.columns.drop(missing_col)] == row[self.columns.drop(missing_col)]).all(axis=1)
        ]

        if not similar.empty:
            value = np.ceil(similar[missing_col].mean())
        else:
            value = np.ceil(self.label_df[missing_col].dropna().mean())

        return int(value)

    def _fill_multiple_missing(self, missing_col):
        return int(np.ceil(self.label_df[missing_col].dropna().mean()))