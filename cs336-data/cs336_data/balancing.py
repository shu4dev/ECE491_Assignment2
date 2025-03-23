import pandas as pd
import numpy as np

# Load the CSV file
data = pd.read_csv('/home/shu4/koa_scratch/ECE491_Assignment2/data/positive_sample/labels.csv')

# Separate the two classes based on the 'label' column
majority = data[data['label'] == 0]
minority = data[data['label'] == 1]

print("Original counts:")
print("Class 0 (majority):", len(majority))
print("Class 1 (minority):", len(minority))

# Set target_count as the average of the two class counts.
# This is one approach to balance the dataset while minimizing the distortion of the original data.
target_count = int((len(majority) + len(minority)) / 2)

# Undersample the majority class (class 0)
majority_sampled = majority.sample(n=target_count, random_state=42)

# Oversample the minority class (class 1) with replacement if necessary
minority_sampled = minority.sample(n=target_count, replace=True, random_state=42)

# Combine the undersampled majority class and the oversampled minority class
balanced_data = pd.concat([majority_sampled, minority_sampled]).reset_index(drop=True)

# Save the balanced dataset to a new CSV file
balanced_data.to_csv("/home/shu4/koa_scratch/ECE491_Assignment2/data/positive_sample/balanced_labels.csv", index=False)

print("Balanced dataset saved as 'balanced_labels.csv'")