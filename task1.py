import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset from seaborn (can also be from a local CSV)
df = sns.load_dataset("iris")

# Display the first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Check data types and missing values
print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

# Clean the dataset (in this case, no missing values, but let's demonstrate)
df_cleaned = df.dropna()  # Or df.fillna(value) if you want to fill instead

# Summary statistics
print("\nBasic Statistical Summary:")
print(df.describe())

# Observations:
print("\nObservations:")
print("- The dataset has 150 rows and 5 columns.")
print("- All features are numeric except the target column 'species'.")
print("- No missing values were found.")
print("- The classes in 'species' are: ", df['species'].unique())

# Visualizations
sns.pairplot(df, hue="species")
plt.suptitle("Pairplot of Iris Dataset", y=1.02)
plt.show()

plt.figure(figsize=(8, 5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap of Numeric Features")
plt.show()
