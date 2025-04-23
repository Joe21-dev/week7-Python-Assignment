import pandas as pd

# Step 1: Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris_df = pd.read_csv(url, header=None, names=column_names)

# Step 2: Compute basic statistics of numerical columns using .describe()
print("Basic statistics of numerical columns:")
print(iris_df.describe())

# Step 3: Group by the 'species' column and compute the mean of numerical columns for each group
print("\nMean of numerical columns grouped by species:")
# Group by 'species' and compute the mean for the numerical columns (exclude 'species' column for mean calculation)
grouped_by_species = iris_df.groupby('species').mean()
print(grouped_by_species)

# Step 4: Identify interesting patterns or findings
# Example analysis: Comparing the mean of 'sepal_length' across species
sepal_length_means = grouped_by_species['sepal_length']
print("\nComparison of mean sepal length across species:")
print(sepal_length_means)

# Additional interesting observation (e.g., correlation between variables)
print("\nCorrelation between numerical columns:")
print(iris_df.corr())
