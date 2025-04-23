import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris_df = pd.read_csv(url, header=None, names=column_names)

# Set the Seaborn style for better visuals
sns.set(style="whitegrid")

# 1. Line Chart - Trend over "time" (simulating an index for time)
plt.figure(figsize=(10, 6))
plt.plot(iris_df.index, iris_df['sepal_length'], label='Sepal Length', color='b')
plt.plot(iris_df.index, iris_df['petal_length'], label='Petal Length', color='r')
plt.title("Trend of Sepal and Petal Length over Index")
plt.xlabel("Index (Simulated Time)")
plt.ylabel("Length (cm)")
plt.legend()
plt.grid(True)
plt.show()

# 2. Bar Chart - Average Petal Length per Species
plt.figure(figsize=(10, 6))
avg_petal_length = iris_df.groupby('species')['petal_length'].mean()
sns.barplot(x=avg_petal_length.index, y=avg_petal_length.values, palette="viridis")
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.show()

# 3. Histogram - Distribution of Sepal Length
plt.figure(figsize=(10, 6))
sns.histplot(iris_df['sepal_length'], kde=True, bins=15, color='purple')
plt.title("Distribution of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter Plot - Relationship between Sepal Length and Petal Length
plt.figure(figsize=(10, 6))
sns.scatterplot(data=iris_df, x='sepal_length', y='petal_length', hue='species', palette="Set1", s=100)
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()
