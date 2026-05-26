import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Dataset laden
df = pd.read_csv("movie_metadata.csv", 
                 sep=",")

# Größe der Grafik
plt.figure(figsize=(12,8))

# Heatmap anzeigen
#sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
sns.boxplot(x=df['budget'])

plt.title("Budget Outliers")
plt.xlim(0, 500000000)
plt.show()
# Titel
#plt.title("Correlation Heatmap")
print(df.isnull().sum())
# Anzeigen
#plt.show()