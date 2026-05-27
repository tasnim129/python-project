import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression

# Dataset laden
df = pd.read_csv("movie_metadata.csv", 
                 sep=",")
 #data cleaning
print(df.head())
print(df.info())
print(df.describe())
print(df.columns)

X=df.drop("imdb_score", axis=1)
y=df["imdb_score"].copy()
X_train_full, y_train_full, X_test, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
X_train,y_train,X_val,y_val=train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.25
    random_state=42
)

#um zu verifizieren:
#print()
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