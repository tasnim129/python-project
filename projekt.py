import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Dataset laden
df = pd.read_csv("movie_metadata.csv", 
                 sep=",")
 #data cleaning
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.columns)
#fehlende Werte entfernen wir konnen auch fullna machen wir schauen mal noch
df = df.dropna()#hier besser mit X_train zu machen 

X = df.select_dtypes(include="number")
X=X.drop("imdb_score", axis=1)
y=df["imdb_score"].copy()
X_train_full,X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
X_train,X_val,y_train,y_val=train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.25,
    random_state=42
)
# Ergebnis: ca. 60% Train, 20% Val, 20% Test
#dann kommt die basline um zu schauen ist unsere linearregression besser als mean oder nicht
baseline=DummyRegressor(strategy="mean")
baseline.fit(X_train,y_train)
pred_baseline=baseline.predict(X_val)
print("Baseline MSE:", mean_squared_error(y_val, pred_baseline))

#wir arbeiten mit regression deshalb brauchen wir einen Skalierung 
scaler=StandardScaler()
scaler.fit(X_train)#er braucht das um die data zu lernen so das er skalieren kann 
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

#Echtes Modell trainieren
model = LinearRegression()

model.fit(X_train_scaled, y_train)

# 10. Auf Validation testen
y_val_pred = model.predict(X_val_scaled)

print("Validation MSE:", mean_squared_error(y_val, y_val_pred))
print("Validation MAE:", mean_absolute_error(y_val, y_val_pred))
print("Validation R2:", r2_score(y_val, y_val_pred))

#auf Test testen
y_test_pred = model.predict(X_test_scaled)

print("Final Test MSE:", mean_squared_error(y_test, y_test_pred))
print("Final Test MAE:", mean_absolute_error(y_test, y_test_pred))
print("Final Test R2:", r2_score(y_test, y_test_pred))








#hier noch nicht fertig!!
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
# Anzeigen
#plt.show()