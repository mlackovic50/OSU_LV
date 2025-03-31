import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = pd.read_csv('data_C02_emission.csv')

numeric_features = ['Engine Size (L)', 'Cylinders', 'Fuel Consumption City (L/100km)',
                   'Fuel Consumption Hwy (L/100km)', 'Fuel Consumption Comb (L/100km)']
categorical_features = ['Fuel Type']

X = data[numeric_features + categorical_features]
y = data['CO2 Emissions (g/km)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

model_with_fuel = LinearRegression()
model_with_fuel.fit(X_train_encoded, y_train)

y_pred_fuel = model_with_fuel.predict(X_test_encoded)

mse_fuel = mean_squared_error(y_test, y_pred_fuel)
mae_fuel = mean_absolute_error(y_test, y_pred_fuel)
r2_fuel = r2_score(y_test, y_pred_fuel)

print("Metrike evaluacije s ukljucenim tipom goriva:")
print(f"MSE: {mse_fuel:.2f}")
print(f"MAE: {mae_fuel:.2f}")
print(f"Rsquared: {r2_fuel:.4f}")

errors = np.abs(y_test - y_pred_fuel)
max_error = errors.max()
max_error_index = errors.idxmax()
vehicle_with_max_error = data.loc[max_error_index]

print(f"\nMaksimalna pogreska u procijeni: {max_error:.2f} g/km")
print("\nPodaci o vozilu s najvecom pogreskom:")
print(vehicle_with_max_error[['Make', 'Model', 'Vehicle Class', 'Fuel Type', 'CO2 Emissions (g/km)']])