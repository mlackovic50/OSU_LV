import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# a)
data = pd.read_csv('data_C02_emission.csv')
input_features = ['Engine Size (L)', 'Cylinders', 'Fuel Consumption City (L/100km)', 
                 'Fuel Consumption Hwy (L/100km)', 'Fuel Consumption Comb (L/100km)', 
                 'Fuel Consumption Comb (mpg)']
X = data[input_features]
y = data['CO2 Emissions (g/km)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# b)
plt.figure(figsize=(10, 6))
plt.scatter(X_train['Fuel Consumption City (L/100km)'], y_train, c='blue', label='Train set')
plt.scatter(X_test['Fuel Consumption City (L/100km)'], y_test, c='red', label='Test set')
plt.xlabel('Fuel Consumption City (L/100km)')
plt.ylabel('CO2 Emissions (g/km)')
plt.title('CO2 Emissions vs. City Fuel Consumption')
plt.legend()
plt.show()

# c)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(X_train['Engine Size (L)'], bins=20)
plt.title('Before Scaling - Engine Size')

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

plt.subplot(1, 2, 2)
plt.hist(X_train_scaled['Engine Size (L)'], bins=20)
plt.title('After Scaling - Engine Size')
plt.show()

# d)
model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("Intercept:", model.intercept_)
print("Koeficijenti:")
for feature, coef in zip(input_features, model.coef_):
    print(f"{feature}: {coef:.4f}")

# e)
y_pred = model.predict(X_test_scaled)

plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Stvarne vrijednosti CO2 emisija (g/km)')
plt.ylabel('Predviđene vrijednosti CO2 emisija (g/km)')
plt.title('Stvarne vs. Predviđene vrijednosti CO2 emisija')
plt.show()

# f)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("---------------------------------------------------")
print("Metrike evaluacije:")
print(f"Srednja kvadratna pogreška (MSE): {mse:.2f}")
print(f"Korijen srednje kvadratne pogreške (RMSE): {rmse:.2f}")
print(f"Srednja apsolutna pogreška (MAE): {mae:.2f}")
print(f"Koeficijent determinacije (R²): {r2:.4f}")

# g)
results = []
for i in range(1, len(input_features)+1):
    selected_features = input_features[:i]
    
    X = data[selected_features]
    y = data['CO2 Emissions (g/km)']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        'num_features': i,
        'features': selected_features,
        'MSE': mse,
        'R2': r2
    })

for result in results:
    print("---------------------------------------------------")
    print(f"Broj ulaznih veličina: {result['num_features']}")
    print(f"Veličine: {result['features']}")
    print(f"MSE: {result['MSE']:.2f}, R²: {result['R2']:.4f}\n")