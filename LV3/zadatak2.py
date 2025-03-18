import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('data_C02_emission.csv')

# a) Histogram emisije CO2 plinova
data['CO2 Emissions (g/km)'].plot(kind='hist', bins=30)
plt.title('Histogram emisije CO2')
plt.show()

#b) Odnos gradske potrošnje goriva i emisije CO2 plinova pomoću dijagrama raspršenja
data['Make'] = data['Make'].astype('category')
data['Model'] = data['Model'].astype('category')
data['Fuel Type'] = data['Fuel Type'].astype('category')

plt.scatter(data['Fuel Consumption City (L/100km)'], data['CO2 Emissions (g/km)'], c=data['Fuel Type'].cat.codes, cmap='viridis')
plt.title('Gradska potrošnja vs Emisija CO2')
plt.xlabel('Gradska potrošnja (L/100km)')
plt.ylabel('Emisija CO2 (g/km)')
plt.colorbar(label='Tip goriva')
plt.show()

# c) Razdioba izvangradske potrošnje s obzirom na tip goriva pomoću kutijastog dijagrama
data.boxplot(column='Fuel Consumption Hwy (L/100km)', by='Fuel Type')
plt.title('Izvangradska potrošnja po tipu goriva')
plt.show()

# d) Broj vozila po tipu goriva pomoću stupčastog dijagrama
data['Fuel Type'].value_counts().plot(kind='bar')
plt.title('Broj vozila po tipu goriva')
plt.show()

# e) Prosječna CO2 emisija goriva s obzirom na broj cilindara pomoću stupčastog grafa
data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean().plot(kind='bar')
plt.title('Prosječna emisija CO2 po broju cilindara')
plt.show()