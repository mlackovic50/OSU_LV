import pandas as pd
import numpy as np

data = pd.read_csv('data_C02_emission.csv')

# a)
print(f"Broj mjerenja: {data.shape[0]}")
print(data.info())
print(f"Broj izostalih vrijednosti: {data.isnull().sum()}")
data = data.dropna()
print(f"Broj dupliciranih vrijednsoti: {data.duplicated().sum()}")
data = data.drop_duplicates()

data['Make'] = data['Make'].astype('category')
data['Model'] = data['Model'].astype('category')
data['Fuel Type'] = data['Fuel Type'].astype('category')

# b)
print("Tri automobila s najvećom, odnosno najmanjom gradskom potrošnjom:")
print(data[['Make', 'Model', 'Fuel Consumption City (L/100km)']].nlargest(3, 'Fuel Consumption City (L/100km)'))
print(data[['Make', 'Model', 'Fuel Consumption City (L/100km)']].nsmallest(3, 'Fuel Consumption City (L/100km)'))

# c)
subset = data[(data['Engine Size (L)'] >= 2.5) & (data['Engine Size (L)'] <= 3.5)]
print(f"Broj vozila s veličinom motora 2.5<x<3.5: {subset.shape[0]}")
print(f"Njihova prosječna potrošnja je: {subset['CO2 Emissions (g/km)'].mean()}")

# d)
audi_data = data[data['Make'] == 'Audi']
print(f"Broj Audi vozila: {audi_data.shape[0]}")
print(f"Prosječna emisija CO2 plinova Audi vozila s 4 cilindra: {audi_data[audi_data['Cylinders'] == 4]['CO2 Emissions (g/km)'].mean()}")

# e)
broj_vozila_po_cilindrima = data['Cylinders'].value_counts().sort_index()
prosjek_emisije_po_cilindrima = data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean()
print(f"Broj vozila po cilindrima: {broj_vozila_po_cilindrima}")
print(f"Prosjek emisija CO2 po clindrima: {prosjek_emisije_po_cilindrima}")

# f)
diesel = data[data['Fuel Type'] == 'D']['Fuel Consumption City (L/100km)']
benzin = data[data['Fuel Type'] == 'X']['Fuel Consumption City (L/100km)']
print(f"Prosječna gradska potrošnja dizelaša: {diesel.mean()}")
print(f"Prosječna gradska potrošnja benzinaca: {benzin.mean()}")
print(f"Medijalna vrijednost potrošnje dizelaša: {diesel.median()}")
print(f"Medijalna vrijednost potrošnje benzinaca: {benzin.median()}")

# g)
print("Vozilo s 4 cilindra koje koristi dizel s najvećom gradskom potrošnjom:")
print(data[(data['Cylinders'] == 4) & (data['Fuel Type'] == 'D')].nlargest(1, 'Fuel Consumption City (L/100km)'))

# h)
print(f"Broj manualnih vozila: {data[data['Transmission'].str.contains('M')].shape[0]}")

# i)
print(data.corr(numeric_only=True))