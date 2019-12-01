import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

#sns.set(context="notebook", palette="Spectral", style = 'darkgrid' ,font_scale = 1.5, color_codes=True)
#import warnings
#warnings.filterwarnings('ignore')
#%matplotlib inline
#plt.style.use('ggplot')
#ggplot is R based visualisation package that provides better graphics with higher level of abstraction
#import os
diamond = pd.read_csv('diamonds.csv')
# Dataset importieren, die csv Datei von Kaggle
print (diamond)
# Die Liste wird im Fenster angezeigt
diamond.info()
print (diamond.info())
# Sie besteht aus 53940 Zeilen und 11 Spalten
# Daten zu Karat, Schnitt, Farbe, Preis, Reinheit, Tiefe, x, y, z
# Karat ist das Gewicht des Diamanten bzw. eines Minerals/Edelsteins
# Schnitt hat eine Skala von Fair, Good, Very Good, Premium, Ideal
# Farbe von D (am besten) bis J (am schlechtesten)
# Preis
#Spalte "Unnamed" brauchen wir nicht, also entfernen:

diamond = diamond.drop (["Unnamed: 0"], axis=1)
diamond.head()
print (diamond)

#Heatmap
#plt.figure(figsize=(20,20))
#p=sns.heatmap(diamond.corr(), annot=True,cmap='RdYlGn',square=True)
#sns.pairplot(diamond)