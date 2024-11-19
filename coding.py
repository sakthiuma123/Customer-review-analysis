import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# read the data
df=pd.read_csv('Reviews_clean.csv')
df.head()
df.tail()
df.info
df.isnull().sum()
df.describe()
df=df.dropna(axis = 0)
df.shape
df.head()
data_pos=data[data["Rating"].isin([4,5])]
data_pos.head()
data_neg=data[data["Rating"].isin([1,2])]
data_neg.head()
data.Rating.value_counts()
37
# Plot histogram grid
df.hist(figsize=(15,15), xrot=-45, bins=10) ## Display the labels rotated by 45 degress
# Clear the text "residue"
plt.show()
import seaborn as sns
sns.kdeplot(df['Rating'])
import seaborn as sns
sns.kdeplot(df['Review Votes'])
sns.boxplot( y=df["Rating"] )
sns.heatmap(df.corr(), annot = True)
plt.show()
import seaborn as sns
sns.barplot(x=data.Rating.value_counts().index,y=data.Rating.value_counts().values)
data_filtered=pd.concat([data_pos[:20000],data_neg[:20000]])
data_filtered.shape
sns.barplot(x=data_filtered.Rating.value_counts().index,y=data_filtered.Rating.value_counts(
).values)
data_filtered["r"]=1
data_filtered["r"][data_filtered["Rating"].isin([1,2])]= 0
data_filtered.head()
data_filtered.tail()
data_filtered.r.value_counts()