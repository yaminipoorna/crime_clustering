########################## All Packages ##########################

import pandas as pd   
import matplotlib.pyplot as plt
import numpy as np   #math calcukations
import seaborn as sns  #advance visualizations

crime=pd.read_csv("C:/Users/yamini/Desktop/crime_data.csv")

crime.columns
crime.isna().sum()  #na values present

####################Outier Treatment########################################
plt.boxplot(crime.Murder);plt.title('Boxplot');plt.show()  # No outliers
plt.boxplot(crime.Assault);plt.title('Boxplot');plt.show()  # No outliers
plt.boxplot(crime.UrbanPop);plt.title('Boxplot');plt.show()  # No outliers
plt.boxplot(crime.Rape);plt.title('Boxplot');plt.show()  # outliers present

#############Winsorization########################
from scipy.stats.mstats import winsorize

crime['Rape']=winsorize(crime.Rape,limits=[0.03, 0.097])   
plt.boxplot(crime['Rape']);plt.title('Boxplot');plt.show()

##################Normalization###################
from sklearn import preprocessing   #package for normalize
crime_normalized = preprocessing.normalize(crime.iloc[:, 1:])
print(crime_normalized)

##########################Univariate, Bivariate################
plt.hist(crime["Assault"])   #Univariate

plt.hist(crime["Rape"])

plt.scatter(crime["Rape"], crime["UrbanPop"]);plt.xlabel('Rape');plt.ylabel('UrbanPop ')  #Bivariate

crime.skew(axis = 0, skipna = True) 

crime.kurtosis(axis = 0, skipna = True)

##################### for creating dendrogram ##########################
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

###Fidning Distance
z = linkage(crime_normalized, method = "complete", metric = "euclidean")

#####Dendogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, leaf_rotation = 0,  leaf_font_size = 5 )  
plt.show()

# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "euclidean").fit(crime_normalized) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

crime['clust'] = cluster_labels # creating a new column and assigning it to new column 

crime.head()

# Aggregate mean of each cluster
crime.iloc[:, 0:].groupby(crime.clust).mean()

# creating a csv file 
crime.to_csv("Crime.csv", encoding = "utf-8")


