#!/usr/bin/env python
# coding: utf-8

# # GRIP: The Spaks Foundation Data Science And Business Analytics Inren

# ### Name: Gebril AbouBakr Ahmed

# ### Task 2 : Prediction using unsupervised ML

# ### Step 1 : Importing important libraries

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans


# ### Step 2: Load dataset

# In[7]:


iris = datasets.load_iris()
iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)
iris_df.head() 


# In[8]:


iris_df.shape


# In[9]:


iris_df.info()


# In[10]:


iris_df.isna()


# In[11]:


iris_df.describe()


# ### Step 3:Visualizing the data

# In[12]:


iris_df.hist("sepal length (cm)")
iris_df.hist("sepal width (cm)")
plt.show()


# In[13]:


iris_df.hist("petal length (cm)")
iris_df.hist("petal width (cm)")
plt.show()


# ### Optimum number of clusters

# In[28]:


plt.figure(figsize=(7,5))
Within_cluster_sum_of_squares = []
clusters_range = range(1, 11)
for i in clusters_range:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(iris_df)
    Within_cluster_sum_of_squares.append(kmeans.inertia_)
plt.plot(clusters_range, Within_cluster_sum_of_squares, "go--", color = "red")
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Within cluster sum of squares')
plt.grid()
plt.show()


# ### Applying kmeans to the dataset 

# In[19]:


# Applying kmeans to the dataset / Creating the kmeans classifier
kmeans = KMeans(n_clusters = 3, init = 'k-means++',max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(iris_df)
print(y_kmeans)


# ### Visualising the clusters

# In[25]:


plt.figure(figsize=(8,5))
x = iris_df.iloc[:, [0, 1, 2, 3]].values
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 50, c = 'red', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 50, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],s = 50, c = 'green', label = 'Iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'Centroids')
plt.legend()
plt.grid()
plt.show()


# In[ ]:




