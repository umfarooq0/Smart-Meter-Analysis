
from cnn_vae import LCLid_
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

LCLid_ = pd.read_csv("/home/usman/Documents/Smart-Meter-analysis/total_hhblock.csv", sep=',', usecols=['LCLid'], squeeze=True)

df  = pd.read_csv('/home/usman/Documents/Smart-Meter-analysis/vae_features_.csv',index_col = False)


distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1, 10)
X = df.iloc[:,1:]
for k in K:
	# Building and fitting the model
	kmeanModel = KMeans(n_clusters=k).fit(X)
	kmeanModel.fit(X)

	distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,
										'euclidean'), axis=1)) / X.shape[0])
	inertias.append(kmeanModel.inertia_)

	mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_,
								'euclidean'), axis=1)) / X.shape[0]
	mapping2[k] = kmeanModel.inertia_

plt.plot(K, distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.show()

plt.plot(K, inertias, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Inertia')
plt.title('The Elbow Method using Inertia')
plt.show()

# we can see from the plots that there is either  5 clusters. 


kmeanModel = KMeans(n_clusters=9).fit(X)
nine_cluster_labels = kmeanModel.labels_
df['nine_cluster_labels'] = nine_cluster_labels
df['LCLid'] = LCLid_.unique()

df_long = pd.read_csv('/home/usman/Documents/Smart-Meter-analysis/weekly_avg_avg_long.csv')

df_clusters = df[['LCLid','nine_cluster_labels']]


df_long = df_long.merge(df_clusters,how = 'inner', on='LCLid')

df_long.groupby('nine_cluster_labels')['LCLid'].count()

'''
0    4350
1       1
2     132
3       1
4    1075
'''
df_long = df_long.iloc[:,1:]

# what we can do is save the clusters per id and save the plots as well 

for i in range(9):
	datashow = df_long[df_long.nine_cluster_labels == i]
	avg_ = datashow.iloc[:,:-2].mean(axis = 0)
	plt.figure(figsize=(12, 6), dpi=80)
	ds_size = datashow.shape[0]
	for j in range(ds_size):
		plt.plot(datashow.iloc[j,:-2],c= 'k',linewidth = 1)
	plt.plot(avg_, c = 'r', linewidth = 2)
	plt.show()
	plt.savefig('/home/usman/Documents/Smart-Meter-analysis/Clustering/vae_five/cluster_{}.png'.format(i))


zzz = '/home/usman/Documents/Smart-Meter-analysis/Clustering/vae_ten/cluster_{}.png'.format(i)
## Nothing that you can tell about those load profiles
# There are lot, majority fall into profile 0. 

# we can look at what the average and std of those profiles is

def get_avg_profile(df,cluster):
	avg_k_zero = df[df.nine_cluster_labels == cluster]
	avg_k_zero_ = avg_k_zero.iloc[:,:-2].mean()
	std_k_zero_ = avg_k_zero.iloc[:,:-2].std()
	return avg_k_zero_,std_k_zero_


def print_avg_profile(mean,std):
	x = np.linspace(0,len(mean),len(mean))
	plt.plot(x, mean, 'k-')
	plt.fill_between(x, mean-std, mean+std)
	plt.show()

for i in range(9):
	mean_,std_ = get_avg_profile(df_long,i)
	plt.plot(mean_)
plt.show()
#VAE seems to have taken the features and made them quite similiar to one another
# what does that mean? VAE didnt converge? 5 features isnt enough? 

zero_mean,zero_std = get_avg_profile(df_long,0)
print_avg_profile(zero_mean,zero_std)

one_mean,one_std = get_avg_profile(df_long,1)
print_avg_profile(one_mean,one_std)

two_mean,two_std = get_avg_profile(df_long,2)
print_avg_profile(two_mean,two_std)

three_mean,three_std = get_avg_profile(df_long,3)
print_avg_profile(three_mean,three_std)

four_mean,four_std = get_avg_profile(df_long,4)
print_avg_profile(four_mean,four_std)

df_clusters.to_csv('/home/usman/Documents/Smart-Meter-analysis/Clustering/simple_kmeans_csv')
