#!/usr/bin/env python
# coding: utf-8

# # Malicious and Benign Websites Project

# ## 1. Environment preparation

# Loading packages and checking their versions.

# In[1]:


import sys #access to system parameters
print("Python version: {}". format(sys.version))

import pandas as pd #collection of functions for data processing and analysis
print("pandas version: {}". format(pd.__version__))

import matplotlib as mpl #collection of functions for visualization
print("matplotlib version: {}". format(mpl.__version__))
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns 
print("Seaborn version: {}". format(sns.__version__))

import numpy as np #a package for scientific computing
print("NumPy version: {}". format(np.__version__))

import scipy as sp #collection of functions for scientific computing and advance mathematics
print("SciPy version: {}". format(sp.__version__)) 

import IPython
from IPython import display #for printing of dataframes in Jupyter notebook
print("IPython version: {}". format(IPython.__version__)) 

import sklearn #collection of machine learning algorithms
print("scikit-learn version: {}". format(sklearn.__version__))
from sklearn import preprocessing, feature_selection, model_selection, metrics, svm, tree, linear_model, neighbors, ensemble
from sklearn import cluster, datasets, mixture, decomposition
from xgboost import XGBClassifier

from datetime import datetime
import time
import warnings


# In[2]:


# import warnings filter
from warnings import simplefilter, filterwarnings
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

from sklearn.exceptions import DataConversionWarning, ConvergenceWarning
filterwarnings('ignore')


# ## 2. Data loading and basic cleaning 

# In[3]:


path =r'dataset.csv' #path to the dataset file 
raw_data = pd.read_csv(path)
raw_data.head(10)


# In[4]:


raw_data.info()


# In[5]:


print(f'There are {raw_data.shape[0]} websites with {raw_data.shape[1]} features.')


# ### Data cleaning
# 
# Completing null values

# In[6]:


print(raw_data.isnull().sum())


# There are 3 features with null values: SERVER, CONTENT_LENGTH, DNS_QUERY_TIMES. 
# 
# The null values of these columns will be corrected with interpolation.

# In[7]:


raw_data.interpolate(inplace = True) #https://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html
print(raw_data.isnull().sum())


# In[8]:


raw_data['SERVER'].fillna("Other server", inplace = True)


# ## 3. Problem definition
# 
# **Problem**: binary classification.<br>
# **Goal**: predict the outcome of the binary event - classify websites as malicious and benign.

# **Common testing procedures:**

# In[9]:


def compare_algorithm_performance(data_x, data_y, algs, cv_split):

  #create table to compare algorithms' metrics
  algs_columns = ['Algorithm', 'Algorithm Parameters', 'Training Time', 'Testing Time', 'Testing Average Accuracy']
  algs_compare = pd.DataFrame(columns = algs_columns)
  for idx,alg in enumerate(algs):
    with warnings.catch_warnings():
      warnings.filterwarnings("ignore", category=ConvergenceWarning)
      #score model with cross validation
      #http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
      cv_results = model_selection.cross_validate(alg, data_x, data_y, cv  = cv_split)
      
    #set name and parameters
    alg_name = alg.__class__.__name__
    algs_compare.loc[idx, 'Algorithm'] = alg_name
    algs_compare.loc[idx, 'Algorithm Parameters'] = str(alg.get_params())   
    algs_compare.loc[idx, 'Training Time'] = cv_results['fit_time'].mean()
    algs_compare.loc[idx, 'Testing Time'] = cv_results['score_time'].mean()
    algs_compare.loc[idx, 'Testing Average Accuracy'] = cv_results['test_score'].mean()      

  #sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html
  algs_compare.sort_values(by = ['Testing Average Accuracy'], ascending = False, inplace = True)
  return algs_compare

def plot_algorithm_performance(algs_compare, clr):
  
  #barplot using https://seaborn.pydata.org/generated/seaborn.barplot.html
  sns.barplot(x='Testing Average Accuracy', y = 'Algorithm', data = algs_compare, color = clr)

  #https://matplotlib.org/api/pyplot_api.html
  plt.title('Machine Learning Algorithm Accuracy Score \n')
  plt.xlabel('Accuracy Score (%)')
  plt.ylabel('Algorithm')

  plt.show()
  
#machine learning algorithms
algs = [
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),
    linear_model.LogisticRegressionCV(),
    linear_model.SGDClassifier(), #stochastic gradient descent
    linear_model.Perceptron(),
    neighbors.KNeighborsClassifier(),   
    svm.SVC(probability=True),  
    tree.DecisionTreeClassifier(),     
    XGBClassifier() #xgboost: http://xgboost.readthedocs.io/en/latest/model.html   
]

#split dataset in cross-validation 
#splitter class: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit
cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .7, random_state = 0 ) # run model 10x with 70/30 split


# ## 4. Baseline
# 

# In[10]:


cleaned_data = raw_data.copy()
cleaned_data.describe()


# In[11]:


numeric_data = cleaned_data.select_dtypes(exclude=['object']).copy()

col_filter = [col for col in numeric_data if col != "Type"]
data_x = numeric_data[col_filter].copy()
data_y = numeric_data.Type.copy()


# In[12]:


algs_compare = compare_algorithm_performance(data_x, data_y, algs, cv_split)
algs_compare


# In[13]:


plot_algorithm_performance(algs_compare, 'r')


# ### Normalization and standartization

# In[14]:


numeric_data_x = data_x.copy()

standard_scaler = preprocessing.StandardScaler()
min_max_scaler = preprocessing.MinMaxScaler()

x = numeric_data_x.values 
x_st_scaled = standard_scaler.fit_transform(x)
x_scaled = min_max_scaler.fit_transform(x_st_scaled)

df = pd.DataFrame(x_scaled)
df.columns = numeric_data_x.columns

for col in df.columns:
  numeric_data_x[col] = df[col].copy()
  numeric_data_x.rename(columns={col:str(col+'_SCALED')}, inplace=True)

numeric_data_x.head()


# In[15]:


algs_compare = compare_algorithm_performance(numeric_data_x, data_y, algs, cv_split)
algs_compare


# In[16]:


plot_algorithm_performance(algs_compare, 'r')


# Some algorithms improved their performance.

# In[17]:


scaled_data = cleaned_data.copy()

for col in df.columns:
  scaled_data[col] = df[col].copy()
  scaled_data.rename(columns={col:str(col+'_SCALED')}, inplace=True)

scaled_data.head()


# ### Dates conversion

# In[18]:


data = scaled_data.copy()


# * **WHOIS_REGDATE**

# In[19]:


for idx, regdate in enumerate(data.WHOIS_REGDATE):
  if regdate == 'None':
    data.at[idx, 'WHOIS_REGDATE'] = None
  else:
    try:
        datetimeObj = datetime.strptime(regdate, '%d/%m/%Y %H:%M')
        data.at[idx, 'WHOIS_REGDATE'] = datetime.timestamp(datetimeObj)
    except ValueError:
        data.at[idx, 'WHOIS_REGDATE'] = None

bool_series = pd.notnull(data.WHOIS_REGDATE) 
notnull_dates = np.array(data.WHOIS_REGDATE[bool_series])
norm_notnull_dates = preprocessing.normalize([notnull_dates])[0]
notnull_idx = np.where(bool_series)[0]

for i in range(len(notnull_idx)):
  idx = notnull_idx[i]
  data.at[idx, 'WHOIS_REGDATE'] = norm_notnull_dates[i]

avg_val = np.average(norm_notnull_dates)

for idx, regdate in enumerate(data.WHOIS_REGDATE):
  if regdate == None:
    data.at[idx, 'WHOIS_REGDATE'] = avg_val

data = data.rename(columns={'WHOIS_REGDATE': 'WHOIS_REGDATE_NORM'})
data.WHOIS_REGDATE_NORM = data.WHOIS_REGDATE_NORM.apply(pd.to_numeric) 
print(data.WHOIS_REGDATE_NORM.head())


# * **WHOIS_UPDATED_DATE**

# In[20]:


for idx, upddate in enumerate(data.WHOIS_UPDATED_DATE):
  if upddate == 'None':
    data.at[idx, 'WHOIS_UPDATED_DATE'] = None
  else:
    try:
        datetimeObj = datetime.strptime(upddate, '%d/%m/%Y %H:%M')
        data.at[idx, 'WHOIS_UPDATED_DATE'] = datetime.timestamp(datetimeObj)
    except ValueError:
        data.at[idx, 'WHOIS_UPDATED_DATE'] = None

bool_series = pd.notnull(data.WHOIS_UPDATED_DATE) 
notnull_dates = np.array(data.WHOIS_UPDATED_DATE[bool_series])
norm_notnull_dates = preprocessing.normalize([notnull_dates])[0]
notnull_idx = np.where(bool_series)[0]

for i in range(len(notnull_idx)):
  idx = notnull_idx[i]
  data.at[idx, 'WHOIS_UPDATED_DATE'] = norm_notnull_dates[i]

avg_val = np.average(norm_notnull_dates)

for idx, upddate in enumerate(data.WHOIS_UPDATED_DATE):
  if upddate == None:
    data.at[idx, 'WHOIS_UPDATED_DATE'] = avg_val

data = data.rename(columns={'WHOIS_UPDATED_DATE': 'WHOIS_UPDATED_DATE_NORM'})
data.WHOIS_UPDATED_DATE_NORM = data.WHOIS_UPDATED_DATE_NORM.apply(pd.to_numeric) 
print(data.WHOIS_UPDATED_DATE_NORM.head())


# In[21]:


scaled_data = data.copy()
scaled_data.head()


# Try with dates columns.

# In[22]:


numeric_data = scaled_data.select_dtypes(exclude=['object']).copy()

col_filter = [col for col in numeric_data if col != "Type"]
data_x = numeric_data[col_filter].copy()
data_y = numeric_data.Type.copy()


# In[23]:


algs_compare = compare_algorithm_performance(data_x, data_y, algs, cv_split)
algs_compare


# In[24]:


plot_algorithm_performance(algs_compare, 'b')


# ## 4. Feature selection and engineering

# In[25]:


data = scaled_data.copy()


# ### Feature selection of categorical values

# There are still some null values in the dataset: some categorical values may have None values as strings ('None'). <br>
# Some categorical features have different formats. <br>
# I will go through all the columns one by one.

# In[26]:


categ_columns = list(['CHARSET', 'SERVER', 'WHOIS_COUNTRY', 'WHOIS_STATEPRO'])
 
for col in categ_columns: 
  print(f'Number of unique values of {col} feature: {len(np.unique(data[col]))}')


# * **URL**

# In[27]:


print(f'Number of unique values of URL feature: {len(np.unique(data.URL))}')


# URL features are completely unique, so the column won't be useful for the further analysis and should be dropped.

# In[28]:


data.drop('URL', axis=1, inplace = True)


# In[29]:


data.head()


# * **SERVER**

# In[30]:


data['SERVER'].value_counts()


# There are lots of different types of servers, and not all of the servers have a full name. 
# Apache, nginx, Microsoft, and mvXXXX.codfw.wmnet types have the largest amounts of values, so it is suggested for the ones that have these names in their names to be renamed.

# In[31]:


most_common_servers = ['Apache', 'nginx', 'Microsoft', 'codfw.wmnet']

for idx, server in enumerate(data.SERVER):
  for server_name in most_common_servers:
    if server_name.lower() in server.lower():
      data.at[idx, 'SERVER'] = server_name
      
print(data['SERVER'].value_counts())


# There are still quite many unique and none values. So I took top 8 most frequently occuring servers, and others including None values were renamed to Other.

# In[32]:


most_freq_servers = list(data['SERVER'].value_counts()[:9].index) # 8 servers + None
most_freq_servers.remove('None')
print(most_freq_servers)

for idx, server in enumerate(data.SERVER):
  if most_freq_servers.count(server) == 0:
    data.at[idx, 'SERVER'] = 'Other server'
    
print(data['SERVER'].value_counts())


# * **CHARSET**

# In[33]:


print(np.unique(data['CHARSET']))
print(data['CHARSET'].value_counts())


# There are the following issues:
# * different capitalization of the same values
# * None as strings
# * ISO-8859-1 and ISO-8859 are of the same standard
# 
# So the entries will be corrected with the appropriate values.

# In[34]:


for idx, charset in enumerate(data.CHARSET):
  data.at[idx, 'CHARSET'] = charset.upper()
  if data.at[idx, 'CHARSET'] == 'NONE':
    data.at[idx, 'CHARSET'] = None
  elif data.at[idx, 'CHARSET'] == 'ISO-8859-1':
    data.at[idx, 'CHARSET'] = 'ISO-8859'

data = data.interpolate()
data = data.dropna(axis=0)
data.index = range(len(data))
print(np.unique(data['CHARSET']))
print(data['CHARSET'].value_counts())


# * **WHOIS_COUNTRY**

# In[35]:


print(data['WHOIS_COUNTRY'].value_counts())


# Most countries are encoded with 2 letters, so full names should be corrected. Capitalization should be treated as well.

# In[36]:


for idx, country in enumerate(data.WHOIS_COUNTRY):
  data.at[idx, 'WHOIS_COUNTRY'] = country.upper()
  if country.upper() == "[U'GB'; U'UK']" or country.upper() == "UNITED KINGDOM":
    data.at[idx, 'WHOIS_COUNTRY'] = 'UK'
  elif country.upper() == "CYPRUS":
    data.at[idx, 'WHOIS_COUNTRY'] = 'CY'
  elif country.upper() == 'NONE':
    data.at[idx, 'WHOIS_COUNTRY'] = 'Other country'
  
print(data['WHOIS_COUNTRY'].value_counts())


# In[37]:


most_freq_countries = list(data['WHOIS_COUNTRY'].value_counts()[:13].index)

for idx, country in enumerate(data.WHOIS_COUNTRY):
  if most_freq_countries.count(country) == 0:
    data.at[idx, 'WHOIS_COUNTRY'] = 'Other country'
    
print(data['WHOIS_COUNTRY'].value_counts())


# * **WHOIS_STATEPRO**

# In[38]:


print(data['WHOIS_STATEPRO'].value_counts())


# This column has a lot of none values, formatting of the values is poor, and there should be a strong correlation between a country and state (different countries may be represented with 1-2 states).
# Since information about different countries is unbalanced, it was decided to drop the column at all.

# In[39]:


data.drop('WHOIS_STATEPRO', axis=1, inplace = True)


# In[40]:


data.head()


# ### Feature engineering for categorical features

# In[41]:


#graph individual features by type
fig = plt.figure(figsize=[30,5])

plt.subplot(131)
sns.barplot(x = 'WHOIS_COUNTRY', y = 'Type', data=data)

plt.subplot(132)
sns.barplot(x = 'SERVER', y = 'Type', data=data)

plt.subplot(133)
sns.barplot(x = 'CHARSET', y = 'Type', data=data)

plt.show()


# In[42]:


dataset_with_dummies = pd.get_dummies(data,prefix_sep='--')
dataset_with_dummies.head()


# In[43]:


numeric_data = dataset_with_dummies.select_dtypes(exclude=['object']).copy()

col_filter = [col for col in numeric_data if col != "Type"]
data_x = numeric_data[col_filter].copy()
data_y = numeric_data.Type.copy()


# In[44]:


algs_compare = compare_algorithm_performance(data_x, data_y, algs, cv_split)
algs_compare


# In[45]:


plot_algorithm_performance(algs_compare, 'g')


# In[46]:


prep_data = dataset_with_dummies.copy()


# ### Feature selection of numerical features

# In[47]:


data = prep_data.copy()


# In[48]:


#correlation heatmap of dataset
def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize': 5 }
    )
    
    plt.title('Pearson Correlation of Features', y=1.05, size=15)

correlation_heatmap(data)


# There are several highly correlated variables.
# 

# In[49]:


corr_matrix = data.corr()
print("Highly correlated variables:")

threshold = 0.75

m_size = corr_matrix.shape[0]

for i in range(m_size):
  for j in range(i+1, m_size):
    if abs(corr_matrix.iloc[i][j]) >= threshold:
      print(f'{corr_matrix.index[i]} and {corr_matrix.columns[j]}: {corr_matrix.iloc[i][j]}')


# It was decided to leave APP_PACKETS, APP_BYTES and URL_LENGTH.

# In[50]:


columns_to_drop = list(['SOURCE_APP_PACKETS_SCALED', 'REMOTE_APP_PACKETS_SCALED', 'TCP_CONVERSATION_EXCHANGE_SCALED', 'REMOTE_APP_BYTES_SCALED', 'NUMBER_SPECIAL_CHARACTERS_SCALED', 'SOURCE_APP_BYTES_SCALED', 'DIST_REMOTE_TCP_PORT_SCALED'])
data.drop(columns_to_drop, axis=1, inplace = True)
data.head()


# In[51]:


correlation_heatmap(data)


# In[52]:


numeric_data = data.copy()

col_filter = [col for col in numeric_data if col != "Type"]
data_x = numeric_data[col_filter].copy()
data_y = numeric_data.Type.copy()


# In[53]:


algs_compare = compare_algorithm_performance(data_x, data_y, algs, cv_split)
algs_compare


# In[54]:


plot_algorithm_performance(algs_compare, 'm')


# In[55]:


data_full = prep_data.copy()


# ## 5. Data exploration

# Exploration of the prepared datasets.

# In[56]:


data = data_full.copy()


# In[57]:


data.describe()


# ### Dataset parameters

# In[58]:


nrow, ncol = data.shape
print(f'Number of observations: {nrow}')
print(f'Number of features: {ncol}')
print(f'Number of malicious websites: {sum(data.Type)}')
print(f'Number of benign websites: {nrow - sum(data.Type)}')

plt.rcParams['figure.figsize'] = (10, 8)

plt.subplot(1, 1, 1)
size = data.Type.value_counts()
labels = 'Benign', 'Malicious'
plt.pie(size, labels = labels, autopct = '.%2f%%')
plt.title('Websites Type Proportion', fontsize = 20)
plt.legend(title="Website Type")

plt.show()


# The dataset is unbalanced: the number of malicious websites is much smaller than benign ones.

# In[59]:


plt.figure(figsize=[25,30])

data = data_full.select_dtypes(exclude=['object']).copy()

col_filter = [col for col in data.columns if len(np.unique(data[col])) > 2] 
numeric_data = data[col_filter]
print(len(numeric_data.columns))

for idx,col in enumerate(numeric_data.columns):
  plt.subplot(5, 3, idx+1)
  plt.hist(x = [numeric_data[data['Type']==1][col], numeric_data[data['Type']==0][col]], stacked=True, color = ['r','g'],label = ['Malicious','Benign'])
  plt.title(f'{col} Histogram by Website Type')
  plt.xlabel(f'{col}')
  plt.ylabel('# of Websites')
  plt.legend()

plt.show()


# ## 5. Modeling

# In[60]:


col_filter = [col for col in data if col != "Type"]
data_x = data[col_filter].copy()
data_y = data.Type.copy()


# ### Clustering

# In[61]:


np.random.seed(0)

plt.figure(figsize=(21, 5))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)

X = data_x
y = data_y

# Principal component analysis
pca = decomposition.PCA(n_components=2)
pca_X = pca.fit(X).transform(X)


# Plot actual classes
plt.figure(figsize=(6.3, 6))
plt.title("Actual classes", size=18)

plt.scatter(pca_X[:, 0], pca_X[:, 1], s=20, c=y)

plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.xticks(())
plt.yticks(())

plt.show()

# Clustering algorithms 
kmeans = cluster.KMeans(n_clusters=2, random_state=0) 
dbscan = cluster.DBSCAN(eps=0.3)  
average_linkage = cluster.AgglomerativeClustering(
        linkage="average", affinity="cityblock",
        n_clusters=2)    
gmm = mixture.GaussianMixture(
        n_components=2, covariance_type='full')

clustering_algorithms = (
        ('KMeans', kmeans),
        ('AgglomerativeClustering', average_linkage),
        ('DBSCAN', dbscan),
        ('GaussianMixture', gmm)
)

plt.figure(figsize=(21, 5))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)

plot_num = 1
for name, algorithm in clustering_algorithms:
  
  t0 = time.time()

  algorithm.fit(X)

  t1 = time.time()

  if hasattr(algorithm, 'labels_'):
    y_pred = algorithm.labels_.astype(np.int)
  else:
    y_pred = algorithm.predict(X)

  # plot
  plt.subplot(1, len(clustering_algorithms), plot_num)
  plt.title(name, size=18)

  plt.scatter(pca_X[:, 0], pca_X[:, 1], s = 20, c=y_pred)

  plt.xlim(-2.5, 2.5)
  plt.ylim(-2.5, 2.5)
  plt.xlabel("PCA1")
  plt.ylabel("PCA2")
  plt.xticks(())
  plt.yticks(())
  plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                 transform=plt.gca().transAxes, size=15,
                 horizontalalignment='right')
  plot_num += 1

plt.show()


# ### Classification

# In[62]:


numeric_data = data_full.copy()

col_filter = [col for col in numeric_data if col != "Type"]
data_x = numeric_data[col_filter].copy()
data_y = numeric_data.Type.copy()


# In[63]:


algs_compare = compare_algorithm_performance(data_x, data_y, algs, cv_split)
algs_compare


# In[64]:


plot_algorithm_performance(algs_compare, 'g')


# In[65]:


def compare_predictions(data_x, data_y, algs, cv_split, parameter, param_vals):
  #create table to compare algorithms metrics
  res_columns = [parameter, 'Accuracy', 'Training Time', 'Testing Time']
  res_compare = pd.DataFrame(columns = res_columns)
  for idx, param_val in enumerate(param_vals):
    alg = algs[idx]  
    #score model with cross validation
    #http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
    cv_results = model_selection.cross_validate(alg, data_x, data_y, cv  = cv_split)
    res_compare.loc[idx, parameter] = param_val
    res_compare.loc[idx, 'Training Time'] = cv_results['fit_time'].mean()
    res_compare.loc[idx, 'Testing Time'] = cv_results['score_time'].mean()
    res_compare.loc[idx, 'Accuracy'] = cv_results['test_score'].mean()      
    
  res_compare.sort_values(by = ['Accuracy'], ascending = False, inplace = True)
  return res_compare


# In[66]:


col_filter = [col for col in data_full if col != "Type"]
data_x = data_full[col_filter].copy()
data_y = data_full.Type.copy()

learning_rates = [pow(10,-i) for i in range(6)]
ns_estimators = [100 * (i+1) for i in range(6)]


# **Gradient Boosting Classifier**

# In[67]:


algs1 = [ensemble.GradientBoostingClassifier(learning_rate=learning_rate) for learning_rate in learning_rates]
comparison_table1 = compare_predictions(data_x, data_y, algs1, cv_split, 'Learning Rate', learning_rates)
comparison_table1


# In[68]:


algs2 = [ensemble.GradientBoostingClassifier(n_estimators=n_estimators) for n_estimators in ns_estimators]
comparison_table2 = compare_predictions(data_x, data_y, algs2, cv_split, 'Number of Estimators', ns_estimators)
comparison_table2 


# In[69]:


best_gbc = [ensemble.GradientBoostingClassifier(learning_rate=0.1, n_estimators=300)]
comparison_table3 = compare_predictions(data_x, data_y, best_gbc, cv_split, '-', ['-'])
comparison_table3


# **Bagging Classifier**

# In[70]:


algs1 = [ensemble.BaggingClassifier(n_estimators=n_estimators) for n_estimators in ns_estimators]
comparison_table1 = compare_predictions(data_x, data_y, algs1, cv_split, 'Number of Estimators', ns_estimators,)
comparison_table1


# In[71]:


best_bc = [ensemble.BaggingClassifier(n_estimators=200)]
comparison_table2 = compare_predictions(data_x, data_y, best_bc, cv_split, '-', ['-'])
comparison_table2


# **AdaBoostClassifier**

# In[72]:


algs1 = [ensemble.AdaBoostClassifier(learning_rate=learning_rate) for learning_rate in learning_rates]
comparison_table1 = compare_predictions(data_x, data_y, algs1, cv_split, 'Learning Rate', learning_rates)
comparison_table1


# In[73]:


algs2 = [ensemble.AdaBoostClassifier(n_estimators=n_estimators) for n_estimators in ns_estimators]
comparison_table2 = compare_predictions(data_x, data_y, algs2, cv_split, 'Number of Estimators', ns_estimators)
comparison_table2


# In[74]:


best_adac = [ensemble.AdaBoostClassifier(learning_rate=1, n_estimators=100)]
comparison_table3 = compare_predictions(data_x, data_y, best_adac, cv_split, '-', ['-'])
comparison_table3


# **Random Forest**

# In[75]:


algs2 = [ensemble.RandomForestClassifier(n_estimators=n_estimators) for n_estimators in ns_estimators]
comparison_table2 = compare_predictions(data_x, data_y, algs2, cv_split, 'Number of Estimators', ns_estimators)
comparison_table2


# In[76]:


best_rfc = [ensemble.RandomForestClassifier(n_estimators=100)]
comparison_table3 = compare_predictions(data_x, data_y, best_rfc, cv_split, '-', ['-'])
comparison_table3


# **XGBoost**

# In[77]:


algs1 = [XGBClassifier(learning_rate=learning_rate) for learning_rate in learning_rates]
comparison_table1 = compare_predictions(data_x, data_y, algs1, cv_split, 'Learning Rate', learning_rates)
comparison_table1


# In[78]:


algs2 = [XGBClassifier(n_estimators=n_estimators) for n_estimators in ns_estimators]
comparison_table2 = compare_predictions(data_x, data_y, algs2, cv_split, 'Number of Estimators', ns_estimators)
comparison_table2


# In[79]:


best_xgbc = [XGBClassifier(learning_rate=1, n_estimators=400)]
comparison_table3 = compare_predictions(data_x, data_y, best_xgbc, cv_split, '-', ['-'])
comparison_table3


# In[80]:


best_algs = [
    best_gbc[0], best_bc[0], best_adac[0], best_rfc[0], best_xgbc[0]    
]

algs_compare = compare_algorithm_performance(data_x, data_y, best_algs, cv_split)
algs_compare


# In[81]:


plot_algorithm_performance(algs_compare, 'g')


# ### Interpretation

# In[82]:


# Print the results
def print_score(classifier,X_train,y_train,X_test,y_test,train=True):
  if train == True:
    print("Training results:")
    print(f'Accuracy: {metrics.accuracy_score(y_train,classifier.predict(X_train))}')
    print(f'Classification Report')
    print(f'{metrics.classification_report(y_train,classifier.predict(X_train))}')
    print('Confusion Matrix:')
    print(f'{metrics.confusion_matrix(y_train,classifier.predict(X_train))}')
              
    res = model_selection.cross_val_score(classifier, X_train, y_train, cv=10, n_jobs=-1, scoring='accuracy')
    print(f'Average Accuracy: {res.mean()}')
              
  elif train == False:
    print("Testing results:")
    print(f'Accuracy: {metrics.accuracy_score(y_test,classifier.predict(X_test))}')
    print(f'Classification Report')
    print(f'{metrics.classification_report(y_test,classifier.predict(X_test))}')
    print('Confusion Matrix:')
    print(f'{metrics.confusion_matrix(y_test,classifier.predict(X_test))}')
    res = model_selection.cross_val_score(classifier, X_test, y_test, cv=10, n_jobs=-1, scoring='accuracy')
    print(f'CV Average Accuracy: {res.mean()}')


# In[83]:


def create_graph(forest, feature_names,tree_name):
  estimator = forest.estimators_[5]
  #https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html
  tree.export_graphviz(estimator, out_file=f'{tree_name}.dot',
                    feature_names = feature_names,
                    class_names = ['benign', 'malicious'],
                    rounded = True, proportion = False, precision = 2, filled = True)

  # Convert to png using system command
  from subprocess import call
  call(['dot', '-Tpng', f'{tree_name}.dot', '-o', f'{tree_name}.png', '-Gdpi=200'])


# In[84]:


def plot_feature_importance(X, feature_importances, title):
  feature_importance_zip = zip(list(X), feature_importances)
  sorted_importance = sorted(feature_importance_zip, key=lambda x: x[1], reverse=True)

  features = X.columns
  importance = feature_importances
  indices = np.argsort(importance).tolist()

  color = plt.cm.Wistia(np.linspace(0, 1, 15))

  plt.rcParams['figure.figsize'] = (15, 10)
  plt.barh(range(len(indices)), importance[indices], color = color)
  plt.yticks(range(len(indices)), features[indices])
  plt.title(title, fontsize = 30)
  plt.grid()
  plt.tight_layout()
  plt.show()


# In[85]:


X = data_x.copy()
y = data_y.copy()


# In[86]:


X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=0)

rf = best_rfc[0]
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)


# In[87]:


print_score(rf,X_train,y_train,X_test,y_test,train=True)       


# In[88]:


print_score(rf,X_train,y_train,X_test,y_test,train=False)


# In[92]:


# with dummies 
X = X_train.copy()
y = y_train.copy()

rf1 = best_rfc[0]
rf1.fit(X_train, y_train)
tree_name = 'tree1'
create_graph(rf1, list(X),tree_name)

from IPython.display import Image
Image(filename = f'{tree_name}.png')


# In[93]:


plot_feature_importance(X, rf1.feature_importances_, 'Feature Importance for Random Forest (with dummies)')


# In[94]:


# without dummies 
col_filter = [col for col in X.columns if not '--' in col]
X_reduced = X[col_filter].copy()

rf2 = best_rfc[0]
rf2.fit(X_reduced, y)
tree_name = 'tree2'
create_graph(rf2, list(X_reduced),tree_name)

from IPython.display import Image
Image(filename = f'{tree_name}.png')


# In[95]:


plot_feature_importance(X_reduced, rf2.feature_importances_, 'Feature Importance for Random Forest (without dummies)')

