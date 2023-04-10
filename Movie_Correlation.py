#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd 
import numpy as np
import seaborn as sns 

import matplotlib 
import matplotlib.pyplot as plt 
plt.style.use('ggplot')

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8) 

# Read the data 
df = pd.read_csv('/Users/dilyshuynh/Desktop/Python/movies.csv')


# In[9]:


# Let's look at the data 
df.head()


# In[17]:


# Let's see if there is any missing data 
for col in df.columns: 
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, pct_missing))


# In[29]:


# Fix missing values in columns 
median_budget = df['budget'].median()
df['budget'].fillna(median_budget, inplace = True)

median_gross = df['gross'].median()
df['gross'].fillna(median_gross, inplace = True)


# In[27]:


# Data types for columns 
df.dtypes


# In[30]:


# Change data type of columns
df['budget'].astype('int64')

df['gross'].astype('int64')


# In[31]:


df['released'].astype('str')


# In[92]:


df.head()


# In[49]:


df.sort_values(by = ['score'], inplace = False, ascending = False)


# In[ ]:


pd.set_option('display.max_row', None)


# In[50]:


# Drop diplicates to show unique value

# In this code line, I want to show the unique names in Company column 
df['company'].drop_duplicates().sort_values()

#In order to drop all duplicate values
#df.drop_duplicates()


# In[51]:


# Budget high correlation 
# company high correlation


# In[54]:


# Scatter plot with budget and gross

plt.scatter(x = df['budget'], y = df['gross'])

plt.title('Budget vs Gross Earning')

plt.xlabel('Budget Earning')
plt.ylabel('Gross Earning')

plt.show()


# In[60]:


# Plot the budget vs gross using regression 

sns.regplot(x = df['budget'], y = df['gross'], data = df, 
            line_kws = {'color':'blue'})


# In[66]:


# Looking at correlation using pearson

df.corr(method = 'pearson')


# In[67]:


# High correlation between budget and gross earning 


# In[71]:


correlation_matrix = df.corr(method = 'pearson')

sns.heatmap(correlation_matrix, annot = True)

plt.title('Correlation Matrix for Numeric Features')

plt.xlabel('Movie Features')

plt.ylabel('Movie Features')

plt.show() 


# In[73]:


# Looking at Company 
df('company').head()


# In[93]:


df_numerized = df 
for col_name in df_numerized.columns: 
    if(df_numerized[col_name].dtype == 'object'): 
        df_numerized[col_name] = df_numerized[col_name].astype('category')
        df_numerized[col_name] = df_numerized[col_name].cat.codes 

df_numerized.head()


# In[82]:


correlation_matrix = df_numerized.corr(method = 'pearson')

sns.heatmap(correlation_matrix, annot = True)

plt.title('Correlation Matrix for Numeric Features')

plt.xlabel('Movie Features')

plt.ylabel('Movie Features')

plt.show() 


# In[83]:


df_numerized.corr()


# In[87]:


correlation_matrix = df_numerized.corr()

corr_pair = correlation_matrix.unstack()

corr_pair 


# In[88]:


sorted_pair = corr_pair.sort_values()
sorted_pair


# In[90]:


# Find high correlation 

high_corr = sorted_pair[(sorted_pair) > 0.5]
high_corr


# In[91]:





# In[ ]:




