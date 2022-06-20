#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install pandas')
get_ipython().system('pip install plotly')
get_ipython().system('pip install dash')
get_ipython().system('pip install dash_bootstrap_components')


# In[3]:


import pandas as pd
df=pd.read_csv("C:\\Users\\omate\\Downloads\\india___monthly_rainfall_data___1901_to_2002 (1).csv")
df.head()


# In[4]:


df=df.iloc[2:]
df


# In[5]:


print("Null values before processing:")
df.isnull().sum()


# In[6]:


df[df[['Year']].isna ().any (axis=1)]


# In[7]:


len(df[df[['Year']].isna ().any (axis=1)])


# In[8]:


df=df[~df[['Year']].isna ().any (axis=1)]
df


# In[9]:


df['vlookup'] = df['vlookup'].fillna(df['State']+df['District'])
df


# In[10]:


print("Null values after processing:")
df.isnull().sum()


# In[11]:


print("Co-Variance Matrix")
df.cov()


# In[12]:


print("Correlation Matrix")
df.corr()


# In[13]:


df['mean_rainfall']=df.iloc[:,3:15].mean(axis=1)
df


# In[14]:


import matplotlib.pyplot as plt

a = df.groupby('State').mean()
plt.figure(figsize=(16,6),dpi=80)
plt.xticks(rotation=90)
plt.plot(a['mean_rainfall'],label='mean_rainfall')
plt.legend(loc='best')
plt.title("Mean rainfall by State")


# In[15]:


df['Dec-Feb']=df[['Dec','Jan','Feb']].sum(axis=1)
df['Mar-Jun']=df[['Mar','Apr','May','Jun']].sum(axis=1)
df['Jul-Nov']=df[['Jul','Aug','Sep','Oct','Nov']].sum(axis=1)

df


# In[16]:


plt.figure(figsize=(16,6),dpi=80)
plt.xticks(rotation=90)
a = df.groupby('State').mean()
plt.plot(a['Dec-Feb'],label='Dec-Feb')
plt.plot(a['Mar-Jun'],label='Mar-Jun')
plt.plot(a['Jul-Nov'],label='Jul-Nov')
plt.legend(loc='best')
plt.title("Seasonal variation in rainfall for different states")


# In[17]:


bplot = df[['State', 'Dec-Feb', 'Mar-Jun','Jul-Nov']].groupby(df['State']).sum().plot.bar(stacked=True,figsize=(20,12))
print("Stacked Bar Graph for Rainfall in Different States")


# In[18]:


df[['State', 'Dec-Feb', 'Mar-Jun','Jul-Nov']].groupby(df['District']).sum()


# In[19]:


import seaborn as sns

fig = plt.figure(figsize=(300, 20))
plt.xticks(rotation='vertical')
sns.boxplot(x='District', y='mean_rainfall', data=df)
plt.title("Mean rainfall for each State")

fig.update_layout(xaxis=dict(rangeslider=dict(visible=True),type="linear"))


# In[20]:


import plotly.express as px
fig = px.line_polar(df, r="mean_rainfall",theta="State",
                    color='Year' ,line_close=True,
                    color_discrete_sequence=px.colors.sequential.Plasma_r,
                    template="plotly_dark")
fig.show()


# In[21]:


fig = px.scatter_polar(df, r="mean_rainfall", theta="State",
                       color="mean_rainfall", symbol="Year", size="mean_rainfall",
                       color_discrete_sequence=px.colors.sequential.Plasma_r)

fig.show()


# In[22]:


px.scatter(df, x="Year", 
           y="mean_rainfall", animation_frame="Year", animation_group="State",
           size="mean_rainfall", color="State", hover_name="State", 
           title='Mean Rainfall of each State from years 1900 to 2002',
           log_x=True, size_max=50, range_x=[1899,2004],range_y=[0,300])


# In[23]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

districts = df['District'].unique()

np.random.seed(100)
mycolors = np.random.choice(list(mpl.colors.XKCD_COLORS.keys()), len(districts), replace=False)

plt.figure(figsize=(16,12), dpi= 80)
for i, y in enumerate(districts):
    if i > 0:
        plt.plot('Year', 'mean_rainfall', data=df.loc[df.District==y, :][['Year','mean_rainfall']], color=mycolors[i], label=y)
        plt.text(df.loc[df.District==y, 'Year'][-1:].values[0]+3, df.loc[df.District==y, 'mean_rainfall'][-1:].values[0], y, fontsize=12, color=mycolors[i])

plt.gca().set(xlim=(1900, 2003), ylim=(0,1000), ylabel='$Mean Rainfall$', xlabel='$Year$')
plt.yticks(fontsize=12, alpha=.7)
plt.title("Time Series of Rainfall Data from 1900 to 2002 for all cities")
plt.show()


# In[24]:



import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

districts = df['District'].unique()

np.random.seed(2)
mycolors = np.random.choice(list(mpl.colors.XKCD_COLORS.keys()), len(districts), replace=False)

plt.figure(figsize=(16,12), dpi= 80)
for i, y in enumerate(districts[:5]):
    plt.plot('Year', 'mean_rainfall', data=df.loc[df.District==y, :][['Year','mean_rainfall']], color=mycolors[i], label=y)
    plt.text(df.loc[df.District==y, 'Year'][-1:].values[0]+3, df.loc[df.District==y, 'mean_rainfall'][-1:].values[0], y, fontsize=12, color=mycolors[i])

        
plt.gca().set(xlim=(1900, 2003), ylim=(0,150), ylabel='$Mean Rainfall$', xlabel='$Year$')
plt.yticks(fontsize=12, alpha=.7)
plt.title("Time Series of Rainfall Data from 1900 to 2002 for 10 cities")
plt.show()


# In[ ]:




