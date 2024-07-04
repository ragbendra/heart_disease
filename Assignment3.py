#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[23]:


file_path='heart-disease.csv' #Load the dataset
df = pd.read_csv(file_path)
df.head()


# In[24]:


df.info()  #display dataset information


# In[25]:


df.describe() #statistical summary


# In[26]:


cat_col = ['cp', 'restecg','slope','thal'] # list of categorical columns to be converted into dummy variables
df = pd.get_dummies(df,columns=cat_col,drop_first=True) # coverting into dummy/indicator variables
df.head() #display the first few rows of the transformed dataset


# In[ ]:


# Calculate the average age of patients with and without heart disease
avg_age_with_disease = df[df['target'] == 1]['age'].mean()
avg_age_without_disease = df[df['target'] == 0]['age'].mean()

print(f"Average age of patients with heart disease: {avg_age_with_disease}")
print(f"Average age of patients without heart disease: {avg_age_without_disease}")


# In[10]:


# to find the distribution of chest pain types among patients
chest_pain_distribution = df[['cp_1', 'cp_2', 'cp_3']].sum()
print("Distribution of chest pain types among patients:")
print(chest_pain_distribution)


# In[11]:


# Find the correlation between thalach and age
correlation_thalach_age = df['thalach'].corr(df['age'])
print(f"Correlation between thalach (maximum heart rate) and age: {correlation_thalach_age}")


# In[27]:


# Analyze the effect of sex on the presence of heart disease
heart_disease_by_sex = df.groupby('sex')['target'].mean()
print("Effect of sex on the presence of heart disease:")
print(heart_disease_by_sex)


# In[32]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[30]:


# Set up the visualizations
plt.figure(figsize=(14, 10))

# Histogram of age distribution
plt.subplot(2, 2, 1)
plt.hist(df['age'], bins=15, color='skyblue', edgecolor='black')
plt.title('Age Distribution of Patients')
plt.xlabel('Age')
plt.ylabel('Frequency')

plt.show()


# In[33]:


# Calculate the distribution of chest pain types
chest_pain_distribution = df[['cp_1', 'cp_2', 'cp_3']].sum()

# Bar chart of chest pain type distribution
plt.figure(figsize=(14, 10))
plt.subplot(2, 2, 2)
chest_pain_distribution.plot(kind='bar', color='salmon', edgecolor='black')
plt.title('Distribution of Chest Pain Types')
plt.xlabel('Chest Pain Type')
plt.ylabel('Number of Patients')
plt.xticks(ticks=[0, 1, 2], labels=['Type 1', 'Type 2', 'Type 3'], rotation=0)

plt.show()


# In[34]:


# Scatter plot of thalach vs age
plt.figure(figsize=(14, 10))
plt.subplot(2, 2, 3)
plt.scatter(df['age'], df['thalach'], color='purple', alpha=0.6)
plt.title('Thalach vs Age')
plt.xlabel('Age')
plt.ylabel('Thalach (Maximum Heart Rate)')

plt.show()


# In[39]:


# Box plot of age distribution with and without heart disease
plt.figure(figsize=(14, 10))
plt.subplot(2, 2, 4)
sns.boxplot(x='target', y='age', data=df, palette='coolwarm')
plt.title('Age Distribution of Patients with and without Heart Disease')
plt.xlabel('Heart Disease (0 = No, 1 = Yes)')
plt.ylabel('Age')

plt.show()


# In[40]:


# Adavanced Analysis(using numpy)


# In[41]:


import numpy as np

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Display the correlation matrix
print(correlation_matrix)


# In[42]:


# Perform a rolling mean analysis on the chol (cholesterol) levels with a window size of 5
rolling_mean_chol = df['chol'].rolling(window=5).mean()

# Plot the rolling mean of cholesterol levels
plt.figure(figsize=(10, 6))
plt.plot(df['chol'], label='Cholesterol')
plt.plot(rolling_mean_chol, label='Rolling Mean (Window=5)', color='orange')
plt.title('Rolling Mean of Cholesterol Levels')
plt.xlabel('Index')
plt.ylabel('Cholesterol')
plt.legend()
plt.show()


# In[ ]:




