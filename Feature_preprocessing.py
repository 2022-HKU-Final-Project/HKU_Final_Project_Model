#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

df = pd.read_csv("7_28_final_data_by_2000w.csv")
print(df.info())


# In[2]:


values = {"jobDiploma": "无", "jobSalary_format": 0, 
         "jobWorkAge": "经验不限","jobDiploma": "学历不限"}
df = df.fillna(value = values)


# In[6]:


print(df['jobDiploma'].value_counts())


# 学历不限 0 
# 初中及以下 1 
# 技校/中技 2 
# 高中 3 
# 大专 4 
# 本科 5 
# 硕士 6 
# 博士 7 
# 无 8

# In[7]:


df.loc[df.jobDiploma == '学历不限', 'jobDiploma'] = 0
df.loc[df.jobDiploma == '初中及以下', 'jobDiploma'] = 1
df.loc[df.jobDiploma == '技校', 'jobDiploma'] = 2
df.loc[df.jobDiploma == '中技', 'jobDiploma'] = 2
df.loc[df.jobDiploma == '中专', 'jobDiploma'] = 2
df.loc[df.jobDiploma == '高中', 'jobDiploma'] = 3
df.loc[df.jobDiploma == '大专', 'jobDiploma'] = 4
df.loc[df.jobDiploma == '本科', 'jobDiploma'] = 5
df.loc[df.jobDiploma == '硕士', 'jobDiploma'] = 6
df.loc[df.jobDiploma == '博士', 'jobDiploma'] = 7
df.loc[df.jobDiploma == '无', 'jobDiploma'] = 8


# In[11]:


#df['jobWorkAge'].value_counts()


# - 经验不限 0
# - 无工作经验 / 无经验 / 经验不限，可接受应届生 1
# - 1年以下 / 1年以下，可接受应届生 2
# - 1年经验 / 2年经验 / 1-2年 / 1-2年，可接受应届生 / 1-3年 3
# - 3-5年 / 3-4年经验 / 3-5年,可接收应届生 4
# - 5-7年经验 / 6-7年 / 6-7年,可接收应届生 / 5-10年 5
# - 8-9年经验 / 8-10年 / 8-10年,可接收应届生 6
# - 10年以上,可接收应届生 / 10年以上 / 10年以上经验 7

# In[16]:


df.loc[df.jobWorkAge == '经验不限', 'jobWorkAge'] = 0
df.loc[df.jobWorkAge == '无工作经验', 'jobWorkAge'] = 1
df.loc[df.jobWorkAge == '无经验', 'jobWorkAge'] = 1
df.loc[df.jobWorkAge == '经验不限,可接收应届生', 'jobWorkAge'] = 1
df.loc[df.jobWorkAge == '1年以下', 'jobWorkAge'] = 2
df.loc[df.jobWorkAge == '1年以下,可接收应届生', 'jobWorkAge'] = 2
df.loc[df.jobWorkAge == '1-3年', 'jobWorkAge'] = 3
df.loc[df.jobWorkAge == '1年经验', 'jobWorkAge'] = 3
df.loc[df.jobWorkAge == '1-2年', 'jobWorkAge'] = 3
df.loc[df.jobWorkAge == '2年经验', 'jobWorkAge'] = 3
df.loc[df.jobWorkAge == '1-2年,可接收应届生', 'jobWorkAge'] = 3
df.loc[df.jobWorkAge == '1-3年', 'jobWorkAge'] = 3
df.loc[df.jobWorkAge == '3-5年', 'jobWorkAge'] = 4
df.loc[df.jobWorkAge == '3-4年经验', 'jobWorkAge'] = 4
df.loc[df.jobWorkAge == '3-5年,可接收应届生', 'jobWorkAge'] = 4

df.loc[df.jobWorkAge == '5-10年', 'jobWorkAge'] = 5
df.loc[df.jobWorkAge == '5-7年经验', 'jobWorkAge'] = 5
df.loc[df.jobWorkAge == '6-7年,可接收应届生', 'jobWorkAge'] = 5
df.loc[df.jobWorkAge == '6-7年', 'jobWorkAge'] = 5
df.loc[df.jobWorkAge == '8-9年经验', 'jobWorkAge'] = 6
df.loc[df.jobWorkAge == '8-10年', 'jobWorkAge'] = 6
df.loc[df.jobWorkAge == '8-10年,可接收应届生', 'jobWorkAge'] = 6
df.loc[df.jobWorkAge == '10年以上,可接收应届生', 'jobWorkAge'] = 7
df.loc[df.jobWorkAge == '10年以上', 'jobWorkAge'] = 7
df.loc[df.jobWorkAge == '10年以上经验', 'jobWorkAge'] = 7


# In[17]:


#df['jobWorkAge'].value_counts()


# In[18]:


print(df.info())

df.to_csv("final_data_0728.csv")

# In[ ]:




