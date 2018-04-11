
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 


# In[2]:


people_data = pd.read_csv('turnover.csv')
people_data.head()


# In[3]:


people_data.info()


# In[4]:


people_data.department.unique()


# In[5]:


people_data.salary.unique()


# In[6]:


people_data.salary = people_data.salary.astype("category").cat.reorder_categories(["low","medium","high"]).cat.codes


# In[7]:


departments = pd.get_dummies(people_data.department)
departments = departments.drop(["accounting"],axis=1)


# In[8]:


people_data = people_data.drop(["department"],axis=1)
people_data.head()


# In[9]:


100.0*people_data.churn.value_counts()/len(people_data)


# In[10]:


sns.heatmap(people_data.corr())
plt.show()


# In[11]:


sns.distplot(people_data.satisfaction_level)
plt.show()


# In[12]:


people_data = people_data.join(departments)


# In[13]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[14]:


model_tree = DecisionTreeClassifier(random_state=42)
model_logit = LogisticRegression(random_state=42)


# In[15]:


y = people_data.churn
x = people_data.drop("churn",axis=1)


# In[16]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=42)


# In[17]:


model_tree.fit(x_train,y_train)


# In[18]:


model_tree.score(x_train,y_train)*100


# In[19]:


model_tree.score(x_test,y_test)*100


# In[21]:


model_logit.fit(x_train,y_train)
model_logit.score(x_train,y_train)*100


# In[22]:


model_logit.score(x_test,y_test)*100


# In[23]:


model_tree_7 = DecisionTreeClassifier(max_depth = 7, random_state=42)
model_tree_7.fit(x_train,y_train)
model_tree_7.score(x_train,y_train)*100


# In[24]:


model_tree_7.score(x_test,y_test)*100


# In[25]:


model_tree_100 = DecisionTreeClassifier(min_samples_leaf=100,random_state=42)
model_tree_100.fit(x_train,y_train)
model_tree_100.score(x_train,y_train)*100


# In[26]:


model_tree_100.score(x_test,y_test)*100 


# In[29]:


from sklearn.model_selection import cross_val_score


# In[30]:


print(cross_val_score(model_tree_7, x, y, cv=10))


# In[31]:


overfitting = [0.98401066, 0.97666667, 0.97733333, 0.974, 0.97333333, 0.97466667, 0.98333333, 0.97666667, 0.98198799, 0.98198799]
print (np.mean(overfitting))


# In[32]:


model_tree_7_b = DecisionTreeClassifier(class_weight="balanced", max_depth=7, random_state=42)


# In[33]:


print(cross_val_score(model_tree_7_b, x, y, cv=10))


# In[34]:


print(np.mean(cross_val_score(model_tree_7_b, x, y, cv=10)))


# In[35]:


from sklearn.metrics import recall_score
prediction = model_tree_7.predict(x_test)
recall_score(y_test,prediction)*100


# In[36]:


model_tree_7_b.fit(x_train, y_train)
prediction_b = model_tree_7_b.predict(x_test)
recall_score(y_test, prediction_b)*100


# In[37]:


from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,prediction)*100


# In[38]:


roc_auc_score(y_test,prediction_b)*100


# In[39]:


importances = model_tree_7_b.feature_importances_


# In[40]:


importances


# In[41]:


features = pd.DataFrame(data=importances, columns = ["importances"], index = x.columns)


# In[42]:


features


# In[44]:


selected = features[features.importances>0.1]


# In[45]:


selected


# In[46]:


selected_features = selected.index  #if columns it would take columns


# In[47]:


selected_features


# In[48]:


x_train_new = x_train[selected_features]


# In[1]:


x_train_new.head()


# In[50]:


model_tree_7_b.fit(x_train_new, y_train)


# In[51]:


x_test_new = x_test[selected_features]
model_tree_7_b.score(x_test_new, y_test)


# In[53]:


prediction_b_selected = model_tree_7_b.predict(x_test_new)
recall_score(y_test, prediction_b_selected)


# In[54]:


model_tree_7_b.predict([[0.8, 0.5, 2]])

