#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


df=pd.read_csv(r"C:/Users/DELL LATITUDE E7450/Desktop/zakaria_ML/music.csv")
#df


# In[4]:


df.shape


# In[5]:


# we are going to split the data


# In[6]:


music_data=df


# In[7]:


music_data


# In[8]:


X= music_data.drop( columns = ['genre'])
X


# In[9]:


Y= music_data['genre']
Y


# In[13]:


X.isnull().sum()


# In[22]:


print(X.shape)
print(Y.shape)


# In[23]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X,Y)
Predictions = model.predict([[21,1],[22,0] ])
predictions


# In[35]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Encode the target
le = LabelEncoder()
Y_encoded = le.fit_transform(Y)

# Create and train the model
model = DecisionTreeClassifier()
model.fit(X, Y_encoded)

# Optional: predict a new sample
sample = [[28]]  # age=28
predictions = model.predict([
    [21, 1],
    [22, 0]
])

# Convert back to genre names
predicted_genres = le.inverse_transform(predictions)
print(predicted_genres)

#pred_genre = le.inverse_transform(pred)
#print(pred_genre)


# In[38]:


from sklearn.metrics import accuracy_score
# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(Y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Create and train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
score = accuracy_score(y_test, predictions)
print("Accuracy:", score)


# In[39]:


print(le.inverse_transform(predictions))


# In[40]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Encode the target
le = LabelEncoder()
Y_encoded = le.fit_transform(Y)

# Create and train the model
model = DecisionTreeClassifier()
model.fit(X, Y_encoded)
#joblib.dump( model, 'music-recommender.joblib')
model= joblib.load( 'music-recommender.joblib')
predictions = model.predict([[21,1]])
predictions
print(le.inverse_transform(predictions))



# In[41]:


# how to visualize the model in graph
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import LabelEncoder

# Encode the target
le = LabelEncoder()
Y_encoded = le.fit_transform(Y)

# Train the model
model = DecisionTreeClassifier()
model.fit(X, Y_encoded)

# Export the decision tree
export_graphviz(
    model,
    out_file="music-recommender.dot",
    feature_names=["age", "gender"],
    class_names=le.classes_,
    filled=True,
    rounded=True,
    label="all"
)


# In[43]:


#from graphviz import Source

# Read your DOT file
#with open("tree.dot") as f:
    #dot_graph = f.read()

# Show the graph in VS Code
#graph = Source(dot_graph)
#graph.view()  # Opens a new window with the tree


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




