#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import tree
import numpy as np
import pydot
import pandas as pd
import graphviz
import seaborn as sns
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from pylab import rcParams
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import plot_confusion_matrix

# In[4]:
# Initializing dataset

heart =pd.read_csv('D:/University Modules/Machine Learning/heart.csv')
# In[5]:
#Checking the dataset
heart

# In[6]:
#Exploring the data

#describing the data

heart.describe()
# In[7]:

#Checking the column types and selecting particular columns
heart_exp = heart.select_dtypes(include=["float64","int64"])
heart_exp.head()


# In[8]:
#Checking the scale of the dataset

heart.shape
# In[9]:
# EXPLORATORY DATA ANALYSIS------------------------------------------------------

#created a box plot for the dataset with respect to each column values

f, axes = plt.subplots(len(heart_exp.columns), 1, constrained_layout = True, figsize=([13,9]))

for i in range(len(heart_exp.columns)):
    
    sns.boxplot(x=heart_exp[heart_exp.columns[i]], ax=axes[i])


# In[10]:
heart.columns
# In[11]:

#Plotted a bar graph for patient with heart disease and no heart disease using the heartdisease as a target column
f = sns.countplot(x='HeartDisease', data=heart)
f.set_title("Heart disease presence distribution")
f.set_xticklabels(['No Heart disease', 'Heart Disease'])
plt.xlabel("");

# In[12]:
#plotting a bar graph for patient having heart diease with respect to their sex( M or F)

f = sns.countplot(x='HeartDisease', data=heart, hue='Sex')
plt.legend(['Female', 'Male'])
f.set_title("Heart disease presence by gender")
f.set_xticklabels(['No Heart disease', 'Heart Disease'])
plt.xlabel("");


# In[13]:
# Created a correlation matrix with respect to all the columns in the heart dataset.

heat_map = sns.heatmap(heart.corr(method='pearson'), annot=True, fmt='.2f', linewidths=2)
heat_map.set_xticklabels(heat_map.get_xticklabels());

sns.set() #

corr = heart.corr()
print(corr)


# In[14]:


corr['Age'].sort_values(ascending=False)


# In[15]:


plt.scatter(x=heart.Age[heart.HeartDisease==1], y=heart.RestingBP[(heart.HeartDisease==1)], c="red", s=60)
plt.scatter(x=heart.Age[heart.HeartDisease==0], y=heart.RestingBP[(heart.HeartDisease==0)], s=60)
plt.legend(["Disease", "No Disease"])
plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate");


# In[16]:
#Created a new column to compare the data of people having heart disease and no heart disease with respect to Age
heart['Age_Category'] = pd.cut(heart['Age'],bins=list(np.arange(25, 85, 5)))


# In[17]:
#Created a bar plot and compared the both with repect to age group of people having heart disease and no heart disease.


plt.subplot(121)
heart[heart['HeartDisease']==1].groupby('Age_Category')['Age'].count().plot(kind='bar')
plt.title('Age Distribution of Patients with +ve Heart Diagonsis')

plt.subplot(122)
heart[heart['HeartDisease']==0].groupby('Age_Category')['Age'].count().plot(kind='bar')
plt.title('Age Distribution of Patients with -ve Heart Diagonsis')


# In[18]:
#Deleting the column created after the comparison as a part of EDA.

del heart['Age_Category']


# In[25]:
#Created a bar plot for the type chest pain people are having comparing the category of chestpaintype
#Comparing with respect to heart disease
f = sns.countplot(x='ChestPainType', data=heart, hue='HeartDisease')
f.set_xticklabels(['ATA', 'NAP', 'TSY','NA']);
f.set_title('Disease presence by chest pain type')
plt.ylabel('Chest Pain Type')
plt.xlabel('')
plt.legend(['No Disease', 'Disease']);


# In[20]:
#Comparing the Exersice Angina to the Sex Column for both male and females.

heart['Sex'] = np.where(heart['Sex'] == "F", 0, 1)
heart['ExerciseAngina'] = np.where(heart['ExerciseAngina'] == "N", 0, 1)


# In[31]:

#Plotted a comparison barplot for patients with different category of chestpain and checking the total amount of people having the type of pain.
sns.countplot(heart['ChestPainType'])
plt.xticks([0,1,2,3],["atypical angina", "non-anginal pain", "asymptomatic", "typical angina" ])
plt.xticks()
plt.show()


# In[32]:


# Showing Fasting Blood Sugar Distribution According To HeartDisease as Variable
sns.countplot(x="FastingBS", hue="HeartDisease", data=heart)
plt.legend(labels = ['No-Disease','Disease'])
plt.show()


# In[34]:


heart['RestingBP'].hist() #Check Resting Blood Pressure Distribution


# In[36]:

#Checking the blood pressure range for the patients with respect to their sex.
g = sns.FacetGrid(heart,hue="Sex",aspect=4)
g.map(sns.kdeplot, 'RestingBP', shade=True)
plt.legend(labels=['Male', 'Female'])

# EXPLORATORY DATA ANALYSIS------------------------------------------------------END
# In[32]:

#MODEL CLASSIFICATION------------------------------------------------------------------------START

# Manipulating data and creating dummy variables for Model processing.

heart = pd.get_dummies(heart)
heart.head().T




# In[81]:

#creating dataset to pass the value for test_train_split    

y=heart["HeartDisease"]
x=heart.drop(["HeartDisease"], axis=1)

x_train, x_test , y_train, y_test = train_test_split(x,y,test_size = 0.25, random_state=42)


# In[35]:
# 1st MODEL DECISION TREE CLASSIFIER


decision=DecisionTreeClassifier().fit(x_train,y_train)

y_pred = decision.predict(x_test)
np.sqrt(mean_squared_error(y_test,y_pred))


# In[36]:
#Calcuating the model accuracy for Test Set and Training Set

print('Training set score: {:.4f}'.format(decision.score(x_train, y_train)*100))
print('Test set score: {:.4f}'.format(decision.score(x_test, y_test)*100))


# In[37]:
#Printed the classification report using the classification_report module from skelearn

print(classification_report(y_pred,y_test))

# In[38]:
#plotting the confusion matrix for decsion tree model

plot_confusion_matrix(decision, x_test, y_test)

#Getting Accuracy Score
decision.score(x_test, y_test)

accuracy_score_dt = round(decision.score(x_test, y_test)*100,2)

accuracy_score_dt

print("The accuracy score achieved using Neural Network is: "+str(accuracy_score_dt)+" %")

# In[85]: #Reference [3] from kaggle mentioned in Report
# HPYER PARAMETER TUNING FOR DECISION TREE CLASSIFIER

#creating a grid search CV classifier using decision tree

def evaluate_model(decision):
    print("Train Accuracy :", accuracy_score(y_train, decision.predict(x_train)))
    print("Train Confusion Matrix:")
    print(confusion_matrix(y_train, decision.predict(x_train)))
    print("-"*50)
    print("Test Accuracy :", accuracy_score(y_test, decision.predict(x_test)))
    print("Test Confusion Matrix:")
    print(confusion_matrix(y_test, decision.predict(x_test)))


# In[86]:
#evaluating the model accuracy
evaluate_model(decision)


# In[88]:


# Create the parameter grid based on the results of random search 
params = {
    'max_depth': [2, 3, 5, 10, 20],
    'min_samples_leaf': [5, 10, 20, 50, 100],
    'criterion': ["gini", "entropy"]
}


# In[90]:


# Instantiate the grid search model
grid_search = GridSearchCV(estimator=decision, 
                           param_grid=params, 
                           cv=4, n_jobs=-1, verbose=1, scoring = "accuracy")


# In[91]:

#Running the Grid Search CV (Cross Validation for 4 folds on decision tree classifier under hyper parameters grid)
get_ipython().run_cell_magic('time', '', 'grid_search.fit(x_train, y_train)')


# In[92]:

#Calculating the accracy for the gridsearchcv model run on decision tree

score_df = pd.DataFrame(grid_search.cv_results_)
score_df.head()


# In[93]:
#Calculating the best score out of the all the results

heart_best=grid_search.best_estimator_

evaluate_model(heart_best)


# In[95]:

#created the classification report for the GridsearchCV classifier for hyper parameter tuning done on decision tree classifer.
print(classification_report(y_test, heart_best.predict(x_test)))
#Created a confusion matrix for gridsearchcv
plot_confusion_matrix(heart_best, x_train, y_train)
# In[46]:
## SECOND MODEL RANDOM FOREST CLASSIFIER

#Building up the random forest model.

rf_model = RandomForestClassifier().fit(x_train, y_train)

# In[47]:
#Creating a function and calculating the mean squared error for random forest

y_pred_rf = rf_model.predict(x_test)
np.sqrt(mean_squared_error(y_test, y_pred_rf))

# In[48]:
#Calculating the accuracy for test set and train for random forest classifier

print('Training set score: {:.4f}'.format(rf_model.score(x_train, y_train)*100))
print('Test set score: {:.4f}'.format(rf_model.score(x_test, y_test)*100))
# In[49]:

#Created a classification report for random forest model accuracy
print(classification_report(y_pred_rf,y_test))

#Getting the accuracy score for random forest classifier

rf_accuracy=accuracy_score(y_pred_rf,y_test)*100
rf_accuracy

print("The accuracy score achieved using Neural Network is: "+str(rf_accuracy)+" %")

# In[50]:
#plotting the confusion matrix for random forest

plot_confusion_matrix(rf_model, x_test, y_test)


# In[3]: # Reference [4] from Towards Data Science mentioned in the report
## Third MODEL USING KERAS BACKED BY TENSORFLOW

#for creating graph on white plots

sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 14, 8

#For classifier in tensorflow

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# In[54]:
#Data PreProcessing
# Tensorflow feature columns are used here as the data contains a mixture of categorial and numerical data.
#so i have created functions for Numeric columns, bucketed other columns having float values and created embedded columns with the string and float values which are amended in the feature column
feature_columns = []

# numeric cols
for header in ['Age', 'RestingECG_Normal' ,'Cholesterol', 'RestingBP', 'Oldpeak']:
      feature_columns.append(tf.feature_column.numeric_column(header))

# bucketized cols
age = tf.feature_column.numeric_column("Age")
age_buckets = tf.feature_column.bucketized_column(age, boundaries=[25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85])
feature_columns.append(age_buckets)

# indicator cols
heart["FastingBS"] = heart["FastingBS"].apply(str)
FBS = tf.feature_column.categorical_column_with_vocabulary_list(
      'FastingBS', ['0', '1', '2'])
FBS_one_hot = tf.feature_column.indicator_column(FBS)
feature_columns.append(FBS_one_hot)

heart["Sex_F"] = heart["Sex_F"].apply(str)
sex_F = tf.feature_column.categorical_column_with_vocabulary_list(
      'Sex_F', ['0', '1'])
sex_F_one_hot = tf.feature_column.indicator_column(sex_F)
feature_columns.append(sex_F_one_hot)

heart["Sex_M"] = heart["Sex_M"].apply(str)
sex_M = tf.feature_column.categorical_column_with_vocabulary_list(
      'Sex_M', ['0', '1'])
sex_M_one_hot = tf.feature_column.indicator_column(sex_M)
feature_columns.append(sex_M_one_hot)

heart["ST_Slope_Up"] = heart["ST_Slope_Up"].apply(str)
slope_Up = tf.feature_column.categorical_column_with_vocabulary_list(
      'ST_Slope_Up', ['0', '1', '2'])
slope_Up_one_hot = tf.feature_column.indicator_column(slope_Up)
feature_columns.append(slope_Up_one_hot)

heart["ST_Slope_Down"] = heart["ST_Slope_Down"].apply(str)
slope_Down = tf.feature_column.categorical_column_with_vocabulary_list(
      'ST_Slope_Down', ['0', '1', '2'])
slope_down_one_hot = tf.feature_column.indicator_column(slope_Down)
feature_columns.append(slope_down_one_hot)

heart["ChestPainType_TA"] = heart["ChestPainType_TA"].apply(str)
cp = tf.feature_column.categorical_column_with_vocabulary_list(
      'ChestPainType_TA', ['0', '1', '2'])
cp_one_hot = tf.feature_column.indicator_column(cp)
feature_columns.append(cp_one_hot)

# embedding cols
thal_embedding = tf.feature_column.embedding_column(FBS, dimension=8)
feature_columns.append(thal_embedding)

# crossed cols
age_FBS_crossed = tf.feature_column.crossed_column([age_buckets, FBS], hash_bucket_size=1000)
age_FBS_crossed = tf.feature_column.indicator_column(age_FBS_crossed)
feature_columns.append(age_FBS_crossed)

cp_slope_crossed = tf.feature_column.crossed_column([cp, slope_Up], hash_bucket_size=1000)
cp_slope_crossed = tf.feature_column.indicator_column(cp_slope_crossed)
feature_columns.append(cp_slope_crossed)


# In[55]:

# Created a pandas dataframe and converted it into a Tensorflow dataset

def create_dataset(dataframe, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('HeartDisease')
    return tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))     .shuffle(buffer_size=len(dataframe))     .batch(batch_size)


# In[56]:

#created a test trian split for the model
train, test = train_test_split(heart, test_size=0.2, random_state=RANDOM_SEED)


# In[57]:
#Created train and test dataset to fit in the model

train_ds = create_dataset(train)
test_ds = create_dataset(test)


# In[58]:
#Creating a binary classifier using tensorflow

model = tf.keras.models.Sequential([
        tf.keras.layers.DenseFeatures(feature_columns=feature_columns),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
])


# In[59]:

#This model will run the classifier build above for the batch size created and will show the accuracy and loss values for each iteration.

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_ds, validation_data=test_ds, epochs=100, use_multiprocessing=True)


# In[60]:

#finded the model accuracy for the test set
model.evaluate(test_ds)


# In[61]:
#finded the model accuracy for train set

model.evaluate(train_ds)


# In[62]:

#plotted a graph for the model accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.ylim((0, 1))
plt.legend(['train','test'], loc='upper left');


# In[63]:
#plotted a graph for the model loss.

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[66]:
#Predicting the heart disease
#here the model is making a binary decsion, so we are taking the maximum binary probability of the output layer

predictions = model.predict(test_ds)
bin_predictions = tf.round(predictions).numpy().flatten()
y_test_keras = tf.round(model.predict(test_ds)).numpy().flatten()


# In[69]:

#Build up a classification report stating the accuracy and other scores.
print(classification_report(y_test_keras,bin_predictions))

#Getting the accuracy score for Keras Model:

keras_accuracy=accuracy_score(y_test_keras, bin_predictions)*100
keras_accuracy

print("The accuracy score achieved using Neural Network is: "+str(keras_accuracy)+" %")

# In[70]:

#Created a function for plotting the confusion matrix using the predicted values and test_labels
cnf_matrix = confusion_matrix(y_test_keras, bin_predictions)
cnf_matrix


# In[71]:
#Plotted the confusion matrix for the model.

class_names = [0,1]
fig,ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)

sns.heatmap(pd.DataFrame(cnf_matrix),annot=True,cmap="Blues",fmt="d",cbar=False)
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.ylabel('Actual label')
plt.xlabel('Predicted label');

##MODEL CLASSIFICATION -----------------------------------------------------------------------------------END

#MODEL ACCURACY COMPARISON # Reference [1] from Github Mentioned in Report

# Created a list of scores and Models

scores=[accuracy_score_dt,rf_accuracy,keras_accuracy]
models= ['DecisionTreeClassifier','Random Forest Classifier', 'Sequnetial Model-Keras']

#Printing the accuracy of each models:

for i in range(len(models)):
    print("The accuracy score achieved using "+models[i]+" is: "+str(scores[i])+" %")


# Comapring the accuracy of each model

sns.set(rc={'figure.figsize':(15,8)})
plt.xlabel("Models")
plt.ylabel("Accuracy score")

sns.barplot(models,scores)  