# Binary classifification of edible/poisonous mushrooms by physical features
# using a neural network

# Dataset provided as public domain on OpenML
# https://www.openml.org/d/24

### 1. Imports
# data analysis
import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow import keras

### 2. Load and describe data
df_obj = pd.read_csv('dataset_24_mushroom.csv')  
print('available features:\n', df_obj.columns.values)
# 23 columns:
#     22 features: 'cap-shape' 'cap-surface' 'cap-color' 'bruises%3F' 'odor'
#               'gill-attachment' 'gill-spacing' 'gill-size' 'gill-color' 
#               'stalk-shape' 'stalk-root' 'stalk-surface-above-ring' 
#               'stalk-surface-below-ring' stalk-color-above-ring' 
#               'stalk-color-below-ring' 'veil-type' 'veil-color' 'ring-number'
#               'ring-type' 'spore-print-color' 'population' 'habitat'
#     label: 'class'
df_obj.info()
# 8124 rows/samples, none with missing of null feature data
# all features and label are objects
df_obj['class'].value_counts()
# 4208 edible 'e' samples, 3916 poisonous 'p' samples
print('_' * 40)

### 3. Preprocessing
# one hot encode all the categorical features, including the label
# df_no_drop = pd.get_dummies(df_obj, columns=None) # for reference
# drop one (redundant) one-hot column taken from each original column
df = pd.get_dummies(df_obj, columns=None, drop_first=True) 
# we now have 96 columns: 95 binary features and 1 binary class ('class_'p'')
# ie the label 'class_'p'' is 0 for edible and 1 for poisonous

# now remove those annoying single quotes from the column index
# (columns does not have str (StringMethods) or replace method, convert columns
# into a Series)
df.columns = pd.Series(df.columns).str.replace("'", "")

# split data into features (X) and label (y) ...
X = df.drop('class_p', axis=1)
y = df.class_p
# ... and into training (64%), development (16%) and testing (20%) sets
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, 
                                                  random_state=123, 
                                                  stratify=y)
X_train, X_dev, y_train, y_dev = train_test_split(X_temp, y_temp, 
                                                  test_size=0.2, 
                                                  random_state=456, 
                                                  stratify=y_temp)
# (mtrain=5199, mdev=1300, mtest=1625)

# normalize training set, and use same values to center and scale the dev and 
# test sets
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_dev = scaler.transform(X_dev)
X_test = scaler.transform(X_test)
print("Training Mean:", X_train.mean(axis=0).round(2)) # confirm feature means are 0
print("Training Std:", X_train.std(axis=0).round(2)) # confirm feature stds are 1
print("Dev Mean:", X_dev.mean(axis=0).round(2)) # confirm feature means are near 0
print("Dev Std:", X_dev.std(axis=0).round(2)) # confirm feature stds are near 1
print("Test Mean:", X_test.mean(axis=0).round(2)) # confirm feature means are near 0
print("Test Std:", X_test.std(axis=0).round(2)) # confirm feature stds are near 1
print('_' * 40)

### 4. Fit a basic network
# will start with a shallow net with very few hidden nodes relative to the 
# number of features
model = keras.Sequential([
        keras.layers.Dense(30, input_dim=95,
                           activation='relu', 
                           kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dense(10, 
                           activation='relu', 
                           kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dense(1,
                           activation='sigmoid',
                           kernel_regularizer=keras.regularizers.l2(0.001))])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=50)
# performance on training data very good
# train: loss=0.0260 acc=1.000

# evaluate on dev and test sets
dev_loss, dev_acc = model.evaluate(X_dev, y_dev)
print('dev set: loss=%0.3f acc=%0.3f' % (dev_loss, dev_acc)) 
# dev: loss=0.025 acc=1.000
# Would normally compare dev and training accuracies to tune hyperparameters, 
# but since they both are very good and do not really vary, there is not much need 
# to change hyperparameters. And since dev scores were not used to cross-validate
# for different combinations of hyperparameters, it would be acceptable to 
# report dev accuracy as a measure of the classifier's performance in place of 
# the testing set accuracy. But let's find the testing score anyway. 
test_loss, test_acc = model.evaluate(X_test, y_test)
print('test set: loss=%0.3f acc=%0.3f' % (test_loss, test_acc)) 
# test: loss=0.025 acc=1.000   (same as dev set)       
        
# export current model
model.save('mushroom_model1.h5')



