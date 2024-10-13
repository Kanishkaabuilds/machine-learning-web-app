import streamlit as st
from sklearn import datasets # Scikit-learn (also sklearn) - most popular open-source ML libraries for Python

# import specific machine learning classifiers:

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from  sklearn.model_selection import train_test_split    # splits your dataset into training and testing sets.
from sklearn.metrics import accuracy_score
from sklearn.decomposition  import PCA    # Principal Component Analysis (PCA), a technique used for dimensionality reduction.


import numpy as np
import matplotlib.pyplot as plt


st.set_page_config(page_title = "Stock Prediction App", page_icon = "🤖")
 
st.title("Streamlit Example")  

st.write("""
# Explore different classifiers
""")

dataset_name = st.sidebar.selectbox("SELECT DATASET", ("Iris", "Breast Cancer", "Wine", "Diabetes", "Heart Disease" ))
classifier_name = st.sidebar.selectbox("SELECT CLASSIFIER", ("KNN", "SVM", "Random Forest"))

# Load dataset from sklearn

def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data =datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    elif  dataset_name == "Wine":
        data = datasets.load_wine()
    elif  dataset_name == "Diabetes":
        data = datasets.load_diabetes()
    else:
        data = datasets.load_heart()

    x = data.data         #variables(data)
    y = data.target       #target labels (class).
    return x,y

x,y = get_dataset(dataset_name)
st.write("Shape of dataset :", x.shape)    
st.write("No.of classes :", len(np.unique(y)))


# shape of the dataset refers to its dimensions in terms of the number of rows and columns
# Rows represent individual data points or samples. Each row contains information about a single observation.
# Columns represent different features or attributes. Each column contains a specific type of information (e.g., height, weight, age) about the samples.'''
# The number of columns is also known as the dimensionality of the dataset.

# different paraameters to modify each classifier

# classes refer to the distinct categories or labels that the model is trying to predict.

# This function creates UI sliders in the sidebar for tuning the hyperparameters of the classifiers:
# For KNN, it adjusts the number of neighbors K.
# For SVM, it adjusts the C parameter (regularization).
# For Random Forest, it adjusts max_depth (tree depth) and n_estimators (number of trees).

def add_parameters_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif  clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    elif  clf_name == "Random Forest":
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100) # no.of trees
        params["max_depth"]=max_depth
        params["n_estimators"] = n_estimators
    return  params

params = add_parameters_ui(classifier_name)

#Load classifier
# initializes the chosen classifier with its respective hyperparameters.

def get_classifier(clf_name,params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif   clf_name == "SVM":
        clf = SVC(C=params["C"])
    else:
        clf = RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"],  random_state=1234)

    return clf

clf = get_classifier(classifier_name, params)

# Classification - Splits the dataset into training (80%) and testing (20%) sets.
x_train,  x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)
clf.fit(x_train, y_train)
y_pred =clf.predict(x_test)

acc = accuracy_score(y_test, y_pred)
st.write(f"classifier = {classifier_name}")
st.write(f"Accuracy = {acc}")

# PLOT
pca = PCA(2) #2D frame
x_projected = pca.fit_transform(x)

x1 = x_projected[:, 0]
x2 = x_projected[:, 1]

fig = plt.figure()

plt.scatter(x1,x2, c=y, alpha = 0.8, cmap = "viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

#plt.show()
st.pyplot(fig)
 