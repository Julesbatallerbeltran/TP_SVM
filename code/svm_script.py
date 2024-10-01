#| echo: false
#| include: false
import sys
from pathlib import Path
sys.path.append(str(Path("..") / "code"))
from svm_source import *

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

from sklearn import svm
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from time import time

scaler = StandardScaler()

import warnings
warnings.filterwarnings("ignore")

plt.style.use('ggplot')


n1 = 200
n2 = 200
mu1 = [1., 1.]
mu2 = [-1./2, -1./2]
sigma1 = [0.9, 0.9]
sigma2 = [0.9, 0.9]
X1, y1 = rand_bi_gauss(n1, n2, mu1, mu2, sigma1, sigma2)

plt.show()
plt.close("all")
plt.ion()
plt.figure(1, figsize=(15, 5))
plt.title('First data set')
plot_2d(X1, y1)

X_train = X1[::2]
Y_train = y1[::2].astype(int)
X_test = X1[1::2]
Y_test = y1[1::2].astype(int)

# fit the model with linear kernel
clf = SVC(kernel='linear')
clf.fit(X_train, Y_train)

# predict labels for the test data base
y_pred = clf.predict(X_test)

# check your score
score = clf.score(X_test, Y_test)
print('Score : %s' % score)

# display the frontiere
def f(xx):
    """Classifier: needed to avoid warning due to shape issues"""
    return clf.predict(xx.reshape(1, -1))

plt.figure()
frontiere(f, X_train, Y_train, w=None, step=50, alpha_choice=1)

#%%
# Same procedure but with a grid search
parameters = {'kernel': ['linear'], 'C': list(np.linspace(0.001, 3, 21))}
clf2 = SVC()
clf_grid = GridSearchCV(clf2, parameters, n_jobs=-1)
clf_grid.fit(X_train, Y_train)

# check your score
print(clf_grid.best_params_)
print('Score : %s' % clf_grid.score(X_test, Y_test))

def f_grid(xx):
    """Classifier: needed to avoid warning due to shape issues"""
    return clf_grid.predict(xx.reshape(1, -1))

# display the frontiere
plt.figure()
frontiere(f_grid, X_train, Y_train, w=None, step=50, alpha_choice=1)

#%%
###############################################################################
#               Iris Dataset
###############################################################################

iris = datasets.load_iris()
X = iris.data
X = scaler.fit_transform(X)
y = iris.target
X = X[y != 0, :2]
y = y[y != 0]

#%%
# split train test
X, y = shuffle(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5)




# fit the model
parameters = {'kernel': ['linear'], 'C': list(np.logspace(-3, 3, 200))}
clf_linear = SVC()
clf_linear_grid = GridSearchCV(clf_linear, parameters, n_jobs=-1)
clf_linear_grid.fit(X_train,y_train)

#calcul de y_pred mais pas forcément utile ici
y_pred=clf_linear_grid.predict(X_test)

# compute the score
print(clf_linear_grid.best_params_)
score = clf_linear_grid.score(X_test,y_test)

print('Generalization score for linear kernel: %s, %s' %
      (clf_linear_grid.score(X_train, y_train),
       clf_linear_grid.score(X_test, y_test)))

# Q2 polynomial kernel
Cs = list(np.logspace(-3, 3, 5))
gammas = 10. ** np.arange(1, 2)
degrees = np.r_[1, 2, 3]

parameters = {'kernel': ['poly'], 'C': Cs, 'gamma': gammas, 'degree': degrees}
clf_poly = SVC()
clf_poly_grid = GridSearchCV(clf_poly, parameters, n_jobs=-1)

clf_poly_grid.fit(X_train,y_train)

#calcul de y_pred mais pas forcément utile ici
y_pred_poly=clf_poly_grid.predict(X_test)

# compute the score
score = clf_poly_grid.score(X_test,y_test)

print(clf_poly_grid.best_params_)
print('Generalization score for polynomial kernel: %s, %s' %
      (clf_poly_grid.score(X_train, y_train),
       clf_poly_grid.score(X_test, y_test)))

#%%
# display your results using frontiere

def f_linear(xx):
    """Classifier: needed to avoid warning due to shape issues"""
    return clf_linear_grid.predict(xx.reshape(1, -1))

def f_poly(xx):
    """Classifier: needed to avoid warning due to shape issues"""
    return clf_poly_grid.predict(xx.reshape(1, -1))

#%%
#sans optimisation des paramètres

clf_linear = SVC(kernel='linear')
clf_linear.fit(X_train,y_train)

#calcul de y_pred mais pas forcément utile ici
y_pred=clf_linear.predict(X_test)

# compute the score
score = clf_linear.score(X_test,y_test)

# Q2 polynomial kernel
clf_poly = SVC(kernel='poly')
clf_poly.fit(X_train,y_train)

print('Generalization score for polynomial kernel: %s, %s' %
      (clf_poly.score(X_train, y_train),
       clf_poly.score(X_test, y_test)))

plt.ion()
plt.figure(figsize=(15, 5))
plt.subplot(131)
plot_2d(X, y)
plt.title("iris dataset")

plt.subplot(132)
frontiere(f_linear, X, y)
plt.title("linear kernel")

plt.subplot(133)
frontiere(f_poly, X, y)
plt.title("polynomial kernel")
plt.tight_layout()
plt.draw()

"""
The dataset used in this example is a preprocessed excerpt
of the "Labeled Faces in the Wild", aka LFW_:

  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

  _LFW: http://vis-www.cs.umass.edu/lfw/
"""

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4,
                              color=True, funneled=False, slice_=None,
                              download_if_missing=True)

images = lfw_people.images
n_samples, h, w, n_colors = images.shape

target_names = lfw_people.target_names.tolist()

names = ['Tony Blair', 'Colin Powell']


idx0 = (lfw_people.target == target_names.index(names[0]))
idx1 = (lfw_people.target == target_names.index(names[1]))
images = np.r_[images[idx0], images[idx1]]
n_samples = images.shape[0]
y = np.r_[np.zeros(np.sum(idx0)), np.ones(np.sum(idx1))].astype(int)

plot_gallery(images, np.arange(12))
plt.show()

X = (np.mean(images, axis=3)).reshape(n_samples, -1)


X -= np.mean(X, axis=0)
X /= np.std(X, axis=0)

indices = np.random.permutation(X.shape[0])
train_idx, test_idx = indices[:X.shape[0] // 2], indices[X.shape[0] // 2:]
X_train, X_test = X[train_idx, :], X[test_idx, :]
y_train, y_test = y[train_idx], y[test_idx]
images_train, images_test = images[
    train_idx, :, :, :], images[test_idx, :, :, :]


print("--- Linear kernel ---")
print("Fitting the classifier to the training set")
start_time = time()

# Fit a classifier (linear) and test all the Cs
Cs = 10. ** np.arange(-5, 6)
scores = []
for C in Cs:
    classifier = SVC(kernel='linear', C=C)
    classifier.fit(X_train, y_train)
    
    score = classifier.score(X_train, y_train)
    scores.append(score)

best_index = np.argmax(scores)
print("Best C: {}".format(Cs[best_index]))

plt.figure()
plt.plot(Cs, scores)
plt.xlabel("Paramètres de régularisation C")
plt.ylabel("Scores d'apprentissage")
plt.xscale("log")
plt.tight_layout()
plt.show()
print("Best score: {}".format(np.max(scores)))

print("Predicting the people names on the testing set")
start_time = time()


# Ensure that Cs and ind are defined
Cs = 10. ** np.arange(-5, 6)
scores = []
for C in Cs:
    classifier = SVC(kernel='linear', C=C)
    classifier.fit(X_train, y_train)
    scores.append(classifier.score(X_train, y_train))

ind = np.argmax(scores)  # Find the index of the best C

start_time = time()
classifier = SVC(kernel='linear', C=Cs[ind])
classifier.fit(X_train, y_train)

print("done in %0.3fs" % (time() - start_time))
# The chance level is the accuracy that will be reached when constantly predicting the majority class.
print("Chance level: %s" % max(np.mean(y), 1. - np.mean(y)))
print("Accuracy: %s" % classifier.score(X_test, y_test))



def run_svm_cv(_X, _y):
    _indices = np.random.permutation(_X.shape[0])
    _train_idx, _test_idx = _indices[:_X.shape[0] // 2], _indices[_X.shape[0] // 2:]
    _X_train, _X_test = _X[_train_idx, :], _X[_test_idx, :]
    _y_train, _y_test = _y[_train_idx], _y[_test_idx]

    _parameters = {'kernel': ['linear'], 'C': list(np.logspace(-3, 3, 5))}
    _svr = svm.SVC()
    _clf_linear = GridSearchCV(_svr, _parameters)
    _clf_linear.fit(_X_train, _y_train)

    print('Generalization score for linear kernel: %s, %s \n' %
          (_clf_linear.score(_X_train, _y_train), _clf_linear.score(_X_test, _y_test)))

print("Score sans variable de nuisance")
run_svm_cv(X, y)

print("Score avec variable de nuisance")
n_features = X.shape[1]
# Ajout de variables de nuisance
sigma = 1
noise = sigma * np.random.randn(n_samples, 300)
X_noisy = np.concatenate((X, noise), axis=1)
X_noisy = X_noisy[np.random.permutation(X.shape[0])]
run_svm_cv(X_noisy, y)

#%% Linear kernel fitting and evaluation
print("--- Linear kernel ---")
print("Fitting the classifier to the training set")
t0 = time()

# Fit a classifier (linear) and test all the Cs
Cs = 10. ** np.arange(-5, 6)
scores = []
for C in Cs:
    clf = SVC(kernel='linear', C=C)
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_train, y_train))

ind = np.argmax(scores)
print("Best C: {}".format(Cs[ind]))

plt.figure()
plt.plot(Cs, scores)
plt.xlabel("Paramètres de régularisation C")
plt.ylabel("Scores d'apprentissage")
plt.xscale("log")
plt.tight_layout()
plt.show()
print("Best score: {}".format(np.max(scores)))

print("Predicting the people names on the testing set")
t0 = time()

#%% Cross-validation error curve
from sklearn.model_selection import cross_val_score
err = []

for C in Cs:
    clf = SVC(kernel='linear', C=C)
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    err.append((1 - scores.mean()) * 100)

plt.figure()
plt.plot(Cs, err)
plt.xscale('log')
plt.xlabel('C')
plt.ylabel('Erreur de classification (%)')
plt.title('Erreur de classification en fonction de C')
plt.grid(True)
plt.show()

#%% Predict labels for the X_test images with the best classifier
clf = SVC(kernel='linear', C=Cs[ind])
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("done in %0.3fs" % (time() - t0))
# The chance level is the accuracy that will be reached when constantly predicting the majority class.
print("Chance level : %s" % max(np.mean(y), 1. - np.mean(y)))
print("Accuracy : %s" % clf.score(X_test, y_test))

#%% Qualitative evaluation of the predictions using matplotlib
prediction_titles = [title(y_pred[i], y_test[i], names)
                     for i in range(y_pred.shape[0])]

plot_gallery(images_test, prediction_titles)
plt.show()

#%% Look at the coefficients
plt.figure()
plt.imshow(np.reshape(clf.coef_, (h, w)))
plt.show()

def run_svm_cv(X, y):
    np.random.seed(18)
    indices = np.random.permutation(X.shape[0])
    train_idx, test_idx = indices[:X.shape[0] // 2], indices[X.shape[0] // 2:]
    X_train, X_test = X[train_idx, :], X[test_idx, :]
    y_train, y_test = y[train_idx], y[test_idx]

    parameters = {'kernel': ['linear'], 'C': list(np.logspace(-3, 3, 5))}
    svr = svm.SVC()
    clf_linear = GridSearchCV(svr, parameters)
    clf_linear.fit(X_train, y_train)

    print('Generalization score for linear kernel: %s, %s \n' %
          (clf_linear.score(X_train, y_train), clf_linear.score(X_test, y_test)))

print("Score sans variable de nuisance")
run_svm_cv(X, y)

print("Score avec variable de nuisance")
n_features = X.shape[1]
# Ajout de variables de nuisance
sigma = 5
noise = sigma * np.random.randn(n_samples, 300)
X_noisy = np.concatenate((X, noise), axis=1)
run_svm_cv(X_noisy, y)

#| echo: false
np.random.seed(404)

# Scale the noisy data before applying PCA
scaler.fit(X_noisy)
X_ncr = scaler.transform(X_noisy)

# Apply PCA with 380 components
n_components_380 = 380
pca_380 = PCA(n_components=n_components_380, svd_solver='randomized').fit(X_ncr)
X_redu_380 = pca_380.transform(X_ncr)

# Print scores before and after reduction
print("Score avant réduction :")
run_svm_cv(X_noisy, y)

print("Score après réduction sur 380 composantes :")
run_svm_cv(X_redu_380, y)

# Apply PCA with 200 components
n_components_200 = 200
pca_200 = PCA(n_components=n_components_200, svd_solver='randomized').fit(X_ncr)
X_redu_200 = pca_200.transform(X_ncr)

# Print score after reduction to 200 components
print("Score après réduction sur 200 composantes :")
run_svm_cv(X_redu_200, y)