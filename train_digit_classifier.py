import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
import pickle

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

train_samples = len(X_train)
test_samples = len(X_test)
X_train = np.array(X_train).reshape((train_samples, -1))
X_test = np.array(X_test).reshape((test_samples, -1))


# Applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components = 200)
X_train = lda.fit_transform(X_train[:5000], y_train[:5000])
X_test = lda.transform(X_test[:100])

# Create a classifier: a support vector classifier
classifier = SVC()

classifier.fit(X_train[:5000], y_train[:5000])

# Now predict the value of the digit on the second half:
expected = y_test[:100]
predicted = classifier.predict(X_test[:100])

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

pickling_on = open("./digit_classifier.pickle","wb")
s = pickle.dump(classifier,pickling_on)
pickling_on.close()

pickling_on = open("./lda.pickle","wb")
s = pickle.dump(lda,pickling_on)
pickling_on.close()