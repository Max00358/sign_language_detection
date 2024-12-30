# This file is used to train classifier model using gesture landscape data
import pickle
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_dict = pickle.load(open('./data.pickle', 'rb'))

trim_data = []
for i, vec in enumerate(data_dict['data']):
    if len(vec) == 42:
        trim_data.append(vec)
    elif len(vec) != 42:
        print(f"Unexpected length ~ index: {i}: {len(vec)} - {vec}")
        trim_data.append(vec[:42])

# data = np.asarray(data_dict['data'])
data = np.asarray(trim_data)
labels = np.asarray(data_dict['labels'])
assert data.shape[0] == labels.shape[0], "Mismatch between data and labels!"

x_train, x_test, y_train, y_test = train_test_split(data,
                                                    labels,
                                                    # Data split into x_train (80%) and x_test (20%), so is Label
                                                    test_size=0.2,
                                                    # Shuffles the dataset before splitting to avoid biases
                                                    shuffle=True,
                                                    # Ensures the split maintains the same label distribution in both
                                                    # training & testing datasets
                                                    stratify=labels)

# Scaler ensures the features in your dataset are scaled
# (typically to zero mean and unit variance)
model = Pipeline([
    ('standardscaler', StandardScaler()),
    ('sgdclassifier', SGDClassifier(
        loss='log_loss',           # Logistic regression loss for probabilistic predictions
        max_iter=1000,             # Maximum number of iterations
        tol=1e-3,                  # Convergence tolerance
        learning_rate='optimal',   # Adaptive learning rate
        eta0=0.01,                 # For better convergence
        alpha=0.001,               # Regularization strength
        random_state=42            # For reproducibility
    ))
])
model.fit(x_train, y_train)

# Uses the trained model to predict labels(y_predict) for the test dataset(x_test)
# Compares the predicted labels (y_predict) with the actual labels (y_test)
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print(f'{score * 100:.2f}% of samples were classified correctly!')
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
