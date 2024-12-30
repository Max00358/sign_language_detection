# This file is used to train classifier model using gesture landscape data
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

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

model = RandomForestClassifier()
model.fit(x_train, y_train)

# Uses the trained model to predict labels(y_predict) for the test dataset(x_test)
y_predict = model.predict(x_test)
# Compares the predicted labels (y_predict) with the actual labels (y_test)
score = accuracy_score(y_predict, y_test)

print(f'{score * 100}% of samples were classified correctly!')

# Saves the trained model to a pickle file (model.p) for later use:
# This allows you to load the model in another script at a later time w/o retraining it
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
