import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, TimeDistributed, Flatten
from tensorflow.keras import Input
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample


def preprocess_binary_classification(normal_dir, abnormal_dir, balance_method='oversample'):
    normal_data = np.loadtxt(normal_dir, delimiter=',')
    abnormal_data = np.loadtxt(abnormal_dir, delimiter=',')

    normal_features = normal_data[:, :-1]
    abnormal_features = abnormal_data[:, :-1]

    normal_labels = normal_data[:, -1]
    abnormal_labels = abnormal_data[:, -1]

    x = np.vstack((normal_features, abnormal_features))
    y = np.array([0] * len(normal_labels) + [1] * len(abnormal_labels))

    n_normal, n_abnormal = len(normal_labels), len(abnormal_labels)

    if n_normal != n_abnormal:
        if balance_method == 'oversample':
            if n_normal < n_abnormal:
                normal_features = resample(normal_features, replace=True, n_samples=n_abnormal, random_state=42)
                normal_labels = np.array([0] * n_abnormal)
            else:
                abnormal_features = resample(abnormal_features, replace=True, n_samples=n_normal, random_state=42)
                abnormal_labels = np.array([1] * n_normal)

        elif balance_method == 'undersample':
            if n_normal > n_abnormal:
                normal_features = resample(normal_features, replace=False, n_samples=n_abnormal, random_state=42)
                normal_labels = np.array([0] * n_abnormal)
            else:
                abnormal_features = resample(abnormal_features, replace=False, n_samples=n_normal, random_state=42)
                abnormal_labels = np.array([1] * n_normal)

    x = np.vstack((normal_features, abnormal_features))
    y = np.concatenate((normal_labels, abnormal_labels))

    return x, y


def build_lstm_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(64)(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax' if num_classes > 1 else 'sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)
    loss = 'categorical_crossentropy' if num_classes > 1 else 'binary_crossentropy'
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

    return model


# Load data for binary classification
normal_dir = 'ptbdb_normal.csv'
abnormal_dir = 'ptbdb_abnormal.csv'
x_binary, y_binary = preprocess_binary_classification(normal_dir, abnormal_dir)

# Split data
x_train_bin, x_test_bin, y_train_bin, y_test_bin = train_test_split(
    x_binary, y_binary, test_size=0.2, random_state=42
)
x_train_bin = x_train_bin[..., np.newaxis]

x_test_bin = x_test_bin[..., np.newaxis]

# Build model
binary_model = build_lstm_model(input_shape=(x_train_bin.shape[1], 1), num_classes=1)


# Train model
class_weights = {0: 1, 1: len(y_binary) / sum(y_binary)}
binary_model.fit(x_train_bin, y_train_bin, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate model
binary_preds = binary_model.predict(x_test_bin)
print(classification_report(y_test_bin, (binary_preds > 0.5).astype(int)))

# --- Save Results and Models ---
binary_model.save('binary_classification_model.keras')

# Save classification reports
with open('binary_classification_report.txt', 'w') as f:
    f.write(classification_report(y_test_bin, (binary_preds > 0.5).astype(int)))


# ---------------------------------------------------------------------


def preprocess_multiclass_classification(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    x_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values

    x_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values

    smote = SMOTE(random_state=42)
    x_train_res, y_train_res = smote.fit_resample(x_train, y_train)

    return x_train_res, y_train_res, x_test, y_test


# Load data for multiclass classification
train_path = 'mitbih_train.csv'
test_path = 'mitbih_test.csv'
x_train_multi, y_train_multi, x_test_multi, y_test_multi = preprocess_multiclass_classification(train_path, test_path)

# One-hot encode labels
y_train_multi = tf.keras.utils.to_categorical(y_train_multi)
y_test_multi = tf.keras.utils.to_categorical(y_test_multi)

# Build model
multiclass_model = build_lstm_model(input_shape=(x_train_multi.shape[1], 1), num_classes=y_train_multi.shape[1])

# Train model
x_train_multi = x_train_multi[..., np.newaxis]
x_test_multi = x_test_multi[..., np.newaxis]
multiclass_model.fit(x_train_multi, y_train_multi, epochs=10, batch_size=128, validation_split=0.2)

# Evaluate model
multiclass_preds = multiclass_model.predict(x_test_multi)
print(classification_report(y_test_multi.argmax(axis=1), multiclass_preds.argmax(axis=1)))

# --- Save Results and Models ---
multiclass_model.save('multiclass_classification_model.keras')

with open('multiclass_classification_report.txt', 'w') as f:
    f.write(classification_report(y_test_multi.argmax(axis=1), multiclass_preds.argmax(axis=1)))

print("Training and evaluation completed.")
