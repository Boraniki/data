import os
import sys
import numpy as np
import warnings
# Ignore warnings
warnings.filterwarnings('ignore')
from PIL import Image
from pathlib import Path
import pandas as pd

base_dir = Path(__file__).resolve().parent

cars_path = base_dir / "cars/"
horse_path = base_dir / "horses/"

cars = os.listdir(cars_path)
horses = os.listdir(horse_path)

train_size = 0.9

def img2vector(path, image_name):
    image = Image.open(os.path.join(path, image_name))
    image = image.resize((50,50))
    image = image.convert('L')
    vector = np.array(image)
    vector = vector.flatten()
    return vector

car_vectors = np.zeros((len(cars), 50*50))
for i in range(0, len(cars)):
    car_vectors[i] = img2vector(cars_path, cars[i])

horse_vectors = np.zeros((len(horses), 50*50))
for i in range(0, len(horses)):
    horse_vectors[i] = img2vector(horse_path, horses[i])

car_df = pd.DataFrame(car_vectors)
horse_df = pd.DataFrame(horse_vectors)

car_df['label'] = 1
horse_df['label'] = -1

car_train = car_df.sample(frac=train_size, random_state=200)
car_test = car_df.drop(car_train.index)

horse_train = horse_df.sample(frac=train_size, random_state=200)
horse_test = horse_df.drop(horse_train.index)

train_df = pd.concat([car_train, horse_train])
test_df = pd.concat([car_test, horse_test])

train_x = train_df.drop('label', axis=1)
train_y = train_df['label']

test_x = test_df.drop('label', axis=1)
test_y = test_df['label']

def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y)**2))

def knn(x, y, k):
    distances = []
    for i in range(0, len(x)):
        distances.append(euclidean_distance(x.iloc[i], y))
    distances = np.array(distances)
    sorted_indices = np.argsort(distances)
    sorted_indices = sorted_indices[:k]
    return sorted_indices

def predict(x, y, k):
    indices = knn(x, y, k)
    labels = train_y.iloc[indices]
    return labels.mode()[0]

def accuracy(x, y, k):
    correct = 0
    for i in range(0, len(x)):
        prediction = predict(train_x, x.iloc[i], k)
        if prediction == y.iloc[i]:
            correct += 1
    return correct / len(x)

accuracy(train_x, train_y, 3)
accuracy(test_x, test_y, 3)

#k loop and plot all accuracies and k

import matplotlib.pyplot as plt

k_list = []
accuracy_list = []

for k in range(1, 30,2):
    accuracy(test_x, test_y, k)
    print(accuracy)
    k_list.append(k)
    accuracy_list.append(accuracy(test_x, test_y, k))


plt.plot(k_list, accuracy_list)
plt.xlabel('k')
plt.ylabel('accuracy')
plt.show()
