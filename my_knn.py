import os
import sys
import numpy as np
import warnings
# Ignore warnings
warnings.filterwarnings('ignore')
from PIL import Image


cars_dirs = "C:/Users/Boraniki/Desktop/dataset/cars/"
horse_dirs = "C:/Users/Boraniki/Desktop/dataset/horses/"

cars = os.listdir(cars_dirs)
horse = os.listdir(horse_dirs)

cars_path = cars_dirs + cars[0]

print(cars_path)

flower_img = Image.open(cars_path)
flower_img.show()
