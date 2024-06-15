# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time

data1_raw = pd.read_csv("D:\研究所課程\深度\DL_HW1\energy_efficiency_data.csv")
data1_raw

data1 = pd.get_dummies(data1_raw, columns=['Orientation', 'Glazing Area Distribution'])
data1
#####
