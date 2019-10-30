import pandas as pd
import numpy as np

train = pd.read_csv("../CF/train.csv")
test = pd.read_csv("../CF/test.csv")

train["bin_3"][train["bin_3"] == "T"] = 1
train["bin_3"][train["bin_3"] == "F"] = 0
train["bin_4"][train["bin_4"] == "Y"] = 1
train["bin_4"][train["bin_4"] == "N"] = 0

train["R"] = 0
train["G"] = 0
train["B"] = 0
for i in range(len(train)):
	if train["nom_0"][i] == "Red":
		train["R"][i] = 1
	elif train["nom_0"][i] == "Green":
		train["G"][i] = 1
	else :
		train["B"][i] = 1



train = train.drop("nom_0", axis = 1)

#pd.Series.value_counts(train["nom_1"])
train["nom_1"][train["nom_1"] == "Trapezoid"] = 0
train["nom_1"][train["nom_1"] == "Square"] = 1
train["nom_1"][train["nom_1"] == "Star"] = 2
train["nom_1"][train["nom_1"] == "Circle"] = 3
train["nom_1"][train["nom_1"] == "Polygon"] = 4
train["nom_1"][train["nom_1"] == "Triangle"] = 5

train["nom_2"][train["nom_2"] == "Lion"] = 0
train["nom_2"][train["nom_2"] == "Cat"] = 1
train["nom_2"][train["nom_2"] == "Snake"] = 2
train["nom_2"][train["nom_2"] == "Dog"] = 3
train["nom_2"][train["nom_2"] == "Axolotl"] = 4
train["nom_2"][train["nom_2"] == "Hamster"] = 5

train.head()
