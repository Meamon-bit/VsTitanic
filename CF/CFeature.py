import pandas as pd
import numpy as np
import collections
from sklearn import tree

train = pd.read_csv("../CF/train.csv")
test = pd.read_csv("../CF/test.csv")

train.bin_3 = train.bin_3.replace("T",1)
train.bin_3 = train.bin_3.replace("F",0)
train.bin_4 = train.bin_4.replace("Y",1)
train.bin_4 = train.bin_4.replace("N",0)

train["R"] = 0
train["G"] = 0
train["B"] = 0
train.loc[train.nom_0 == "Red", "R"] = 1
train.loc[train.nom_0 == "Green", "G"] = 1
train.loc[train.nom_0 == "Blue", "B"] = 1
train = train.drop("nom_0", axis = 1)

pd.Series.value_counts(train["nom_1"])
train.nom_1 = train.nom_1.replace("Trapezoid",0)
train.nom_1 = train.nom_1.replace("Square",1)
train.nom_1 = train.nom_1.replace("Star",2)
train.nom_1 = train.nom_1.replace("Circle",3)
train.nom_1 = train.nom_1.replace("Polygon",4)
train.nom_1 = train.nom_1.replace("Triangle",5)

train.nom_2 = train.nom_2.replace("Lion",0)
train.nom_2 = train.nom_2.replace("Cat",1)
train.nom_2 = train.nom_2.replace("Snake",2)
train.nom_2 = train.nom_2.replace("Dog",3)
train.nom_2 = train.nom_2.replace("Axolotl",4)
train.nom_2 = train.nom_2.replace("Hamster",5)

train.nom_3 = train.nom_3.replace("Russia",0)
train.nom_3 = train.nom_3.replace("Canada",1)
train.nom_3 = train.nom_3.replace("China",2)
train.nom_3 = train.nom_3.replace("Finland",3)
train.nom_3 = train.nom_3.replace("Costa Rica",4)
train.nom_3 = train.nom_3.replace("India",5)

train.nom_4 = train.nom_4.replace("Oboe", 0)
train.nom_4 = train.nom_4.replace("Piano", 1)
train.nom_4 = train.nom_4.replace("Bassoon", 2)
train.nom_4 = train.nom_4.replace("Theremin", 3)

train.nom_5 = train.nom_5.str.upper()
train.nom_5 = train.nom_5.map(lambda x: int(x,16))

train.nom_6 = train.nom_6.str.upper()
train.nom_6 = train.nom_6.map(lambda x: int(x,16))

train.nom_7 = train.nom_7.str.upper()
train.nom_7 = train.nom_7.map(lambda x: int(x,16))

train.nom_8 = train.nom_8.str.upper()
train.nom_8 = train.nom_8.map(lambda x: int(x,16))

train.nom_9 = train.nom_9.str.upper()
train.nom_9 = train.nom_9.map(lambda x: int(x,16))

train.ord_1 = train.ord_1.replace("Novice", 0)
train.ord_1 = train.ord_1.replace("Contributor", 1)
train.ord_1 = train.ord_1.replace("Expert", 2)
train.ord_1 = train.ord_1.replace("Master", 3)
train.ord_1 = train.ord_1.replace("Grandmaster", 4)

train.ord_2 = train.ord_2.replace("Freezing", 0)
train.ord_2 = train.ord_2.replace("Cold", 1)
train.ord_2 = train.ord_2.replace("Warm", 2)
train.ord_2 = train.ord_2.replace("Hot", 3)
train.ord_2 = train.ord_2.replace("Boiling Hot", 4)
train.ord_2 = train.ord_2.replace("Lava Hot", 5)

train.ord_3 = train.ord_3.replace("a", 0)
train.ord_3 = train.ord_3.replace("b", 1)
train.ord_3 = train.ord_3.replace("c", 2)
train.ord_3 = train.ord_3.replace("d", 3)
train.ord_3 = train.ord_3.replace("e", 4)
train.ord_3 = train.ord_3.replace("f", 5)
train.ord_3 = train.ord_3.replace("g", 6)
train.ord_3 = train.ord_3.replace("h", 7)
train.ord_3 = train.ord_3.replace("i", 8)
train.ord_3 = train.ord_3.replace("j", 9)
train.ord_3 = train.ord_3.replace("k", 10)
train.ord_3 = train.ord_3.replace("l", 11)
train.ord_3 = train.ord_3.replace("m", 12)
train.ord_3 = train.ord_3.replace("n", 13)
train.ord_3 = train.ord_3.replace("o", 14)

train.ord_4 = train.ord_4.replace("A", 0)
train.ord_4 = train.ord_4.replace("B", 1)
train.ord_4 = train.ord_4.replace("C", 2)
train.ord_4 = train.ord_4.replace("D", 3)
train.ord_4 = train.ord_4.replace("E", 4)
train.ord_4 = train.ord_4.replace("F", 5)
train.ord_4 = train.ord_4.replace("G", 6)
train.ord_4 = train.ord_4.replace("H", 7)
train.ord_4 = train.ord_4.replace("I", 8)
train.ord_4 = train.ord_4.replace("J", 9)
train.ord_4 = train.ord_4.replace("K", 10)
train.ord_4 = train.ord_4.replace("L", 11)
train.ord_4 = train.ord_4.replace("M", 12)
train.ord_4 = train.ord_4.replace("N", 13)
train.ord_4 = train.ord_4.replace("O", 14)
train.ord_4 = train.ord_4.replace("P", 15)
train.ord_4 = train.ord_4.replace("Q", 16)
train.ord_4 = train.ord_4.replace("R", 17)
train.ord_4 = train.ord_4.replace("S", 18)
train.ord_4 = train.ord_4.replace("T", 19)
train.ord_4 = train.ord_4.replace("U", 20)
train.ord_4 = train.ord_4.replace("V", 21)
train.ord_4 = train.ord_4.replace("W", 22)
train.ord_4 = train.ord_4.replace("X", 23)
train.ord_4 = train.ord_4.replace("Y", 24)
train.ord_4 = train.ord_4.replace("Z", 25)

counter = collections.Counter(train.ord_5.values)
count_dict = dict(counter.most_common())
train.ord_5 = train.ord_5.map(lambda x: count_dict[x]).values
print(train)



