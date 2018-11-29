import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

DATA_DIM = 99
SOM_X_DIM = 16
SOM_Y_DIM = 11

MAPSIZE = SOM_X_DIM * SOM_Y_DIM


codebook2D = pd.read_csv("codebook.txt", sep=",", header=None)

codebook2D = codebook2D.values

# print(codebook2D)

def toX(coor):
    return coor // SOM_Y_DIM

def toY(coor):
    return coor % SOM_Y_DIM

codebook = np.random.random([SOM_X_DIM, SOM_Y_DIM, DATA_DIM])

for i in range(MAPSIZE):
    codebook[toX(i)][toY(i)] = codebook2D[i]

umat_X_DIM = SOM_X_DIM * 2 - 1
umat_Y_DIM = SOM_Y_DIM * 2 - 1
umat = np.empty([umat_X_DIM, umat_Y_DIM])
umat[:] = np.nan
# print(umat.shape)


def dist(x, y):
    res = 0
    for i in range(DATA_DIM):
        tmp = x[i] - y[i]
        res += tmp * tmp
    return np.sqrt(res)

for i in range(SOM_X_DIM):
    for j in range(SOM_Y_DIM-1):
        umat[2*i, 2*j+1] = dist(codebook[i][j], codebook[i][j+1])

for j in range(SOM_Y_DIM):
    for i in range(SOM_X_DIM-1):
            umat[2*i+1, 2*j] = dist(codebook[i][j], codebook[i+1][j])


# for i in range(SOM_X_DIM):
#     s = ""
#     for j in range(SOM_Y_DIM):
#         s += str(umat[i][j]) + " "
#     print(s)


dir = [(0,1), (1,0), (-1, 0), (0, -1)]

def inBound(x, y):
    return (0 <= x and x < umat_X_DIM and 0 <= y and y < umat_Y_DIM)

def avgNeighbors(u, v):

    sum = 0
    num = 0
    for i in range(len(dir)):
        x = u + dir[i][0]
        y = v + dir[i][1]
        # print (str(u) + " " + str(v) + " " + str(x) + " " + str(y))
        if (inBound(x,y) and np.isnan(umat[x][y]) == False):
            sum += umat[x][y]
            num += 1

    return sum / num

for i in range(umat_X_DIM):
    for j in range(umat_Y_DIM):
        if (i % 2 == 0 and j % 2 == 0):
            umat[i][j] = avgNeighbors(i,j)
            if (umat[i][j] == 0):
                print(str(i) + " " + str(j) )


fig, ax = plt.subplots()
im = plt.imshow(umat.T, origin="lower")
cbar = plt.colorbar(im, ax=ax)
plt.yticks(np.arange(0,21,step=2))
plt.show()
