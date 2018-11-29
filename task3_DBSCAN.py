import pandas as pd
import numpy as np

mapping = pd.read_csv("mappings.dat", sep=" ", header=None)

mapping = mapping.drop(columns=[2])

points = np.unique(mapping, axis=0)

eps = 1
minPts = 5

def dist(i,j):
    return np.sqrt((points[i, 0]-points[j, 0]) ** 2 + (points[i, 1] - points[j, 1]) ** 2)

def densityOf(i):
    density = 0
    for j in range(len(points)):
        if (dist(i, j) <= eps):
            density += 1
    return density


pointsType = np.zeros([len(points)])

# Find core points
for i in range(len(points)):

    density = densityOf(i)

    if (density >= minPts):
        pointsType[i] = 1

def isNeighbor(i):
    for j in range(len(points)):
        if (dist(i,j) <= eps and pointsType[j] == 1):
            return True
    return False


# Find border points
for i in range(len(points)):
    density = densityOf(i)

    if (density < minPts and isNeighbor(i)):
        pointsType[i] = 2


# Find label for cluster
current_label = 0
label = np.empty([len(points)])
label[:] = np.nan

def DFS(i, current_label):
    label[i] = current_label

    if (pointsType[i] == 2):
        return

    for j in range(len(points)):
        if (dist(i,j) <= eps and pointsType[j] > 0 and np.isnan(label[j])):
            DFS(j,current_label)

for i in range(len(points)):
    if (pointsType[i] == 1 and np.isnan(label[i])):
        current_label += 1
        DFS(i,current_label)


# #############################################################################
# Plot result
import matplotlib.pyplot as plt

colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, current_label)]

fig, ax = plt.subplots()

for k in range(current_label):
    core = []
    border = []
    count = 0
    for i in range(len(points)):
        if ((np.isnan(label[i]) == False) and (int(label[i]) == k + 1)):
            if (pointsType[i] == 1):
                count += 1
                core.append(points[i])
            else:
                border.append(points[i])

    # print(count)

    core = pd.DataFrame(core)
    border = pd.DataFrame(border)
    # print(core)

    core = core.values
    border = border.values

    if (len(core) > 0):
        ax.plot(core[:,0], core[:,1], 'o', markerfacecolor=tuple(colors[k]),
                        markeredgecolor='k', markersize=14, label="cores of cls "+ str(k+1))

    if (len(border) > 0):
        ax.plot(border[:,0], border[:,1], 'o', markerfacecolor=tuple(colors[k]),
                    markeredgecolor='k', markersize=6, label="borders of cls "+ str(k+1))

noise = []
for i in range(len(points)):
    if (np.isnan(label[i]) == True):
        if (pointsType[i] == 0):
            noise.append(points[i])
noise = pd.DataFrame(noise)
noise = noise.values

if (len(noise) > 0):
    ax.plot(noise[:,0], noise[:,1], 'o', markerfacecolor=tuple([0,0,0,1]),
                        markeredgecolor='k', markersize=6, label="noise")


# Shrink current axis's height by 10% on the bottom
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=2)

plt.title("Estimated number of clusters: " + str(current_label))
plt.show()





