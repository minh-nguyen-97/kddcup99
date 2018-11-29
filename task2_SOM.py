import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-alpha",   type=float,     required=False,     default=0.8 ,   dest="alpha",       help="Defines the initial learningrate at iteration zero")
parser.add_argument("-data",    type=str,       required=True,      default=None,  dest="csvFile",     help="Defines the name of the data file (CSV file)")
parser.add_argument("-iter",    type=int,       required=False,     default=10 ,    dest="iter",        help="Defines the number oftraining iterations")
parser.add_argument("-sigma",   type=float,     required=False,     default=None,  dest="sigma",       help="Defines the std.dev. Of the gaussian neighborhood function")
parser.add_argument("-mapsize",   type=str,     required=False,     default="8,16", dest="mapsizeStr" ,   help="Defines the size of the(2D-)map of the SOM")

args = parser.parse_args()

alpha = args.alpha
csvFile = args.csvFile
iter = args.iter

mapsizeStr = args.mapsizeStr

mapsize = [int(x) for x in mapsizeStr.split(',')]

if (args.sigma == None):
    sigma = 0.3 * max(mapsize)
else:
    sigma = args.sigma

def accepted():
    if (alpha > 0 and csvFile != "" and iter >= 0 and sigma > 0 and mapsize[0] > 0 and mapsize[1] > 0):
        return True
    return False

if (not accepted()):
    print("One of argument values is not accepted")
    exit(0)

# alpha = 0.8
# csvFile = "task1_Preprocessing.csv"
# iter = 300
# mapsizeStr = "16,11"
#
# mapsize = [int(x) for x in mapsizeStr.split(',')]
# sigma = 40 #0.3 * max(mapsize)

kdd = pd.read_csv(csvFile, sep=',', header=None)

SAMPLE_SIZE = kdd.shape[0]
DATA_DIM = kdd.shape[1]

SOM_X_DIM = mapsize[0]
SOM_Y_DIM = mapsize[1]

MAPSIZE = SOM_X_DIM * SOM_Y_DIM

codebook = np.random.random([MAPSIZE, DATA_DIM])

r = np.empty([MAPSIZE, MAPSIZE])

for i in range(MAPSIZE):
    for j in range(MAPSIZE):
        r[i][j] = - ((i - j) * (i - j) / 2)

PI = np.pi

x = np.array(kdd)

totalTime = 0

for it in range(iter):

    sigmaT = sigma * np.exp((-it*it * PI * PI) / (iter*iter))

    alphaT = alpha * (iter - it) / iter

    quantErr = 0

    h = r / (sigmaT * sigmaT)

    h = alphaT * np.exp(h)

    start = time.time()

    for i in range(SAMPLE_SIZE):

        # competitive
        diff = x[i] - codebook

        dist = np.einsum('ij,ij->i', diff, diff)

        argmin = dist.argmin(axis=0)

        # cooperative
        delta = np.einsum('ij,i->ij', diff, h[argmin])

        codebook = codebook + delta

    for i in range(SAMPLE_SIZE):
        # competitive
        diff = x[i] - codebook

        dist = np.einsum('ij,ij->i', diff, diff)

        argmin = dist.argmin(axis=0)

        quantDiff = x[i] - codebook[argmin]

        quantDist = np.einsum('i,i->', quantDiff, quantDiff)

        quantErr += np.sqrt(quantDist)


    end = time.time()

    timeTaken = end - start

    totalTime += timeTaken

    quantErr /= SAMPLE_SIZE

    print("Iter ", it, ": ", timeTaken, " ", quantErr)


print("Total time: ", totalTime)

# write to mappings.dat
f = open("mappings.dat", "w")

quantSum = 0

def toX(coor):
    return coor // SOM_Y_DIM

def toY(coor):
    return coor % SOM_Y_DIM

for i in range(SAMPLE_SIZE):
    # competitive
    diff = x[i] - codebook

    dist = np.einsum('ij,ij->i', diff, diff)

    argmin = dist.argmin(axis=0)

    quantDiff = x[i] - codebook[argmin]

    quantDist = np.einsum('i,i->', quantDiff, quantDiff)

    quantErr = np.sqrt(quantDist)

    f.write(str(toX(argmin)) + " " + str(toY(argmin)) + " " + str(quantErr) + "\n")

    quantSum += quantErr

quantMean = quantSum / SAMPLE_SIZE

print("Mean quantization error: ",quantMean)

f = open("codebook.txt", "w")

for i in range(MAPSIZE):
    fc = ""
    for j in range(DATA_DIM):
        fc += str(codebook[i][j])
        if (j < DATA_DIM-1):
            fc += ","
    fc += "\n"
    f.write(fc)
