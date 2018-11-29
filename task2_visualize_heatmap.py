import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SAMPLE_SIZE = int(100000)
HALF_SAMPLE_SIZE = int(SAMPLE_SIZE / 2)

kdd99 = pd.read_csv("kddcup.data_10_percent.", sep=',', header=None)

kdd99.columns=['duration','protocol_type','service','flag','src_bytes','dst _bytes','land','wrong_fragment','urgent','hot','num_failed_logins','logged_ in','num_compromised','root_shell','su_attempted','num_root','num_file_crea tions','num_shells','num_access_files','num_outbound_cmds','is_host_login', 'is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerro r_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_ra te','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_hos t_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate ','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate', 'dst_host_srv_rerror_rate','attack']

attack = kdd99.attack

attack = attack.apply(lambda x: 1 if x == 'normal.' else 0) # normal == 1 else == 0

trainNormal = attack[attack == 1][:HALF_SAMPLE_SIZE]

trainAttack = attack[attack == 0][:HALF_SAMPLE_SIZE]

trainSet = pd.concat([trainNormal, trainAttack])

trainSet = np.array(trainSet)

classes = pd.DataFrame(trainSet)

# read mapping

mapping = pd.read_csv("mappings.dat", sep=" ", header=None)

mapping = mapping.drop(columns=[2])

data = pd.concat([mapping, classes], axis=1)

# print(data)

# vals = np.unique(data, axis=0)
#
# positives=vals[vals[:,2] == 1][:,0:2] #select samples from class 1 w/o label
# negatives=vals[vals[:,2] == 0][:,0:2] #select samples from class 0 w/o label
#
# fig, ax = plt.subplots()
# ax.plot(positives[:,0], positives[:,1], "k+")
# ax.plot(negatives[:,0], negatives[:,1], "rx")
# plt.plot(markers=["k+","rx"], labels=["Normal","Attack"])
# plt.xticks(np.arange(0,16,step=1))
# plt.yticks(np.arange(0,11,step=1))
# plt.show()

xy = data.values[:,0:2]

# print(xy)

xdim = int(xy[:,0].max() + 1)
ydim = int(xy[:,1].max() + 1)
matrix = np.zeros([ydim,xdim])

for i in range(data.shape[0]):
    matrix[ydim-xy[i,1]-1][xy[i,0]] += 1

#plot the heatmap
fig, ax = plt.subplots()
im = plt.imshow(matrix, origin="lower")
cbar = plt.colorbar(im, ax=ax)
plt.xticks(np.arange(0,16,step=1))
plt.yticks(np.arange(0,11,step=1))
plt.show()