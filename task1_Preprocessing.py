import pandas as pd

kdd99 = pd.read_csv("kddcup.data_10_percent.", sep=',', header=None)

kdd99.columns=['duration','protocol_type','service','flag','src_bytes','dst _bytes','land','wrong_fragment','urgent','hot','num_failed_logins','logged_ in','num_compromised','root_shell','su_attempted','num_root','num_file_crea tions','num_shells','num_access_files','num_outbound_cmds','is_host_login', 'is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerro r_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_ra te','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_hos t_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate ','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate', 'dst_host_srv_rerror_rate','attack']

# drop columns that only have value 0
newData = kdd99.drop(columns=['is_host_login', 'num_outbound_cmds'])

# one hot encoding with symbolic values
symbolics = ['protocol_type', 'service','flag']

for symbolic in symbolics:
    newData = newData.join(pd.get_dummies(kdd99[symbolic], prefix=symbolic))
    newData = newData.drop(columns=[symbolic])


# remove strong correlation
def strongCorr(newData):

    strongCorr = []
    corr = newData.corr()
    for i in range(corr.shape[0]):
        # print (corr.columns[i])
        for j in range(i+1, corr.shape[1]):
            if (abs(corr.values[i,j]) > 0.9):
                # print ("\t" + corr.columns[j])
                strongCorr.append(corr.columns[j])

    strongCorr = list(set(strongCorr))

    for i in range(len(strongCorr)):
        newData = newData.drop(columns=[strongCorr[i]])

    return newData

newData = strongCorr(newData)


SAMPLE_SIZE = int(100000)
HALF_SAMPLE_SIZE = int(SAMPLE_SIZE/2)

normal = newData[newData.attack == "normal."].head(HALF_SAMPLE_SIZE)

attack = newData[newData.attack != "normal."].head(HALF_SAMPLE_SIZE)

newData = pd.concat([normal, attack])

# print(newData)

# drop column 'attack'
newData = newData.drop(columns=['attack'])

newData.to_csv("task1_Preprocessing.csv", index=False, header=None)
