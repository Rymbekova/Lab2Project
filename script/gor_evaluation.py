from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef, classification_report
import os
import pandas as pd
from scipy.stats import sem
import statistics

def my_MCC(test_true,test_pred, ss):
    MCC = dict.fromkeys(ss, 0)
    for structure in ss:
        test_true_temp = [structure if i == structure else 'X' for i in test_true]
        test_pred_temp = [structure if i == structure else 'X' for i in test_pred]
        MCC[structure] = matthews_corrcoef(test_true_temp, test_pred_temp)
    return(MCC)

def generate_df(ss):
    columns = ['H', '-', 'E'] 
    iterables = [['set0','set1', 'set2', 'set3', 'set4', 'test'], ['MCC']]
    my_index = pd.MultiIndex.from_product(iterables, names=['data', 'score'])
    df = pd.DataFrame(index= my_index, columns=columns )
    return(df)

ss = ['H', '-', 'E']
df = generate_df(ss)


Q3_list_cv = []
for i in range(1,5):
    test_true = list(open('/home/um36/lab2/evaluation/set' + str(i) + '_true.txt').read())
    test_pred = list(open('/home/um36/lab2/evaluation/set' + str(i) + '_pred.txt').read())
    MCC = my_MCC(test_true, test_pred, ss)
    report = classification_report(test_true, test_pred, labels=ss, output_dict=True)
    Q3 = accuracy_score(test_true, test_pred)
    Q3_list_cv.append(Q3)
    for s in ss:
         df.loc['set'+str(i), 'MCC'][s] = MCC[s]
Q3_set = statistics.mean(Q3_list_cv)


test_true = list(open('/home/um36/lab2/evaluation/test_true.txt').read())
test_pred = list(open('/home/um36/lab2/evaluation/test_pred.txt').read())
MCC = my_MCC(test_true, test_pred, ss) 
report = classification_report(test_true, test_pred, labels=ss, output_dict=True) 
Q3_test = accuracy_score(test_true, test_pred)
for s in ss:
    df.loc['test', 'MCC'][s] = MCC[s]
print(df)
print("Q3_cv:", Q3_set)
print("Q3_test:", Q3_test)
