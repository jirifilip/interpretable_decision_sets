import pandas as pd
import numpy as np


from multiprocessing import Pool
from rule_miner import run_fim_apriori
from IDS_smooth_local import (
    smooth_local_search,
    func_evaluation,
    createrules,
    run_apriori,
    prepare_caches,
    prepare_overlap
)
from rules import predict

from sklearn.metrics import accuracy_score

from pyarc import TransactionDB
from pyarc.algorithms import (
    top_rules,
    createCARs,
    M1Algorithm
)
from pyarc import CBA, TransactionDB
        


"""
lambda_array = [1.0]*7     # use separate hyperparamter search routine
s1 = smooth_local_search(list_of_rules, df, Y, lambda_array, 0.33, 0.33)
s2 = smooth_local_search(list_of_rules, df, Y, lambda_array, 0.33, -1.0)
f1 = func_evaluation(s1, list_of_rules, df, Y, lambda_array)
f2 = func_evaluation(s2, list_of_rules, df, Y, lambda_array)

result_set = {}
if f1 > f2:
    print("The Solution Set is: "+str(s1))
    result_set = list(s1)
else:
    print("The Solution Set is: "+str(s2))
    result_set = list(s2)
"""

df = pd.read_csv('data/segment0.csv', ',')
df_raw = df.iloc[:, :-1]
Y = df.iloc[:, -1]

txns_train = TransactionDB.from_DataFrame(df)

rules = run_fim_apriori(df_raw, 0.8)
list_of_rules = createrules(rules, list(set(Y)))

prepare_caches(list_of_rules, df, Y)
prepare_overlap(list_of_rules, df)

def optimize(param):
    lambda_array = [1.0]*7     # use separate hyperparamter search routine
    s1 = smooth_local_search(list_of_rules, df, Y, lambda_array, 0.33, param)
    f1 = func_evaluation(s1, list_of_rules, df, Y, lambda_array, len(list_of_rules))

    return f1, s1

def f(x):
    return x * x

if __name__ == "__main__":
   

    print("---------------------")
    print("Training CBA model")
    print("---------------------")

    cba = CBA(support=0.55, confidence=0, algorithm="m1")
    cba.fit(txns_train)

    cba_accuracy = cba.rule_model_accuracy(txns_train)

    # precompute values for cover and correct cover
    

    print("---------------------")
    print("All mined rules for IDS")
    print("---------------------")
    for r in list_of_rules:
        r.print_rule()

    print("---------------------")
    print("Starting SLS")
    print("---------------------")
    

    

    process_pool = Pool(2)
    results = process_pool.map(optimize, [0.33, -1.0])

    #results = process_pool.map(f, [2, 3])

    (f1, s1), (f2, s2) = results 

    result_set = {}
    if f1 > f2:
        print("The Solution Set is: "+str(s1))
        result_set = list(s1)
    else:
        print("The Solution Set is: "+str(s2))
        result_set = list(s2)




    np_rules = np.array(list_of_rules)
    solution_rules = np_rules[result_set]

    print("---------------------")
    print("IDS rules")
    print("---------------------")
    for r in solution_rules:
        r.print_rule()
    
    print("---------------------")
    print("CBA rules")
    print("---------------------")
    for r in cba.clf.rules:
        print(r)

    print("----------------------")

    pred = predict(result_set, list_of_rules, df, Y)
    print("IDS accuracy", accuracy_score(pred, Y))
    print("CBA accuracy", cba_accuracy)


