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

import random
        

def random_sample():
    pass


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

df = pd.read_csv('data/iris0.csv', ',')
df_raw = df.iloc[:, :-1]
Y = df.iloc[:, -1]

txns_train = TransactionDB.from_DataFrame(df)

rules = run_fim_apriori(df_raw, 0.35)

list_of_rules = createrules(rules, list(set(Y)))
print(len(list_of_rules), "rules created")

prepare_caches(list_of_rules, df, Y)
prepare_overlap(list_of_rules, df)
print("caches prepared")

def optimize(param):
    lambda_array = [1.0]*7     # use separate hyperparamter search routine
    #lambda_array = [1, 0, 0, 1, 0, 0, 0]
    
    s1 = smooth_local_search(list_of_rules, df, Y, lambda_array, 0.33, param)
    f1 = func_evaluation(s1, list_of_rules, df, Y, lambda_array, len(list_of_rules))

    return f1, s1

def f(x):
    return x * x

if __name__ == "__main__":
   

    print("---------------------")
    print("Training CBA model")
    print("---------------------")

    cba = CBA(support=0.35, confidence=0, algorithm="m1")
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

    result_obj = 0
    result_set = {}
    if f1 > f2:
        print("The Solution Set is: "+str(s1))
        result_set = list(s1)
        result_obj = f1
    else:
        print("The Solution Set is: "+str(s2))
        result_set = list(s2)
        result_obj = f1




    np_rules = np.array(list_of_rules)
    solution_rules = np_rules[result_set]

    n = len(list_of_rules)
    random_sample_soln_set = random.sample(range(0, n), len(result_set))
    random_sample_objective = func_evaluation(random_sample_soln_set, list_of_rules, df, Y, [1.0]*7, n)

    empty_obj = func_evaluation({}, list_of_rules, df, Y, [1.0]*7, n)

    print("---------------------")
    print("Objective with empty set")
    print(empty_obj)
    print("----------------------")


    print("---------------------")
    print("IDS rules")
    print("---------------------")
    print("objective_value", result_obj)
    print("scaled objective value", result_obj - empty_obj)
    print("---------------------")
    for r in solution_rules:
        r.print_rule()
    
    print("---------------------")
    print("CBA rules")
    print("---------------------")
    for r in cba.clf.rules:
        print(r)

    print("---------------------")
    print("Random sampled rules")
    print("---------------------")
    print("objective_value:", random_sample_objective)
    print("scaled objective value", random_sample_objective - empty_obj)
    print("---------------------")
    for rule in np_rules[random_sample_soln_set]:
        rule.print_rule()
    
    



    print("----------------------")



    pred = predict(result_set, list_of_rules, df, Y)
    print("IDS accuracy", accuracy_score(pred, Y))
    print("CBA accuracy", cba_accuracy)


    print("---------------------")
    print("Number of rules:")
    print("Total:", len(list_of_rules))
    print("CBA:", len(cba.clf.rules))
    print("IDS smooth:", len(solution_rules))
    print("Random:", len(random_sample_soln_set))
    print("---------------------")


    print()
    print("---------------------")
    print("Random test")

    for i in range(len(list_of_rules)):
        rand_soln = random.sample(range(len(list_of_rules)), i)
        obj_val = func_evaluation(rand_soln, list_of_rules, df, Y, [1.0]*7, n)
        print("{} rules sampled, objective value: {}, scaled obj value: {}".format(i, obj_val, obj_val - empty_obj))

    print("---------------------")


