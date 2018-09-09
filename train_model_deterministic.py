import pandas as pd

from IDS_deterministic_local import (
    run_fim_apriori,
    createrules,
    deterministic_local_search
)

df = pd.read_csv('data/iris_train.tab')
df1 = pd.read_csv('data/iris0.csv')
Y = list(df1['class'].values)


itemsets = run_fim_apriori(df, 0.1)
print("-----------\nrules mined\n-----------")
list_of_rules = createrules(itemsets, list(set(Y)))
print("-----------\nrules created\n-----------")

print("----------------------")
"""
for r in list_of_rules:
    r.print_rule()
"""


lambda_array = [0.5]*7     # use separate hyperparamter search routine
epsilon = 0.05
soln_set, obj_val = deterministic_local_search(list_of_rules, df, Y, lambda_array, epsilon)
print(soln_set)
print(obj_val)
