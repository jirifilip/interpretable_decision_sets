import pandas as pd
import numpy as np
from rule_miner import run_fim_apriori
from IDS_smooth_local import (
    smooth_local_search,
    func_evaluation,
    createrules,
    run_apriori
)


df = pd.read_csv('data/iris_train.tab',' ')
df1 = pd.read_csv('data/iris0.csv', ',')
Y = list(df1['class'].values)




itemsets = run_fim_apriori(df, 0.8)
print("----------list of rules------------")
list_of_rules = createrules(itemsets, list(set(Y)))
print("----------------------")
for r in list_of_rules:
    r.print_rule()

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



np_rules = np.array(list_of_rules)
solution_rules = np_rules[result_set]

list(map(lambda r: r.print_rule(), solution_rules))