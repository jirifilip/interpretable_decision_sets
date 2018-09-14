from sklearn.metrics import classification_report, f1_score
from scipy.stats import mode

dummy_label = "N/A"

def predict(soln_set, list_rules, df, Y):
   
    y_pred_dict = {}
    for index in soln_set:
        rule = list_rules[index]

        y_pred_per_rule = list(rule.get_y_pred_per_rule(df, Y))
        print('rule {} score:'.format(index))
        print(classification_report(Y, y_pred_per_rule))
        rule_f1_score = f1_score(Y, y_pred_per_rule, average='micro')
        y_pred_dict.update({rule_f1_score: y_pred_per_rule})

    y_pred = []
    top_f1_score = sorted(y_pred_dict.keys(), reverse=True)[0]

    for subscript in range(len(Y)):
        v_list = []
        for k, v in y_pred_dict.items():
            v_list.append(v[subscript])
        set_v_list = set(v_list)
       
        if list(set_v_list)[0] == dummy_label:         # "For data points that satisfy zero itemsets, we predict the majority class label in the training data,"
            y_pred.append(mode(Y).mode[0])
        elif len(list(set_v_list)) - 1 >  len(set(Y)): # "and for data points that satisfy more than one itemset, we predict using the rule with the highest F1 score on the training data."
            y_pred.append(y_pred_dict[top_f1_score][subscript])
        else:                                          # unique
            y_pred.append(list(set(v_list))[0])
   
    return y_pred