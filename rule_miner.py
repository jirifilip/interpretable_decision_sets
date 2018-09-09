import fim


def run_fim_apriori(df, support_thres):
 
    dataset = []
    for i in range(0,df.shape[0]):
        temp = []
        for col_name in df.columns:
            temp.append(col_name+"="+str(df[col_name][i]))
        dataset.append(temp)

    # must input support as percentage
    result_raw = fim.apriori(dataset, supp=support_thres * 10)
    result = list(map(lambda i: list(i[0]), result_raw))
        
    return result