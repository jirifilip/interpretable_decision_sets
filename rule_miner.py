import fim

def process_dataset(df):
    dataset = []
    for i in range(0,df.shape[0]):
        temp = []
        for col_name in df.columns:
            temp.append(col_name+"="+str(df[col_name][i]))
        dataset.append(temp)
        
    return dataset
    

def run_fim_apriori(df, minsup):
    print("running fim apriori function")
    processed_df = process_dataset(df)
    print("dataset processed")
    result_raw = fim.apriori(processed_df, supp=(minsup*100))
    print("apriori runned")
    result = list(map(lambda i: list(i[0]), result_raw))
    print("apriori results processed")
    return result