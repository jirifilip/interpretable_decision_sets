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
    processed_df = process_dataset(df)
    
    result_raw = fim.apriori(processed_df, supp=(minsup*100))
    result = list(map(lambda i: list(i[0]), result_raw))
    
    return result