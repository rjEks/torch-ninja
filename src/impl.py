import bicycle
import mlp_impl
import pandas as pd

#Running cycles and epochs

def running():
    
    df = pd.read_csv("../data/hour.csv")
    print(len(df))
    
    #input_size = bicycle('train.csv')
    #test_set = bicycle('test.csv')
    
if __name__=="__main__":
    running()
    