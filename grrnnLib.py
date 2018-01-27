import pandas as pd

def createModel(strGraphFilename):
    graph_frame = pd.read_table(strGraphFilename,header=None)
    inputs = graph_frame[0]
    outs = graph_frame[1]
    return inputs,outs

def loadData(dicDataSettings):
    #data_frame = pd.read_table("./data/app18_net9s/")
    #inputs
    KOS = pd.read_table('data/app18_net9s/KO.tsv')
    TFS = pd.read_table('data/app18_net9s/TFs.tsv')
    #outputs
    NONTFS = pd.read_table('data/app18_net9s/NonTFs.tsv')
    return None

def save():
    raise NotImplementedError
    
class GRRNN:
    def __init__():
        raise NotImplementedError
        
    def train(input, target):
        raise NotImplementedError
    
    def predict(input):
        #test inputs
        tKO = pd.read_table('data/app18_net9s/test_KO.tsv')
        tTFS = pd.read_table('data/app18_net9s/test_TFs.tsv')
        raise NotImplementedError

if __name__ == '__main__':
    createModel('./data/app18_net9s/Graph.tsv')
