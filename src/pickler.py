import pickle

def write_model(filename, model):
    '''
    Writes a model to a file
    '''
    with open(filename, 'wb') as outfile:
        pickle.dump(model, outfile)
        
def read_model(filename):
    '''
    Reads a model from a file
    '''
    with open(filename, 'rb') as infile:
        model = pickle.load(infile)
    return model
