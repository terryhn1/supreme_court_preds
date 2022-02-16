import extraction
import pandas as pd
import numpy as np



def get_winner_names():
    names = list()

    for i in range(len(dataset)):
        if dataset["winner_label"][i]:
            names.append(dataset["first_party"][i])
        else:
            names.append(dataset["second_party"][i])
    
    return pd.DataFrame(np.array(names))

def initialize():
    global dataset
    dataset, _ = extraction.extract_data()
    
    winner_names = get_winner_names(dataset)
    
    print(winner_names)



if __name__ == "__main__":
    initialize()
