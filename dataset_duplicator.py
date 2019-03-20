#!/usr/bin/python -i

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    test_set = "training_set"
    test_labels = "training_labels"

    with open(f"{test_set}.P", mode="rb") as data_file, open(f"{test_labels}.P", mode="rb") as label_file:
        print(f"Reading {test_set}.P.")
        data = np.load(data_file)
        labels = np.load(label_file)

def get_age(p): return int(p//12)

def duplicate_rate(labels):
    duplicates = np.zeros(int(max(labels)+1))
    
    for e in np.nditer(labels):
        duplicates[get_age(e)] += 1
    
    max_frequency = max(duplicates)
    for i in range(len(duplicates)):
        if duplicates[i] == 0: continue
        duplicates[i] = max_frequency/duplicates[i]
        
    return duplicates

def duplicate(data, labels, sampling_rate=1):
    rate = duplicate_rate(labels)*sampling_rate
    
    # keep track of sample count for each label
    counts = np.zeros(len(rate))
    
    new_data = []
    new_labels = []
    #ages = np.zeros(len(rate))
    for d, l in zip(data, labels):
        age = get_age(l)
        count = counts[age]
        
        # number of duplicate samples
        ds = int(np.floor(count + rate[age]) - np.floor(count))
        counts[age] += rate[age]
        
        #ages[age] += ds
        new_data   += [d]*ds
        new_labels += [l]*ds
        #if ds > 1:
        #    print(f"Added {ds} duplicates of image of age {age}")
    
    return np.array(new_data), np.array(new_labels)

if __name__ == "__main__":
    print(len(data))
    data, label = duplicate(data, labels)
    print(len(data))

    #plt.plot(data)
    #plt.show()