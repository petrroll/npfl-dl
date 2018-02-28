#!/usr/bin/env python3
import numpy as np
import collections as col

def shannon_entropy(P):
    P = P[P > 0]
    return -np.sum(P * np.log(P))

def cross_entropy(P, Q):
    mask = P > 0
    P = P[mask]
    Q = Q[mask]
    return -np.sum(P * np.log(Q))

def kl_divergence(P, Q):
    return cross_entropy(P, Q) - shannon_entropy(P)

if __name__ == "__main__":
    data_freq_dict = {}
    # Load data distribution, each data point on a line
    with open("numpy_entropy_data.txt", "r") as data:
        for line in data:
            line = line.rstrip("\n")
            data_freq_dict[line] = data_freq_dict.get(line, 0) + 1

    # Load model distribution, each line `word \t probability`, creating
    # a NumPy array containing the model distribution
    model_freq_dict = {}
    with open("numpy_entropy_model.txt", "r") as model:
        for line in model:
            line = line.rstrip("\n")
            name, freq = line.split()
            model_freq_dict[name] = float(freq)

    # Preprocess so that both data and model contain info about all elements
    for (key, val) in data_freq_dict.items():
        if not (key in model_freq_dict):
            model_freq_dict[key] = 0

    for (key, val) in model_freq_dict.items():
        if not (key in data_freq_dict):
            data_freq_dict[key] = 0 
    
    # Sort data and model probabilities the same way & create numpy array
    data_freq = [data_freq_dict[k] for k in sorted(data_freq_dict.keys())]
    data_freq = np.fromiter(data_freq, int)

    model_probs = [model_freq_dict[k] for k in sorted(model_freq_dict.keys())]
    model_probs = np.fromiter(model_probs, float)

    # Create data probabilites vector (zero when no data recorded)
    data_count = data_freq.sum()
    data_probs = data_freq / data_count if data_count > 0 else np.zeros(len(model_probs))

    print("{:.2f}".format(shannon_entropy(data_probs)))
    print("{:.2f}".format(cross_entropy(data_probs, model_probs)))
    print("{:.2f}".format(kl_divergence(data_probs, model_probs)))
