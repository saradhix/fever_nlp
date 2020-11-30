import json
import sys
import pickle
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import libglove as emb
#import neural_network_600relu as nn
import neural_network as nn
def pre_process(sentence):
    ##Replace brackets
    brackets = ['-LRB-', '-LSB-', '-RRB-', '-RSB-']
    for bracket in brackets:
        sentence = sentence.replace(bracket, " ")
    return sentence

def get_training_data():
    samples_to_use = 368892
    label_dict = {"SUPPORTS":0, "REFUTES": 1, "NOT ENOUGH INFO": 2}

    X_all = []
    y_all = []
    count = 0
    #train_file = "formatted_data_train.jsonl"
    train_file = "formatted_data_train_3_class.jsonl"
    fp = open(train_file, 'r')
    for line in fp:
        if count >= samples_to_use:
            break
        obj = json.loads(line.strip())
        claim = pre_process(obj['claim'])
        evidence = pre_process(obj['evidence'])
        X = (claim, evidence)
        y = label_dict[obj['label']]
        X_all.append(X)
        y_all.append(y)
        count += 1
    return X_all, y_all

def main():
    (X_all, y_all) = get_training_data()

    claims = [claim for (claim, _) in X_all]
    evidences = [evidence for (_, evidence) in X_all]

    print("Transforming claims and evidences...")

    u = emb.get_vectors(claims)
    v = emb.get_vectors(evidences)

    print("Shape of u=", u.shape)
    print("Shape of v=", v.shape)
    uplusv = u + v
    uminusv = u - v
    ubyv = u * v
    print("Resulting shapes=", uplusv.shape, uminusv.shape, ubyv.shape)
    all_features = np.concatenate((u, v, uplusv, uminusv, ubyv), axis=1)
    print("Shape of all=", all_features.shape)
    X_train, X_test, y_train, y_test = train_test_split(all_features, y_all, test_size=0.2, random_state=42)
    print("#Train=", len(X_train), len(y_train))
    print("#Test=", len(X_test), len(y_test))

    classifier_model_name = "baseline_classifier_inferglove_model.h5"

    nn.fit_predict(X_train, y_train, X_test, y_test, classifier_model_name)



if __name__ == "__main__":
    main()
