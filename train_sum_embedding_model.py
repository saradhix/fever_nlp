import json
import sys
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import neural_network_600relu as nn
#import libglove as emb
import libfasttext as emb
def pre_process(sentence):
    ##Replace brackets
    brackets = ['-LRB-', '-LSB-', '-RRB-', '-RSB-']
    for bracket in brackets:
        sentence = sentence.replace(bracket, " ")
    return sentence

def get_training_data():
    samples_to_use = 4000
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

    print("Transforming claims")
    claims_features = emb.get_vectors(claims)

    print("Transforming evidences")
    evidence_features = emb.get_vectors(evidences)

    print("Shape of claims=", claims_features.shape)
    print("Shape of evidences=", evidence_features.shape)
    all_features = np.concatenate((claims_features, evidence_features), axis=1)
    print("Shape of all=", all_features.shape)
    X_raw_train, X_raw_test, X_train, X_test, y_train, y_test = train_test_split(X_all, all_features, y_all, test_size=0.2, random_state=42)
    print("#Train=", len(X_train), len(y_train))
    print("#Test=", len(X_test), len(y_test))

    classifier_model_name = "sum_embed_fasttext_classifier_model.h5"

    y_pred = nn.fit_predict(X_train, y_train, X_test, y_test, classifier_model_name)
    fp=open("result_analysis.txt", "w")
    for(X, gold, pred) in zip(X_raw_test, y_test, y_pred):
        if gold!=pred:
            line = X[0]+"\t"+X[1]+"\t"+"gold="+str(gold)+"\t"+"pred="+str(pred)+"\n"
            fp.write(line)
    fp.close()




if __name__ == "__main__":
    main()
