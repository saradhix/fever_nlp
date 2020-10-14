import json
import sys
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import neural_network as nn
def pre_process(sentence):
    ##Replace brackets
    brackets = ['-LRB-', '-LSB-', '-RRB-', '-RSB-']
    for bracket in brackets:
        sentence = sentence.replace(bracket, " ")
    return sentence

def get_training_data():
    samples_to_use = 150000
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

    corpus = claims + evidences
    print("Length of raw corpus=", len(corpus))

    print("Vectorizing...")

    num_tfidf_features=5000

    vectorizer = TfidfVectorizer(max_features=num_tfidf_features, strip_accents='unicode',analyzer='word', token_pattern=r'\w{1,}',ngram_range=(1,1),
                                 use_idf=1,smooth_idf=1,stop_words='english',)
    vectorizer.fit(corpus)

    model_pickle = "tfidf_vectorizer_"+str(num_tfidf_features)+"_features.pickle"
    fp = open(model_pickle, 'wb')
    pickle.dump(vectorizer, fp)
    fp.close() #To close the pickle file
    claims_features = vectorizer.transform(claims)
    evidence_features = vectorizer.transform(evidences)

    print("Shape of claims=", claims_features.shape)
    print("Shape of evidences=", evidence_features.shape)
    print("Converting claims to dense")
    claims_features = claims_features.todense()
    print("Converting evidence to dense")
    evidence_features = evidence_features.todense()
    dot_products = []
    for (c, e) in zip(claims_features, evidence_features):
        dp = np.dot(c, e.T)
        dot_products.append(dp)
    dot_products = np.array(dot_products)
    print("Shape of dot_products=", dot_products.shape)
    dot_products = dot_products.reshape(evidence_features.shape[0], -1)
    print("AFter reshaping Shape of dot_products=", dot_products.shape)
    all_features = np.concatenate((claims_features, dot_products, evidence_features), axis=1)
    print("Shape of all=", all_features.shape)
    X_train, X_test, y_train, y_test = train_test_split(all_features, y_all, test_size=0.2, random_state=42)
    print("#Train=", len(X_train), len(y_train))
    print("#Test=", len(X_test), len(y_test))

    classifier_model_name = "baseline_classifier_"+str(num_tfidf_features)+"_model.h5"

    nn.fit_predict(X_train, y_train, X_test, y_test, classifier_model_name)



if __name__ == "__main__":
    main()
