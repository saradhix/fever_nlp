import sys
import json
from tqdm import tqdm
from random import sample


def load_wiki_data(filename):
    wiki_dict = {}
    fp=open(filename, 'r')
    for line in tqdm(fp, total=5416537):
        obj = json.loads(line.strip())
        for key, sentences_list in obj.items(): #Only 1 item will be there
            for sentence_id, sentence in sentences_list: 
                if wiki_dict.get(key, 0) == 0:
                    wiki_dict[key]={}
                if sentence_id.isalpha(): continue
                s_len = len(sentence_id.strip())
                if s_len==0 or s_len >=4: continue
                #print("key=", key,"sentence_id=", sentence_id)
                wiki_dict[key][int(sentence_id)]=sentence
    #print(wiki_dict)
    return wiki_dict

def get_training_data(wiki_dict):
    X_all = []
    y_all = []
    claim_evidences_map = {}
    labels_map = {}
    train_file = 'train.jsonl'
    fp = open(train_file, 'r')
    for line in fp:
        obj = json.loads(line.strip())
        #print(obj)
        if obj['verifiable'] == 'VERIFIABLE':
            label = obj['label']
            claim = obj['claim']
            evidences = obj['evidence']
            for evidence_list in evidences:
                for evidence in evidence_list:
                    print("Claim=", claim, "label=", label)
                    page_id = evidence[2]
                    line_id = evidence[3]
                    try:
                        evidence_sentence = wiki_dict[page_id][line_id]
                        #print("Claim={} evidence_sentence={}".format(claim, evidence_sentence))
                        claim_evidences_map[claim] = claim_evidences_map.get(claim,'') + evidence_sentence
                        labels_map[claim] = label
                        #print(claim_evidences_map)
                    except:
                        pass
    fp.close()
    for (claim, evidence_concatenated) in claim_evidences_map.items():
        X_all.append((claim, evidence_concatenated))
        y_all.append(labels_map[claim])
    return X_all, y_all


def main():
    wiki_data = load_wiki_data('combined_replaced_pronouns.jsonl')
    X_all, y_all = get_training_data(wiki_data)
    print("Training data #X_train=", len(X_all), "#y_all", len(y_all))
    print("Generating training file")
    fp=open("formatted_data_train_concat_2_class.jsonl", "w")
    for (X, y) in zip(X_all, y_all):
        obj={}
        obj['claim']=X[0]
        obj['evidence_concatenated']=X[1]
        obj['label']=y
        fp.write(json.dumps(obj)+'\n')
    fp.close()
if __name__ == "__main__":
    main()
