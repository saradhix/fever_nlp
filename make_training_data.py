import sys
import json
from tqdm import tqdm


def load_wiki_data(filename):
    wiki_dict = {}
    fp=open(filename, 'r')
    for line in tqdm(fp):
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
                    #print("Evidence=", evidence)
                    page_id = evidence[2]
                    line_id = evidence[3]
                    try:
                        evidence_sentence = wiki_dict[page_id][line_id]
                        #print(label, claim, page_id, evidence_sentence)
                        X_all.append((claim, evidence_sentence))
                        y_all.append(label)
                    except:
                        continue
    fp.close()
    return X_all, y_all


def main():
    wiki_data = load_wiki_data('combined_replaced_pronouns.jsonl')
    X_all, y_all = get_training_data(wiki_data)
    print("Training data #X_train=", len(X_all), "#y_all", len(y_all))
    print("Generating training file")
    fp=open("formatted_data_train.jsonl", "w")
    for (X, y) in zip(X_all, y_all):
        obj={}
        obj['claim']=X[0]
        obj['evidence']=X[1]
        obj['label']=y
        fp.write(json.dumps(obj)+'\n')
    fp.close()
if __name__ == "__main__":
    main()
