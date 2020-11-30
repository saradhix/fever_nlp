import json
from random import shuffle

train_file = 'train.jsonl'

lines = []
pages = []
fp=open(train_file, 'r')
for line in fp:
    obj = json.loads(line)
    label = obj['label']
    #print(obj)
    evidences=obj['evidence'][0]
    for evidence in evidences:
        #print(evidence)
        page = evidence[2]
        #print(page)
        pages.append(page)

fp.close()

print("#pages=", len(pages))
print("Unique=", len(set(pages)))
for page in set(pages):
    print(page)
