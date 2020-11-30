import json
from random import shuffle

train_file = 'train.jsonl'

lines = []
fp=open(train_file, 'r')
for line in fp:
    lines.append(line.strip())
fp.close()

print("Read lines = {}".format(len(lines)))
print("First 4 lines=", lines[:5])
shuffle(lines)
print("After shuffling")
print("First 4 lines=", lines[:5])


#Number of samples to dump
test_samples = 5
fp=open('test_generated.jsonl', 'w')
ft=open('test_predictions.jsonl','w')
for line in lines[:test_samples]:
    obj = json.loads(line)
    #print(obj)
    fp.write(json.dumps(obj)+'\n')
    obj['evidence']=obj['evidence'][0]
    ft.write(json.dumps(obj)+'\n')
fp.close()
ft.close()



