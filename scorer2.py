from scorer import fever_score
import json

instance1 = {"predicted_label": "REFUTES", "predicted_evidence": [ #is not strictly correct - missing (page2,2)
    ["page1", 1]                                    #page name, line number
]}

instance2 = {"predicted_label": "REFUTES", "predicted_evidence": [
    ["page1", 1],                                   #page name, line number
    ["page2", 2],
    ["page3", 3]
]}

actuals = [
    {"label": "REFUTES", "evidence":
        [
            [
                [None, None, "page1", 1],
                [None, None, "page2", 2],
            ]
        ]},
    {"label": "REFUTES", "evidence":
        [
            [
                [None, None, "page1", 1],
                [None, None, "page2", 2],
            ]
        ]}
]
prediction_file = 'test_predictions.jsonl'
actual_file = 'test_generated.jsonl'
predictions=[]
actuals=[]
fp=open(prediction_file, 'r')
for line in fp:
    obj = json.loads(line)
    pred={}
    pred["predicted_label"]=obj['label']
    pred["predicted_evidence"]=obj['evidence']
    predictions.append(pred)
print("Predictions=", predictions)
fp=open(actual_file, 'r')
for line in fp:
    obj = json.loads(line)
    act={}
    act["label"]=obj['label']
    act["evidence"]=obj['evidence']
    actuals.append(act)
print("Actuals=", actuals)


strict_score, label_accuracy, precision, recall, f1 = fever_score(predictions, actuals)
print("strict_score={} label_acccuracy={} precision={} recall={} f1={}".format(strict_score, label_accuracy, precision, recall, f1))
