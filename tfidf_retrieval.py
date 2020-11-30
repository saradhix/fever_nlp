def pre_process(sentence):
    ##Replace brackets
    brackets = ['-LRB-', '-LSB-', '-RRB-', '-RSB-']
    for bracket in brackets:
      sentence = sentence.replace(bracket, " ")
    return sentence

def get_training_data():
    X_all = []
    y_all = []
    count = 0
    train_file = "retrieval_data_2_class.jsonl"
    fp = open(train_file, 'r')
    for line in fp:
      obj = json.loads(line.strip())
      claim = pre_process(obj['claim'])
      evidence = pre_process(obj['evidence'])
      X = (claim, evidence)
      y = int(obj['label'])
      X_all.append(X)
      y_all.append(y)
      count += 1
    return X_all, y_all

(X_all, y_all) = get_training_data()

X_raw_train, X_raw_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=0)
X_train = []
for X, y in zip(X_raw_train, y_train):
  X_train.append([X[0],X[1], y])
print(X_train[:5])
X_eval=[]
for X, y in zip(X_raw_test, y_test):
  X_eval.append([X[0],X[1], y])
print(X_eval[:5])
