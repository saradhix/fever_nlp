# Team 23: Fact Extraction and Verification

This is the code repository for the code developed by Vijayasaradhi Indurthi and Tathagatha Raha as part of the Advanced Natural Language Processing course.  This code shows our contributions for the FEVER shared task

# Background
This code repository can be divided into multiple sections
Utility code - Code for preprocessing the wikipedia etc
Model scripts - Python scripts for training the models. We have written some scripts which will train some of the models on the linux machine. We had to implement them as scripts as these scripts consume a lot of memory due to which these could not be run in Colab, hence these are not in notebook form
Model notebooks - Colab Notebooks for training the models. These notebooks were run on Colab as these notebooks require the GPU to run, hence could not create scripts for these


# Preprocessing the wikipedia dump
# Running the baseline models
# Running the notebooks
# List of models and their scripts
| Model |Script for training  |Link to the model|
|---|---|---|
|NN_TFIDF|train_baseline_model.py|baseline_classifier_5000_model.h5 |
| NN_SUM_EMB|train_sum_embedding_model.py  | sum_embed_glove_classifier_model.h5|
| NN_SUM_EMB|train_sum_embedding_model.py  | sum_embed_fasttext_classifier_model.h5|
| Vanilla BERT|FEVER_simple_transformers_bert.ipynb  | Directory fever_bert_base_encased|
| Vanilla RoBERTa|FEVER_simple_transformers_roberta.ipynb | Directory fever_roberta_base|
| InferGlove|train_infer_glove.py | baseline_classifier_inferglove_model.h5|
| InferGlove|train_infer_fasttext.py | baseline_classifier_inferfasttext_model.h5|
| InferBERT|FEVER_InferBERT_sentence_transformers.ipynb | baseline_classifier_inferbert_model.h5|
| InferRoBERTa|FEVER_InferRoBERTa_sentence_transformers.ipynb | baseline_classifier_infer_roberta_model.h5|
| End to end evaluation|FEVER_end_to_end_score.ipynb | NA|



# Location of all the models

All the models can be found here
```bash
https://drive.google.com/drive/folders/1L47W5bN8I3uRZqPJGCa4FBInxHo4BSE9?usp=sharing
```
# Running the training
For the scripts, just run the associated py file
```bash
$python run_baseline_model.py
```
For the notebooks, please open in Colab or in jupyter and run the required cells!

Have fun with FEVER task!
