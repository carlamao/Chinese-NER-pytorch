# Chinese-NER-pytorch
Chinese NER task on pytorch using BERT+ softmax, MacBERT + softmax and RoBERTa + softmax

To download the file, 

The msra data can be downloaded here: only download msra_test_bio and msra_train_bio

To run the code on the base parameters and MSRA data:
```
python train.py 
```
To change the parameters for example:
To run the code using MacBERT softmax on the Weibo dataset:
```
python train.py --data_dir data/wei --bert_model_dir hfl/chinese-macbert-base --model_dir experiments/base_model
```

To run the model with CRF:
```
python train.py --data_dir data/wei --bert_model_dir hfl/chinese-macbert-base --model_dir experiments/base_model_CRF
```


