# Chinese-NER-pytorch
Chinese NER task on pytorch using BERT+ softmax, MacBERT + softmax and RoBERTa + softmax

To download the file
```
git clone https://github.com/carlamao/Chinese-NER-pytorch
```

Requirements:
```
pip install transformers
```

The msra data can be downloaded here: only download msra_test_bio and msra_train_bio https://github.com/lemonhu/NER-BERT-pytorch/tree/master/data/msra

To preprocess the MSRA data, run:
```
python build_msra_dataset_tags.py
```

To preprocess the Weibo data, run:
```
python wei_dataset.py
```

To preprocess the People's Daily data, run:
```
python People'sDaily.py
```

To preprocess the Resume data, run:
```
python resume_dataset.py
```
To preprocess the data, Chinese medical data run:
```
python CNMER_dataset.py
```

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


