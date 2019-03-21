#
Script

```buildoutcfg
python evaluate.py sentence1 sentence2
```


# Web server
```
curl -X POST 'http://10.10.10.10/?sentence1=This%20is%20sentence%201.&sentence2=This%20is%20sentence%202.'
```

#Network Architechture

Bidirectional-LSTM. Concat the final states from the model for both the sentences and apply a fully-connected layer to get the classification layer

#word embedding source

[Google word2vec](https://code.google.com/archive/p/word2vec/)


