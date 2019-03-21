#### Script

```buildoutcfg
python evaluate.py sentence1 sentence2

i.e. python evaluate.py "what do you think of bollywood" "what is your view on bollywood"

result - 'Yes'
```


#### Web server(accessable from machine)
```
curl -X POST 'http://0.0.0.0:5000/sentence1=This%20is%20sentence%201.&sentence2=This%20is%20sentence%202.'
```

#### Network Architechture

Bidirectional-LSTM. Concat the final states from the model for both the sentences and apply a fully-connected layer to get the classification layer.
Entopy has been used as a loss function for the same.

#### word embedding source

[Google word2vec](https://code.google.com/archive/p/word2vec/)


##### Baseline accuracy:
Train accuracy: 80.76 %

Test accuracy: 72.83 %