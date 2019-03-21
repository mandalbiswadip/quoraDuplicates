#### Script

```buildoutcfg
python evaluate.py sentence1 sentence2

i.e. python evaluate.py "what do you think of bollywood" "what is your view on bollywood"

result - 'Yes'
```
NOTE: The above script can be changed to have multiple data points as inputs. 
The script will load the word embedding each time it is run using the avove command. It is better to modify the above script if multiple data points are to be tested


#### Web server(accessable from machine)
```
curl -X POST 'http://0.0.0.0:5000/sentence1=This%20is%20sentence%201.&sentence2=This%20is%20sentence%202.'
i.e. curl -X POST "http://0.0.0.0:5000/sentence1=what%20is%20your%20view%20on%20bollywood&sentence2=what%20do%20you%20think%20of%20bollywood"
```

#### Network Architechture

Bidirectional-LSTM. Concat the final states from the model for both the sentences and apply a fully-connected layer to get the classification layer.
Entopy has been used as a loss function for the same. The architecure details such as number of layers, batch size etc. can be found in the model/config file

#### word embedding source

[Google word2vec](https://code.google.com/archive/p/word2vec/)


##### Baseline accuracy:
Train accuracy: 80.76 %

Test accuracy: 72.83 %