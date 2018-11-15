# Simple Implementation of Neural Machine Translation
## Basics
This repository implements a simple neural machine translation (NMT) to translate English to Vietnamese. Given an input English sentence, NMT first embeds each English word in the sentence, then feeds the embeddings to the network, and finally outputs the probability of each predicted Vietnamese word. 

The implementation mainly follows the [Tensorflow Neural Machine Translation Tutorial](https://github.com/tensorflow/nmt), but in a simple and straight way. This implementations contains:
- Multilayer bidirectional LSTM as encoder;
- Attention mechanism at decoder;
- Beam Search strategy decoder during inference.

For more detailed information, please refer to this [tutorial](https://github.com/tensorflow/nmt).

## Requirement
To run the code you need the latest version of TensorFlow (>=1.12) installed.

## Usage
**Note**: Hyperparameters can be modified in config.py.

Run training.py to train the model. The training dataset will be automatically downloaded if they do not exist.
```
python training.py
```

Run evaluation.py to evaluate the trained model. An average BLEU score over the selected dataset will be returned.
```
python evaluation.py
```

Run test.py to test the trained model. Input a list of English sentences and the model will output the Vietnamese translations.
```
python test.py
```

## Performance
If you follow the above steps without modifying the hyperparameters, you will see a BLEU score around 20 for the test dataset "tst2012". Better score can be achieved by tuning the hyperparameters (e.g. hidden units, learning rate, or even the network layer).

## Reference
- [Tensorflow Neural Machine Translation Tutorial](https://github.com/tensorflow/nmt)

