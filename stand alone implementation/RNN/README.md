# Simple Implementation of Recurrent Neural Network
## Basics
This repository implements a simple recurrent neural network (RNN) to do the classification on MNIST dataset. RNN reads the input image line by line, and outputs the label of the image when the last input sequence (the last row of the image) is fed into RNN.
## Performance
The training and validation accuracy is recorded for each of the following three networks.
### Performance of a Recurrent Neural Network:
<p float="center">
  <img src="/stand%20alone%20implementation/RNN/Figures/rnn.png" alt="RNN" width="75%"/>
</p>

### Performance of a Long Short-Term Memory Network:
<p float="center">
  <img src="/stand%20alone%20implementation/RNN/Figures/lstm.png" alt="LSTM" width="75%"/>
</p>

### Performance of a Fully Connected Network (for comparison):
<p float="center">
  <img src="/stand%20alone%20implementation/RNN/Figures/fc.png" alt="FC" width="75%"/>
</p>

## Reference
- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
