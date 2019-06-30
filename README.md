# ECG Time-Series Classification
The TensorFlow code in this project classifies a single heartbeat from an ECG recording. Three classification models were tested: a 1-D convolutional neural network (CNN); a recurrent neural network (RNN); and a Bayesian neural network (BNN) based on the CNN architecture. The CNN model is implemented in both Swift and Python; the RNN and BNN models are Python-only.

## Data
This analysis used segmented time-series data obtained from https://www.kaggle.com/shayanfazeli/heartbeat
* Time series are zero-padded, 187-element vectors containing the ECG lead II signal for one heartbeat.
* Labels [0, ..., 4] represent normal heartbeats and 4 classes of arrhythmia ['N', 'S', 'V', 'F', 'Q'].
* The class distribution is highly skewed. N = [90589, 2779, 7236, 803, 8039].
* In `PreprocessECG.ipynb` I take 100 examples from each class for the test set, and use the remainder for the training set. Under-represented classes are upsampled to balance the class ratios for training.

Thank you to Shayan Fazeli for providing this data set.

## Models
#### Convolutional Model
* The convolutional model was taken from [Kachuee, Fazeli, & Sarrafzadeh \(2018\)](https://arxiv.org/pdf/1805.00794.pdf)

Model consists of:
* An initial 1-D convolutional layer
* 5 repeated residual blocks (containing two 1-D convolutional layers with a passthrough connection and `same` padding; and a max pool layer)
* A fully-connected layer
* A linear layer with softmax output
* No regularization was used except for early stopping

#### Recurrent Model

Model consists of:
* Two stacked bidirectional GRU layers (input is masked to the variable dimension of the heartbeat vector)
* Two fully-connected layers connected to the last output-pair of the downstream (bidirectional) GRU layer
* A linear layer with softmax output
* Dropout regularization was used for the GRU layers

Since the model operates on segmented heartbeat samples, we can use a bidirectional RNN because the whole segment is available for processing at one time. It is also a more \"fair\" comparison with the CNN.

#### Bayesian Model

This model used the same network architecture as the convolutional (CNN) model above. However, the weights were stochastic, and posterior distributions of weights were trained using the Flipout method [\(Wen, Vicol, Ba, Tran, \& Grosse, 2018\)](https://arxiv.org/abs/1803.04386).

### Training
The CNN and RNN models were trained for 8000 parameter updates with a mini-batch size of 200 using the Adam optimizer with exponential learning rate decay. See notebook for parameter values. The RNN model took about 10x longer to train (wall time) than the CNN model.

The BNN model was trained for 3.5M parameter updates with a mini-batch size of 125 using the Adam optimizer with a fixed learning rate. The KL-divergence loss was annealed according to the [TensorFlow Probability example scheme](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/cifar10_bnn.py). See notebook for parameter values.

## Results
All models exhibited sufficient capacity to learn the training distribution with high accuracy. The error rates for all models were highest for the classes with the fewest examples. Collecting more data for the S- and F-type arrhythmias would likely increase the overall accuracy of the trained models.

In contrast with [Kachuee, Fazeli, & Sarrafzadeh \(2018\)](https://arxiv.org/pdf/1805.00794.pdf), I chose to upsample the under-represented classes rather than augment data as we do not have a physiologically valid generative model for heartbeats. Kachuee _et al._ also used augmented data as part of their test set without justification and I did not. As a consequence, my test set is much smaller. That said, my results for the convolutional model appear to be consistent with theirs.

#### Convolutional Model
```
       class  precision    recall  f1-score   support

           0       0.88      0.98      0.92       100
           1       0.98      0.91      0.94       100
           2       0.91      0.97      0.94       100
           3       0.98      0.87      0.92       100
           4       1.00      0.99      0.99       100

   micro avg       0.94      0.94      0.94       500
   macro avg       0.95      0.94      0.94       500
weighted avg       0.95      0.94      0.94       500
 ```
Confusion Matrix

 ![alt text](https://github.com/dave-fernandes/ECGClassifier/blob/master/images/CM-CNN.png "Confusion matrix for CNN classifier.")

#### Recurrent Model
```
       class  precision    recall  f1-score   support

           0       0.84      0.97      0.90       100
           1       0.98      0.89      0.93       100
           2       0.91      0.92      0.92       100
           3       0.98      0.89      0.93       100
           4       0.97      0.99      0.98       100

   micro avg       0.93      0.93      0.93       500
   macro avg       0.94      0.93      0.93       500
weighted avg       0.94      0.93      0.93       500
 ```
Confusion Matrix

 ![alt text](https://github.com/dave-fernandes/ECGClassifier/blob/master/images/CM-RNN.png "Confusion matrix for RNN classifier.")

#### Bayesian Model
For the Bayesian model, I obtained a Monte Carlo estimate for the most probable class. This class was then evaluated as above based on precision, recall, and the confusion matrix.

```
       class  precision    recall  f1-score   support

           0       0.88      0.98      0.92       100
           1       0.97      0.91      0.94       100
           2       0.92      0.98      0.95       100
           3       0.99      0.88      0.93       100
           4       1.00      0.99      0.99       100

   micro avg       0.95      0.95      0.95       500
   macro avg       0.95      0.95      0.95       500
weighted avg       0.95      0.95      0.95       500
 ```
Confusion Matrix

 ![alt text](https://github.com/dave-fernandes/ECGClassifier/blob/master/images/CM-BNN.png "Confusion matrix for BNN classifier.")

## Discussion
#### CNN versus RNN
The CNN model has 53,957 parameters and the RNN model has 240,293. Moreover, the serial nature of the RNN causes it to be less parallelizable than the CNN. Given that the CNN is slightly more accurate than the RNN, it provides an all-around better solution.

#### Maximum Likelihood versus Bayesian Estimate
The Bayesian model has slightly better performance than the standard CNN model with its maximum likelihood estimate (MLE). However, due to the small test-set size, this difference in performance may not be statistically significant. Still, the KL-divergence term in the loss for the Bayesian model should have a regularizing effect and allow the BNN model to generalize better.

#### Probability Estimation
The Bayesian neural network lore states that Bayesian networks produce better probability estimates than their standard \(maximum likelihood\) NN counterparts. We can check this by comparing the accuracy of the softmax \"probability\" estimate in the standard CNN model with the accuracy of the Monte Carlo probability estimate from the Bayesian network.

Fraction of correct CNN classifications versus softmax \"probability\" estimate, binned by estimate value. Error bars are 68% binomial confidence limits. The one-to-one line is the expected value if the estimate were perfect.

 ![alt text](https://github.com/dave-fernandes/ECGClassifier/blob/master/images/CNN-probability.png "Probability estimates for CNN classifier.")

Fraction of correct Bayesian network classifications versus Monte Carlo probability estimate, binned by estimate value. Error bars are 68% binomial confidence limits. The one-to-one line is the expected value if the estimate were perfect.

 ![alt text](https://github.com/dave-fernandes/ECGClassifier/blob/master/images/BNN-probability.png "Probability estimates for Bayesian classifier.")
 
 It is clear from the plots that the standard \(maximum likelihood\) network is estimating probability at least as well as the Bayesian network.

## Files
* `PreprocessECG.ipynb` is a Jupyter notebook used to format and balance the data.
* `ClassifyECG.ipynb` is a Jupyter notebook containing the CNN and RNN classification models, as well as training and evaluation code.
* `BayesClassifierECG.ipynb` is a Jupyter notebook containing the Bayesian classification model, as well as training and evaluation code.
* `ECG.xcodeproj` is an Xcode 11 project file that builds the Swift source from the `ECG` subdirectory to train the CNN model.

## Implementation Notes
* Python implementation tested with Python 3.6.7, TensorFlow 1.13.1, and TensorFlow Probability 0.6.0
* Swift implementation tested in Xcode 11 with Swift for Tensorflow toolchain 0.4.0
