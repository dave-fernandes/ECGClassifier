# ECG Time-Series Classification
The TensorFlow code here classifies a single heartbeat from an ECG recording. Two classification models are tested: a 1-D convolutional neural network (CNN); and recurrent neural network (RNN). The CNN model is implemented in both Swift and Python; the RNN model is Python-only.

## Data
This analysis uses segmented time-series data obtained from https://www.kaggle.com/shayanfazeli/heartbeat
* Time series are zero-padded, 187-element vectors containing the ECG lead II signal for one heartbeat
* Labels are [0, ..., 4] representing normal heartbeats and 4 classes of arrhythmia ['N', 'S', 'V', 'F', 'Q']
* The class distribution is highly skewed. N = [90589, 2779, 7236, 803, 8039]
* In `PreprocessECG.ipynb` we take 100 examples from each class for the test set, and use the remainder for the training set. Under-represented classes are upsampled to balance the class ratios for training.

Thank you to Shayan Fazeli for providing this data set.

## Models
#### Convolutional Model
* The convolutional model is taken from [Kachuee, Fazeli, & Sarrafzadeh \(2018\)](https://arxiv.org/pdf/1805.00794.pdf)

Model consists of:
* An initial 1-D convolutional layer
* 5 repeated residual blocks (containing two 1-D convolutional layers with a passthrough connection and `same` padding; and a max pool layer)
* A fully-connected layer
* A linear layer with softmax output
* Dropout regularization was used for the convolutional layers

#### Recurrent Model

Model consists of:
* Two stacked bidirectional GRU layers (input is masked to the variable dimension of the heartbeat vector)
* Two fully-connected layers connected to the last output-pair of the second (bidirectional) GRU layer
* A linear layer with softmax output
* Dropout regularization was used for the GRU layers

Since the model operates on segmented heartbeat samples, we can use a bidirectional RNN because the whole segment is available for processing at one time. It is also a more \"fair\" comparison with the CNN.

### Training
Both models are trained for 8000 parameter updates with a mini-batch size of 200 using the Adam optimizer with exponential learning rate decay. See code for parameter values. The RNN model took about 10 times as long to train (wall time) as the CNN model.

## Results
#### Convolutional Model
```
              precision    recall  f1-score   support

           0       0.88      0.98      0.92       100
           1       0.98      0.91      0.94       100
           2       0.91      0.97      0.94       100
           3       0.98      0.87      0.92       100
           4       1.00      0.99      0.99       100

   micro avg       0.94      0.94      0.94       500
   macro avg       0.95      0.94      0.94       500
weighted avg       0.95      0.94      0.94       500

Confusion Matrix
[[98  0  2  0  0]
 [ 7 91  2  0  0]
 [ 0  1 97  2  0]
 [ 6  1  6 87  0]
 [ 1  0  0  0 99]]
```

#### Recurrent Model
```
              precision    recall  f1-score   support

           0       0.84      0.97      0.90       100
           1       0.98      0.89      0.93       100
           2       0.91      0.92      0.92       100
           3       0.98      0.89      0.93       100
           4       0.97      0.99      0.98       100

   micro avg       0.93      0.93      0.93       500
   macro avg       0.94      0.93      0.93       500
weighted avg       0.94      0.93      0.93       500

Confusion Matrix
[[97  2  1  0  0]
 [10 89  0  0  1]
 [ 4  0 92  2  2]
 [ 4  0  7 89  0]
 [ 0  0  1  0 99]]
 ```

## Discussion
Both models exhibited sufficient capacity to learn the training distribution with high accuracy. The error rates for both models are highest for the classes with the fewest examples. Collecting more data for the S- and F-type arrhythmias would likely increase the overall accuracy of the trained models.

In contrast with [Kachuee, Fazeli, & Sarrafzadeh \(2018\)](https://arxiv.org/pdf/1805.00794.pdf), we chose to upsample the under-represented classes rather than augment data as we do not have a physiologically valid generative model for heartbeats. Kachuee _et al._ also used augmented data as part of their test set without justification and we did not. As a consequence, our test set is much smaller. That said, our results for the convolutional model appear to be consistent with theirs.

The CNN model has 53,957 parameters and the RNN model has 240,293. Moreover, the serial nature of the RNN causes it to be less parallelizable than the CNN. Given that the CNN is slightly more accurate than the RNN, it provides an all-around better solution.

## Files
* `PreprocessECG.ipynb` is a Jupyter notebook used to format and balance the data. It balances the class-distribution in the training set by upsampling under-represented classes.
* `ClassifyECG.ipynb` is a Jupyter notebook containing the classification model, training and evaluation code.
* `ECG.xcodeproj` is an Xcode 10 project file that builds the Swift source from the `ECG` subdirectory.

## Implementation Notes
* Python implementation tested with Python 3.6.7 and TensorFlow 1.12.0
* Swift implementation tested with Swift compiler commit `apple/swift:tensorflow 9bf0fc1eb5071ae9856e15cb75d9b1aead415d80` and TensorFlow library commit `tensorflow/swift-apis:master d87fab3a9b68c096a07a4331bcfbb2abd4e85be1`
