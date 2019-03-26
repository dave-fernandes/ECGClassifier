# ECG Time-Series Classification
The TensorFlow code here classifies a single heartbeat from an ECG recording. Two classification models are tested: a 1-D convolutional neural network (CNN); and recurrent neural network (RNN). The models are implemented in both Swift and Python.

## Data
This analysis uses segmented time-series data obtained from https://www.kaggle.com/coni57/model-from-arxiv-1805-00794

## Models

## Results
RNN Model
```

```

## Files
* `PreprocessECG.ipynb` is a Jupyter notebook used to format and balance the data. It balances the class-distribution in the training set by upsampling under-represented classes.
* `ClassifyECG.ipynb` is a Jupyter notebook containing the classification model, training and evaluation code.
* `ECG.xcodeproj` is an Xcode 10 project file that builds the Swift source from the `ECG` subdirectory.

## Implementation Notes
* Python implementation tested with Python 3.6.7 and TensorFlow 1.12.0
* Swift implementation tested with Swift for TensorFlow nightly build for Xcode from March 26, 2019
