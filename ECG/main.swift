//
//  main.swift
//  ECG
//
//  Created by Dave Fernandes on 2019-03-04.
//  Copyright © 2019 MintLeaf Software Inc. All rights reserved.
//

import TensorFlow
import Python
import Foundation

let batchSize: Int = 200
let maxEpochs: Int = 4

let (trainDataset, testDataset) = loadDatasets()
let testBatches = testDataset.batched(1000)

var model = ECGModel()
let optimizer = Adam(for: model, learningRate: 0.001, decay: 0)

// Training loop
for epoch in 1...maxEpochs {
    print("Epoch \(epoch), training...")
    
    var trainingLossSum: Float = 0
    var trainingBatchCount = 0
    let trainingShuffled = trainDataset.shuffled(sampleCount: 500000, randomSeed: Int64(epoch))
    let t0 = Date()
    
    // Loop over mini-batches in training set
    for batch in trainingShuffled.batched(batchSize) {
        let gradients = gradient(at: model) {
            (model: ECGModel) -> Tensor<Float> in
            
            let thisLoss = loss(model: model, examples: batch)
            trainingLossSum += thisLoss.scalarized()
            trainingBatchCount += 1
            return thisLoss
        }
        optimizer.update(&model.allDifferentiableVariables, along: gradients)
    }
    
    let t1 = Date()
    print("  training loss: \(trainingLossSum / Float(trainingBatchCount))  step: \(trainingBatchCount * epoch) (\(t1.timeIntervalSince(t0)) sec)")
    
    var testLossSum: Float = 0
    var testBatchCount = 0
    
    // Loop over test set
    for batch in testBatches {
        testLossSum += loss(model: model, examples: batch).scalarized()
        testBatchCount += 1
    }
    print("  test loss: \(testLossSum / Float(testBatchCount))")
}

// Print metrics and confusion matrix
var yActual = [Int32]()
var yPredicted = [Int32]()

for batch in testBatches {
    let labelValues = batch.labels.scalars
    let predictedValues = model.predictedClasses(for: batch.series).scalars
    yActual.append(contentsOf: labelValues)
    yPredicted.append(contentsOf: predictedValues)
}

let skm = Python.import("sklearn.metrics")
let report = skm.classification_report(yActual, yPredicted)
print(report)
let confusionMatrix = skm.confusion_matrix(yActual, yPredicted)
print(confusionMatrix)
