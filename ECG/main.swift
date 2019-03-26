//
//  main.swift
//  ECG
//
//  Created by Dave Fernandes on 2019-03-04.
//  Copyright © 2019 MintLeaf Software Inc. All rights reserved.
//

import TensorFlow
import Python

let batchSize: Int32 = 500

let (trainDataset, testDataset) = loadDatasets()
let trainingBatches = trainDataset.batched(Int64(batchSize))
let testBatches = testDataset.batched(Int64(batchSize))

let trainContext = Context(learningPhase: .training)
let testContext = Context(learningPhase: .inference)
var model = ECGModel()
let optimizer = Adam<ECGModel, Float>(learningRate: 0.001, decay: 0)

for epoch in 1...50 {
    print("Epoch \(epoch), training...")
    
    var trainingLossSum: Float = 0
    var trainingBatchCount = 0
    
    for batch in trainingBatches {
        let gradients = gradient(at: model) {
            (model: ECGModel) -> Tensor<Float> in
            
            let thisLoss = loss(model: model, series: batch.second, labels: batch.first, in: trainContext)
            trainingLossSum += thisLoss.scalarized()
            trainingBatchCount += 1
            return thisLoss
        }
        optimizer.update(&model.allDifferentiableVariables, along: gradients)
    }
    print("  training loss: \(trainingLossSum / Float(trainingBatchCount))")
    
    var testLossSum: Float = 0
    var testBatchCount = 0
    
    for batch in testBatches {
        testLossSum += loss(model: model, series: batch.second, labels: batch.first, in: testContext).scalarized()
        testBatchCount += 1
    }
    print("  test loss: \(testLossSum / Float(testBatchCount))")
}

var yActual = [Int32]()
var yPredicted = [Int32]()

for batch in testBatches {
    let labelValues = batch.first.scalars
    let predictedValues = model.predictedClasses(for: batch.second).scalars
    yActual.append(contentsOf: labelValues)
    yPredicted.append(contentsOf: predictedValues)
}

let skm = Python.import("sklearn.metrics")
let report = skm.classification_report(yActual, yPredicted)
print(report)
let confusionMatrix = skm.confusion_matrix(yActual, yPredicted)
print(confusionMatrix)
