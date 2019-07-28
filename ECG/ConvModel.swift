//
//  ConvModel.swift
//  ECG
//
//  Created by Dave Fernandes on 2019-03-04.
//  Copyright © 2019 MintLeaf Software Inc. All rights reserved.
//

// Model is from: https://arxiv.org/pdf/1805.00794.pdf

import TensorFlow

public struct ConvUnit<Scalar: TensorFlowFloatingPoint> : Layer {
    var conv1: Conv1D<Scalar>
    var conv2: Conv1D<Scalar>
    var pool: MaxPool1D<Scalar>
    
    public init(kernelSize: Int, channels: Int) {
        conv1 = Conv1D<Scalar>(filterShape: (kernelSize, channels, channels), padding: .same, activation: relu)
        conv2 = Conv1D<Scalar>(filterShape: (kernelSize, channels, channels), padding: .same)
        pool = MaxPool1D<Scalar>(poolSize: kernelSize, stride: 2, padding: .valid)
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        var tmp = input.sequenced(through: conv1, conv2)
        tmp = pool(relu(tmp + input))
        return tmp
    }
}

public struct ConvModel : Layer {
    var conv1: Conv1D<Float>
    var convUnit = [ConvUnit<Float>]()
    var dense1: Dense<Float>
    var dense2: Dense<Float>
    
    @noDerivative let convUnitCount = 5
    
    public init() {
        conv1 = Conv1D<Float>(filterShape: (5, 1, 32), stride: 1, padding: .same)
        for _ in 0..<convUnitCount {
            convUnit.append(ConvUnit<Float>(kernelSize: 5, channels: 32))
        }
        dense1 = Dense<Float>(inputSize: 64, outputSize: 32, activation: relu)
        dense2 = Dense<Float>(inputSize: 32, outputSize: 5)
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var tmp = conv1(input.expandingShape(at: 2))
        
        for i in 0..<convUnitCount {
            let unit = convUnit[i]
            tmp = unit(tmp)
        }
        
        tmp = tmp.reshaped(to: [-1, 64])
        tmp = tmp.sequenced(through: dense1, dense2)
        return tmp
    }
    
    public func predictedClasses(for input: Tensor<Float>) -> Tensor<Int32> {
        return model.inferring(from: input).argmax(squeezingAxis: 1)
    }
}

typealias ECGModel = ConvModel

@differentiable(wrt: model)
func loss(model: ECGModel, examples: Example) -> Tensor<Float> {
    let logits = model(examples.series)
    return softmaxCrossEntropy(logits: logits, labels: examples.labels)
}
