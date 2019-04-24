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
    public func call(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        var tmp = conv2(conv1(input))
        tmp = pool(relu(tmp + input))
        return tmp
    }
}

public struct ConvModel : Layer {
    var conv1: Conv1D<Float>
    var unit1: ConvUnit<Float>
    var unit2: ConvUnit<Float>
    var unit3: ConvUnit<Float>
    var unit4: ConvUnit<Float>
    var unit5: ConvUnit<Float>
    var dense1: Dense<Float>
    var dense2: Dense<Float>
    var dense3: Dense<Float>
    
    public init() {
        conv1 = Conv1D<Float>(filterShape: (5, 1, 32), stride: 1, padding: .same)
        unit1 = ConvUnit<Float>(kernelSize: 5, channels: 32)
        unit2 = ConvUnit<Float>(kernelSize: 5, channels: 32)
        unit3 = ConvUnit<Float>(kernelSize: 5, channels: 32)
        unit4 = ConvUnit<Float>(kernelSize: 5, channels: 32)
        unit5 = ConvUnit<Float>(kernelSize: 5, channels: 32)
        dense1 = Dense<Float>(inputSize: 64, outputSize: 32, activation: relu)
        dense2 = Dense<Float>(inputSize: 32, outputSize: 32, activation: relu)
        dense3 = Dense<Float>(inputSize: 32, outputSize: 5)
    }
    
    @differentiable
    public func call(_ input: Tensor<Float>) -> Tensor<Float> {
        var tmp = conv1(input.expandingShape(at: 2))
        tmp = unit2(unit1(tmp))
        tmp = unit4(unit3(tmp))
        tmp = unit5(tmp).reshaped(to: [-1, 64])
        tmp = dense2(dense1(tmp))
        tmp = dense3(tmp)
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
