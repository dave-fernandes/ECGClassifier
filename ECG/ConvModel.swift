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
    public func applied(to input: Tensor<Scalar>, in context: Context) -> Tensor<Scalar> {
        var tmp = conv2.applied(to: conv1.applied(to: input, in: context), in: context)
        tmp = pool.applied(to: relu(tmp + input), in: context)
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
    public func applied(to input: Tensor<Float>, in context: Context) -> Tensor<Float> {
        var tmp = conv1.applied(to: input.expandingShape(at: 2), in: context)
        tmp = unit2.applied(to: unit1.applied(to: tmp, in: context), in: context)
        tmp = unit4.applied(to: unit3.applied(to: tmp, in: context), in: context)
        tmp = unit5.applied(to: tmp, in: context).reshaped(to: [-1, 64])
        tmp = dense2.applied(to: dense1.applied(to: tmp, in: context), in: context)
        tmp = dense3.applied(to: tmp, in: context)
        return tmp
    }
    
    public func predictedClasses(for input: Tensor<Float>) -> Tensor<Int32> {
        return model.inferring(from: input).argmax(squeezingAxis: 1)
    }
}

typealias ECGModel = ConvModel

@differentiable(wrt: model)
func loss(model: ECGModel, series: Tensor<Float>, labels: Tensor<Int32>, in context: Context) -> Tensor<Float> {
    let logits = model.applied(to: series, in: context)
    return softmaxCrossEntropy(logits: logits, labels: labels)
}
