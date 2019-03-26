//
//  loadDatasets.swift
//  ECG
//
//  Created by Dave Fernandes on 2019-03-05.
//  Copyright © 2019 MintLeaf Software Inc. All rights reserved.
//

import Python
import TensorFlow

func loadTimeSeries(from path: String) -> (Tensor<Int32>, Tensor<Float32>) {
    let pickle = Python.import("pickle")
    let file = Python.open(path, "rb")
    let pyDict = pickle.load(file, encoding: "bytes")
    let dict = Dictionary<String, PythonObject>(pyDict)

    guard let series = dict?["x"],
        let labels = dict?["y"] else {
        fatalError()
    }

    let labelsTensor = Tensor<Int64>(numpy: labels)!
    let seriesTensor = Tensor<Float64>(numpy: series)!
    return (Tensor<Int32>(labelsTensor), Tensor<Float32>(seriesTensor))
}

extension Dataset where Element == TensorPair<Tensor<Int32>, Tensor<Float>> {
    init(fromTuple: (Tensor<Int32>, Tensor<Float>)) {
        self = zip(Dataset<Tensor<Int32>>(elements: fromTuple.0), Dataset<Tensor<Float>>(elements: fromTuple.1))
    }
}

public typealias ECGDataset = Dataset<TensorPair<Tensor<Int32>, Tensor<Float>>>
public func loadDatasets() -> (ECGDataset, ECGDataset) {
    let trainDataset = ECGDataset(fromTuple: loadTimeSeries(from: "train_set.pickle"))
    let testDataset = ECGDataset(fromTuple: loadTimeSeries(from: "test_set.pickle"))
    return (trainDataset, testDataset)
}
