//
//  loadDatasets.swift
//  ECG
//
//  Created by Dave Fernandes on 2019-03-05.
//  Copyright © 2019 MintLeaf Software Inc. All rights reserved.
//

import Python
import TensorFlow

struct Example: TensorGroup {
    var labels: Tensor<Int32>
    var series: Tensor<Float>
}

func loadTimeSeries(from path: String) -> Example {
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
    return Example(labels: Tensor<Int32>(labelsTensor), series: Tensor<Float32>(seriesTensor))
}

func loadDatasets() -> (training: Dataset<Example>, test: Dataset<Example>) {
    let trainingDataset = Dataset<Example>(elements: loadTimeSeries(from: "train_set.pickle"))
    let testDataset = Dataset<Example>(elements: loadTimeSeries(from: "test_set.pickle"))
    return (training: trainingDataset, test: testDataset)
}
