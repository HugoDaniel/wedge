"use client"

import { compareTensors, createSequentialTensor } from "@/lib/tests/testHelpers";
import { createNNShaders } from "@/lib/wedge/create";
import * as tf from '@tensorflow/tfjs';
import { expect } from "chai";
import { Test, TestContainer } from "react-browser-tests";

export default function TestPage() {

  return <TestContainer>
    <Test title="Relu test" skip fn={async () => {
      const inputHeight = 3;
      const inputWidth = 3;
      const inputDepth = 1;
      const shape = [inputHeight, inputWidth, inputDepth];

      // Start the sequence at -2 so the sequentialData has negative values,
      // making the ReLU function actually cap some values at 0.
      const sequenceStart = -2;
      const sequentialData = createSequentialTensor(shape, sequenceStart);
      const sequentialDataArray = sequentialData.dataSync();
      const sequentialInput = tf.input({ shape });

      // Add the sequential input and zero input
      let result = tf.layers.reLU().apply(sequentialInput) as tf.SymbolicTensor;

      const model: tf.LayersModel = tf.model({ inputs: sequentialInput, outputs: result });

      const nns = await createNNShaders(model);

      const unsqueezedSequentialData = tf.expandDims(sequentialData, 0);
      const tfjsPrediction = model.predict(unsqueezedSequentialData) as tf.Tensor;
      const nnsPrediction = nns.predict([sequentialDataArray]) as tf.Tensor;

      const compareResult = compareTensors(tf.squeeze(tfjsPrediction, [0]), nnsPrediction, 0.1);

      expect(compareResult).to.equal(true);
    }} />
  </TestContainer>
}