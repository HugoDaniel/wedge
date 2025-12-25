"use client"

import { compareTensors, createSequentialTensor } from "@wedge/core/tests/testHelpers";
import { createWedge } from "@wedge/core/create";
import * as tfOriginal from '@tensorflow/tfjs';
import { expect, Test, TestContainer } from "react-browser-tests";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const tf = tfOriginal as any;

export function SingleInputOpsTests() {

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
      const result = tf.layers.reLU().apply(sequentialInput);

      const model = tf.model({ inputs: sequentialInput, outputs: result });

      const nns = await createWedge(model);

      const unsqueezedSequentialData = tf.expandDims(sequentialData, 0);
      const tfjsPrediction = model.predict(unsqueezedSequentialData);
      const nnsPrediction = nns.predict([sequentialDataArray]);

      const compareResult = compareTensors(tf.squeeze(tfjsPrediction, [0]), nnsPrediction, 0.1);

      expect(compareResult).to.equal(true);
    }} />
  </TestContainer>
}