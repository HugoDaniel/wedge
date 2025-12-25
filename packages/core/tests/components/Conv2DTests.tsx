"use client";

import { compareTensors, createSequentialTensor } from "@wedge/core/tests/testHelpers";
import { convertShapeToTexture2DShape } from "@wedge/core/backends/webgl/buffersAndTextures";
import { defaultOptions } from "@wedge/core/constants";
import { createWedge } from "@wedge/core/create";
import { padChannels } from "@wedge/core/transforms";
import { WedgeOptions } from "@wedge/core/backends/webgl/types";
import * as tfOriginal from '@tensorflow/tfjs';
import { expect, Test, TestContainer } from "react-browser-tests";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const tf = tfOriginal as any;

const defaultOptionsWithoutBatchDim: WedgeOptions = {
  ...defaultOptions,
  hasBatchDimension: false
}

type ConvLayerArgs = {
  kernelSize: number,
  filters: number,
  padding: "valid" | "same",
  weights?: any[],
  inputDepth?: number,
  kernelInitializer?: any,
  biasInitializer?: any,
  strides?: number,
}

async function predictAndCompare(
  input: any,
  convLayerArgs: ConvLayerArgs,
  inputDimension: number,
  inputDepth: number,
  numConvLayers: number = 1,
  nnShadersOptions?: WedgeOptions) {

  let result = tf.layers.conv2d(convLayerArgs).apply(input);

  for (let i = 0; i < numConvLayers - 1; i++) {
    result = tf.layers.conv2d(convLayerArgs).apply(result);
  }

  // result = tf.layers.conv2d(convLayerArgs).apply(result);
  const model = tf.model({ inputs: input, outputs: result });

  const nns = await createWedge(model, nnShadersOptions || defaultOptionsWithoutBatchDim);

  // const inputTensor = createSequentialTensor([1, inputDimension, inputDimension, inputDepth]);
  const inputTensor = tf.ones([1, inputDimension, inputDimension, inputDepth]);

  // console.log("inputTensor")
  // tf.print(inputTensor.squeeze([0]));

  // Get prediction from TensorFlow.js model
  const tfjsPrediction = model.predict(inputTensor);

  // For Wedge: squeeze batch dimension if hasBatchDimension is false, then pad channels
  const options = nnShadersOptions || defaultOptionsWithoutBatchDim;
  const inputForWedge = options.hasBatchDimension ? inputTensor : tf.squeeze(inputTensor, [0]);
  const channelPaddedInput = padChannels(inputForWedge, "testInput");
  const [textWidth, textHeight, _] = convertShapeToTexture2DShape(channelPaddedInput.shape, "testInput");
  let channelPaddedAndTexturePadded = new Float32Array(textWidth * textHeight * 4);
  channelPaddedAndTexturePadded.set(channelPaddedInput.dataSync(), 0);

  const nnsPrediction = nns.predict([channelPaddedAndTexturePadded]);
  const nnsPredictionTensor = tf.tensor(nnsPrediction, nns.finalOutputData!.originalShape);

  return compareTensors(tf.squeeze(tfjsPrediction, [0]), nnsPredictionTensor, 0.1);
}

type ConvLayerTestArgs = {
  kernelSize: number,
  inputDepth: number,
  filters: number,
  inputDimension?: number,
  useInitializers?: boolean,
  numConvLayers?: number,
  strides?: number,
  nnShadersOptions?: WedgeOptions
}

async function createConvLayerTest({
  kernelSize,
  inputDepth,
  filters,
  inputDimension = 3,
  useInitializers = false,
  numConvLayers = 1,
  strides = 1,
  nnShadersOptions = defaultOptionsWithoutBatchDim
}: ConvLayerTestArgs) {
  const input = tf.input({ shape: [inputDimension, inputDimension, inputDepth] });

  let convLayerArgs: ConvLayerArgs = {
    filters,
    kernelSize,
    strides,
    padding: "same",
    // weights: []
  };

  let kernelWeights: any;

  if (useInitializers) {
    // Use constant initializers
    convLayerArgs.kernelInitializer = tf.initializers.constant({ value: 2.0 });
    convLayerArgs.biasInitializer = tf.initializers.constant({ value: 1.0 });
  } else {
    // Use sequential tensor weights
    kernelWeights = createSequentialTensor([kernelSize, kernelSize, inputDepth, filters]);
    const biasWeights = createSequentialTensor([filters]);
    convLayerArgs.weights = [kernelWeights, biasWeights];

    console.log("kernelWeights with format HWCN:");
    tf.print(kernelWeights);
  }

  const results = await predictAndCompare(
    input,
    convLayerArgs,
    inputDimension,
    inputDepth,
    numConvLayers,
    nnShadersOptions)

  expect(results).to.equal(true);
}

export function Conv2DTests() {

  return <TestContainer>
    <Test title="kernel size 1 & input depth 1 & filters 1" fn={async () => {
      await createConvLayerTest({ kernelSize: 1, inputDepth: 1, filters: 1, useInitializers: true });
    }} />

    <Test title="kernel size 3 & input depth 1 & filters 1" fn={async () => {
      await createConvLayerTest({ kernelSize: 3, inputDepth: 1, filters: 1 });
    }} />

    <Test title="kernel size 3 & input depth 1 & filters 2" fn={async () => {
      await createConvLayerTest({ kernelSize: 3, inputDepth: 1, filters: 2 });
    }} />

    <Test title="kernel size 3 & input depth 1 & filters 6" fn={async () => {
      await createConvLayerTest({ kernelSize: 1, inputDepth: 1, filters: 6 });
    }} />

    <Test title="kernel size 3 & input depth 5 & filters 1" fn={async () => {
      await createConvLayerTest({ kernelSize: 3, inputDepth: 5, filters: 1 });
    }} />

    <Test title="kernel size 3 & input depth 5 & filters 6" fn={async () => {
      await createConvLayerTest({ kernelSize: 3, inputDepth: 5, filters: 6 });
    }} />

    <Test title="kernel size 3 & input depth 10 & filters 10" fn={async () => {
      await createConvLayerTest({ kernelSize: 3, inputDepth: 10, filters: 10 });
    }} />

    <Test title="RENDER TARGETS - 2 render targets with inputDimension 3" fn={async () => {
      const nnShadersOptions = {
        ...defaultOptionsWithoutBatchDim,
        renderTargetBreakpoints: [
          { outputTextureElementCount: 1, numberOfRenderTargets: 2 },
        ]
      };
      await createConvLayerTest({
        kernelSize: 3, inputDepth: 3, filters: 1, inputDimension: 3,
        nnShadersOptions
      });
    }} />

    <Test title="RENDER TARGETS - 64 by 64 input of 1s, and 8 render targets" fn={async () => {
      const nnShadersOptions = {
        ...defaultOptionsWithoutBatchDim,
        renderTargetBreakpoints: [
          { outputTextureElementCount: 1, numberOfRenderTargets: 8 },
        ]
      };
      await createConvLayerTest({
        kernelSize: 3, inputDepth: 3, filters: 1, inputDimension: 64,
        nnShadersOptions
      });
    }} />

    <Test title="stride 3 test" fn={async () => {
      await createConvLayerTest({ kernelSize: 3, inputDepth: 3, filters: 1, inputDimension: 5, strides: 3 });
    }} />

    <Test title="3 sequential convs kernel size 1" fn={async () => {
      await createConvLayerTest({ kernelSize: 1, inputDepth: 3, filters: 3, inputDimension: 10, numConvLayers: 3, strides: 3 });
    }} />

    <Test title="3 sequential convs kernel size 3" fn={async () => {
      await createConvLayerTest({
        kernelSize: 3, inputDepth: 3, filters: 3, inputDimension: 3,
        numConvLayers: 3, strides: 3
      });
    }} />
  </TestContainer>
}
