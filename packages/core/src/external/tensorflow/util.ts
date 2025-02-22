import { LayersModel } from "@tensorflow/tfjs";
import { getElementCount } from "../../backends/webgl/buffersAndTextures";
import { getValidShape } from "../../backends/webgl/helpers";
import { TypedArray } from "../../types";
import { TFTensor } from "./core/tensor";
import { GraphModel } from "./graph_model";
import { tensor } from "./ops/tensor";
import { NamedTensorMap, TFNode } from "./types";

export function getAttrParams(layer: any): TFNode['attrParams'] {
  switch (layer.constructor.name) {
    case 'Dense':
      return {
        units: { value: layer.units, type: 'number' },
        activation: { value: layer.activation, type: 'string' }
      };
    case 'Conv2D':
      return {
        filters: { value: layer.filters, type: 'number' },
        strides: { value: layer.strides, type: 'number[]' },
        pad: { value: layer.padding, type: 'string' },
        activation: { value: layer.activation.getClassName(), type: 'string' }
      };
    case 'DepthwiseConv2D':
      return {
        kernelSize: { value: layer.kernelSize, type: 'number[]' },
        strides: { value: layer.strides, type: 'number[]' },
        pad: { value: layer.padding, type: 'string' },
        activation: { value: layer.activation.getClassName(), type: 'string' }
      };
    // Add more cases as needed for other layer types
    default:
      return {}; // No attrParams for unknown layer types
  }
}

// The dummy input tensors are create at the start of the model execution.
// They are used when the real input tensors are not available.
export function createDummyInputsNamedTensorMap(model: LayersModel | GraphModel): NamedTensorMap {
  let dummyInputsNamedTensorMap: NamedTensorMap = {};

  if (model instanceof GraphModel) {
    model.inputs.forEach(input => {
      const inputShape = getValidShape(input.shape as number[]);
      const elementCount = getElementCount(inputShape);
      let dummyInputTensor: TFTensor;

      if (input.dtype === "int32") {
        dummyInputTensor = tensor(new Int32Array(elementCount), inputShape);
      } else if (input.dtype === "float32") {
        dummyInputTensor = tensor(new Float32Array(elementCount), inputShape);
      } else {
        throw new Error("createInputsNamedTensorMap - input.dtype not supported: " + input.dtype);
      }

      dummyInputsNamedTensorMap[input.name] = dummyInputTensor;
    });
  } else if (model instanceof LayersModel) {
    model.inputLayers.forEach(layer => {
      const inputTensorShape = layer.batchInputShape.slice(1);
      if (inputTensorShape.includes(null)) {
        throw new Error("createInputsNamedTensorMap - inputTensorShape has null values: " + inputTensorShape);
      }

      const elementCount = getElementCount(inputTensorShape as number[]);
      const dummyInputTensor = tensor(new Float32Array(elementCount), inputTensorShape as number[]);
      dummyInputsNamedTensorMap[layer.name] = dummyInputTensor;
    });
  } else {
    throw new Error("createInputsNamedTensorMap - modelType not supported.");
  }

  return dummyInputsNamedTensorMap;
}


/**
 * Returns the size (number of elements) of the tensor given its shape.
 *
 * ```js
 * const shape = [3, 4, 2];
 * const size = tf.util.sizeFromShape(shape);
 * console.log(size);
 * ```
 *
 * @doc {heading: 'Util', namespace: 'util'}
 */
export function sizeFromShape(shape: number[]): number {
  if (shape.length === 0) {
    // Scalar.
    return 1;
  }
  let size = shape[0];
  for (let i = 1; i < shape.length; i++) {
    size *= shape[i];
  }
  return size;
}

export function isScalarShape(shape: number[]): boolean {
  return shape.length === 0;
}


export function processOutput(
  outputData: TypedArray,
  finalOutputData: TypedArray,
  returnFlatArray: boolean) {
  if (finalOutputData !== null) {
    if (returnFlatArray) {
      // if (!!true) {
      console.log("outputData", outputData);
      return outputData;
    } else {
      if (options.transformations.padChannels) {
        const outputData = removePadChannels(
          outputData,
          finalOutputData.originalElementCount,
          finalOutputData.originalShape);
      }

      // Only get originalElementCount elements out of the outputData float32array.
      outputData = outputData.slice(0, finalOutputData.originalElementCount);

      return outputData;
    }
  } else {
    throw new Error("predict - error: finalOutputData is null.");
  }

}