
/*

Original:
https://github.com/tensorflow/tfjs/blob/master/tfjs-core/src/io/io_utils.ts

*/

import { TypedArray } from "../../../types";
import { sizeFromShape } from "../helpers";
import { tensor } from "../ops/tensor";
import { NamedTensorMap, TFTensor } from "../types";
import { CompositeArrayBuffer } from "./composite_array_buffer";
import { DTYPE_VALUE_SIZE_MAP, ModelArtifacts, ModelArtifactsInfo, ModelJSON, WeightData, WeightsManifestConfig, WeightsManifestEntry } from "./types";

export function getModelJSONForModelArtifacts(
  artifacts: ModelArtifacts, manifest: WeightsManifestConfig): ModelJSON {
  const result: ModelJSON = {
    modelTopology: artifacts.modelTopology as any,
    format: artifacts.format,
    generatedBy: artifacts.generatedBy,
    convertedBy: artifacts.convertedBy,
    weightsManifest: manifest
  };
  if (artifacts.signature != null) {
    result.signature = artifacts.signature;
  }
  if (artifacts.userDefinedMetadata != null) {
    result.userDefinedMetadata = artifacts.userDefinedMetadata;
  }
  if (artifacts.modelInitializer != null) {
    result.modelInitializer = artifacts.modelInitializer;
  }
  if (artifacts.initializerSignature != null) {
    result.initializerSignature = artifacts.initializerSignature;
  }
  if (artifacts.trainingConfig != null) {
    result.trainingConfig = artifacts.trainingConfig;
  }
  return result;
}

// Use Buffer on Node.js instead of Blob/atob/btoa
const useNodeBuffer = typeof Buffer !== 'undefined' &&
  (typeof Blob === 'undefined' || typeof atob === 'undefined' ||
    typeof btoa === 'undefined');

export function stringByteLength(str: string): number {
  if (useNodeBuffer) {
    return Buffer.byteLength(str, 'utf8');
  }
  return new Blob([str]).size;
}

/**
 * Populate ModelArtifactsInfo fields for a model with JSON topology.
 * @param modelArtifacts
 * @returns A ModelArtifactsInfo object.
 */
export function getModelArtifactsInfoForJSON(modelArtifacts: ModelArtifacts):
  ModelArtifactsInfo {
  if (modelArtifacts.modelTopology instanceof ArrayBuffer) {
    throw new Error('Expected JSON model topology, received ArrayBuffer.');
  }

  return {
    dateSaved: new Date(),
    modelTopologyType: 'JSON',
    modelTopologyBytes: modelArtifacts.modelTopology == null ?
      0 :
      stringByteLength(JSON.stringify(modelArtifacts.modelTopology)),
    weightSpecsBytes: modelArtifacts.weightSpecs == null ?
      0 :
      stringByteLength(JSON.stringify(modelArtifacts.weightSpecs)),
    weightDataBytes: modelArtifacts.weightData == null ?
      0 :
      new CompositeArrayBuffer(modelArtifacts.weightData).byteLength,
  };
}


/**
 * Create `ModelArtifacts` from a JSON file and weights.
 *
 * @param modelJSON Object containing the parsed JSON of `model.json`
 * @param weightSpecs The list of WeightsManifestEntry for the model. Must be
 *     passed if the modelJSON has a weightsManifest.
 * @param weightData An ArrayBuffer or array of ArrayBuffers of weight data for
 *     the model corresponding to the weights in weightSpecs. Must be passed if
 *     the modelJSON has a weightsManifest.
 * @returns A Promise of the `ModelArtifacts`, as described by the JSON file.
 */
export function getModelArtifactsForJSONSync(
  modelJSON: ModelJSON, weightSpecs?: WeightsManifestEntry[],
  weightData?: WeightData): ModelArtifacts {

  const modelArtifacts: ModelArtifacts = {
    modelTopology: modelJSON.modelTopology,
    format: modelJSON.format,
    generatedBy: modelJSON.generatedBy,
    convertedBy: modelJSON.convertedBy
  };

  if (modelJSON.trainingConfig != null) {
    modelArtifacts.trainingConfig = modelJSON.trainingConfig;
  }
  if (modelJSON.weightsManifest != null) {
    if (!weightSpecs) {
      throw new Error('modelJSON has weightsManifest but weightSpecs is null');
    }
    if (!weightData) {
      throw new Error('modelJSON has weightsManifest but weightData is null');
    }
    modelArtifacts.weightSpecs = weightSpecs;
    modelArtifacts.weightData = weightData;
  }
  if (modelJSON.signature != null) {
    modelArtifacts.signature = modelJSON.signature;
  }
  if (modelJSON.userDefinedMetadata != null) {
    modelArtifacts.userDefinedMetadata = modelJSON.userDefinedMetadata;
  }
  if (modelJSON.modelInitializer != null) {
    modelArtifacts.modelInitializer = modelJSON.modelInitializer;
  }
  if (modelJSON.initializerSignature != null) {
    modelArtifacts.initializerSignature = modelJSON.initializerSignature;
  }

  return modelArtifacts;
}

/**
 * Create `ModelArtifacts` from a JSON file.
 *
 * @param modelJSON Object containing the parsed JSON of `model.json`
 * @param loadWeights Function that takes the JSON file's weights manifest,
 *     reads weights from the listed path(s), and returns a Promise of the
 *     weight manifest entries along with the weights data.
 * @returns A Promise of the `ModelArtifacts`, as described by the JSON file.
 */
export async function getModelArtifactsForJSON(
  modelJSON: ModelJSON,
  loadWeights: (weightsManifest: WeightsManifestConfig) => Promise<[
    /* weightSpecs */ WeightsManifestEntry[], WeightData,
  ]>): Promise<ModelArtifacts> {
  let weightSpecs: WeightsManifestEntry[] | undefined;
  let weightData: WeightData | undefined;

  if (modelJSON.weightsManifest != null) {
    [weightSpecs, weightData] = await loadWeights(modelJSON.weightsManifest);
  }

  return getModelArtifactsForJSONSync(modelJSON, weightSpecs, weightData);
}


/**
 * Concatenate the weights stored in a WeightsManifestConfig into a list of
 * WeightsManifestEntry
 *
 * @param weightsManifest The WeightsManifestConfig to extract weights from.
 * @returns A list of WeightsManifestEntry of the weights in the weightsManifest
 */
export function getWeightSpecs(weightsManifest: WeightsManifestConfig):
  WeightsManifestEntry[] {
  const weightSpecs: WeightsManifestEntry[] = [];
  for (const entry of weightsManifest) {
    weightSpecs.push(...entry.weights);
  }
  return weightSpecs;
}


function decodeWeight(
  spec: WeightsManifestEntry,
  byteBuffer: ArrayBuffer): TFTensor {

  const name = spec.name;
  const dtype = spec.dtype;
  const shape = spec.shape;
  const size = sizeFromShape(shape);
  let values: TypedArray | string[] | Uint8Array[];
  let offset = 0;

  if ('quantization' in spec) {
    const quantization = spec.quantization;
    if (quantization.dtype === 'uint8' || quantization.dtype === 'uint16') {
      if (!('min' in quantization && 'scale' in quantization)) {
        throw new Error(
          `Weight ${spec.name} with quantization ${quantization.dtype} ` +
          `doesn't have corresponding metadata min and scale.`);
      }
    } else if (quantization.dtype === 'float16') {
      if (dtype !== 'float32') {
        throw new Error(
          `Weight ${spec.name} is quantized with ${quantization.dtype} ` +
          `which only supports weights of type float32 not ${dtype}.`);
      }
    } else {
      throw new Error(
        `Weight ${spec.name} has unknown ` +
        `quantization dtype ${quantization.dtype}. ` +
        `Supported quantization dtypes are: ` +
        `'uint8', 'uint16', and 'float16'.`);
    }
    const quantizationSizeFactor = DTYPE_VALUE_SIZE_MAP[quantization.dtype];
    const quantizedArray = (quantization.dtype === 'uint8') ?
      new Uint8Array(byteBuffer) :
      new Uint16Array(byteBuffer);
    if (dtype === 'float32') {
      if (quantization.dtype === 'uint8' || quantization.dtype === 'uint16') {
        values = new Float32Array(quantizedArray.length);
        for (let i = 0; i < quantizedArray.length; i++) {
          const v = quantizedArray[i];
          values[i] = v * quantization.scale + quantization.min;
        }
      } else if (quantization.dtype === 'float16') {
        // TODO: This is inefficient. Make getFloat16Decoder efficient.
        const float16Decode = getFloat16Decoder();
        values = float16Decode(quantizedArray as Uint16Array);
      } else {
        throw new Error(
          `Unsupported quantization type ${quantization.dtype} ` +
          `for weight type float32.`);
      }
    } else if (dtype === 'int32') {
      if (quantization.dtype !== 'uint8' && quantization.dtype !== 'uint16') {
        throw new Error(
          `Unsupported quantization type ${quantization.dtype} ` +
          `for weight type int32.`);
      }
      values = new Int32Array(quantizedArray.length);
      for (let i = 0; i < quantizedArray.length; i++) {
        const v = quantizedArray[i];
        values[i] = Math.round(v * quantization.scale + quantization.min);
      }
    } else {
      throw new Error(`Unsupported dtype in weight '${name}': ${dtype}`);
    }
    offset += size * quantizationSizeFactor;
  } else if (dtype === 'string') {
    const size = sizeFromShape(spec.shape);
    values = [];
    for (let i = 0; i < size; i++) {
      const byteLength = new Uint32Array(
        byteBuffer.slice(offset, offset + NUM_BYTES_STRING_LENGTH))[0];
      offset += NUM_BYTES_STRING_LENGTH;
      const bytes = new Uint8Array(
        byteBuffer.slice(offset, offset + byteLength));
      (values as Uint8Array[]).push(bytes);
      offset += byteLength;
    }
  } else {
    const dtypeFactor = DTYPE_VALUE_SIZE_MAP[dtype];
    if (dtype === 'float32') {
      values = new Float32Array(byteBuffer);
    } else if (dtype === 'int32') {
      values = new Int32Array(byteBuffer);
    } else if (dtype === 'bool') {
      values = new Uint8Array(byteBuffer);
    } else if (dtype === 'complex64') {
      values = new Float32Array(byteBuffer);
      const real = new Float32Array(values.length / 2);
      const image = new Float32Array(values.length / 2);
      for (let i = 0; i < real.length; i++) {
        real[i] = values[i * 2];
        image[i] = values[i * 2 + 1];
      }
      const realTensor = tensor(real, shape, 'float32');
      const imageTensor = tensor(image, shape, 'float32');
      const complexTensor = complex(realTensor, imageTensor);
      realTensor.dispose();
      imageTensor.dispose();
      return complexTensor;
    } else {
      throw new Error(`Unsupported dtype in weight '${name}': ${dtype}`);
    }
    offset += size * dtypeFactor;
  }

  return tensor(values, shape, dtype);
}


function getWeightBytelength(spec: WeightsManifestEntry,
  slice: (start: number, end: number) => ArrayBuffer): number {

  const size = sizeFromShape(spec.shape);
  let bytesPerValue: number;
  if ('quantization' in spec) {
    const quantization = spec.quantization;
    bytesPerValue = DTYPE_VALUE_SIZE_MAP[quantization.dtype];
  } else if (spec.dtype === 'string') {
    // Can not statically determine string length.
    let byteLength = 0;
    for (let i = 0; i < size; i++) {
      byteLength += NUM_BYTES_STRING_LENGTH + new Uint32Array(
        slice(byteLength, byteLength + NUM_BYTES_STRING_LENGTH))[0];
    }
    return byteLength;
  } else {
    bytesPerValue = DTYPE_VALUE_SIZE_MAP[spec.dtype];
  }

  return size * bytesPerValue;
}


/**
 * Decode flat ArrayBuffer as weights.
 *
 * This function does not handle sharding.
 *
 * This function is the reverse of `encodeWeights`.
 *
 * @param weightData A flat ArrayBuffer or an array of ArrayBuffers carrying the
 *   binary values of the tensors concatenated in the order specified in
 *   `specs`.
 * @param specs Specifications of the names, dtypes and shapes of the tensors
 *   whose value are encoded by `buffer`.
 * @return A map from tensor name to tensor value, with the names corresponding
 *   to names in `specs`.
 * @throws Error, if any of the tensors has unsupported dtype.
 */
export function decodeWeights(
  weightData: WeightData,
  specs: WeightsManifestEntry[]): NamedTensorMap {
  // TODO: Support quantization.
  const compositeBuffer = new CompositeArrayBuffer(weightData);
  const out: NamedTensorMap = {};
  let offset = 0;
  for (const spec of specs) {
    const byteLength = getWeightBytelength(spec, (start, end) => {
      return compositeBuffer.slice(offset + start, offset + end);
    });
    out[spec.name] = decodeWeight(spec, compositeBuffer
      .slice(offset, offset + byteLength));
    offset += byteLength;
  }
  return out;
}

type FetchFuncOptions = { isBinary: boolean };

export function myFetchFunc(
  input: RequestInfo | URL,
  init?: RequestInit,
  fetchOptions?: FetchFuncOptions) {
  // XXX: fetchOptions is not currently used.

  return fetch(input, init);
}