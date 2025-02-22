/*

Original:
https://github.com/tensorflow/tfjs/blob/master/tfjs-core/src/ops/tensor_ops_util.ts

*/

import { assert } from '../../../util';
import { TFTensor } from '../core/tensor';
import { DataType, isWebGLData, isWebGPUData, TensorLike, TypedArray, WebGLData, WebGPUData } from '../core/types';
import { inferDtype, isTypedArray } from '../core/util';
import { ENGINE } from '../engine';
import { sizeFromShape } from '../util';

/** This is shared code across all tensor creation methods. */
export function makeTensor(
  values: TensorLike | WebGLData | WebGPUData, shape: number[] | undefined,
  inferredShape: number[], dtype?: DataType): TFTensor {
  if (dtype == null) {
    dtype = inferDtype(values);
  } else if (dtype === 'complex64') {
    throw new Error(
      `Cannot construct a complex64 tensor directly. ` +
      `Please use tf.complex(real, imag).`);
  }

  if (isWebGPUData(values) || isWebGLData(values)) {
    if (dtype !== 'float32' && dtype !== 'int32') {
      throw new Error(
        `Creating tensor from GPU data only supports ` +
        `'float32'|'int32' dtype, while the dtype is ${dtype}.`);
    }
    return ENGINE.backend.createTensorFromGPUData(
      values, shape || inferredShape, dtype);
  }

  if (!isTypedArray(values) && !Array.isArray(values) &&
    typeof values !== 'number' && typeof values !== 'boolean' &&
    typeof values !== 'string') {
    throw new Error(
      'values passed to tensor(values) must be a number/boolean/string or ' +
      'an array of numbers/booleans/strings, or a TypedArray');
  }
  // Verify that the shape matches the inferred shape.
  if (shape != null) {
    assertNonNegativeIntegerDimensions(shape);

    const providedSize = sizeFromShape(shape);
    const inferredSize = sizeFromShape(inferredShape);
    assert(
      providedSize === inferredSize,
      () =>
        `Based on the provided shape, [${shape}], the tensor should have ` +
        `${providedSize} values but has ${inferredSize}`);

    for (let i = 0; i < inferredShape.length; ++i) {
      const inferred = inferredShape[i];
      const flatDimsDontMatch = i === inferredShape.length - 1 ?
        inferred !== sizeFromShape(shape.slice(i)) :
        true;
      assert(
        inferredShape[i] === shape[i] || !flatDimsDontMatch,
        () => `Error creating a new Tensor. Inferred shape ` +
          `(${inferredShape}) does not match the provided ` +
          `shape (${shape}). `);
    }
  }

  if (!isTypedArray(values) && !Array.isArray(values)) {
    values = [values] as number[];
  }

  shape = shape || inferredShape;
  values = dtype !== 'string' ?
    toTypedArray(values, dtype) :
    flatten(values as string[], [], true) as string[];

  return ENGINE.makeTensor(values as TypedArray, shape, dtype);
}