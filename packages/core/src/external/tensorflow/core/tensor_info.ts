/*

Original:
https://github.com/tensorflow/tfjs/blob/master/tfjs-core/src/tensor_info.ts

*/

import { DataType } from './types';

/**
 * We wrap data id since we use weak map to avoid memory leaks.
 * Since we have our own memory management, we have a reference counter
 * mapping a tensor to its data, so there is always a pointer (even if that
 * data is otherwise garbage collectable).
 * See https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/
 * Global_Objects/WeakMap
 */
export type DataId = object;  // object instead of {} to force non-primitive.

/** Holds metadata for a given tensor. */
export interface TensorInfo {
  dataId: DataId;
  shape: number[];
  dtype: DataType;
}