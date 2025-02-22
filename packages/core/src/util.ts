import { DataType, DataTypeMap, TensorLike } from "./external/tensorflow/core/types";

export function assert(expr: boolean, msg: string | (() => string)) {
  if (!expr) {
    throw new Error(typeof msg === 'string' ? msg : msg());
  }
}

export function isTypedArrayBrowser(a: unknown): a is Uint8Array
  | Float32Array | Int32Array | Uint8ClampedArray {
  return a instanceof Float32Array || a instanceof Int32Array ||
    a instanceof Uint8Array || a instanceof Uint8ClampedArray;
}

export function isPromise(object: any): object is Promise<unknown> {
  return object && object.then && typeof object.then === 'function';
}

export function checkConversionForErrors<D extends DataType>(
  vals: DataTypeMap[D] | number[], dtype: D): void {
  for (let i = 0; i < vals.length; i++) {
    const num = vals[i] as number;
    if (isNaN(num) || !isFinite(num)) {
      throw Error(`A tensor of type ${dtype} being uploaded contains ${num}.`);
    }
  }
}

function noConversionNeeded(a: TensorLike, dtype: DataType): boolean {
  return (a instanceof Float32Array && dtype === 'float32') ||
    (a instanceof Int32Array && dtype === 'int32') ||
    (a instanceof Uint8Array && dtype === 'bool');
}
