import { TensorLike, WebGLData, WebGPUData } from "@tensorflow/tfjs";
import { DEBUG } from "../../../constants";
import { checkConversionForErrors, isPromise, isTypedArrayBrowser } from "../../../util";
import { DataType, DataTypeMap, FlatVector, RecursiveArray, TypedArray } from "./types";

export function decodeString(bytes: Uint8Array, encoding = 'utf-8'): string {
  encoding = encoding || 'utf-8';
  return new TextDecoder(encoding).decode(bytes);
}

export function isTypedArray(a: {}): a is Float32Array | Int32Array | Uint8Array |
  Uint8ClampedArray {
  return isTypedArrayBrowser(a);
}

function createNestedArray(
  offset: number, shape: number[], a: TypedArray, isComplex = false) {
  const ret = new Array();
  if (shape.length === 1) {
    const d = shape[0] * (isComplex ? 2 : 1);
    for (let i = 0; i < d; i++) {
      ret[i] = a[offset + i];
    }
  } else {
    const d = shape[0];
    const rest = shape.slice(1);
    const len = rest.reduce((acc, c) => acc * c) * (isComplex ? 2 : 1);
    for (let i = 0; i < d; i++) {
      ret[i] = createNestedArray(offset + i * len, rest, a, isComplex);
    }
  }
  return ret;
}


export function toNestedArray(
  shape: number[], a: TypedArray, isComplex = false) {
  if (shape.length === 0) {
    // Scalar type should return a single number.
    return a[0];
  }
  const size = shape.reduce((acc, c) => acc * c) * (isComplex ? 2 : 1);
  if (size === 0) {
    // A tensor with shape zero should be turned into empty list.
    return [];
  }
  if (size !== a.length) {
    throw new Error(`[${shape}] does not match the input size ${a.length}${isComplex ? ' for a complex tensor' : ''}.`);
  }

  return createNestedArray(0, shape, a, isComplex);
}

export function computeStrides(shape: number[]): number[] {
  const rank = shape.length;
  if (rank < 2) {
    return [];
  }

  // Last dimension has implicit stride of 1, thus having D-1 (instead of D)
  // strides.
  const strides = new Array(rank - 1);
  strides[rank - 2] = shape[rank - 1];
  for (let i = rank - 3; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

export function getArrayFromDType<D extends DataType>(
  dtype: D, size: number): DataTypeMap[D] {
  if (dtype == null || dtype === 'float32') {
    return new Float32Array(size) as DataTypeMap[D];
  } else if (dtype === 'int32') {
    return new Int32Array(size) as DataTypeMap[D];
  } else if (dtype === 'bool') {
    return new Uint8Array(size) as DataTypeMap[D];
  } else if (dtype === 'string') {
    return new Array<string>(size) as DataTypeMap[D];
  }

  throw new Error(`Unknown data type ${dtype}`);
}

export function arraysEqual(n1: FlatVector, n2: FlatVector) {
  if (n1 === n2) {
    return true;
  }
  if (n1 == null || n2 == null) {
    return false;
  }

  if (n1.length !== n2.length) {
    return false;
  }
  for (let i = 0; i < n1.length; i++) {
    if (n1[i] !== n2[i]) {
      return false;
    }
  }
  return true;
}

export function isNumber(value: {}): boolean {
  return typeof value === 'number';
}

export function isString(value: {}): value is string {
  return typeof value === 'string' || value instanceof String;
}

export function isBoolean(value: {}): boolean {
  return typeof value === 'boolean';
}

export function inferDtype(values: TensorLike | WebGLData | WebGPUData): DataType {
  if (Array.isArray(values)) {
    return inferDtype(values[0]);
  }
  if (values instanceof Float32Array) {
    return 'float32';
  } else if (
    values instanceof Int32Array || values instanceof Uint8Array ||
    values instanceof Uint8ClampedArray) {
    return 'int32';
  } else if (isNumber(values)) {
    return 'float32';
  } else if (isString(values)) {
    return 'string';
  } else if (isBoolean(values)) {
    return 'bool';
  }
  return 'float32';
}

function noConversionNeeded(a: TensorLike, dtype: DataType): boolean {
  return (a instanceof Float32Array && dtype === 'float32') ||
    (a instanceof Int32Array && dtype === 'int32') ||
    (a instanceof Uint8Array && dtype === 'bool');
}


export function toTypedArray(a: TensorLike, dtype: DataType): TypedArray {
  if (dtype === 'string') {
    throw new Error('Cannot convert a string[] to a TypedArray');
  }
  if (Array.isArray(a)) {
    a = flatten(a);
  }

  if (DEBUG) {
    checkConversionForErrors(a as number[], dtype);
  }

  if (noConversionNeeded(a, dtype)) {
    return a as TypedArray;
  }

  if (dtype == null || dtype === 'float32' || dtype === 'complex64') {
    return new Float32Array(a as number[]);
  } else if (dtype === 'int32') {
    return new Int32Array(a as number[]);
  } else if (dtype === 'bool') {
    const bool = new Uint8Array((a as number[]).length);
    for (let i = 0; i < bool.length; ++i) {
      if (Math.round((a as number[])[i]) !== 0) {
        bool[i] = 1;
      }
    }
    return bool;
  } else {
    throw new Error(`Unknown data type ${dtype}`);
  }
}


// NOTE: We explicitly type out what T extends instead of any so that
// util.flatten on a nested array of number doesn't try to infer T as a
// number[][], causing us to explicitly type util.flatten<number>().
/**
 *  Flattens an arbitrarily nested array.
 *
 * ```js
 * const a = [[1, 2], [3, 4], [5, [6, [7]]]];
 * const flat = tf.util.flatten(a);
 * console.log(flat);
 * ```
 *
 *  @param arr The nested array to flatten.
 *  @param result The destination array which holds the elements.
 *  @param skipTypedArray If true, avoids flattening the typed arrays. Defaults
 *      to false.
 *
 * @doc {heading: 'Util', namespace: 'util'}
 */
export function
  flatten<T extends number | boolean | string | Promise<number> | TypedArray>(
    arr: T | RecursiveArray<T>, result: T[] = [], skipTypedArray = false): T[] {
  if (result == null) {
    result = [];
  }
  if (typeof arr === 'boolean' || typeof arr === 'number' ||
    typeof arr === 'string' || isPromise(arr) || arr == null ||
    isTypedArray(arr) && skipTypedArray) {
    result.push(arr as T);
  } else if (Array.isArray(arr) || isTypedArray(arr)) {
    for (let i = 0; i < arr.length; ++i) {
      flatten(arr[i], result, skipTypedArray);
    }
  } else {
    let maxIndex = -1;
    for (const key of Object.keys(arr)) {
      // 0 or positive integer.
      if (/^([1-9]+[0-9]*|0)$/.test(key)) {
        maxIndex = Math.max(maxIndex, Number(key));
      }
    }
    for (let i = 0; i <= maxIndex; i++) {
      // tslint:disable-next-line: no-unnecessary-type-assertion
      flatten((arr as RecursiveArray<T>)[i], result, skipTypedArray);
    }
  }
  return result;
}
