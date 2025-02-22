
import { assert } from '../../../util';
import { TFTensorInfo } from '../types';
import { sizeFromShape } from '../util';
import { getGlobal } from './global_util';
import { DataId } from './tensor_info';
import { ArrayMap, BackendValues, DataType, DataTypeMap, DataValues, NumericDataType, Rank, ShapeMap, SingleValueMap, TypedArray } from './types';
import { arraysEqual, computeStrides, decodeString, getArrayFromDType, toNestedArray } from './util';

export interface TFTensorData<D extends DataType> {
  dataId?: DataId;
  values?: DataTypeMap[D];
}

// This interface mimics KernelBackend (in backend.ts), which would create a
// circular dependency if imported.
export interface Backend { }

/**
 * A mutable object, similar to `tf.TFTensor`, that allows users to set values
 * at locations before converting to an immutable `tf.TFTensor`.
 *
 * See `tf.buffer` for creating a tensor buffer.
 *
 * @doc {heading: 'TFTensors', subheading: 'Classes'}
 */
export class TFTensorBuffer<R extends Rank, D extends DataType = 'float32'> {
  size: number;
  shape: ShapeMap[R];
  strides: number[];
  values: DataTypeMap[D];

  constructor(shape: ShapeMap[R], public dtype: D, values?: DataTypeMap[D]) {
    this.shape = shape.slice() as ShapeMap[R];
    this.size = sizeFromShape(shape);

    if (values != null) {
      const n = values.length;
      assert(
        n === this.size,
        () => `Length of values '${n}' does not match the size ` +
          `inferred by the shape '${this.size}'.`);
    }
    if (dtype === 'complex64') {
      throw new Error(
        `complex64 dtype TFTensorBuffers are not supported. Please create ` +
        `a TFTensorBuffer for the real and imaginary parts separately and ` +
        `call tf.complex(real, imag).`);
    }
    this.values = values || getArrayFromDType(dtype, this.size);
    this.strides = computeStrides(shape);
  }

  /**
   * Sets a value in the buffer at a given location.
   *
   * @param value The value to set.
   * @param locs  The location indices.
   *
   * @doc {heading: 'TFTensors', subheading: 'Creation'}
   */
  set(value: SingleValueMap[D], ...locs: number[]): void {
    if (locs.length === 0) {
      locs = [0];
    }
    assert(
      locs.length === this.rank,
      () => `The number of provided coordinates (${locs.length}) must ` +
        `match the rank (${this.rank})`);

    const index = this.locToIndex(locs);
    this.values[index] = value as number;
  }

  /**
   * Returns the value in the buffer at the provided location.
   *
   * @param locs The location indices.
   *
   * @doc {heading: 'TFTensors', subheading: 'Creation'}
   */
  get(...locs: number[]): SingleValueMap[D] {
    if (locs.length === 0) {
      locs = [0];
    }
    let i = 0;
    for (const loc of locs) {
      if (loc < 0 || loc >= this.shape[i]) {
        const msg = `Requested out of range element at ${locs}. ` +
          `  Buffer shape=${this.shape}`;
        throw new Error(msg);
      }
      i++;
    }
    let index = locs[locs.length - 1];
    for (let i = 0; i < locs.length - 1; ++i) {
      index += this.strides[i] * locs[i];
    }
    return this.values[index] as SingleValueMap[D];
  }

  locToIndex(locs: number[]): number {
    if (this.rank === 0) {
      return 0;
    } else if (this.rank === 1) {
      return locs[0];
    }
    let index = locs[locs.length - 1];
    for (let i = 0; i < locs.length - 1; ++i) {
      index += this.strides[i] * locs[i];
    }
    return index;
  }

  indexToLoc(index: number): number[] {
    if (this.rank === 0) {
      return [];
    } else if (this.rank === 1) {
      return [index];
    }
    const locs: number[] = new Array(this.shape.length);
    for (let i = 0; i < locs.length - 1; ++i) {
      locs[i] = Math.floor(index / this.strides[i]);
      index -= locs[i] * this.strides[i];
    }
    locs[locs.length - 1] = index;
    return locs;
  }

  get rank() {
    return this.shape.length;
  }

  /**
   * Creates an immutable `tf.TFTensor` object from the buffer.
   *
   * @doc {heading: 'TFTensors', subheading: 'Creation'}
   */
  toTFTensor(): TFTensor<R> {
    return trackerFn!().makeTFTensor(this.values, this.shape, this.dtype) as
      TFTensor<R>;
  }
}

export interface DataToGPUWebGLOption {
  customTexShape?: [number, number];
}

export type DataToGPUOptions = DataToGPUWebGLOption;

export interface GPUData {
  tensorRef: TFTensor;
  texture?: WebGLTexture;
  buffer?: GPUBuffer;
  texShape?: [number, number];
}

export interface TFTensorTracker {
  makeTFTensor(
    values: DataValues, shape: number[], dtype: DataType,
    backend?: Backend): TFTensor;
  makeVariable(
    initialValue: TFTensor, trainable?: boolean, name?: string,
    dtype?: DataType): Variable;
  incRef(a: TFTensor, backend: Backend | null): void;
  disposeTFTensor(t: TFTensor): void;
  disposeVariable(v: Variable): void;
  read(dataId: DataId): Promise<BackendValues>;
  readSync(dataId: DataId): BackendValues;
  readToGPU(dataId: DataId, options?: DataToGPUOptions): GPUData;
}

/**
 * The TFTensor class calls into this handler to delegate chaining operations.
 */
export interface OpHandler {
  cast<T extends TFTensor>(x: T, dtype: DataType): T;
  buffer<R extends Rank, D extends DataType>(
    shape: ShapeMap[R], dtype: D,
    values?: DataTypeMap[D]): TFTensorBuffer<R, D>;
  print<T extends TFTensor>(x: T, verbose: boolean): void;
  clone<T extends TFTensor>(x: T): T;
  // TODO(yassogba) bring reshape back?
}

// For tracking tensor creation and disposal.
let trackerFn: (() => TFTensorTracker) | null = null;
// Used by chaining methods to call into ops.
let opHandler: OpHandler | null = null;
// Used to warn about deprecated methods.
let deprecationWarningFn: ((msg: string) => void) | null = null;
// This here so that we can use this method on dev branches and keep the
// functionality at master.
// tslint:disable-next-line:no-unused-expression
[deprecationWarningFn];

/**
 * An external consumer can register itself as the tensor tracker. This way
 * the TFTensor class can notify the tracker for every tensor created and
 * disposed.
 */
export function setTFTensorTracker(fn: () => TFTensorTracker) {
  trackerFn = fn;
}

/**
 * An external consumer can register itself as the op handler. This way the
 * TFTensor class can have chaining methods that call into ops via the op
 * handler.
 */
export function setOpHandler(handler: OpHandler) {
  opHandler = handler;
}

/**
 * Sets the deprecation warning function to be used by this file. This way the
 * TFTensor class can be a leaf but still use the environment.
 */
export function setDeprecationWarningFn(fn: (msg: string) => void) {
  deprecationWarningFn = fn;
}

// Declare this namespace to make TFTensor class augmentation work in google3.
export declare namespace TFTensor { }
/**
 * A `tf.TFTensor` object represents an immutable, multidimensional array of
 * numbers that has a shape and a data type.
 *
 * For performance reasons, functions that create tensors do not necessarily
 * perform a copy of the data passed to them (e.g. if the data is passed as a
 * `Float32Array`), and changes to the data will change the tensor. This is not
 * a feature and is not supported. To avoid this behavior, use the tensor before
 * changing the input data or create a copy with `copy = tf.add(yourTFTensor, 0)`.
 *
 * See `tf.tensor` for details on how to create a `tf.TFTensor`.
 *
 * @doc {heading: 'TFTensors', subheading: 'Classes'}
 */
export class TFTensor<R extends Rank = Rank> implements TFTensorInfo {
  /** Unique id of this tensor. */
  readonly id: number;
  /**
   * Id of the bucket holding the data for this tensor. Multiple arrays can
   * point to the same bucket (e.g. when calling array.reshape()).
   */
  dataId: DataId;
  /** The shape of the tensor. */
  readonly shape: ShapeMap[R];
  /** Number of elements in the tensor. */
  readonly size: number;
  /** The data type for the array. */
  readonly dtype: DataType;
  /** The rank type for the array (see `Rank` enum). */
  readonly rankType: R;

  /** Whether this tensor has been globally kept. */
  kept = false;
  /** The id of the scope this tensor is being tracked in. */
  scopeId: number;
  /** The keras mask that some keras layers attach to the tensor */
  kerasMask?: TFTensor;

  /**
   * Number of elements to skip in each dimension when indexing. See
   * https://docs.scipy.org/doc/numpy/reference/generated/\
   * numpy.ndarray.strides.html
   */
  readonly strides: number[];

  constructor(shape: ShapeMap[R], dtype: DataType, dataId: DataId, id: number) {
    this.shape = shape.slice() as ShapeMap[R];
    this.dtype = dtype || 'float32';
    this.size = sizeFromShape(shape);
    this.strides = computeStrides(shape);
    this.dataId = dataId;
    this.id = id;
    this.rankType = (this.rank < 5 ? this.rank.toString() : 'higher') as R;
  }
  name: string;

  get rank(): number {
    return this.shape.length;
  }

  /**
   * Returns a promise of `tf.TFTensorBuffer` that holds the underlying data.
   *
   * @doc {heading: 'TFTensors', subheading: 'Classes'}
   */
  async buffer<D extends DataType = 'float32'>(): Promise<TFTensorBuffer<R, D>> {
    const vals = await this.data<D>();
    return opHandler!.buffer(this.shape, this.dtype as D, vals);
  }

  /**
   * Returns a `tf.TFTensorBuffer` that holds the underlying data.
   * @doc {heading: 'TFTensors', subheading: 'Classes'}
   */
  bufferSync<D extends DataType = 'float32'>(): TFTensorBuffer<R, D> {
    return opHandler!.buffer(this.shape, this.dtype as D, this.dataSync());
  }

  /**
   * Returns the tensor data as a nested array. The transfer of data is done
   * asynchronously.
   *
   * @doc {heading: 'TFTensors', subheading: 'Classes'}
   */
  async array(): Promise<ArrayMap[R]> {
    const vals = await this.data();
    return toNestedArray(this.shape, vals, this.dtype === 'complex64') as
      ArrayMap[R];
  }

  /**
   * Returns the tensor data as a nested array. The transfer of data is done
   * synchronously.
   *
   * @doc {heading: 'TFTensors', subheading: 'Classes'}
   */
  arraySync(): ArrayMap[R] {
    return toNestedArray(
      this.shape, this.dataSync(), this.dtype === 'complex64') as
      ArrayMap[R];
  }

  /**
   * Asynchronously downloads the values from the `tf.TFTensor`. Returns a
   * promise of `TypedArray` that resolves when the computation has finished.
   *
   * @doc {heading: 'TFTensors', subheading: 'Classes'}
   */
  async data<D extends DataType = NumericDataType>(): Promise<DataTypeMap[D]> {
    this.throwIfDisposed();
    const data = trackerFn!().read(this.dataId);
    if (this.dtype === 'string') {
      const bytes = await data as Uint8Array[];
      try {
        return bytes.map(b => decodeString(b)) as DataTypeMap[D];
      } catch {
        throw new Error(
          'Failed to decode the string bytes into utf-8. ' +
          'To get the original bytes, call tensor.bytes().');
      }
    }
    return data as Promise<DataTypeMap[D]>;
  }

  /**
   * Copy the tensor's data to a new GPU resource. Comparing to the `dataSync()`
   * and `data()`, this method prevents data from being downloaded to CPU.
   *
   * For WebGL backend, the data will be stored on a densely packed texture.
   * This means that the texture will use the RGBA channels to store value.
   *
   * For WebGPU backend, the data will be stored on a buffer. There is no
   * parameter, so can not use a user-defined size to create the buffer.
   *
   * @param options:
   *     For WebGL,
   *         - customTexShape: Optional. If set, will use the user defined
   *     texture shape to create the texture.
   *
   * @returns For WebGL backend, a GPUData contains the new texture and
   *     its information.
   *     {
   *        tensorRef: The tensor that is associated with this texture,
   *        texture: WebGLTexture,
   *        texShape: [number, number] // [height, width]
   *     }
   *
   *     For WebGPU backend, a GPUData contains the new buffer.
   *     {
   *        tensorRef: The tensor that is associated with this buffer,
   *        buffer: GPUBuffer,
   *     }
   *
   *     Remember to dispose the GPUData after it is used by
   *     `res.tensorRef.dispose()`.
   *
   * @doc {heading: 'TFTensors', subheading: 'Classes'}
   */
  dataToGPU(options?: DataToGPUOptions): GPUData {
    this.throwIfDisposed();
    return trackerFn!().readToGPU(this.dataId, options);
  }

  /**
   * Synchronously downloads the values from the `tf.TFTensor`. This blocks the
   * UI thread until the values are ready, which can cause performance issues.
   *
   * @doc {heading: 'TFTensors', subheading: 'Classes'}
   */
  dataSync<D extends DataType = NumericDataType>(): DataTypeMap[D] {
    this.throwIfDisposed();
    const data = trackerFn!().readSync(this.dataId);
    if (this.dtype === 'string') {
      try {
        return (data as Uint8Array[]).map(b => decodeString(b)) as
          DataTypeMap[D];
      } catch {
        throw new Error(
          'Failed to decode the string bytes into utf-8. ' +
          'To get the original bytes, call tensor.bytes().');
      }
    }
    return data as DataTypeMap[D];
  }

  /** Returns the underlying bytes of the tensor's data. */
  async bytes(): Promise<Uint8Array[] | Uint8Array> {
    this.throwIfDisposed();
    const data = await trackerFn!().read(this.dataId);
    if (this.dtype === 'string') {
      return data as Uint8Array[];
    } else {
      return new Uint8Array((data as TypedArray).buffer);
    }
  }

  /**
   * Disposes `tf.TFTensor` from memory.
   *
   * @doc {heading: 'TFTensors', subheading: 'Classes'}
   */
  dispose(): void {
    if (this.isDisposed) {
      return;
    }
    if (this.kerasMask) {
      this.kerasMask.dispose();
    }
    trackerFn!().disposeTFTensor(this);
    this.isDisposedInternal = true;
  }

  protected isDisposedInternal = false;
  get isDisposed(): boolean {
    return this.isDisposedInternal;
  }

  throwIfDisposed() {
    if (this.isDisposed) {
      throw new Error(`TFTensor is disposed.`);
    }
  }

  /**
   * Prints the `tf.TFTensor`. See `tf.print` for details.
   *
   * @param verbose Whether to print verbose information about the tensor,
   *    including dtype and size.
   *
   * @doc {heading: 'TFTensors', subheading: 'Classes'}
   */
  print(verbose = false): void {
    return opHandler!.print(this, verbose);
  }

  /**
   * Returns a copy of the tensor. See `tf.clone` for details.
   * @doc {heading: 'TFTensors', subheading: 'Classes'}
   */
  clone<T extends TFTensor>(this: T): T {
    this.throwIfDisposed();
    return opHandler!.clone(this);
  }

  /**
   * Returns a human-readable description of the tensor. Useful for logging.
   *
   * @doc {heading: 'TFTensors', subheading: 'Classes'}
   */
  // toString(verbose = false): string {
  //   const vals = this.dataSync();
  //   return tensorToString(vals, this.shape, this.dtype, verbose);
  // }

  cast<T extends this>(dtype: DataType): T {
    this.throwIfDisposed();
    return opHandler!.cast(this as T, dtype);
  }
  variable(trainable = true, name?: string, dtype?: DataType): Variable<R> {
    this.throwIfDisposed();
    return trackerFn!().makeVariable(this, trainable, name, dtype) as
      Variable<R>;
  }
}

Object.defineProperty(TFTensor, Symbol.hasInstance, {
  value: (instance: TFTensor) => {
    // Implementation note: we should use properties of the object that will be
    // defined before the constructor body has finished executing (methods).
    // This is because when this code is transpiled by babel, babel will call
    // classCallCheck before the constructor body is run.
    // See https://github.com/tensorflow/tfjs/issues/3384 for backstory.
    return !!instance && instance.data != null && instance.dataSync != null &&
      instance.throwIfDisposed != null;
  }
});

export function getGlobalTFTensorClass() {
  // Use getGlobal so that we can augment the TFTensor class across package
  // boundaries because the node resolution alg may result in different modules
  // being returned for this file depending on the path they are loaded from.
  return getGlobal('TFTensor', () => {
    return TFTensor;
  });
}

// Global side effect. Cache global reference to TFTensor class
getGlobalTFTensorClass();

export interface NumericTFTensor<R extends Rank = Rank> extends TFTensor<R> {
  dtype: NumericDataType;
  dataSync<D extends DataType = NumericDataType>(): DataTypeMap[D];
  data<D extends DataType = NumericDataType>(): Promise<DataTypeMap[D]>;
  dataToGPU(options?: DataToGPUOptions): GPUData;
}

export interface StringTFTensor<R extends Rank = Rank> extends TFTensor<R> {
  dtype: 'string';
  dataSync<D extends DataType = 'string'>(): DataTypeMap[D];
  data<D extends DataType = 'string'>(): Promise<DataTypeMap[D]>;
}

/** @doclink TFTensor */
export type Scalar = TFTensor<Rank.R0>;
/** @doclink TFTensor */
export type TFTensor1D = TFTensor<Rank.R1>;
/** @doclink TFTensor */
export type TFTensor2D = TFTensor<Rank.R2>;
/** @doclink TFTensor */
export type TFTensor3D = TFTensor<Rank.R3>;
/** @doclink TFTensor */
export type TFTensor4D = TFTensor<Rank.R4>;
/** @doclink TFTensor */
export type TFTensor5D = TFTensor<Rank.R5>;
/** @doclink TFTensor */
export type TFTensor6D = TFTensor<Rank.R6>;

/**
 * A mutable `tf.TFTensor`, useful for persisting state, e.g. for training.
 *
 * @doc {heading: 'TFTensors', subheading: 'Classes'}
 */
export class Variable<R extends Rank = Rank> extends TFTensor<R> {
  name: string;

  constructor(
    initialValue: TFTensor<R>, public trainable: boolean, name: string,
    tensorId: number) {
    super(
      initialValue.shape, initialValue.dtype, initialValue.dataId, tensorId);
    this.name = name;
  }

  /**
   * Assign a new `tf.TFTensor` to this variable. The new `tf.TFTensor` must have
   * the same shape and dtype as the old `tf.TFTensor`.
   *
   * @param newValue New tensor to be assigned to this variable.
   *
   * @doc {heading: 'TFTensors', subheading: 'Classes'}
   */
  assign(newValue: TFTensor<R>): void {
    if (newValue.dtype !== this.dtype) {
      throw new Error(
        `dtype of the new value (${newValue.dtype}) and ` +
        `previous value (${this.dtype}) must match`);
    }
    if (!arraysEqual(newValue.shape, this.shape)) {
      throw new Error(
        `shape of the new value (${newValue.shape}) and ` +
        `previous value (${this.shape}) must match`);
    }
    trackerFn!().disposeTFTensor(this);
    this.dataId = newValue.dataId;
    trackerFn!().incRef(this, null /* backend */);
  }

  override dispose(): void {
    trackerFn!().disposeVariable(this);
    this.isDisposedInternal = true;
  }
}

Object.defineProperty(Variable, Symbol.hasInstance, {
  value: (instance: Variable) => {
    return instance instanceof TFTensor && instance.assign != null &&
      instance.assign instanceof Function;
  }
});