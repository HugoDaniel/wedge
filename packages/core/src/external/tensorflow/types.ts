import { ParamType, ValueType } from "../../graph/types";
import { TFTensor } from "./core/tensor";

export interface DataTypeMap {
  float32: Float32Array;
  int32: Int32Array;
  bool: Uint8Array;
  complex64: Float32Array;
  string: string[];
}

export type DataType = keyof DataTypeMap;

export type Category = 'arithmetic' | 'basic_math' | 'control' | 'convolution' | 'creation' | 'custom' | 'dynamic' | 'evaluation' | 'graph' | 'hash_table' | 'image' | 'logical' | 'matrices' | 'normalization' | 'ragged' | 'reduction' | 'slice_join' | 'sparse' | 'spectral' | 'string' | 'transformation';

export declare enum Rank {
  R0 = "R0",
  R1 = "R1",
  R2 = "R2",
  R3 = "R3",
  R4 = "R4",
  R5 = "R5",
  R6 = "R6"
}

// export interface TFTensor<R extends Rank = Rank> {
//   abs<T extends TFTensor>(this: T): T;
// }

export declare interface ParamValue {
  value?: ValueType;
  type: ParamType;
}

export type NamedTensorMap = {
  [key: string]: TFTensor;
};

export type NamedTensorsMap = {
  [key: string]: TFTensor[];
};

/*
export type TFTensorArrayMap = {
  [key: number]: TFTensorArray;
};
export type TFTensorListMap = {
  [key: number]: TFTensorList;
};
export type HashTableMap = {
  [key: number]: HashTable;
};
*/
export interface TFTensorInfo {
  name: string;
  shape?: number[];
  dtype: DataType;
}


export interface ModelTensorInfo {
  // Name of the tensor.
  name: string;
  // Tensor shape information, Optional.
  shape?: number[];
  // Data type of the tensor.
  dtype: DataType;
  // TensorFlow native Data type of the tensor.
  tfDtype?: string;
}

export interface ModelPredictConfig {
  /**
   * Optional. Batch size (Integer). If unspecified, it will default to 32.
   */
  batchSize?: number;

  /**
   * Optional. Verbosity mode. Defaults to false.
   */
  verbose?: boolean;
}

// For mapping input or attributes of NodeDef into TensorFlow.js op param.
export declare interface ParamMapper {
  // tensorflow.js name for the field, it should be in camelcase format.
  name: string;
  type: ParamType;
  defaultValue?: ValueType;
  notSupported?: boolean;
}

// For mapping the input of TensorFlow NodeDef into TensorFlow.js Op param.
export declare interface InputParamMapper extends ParamMapper {
  // The first number is the starting index of the param, the second number is
  // the length of the param. If the length value is positive number, it
  // represents the true length of the param. Otherwise, it represents a
  // variable length, the value is the index go backward from the end of the
  // array.
  // For example `[0, 5]`: this param is the array of input tensors starting at
  // index 0 and with the length of 5.
  // For example `[1, -1]`: this param is the array of input tensors starting at
  // index 1 and with the `inputs.length - 1`.
  // Zero-based index at where in the input array this param starts.
  // A negative index can be used, indicating an offset from the end of the
  // sequence. slice(-2) extracts the last two elements in the sequence.
  start: number;
  // Zero-based index before where in the input array the param ends. The
  // mapping is up to but not including end. For example, start = 1, end = 4
  // includes the second element through the fourth element (elements indexed 1,
  // 2, and 3). A negative index can be used, indicating an offset from the end
  // of the sequence. start = 2, end = -1 includes the third element through the
  // second-to-last element in the sequence. If end is omitted, end is set to
  // start + 1, the mapping only include the single element at start index. If
  // end is set to 0, the mapping is through the end of the input array
  // (arr.length). If end is greater than the length of the inputs, mapping
  // inncludes through to the end of the sequence (arr.length).
  end?: number;
}

// For mapping the attributes of TensorFlow NodeDef into TensorFlow.js op param.
export declare interface AttrParamMapper extends ParamMapper {
  // TensorFlow attribute name, this should be set if the tensorflow attribute
  // name is different form the tensorflow.js name.
  tfName?: string;
  // TensorFlow deprecated attribute name, this is used to support old models.
  tfDeprecatedName?: string;
}


export declare interface OpMapper {
  tfOpName: string;
  category?: Category;
  inputs?: InputParamMapper[];
  attrs?: AttrParamMapper[];
  outputs?: string[];
  customExecutor?: any; // OpExecutor;
}