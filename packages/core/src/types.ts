import { GraphWebGL } from "./backends/webgl/types";
import { GraphNode } from "./graph/types";
import { OpInput } from "./ops/types";
import { TensorWebGL } from "./tensor/TensorWebGL";

export type DataFormat = "NHWC" | "HWC" | "VEC";

export type CustomShapeUpdate = (
  shapeWithPaddedChannels: number[],
  currentWidth: number,
  currentHeight: number,
  numRenderTargets: number,
  nodeName: string) => [number, number, number, number]

export type NamedArrayBufferViewMap = {
  [key: string]: ArrayBufferView;
};

export type TypedArray = Float32Array | Int32Array | Uint8Array;

export type CompilationOptions = {
  padChannels?: boolean;
}

export type TensorType = TensorWebGL;

export interface WedgeBase {
  backend: "webgl" | "webgpu";
  originalGraph: GraphWebGL | null;
  compiledGraph: GraphWebGL | null;
  orderedNodes: GraphNode[] | [];
  hasGraphUpdatedSinceLastRun: boolean;

  // orderedNodes: GraphNode[] = [];
  // nodeWebGLDataMap: NodeWebGLDataMap = new Map();
  // opNodeMap: WebGLOpNodeMap = new Map();
  // opNodeWithProgramMap: WebGLOpNodeWithProgramMap = new Map();
  // inputsNamedTensorMap: NamedTensorMap = { };
  // inputTensorNames: Set<string> = new Set();

  compile: (compilationOptions?: CompilationOptions) => void;

  run: (inputRawData: ArrayBufferView[]) => Float32Array;

  // Creates a tensor.
  tensor: (data: OpInput, shape?: number[]) => TensorType;

  loadGraphModel: (modelPath: string) => Promise<void>;
}