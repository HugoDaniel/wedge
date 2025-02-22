import { Graph } from "../../graph/types";
import { OpParams, Ops } from "../../ops/types";
import { TensorWebGL } from "../../tensor/TensorWebGL";
import { DataArray } from "../../tensor/types";

export type Vector3String = {
  x: string;
  y: string;
  z: string;
};

export type ModelConfig = {
  backend: "webgl";
  runtime: "tfjs" | "mediapipe" | "tfjs-tflite" | "wedge";
  url: string;
}

export type GlobalState = {
  targetFPS: number;
  cameraWidth: number;
  cameraHeight: number;
  isVideoStreamLoaded: boolean;
  isCameraCanvasLoaded: boolean;
  modelConfig: ModelConfig;
}

export type InitWebGLReturn = {
  canvas: OffscreenCanvas;
  gl: WebGL2RenderingContext;
  maxColorAttachments: number;
}

export type ProgramInfo = {
  program: WebGLProgram;
  attribLocations: {
    vertexPosition: number;
  };
}

// FBO - Frame Buffer Object.
export type TexturesAndOutputFBO = {
  inputTextures: WebGLTexture[],
  outputTextures: WebGLTexture[],
  weightTextures: WebGLTexture[],
  frameBuffer: WebGLFramebuffer,
}

export type OutputShape = number[];
export type InputShape = number[];

export type ModelType = "GraphModel" | "LayersModel" | "onnx" | "tflite";

export type WedgeOptions = {
  // XXX: NOTE: these width and height values may not be necessary? As we're drawing to a texture, not the canvas.
  canvasWidth: number;
  canvasHeight: number;
  viewportMaxSize: number;

  hasBatchDimension: boolean;

  transformations: {
    padChannels: boolean;
  }
  renderTargetBreakpoints: {
    outputTextureElementCount: number;
    numberOfRenderTargets: number;
  }[];
}

export type WebGLDataNodeBase = {
  nodeName: string;
  uniformName: string;
};

export type WebGLDataNodeNonTexture = WebGLDataNodeBase & {
  webGLType: "float" | "vec2" | "vec3" | "vec4";
  data: number[];
};

// Note: For now, webGL textures do not have a data field. This is because textures can be too large.
export type WebGLDataNodeTextureBase = WebGLDataNodeBase & {
  texture: WebGLTexture;

  // Original dimensions:
  originalShape: number[];
  originalElementCount: number;

  // Texture dimensions:
  RGBATextureShape: number[];
  RGBATextureElementCount: number;
}

export type WebGLDataNodeTexture = WebGLDataNodeTextureBase & {
  webGLType: "sampler2D";
};

export type WebGLDataNodeTextureArray = WebGLDataNodeTextureBase & {
  webGLType: "sampler2DArray";
};

// export type WebGLDataNode = WebGLDataNodeNonTexture | WebGLDataNodeTexture | WebGLDataNodeTextureArray;

export type ArithmeticOpName = "AddV2" | "Mul"

export type SingleInputBasicOpName = "Relu"

// Here I'm creating 2 types for op names, as I feel some of the graph model ops really shouldn't be called ops.
// For example, Placeholder and Const. These are more like "data" nodes.
export type OpName = ArithmeticOpName
  | SingleInputBasicOpName
  | "Conv2D"
  | "_FusedConv2D"
  | "DepthwiseConv2D"
  | "DepthwiseConv2dNative"
  | "FusedDepthwiseConv2dNative"
  | "ResizeBilinear"
  | "NotSupported"

export type GraphModelOpNames = OpName | "Placeholder" | "Const"

export type LayersModelLayerClass = "Conv2D"
  | "DenseLayer"
  | "InputLayer"
  | "MaxPooling2DLayer"
  | "ReLU"
  | "DepthwiseConv2D"
  | "Add"


// Operation nodes have a corresponding WebGL program - with vertex and fragment shaders.
export type WebGLOpNode = {
  // node: Node;
  name: string;
  // Operations have input(s) and output(s) - and possibly weight(s).
  inputs: (TensorWebGL | null)[];
  output: WebGLDataNodeTextureArray | null;
  weights: WebGLDataNodeTextureArray[];
  opParams: OpParams | null;
  type: OpName;
  fsSource: string;
};

export type WebGLOpNodeWithProgram = {
  opNode: WebGLOpNode;
  programInfo: ProgramInfo;
}

export type WebGLOpNodeWithProgramMap = Map<string, WebGLOpNodeWithProgram>;

// export type WebGLData = {
//   isOperation: boolean;
//   shape: number[];
//   texture: WebGLTexture | null;
//   scalar: number | null;
// }

export type WebGLOpInput = TensorWebGL | DataArray;
export type WebGLOpOutput = TensorWebGL;

export type WebGLOps = Ops<WebGLOpInput, WebGLOpOutput>
export type GraphWebGL = Graph<TensorWebGL>;
