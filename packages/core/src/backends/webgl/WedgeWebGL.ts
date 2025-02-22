import * as tf from '@tensorflow/tfjs';
import { NamedTensorMap } from '@tensorflow/tfjs-converter/dist/data/types';
import { GraphNode } from '../../graph/types';
import { TensorWebGL } from '../../tensor/TensorWebGL';
import { CompilationOptions, TensorData, WedgeBase } from '../../types';
import { initWebGLData, updateUniformsForProgram } from './backends/webgl/webGLData';
import { createAndBindFramebuffer, updateFramebufferTextureLayer } from './buffersAndTextures';
import { defaultOptions } from './constants';
import { loadGraphModel } from './loaders/graphModel';
import { createOpNodeMapPrograms } from './modelHelpers';
import { initWebGL } from './setupShadersAndWebGL';
import { GraphWebGL, NodeWebGLDataMap, WebGLDataTextureArray, WebGLOpNodeMap, WebGLOpNodeWithProgramMap, WebGLOps, WedgeOptions } from './types';

/*

Info:
loadGraphModel source code:
https://github.com/tensorflow/tfjs/blob/f0f981fe306bf548e300536aca485c0ffdd6619e/tfjs-converter/src/executor/graph_model.ts#L624

*/

export class WedgeWebGL implements WedgeBase, WebGLOps {
  public originalGraph: GraphWebGL | null;
  public compiledGraph: GraphWebGL | null;
  public orderedNodes: GraphNode[] | [];
  public hasGraphUpdatedSinceLastRun: boolean;

  public canvas: OffscreenCanvas | HTMLCanvasElement | null = null;
  public gl: WebGL2RenderingContext;

  public maxColorAttachments: number;

  public frameBuffer: WebGLFramebuffer | null = null;
  public nodeWebGLDataMap: NodeWebGLDataMap = new Map();
  public opNodeMap: WebGLOpNodeMap = new Map();
  public opNodeWithProgramMap: WebGLOpNodeWithProgramMap = new Map();
  public inputsNamedTensorMap: NamedTensorMap = {};
  public inputTensorNames: Set<string> = new Set();

  public initialSetupComplete = false;

  public finalOutputData: WebGLDataTextureArray | null = null;

  public layersModel: tf.LayersModel | null = null;

  constructor(
    public options: WedgeOptions = defaultOptions,
  ) {
    // 1st step - create offscreen canvas.
    const { canvas, gl, maxColorAttachments } = initWebGL(
      this.options.canvasWidth,
      this.options.canvasHeight,
      this.options.viewportMaxSize);

    this.canvas = canvas;
    this.gl = gl;
    this.maxColorAttachments = maxColorAttachments;

    for (const renderTargetBreakpoint of this.options.renderTargetBreakpoints) {
      const numberOfRenderTargets = renderTargetBreakpoint.numberOfRenderTargets;
      if (numberOfRenderTargets > this.maxColorAttachments) {
        throw new Error("Error: numberOfRenderTargets is greater than maxColorAttachments. numberOfRenderTargets: " + numberOfRenderTargets + ", maxColorAttachments: " + this.maxColorAttachments);
      }
    }
  }
  backend: 'webgl' | 'webgpu';
  compile: (compilationOptions?: CompilationOptions) => void;

  cleanGraph: GraphWebGL | null;

  tensor(data: TensorData, shape?: number[]): TensorWebGL {
    return new TensorWebGL(data, this.gl, { shape });
  }

  readOutput(): Float32Array {
    // hmmm maybe this checkFramebufferStatus call is not necessary?
    // if (this.gl.checkFramebufferStatus(this.gl.FRAMEBUFFER) !== this.gl.FRAMEBUFFER_COMPLETE) {
    //   console.error('readOutput - Framebuffer not complete');
    //   return new Float32Array(0);
    // }

    let [outputTexturesWidth, outputTexturesHeight, numberOfTextures, _] = this.finalOutputData!.RGBATextureShape;

    if (!outputTexturesWidth || !outputTexturesHeight || !numberOfTextures) {
      throw new Error("Error: outputTexturesWidth, outputTexturesHeight, or numberOfTextures is not defined.");
    }

    let sizePerLayer = outputTexturesWidth * outputTexturesHeight * 4;
    let output = new Float32Array(sizePerLayer * numberOfTextures);

    for (let layer = 0; layer < numberOfTextures; layer++) {
      // Reattach the layer to the framebuffer for reading.
      this.gl.framebufferTextureLayer(
        this.gl.FRAMEBUFFER,
        this.gl.COLOR_ATTACHMENT0 + layer,
        this.finalOutputData!.texture,
        0,  // mipmap level
        layer
      );

      // Set the read buffer to the current attachment.
      this.gl.readBuffer(this.gl.COLOR_ATTACHMENT0 + layer);

      // Read the pixels from the currently attached layer.
      let layerData = new Float32Array(sizePerLayer);
      this.gl.readPixels(0, 0, outputTexturesWidth, outputTexturesHeight, this.gl.RGBA, this.gl.FLOAT, layerData);

      // Copy the layer data into the output array at the correct offset.
      output.set(layerData, layer * sizePerLayer);
    }

    return output;
  }

  // The LayersModel's execute function is overridden. Original model's execute function:
  // https://github.com/tensorflow/tfjs/blob/master/tfjs-layers/src/engine/training.ts#L936
  layersModelExecute(inputRawData: ArrayBufferView[]): Float32Array {
    if (!this.layersModel) {
      throw new Error("layersModelExecute - error: layersModel is not defined.");
    }

    if (!this.initialSetupComplete) {
      throw new Error("layersModelExecute - error: model not initialized. Call createWedge first.");
    }

    const result = this.run(inputRawData);
    return result;
  }


  initializeGraphModel(): void {
    if (!this.graphModel) {
      throw new Error("initializeGraphModel - error: graphModel is not defined.");
    }

    if (!this.initialSetupComplete) {
      throw new Error("initializeGraphModel - error: model not initialized. Call createWedge first.");
    }

    const { opNodeMap, nodeWebGLDataMap } = initWebGLData(
      this.gl,
      this.executor.weightMap,
      this.orderedNodes,
      "GraphModel",
      this.options);

    // Create WebGL programs for all the operations.
    this.opNodeWithProgramMap = createOpNodeMapPrograms(opNodeMap, this.gl, this.executor.weightMap);
    this.opNodeMap = opNodeMap;
    this.nodeWebGLDataMap = nodeWebGLDataMap;
    this.frameBuffer = createAndBindFramebuffer(this.gl);
    this.initialSetupComplete = true;
  }

  // This is a modified version of both the graph model and the executor's execute function.
  // https://github.com/tensorflow/tfjs/blob/master/tfjs-converter/src/executor/graph_model.ts#L507
  // https://github.com/tensorflow/tfjs/blob/master/tfjs-converter/src/executor/graph_executor.ts#L231
  graphModelExecute(inputRawData: ArrayBufferView[]): ArrayBufferView {
    if (!this.initialSetupComplete) {
      this.initializeGraphModel();
    }

    const result = this.run(inputRawData);

    return result;
  }

  async loadGraphModel(modelPath: string): Promise<void> {
    const { model, executor, orderedNodes, inputsNamedTensorMap, inputTensorNames } = await loadGraphModel(modelPath);

    return;
  }

  run(inputRawData: ArrayBufferView[]): Float32Array {
    if (!this.gl) {
      throw new Error("run - error: gl is not defined.");
    }

    //
    ////
    ////// Run all the WebGL programs.
    let opNodesRan = 0;
    this.opNodeWithProgramMap.forEach((opNodeWithProgam, opNodeName) => {
      this.gl.useProgram(opNodeWithProgam.programInfo.program);

      updateUniformsForProgram(this.gl, opNodeWithProgam, this.inputTensorNames, inputRawData, this.options);
      updateFramebufferTextureLayer(this.gl, opNodeWithProgam.opNode.output!);

      // There are 2 triangles that make up the square that covers the texture / canvas. 
      // 2 triangles means 6 vertices - hence the following 6:
      this.gl.drawArrays(this.gl.TRIANGLES, 0, 6);
      this.finalOutputData = opNodeWithProgam.opNode.output;

      opNodesRan++;
    });

    // Read the output from the framebuffer
    const outputData = this.readOutput();

    // const dummyArray = new Float32Array(100);

    return outputData;
  }


}
