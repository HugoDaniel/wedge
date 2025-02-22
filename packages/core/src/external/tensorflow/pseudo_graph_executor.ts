// https://github.com/tensorflow/tfjs/blob/master/tfjs-converter/src/executor/graph_executor.ts



import { ISignatureDef } from './compiled_api';
import { Graph } from './types';

export class PseudoGraphExecutor {
  // private compiledMap = new Map<string, ReturnType<typeof this.compile>>();
  // private parseNodeNameCache = new Map<string, [string, number, string?]>();
  // private _weightMap: NamedTensorsMap = {};
  // private _weightIds: number[];
  private _signature: ISignatureDef | undefined;
  // private _inputs: Node[];
  // private _outputs: Node[];
  // private _initNodes: Node[];  // Internal init nodes to start initialization.
  // private SEPARATOR = ',';
  // private _functions: { [key: string]: Graph } = {};
  // private _functionExecutorMap: { [key: string]: FunctionExecutor } = {};

  /**
   *
   * @param graph Graph the model or function graph to be executed.
   * @param parent When building function exector you need to set the parent
   * executor. Since the weights and function executor maps are set at parant
   * level, that function executor can access the function maps and weight maps
   * through the parent.
   */
  constructor(public graph: Graph) { //private parent?: GraphExecutor) {
    // this._outputs = graph.outputs;
    // this._inputs = graph.inputs;
    // this._initNodes = graph.initNodes;
    this._signature = graph.signature;
    // this._functions = graph.functions;
    // // create sub-graph executors
    // if (graph.functions != null) {
    //   Object.keys(graph.functions).forEach(name => {
    //     this._functionExecutorMap[name] =
    //       new GraphExecutor(graph.functions[name], this);
    //   });
    // }
  }


}