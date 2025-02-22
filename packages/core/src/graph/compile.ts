import { GraphType } from "./types";

export function basicGraphCompile(graph: GraphType) {
  let orderedNodes: TFNode[] = [];

  inputs = executor.mapInputs(inputs);
  executor.checkInputs(inputs);
  executor.checkInputShapeAndType(inputs);
  outputs = executor.mapOutputs(outputs);
  executor.checkOutputs(outputs);

  const outputNodeNames = outputs!.map(name => parseNodeName(name)[0]);
  // const outputNodeNameSet = new Set(outputNodeNames);
  let outputNodes: TFNode[] = outputNodeNames.map(name => executor.graph.nodes[name]);

  // If no outputs are specified, then use the default outputs of the model.
  if (outputNodes.length === 0) {
    throw new Error("Error: outputNodes.length is 0.");
  }

  // Add inputs to the weightMap. This way, weightMaps will have all the weights + input tensors.
  Object.keys(inputs).forEach(name => {
    let inputTensor = inputs[name];

    // Remove batch dimension - do this instead of calling removeBatchDimension
    if (hasBatchDimension) {
      inputTensor = squeeze(inputTensor, [0]);
      inputTensor.
    }

    const [nodeName, index] = parseNodeName(name);
    executor.weightMap[nodeName] = [inputTensor];
  });



  // Original compile function:
  // https://github.com/tensorflow/tfjs/blob/master/tfjs-converter/src/executor/graph_executor.ts#L167
  const compilation = compile(inputs, outputNodes);
  orderedNodes = compilation.orderedNodes;

  // FIXME: TODO: Removing identity nodes is not ideal - if there's an identity node somewhere
  // in the middle of the graph, it will be skipped. Assuming for now they're always unnecessary.
  orderedNodes = orderedNodes.filter(node => node.op !== "Identity");

  console.log("executor.weightMap", executor.weightMap)

  return orderedNodes;
}
