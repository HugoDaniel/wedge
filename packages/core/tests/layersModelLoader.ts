import * as tf from '@tensorflow/tfjs';
import { Node } from '@tensorflow/tfjs-converter/dist/operations/types';
import { createDummyInputsNamedTensorMap, processLayersModel } from '../src/modelHelpers';

export function getLayersModelWeightNodes(weightTensors: tf.Tensor[], layerName: string): Node[] {
  return weightTensors.map((tensor, index) => ({
    name: createWeightName(layerName, index),
    op: 'Const',
    inputNames: [],
    inputs: [],
    attrParams: {},
    category: 'custom', // not sure what's the correct category for weights. But, it shouldn't matter much.
    inputParams: {},
    children: []
  }));
}

export function getLayersModelInputNodes(layer: layers.Layer, orderedNodes: Node[]) {
  // Assuming each layer takes input from the immediately preceding layer in a linear topology
  const inputNodeNames = layer.inboundNodes.map(node => node.inboundLayers.map(layer => layer.name)).flat();
  const inputNodes = orderedNodes.filter(node => inputNodeNames.includes(node.name));
  const layerClassName = layer.constructor.name;

  if (!inputNodes.length && layerClassName !== 'InputLayer') {
    // If no input nodes are found, and if the layer class name is not InputLayer, then throw an error.
    throw new Error("Error: no input nodes found for layer: " + layer.name);
  }

  return inputNodes;
}

type ProcessLayersModelReturn = {
  orderedNodes: Node[],
  layersModelWeightMap: NamedTensorsMap
}

export function processLayersModel(
  layersModel: LayersModel,
  inputs: NamedTensorMap,
  hasBatchDimension: boolean): ProcessLayersModelReturn {
  const layersModelWeightMap: NamedTensorsMap = {};
  const orderedNodes: Node[] = [];

  for (let i = 0; i < layersModel.layers.length; i++) {
    const layer = layersModel.layers[i];
    const layerName = `${layer.name}`;
    const weightTensors = layer.getWeights(); // Get the weight tensors of the layer

    if (weightTensors.length > 2) {
      // Currently only support max 2 weights - one for the kernel and one for the bias. 
      throw new Error("Error: only 2 weights max. are supported right now. Layer " + layerName + " has " + weightTensors.length + " weights.");
    }

    const inputNodes = getLayersModelInputNodes(layer, orderedNodes);
    const weightNodes = getLayersModelWeightNodes(weightTensors, layerName);
    const opName = mapLayerClassesToOpName(layer.constructor.name);
    const attrParams = getAttrParams(layer);

    const finalInputNodes: Node[] = inputNodes.concat(weightNodes);

    if (layer.constructor.name === 'InputLayer') {
      // Handle InputLayer specially
      const inputLayerNode: Node = {
        attrParams,
        name: layerName,
        op: opName,
        inputNames: [], // InputLayer does not have incoming nodes
        inputs: [],
        category: 'custom',
        inputParams: {},
        children: []
      };

      // Add the input tensors to the weight map. This is also done for GraphModels.
      Object.keys(inputs).forEach(name => {
        // If the input tensor name does not match the layer name, then skip it.
        if (name !== layerName) {
          return;
        }

        let inputTensor = inputs[name];
        // Remove batch dimension - do this instead of calling removeBatchDimension
        if (hasBatchDimension) {
          inputTensor = squeeze(inputTensor, [0]);
        }

        layersModelWeightMap[layerName] = [inputTensor];
      });

      // Add the input layer node to the start of the ordered nodes list.
      orderedNodes.unshift(inputLayerNode);
    } else {
      const opNode: Node = {
        attrParams,
        name: layerName,
        op: opName,
        inputNames: finalInputNodes.map(node => node.name),
        inputs: finalInputNodes,
        category: 'custom',
        inputParams: {},
        children: []
      };

      orderedNodes.push(opNode)
    }

    if (weightTensors.length !== weightNodes.length) {
      throw new Error("Error: weightTensors and weightNodes length mismatch for layer: " + layerName);
    }

    // Regardless of layer type, process weights
    for (let i = 0; i < weightTensors.length; i++) {
      // XXX: I don't think we need the weights in the orderedNodes list?
      // orderedNodes.push(weightNodes[i]);
      layersModelWeightMap[createWeightName(layerName, i)] = [weightTensors[i]];
    }
  }

  return { orderedNodes, layersModelWeightMap };
}


export async function loadLayersModel(model: tf.LayersModel, hasBatchDimension: boolean = true): Promise<{
  model: tf.LayersModel;
  orderedNodes: Node[];
  inputsNamedTensorMap: tf.NamedTensorMap;
  inputTensorNames: Set<string>;
  layersModelWeightMap: tf.NamedTensorMap;
}> {
  // Create input tensor map and get tensor names
  const inputsNamedTensorMap = createDummyInputsNamedTensorMap(model);
  const inputTensorNames = new Set(Object.keys(inputsNamedTensorMap));

  // Process the layers model
  const { layersModelWeightMap, orderedNodes } = processLayersModel(
    model,
    inputsNamedTensorMap,
    hasBatchDimension
  );

  return {
    model,
    orderedNodes,
    inputsNamedTensorMap,
    inputTensorNames,
    layersModelWeightMap
  };
} 