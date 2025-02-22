import { parseNodeName } from "../external/tensorflow/executor_utils";
import { NamedTensorMap } from "../external/tensorflow/types";
import { assert } from "../util";

export function parseNodeName(
  name: string, context?: ExecutionContext): [string, number, string?] {
  if (name === '') {
    return ['', 0, undefined];
  }

  const isCacheEnabled = context != null && context.parseNodeNameCache != null;
  if (isCacheEnabled) {
    const cachedResult = context.parseNodeNameCache.get(name);
    if (cachedResult != null) {
      return cachedResult;
    }
  }
  const parts = name.split(':');
  let result: [string, number, string?];
  if (parts.length === 1) {
    result = [name, 0, undefined];
  } else {
    const nodeName = parts[0];
    const outputName = parts.length === 3 ? parts[1] : undefined;
    const index = Number(parts[parts.length - 1]);
    result = [nodeName, index, outputName];
  }
  if (isCacheEnabled) {
    context.parseNodeNameCache.set(name, result);
  }
  return result;
}

export function checkInputShapeAndType(inputs: NamedTensorMap) {
  Object.keys(inputs).forEach(name => {
    const input = inputs[name];
    const [nodeName,] = parseNodeName(name);
    const node = this.graph.nodes[nodeName];
    if (node.attrParams['shape'] && node.attrParams['shape'].value) {
      const shape = node.attrParams['shape'].value as number[];
      const match = shape.length === input.shape.length &&
        input.shape.every(
          (dim, index) => shape[index] === -1 || shape[index] === dim);
      assert(
        match,
        () => `The shape of dict['${node.name}'] provided in ` +
          `model.execute(dict) must be [${shape}], but was ` +
          `[${input.shape}]`);
    }
    if (node.attrParams['dtype'] && node.attrParams['dtype'].value) {
      assert(
        input.dtype === node.attrParams['dtype'].value as string,
        () => `The dtype of dict['${node.name}'] provided in ` +
          `model.execute(dict) must be ` +
          `${node.attrParams['dtype'].value}, but was ${input.dtype}`);
    }
  });
}

export function mapInputs(inputs: NamedTensorMap) {
  const result: NamedTensorMap = {};
  for (const inputName in inputs) {
    const tensor = this._signature?.inputs?.[inputName];
    if (tensor != null) {

      if (!tensor.name) {
        throw new Error("tensor.name is null :(");
      }

      result[tensor.name] = inputs[inputName];
    } else {
      result[inputName] = inputs[inputName];
    }
  }
  return result;
}

export function checkInputs(inputs: NamedTensorMap) {
  const notInGraph = Object.keys(inputs).filter(name => {
    const [nodeName] = parseNodeName(name);
    return this.graph.nodes[nodeName] == null;
  });
  if (notInGraph.length > 0) {
    throw new Error(
      `The dict provided in model.execute(dict) has ` +
      `keys: [${notInGraph}] that are not part of graph`);
  }
}

export function mapOutputs(outputs: string[]): string[] {
  const finalResult = outputs.map(name => {
    const tensor = this._signature?.outputs?.[name];
    if (tensor != null) {
      return tensor.name;
    }
    return name;
  }, {})

  return finalResult.filter(name => name != null) as string[];
}

export function checkOutputs(outputs: string[]): void {
  outputs.forEach(name => {
    const [normalizedName] = parseNodeName(name);
    if (!this.graph.nodes[normalizedName]) {
      throw new Error(`The output '${name}' is not found in the graph`);
    }
  });
}