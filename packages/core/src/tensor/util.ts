export function makeTensor(
  values: DataValues, shape: number[], dtype: DataType): TFTensor {
  if (values == null) {
    throw new Error('Values passed to engine.makeTensor() are null');
  }
  dtype = dtype || 'float32';
  backend = backend || this.backend;
  let backendVals = values as BackendValues;
  if (dtype === 'string' && util.isString(values[0])) {
    backendVals = (values as string[]).map(d => encodeString(d));
  }
  const dataId = backend.write(backendVals, shape, dtype);
  const t = new TFTensor(shape, dtype, dataId, this.nextTensorId());
  // this.trackTensor(t, backend);

  // Count bytes for string tensors.
  if (dtype === 'string') {
    const info = this.state.tensorInfo.get(dataId);
    const newBytes = bytesFromStringArray(backendVals as Uint8Array[]);
    this.state.numBytes += newBytes - info.bytes;
    info.bytes = newBytes;
  }
  return t;
}