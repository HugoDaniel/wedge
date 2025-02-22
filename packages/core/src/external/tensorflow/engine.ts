/*

Original:
https://github.com/tensorflow/tfjs/blob/master/tfjs-core/src/engine.ts
*/



/**
 * Internal method used by public APIs for tensor creation. Makes a new
 * tensor with the provided shape, dtype and values. It always
 * creates a new data id and writes the values to the underlying backend.
 */
export class ENGINE {
  /*
    trackTensor(a: TFTensor, backend: KernelBackend): void {
      this.state.numTensors++;
      if (a.dtype === 'string') {
        this.state.numStringTensors++;
      }
      // Bytes for complex numbers are counted by their components. Bytes for
      // string tensors are counted when writing values.
      let bytes = 0;
      if (a.dtype !== 'complex64' && a.dtype !== 'string') {
        bytes = a.size * util.bytesPerElement(a.dtype);
      }
      this.state.numBytes += bytes;
  
      if (!this.state.tensorInfo.has(a.dataId)) {
        this.state.numDataBuffers++;
        this.state.tensorInfo.set(a.dataId, {
          backend: backend || this.backend,
          dtype: a.dtype,
          shape: a.shape,
          bytes
        });
      }
  
      if (!(a instanceof Variable)) {
        this.track(a);
      }
    }

  
  static makeTensor(
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
    this.trackTensor(t, backend);

    // Count bytes for string tensors.
    if (dtype === 'string') {
      const info = this.state.tensorInfo.get(dataId);
      const newBytes = bytesFromStringArray(backendVals as Uint8Array[]);
      this.state.numBytes += newBytes - info.bytes;
      info.bytes = newBytes;
    }
    return t;
  }
  */
}