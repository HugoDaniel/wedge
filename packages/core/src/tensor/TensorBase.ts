import { OpNode } from "../graph/types";
import { DataType, TensorData } from "./types";

export abstract class TensorBase {
  public readonly shape: number[];
  public readonly dtype: DataType;
  public readonly nodeName: string;
  public readonly size: number;
  // Abstract operation node:
  public readonly operationNode?: OpNode;

  constructor(data: TensorData, options?: { shape?: number[], dtype?: DataType, nodeName?: string }) {
    // For symbolic tensors (empty data array), shape must be provided
    if ((!data || (Array.isArray(data) && data.length === 0)) && !options?.shape) {
      throw new Error('Shape must be provided when creating a symbolic tensor');
    }

    this.shape = options?.shape || this.inferShape(data);
    this.dtype = options?.dtype || 'float32';
    this.nodeName = options?.nodeName || '';
    this.size = this.getElementCount(this.shape);
    this.initializeData(data);
  }

  protected abstract initializeData(data: TensorData): void;

  protected abstract inferShape(data: TensorData): number[];

  protected getElementCount(shape: number[]): number {
    return shape.reduce((acc, dim) => acc * dim, 1);
  }

  abstract toString(): string;

  abstract dispose(): void;
}