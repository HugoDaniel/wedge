import { Model } from "@/lib/types";
import { NNShaders } from "@/lib/wedge/NNShaders";
import { NNShadersOptions, WebGLOpNode } from "@/lib/wedge/types";
import { FC } from "react";


type ViewOpNodeProps = {
  opNode: WebGLOpNode;
  onClick: () => void;
  isActive: boolean;
}

type ModelStatsProps = {
  model: Model;
  nnShadersOptions?: NNShadersOptions;
}

export const ModelStats: FC<ModelStatsProps> = ({
  model,
  nnShadersOptions }) => {
  const opNodesIterableIterator = (model as NNShaders).opNodeMap.values();
  const opNodes = Array.from(opNodesIterableIterator);
  const numberOfOpNodes = opNodes.length

  return (
    <>
      <div className={"model-stats"}>
        Number of Op Nodes: {numberOfOpNodes}
      </div >
    </>
  );

}