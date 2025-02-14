"use client";

import { sidebarMenu } from "@/lib/tests/constants";
import { MultiplePageOverview, TestContainer } from "react-browser-tests";

export default function TestPage() {
  const pagesWithoutHomes = Object.keys(sidebarMenu).filter((key) => key !== "/tests");

  return <TestContainer>
    <p>Multiple test page overview:</p>
    <MultiplePageOverview
      urls={pagesWithoutHomes}
      singleIframeMode
    />
  </TestContainer>
}
