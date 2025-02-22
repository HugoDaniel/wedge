"use client";

import { sidebarMenu } from "@/lib/tests/constants";
import React from "react";

export default function TestPage() {
  const pagesWithoutHomes = Object.keys(sidebarMenu).filter((key) => key !== "/tests");

  return <div>TODO: Use the test components here</div>
}
