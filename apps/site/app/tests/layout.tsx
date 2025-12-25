"use client";

import { CustomSidebarLayout } from "@/components/test/CustomSidebarLayout";

export default function TestsLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  if (process.env.NODE_ENV !== "development") {
    return <div>What are you doing here?</div>
  }

  return (
    <CustomSidebarLayout>
      {children}
    </CustomSidebarLayout>
  );
}
