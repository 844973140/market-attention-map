import type { Metadata } from "next";

import "@/app/globals.css";

export const metadata: Metadata = {
  title: "饼干市场热力图",
  description: "通过公开市场数据，观察全球金融市场当前关注焦点。",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
