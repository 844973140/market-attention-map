"use client";

import { useEffect, useState } from "react";

import type { MarketSelection } from "@/lib/types";

type Headline = { title: string; url: string; source: string; seenAt: string };
type Analysis = { summary: string; sentiment: string; signal: string; risks: string; evidence: string[] };

interface IntelligenceStudioProps { selection: MarketSelection; }

export function IntelligenceStudio({ selection }: IntelligenceStudioProps) {
  const [headlines, setHeadlines] = useState<Headline[]>([]);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [status, setStatus] = useState<"idle" | "loading-news" | "thinking" | "error">("idle");
  const [message, setMessage] = useState("选择一个市场主题，先收集新闻证据，再生成解读。");

  useEffect(() => { setHeadlines([]); setAnalysis(null); setStatus("idle"); setMessage(`已切换至「${selection.name}」。`); }, [selection.id, selection.name]);

  async function collectNews() {
    setStatus("loading-news"); setMessage("正在整理公开新闻线索…");
    try {
      const response = await fetch(`/api/intelligence?market=${encodeURIComponent(selection.name)}`);
      const data = await response.json() as { headlines?: Headline[]; error?: string };
      const nextHeadlines = data.headlines ?? [];
      setHeadlines(nextHeadlines); setMessage(data.error ?? (nextHeadlines.length ? "已收集公开新闻线索。" : "暂未找到可用新闻线索。")); setStatus("idle");
      return nextHeadlines;
    } catch { setStatus("error"); setMessage("新闻服务暂时不可用，请稍后重试。"); return []; }
  }

  async function runAnalysis() {
    const evidence = headlines.length ? headlines : await collectNews();
    setStatus("thinking"); setMessage("正在基于数据与新闻证据生成克制解读…");
    try {
      const response = await fetch("/api/intelligence", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ market: selection, headlines: evidence }) });
      const data = await response.json() as { analysis?: Analysis; error?: string };
      if (!response.ok || !data.analysis) throw new Error(data.error ?? "AI 解读暂时不可用");
      setAnalysis(data.analysis); setStatus("idle"); setMessage("解读已生成；请结合原始来源判断。");
    } catch (error) { setStatus("error"); setMessage(error instanceof Error ? error.message : "AI 解读暂时不可用"); }
  }

  return <section className="intelligence-studio" id="intelligence" aria-live="polite">
    <div className="studio-intro"><div><p className="eyebrow">市场情报台</p><h2>从证据到解释，始终保留判断距离</h2></div><p>当前主题：<strong>{selection.name}</strong></p></div>
    <div className="studio-actions"><button onClick={collectNews} disabled={status !== "idle"}>{status === "loading-news" ? "正在检索" : "收集新闻线索"}</button><button className="primary" onClick={runAnalysis} disabled={status !== "idle"}>{status === "thinking" ? "正在解读" : "生成智能解读"}</button><span>{message}</span></div>
    <div className="studio-grid">
      <article className="evidence-card"><h3>公开新闻证据</h3><p>通过 GDELT 检索当前主题；点击可查看原始页面。</p>{headlines.length ? <ul>{headlines.map((item) => <li key={item.url}><a href={item.url} target="_blank" rel="noreferrer">{item.title}</a><span>{item.source}</span></li>)}</ul> : <div className="empty-state">尚未收集线索</div>}</article>
      <article className="insight-card"><h3>智能解读</h3><p>由 DeepSeek 基于当前数据和已收集标题生成。</p>{analysis ? <><div className="insight-signal"><span>{analysis.sentiment}</span><strong>{analysis.signal}</strong></div><p className="insight-summary">{analysis.summary}</p><p className="risk-note">注意：{analysis.risks}</p></> : <div className="empty-state">先收集线索，再生成解读</div>}</article>
    </div>
    <p className="studio-disclaimer">智能解读只服务于新闻、搜索与情绪辅助，不参与市场关注度指数计算，也不构成投资建议。</p>
  </section>;
}
