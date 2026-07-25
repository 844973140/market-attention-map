import { NextRequest, NextResponse } from "next/server";

import type { MarketNode } from "@/lib/types";

const newsTimeoutMs = 8_000;
const aiTimeoutMs = 18_000;

type Headline = { title: string; url: string; source: string; seenAt: string };

function fallbackName(market: string) {
  return market.replace(/[<>]/g, "").slice(0, 48);
}

async function fetchHeadlines(market: string): Promise<Headline[]> {
  const query = new URLSearchParams({ query: fallbackName(market), mode: "artlist", format: "json", maxrecords: "8", timespan: "7d" });
  const response = await fetch(`https://api.gdeltproject.org/api/v2/doc/doc?${query}`, {
    next: { revalidate: 300 }, signal: AbortSignal.timeout(newsTimeoutMs),
  });
  if (!response.ok) throw new Error("新闻检索暂时不可用");
  const payload = await response.json() as { articles?: Array<Record<string, unknown>> };
  return (payload.articles ?? []).flatMap((article) => {
    const title = typeof article.title === "string" ? article.title.trim() : "";
    const url = typeof article.url === "string" ? article.url : "";
    if (!title || !url) return [];
    return [{ title: title.slice(0, 180), url, source: typeof article.domain === "string" ? article.domain : "GDELT", seenAt: typeof article.seendate === "string" ? article.seendate : "" }];
  });
}

function safeAnalysis(value: unknown) {
  const data = value && typeof value === "object" ? value as Record<string, unknown> : {};
  const read = (key: string, max: number) => typeof data[key] === "string" ? data[key].slice(0, max) : "";
  const evidence = Array.isArray(data.evidence) ? data.evidence.filter((item): item is string => typeof item === "string").slice(0, 3).map((item) => item.slice(0, 160)) : [];
  return { summary: read("summary", 320) || "暂未形成可用解读。", sentiment: read("sentiment", 24) || "中性", signal: read("signal", 80) || "关注信号待确认", risks: read("risks", 180) || "公开信息可能滞后或不完整。", evidence };
}

export async function GET(request: NextRequest) {
  const market = fallbackName(request.nextUrl.searchParams.get("market") ?? "");
  if (!market) return NextResponse.json({ error: "缺少市场主题" }, { status: 400 });
  try {
    const headlines = await fetchHeadlines(market);
    return NextResponse.json({ market, headlines, source: "GDELT 新闻检索", updatedAt: new Date().toISOString() }, { headers: { "Cache-Control": "private, max-age=120" } });
  } catch (error) {
    return NextResponse.json({ market, headlines: [], source: "GDELT 新闻检索", error: error instanceof Error ? error.message : "新闻检索暂时不可用" }, { status: 200 });
  }
}

export async function POST(request: NextRequest) {
  const key = process.env.DEEPSEEK_API_KEY?.trim();
  if (!key) return NextResponse.json({ error: "尚未配置 AI 服务" }, { status: 503 });
  const body = await request.json().catch(() => null) as { market?: MarketNode; headlines?: Headline[]; question?: string } | null;
  if (!body?.market) return NextResponse.json({ error: "缺少市场数据" }, { status: 400 });
  const headlines = (body.headlines ?? []).slice(0, 8).map((item) => item.title);
  const market = body.market;
  const prompt = JSON.stringify({ market: { name: market.name, category: market.category, attentionScore: market.attentionScore, priceChange24h: market.priceChange24h, volumeChange24h: market.volumeChange24h, newsHeat: market.newsHeat, dataStatus: market.dataStatus }, headlines, question: (body.question ?? "").slice(0, 240) });
  try {
    const response = await fetch("https://api.deepseek.com/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${key}` },
      body: JSON.stringify({ model: "deepseek-v4-flash", temperature: 0.3, max_tokens: 500, response_format: { type: "json_object" }, messages: [
        { role: "system", content: "你是中文金融信息解读助手。只解释给定市场关注度数据与新闻标题，描述已经观察到的现象。严禁预测未来、使用‘可能/预计/将会’等预测词，严禁给买卖建议。必须输出 JSON：{summary:string,sentiment:string,signal:string,risks:string,evidence:string[]}。sentiment 仅可为偏正面、偏负面或中性。" },
        { role: "user", content: `请基于以下 JSON 生成克制、可核验的说明：${prompt}` },
      ] }), signal: AbortSignal.timeout(aiTimeoutMs),
    });
    if (!response.ok) throw new Error(`AI 服务返回 ${response.status}`);
    const payload = await response.json() as { choices?: Array<{ message?: { content?: string } }> };
    const content = payload.choices?.[0]?.message?.content ?? "{}";
    return NextResponse.json({ analysis: safeAnalysis(JSON.parse(content)) });
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : "AI 解读暂时不可用" }, { status: 502 });
  }
}
