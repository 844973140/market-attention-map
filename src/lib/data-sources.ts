import { demoMarketInputs } from "@/data/demo";
import { calculateAttention } from "@/lib/attention";
import type { MarketDataResponse, MarketMetricInput } from "@/lib/types";

const REQUEST_TIMEOUT_MS = 6_500;

const GDELT_KEYWORDS: Record<string, string> = {
  bitcoin: "bitcoin",
  ethereum: "ethereum",
  "ai-crypto": '"artificial intelligence" finance',
  technology: '"technology stocks"',
  gold: "gold market",
  "crude-oil": '"crude oil" market',
};

interface CoinMarketResponse {
  id: string;
  market_cap: number;
  total_volume: number;
  price_change_percentage_24h: number | null;
}

interface CoinSnapshot {
  priceChange24h: number;
  volumeChange24h: number;
  marketCap: number;
  totalVolume: number;
}

function signal() {
  return AbortSignal.timeout(REQUEST_TIMEOUT_MS);
}

function coingeckoHeaders(): HeadersInit {
  const apiKey = process.env.COINGECKO_API_KEY?.trim();
  return apiKey ? { "x-cg-demo-api-key": apiKey } : {};
}

async function fetchJson<T>(url: string, headers: HeadersInit = {}): Promise<T> {
  const init: RequestInit & { next: { revalidate: number } } = {
    headers: { Accept: "application/json", ...headers },
    next: { revalidate: 300 },
    signal: signal(),
  };
  const response = await fetch(url, init);
  if (!response.ok) throw new Error(`请求失败：${response.status}`);
  return (await response.json()) as T;
}

function nearestVolumeChange(points: [number, number][]): number {
  if (points.length < 2) return 0;
  const latest = points.at(-1)!;
  const targetTime = latest[0] - 24 * 60 * 60 * 1_000;
  const prior = points.reduce((closest, point) =>
    Math.abs(point[0] - targetTime) < Math.abs(closest[0] - targetTime) ? point : closest,
  );
  if (!prior[1]) return 0;
  return ((latest[1] - prior[1]) / prior[1]) * 100;
}

async function fetchCoinGecko(): Promise<Record<string, CoinSnapshot>> {
  const base = "https://api.coingecko.com/api/v3";
  const headers = coingeckoHeaders();
  const markets = await fetchJson<CoinMarketResponse[]>(
    `${base}/coins/markets?vs_currency=usd&ids=bitcoin%2Cethereum&price_change_percentage=24h`,
    headers,
  );
  const charts = await Promise.all(
    ["bitcoin", "ethereum"].map(async (id) => {
      const chart = await fetchJson<{ total_volumes: [number, number][] }>(
        `${base}/coins/${id}/market_chart?vs_currency=usd&days=2&interval=hourly`,
        headers,
      );
      return [id, nearestVolumeChange(chart.total_volumes)] as const;
    }),
  );
  const volumeChanges = Object.fromEntries(charts);

  return Object.fromEntries(
    markets.map((market) => [
      market.id,
      {
        priceChange24h: market.price_change_percentage_24h ?? 0,
        volumeChange24h: volumeChanges[market.id] ?? 0,
        marketCap: market.market_cap,
        totalVolume: market.total_volume,
      },
    ]),
  );
}

function timelineValues(payload: unknown): number[] {
  if (!payload || typeof payload !== "object") return [];
  const timeline = (payload as { timeline?: unknown }).timeline;
  const series = Array.isArray(timeline) ? timeline[0] : undefined;
  if (!series || typeof series !== "object") return [];
  const data = (series as { data?: unknown }).data;
  if (!Array.isArray(data)) return [];
  return data
    .map((point) => {
      if (!point || typeof point !== "object") return Number.NaN;
      return Number((point as { value?: unknown }).value);
    })
    .filter(Number.isFinite);
}

async function fetchGdeltHeat(keyword: string): Promise<number> {
  const params = new URLSearchParams({
    query: keyword,
    mode: "timelinevol",
    format: "json",
    timespan: "24h",
    maxrecords: "250",
  });
  const payload = await fetchJson<unknown>(
    `https://api.gdeltproject.org/api/v2/doc/doc?${params.toString()}`,
  );
  const values = timelineValues(payload);
  if (!values.length) throw new Error("没有可用的新闻时间序列");
  const average = values.reduce((sum, value) => sum + value, 0) / values.length;
  const latest = values.at(-1) ?? average;
  if (!average) return 35;
  return Math.max(10, Math.min(100, (latest / average) * 50));
}

export async function getMarketData(): Promise<MarketDataResponse> {
  const warnings: string[] = [];
  let publicSourceCount = 0;
  let coins: Record<string, CoinSnapshot> = {};

  try {
    coins = await fetchCoinGecko();
    publicSourceCount += Object.keys(coins).length;
  } catch {
    warnings.push("CoinGecko 暂时不可用，已使用模拟数据。");
  }

  const newsEntries = await Promise.all(
    Object.entries(GDELT_KEYWORDS).map(async ([id, keyword]) => {
      try {
        return [id, await fetchGdeltHeat(keyword)] as const;
      } catch {
        return [id, null] as const;
      }
    }),
  );
  const newsHeat = Object.fromEntries(newsEntries) as Record<string, number | null>;
  const liveNewsCount = newsEntries.filter(([, heat]) => heat !== null).length;
  publicSourceCount += liveNewsCount;
  if (liveNewsCount < newsEntries.length) {
    warnings.push("部分 GDELT 新闻数据不可用，已保留演示热度。");
  }

  const merged: MarketMetricInput[] = demoMarketInputs.map((demo) => {
    const coin = coins[demo.id];
    const liveHeat = newsHeat[demo.id];
    const hasLiveHeat = typeof liveHeat === "number";
    const isStrictDemoCategory = demo.category !== "加密资产";
    const hasAnyPublicData = Boolean(coin) || hasLiveHeat;

    let source = "模拟数据";
    let dataStatus: MarketMetricInput["dataStatus"] = "模拟数据";
    if (isStrictDemoCategory && hasLiveHeat) {
      source = "模拟数据（GDELT 新闻参考）";
    } else if (coin && hasLiveHeat) {
      source = "CoinGecko · GDELT";
      dataStatus = "公开数据";
    } else if (coin) {
      source = "CoinGecko";
      dataStatus = "公开数据";
    } else if (hasAnyPublicData) {
      source = "GDELT · 市场指标为模拟数据";
      dataStatus = "混合数据";
    }

    return {
      ...demo,
      priceChange24h: coin?.priceChange24h ?? demo.priceChange24h,
      volumeChange24h: coin?.volumeChange24h ?? demo.volumeChange24h,
      newsHeat: hasLiveHeat ? liveHeat : demo.newsHeat,
      marketCap: coin?.marketCap ?? demo.marketCap,
      totalVolume: coin?.totalVolume ?? demo.totalVolume,
      source,
      dataStatus,
    };
  });

  return {
    updatedAt: new Date().toISOString(),
    mode: publicSourceCount ? "公开与模拟混合" : "演示模式",
    items: calculateAttention(merged),
    warnings,
  };
}
