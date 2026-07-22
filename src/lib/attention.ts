import type {
  DataStatus,
  MarketCategory,
  MarketGroup,
  MarketMetricInput,
  MarketNode,
} from "@/lib/types";

const CATEGORY_ORDER: MarketCategory[] = ["加密资产", "股票市场", "大宗商品"];

const CATEGORY_DESCRIPTION: Record<MarketCategory, string> = {
  加密资产: "观察主流加密资产与链上主题的市场讨论、波动和交易活跃度。",
  股票市场: "观察全球股票行业主题的相对关注变化，市场指标使用模拟数据。",
  大宗商品: "观察黄金、原油与铜等宏观资产的相对关注变化，市场指标使用模拟数据。",
};

function clamp(value: number, min = 0, max = 100) {
  return Math.min(max, Math.max(min, value));
}

function round(value: number, digits = 1) {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

function normalize(values: number[]): number[] {
  if (!values.length) return [];
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (max === min) return values.map(() => 50);
  return values.map((value) => clamp(((value - min) / (max - min)) * 100));
}

export function calculateAttention(items: MarketMetricInput[]): MarketNode[] {
  const volumeScores = normalize(items.map((item) => Math.abs(item.volumeChange24h)));
  const priceScores = normalize(items.map((item) => Math.abs(item.priceChange24h)));
  const newsScores = normalize(items.map((item) => clamp(item.newsHeat)));

  const weighted = items.map(
    (_, index) =>
      volumeScores[index] * 0.5 + priceScores[index] * 0.3 + newsScores[index] * 0.2,
  );
  const finalScores = normalize(weighted);

  return items.map((item, index) => ({
    ...item,
    kind: "市场",
    attentionScore: Math.round(finalScores[index]),
    components: {
      volume: Math.round(volumeScores[index]),
      price: Math.round(priceScores[index]),
      news: Math.round(newsScores[index]),
    },
  }));
}

function groupStatus(children: MarketNode[]): DataStatus {
  if (children.every((item) => item.dataStatus === "模拟数据")) return "模拟数据";
  if (children.every((item) => item.dataStatus === "公开数据")) return "公开数据";
  return "混合数据";
}

export function buildMarketGroups(items: MarketNode[]): MarketGroup[] {
  return CATEGORY_ORDER.map((category) => {
    const children = items.filter((item) => item.category === category);
    const average = (selector: (item: MarketNode) => number) =>
      children.length
        ? children.reduce((sum, item) => sum + selector(item), 0) / children.length
        : 0;
    const status = groupStatus(children);

    return {
      id: `category-${category}`,
      kind: "分类",
      name: category,
      category,
      attentionScore: Math.round(average((item) => item.attentionScore)),
      priceChange24h: round(average((item) => item.priceChange24h)),
      volumeChange24h: round(average((item) => item.volumeChange24h)),
      newsHeat: Math.round(average((item) => item.newsHeat)),
      source: status === "模拟数据" ? "模拟数据" : "公开数据与模拟数据",
      dataStatus: status,
      description: CATEGORY_DESCRIPTION[category],
      children,
    };
  });
}

export function formatChange(value: number) {
  if (value === 0) return "0.0%";
  return `${value > 0 ? "+" : ""}${value.toFixed(1)}%`;
}

export function formatCompactCurrency(value?: number) {
  if (value === undefined) return "—";
  return new Intl.NumberFormat("zh-CN", {
    style: "currency",
    currency: "USD",
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(value);
}
