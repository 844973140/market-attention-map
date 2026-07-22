export type MarketCategory = "加密资产" | "股票市场" | "大宗商品";

export type DataStatus = "公开数据" | "混合数据" | "模拟数据";

export interface MarketMetricInput {
  id: string;
  name: string;
  category: MarketCategory;
  priceChange24h: number;
  volumeChange24h: number;
  newsHeat: number;
  source: string;
  dataStatus: DataStatus;
  description: string;
  marketCap?: number;
  totalVolume?: number;
}

export interface MarketNode extends MarketMetricInput {
  kind: "市场";
  attentionScore: number;
  components: {
    volume: number;
    price: number;
    news: number;
  };
}

export interface MarketGroup {
  id: string;
  kind: "分类";
  name: MarketCategory;
  category: MarketCategory;
  attentionScore: number;
  priceChange24h: number;
  volumeChange24h: number;
  newsHeat: number;
  source: string;
  dataStatus: DataStatus;
  description: string;
  children: MarketNode[];
}

export type MarketSelection = MarketNode | MarketGroup;

export interface MarketDataResponse {
  updatedAt: string;
  mode: "公开与模拟混合" | "演示模式";
  items: MarketNode[];
  warnings: string[];
}
