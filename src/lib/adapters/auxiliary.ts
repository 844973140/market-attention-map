import type { MarketMetricInput } from "@/lib/types";

export type AuxiliaryCapability = "数据采集" | "情绪分析" | "搜索辅助";

export interface AuxiliaryObservation {
  itemId: string;
  newsHeat?: number;
  sentimentScore?: number;
  references?: Array<{
    title: string;
    url: string;
  }>;
}

/**
 * 第三方 Skill 只能实现此辅助层契约，不能参与市场关注度指数的公式实现。
 * 核心归一化与加权始终由 src/lib/attention.ts 负责。
 */
export interface AuxiliaryDataAdapter {
  readonly id: string;
  readonly capability: AuxiliaryCapability;
  isAvailable(): boolean;
  collect(items: readonly MarketMetricInput[]): Promise<readonly AuxiliaryObservation[]>;
}

/**
 * 隔离单个第三方数据源的失败，便于随时替换不稳定或付费的实现。
 */
export async function collectAuxiliaryObservations(
  adapters: readonly AuxiliaryDataAdapter[],
  items: readonly MarketMetricInput[],
): Promise<AuxiliaryObservation[]> {
  const activeAdapters = adapters.filter((adapter) => adapter.isAvailable());
  const results = await Promise.allSettled(
    activeAdapters.map((adapter) => adapter.collect(items)),
  );

  return results.flatMap((result) => (result.status === "fulfilled" ? [...result.value] : []));
}
