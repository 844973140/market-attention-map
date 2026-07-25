import type { MarketNode } from "@/lib/types";

/** AI 分析层的可替换边界，不得修改本仓库的指数计算公式。 */
export interface MarketAnalysisAdapter {
  readonly id: string;
  analyze(input: Pick<MarketNode, "id" | "name" | "category" | "newsHeat">): Promise<{
    summary: string;
    evidence: string[];
    generatedAt: string;
  }>;
}
