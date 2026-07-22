"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import { AttentionTreemap } from "@/components/AttentionTreemap";
import { DetailPanel } from "@/components/DetailPanel";
import { demoMarketInputs } from "@/data/demo";
import { buildMarketGroups, calculateAttention } from "@/lib/attention";
import type { MarketDataResponse, MarketSelection } from "@/lib/types";

const initialItems = calculateAttention(demoMarketInputs);

function timeLabel(value: string) {
  return new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(new Date(value));
}

export function MarketAttentionMap() {
  const [items, setItems] = useState(initialItems);
  const [updatedAt, setUpdatedAt] = useState(new Date().toISOString());
  const [mode, setMode] = useState<MarketDataResponse["mode"]>("演示模式");
  const [warnings, setWarnings] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const groups = useMemo(() => buildMarketGroups(items), [items]);
  const [selectedId, setSelectedId] = useState(initialItems[0].id);

  const selections = useMemo<MarketSelection[]>(
    () => groups.flatMap((group) => [group, ...group.children]),
    [groups],
  );
  const selection = selections.find((item) => item.id === selectedId) ?? items[0];

  useEffect(() => {
    const controller = new AbortController();
    fetch("/api/market-data", { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error("市场数据暂时不可用");
        return response.json() as Promise<MarketDataResponse>;
      })
      .then((data) => {
        setItems(data.items);
        setUpdatedAt(data.updatedAt);
        setMode(data.mode);
        setWarnings(data.warnings);
      })
      .catch(() => {
        if (!controller.signal.aborted) {
          setWarnings(["公开数据暂时不可用，当前展示完整模拟数据。"]);
        }
      })
      .finally(() => setIsLoading(false));
    return () => controller.abort();
  }, []);

  const handleSelect = useCallback((next: MarketSelection) => setSelectedId(next.id), []);
  const focus = [...items].sort((a, b) => b.attentionScore - a.attentionScore)[0];

  return (
    <main className="site-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="饼干市场热力图首页">
          <span className="brand-mark" aria-hidden="true">饼</span>
          <span>饼干市场热力图</span>
        </a>
        <nav aria-label="主导航">
          <a className="active" href="#overview">市场概览</a>
          <a href="#analysis">数据分析</a>
          <a href="#method">方法说明</a>
        </nav>
        <div className="live-status">
          <i className={isLoading ? "loading" : ""} />
          {isLoading ? "正在连接公开数据" : mode}
        </div>
      </header>

      <section className="hero" id="top">
        <div>
          <p className="eyebrow">全球市场 · 公开数据观察</p>
          <h1>饼干市场热力图</h1>
          <p className="subtitle">观察全球资本关注正在集中于哪些市场。</p>
        </div>
        <div className="hero-note">
          <span>市场关注度指数</span>
          <strong>0—100</strong>
          <small>衡量关注，不判断方向</small>
        </div>
      </section>

      <section className="signal-strip" aria-label="市场摘要">
        <div>
          <span>当前关注焦点</span>
          <strong>{focus.name}</strong>
          <em>指数 {focus.attentionScore}</em>
        </div>
        <div>
          <span>覆盖范围</span>
          <strong>3 类市场 · {items.length} 个主题</strong>
        </div>
        <div>
          <span>数据更新时间</span>
          <strong>{timeLabel(updatedAt)}</strong>
          <em>约 5 分钟刷新</em>
        </div>
      </section>

      <section className="market-workspace" id="overview">
        <div className="map-panel">
          <div className="section-heading">
            <div>
              <p>全球市场</p>
              <h2>注意力分布</h2>
            </div>
            <div className="map-legend" aria-label="关注度图例">
              <span>低</span>
              <i /><i /><i /><i />
              <span>高</span>
            </div>
          </div>
          <p className="interaction-hint">点击分类进入局部，点击市场查看右侧详情</p>
          <AttentionTreemap groups={groups} selectedId={selectedId} onSelect={handleSelect} />
        </div>
        <DetailPanel selection={selection} />
      </section>

      {warnings.length ? (
        <div className="data-warning" role="status">
          <strong>数据提示</strong>
          <span>{warnings.join(" ")}</span>
        </div>
      ) : null}

      <section className="method-section" id="method">
        <div>
          <p className="eyebrow">方法说明</p>
          <h2>让“被关注”与“值得买”保持距离</h2>
        </div>
        <div className="method-copy">
          <p>
            指数仅衡量成交、波动与新闻讨论的相对活跃程度。价格变化使用绝对幅度参与计算，涨跌方向单独展示。
          </p>
          <p>
            股票市场与大宗商品的市场指标为模拟数据，并在地图和详情中明确标注。所有公开接口均可自动降级。
          </p>
        </div>
      </section>

      <footer>
        <span>饼干市场热力图 · 原型演示</span>
        <span>非资金流、非价格预测、非投资建议</span>
      </footer>
    </main>
  );
}
