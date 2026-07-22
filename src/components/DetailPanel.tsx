"use client";

import { formatChange, formatCompactCurrency } from "@/lib/attention";
import type { MarketSelection } from "@/lib/types";

interface DetailPanelProps {
  selection: MarketSelection;
}

export function DetailPanel({ selection }: DetailPanelProps) {
  const isMarket = selection.kind === "市场";
  const changeClass = selection.priceChange24h >= 0 ? "positive" : "negative";

  return (
    <aside className="detail-panel" id="analysis" aria-live="polite">
      <div className="detail-kicker">
        <span>详细信息</span>
        <span className={`data-badge ${selection.dataStatus === "模拟数据" ? "demo" : ""}`}>
          {selection.dataStatus}
        </span>
      </div>

      <div className="detail-heading">
        <div>
          <p className="detail-category">{selection.category}</p>
          <h2>{selection.name}</h2>
        </div>
        <div className="score-lockup" aria-label={`市场关注度指数 ${selection.attentionScore}`}>
          <strong>{selection.attentionScore}</strong>
          <span>/ 100</span>
        </div>
      </div>

      <div className="score-track" aria-hidden="true">
        <span style={{ width: `${selection.attentionScore}%` }} />
      </div>

      <div className="detail-metrics">
        <div>
          <span>24小时变化</span>
          <strong className={changeClass}>{formatChange(selection.priceChange24h)}</strong>
        </div>
        <div>
          <span>新闻热度</span>
          <strong>{Math.round(selection.newsHeat)}</strong>
        </div>
        <div>
          <span>成交活跃变化</span>
          <strong>{formatChange(selection.volumeChange24h)}</strong>
        </div>
      </div>

      {isMarket && (selection.marketCap || selection.totalVolume) ? (
        <div className="market-facts">
          <div>
            <span>市值</span>
            <strong>{formatCompactCurrency(selection.marketCap)}</strong>
          </div>
          <div>
            <span>24小时成交量</span>
            <strong>{formatCompactCurrency(selection.totalVolume)}</strong>
          </div>
        </div>
      ) : null}

      <section className="detail-section">
        <h3>数据来源</h3>
        <p>{selection.source}</p>
        {selection.dataStatus === "模拟数据" ? (
          <p className="demo-notice">本节点市场指标为模拟数据，仅用于界面与交互展示。</p>
        ) : null}
      </section>

      <section className="detail-section">
        <h3>指标说明</h3>
        <p>
          市场关注度指数由成交量变化 50%、价格变化 30%、新闻热度 20% 加权，并在全市场样本中归一化至
          0–100。
        </p>
        {isMarket ? (
          <div className="component-bars">
            {[
              ["成交量", selection.components.volume, "50%"],
              ["价格", selection.components.price, "30%"],
              ["新闻", selection.components.news, "20%"],
            ].map(([label, value, weight]) => (
              <div className="component-row" key={label}>
                <span>{label}</span>
                <div><i style={{ width: `${value}%` }} /></div>
                <b>{weight}</b>
              </div>
            ))}
          </div>
        ) : null}
      </section>

      <section className="detail-section interpretation">
        <h3>智能解读</h3>
        <p>{selection.description}</p>
        <small>此处仅解释关注度构成，不代表趋势判断或买卖建议。</small>
      </section>
    </aside>
  );
}
