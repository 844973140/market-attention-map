"use client";

type Contribution = { label: string; value: number; weight: string };

interface AttentionCompositionProps { items: Contribution[]; }

/** 基于 Lieflat Charts Lupi Basics F5「Tick Rows」：一根刻度代表 5 个归一化点。 */
export function AttentionComposition({ items }: AttentionCompositionProps) {
  const maxTicks = 20;
  const startX = 92;
  const tickWidth = 8.1;

  return (
    <div className="attention-composition" aria-label="市场关注度指数构成">
      <div className="composition-head"><span>指数构成</span><small>一根刻度 = 5 个归一化点</small></div>
      <svg viewBox="0 0 320 170" role="img" aria-label="成交量、价格与新闻热度的指数构成">
        {items.map((item, row) => {
          const tickCount = Math.round(Math.min(100, Math.max(0, item.value)) / 5);
          const y = 34 + row * 43;
          return <g className="tick-row" key={item.label} style={{ animationDelay: `${row * 90}ms` }}>
            <text className="tick-label" x="82" y={y + 3} textAnchor="end">{item.label}</text>
            <line className="tick-baseline" x1={startX} y1={y + 8} x2={startX + maxTicks * tickWidth} y2={y + 8} />
            {Array.from({ length: tickCount }, (_, index) => {
              const x = startX + index * tickWidth + tickWidth / 2;
              const height = 9 + ((index + row * 3) % 4) * 1.5;
              return <g key={index}><line className="tick-mark" x1={x} y1={y + 8} x2={x} y2={y + 8 - height} />{index % 5 === 4 ? <circle className="tick-dot" cx={x} cy={y + 12} r="1.1" /> : null}</g>;
            })}
            <text className="tick-value" x="274" y={y + 4}>{Math.round(item.value)}</text>
            <text className="tick-weight" x="303" y={y + 4} textAnchor="end">{item.weight}</text>
          </g>;
        })}
        <text className="tick-caption" x="182" y="158" textAnchor="middle">每第五根以圆点标记 · 权重用于最终加权</text>
      </svg>
    </div>
  );
}
