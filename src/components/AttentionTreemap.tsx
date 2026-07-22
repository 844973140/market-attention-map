"use client";

import * as echarts from "echarts";
import type { ECElementEvent } from "echarts";
import { useEffect, useMemo, useRef } from "react";

import { formatChange } from "@/lib/attention";
import type { MarketGroup, MarketSelection } from "@/lib/types";

interface AttentionTreemapProps {
  groups: MarketGroup[];
  selectedId: string;
  onSelect: (selection: MarketSelection) => void;
}

interface TreemapDatum {
  id: string;
  name: string;
  value: number;
  kind: "分类" | "市场";
  attentionScore: number;
  change: number;
  source: string;
  status: string;
  selection: MarketSelection;
  itemStyle?: Record<string, unknown>;
  children?: TreemapDatum[];
}

interface TreemapParams {
  data?: TreemapDatum;
}

function attentionColor(score: number) {
  if (score >= 82) return "#b68a38";
  if (score >= 66) return "#7f6d37";
  if (score >= 48) return "#505035";
  if (score >= 28) return "#343b31";
  return "#252a27";
}

function shortSource(source: string) {
  if (source.includes("模拟数据")) return "模拟数据";
  return source.replace(" · ", " + ");
}

function toTreeData(groups: MarketGroup[], selectedId: string): TreemapDatum[] {
  return groups.map((group) => ({
    id: group.id,
    name: group.name,
    value: Math.max(
      1,
      group.children.reduce((sum, item) => sum + Math.max(item.attentionScore, 8), 0),
    ),
    kind: "分类",
    attentionScore: group.attentionScore,
    change: group.priceChange24h,
    source: group.source,
    status: group.dataStatus,
    selection: group,
    itemStyle: {
      borderColor: group.id === selectedId ? "#e0b861" : "#111512",
      borderWidth: group.id === selectedId ? 2 : 1,
    },
    children: group.children.map((item) => ({
      id: item.id,
      name: item.name,
      value: Math.max(item.attentionScore, 8),
      kind: "市场",
      attentionScore: item.attentionScore,
      change: item.priceChange24h,
      source: item.source,
      status: item.dataStatus,
      selection: item,
      itemStyle: {
        color: attentionColor(item.attentionScore),
        borderColor:
          item.id === selectedId
            ? "#f0cb72"
            : item.priceChange24h >= 0
              ? "rgba(102, 150, 104, .42)"
              : "rgba(171, 82, 73, .42)",
        borderWidth: item.id === selectedId ? 3 : 1,
      },
    })),
  }));
}

export function AttentionTreemap({ groups, selectedId, onSelect }: AttentionTreemapProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<echarts.EChartsType | null>(null);
  const onSelectRef = useRef(onSelect);
  const treeData = useMemo(() => toTreeData(groups, selectedId), [groups, selectedId]);

  useEffect(() => {
    onSelectRef.current = onSelect;
  }, [onSelect]);

  useEffect(() => {
    if (!containerRef.current) return;
    const chart = echarts.init(containerRef.current, undefined, { renderer: "canvas" });
    chartRef.current = chart;

    chart.setOption({
      backgroundColor: "transparent",
      animationDuration: 700,
      animationEasing: "cubicOut",
      tooltip: {
        confine: true,
        backgroundColor: "rgba(13, 16, 14, .96)",
        borderColor: "#5d5133",
        borderWidth: 1,
        padding: [12, 14],
        textStyle: { color: "#e7e3d8", fontSize: 12 },
        formatter: (params: TreemapParams) => {
          const data = params.data;
          if (!data) return "";
          return [
            `<strong>${data.name}</strong>`,
            `市场关注度指数：${data.attentionScore}`,
            `24小时变化：${formatChange(data.change)}`,
            `数据来源：${data.source}`,
          ].join("<br/>");
        },
      },
      series: [
        {
          type: "treemap",
          name: "全球市场",
          data: [],
          top: 8,
          right: 8,
          bottom: 8,
          left: 8,
          roam: false,
          nodeClick: "zoomToNode",
          zoomToNodeRatio: 0.7,
          leafDepth: 2,
          squareRatio: 1.18,
          breadcrumb: { show: false },
          visualMin: 0,
          silent: false,
          label: {
            show: true,
            position: "insideTopLeft",
            padding: [12, 12],
            overflow: "truncate",
            formatter: (params: TreemapParams) => {
              const data = params.data;
              if (!data || data.kind === "分类") return "";
              const direction = data.change >= 0 ? "rise" : "fall";
              return [
                `{name|${data.name}}`,
                `{score|关注度 ${data.attentionScore}}`,
                `{${direction}|24小时 ${formatChange(data.change)}}`,
                `{source|来源 ${shortSource(data.source)}}`,
              ].join("\n");
            },
            rich: {
              name: {
                color: "#f2eee4",
                fontSize: 17,
                fontWeight: 700,
                lineHeight: 28,
              },
              score: { color: "#d8bd79", fontSize: 13, lineHeight: 22 },
              rise: { color: "#9cc69a", fontSize: 12, lineHeight: 20 },
              fall: { color: "#d8897e", fontSize: 12, lineHeight: 20 },
              source: { color: "rgba(235, 231, 218, .62)", fontSize: 10, lineHeight: 18 },
            },
          },
          upperLabel: {
            show: true,
            height: 36,
            color: "#ddd7c8",
            fontSize: 13,
            fontWeight: 600,
            padding: [0, 10],
            formatter: (params: TreemapParams) => {
              const data = params.data;
              return data?.kind === "分类"
                ? `${data.name}  ·  关注度 ${data.attentionScore}`
                : data?.name ?? "";
            },
          },
          itemStyle: {
            gapWidth: 2,
            borderColor: "#101310",
            borderWidth: 1,
          },
          levels: [
            {
              itemStyle: { borderWidth: 0, gapWidth: 8, borderColor: "transparent" },
            },
            {
              color: ["#1b211d"],
              itemStyle: { borderColor: "#0b0e0c", borderWidth: 2, gapWidth: 3 },
              upperLabel: { show: true },
            },
            {
              itemStyle: { borderWidth: 1, gapWidth: 2 },
            },
          ],
          emphasis: {
            itemStyle: {
              borderColor: "#e0b861",
              borderWidth: 2,
              shadowBlur: 18,
              shadowColor: "rgba(224, 184, 97, .18)",
            },
          },
        },
      ],
    });

    const handleClick = (params: ECElementEvent) => {
      const data = params.data as unknown as TreemapDatum | undefined;
      if (data?.selection) onSelectRef.current(data.selection);
    };
    chart.on("click", handleClick);

    const observer = new ResizeObserver(() => chart.resize());
    observer.observe(containerRef.current);
    return () => {
      observer.disconnect();
      chart.off("click", handleClick);
      chart.dispose();
      chartRef.current = null;
    };
  }, []);

  useEffect(() => {
    chartRef.current?.setOption({ series: [{ data: treeData }] });
  }, [treeData]);

  return <div ref={containerRef} className="treemap-canvas" aria-label="全球市场关注度树图" />;
}
