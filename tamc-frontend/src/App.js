/*  App.js - FARMER-FRIENDLY VERSION WITH SIMPLIFIED SENTIMENT ANALYSIS  */
import React, { useState, useEffect, useRef, useCallback, useMemo } from "react";
import axios from "axios";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer
} from "recharts";
import { v4 as uuidv4 } from "uuid";
import "./App.css";

/* ---------- Helpers ---------- */
const formatDate = (d) => (d ? d.split("-").reverse().join("/") : "");
const isInt = (f) => /bag|arrivals|count|lots|farmers/i.test(f);
const formatValue = (f, v) => {
  if (typeof v !== "number") return v;
  if (isInt(f) || /chilli|paddy|cotton|onion|bags|arrivals/i.test(f)) return Math.round(v);
  return Math.round(v);
};

const isWeightMetric = (metricName) => {
  if (!metricName) return false;
  const lower = metricName.toLowerCase();
  return /quantity|weight|quintal/.test(lower);
};

const toNumber = (value) => {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
};

const convertQuintalsToKg = (quintals) => Math.round(quintals * 100);
const colors = ["#16a34a", "#ef4444", "#f59e0b", "#3b82f6", "#8b5cf6", "#ec4899", "#06b6d4"];

const mentionsCommodity = (text = "") => {
  if (!text) return false;
  return /\b(chilli|cotton|paddy|onion|groundnut|turmeric|maize|rice|wheat|soybean|sugarcane)\b/i.test(text);
};

const summarizeForecast = (response, weatherData, isPriceData) => {
  if (!response || !Array.isArray(response.total_predicted) || response.total_predicted.length === 0) {
    return [];
  }

  const data = response.total_predicted;
  const firstEntry = data[0];
  const lastEntry = data[data.length - 1];
  const suffix = isPriceData ? "/Q" : getUnitSuffix(response.metric_name);
  const prefix = isPriceData ? "₹" : "";
  const totalSum = data.reduce((sum, item) => sum + (item.total_predicted_value || 0), 0);
  const averageValue = totalSum / data.length;
  const changePct = firstEntry && firstEntry.total_predicted_value
    ? ((lastEntry.total_predicted_value - firstEntry.total_predicted_value) / firstEntry.total_predicted_value) * 100
    : 0;

  const trendBadge = changePct > 5 ? "up" : changePct < -5 ? "down" : "flat";
  const trendIcon = trendBadge === "up" ? "📈" : trendBadge === "down" ? "📉" : "➖";
  const formatMetricValue = (value) => `${prefix}${Math.round(value).toLocaleString()}${suffix}`;
  const summary = [
    {
      icon: "📆",
      title: `${formatDate(firstEntry.date)} → ${formatDate(lastEntry.date)}`,
      detail: `Covering ${data.length} day${data.length > 1 ? "s" : ""}`,
      badge: null
    },
    {
      icon: isPriceData ? "💰" : "📦",
      title: isPriceData ? `Avg Price ${formatMetricValue(averageValue)}` : `${response.metric_name || "Forecast"}`,
      detail: `${formatMetricValue(lastEntry.total_predicted_value)} on ${formatDate(lastEntry.date)}`,
      badge: {
        label: `${trendIcon} ${changePct >= 0 ? "+" : ""}${changePct.toFixed(1)}%`,
        tone: trendBadge
      }
    }
  ];

  if (!isPriceData && response.commodity_daily) {
    const totals = Object.entries(response.commodity_daily).map(([commodity, entries]) => ({
      commodity,
      total: entries.slice(0, data.length).reduce((sum, entry) => sum + (entry.predicted_value || 0), 0)
    }));
    totals.sort((a, b) => b.total - a.total);
    const top = totals.slice(0, 2);
    if (top.length > 0) {
      summary.push({
        icon: "🌾",
        title: "Top Commodities",
        detail: top.map((item) => `${item.commodity}: ${Math.round(item.total).toLocaleString()}`).join(" • "),
        badge: null
      });
    }
  }

  if (weatherData?.weather_summary) {
    const weatherSummary = weatherData.weather_summary;
    const factor = weatherData.weather_factor_summary || {};
    const heavy = factor.heavy || 0;
    const moderate = factor.moderate || 0;
    const badgeLabel = heavy > 0 ? `${heavy} heavy impact day${heavy > 1 ? "s" : ""}` :
      moderate > 0 ? `${moderate} moderate impact day${moderate > 1 ? "s" : ""}` : "Stable weather";

    summary.push({
      icon: "🌤️",
      title: `${weatherSummary.condition} • ${weatherSummary.temp_c ?? "--"}°C`,
      detail: `Rain ${weatherSummary.rain_mm ?? 0}mm`,
      badge: {
        label: badgeLabel,
        tone: heavy > 0 ? "down" : moderate > 0 ? "flat" : "up"
      }
    });
  }

  return summary;
};

// Helper to get unit suffix based on metric type
const getUnitSuffix = (metricName) => {
  if (!metricName) return "";
  const lower = metricName.toLowerCase();
  if (lower.includes("price")) return " /Q"; // Already handled separately
  if (lower.includes("weight") || lower.includes("quintal") || lower.includes("quantity")) return " Q"; // Quintals
  if (lower.includes("revenue") || lower.includes("income")) return " ₹";
  if (lower.includes("bags")) return " bags";
  if (lower.includes("lots")) return " lots";
  if (lower.includes("farmers")) return " farmers";
  if (lower.includes("arrivals")) return " arrivals";
  return ""; // No unit
};

// Helper to detect if query is asking for single day
const isSingleDayQuery = (query) => {
  const lowerQuery = query.toLowerCase();
  return /\b(tomorrow|today)\b/.test(lowerQuery) ||
    /\b(1|one)\s*(day)\b/.test(lowerQuery) ||
    /price.*tomorrow/i.test(lowerQuery) ||
    /tomorrow.*price/i.test(lowerQuery);
};

/* ---------- 📰 FARMER-FRIENDLY SENTIMENT PANEL ---------- */
const SentimentPanel = React.memo(function SentimentPanel({ sentimentData }) {
  const [isExpanded, setIsExpanded] = useState(true); // Default expanded

  if (!sentimentData || sentimentData.news_count === 0) return null;

  const { avg_sentiment, sentiment_label, news_count, top_keywords, sample_headlines } = sentimentData;

  // Simplified farmer-friendly messages
  const getFarmerMessage = () => {
    if (sentiment_label === "Positive") {
      return {
        icon: "📈",
        title: "Good News for Farmers!",
        message: "Market conditions are favorable. Prices may increase.",
        color: "#22c55e",
        bgGradient: "linear-gradient(135deg, #d1fae5, #a7f3d0)",
        action: "Good time to plan your sales"
      };
    } else if (sentiment_label === "Negative") {
      return {
        icon: "📉",
        title: "Market Alert",
        message: "Prices may decrease. Consider selling soon.",
        color: "#ef4444",
        bgGradient: "linear-gradient(135deg, #fee2e2, #fecaca)",
        action: "Act quickly to avoid losses"
      };
    } else {
      return {
        icon: "➡️",
        title: "Stable Market",
        message: "No major changes expected in prices.",
        color: "#f59e0b",
        bgGradient: "linear-gradient(135deg, #fef3c7, #fde68a)",
        action: "Normal market conditions"
      };
    }
  };

  const farmerMsg = getFarmerMessage();

  return (
    <div className="mt-4 rounded-xl p-5 border-2 shadow-lg transform transition-all duration-300 hover:shadow-xl" style={{
      background: farmerMsg.bgGradient,
      borderColor: farmerMsg.color,
    }}>
      {/* Main Message - Big and Clear with toggle */}
      <div className="flex items-center gap-4 cursor-pointer" style={{
        marginBottom: isExpanded ? "1rem" : "0"
      }} onClick={() => setIsExpanded(!isExpanded)}>
        <span className="text-5xl">{farmerMsg.icon}</span>
        <div className="flex-1">
          <h3 className="m-0 text-xl font-bold mb-1" style={{ color: farmerMsg.color }}>
            {farmerMsg.title}
          </h3>
          <p style={{
            margin: 0,
            fontSize: "1.05rem",
            color: "#374151",
            fontWeight: "600"
          }}>
            {farmerMsg.message}
          </p>
        </div>
        <div style={{
          width: "28px",
          height: "28px",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "rgba(255, 255, 255, 0.6)",
          borderRadius: "8px",
          fontSize: "0.9rem",
          transform: isExpanded ? "rotate(180deg)" : "rotate(0deg)",
          transition: "transform 0.2s ease"
        }}>
          ▼
        </div>
      </div>

      {isExpanded && (
        <div>

          {/* Action Box */}
          <div style={{
            background: "white",
            borderRadius: "8px",
            padding: "1rem",
            marginBottom: "1rem",
            boxShadow: "0 2px 6px rgba(0,0,0,0.1)",
            borderLeft: `4px solid ${farmerMsg.color}`
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
              <span style={{ fontSize: "1.5rem" }}>💡</span>
              <span style={{
                fontSize: "1rem",
                fontWeight: "600",
                color: "#374151"
              }}>
                {farmerMsg.action}
              </span>
            </div>
          </div>

          {/* Simple Visual Indicator */}
          <div style={{
            background: "white",
            borderRadius: "8px",
            padding: "1rem",
            marginBottom: "1rem",
            boxShadow: "0 2px 4px rgba(0,0,0,0.05)"
          }}>
            <div style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: "0.75rem"
            }}>
              <span style={{ fontSize: "0.95rem", color: "#6b7280", fontWeight: "600" }}>
                📰 Based on {news_count} recent market reports
              </span>
              <div style={{
                background: farmerMsg.color,
                color: "white",
                padding: "0.35rem 0.9rem",
                borderRadius: "20px",
                fontSize: "0.85rem",
                fontWeight: "700"
              }}>
                {sentiment_label}
              </div>
            </div>

            {/* Simple 3-level indicator */}
            <div style={{
              display: "flex",
              gap: "0.5rem",
              marginTop: "0.75rem"
            }}>
              <div style={{
                flex: 1,
                height: "8px",
                borderRadius: "4px",
                background: sentiment_label === "Negative" ? "#ef4444" : "#e5e7eb"
              }} />
              <div style={{
                flex: 1,
                height: "8px",
                borderRadius: "4px",
                background: sentiment_label === "Neutral" ? "#f59e0b" : "#e5e7eb"
              }} />
              <div style={{
                flex: 1,
                height: "8px",
                borderRadius: "4px",
                background: sentiment_label === "Positive" ? "#22c55e" : "#e5e7eb"
              }} />
            </div>
            <div style={{
              display: "flex",
              justifyContent: "space-between",
              marginTop: "0.5rem",
              fontSize: "0.75rem",
              color: "#9ca3af",
              fontWeight: "600"
            }}>
              <span>Prices Down</span>
              <span>Stable</span>
              <span>Prices Up</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
});

/* ---------- SIMPLIFIED PRICE FORECAST ---------- */
const EnhancedPriceForecast = React.memo(function EnhancedPriceForecast({ forecast }) {
  if (!forecast) return null;
  const {
    date,
    baseline_price,
    weather_adjustment,
    disease_adjustment,
    sentiment_adjustment,
    final_price,
    min_price,
    max_price,
    price_unit,  // NEW: Extract price_unit
    weather_reason,
    disease_reason,
    sentiment_reason
  } = forecast;

  // Format the unit display
  const unitDisplay = price_unit === "per cover" ? "/cover" : "/Q";


  // Calculate total impact
  const totalAdjustment = weather_adjustment + disease_adjustment + sentiment_adjustment;
  const hasPositiveImpact = totalAdjustment > 0;

  return (
    <div style={{
      background: "white",
      borderRadius: "10px",
      padding: "1.25rem",
      marginBottom: "0.75rem",
      border: "2px solid #e5e7eb",
      boxShadow: "0 2px 8px rgba(0,0,0,0.08)"
    }}>
      {/* Date */}
      <div style={{
        fontSize: "0.9rem",
        fontWeight: "600",
        color: "#6b7280",
        marginBottom: "0.75rem",
        display: "flex",
        alignItems: "center",
        gap: "0.5rem"
      }}>
        <span>📅</span>
        {formatDate(date)}
      </div>

      {/* Main Price Display - Big and Clear with Range */}
      <div style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        marginBottom: "1rem"
      }}>
        <div>
          <div style={{ fontSize: "0.8rem", color: "#9ca3af", marginBottom: "0.25rem" }}>
            Expected Price Range
          </div>
          <div style={{
            fontSize: "1.75rem",
            fontWeight: "700",
            color: "#16a34a",
            display: "flex",
            alignItems: "baseline",
            gap: "0.5rem"
          }}>
            <span>₹{Math.round(min_price).toLocaleString()}</span>
            <span style={{ fontSize: "1.2rem", color: "#6b7280", fontWeight: "500" }}>-</span>
            <span>₹{Math.round(max_price).toLocaleString()}</span>
            <span style={{ fontSize: "0.9rem", color: "#9ca3af" }}>{unitDisplay}</span>
          </div>
          <div style={{
            fontSize: "0.75rem",
            color: "#6b7280",
            marginTop: "0.25rem",
            fontWeight: "500"
          }}>
            Mid-point: ₹{Math.round(final_price).toLocaleString()}{unitDisplay}
          </div>
        </div>

        {/* Impact Indicator */}
        {Math.abs(totalAdjustment) > 0.5 && (
          <div style={{
            background: hasPositiveImpact ? "#dcfce7" : "#fee2e2",
            color: hasPositiveImpact ? "#166534" : "#991b1b",
            padding: "0.5rem 1rem",
            borderRadius: "20px",
            fontSize: "0.95rem",
            fontWeight: "700",
            display: "flex",
            alignItems: "center",
            gap: "0.5rem"
          }}>
            {hasPositiveImpact ? "📈" : "📉"}
            {hasPositiveImpact ? "+" : ""}{totalAdjustment.toFixed(1)}%
          </div>
        )}
      </div>

      {/* Simple Impact Summary - Only if significant */}
      {Math.abs(totalAdjustment) > 0.5 && (
        <div style={{
          background: "#f9fafb",
          borderRadius: "8px",
          padding: "0.75rem",
          fontSize: "0.85rem",
          color: "#6b7280"
        }}>
          <div style={{ fontWeight: "600", color: "#374151", marginBottom: "0.5rem" }}>
            What's affecting the price:
          </div>
          {Math.abs(weather_adjustment) > 0.5 && (
            <div style={{ marginBottom: "0.25rem" }}>
              🌤️ Weather: {weather_adjustment > 0 ? "Favorable" : "Challenging"}
              ({weather_adjustment > 0 ? "+" : ""}{weather_adjustment.toFixed(1)}%)
            </div>
          )}
          {Math.abs(sentiment_adjustment) > 0.5 && (
            <div style={{ marginBottom: "0.25rem" }}>
              📰 Market News: {sentiment_adjustment > 0 ? "Positive" : "Negative"}
              ({sentiment_adjustment > 0 ? "+" : ""}{sentiment_adjustment.toFixed(1)}%)
            </div>
          )}
          {Math.abs(disease_adjustment) > 0.5 && (
            <div>
              🦠 Disease Risk: Alert
              ({disease_adjustment.toFixed(1)}%)
            </div>
          )}
        </div>
      )}

      {/* Expandable Details - Only for interested users */}
      {(weather_reason || disease_reason || sentiment_reason) && (
        <details style={{ marginTop: "0.75rem" }}>
          <summary style={{
            fontSize: "0.85rem",
            color: "#6b7280",
            cursor: "pointer",
            padding: "0.5rem",
            background: "#f9fafb",
            borderRadius: "6px",
            fontWeight: "500"
          }}>
            📝 More Details
          </summary>
          <div style={{
            marginTop: "0.5rem",
            fontSize: "0.8rem",
            color: "#6b7280",
            padding: "0.75rem",
            background: "#f9fafb",
            borderRadius: "6px",
            lineHeight: "1.6"
          }}>
            {weather_reason && <div style={{ marginBottom: "0.5rem" }}>🌤️ {weather_reason}</div>}
            {sentiment_reason && <div style={{ marginBottom: "0.5rem" }}>📰 {sentiment_reason}</div>}
            {disease_reason && <div>🦠 {disease_reason}</div>}
          </div>
        </details>
      )}
    </div>
  );
});

/* ---------- SIMPLIFIED AI INSIGHTS PANEL ---------- */
const AIInsightsPanel = React.memo(function AIInsightsPanel({ insights, queryType }) {
  const [isExpanded, setIsExpanded] = useState(false);
  if (!insights) return null;

  const interpretation = insights.interpretation || {};
  const recommendations = insights.recommendations || [];
  const risks = insights.risk_assessment?.specific_risks || [];
  const opportunities = insights.opportunities || [];

  const marketToneMap = {
    shortage: { icon: "📈", color: "bg-green-100 text-green-800", text: "Shortage" },
    oversupply: { icon: "📉", color: "bg-red-100 text-red-800", text: "Oversupply" },
    balanced: { icon: "⚖️", color: "bg-yellow-100 text-yellow-800", text: "Balanced" }
  };
  const tone = marketToneMap[interpretation.market_condition] || marketToneMap.balanced;

  return (
    <div className="mt-6 border border-emerald-200 bg-emerald-50 rounded-2xl p-6 shadow-sm">
      <div
        className="flex items-center justify-between cursor-pointer"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-3">
          <span className="text-2xl">🧠</span>
          <div>
            <h3 className="m-0 text-lg font-bold text-gray-900">Smart Market Advice</h3>
            {insights.summary && (
              <p className="m-0 text-sm text-gray-700">{insights.summary}</p>
            )}
          </div>
        </div>
        <div className={`w-8 h-8 flex items-center justify-center rounded-full bg-white text-sm font-semibold text-emerald-600 shadow ${isExpanded ? "rotate-180" : ""}`}>
          ▼
        </div>
      </div>

      {isExpanded && (
        <div className="mt-4 space-y-4">
          <div className="flex items-center gap-3">
            <span className={`px-3 py-1 rounded-full text-xs font-semibold ${tone.color}`}>
              {tone.icon} {tone.text}
            </span>
            <span className="text-xs text-gray-600">
              Confidence: {interpretation.confidence_level || "unknown"}
            </span>
          </div>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white rounded-xl p-4 border border-emerald-100">
              <h4 className="text-sm font-semibold text-gray-800 mb-2">💡 Recommended Actions</h4>
              {recommendations.length === 0 && <p className="text-xs text-gray-500">No recommendations generated.</p>}
              {recommendations.slice(0, 3).map((rec, idx) => (
                <div key={idx} className="mb-3 last:mb-0">
                  <div className="text-sm font-semibold text-gray-900">{idx + 1}. {rec.action}</div>
                  <div className="text-xs text-gray-600">⏰ {rec.timing}</div>
                  {rec.expected_outcome && (
                    <div className="text-xs text-gray-500 mt-1">✓ {rec.expected_outcome}</div>
                  )}
                </div>
              ))}
            </div>

            <div className="bg-white rounded-xl p-4 border border-red-100">
              <h4 className="text-sm font-semibold text-gray-800 mb-2">⚠️ Risks & Mitigation</h4>
              {risks.length === 0 && <p className="text-xs text-gray-500">No critical risks detected.</p>}
              {risks.slice(0, 2).map((risk, idx) => (
                <div key={idx} className="mb-3 last:mb-0">
                  <div className="text-sm font-semibold text-gray-900">{risk.risk}</div>
                  <div className="text-xs text-gray-600">Impact: {risk.impact || "n/a"} | Probability: {risk.probability || "n/a"}</div>
                  {risk.mitigation && (
                    <div className="text-xs text-gray-500 mt-1">💡 {risk.mitigation}</div>
                  )}
                </div>
              ))}
            </div>
          </div>

          {opportunities.length > 0 && (
            <div className="bg-white rounded-xl p-4 border border-blue-100">
              <h4 className="text-sm font-semibold text-gray-800 mb-2">🚀 Opportunities</h4>
              <div className="grid md:grid-cols-2 gap-3">
                {opportunities.slice(0, 2).map((opp, idx) => (
                  <div key={idx} className="text-sm text-gray-800">
                    <div className="font-semibold">{opp.opportunity}</div>
                    {opp.action_required && (
                      <div className="text-xs text-gray-600">Action: {opp.action_required}</div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
});

/* Interactive Options - ChatGPT Style Buttons */
const InteractiveOptions = React.memo(({ prompt, options, onSelect, disabled }) => {
  if (!options || options.length === 0) return null;
  return (
    <div className="mt-4 p-4 bg-gradient-to-r from-gray-50 to-gray-100 rounded-xl border border-gray-200">
      <p className="text-sm font-semibold text-gray-700 mb-3">{prompt}</p>
      <div className="flex flex-wrap gap-2">
        {options.map((opt, idx) => (
          <button
            key={opt.value || idx}
            type="button"
            onClick={() => onSelect(opt.value || opt.label, opt.type || "selection")}
            disabled={disabled}
            className="px-4 py-2.5 bg-white hover:bg-gray-50 border-2 border-gray-300 hover:border-green-500 rounded-lg text-gray-800 font-medium text-sm transition-all duration-200 transform hover:-translate-y-0.5 hover:shadow-md disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:translate-y-0"
          >
            {opt.label}
          </button>
        ))}
      </div>
    </div>
  );
});

/* Weather Impact Display Component */
const WeatherImpactDisplay = React.memo(({ weatherData }) => {
  if (!weatherData || !weatherData.weather_summary) return null;

  const { weather_summary, weather_factor_summary } = weatherData;

  return (
    <div className="mt-3 bg-gradient-to-br from-sky-50 via-blue-50 to-cyan-50 border-2 border-sky-300 rounded-xl p-4 shadow-md">
      <h5 className="flex items-center gap-2 text-sky-700 font-bold text-base mb-3">
        <span className="text-2xl">🌤️</span>
        Weather Impact Analysis
      </h5>
      <div className="space-y-2.5">
        <div className="flex justify-between items-center px-3 py-2 bg-white rounded-lg text-sm">
          <strong className="text-sky-900">Condition:</strong>
          <span className="text-gray-700 font-medium">{weather_summary.condition || 'Unknown'}</span>
        </div>
        {weather_summary.rain_mm !== undefined && (
          <div className="flex justify-between items-center px-3 py-2 bg-white rounded-lg text-sm">
            <strong className="text-sky-900">Precipitation:</strong>
            <span className="text-gray-700 font-medium">{weather_summary.rain_mm} mm</span>
          </div>
        )}
        {weather_summary.temp_c !== undefined && (
          <div className="flex justify-between items-center px-3 py-2 bg-white rounded-lg text-sm">
            <strong className="text-sky-900">Temperature:</strong>
            <span className="text-gray-700 font-medium">{weather_summary.temp_c}°C</span>
          </div>
        )}
        {weather_factor_summary && (
          <div className="px-3 py-2 bg-white rounded-lg text-sm">
            <strong className="text-sky-900 block mb-2">Impact Days:</strong>
            <div className="flex flex-wrap gap-2">
              {weather_factor_summary.heavy > 0 && (
                <span className="px-3 py-1 bg-red-100 text-red-800 rounded-full text-xs font-semibold">
                  Heavy: {weather_factor_summary.heavy} days
                </span>
              )}
              {weather_factor_summary.moderate > 0 && (
                <span className="px-3 py-1 bg-yellow-100 text-yellow-800 rounded-full text-xs font-semibold">
                  Moderate: {weather_factor_summary.moderate} days
                </span>
              )}
              {weather_factor_summary.none > 0 && (
                <span className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-xs font-semibold">
                  Normal: {weather_factor_summary.none} days
                </span>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
});

/* ---------- Prediction Result ---------- */
const PredictionResult = React.memo(function PredictionResult({ response, isPriceData, isSingleDay, weatherData }) {
  const [viewMode, setViewMode] = useState("total");
  const [isExpanded, setIsExpanded] = useState(true); // Default expanded
  const totalData = response?.total_predicted || [];
  let commodityData = response?.commodity_daily || {};

  // 🔥 Override commodityData when aggregate mode + breakdown exists
  if (response.mode === "aggregate" && response.commodity_breakdown) {
    commodityData = {};
    response.commodity_breakdown.forEach((item) => {
      commodityData[item.commodity] = [
        {
          date: response.total_predicted?.[0]?.date,
          predicted_value: item.total_predicted_value,
        },
      ];
    });
  }
  const isWeightMetricSelected = !isPriceData && isWeightMetric(response?.metric_name);

  // If single day query, show only first day
  const displayData = isSingleDay ? totalData.slice(0, 1) : totalData;

  const commodityChartData = useMemo(() => {
    if (!displayData.length) return [];
    if (!commodityData || typeof commodityData !== "object") return [];
    return displayData.map((day) => {
      const point = { date: day.date, total: day.total_predicted_value };
      Object.entries(commodityData).forEach(([commodity, data]) => {
        const cd = Array.isArray(data)
          ? data.find((d) => d.date === day.date)
          : null;
        point[commodity] = cd ? cd.predicted_value : 0;
      });
      return point;
    });
  }, [displayData, commodityData]);

  const summaryItems = useMemo(() => {
    if (!response) return [];
    return summarizeForecast(response, weatherData, isPriceData);
  }, [response, weatherData, isPriceData]);

  if (!response) return null;

  const hasCommodityData = Object.keys(commodityData).length > 0;

  const buildTooltipValue = (value, label = response.metric_name) => {
    const numericValue = toNumber(value);
    const roundedValue = formatValue(response.metric_name, numericValue).toLocaleString();
    if (isWeightMetricSelected) {
      const kgValue = convertQuintalsToKg(numericValue).toLocaleString();
      return [`${roundedValue} Q (≈ ${kgValue} kg)`, label];
    }
    const suffix = isPriceData ? "/Q" : getUnitSuffix(response.metric_name);
    const prefix = isPriceData ? "₹" : "";
    return [`${prefix}${roundedValue}${suffix}`, label];
  };

  return (
    <div className="mt-4">
      {/* Collapse/Expand Toggle */}
      <div
        className="flex items-center justify-between px-4 py-3 bg-gradient-to-r from-gray-100 to-gray-50 border border-gray-300 rounded-xl cursor-pointer hover:from-gray-200 hover:to-gray-100 transition-all shadow-sm"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-2 font-semibold text-gray-800 text-sm">
          <span className="text-xl">{isPriceData ? "💰" : "📊"}</span>
          <span>{isPriceData ? "Price Forecast" : "Arrival Forecast"}</span>
          <span className="text-xs text-gray-500 font-normal">
            ({displayData.length} {displayData.length === 1 ? "day" : "days"})
          </span>
        </div>
        <div
          className={`flex items-center justify-center w-6 h-6 bg-white rounded-md text-xs transition-transform duration-200 ${isExpanded ? "rotate-180" : ""
            }`}
        >
          ▼
        </div>
      </div>

      {/* Collapsible Content */}
      {isExpanded && (
        <div>
          {/* Summary Explanation Section */}
          {summaryItems.length > 0 && (
            <div className="mt-3 bg-white border border-gray-200 rounded-2xl p-4 shadow-sm">
              <h4 className="font-bold text-gray-900 text-sm mb-3 flex items-center gap-2">
                <span className="text-lg">💡</span>
                <span>Forecast Highlights</span>
              </h4>
              <div className="grid gap-3 md:grid-cols-2">
                {summaryItems.map((item, idx) => (
                  <div
                    key={`${item.title}-${idx}`}
                    className="flex items-start gap-3 p-3 rounded-xl border border-gray-100 bg-gradient-to-r from-gray-50 to-white"
                  >
                    <span className="text-xl">{item.icon}</span>
                    <div className="flex-1">
                      <div className="text-sm font-semibold text-gray-900 flex items-center gap-2">
                        {item.title}
                        {item.badge && (
                          <span
                            className={`text-xs font-semibold px-2 py-0.5 rounded-full ${item.badge.tone === "up"
                              ? "bg-green-100 text-green-700"
                              : item.badge.tone === "down"
                                ? "bg-red-100 text-red-700"
                                : "bg-yellow-100 text-yellow-700"
                              }`}
                          >
                            {item.badge.label}
                          </span>
                        )}
                      </div>
                      <div className="text-xs text-gray-600 mt-0.5">{item.detail}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {hasCommodityData && !isSingleDay && (
            <div className="flex gap-2 mt-3 mb-3">
              <button
                className={`px-4 py-2 rounded-lg font-medium text-sm transition-all ${viewMode === "total"
                  ? "bg-green-600 text-white shadow-md"
                  : "bg-gray-200 text-gray-700 hover:bg-gray-300"
                  }`}
                onClick={() => setViewMode("total")}
              >
                📊 Total Overview
              </button>
              <button
                className={`px-4 py-2 rounded-lg font-medium text-sm transition-all ${viewMode === "commodity"
                  ? "bg-green-600 text-white shadow-md"
                  : "bg-gray-200 text-gray-700 hover:bg-gray-300"
                  }`}
                onClick={() => setViewMode("commodity")}
              >
                🌾 Commodity Details
              </button>
            </div>
          )}

          {viewMode === "total" && (
            <>
              {displayData.length > 0 && (
                <div className="mt-3 bg-gradient-to-br from-gray-50 to-white border border-gray-200 rounded-xl p-4 shadow-sm">
                  <h4 className="font-bold text-gray-800 mb-3 text-sm flex items-center gap-2">
                    📅 {isSingleDay ? (isPriceData ? "Tomorrow's Price" : `Tomorrow's ${response.metric_name}`) : "Forecast Summary"}
                  </h4>
                  <div className="overflow-x-auto">
                    <table className="w-full border-collapse">
                      <thead>
                        <tr className="bg-gradient-to-r from-gray-100 to-gray-50">
                          <th className="px-4 py-2.5 text-left text-xs font-bold text-gray-700 border-b-2 border-gray-300">Date</th>
                          <th className="px-4 py-2.5 text-left text-xs font-bold text-gray-700 border-b-2 border-gray-300">{response.metric_name}</th>
                        </tr>
                      </thead>
                      <tbody>
                        {displayData.map((item, i) => (
                          <tr key={i} className={`${i % 2 === 0 ? "bg-white" : "bg-gray-50"} hover:bg-blue-50 transition-colors`}>
                            <td className="px-4 py-2.5 text-sm text-gray-700 border-b border-gray-200">{formatDate(item.date)}</td>
                            <td className="px-4 py-2.5 text-sm font-semibold text-gray-900 border-b border-gray-200">
                              {(() => {
                                const numericValue = toNumber(item.total_predicted_value);
                                const roundedValue = formatValue(response.metric_name, numericValue);

                                const formattedRoundedValue =
                                  typeof roundedValue === "number" ? roundedValue.toLocaleString() : roundedValue;

                                const unitSuffix = isPriceData
                                  ? (item.price_unit === "per cover" ? "/cover" : "/Q")
                                  : getUnitSuffix(response.metric_name);
                                const kgValue = isWeightMetricSelected
                                  ? convertQuintalsToKg(numericValue).toLocaleString()
                                  : null;

                                // 🔥 ADD PRICE RANGE (only for PRICE type)
                                const PRICE_RANGE = 0.05; // ±5%
                                const minPrice = numericValue * (1 - PRICE_RANGE);
                                const maxPrice = numericValue * (1 + PRICE_RANGE);

                                return (
                                  <div className="flex flex-col">
                                    {/* If PRICE → show RANGE */}
                                    {isPriceData ? (
                                      <>
                                        <span>
                                          ₹{Math.round(minPrice).toLocaleString()} - ₹{Math.round(maxPrice).toLocaleString()} {unitSuffix}
                                        </span>
                                        <span className="text-xs text-gray-500 font-medium mt-0.5">
                                          Mid: ₹{formattedRoundedValue} {unitSuffix}
                                        </span>
                                      </>
                                    ) : (
                                      // If NOT PRICE → show normal metric
                                      <span>
                                        {formattedRoundedValue}
                                        {unitSuffix}
                                      </span>
                                    )}

                                    {/* Weight Conversion */}
                                    {isWeightMetricSelected && kgValue && (
                                      <span className="text-xs text-gray-500 font-medium mt-0.5">
                                        ≈ {kgValue} kg
                                      </span>
                                    )}
                                  </div>
                                );
                              })()}
                            </td>

                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* Display Weather Impact if available */}
              {weatherData && (
                <WeatherImpactDisplay weatherData={weatherData} />
              )}

              {!isSingleDay && (
                <div className="mt-3 bg-gradient-to-br from-gray-50 to-white border border-gray-200 rounded-xl p-4 shadow-sm">
                  <h4 className="font-bold text-gray-800 mb-3 text-sm flex items-center gap-2">📈 Trend Chart</h4>
                  {displayData.length > 0 ? (
                    <ResponsiveContainer width="100%" height={220}>
                      <LineChart data={displayData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                        <XAxis dataKey="date" tickFormatter={(d) => d.slice(5)} />
                        <YAxis />
                        <Tooltip formatter={(value) => buildTooltipValue(value)} />
                        <Line dataKey="total_predicted_value" name={response.metric_name} type="monotone" stroke={isPriceData ? "#f59e0b" : "#16a34a"} strokeWidth={2} />
                      </LineChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="text-center text-gray-500 py-4">No forecast data</div>
                  )}
                </div>
              )}
            </>
          )}

          {viewMode === "commodity" && hasCommodityData && !isSingleDay && (
            <div className="mt-3 bg-gradient-to-br from-gray-50 to-white border border-gray-200 rounded-xl p-4 shadow-sm">
              <h4 className="font-bold text-gray-800 mb-3 text-sm flex items-center gap-2">📈 Commodity Comparison</h4>
              {commodityChartData.length > 0 ? (
                <ResponsiveContainer width="100%" height={250}>
                  <LineChart data={commodityChartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis dataKey="date" tickFormatter={(d) => d.slice(5)} />
                    <YAxis />
                    <Tooltip formatter={(value, name) => buildTooltipValue(value, name)} />
                    {Object.keys(commodityData).map((c, i) => (
                      <Line key={c} type="monotone" dataKey={c} name={c} stroke={colors[i % colors.length]} strokeWidth={2} />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="no-chart">No commodity data</div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
});

/* ---------- Message Formatter ---------- */
const formatMessage = (message) => {
  if (!message) return "";

  // Split into lines for processing
  const lines = message.split('\n').map(l => l.trim()).filter(l => l);
  let result = [];
  let inList = false;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // Check if line is a section header (emoji + text ending with :)
    if (line.match(/^([🔍🔴🟡💡📊📈📉⚠️✅💰📰🌤️🦠👥⏰🤖])\s*(.+):$/)) {
      // Close any open list before header
      if (inList) {
        result.push('</ul>');
        inList = false;
      }
      result.push(`<div class="font-semibold text-gray-900 mt-4 mb-2">${line}</div>`);
    }
    // Check if line is an emoji bullet point (🔴, 🟡, 💡, etc. at start)
    else if (line.match(/^([🔴🟡💡🔵⚫🟢🟠🟣⚪🟤])\s+(.+)$/)) {
      const match = line.match(/^([🔴🟡💡🔵⚫🟢🟠🟣⚪🟤])\s+(.+)$/);
      if (!inList) {
        result.push('<ul class="space-y-2 my-3 ml-4">');
        inList = true;
      }
      result.push(`<li class="flex items-start gap-2"><span class="flex-shrink-0">${match[1]}</span><span class="text-gray-800">${match[2]}</span></li>`);
    }
    // Check for regular bullet points
    else if (line.match(/^[•\-–—]\s+(.+)$/)) {
      if (!inList) {
        result.push('<ul class="list-disc pl-6 space-y-2 my-3">');
        inList = true;
      }
      const content = line.replace(/^[•\-–—]\s+/, '');
      result.push(`<li class="text-gray-800">${content}</li>`);
    }
    // Check for numbered lists
    else if (line.match(/^(\d+)\.\s+(.+)$/)) {
      const match = line.match(/^(\d+)\.\s+(.+)$/);
      if (!inList || match[1] === '1') {
        if (inList) result.push('</ul>');
        result.push('<ol class="list-decimal pl-6 space-y-2 my-3">');
        inList = true;
      }
      result.push(`<li class="text-gray-800">${match[2]}</li>`);
    }
    // Regular text
    else {
      if (inList) {
        result.push('</ul>');
        inList = false;
      }
      result.push(`<div class="text-gray-800 mb-3">${line}</div>`);
    }
  }

  // Close any open list at the end
  if (inList) {
    result.push('</ul>');
  }

  return result.join('');
};

/* ---------- Chat Message ---------- */
const ChatMessage = React.memo(({
  message,
  isBot,
  predictionData,
  isPriceData,
  aiInsights,
  sentimentInfo,
  detailedForecasts,
  isSingleDay,
  interactiveOptions,
  isTyping,
  onOptionSelect,
  weatherData,
  userQuery,
  queryType,
  sessionId
}) => {
  const shouldHideText = isBot && (predictionData || (aiInsights && queryType === "advisory_only"));
  return (
    <div className="bg-white">
      <div className={`max-w-3xl mx-auto px-4 py-6 flex gap-6 ${!isBot ? "justify-end" : ""}`}>
        {isBot && (
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-green-500 to-emerald-600 flex items-center justify-center flex-shrink-0 shadow-sm">
            <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
          </div>
        )}
        <div className={`${isBot ? "flex-1" : "max-w-[80%]"} space-y-4`}>
          {isTyping ? (
            <ProgressiveLoadingIndicator sessionId={sessionId} />
          ) : (
            <>
              {/* Hide text response if we have chart data to show (arrival/price predictions) */}
              {!shouldHideText && (
                <div className="text-gray-800 leading-relaxed text-base message-content" dangerouslySetInnerHTML={{ __html: formatMessage(message) }} />
              )}
              {isBot && interactiveOptions && (
                <InteractiveOptions
                  prompt={interactiveOptions.question}
                  options={interactiveOptions.options}
                  onSelect={onOptionSelect}
                  disabled={false}
                />
              )}

              {/* Display detailed price forecasts - limit to 1 if single day */}
              {isBot && detailedForecasts && detailedForecasts.length > 0 && (
                <div style={{ marginTop: "1rem" }}>
                  {isSingleDay && detailedForecasts.length > 1 && (
                    <div style={{
                      background: "#dbeafe",
                      padding: "0.75rem",
                      borderRadius: "8px",
                      marginBottom: "0.75rem",
                      fontSize: "0.9rem",
                      color: "#1e40af",
                      fontWeight: "600",
                      display: "flex",
                      alignItems: "center",
                      gap: "0.5rem"
                    }}>
                      <span>ℹ️</span>
                      <span>You asked for {/\btoday\b/i.test(userQuery || "") ? "today's" : "tomorrow's"} price - showing only the first day from {detailedForecasts.length} days of data</span>
                    </div>
                  )}
                  <h4 style={{
                    fontSize: "1rem",
                    fontWeight: "600",
                    color: "#14532d",
                    marginBottom: "0.75rem"
                  }}>
                    {isSingleDay ? `📊 ${/\btoday\b/i.test(userQuery || "") ? "Today's" : "Tomorrow's"} Price Details` : "📊 Detailed Price Breakdown"}
                  </h4>
                  {(isSingleDay ? detailedForecasts.slice(0, 1) : detailedForecasts).map((forecast, idx) => (
                    <EnhancedPriceForecast key={idx} forecast={forecast} />
                  ))}
                </div>
              )}

              {/* Keep existing prediction display */}
              {isBot && predictionData && (
                <PredictionResult
                  response={predictionData}
                  isPriceData={isPriceData}
                  isSingleDay={isSingleDay}
                  weatherData={weatherData}
                />
              )}

              {isBot && aiInsights && (
                <AIInsightsPanel insights={aiInsights} queryType={queryType} />
              )}
            </>
          )}
        </div>
        {!isBot && (
          <div className="w-8 h-8 rounded-full bg-purple-600 flex items-center justify-center flex-shrink-0">
            <svg className="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" clipRule="evenodd" />
            </svg>
          </div>
        )}
      </div>
    </div>
  );
});

/* ---------- Real-Time Loading Indicator (Gets status from backend) ---------- */
const ProgressiveLoadingIndicator = ({ sessionId }) => {
  const [currentStatus, setCurrentStatus] = useState("Analyzing your query...");
  const [prevStatus, setPrevStatus] = useState("");

  useEffect(() => {
    // Poll backend for real status every 500ms
    const pollInterval = setInterval(async () => {
      try {
        const backendUrl = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8005';
        const response = await axios.get(`${backendUrl}/status/${sessionId}`);
        if (response.data && response.data.status) {
          if (response.data.status !== currentStatus) {
            setPrevStatus(currentStatus);
            setCurrentStatus(response.data.status);
          }
        }
      } catch (error) {
        console.log("Status poll:", error.message);
      }
    }, 500);

    return () => clearInterval(pollInterval);
  }, [sessionId, currentStatus]);

  // Get icon and color based on status
  const getStatusInfo = (status) => {
    if (status.includes("Analyzing")) {
      return { icon: "🔍", color: "#3b82f6", bg: "#dbeafe" };
    } else if (status.includes("Fetching")) {
      return { icon: "📊", color: "#8b5cf6", bg: "#ede9fe" };
    } else if (status.includes("Processing")) {
      return { icon: "🧠", color: "#ec4899", bg: "#fce7f3" };
    } else if (status.includes("Preparing")) {
      return { icon: "✨", color: "#10b981", bg: "#d1fae5" };
    }
    return { icon: "⏳", color: "#6b7280", bg: "#f3f4f6" };
  };

  const statusInfo = getStatusInfo(currentStatus);

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: '10px',
      padding: '4px 0'
    }}>
      {/* Animated spinning circle */}
      <div style={{
        width: '16px',
        height: '16px',
        border: '2px solid #e5e7eb',
        borderTopColor: statusInfo.color,
        borderRadius: '50%',
        animation: 'smoothSpin 0.8s linear infinite'
      }} />

      {/* Status text - clean and simple */}
      <div style={{
        fontSize: '14px',
        color: '#6b7280',
        fontWeight: '500',
        animation: 'fadeInText 0.3s ease-out'
      }}>
        {currentStatus}
      </div>
    </div>
  );
};

/* ---------- Main App ---------- */
function App() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [language, setLanguage] = useState(() => {
    return localStorage.getItem("selected_language") || "English";
  });
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content: "👋 <strong>Welcome!</strong> I'm CropCast AI, your intelligent agricultural assistant.<br/><br/>Ask me about commodity prices, market forecasts, and agricultural insights."
    }
  ]);

  const messagesEndRef = useRef(null);
  const composerRef = useRef(null);
  const [sessionId] = useState(() => {
    // Always generate a fresh session on page load/refresh
    // This ensures backend doesn't reuse old context (commodity, variant, etc.)
    const id = uuidv4();
    localStorage.setItem("chat_session", id);
    return id;
  });

  const [originalDays, setOriginalDays] = useState(7);
  const [pendingClarification, setPendingClarification] = useState(null);
  const [isListening, setIsListening] = useState(false);

  const handleLanguageChange = (lang) => {
    setLanguage(lang);
    localStorage.setItem("selected_language", lang);
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const recognitionRef = useRef(null);
  const startVoiceInput = () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert("Voice recognition not supported in this browser. Please use Chrome or Edge.");
      return;
    }

    if (!recognitionRef.current) {
      const recognition = new SpeechRecognition();

      // Language mapping based on selected language
      const langMap = {
        "English": "en-IN",
        "తెలుగు": "te-IN",
        "हिंदी": "hi-IN"
      };
      recognition.lang = langMap[language] || "en-IN";
      recognition.continuous = false;
      recognition.interimResults = false;

      // When speech is detected
      recognition.onstart = () => {
        setIsListening(true);
        console.log("🎙️ Listening...");
      };

      // When speech recognition ends
      recognition.onend = () => {
        setIsListening(false);
        console.log("🛑 Stopped listening");
      };

      // When result is received
      recognition.onresult = (e) => {
        const transcript = e.results[0][0].transcript;
        setQuery(transcript);
        setIsListening(false);
        console.log("📝 Transcript:", transcript);
      };

      // Error handling
      recognition.onerror = (e) => {
        setIsListening(false);
        console.error("❌ Speech recognition error:", e.error);

        if (e.error === 'no-speech') {
          alert("No speech detected. Please try again.");
        } else if (e.error === 'not-allowed') {
          alert("Microphone access denied. Please allow microphone access in browser settings.");
        } else {
          alert(`Speech recognition error: ${e.error}`);
        }
      };

      recognitionRef.current = recognition;
    }

    try {
      recognitionRef.current.start();
    } catch (error) {
      console.error("Error starting recognition:", error);
      setIsListening(false);
    }
  };

  // Handler for variety/metric/commodity selection
  const handleOptionSelect = useCallback(async (selectedValue, selectionType, context) => {
    console.log("🔘 Button clicked:", selectedValue);
    console.log("📋 Pending clarification:", pendingClarification);

    setLoading(true);

    // Keep the interactive options visible AND clickable (don't remove or disable)
    // This allows users to change their selection by clicking another button

    // Format the selected value for display
    const metricLabels = {
      total_bags: "📦 Total Bags",
      number_of_arrivals: "🔢 Arrivals Count",
      number_of_lots: "📊 Number of Lots",
      total_weight: "⚖️ Quantity (Quintals)",
      number_of_farmers: "👥 Number of Farmers",
      total_revenue: "💰 Revenue"
    };

    const displayValue = metricLabels[selectedValue] || selectedValue;

    setMessages((p) => [...p, { role: "user", content: displayValue }]);
    setMessages((p) => [...p, { role: "assistant", content: "⏳ Processing your selection...", isTyping: true }]);

    try {
      let queryToSend = selectedValue;
      const requestContext = {};
      const isMetricSelection = selectionType === "metric" || pendingClarification?.type === "metric";
      const metricPhrases = {
        total_bags: "Predict total bags",
        number_of_arrivals: "Predict number of arrivals",
        number_of_lots: "Predict number of lots",
        total_weight: "Predict total weight in quintals",
        number_of_farmers: "Predict number of farmers",
        total_revenue: "Predict total revenue"
      };

      // If this is a variety selection
      if (pendingClarification && pendingClarification.isVarietySelection && pendingClarification.originalQuery) {
        const originalQuery = pendingClarification.originalQuery;
        // Append variety to original query
        queryToSend = `${originalQuery} variety ${selectedValue}`;
        console.log("✅ Variety selection query:", queryToSend);
        console.log("   Original query was:", originalQuery);
        console.log("   Selected variety:", selectedValue);

        // Keep advisory flag for next request
        const isAdvisory = pendingClarification.isAdvisoryQuery;
        setPendingClarification({ isAdvisoryQuery: isAdvisory });
      }
      // If this is a backend CLARIFICATION (metric/commodity selection)
      else if (selectionType === "clarification") {
        // For clarification responses, send ONLY the selected label
        // Backend session already has the context stored
        queryToSend = selectedValue;
        console.log("✅ CLARIFICATION response - sending label only:", queryToSend);
        setPendingClarification(null);
      }
      else if (isMetricSelection) {
        // Rerun the original arrival query but include metric context
        requestContext.metric = selectedValue;
        requestContext.force_arrival = true;
        const original = pendingClarification?.originalQuery || "";
        const cleanQuery = original.replace(/\b(expected\s+)?arrivals?\s+/i, "").trim();
        const metricPrompt = metricPhrases[selectedValue] || selectedValue;
        queryToSend = cleanQuery ? `${metricPrompt} ${cleanQuery}`.trim() : metricPrompt;
        if (!mentionsCommodity(original)) {
          requestContext.clear_commodity = true;
        }
        console.log("✅ Metric selected:", selectedValue, "| Rerunning query:", queryToSend);
        setPendingClarification(null);
      }
      // If this is a metric selection from frontend - DON'T rebuild query!
      // Just send the metric name and let backend handle it with its session
      else if (pendingClarification && pendingClarification.originalQuery) {
        // Send a clear query that emphasizes the metric to avoid AI confusion
        // Remove "arrivals" from original query and replace with specific metric
        // Extract location and timeframe from original query
        const original = pendingClarification.originalQuery;
        // Remove generic "arrivals" word but KEEP preposition (in/for/at)
        const cleanQuery = original.replace(/\b(expected\s+)?arrivals?\s+/i, '').trim();

        // Build new clear query with metric first
        queryToSend = `${metricPhrases[selectedValue]} ${cleanQuery}`;

        console.log("✅ Sending metric selection with original query:", queryToSend);
        console.log("   Original query:", pendingClarification.originalQuery);
        console.log("   Selected metric:", selectedValue);
        console.log("   Cleaned query:", cleanQuery);

        setPendingClarification(null);
      } else {
        console.warn("⚠️ No pending clarification found!");
      }

      // Send the selection back to backend
      console.log("📡 Sending to backend:", queryToSend);
      const requestBody = {
        query: queryToSend,
        session_id: sessionId
      };
      if (Object.keys(requestContext).length > 0) {
        requestBody.context = requestContext;
      }
      const res = await axios.post(
        `${process.env.REACT_APP_BACKEND_URL}/ai/chat`,
        requestBody,
        { timeout: 180000 }
      );

      const aiResponse = res.data.response || "Selection processed.";
      const toolResults = res.data.tool_results || {};
      const aiInsights = res.data.ai_insights;
      const queryType = res.data.query_type || "prediction";

      let predictionData = null;
      let isPriceData = false;
      let sentimentInfo = null;
      let detailedForecasts = [];
      let weatherData = null;

      // Process arrivals
      if (toolResults.arrival?.success) {
        predictionData = toolResults.arrival.data;
        isPriceData = false;
        weatherData = toolResults.arrival.data; // Weather data is part of arrival response
      }

      // Process prices
      if (toolResults.price?.success) {
        const d = toolResults.price.data;
        isPriceData = true;

        // Extract sentiment info
        if (Array.isArray(d.variants) && d.variants[0]?.sentiment_info) {
          sentimentInfo = d.variants[0].sentiment_info;
        }

        // Extract detailed forecasts
        if (Array.isArray(d.variants) && d.variants[0]?.forecasts) {
          detailedForecasts = d.variants[0].forecasts;
        }

        let aggregated = [];
        if (Array.isArray(d.variants)) {
          const groupedByDate = {};
          d.variants.forEach((v) => {
            (v.forecasts || []).forEach((f) => {
              const date = f.date;
              const val = f.final_price ?? f.predicted_price ?? 0;
              if (!groupedByDate[date]) groupedByDate[date] = [];
              if (val > 0) groupedByDate[date].push(val);
            });
          });
          aggregated = Object.entries(groupedByDate).map(([date, prices]) => ({
            date,
            total_predicted_value: Math.round(prices.reduce((a, b) => a + b, 0) / prices.length)
          }));
        }

        if (aggregated.length > 0) {
          predictionData = {
            metric_name: "Price (₹/Quintal)",
            total_predicted: aggregated,
            commodity: d.commodity,
            district: d.market || d.district
          };
        }
      }

      setMessages((prev) => {
        const newMessages = prev.slice(0, -1);
        return [
          ...newMessages,
          {
            role: "assistant",
            content: aiResponse,
            predictionData,
            isPriceData,
            aiInsights,
            sentimentInfo,
            detailedForecasts,
            weatherData,
            isSingleDay: context?.isSingleDay || false,
            queryType
          }
        ];
      });
    } catch (e) {
      const errorMsg = e.response?.data?.detail || e.message;
      setMessages((prev) => {
        const newMessages = prev.slice(0, -1);
        return [...newMessages, { role: "assistant", content: `❌ Error: ${errorMsg}` }];
      });
    } finally {
      setLoading(false);
    }
  }, [sessionId, pendingClarification]);

  // Helper function to send a message (for both manual and auto-send)
  const sendMessage = useCallback(async (messageText) => {
    if (!messageText.trim() || loading) return;

    const userMessage = messageText;
    const lower = userMessage.toLowerCase();
    const checkSingleDay = isSingleDayQuery(userMessage);

    // NEW: Set original days based on query
    const days = checkSingleDay ? 1 : 7;
    setOriginalDays(days);

    setMessages((p) => [...p, { role: "user", content: userMessage }]);
    setQuery("");
    setLoading(true);

    // ✅ FRONTEND DETECTION: Check if arrival query without explicit metric
    const isPriceQuery = /\b(price|cost|rate|₹)\b/i.test(lower);
    const isCapabilityQuestion = /\b(can you|do you|are you able|is it possible|does this)\b/i.test(lower);
    const isArrivalQuery = /arrivals?|forecast|expect/i.test(lower);  // Removed "predict" - let agent decide
    const hasExplicitMetric = /\b(bags?|lots?|quintals?|quantity|weight|farmers?|revenue)\b/i.test(lower);

    // Only show metric selection for actual arrival requests (not price queries or capability questions)
    if (isArrivalQuery && !hasExplicitMetric && !isPriceQuery && !isCapabilityQuestion) {
      // Show interactive metric selection immediately
      setMessages((p) => [
        ...p,
        {
          role: "assistant",
          content: "📊 What would you like to forecast?",
          interactiveOptions: {
            question: "Select the metric:",
            options: [
              { label: "📦 Total Bags", value: "total_bags", type: "metric" },
              { label: "🔢 Arrivals Count", value: "number_of_arrivals", type: "metric" },
              { label: "📊 Number of Lots", value: "number_of_lots", type: "metric" },
              { label: "⚖️ Quantity (Quintals)", value: "total_weight", type: "metric" },
              { label: "👥 Number of Farmers", value: "number_of_farmers", type: "metric" }
            ]
          }
        }
      ]);
      setPendingClarification({ originalQuery: userMessage, isSingleDay: checkSingleDay, type: "metric" });
      setLoading(false);
      return;
    }

    setMessages((p) => [...p, { role: "assistant", content: "🧠 AI analyzing with market sentiment...", isTyping: true }]);

    try {
      // 🔧 FIX: Pass advisory flag through context for variety selections
      const isAdvisoryContext = pendingClarification?.isAdvisoryQuery || false;
      const requestBody = {
        query: userMessage,
        session_id: sessionId
      };

      // If this is a variety selection for an advisory query, pass the flag to backend
      if (isAdvisoryContext) {
        requestBody.context = { is_advisory_query: true };
        console.log("🎯 Passing advisory flag to backend via context");
      }

      const res = await axios.post(
        `${process.env.REACT_APP_BACKEND_URL}/ai/chat`,
        requestBody,
        { timeout: 180000 }
      );

      // Handle variety selection (works for both price and arrival tools)
      if (res.data.has_varieties && res.data.varieties?.variants?.length) {
        const aiResponse = res.data.response || "Please select a variety:";
        const variants = res.data.varieties.variants || [];

        const interactiveOptions = {
          question: aiResponse,
          options: variants.map((v) => ({
            label: v,
            value: v,
            type: "variety"
          }))
        };

        // Store the original query context for variety selection
        // Also detect if this is an advisory/market trends query
        const isAdvisory = /should|advice|recommend|bring|decision|what to do|whether|market trends/i.test(userMessage);

        setPendingClarification({
          originalQuery: userMessage,
          isSingleDay: checkSingleDay,
          isVarietySelection: true,
          isAdvisoryQuery: isAdvisory
        });

        setMessages((prev) => {
          const newMessages = prev.slice(0, -1);
          return [
            ...newMessages,
            {
              role: "assistant",
              content: aiResponse.replace(/\n/g, "<br/>"),
              interactiveOptions,
              isSingleDay: checkSingleDay,
              originalQuery: userMessage // Store original query
            }
          ];
        });
        setLoading(false);
        return;
      }

      // Handle clarification requests (arrival tool - metric/commodity selection)
      if (res.data.query_type === "CLARIFICATION") {
        const clarificationMsg = res.data.response || "";

        // Parse response to extract options (backend sends formatted text with numbered options)
        const lines = clarificationMsg.split('\n');
        const options = [];

        lines.forEach((line) => {
          const match = line.match(/^(\d+)\.\s*(.+)$/);
          if (match) {
            const [, num, label] = match;
            options.push({
              label: label.trim(),
              value: label.trim(),
              type: "clarification"
            });
          }
        });

        const interactiveOptions = options.length > 0 ? {
          question: lines[0], // First line is usually the question
          options
        } : null;

        setMessages((prev) => {
          const newMessages = prev.slice(0, -1);
          return [
            ...newMessages,
            {
              role: "assistant",
              content: clarificationMsg.replace(/\n/g, "<br/>"),
              interactiveOptions,
              isSingleDay: checkSingleDay,
              originalQuery: pendingClarification?.originalQuery || userMessage // Store original query
            }
          ];
        });
        setLoading(false);
        return;
      }

      const aiResponse = res.data.response || "Analysis complete.";
      const toolResults = res.data.tool_results || {};
      const aiInsights = res.data.ai_insights;
      const queryType = res.data.query_type || "prediction";

      let predictionData = null;
      let isPriceData = false;
      let sentimentInfo = null;
      let detailedForecasts = [];
      let weatherData = null;

      // 🔧 FIX: Use backend's query_type flag to determine if this is advisory-only
      // Advisory queries should NOT show detailed prediction charts - only text summary + recommendations
      const isAdvisoryOnly = res.data.query_type === "advisory_only";

      if (isAdvisoryOnly) {
        console.log("🎯 ADVISORY-ONLY QUERY - Skipping all detailed chart rendering");
      }

      // Process arrivals (but skip charts if advisory-only query)
      if (toolResults.arrival?.success && !isAdvisoryOnly) {
        predictionData = toolResults.arrival.data;
        isPriceData = false;
        weatherData = toolResults.arrival.data; // Weather data is part of arrival response
      }

      // Process prices (but skip charts if advisory-only query)
      if (toolResults.price?.success && !isAdvisoryOnly) {
        const d = toolResults.price.data;
        isPriceData = true;

        // Extract sentiment info
        if (Array.isArray(d.variants) && d.variants[0]?.sentiment_info) {
          sentimentInfo = d.variants[0].sentiment_info;
        }

        // Extract detailed forecasts
        if (Array.isArray(d.variants) && d.variants[0]?.forecasts) {
          detailedForecasts = d.variants[0].forecasts;
        }

        let aggregated = [];
        if (Array.isArray(d.variants)) {
          const groupedByDate = {};
          d.variants.forEach((v) => {
            (v.forecasts || []).forEach((f) => {
              const date = f.date;
              const val = f.final_price ?? f.predicted_price ?? 0;
              if (!groupedByDate[date]) groupedByDate[date] = [];
              if (val > 0) groupedByDate[date].push(val);
            });
          });
          aggregated = Object.entries(groupedByDate).map(([date, prices]) => ({
            date,
            total_predicted_value: Math.round(prices.reduce((a, b) => a + b, 0) / prices.length)
          }));
        } else {
          const raw = d.predictions || d.predicted_prices || [];
          raw.forEach((p) => {
            const val = p.final_price ?? p.predicted_price ?? 0;
            aggregated.push({ date: p.date, total_predicted_value: Math.round(val) });
          });
        }

        if (aggregated.length > 0) {
          predictionData = {
            metric_name: "Price (₹/Quintal)",
            total_predicted: aggregated,
            commodity: d.commodity,
            district: d.market || d.district
          };
        }
      }

      setMessages((prev) => {
        const newMessages = prev.slice(0, -1);
        return [
          ...newMessages,
          {
            role: "assistant",
            content: aiResponse,
            predictionData,
            isPriceData,
            aiInsights,
            sentimentInfo,
            detailedForecasts,
            weatherData,
            isSingleDay: checkSingleDay,
            originalQuery: pendingClarification?.originalQuery || messageText, // Store original query for today/tomorrow detection
            queryType
          }
        ];
      });

      // Clear advisory flag after response is added
      if (pendingClarification?.isAdvisoryQuery && !pendingClarification.originalQuery) {
        setPendingClarification(null);
      }
    } catch (e) {
      const errorMsg = e.response?.data?.detail || e.message;
      setMessages((prev) => {
        const newMessages = prev.slice(0, -1);
        return [...newMessages, { role: "assistant", content: `❌ Error: ${errorMsg}` }];
      });
    } finally {
      setLoading(false);
    }
  }, [loading, sessionId]);

  // Wrapper for handleSend that uses current query state
  const handleSend = useCallback(() => {
    sendMessage(query);
  }, [query, sendMessage]);

  const handleComposerKeyDown = useCallback(
    (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend]
  );

  return (
    <div className="flex h-screen bg-white overflow-hidden">
      {/* Main Chat Area - Full Width */}
      <div className="flex-1 flex flex-col bg-white w-full">
        {/* Top Bar */}
        <div className="flex items-center justify-between px-8 py-3 bg-white border-b border-gray-200">
          {/* Title */}
          <div className="flex items-center gap-3">
            <span className="text-2xl">🌾</span>
            <h1 className="text-gray-900 font-semibold text-lg m-0">CropCast AI</h1>
          </div>

          {/* Right side controls */}
          <div className="flex items-center gap-3">
            {/* Language Selector */}
            <div className="relative group">
              <button className="flex items-center gap-2 px-3 py-1.5 hover:bg-gray-100 rounded-md text-gray-700 text-sm font-medium transition-all">
                <span>🌐</span>
                <span>{language}</span>
                <span className="text-xs">▼</span>
              </button>

              {/* Dropdown */}
              <div className="absolute right-0 mt-2 w-40 bg-white border border-gray-200 rounded-lg shadow-lg opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-50">
                <button
                  onClick={() => handleLanguageChange("English")}
                  className={`w-full text-left px-4 py-2 hover:bg-gray-50 first:rounded-t-lg transition-all text-sm ${language === "English" ? "bg-gray-100 font-semibold" : ""
                    }`}
                >
                  English
                </button>
                <button
                  onClick={() => handleLanguageChange("తెలుగు")}
                  className={`w-full text-left px-4 py-2 hover:bg-gray-50 transition-all text-sm ${language === "తెలుగు" ? "bg-gray-100 font-semibold" : ""
                    }`}
                >
                  తెలుగు (Telugu)
                </button>
                <button
                  onClick={() => handleLanguageChange("हिंदी")}
                  className={`w-full text-left px-4 py-2 hover:bg-gray-50 last:rounded-b-lg transition-all text-sm ${language === "हिंदी" ? "bg-gray-100 font-semibold" : ""
                    }`}
                >
                  हिंदी (Hindi)
                </button>
              </div>
            </div>

            {/* User Icon */}
            <button className="w-8 h-8 flex items-center justify-center hover:bg-gray-100 rounded-md transition-all">
              <svg className="w-5 h-5 text-gray-700" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" clipRule="evenodd" />
              </svg>
            </button>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto flex flex-col">
          {messages.length <= 1 ? (
            /* Welcome Screen - Show when no messages */
            <div className="flex-1 flex flex-col items-center justify-center px-4">
              <h1 className="text-4xl font-semibold text-gray-800 mb-12">What can I help with?</h1>

              {/* Suggestion Cards */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-w-3xl w-full">
                <button
                  onClick={() => sendMessage("What will be cotton price in Warangal tomorrow")}
                  disabled={loading}
                  className="p-4 bg-white border border-gray-200 rounded-2xl hover:bg-gray-50 hover:border-gray-300 transition-all text-left group disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <div className="flex items-start gap-3">
                    <span className="text-2xl">💰</span>
                    <div>
                      <div className="font-medium text-gray-800 mb-1">Price Forecast</div>
                      <div className="text-sm text-gray-600">What will be cotton price in Warangal tomorrow</div>
                    </div>
                  </div>
                </button>

                <button
                  onClick={() => sendMessage("Expected arrivals in Khammam for next 7 days")}
                  disabled={loading}
                  className="p-4 bg-white border border-gray-200 rounded-2xl hover:bg-gray-50 hover:border-gray-300 transition-all text-left group disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <div className="flex items-start gap-3">
                    <span className="text-2xl">📊</span>
                    <div>
                      <div className="font-medium text-gray-800 mb-1">Arrival Predictions</div>
                      <div className="text-sm text-gray-600">Expected arrivals in Khammam for next 7 days</div>
                    </div>
                  </div>
                </button>

                <button
                  onClick={() => sendMessage("Should I bring cotton to market today in Warangal")}
                  disabled={loading}
                  className="p-4 bg-white border border-gray-200 rounded-2xl hover:bg-gray-50 hover:border-gray-300 transition-all text-left group disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <div className="flex items-start gap-3">
                    <span className="text-2xl">🌤️</span>
                    <div>
                      <div className="font-medium text-gray-800 mb-1">Market Advisory</div>
                      <div className="text-sm text-gray-600">Should I bring cotton to market today in Warangal</div>
                    </div>
                  </div>
                </button>

                <button
                  onClick={() => sendMessage("Chilli prices in Warangal tomorrow")}
                  disabled={loading}
                  className="p-4 bg-white border border-gray-200 rounded-2xl hover:bg-gray-50 hover:border-gray-300 transition-all text-left group disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <div className="flex items-start gap-3">
                    <span className="text-2xl">💰</span>
                    <div>
                      <div className="font-medium text-gray-800 mb-1">Price Forecast</div>
                      <div className="text-sm text-gray-600">Chilli prices in Warangal next week</div>
                    </div>
                  </div>
                </button>
              </div>
            </div>
          ) : (
            /* Chat Messages */
            <div className="max-w-3xl mx-auto px-4 py-6 space-y-0 w-full">
              {messages.slice(1).map((msg, i) => (
                <ChatMessage
                  key={i}
                  message={msg.content}
                  isBot={msg.role === "assistant"}
                  predictionData={msg.predictionData}
                  isPriceData={msg.isPriceData}
                  aiInsights={msg.aiInsights}
                  sentimentInfo={msg.sentimentInfo}
                  detailedForecasts={msg.detailedForecasts}
                  weatherData={msg.weatherData}
                  interactiveOptions={msg.interactiveOptions}
                  isTyping={msg.isTyping}
                  onOptionSelect={(value, type) => handleOptionSelect(value, type, { isSingleDay: msg.isSingleDay })}
                  isSingleDay={msg.isSingleDay || false}
                  userQuery={msg.originalQuery || ""}
                  queryType={msg.queryType || "prediction"}
                  sessionId={sessionId}
                />
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input Section */}
        <div className="pb-11 px-3">
          <div className="max-w-3xl mx-auto">
            <div className="bg-white rounded-3xl border border-gray-300 shadow-lg overflow-hidden">
              <div className="px-5 pt-4">
                <textarea
                  ref={composerRef}
                  rows={1}
                  className="w-full border-none resize-none text-base outline-none bg-transparent placeholder-gray-500 text-gray-900"
                  placeholder="Ask anything"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={handleComposerKeyDown}
                  disabled={loading}
                  style={{ minHeight: '24px', maxHeight: '200px' }}
                />
              </div>
              <div className="flex justify-between items-center px-4 pb-3 pt-2">
                <div className="flex gap-2 items-center">
                  <button
                    type="button"
                    className={`p-2 text-sm rounded-lg transition-all flex items-center gap-2 ${isListening
                      ? 'bg-red-100 text-red-600 listening-pulse'
                      : 'text-gray-600 hover:bg-gray-100'
                      }`}
                    onClick={startVoiceInput}
                    title={isListening ? "Listening..." : "Click to speak"}
                    disabled={isListening}
                  >
                    {isListening ? (
                      <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z" />
                        <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z" />
                      </svg>
                    ) : (
                      <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z" />
                        <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z" />
                      </svg>
                    )}
                    {isListening && (
                      <span className="text-xs font-semibold animate-pulse">Listening...</span>
                    )}
                  </button>
                </div>
                <button
                  type="button"
                  className="w-8 h-8 rounded-full bg-black hover:bg-gray-800 text-white font-bold transition-all disabled:opacity-30 disabled:cursor-not-allowed flex items-center justify-center"
                  onClick={handleSend}
                  disabled={loading || !query.trim()}
                >
                  {loading ? "..." : "↑"}
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
      {/* Footer: Version Number */}
      <div className="fixed bottom-0 left-0 w-full text-center py-3 text-gray-500 text-xs bg-white border-t shadow-sm">
        CropCast AI • Version 1.0
      </div>
    </div>
  );
}

export default App;