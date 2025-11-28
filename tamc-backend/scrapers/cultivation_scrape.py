"""
Cultivation Data Integration Module for Agricultural Forecasting
Fetches and processes UPAG Telangana cultivation data to enhance arrival predictions
"""

import json
import requests
import pandas as pd
import logging
from typing import Dict, Optional
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO)

# ---------- CORE SERVICE ------------------------------------------------------
class CultivationDataService:
    def __init__(self, cache_dir: str = "./cultivation_cache"):
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, "telangana_cultivation_data.csv")
        os.makedirs(cache_dir, exist_ok=True)
        self.api_url = "https://dash.upag.gov.in/_dash-update-component"

    # -------------------------------------------------------------------------
    # 1.  FETCH DATA (with 24-h cache)
    # -------------------------------------------------------------------------
    def fetch_telangana_cultivation_data(self, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        if not force_refresh and os.path.exists(self.cache_file):
            age_h = (datetime.now().timestamp() - os.path.getmtime(self.cache_file)) / 3600
            if age_h < 24:
                logging.info("✅ Using cached cultivation data")
                try:
                    return pd.read_csv(self.cache_file)
                except Exception as e:
                    logging.warning(f"Failed to read cache: {e}")

        logging.info("🌾 Fetching UPAG cultivation data (Telangana only)...")
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "origin": "https://dash.upag.gov.in",
            "referer": "https://dash.upag.gov.in/statewiseapy?t=&stateID=0&rtab=Area%2C%20Production%20%26%20Yield&rtype=reports",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        }
        payload = {
            "output": (
                "..swapy-store.data...swapy-suffix-title.children..."
                "..swapy-notification1.children...swapy-notification2.children..."
                "..swapy-notification4.children...swapy-sheetname.children.."
            ),
            "outputs": [
                {"id": "swapy-store", "property": "data"},
                {"id": "swapy-suffix-title", "property": "children"},
                {"id": "swapy-notification1", "property": "children"},
                {"id": "swapy-notification2", "property": "children"},
                {"id": "swapy-notification4", "property": "children"},
                {"id": "swapy-sheetname", "property": "children"},
            ],
            "inputs": [
                {
                    "id": "swapy-filters-store",
                    "property": "data",
                    "value": {
                        "fromyear": "",
                        "toyear": "",
                        "crop": ["All"],
                        "metric": ["Area", "Production", "Yield"],
                        "uom": "Lakh",
                    },
                },
            ],
            "changedPropIds": ["swapy-filters-store.data"],
            "state": [
                {
                    "id": "url",
                    "property": "search",
                    "value": "?t=&stateID=0&rtab=Area%2C%20Production%20%26%20Yield&rtype=reports",
                },
            ],
        }

        try:
            resp = requests.post(self.api_url, headers=headers, json=payload, timeout=3)
            logging.info(f"🔍 Status Code: {resp.status_code}")
            if resp.status_code != 200:
                logging.error("⚠️ Failed to fetch cultivation data")
                return None
            data = resp.json().get("response", {}).get("swapy-store", {}).get("data", [])
            if not data:
                logging.warning("⚠️ No data found in response")
                return None
            df = pd.DataFrame(data)
            df_tel = df[df["State"].str.lower() == "telangana"].copy()
            if df_tel.empty:
                logging.warning("⚠️ No Telangana data found")
                return None
            df_tel.to_csv(self.cache_file, index=False)
            logging.info("✅ Telangana cultivation data saved to cache")
            return df_tel
        except Exception as e:
            logging.error(f"❌ Unexpected error: {e}")
            return None

    # -------------------------------------------------------------------------
    # 2.  CULTIVATION FACTOR
    # -------------------------------------------------------------------------
    def get_cultivation_factor(self, commodity: str, district: Optional[str] = None) -> Dict:
        df = self.fetch_telangana_cultivation_data()
        if df is None or df.empty:
            return {"status": "unavailable", "impact_factor": 1.0, "message": "Cultivation data not available"}

        try:
            # pick exact string column 'Crop'
            crop_col = None
            for c in df.columns:
                if c.strip().lower() == "crop":
                    crop_col = c
                    break
            if crop_col is None:
                return {"status": "error", "impact_factor": 1.0, "message": "Crop column not found"}

            df[crop_col] = df[crop_col].astype(str)
            commodity_lower = commodity.lower().strip()
            crop_match = df[df[crop_col].str.lower().str.contains(commodity_lower, na=False)]
            if crop_match.empty:
                return {"status": "not_found", "impact_factor": 1.0, "message": f"No cultivation data found for {commodity}"}

            if "Latest Estimate" in crop_match.columns:
                crop_match = crop_match[crop_match["Latest Estimate"] == 1]

            # year
            year_col = None
            for c in df.columns:
                if c.strip().lower() == "crop year":
                    year_col = c
                    break
            if year_col:
                latest_year = crop_match[year_col].max()
                latest_data = crop_match[crop_match[year_col] == latest_year]
            else:
                latest_data = crop_match.tail(10)
                latest_year = "Latest"

            # metric / value
            metric_col = value_col = None
            for c in df.columns:
                if c.strip().lower() == "metric":
                    metric_col = c
                elif c.strip().lower() == "value":
                    value_col = c

            production = area = yield_val = 0.0
            if metric_col and value_col:
                production = float(latest_data[latest_data[metric_col] == "Production"][value_col].sum() or 0)
                area = float(latest_data[latest_data[metric_col] == "Area"][value_col].sum() or 0)
                yield_val = float(latest_data[latest_data[metric_col] == "Yield"][value_col].sum() or 0)
            else:
                production = float(latest_data["Value"].sum())

            # impact factor
            impact_factor = 1.0
            if production > 0 and year_col and metric_col and value_col:
                all_prod = df[(df[year_col] == latest_year) & (df[metric_col] == "Production")]
                avg = all_prod[value_col].mean()
                if avg and avg > 0:
                    impact_factor = 0.9 + (production / avg) * 0.2
                    impact_factor = max(0.8, min(1.2, impact_factor))

            return {
                "status": "success",
                "commodity": latest_data[crop_col].iat[0],
                "year": str(latest_year),
                "area": round(area, 2),
                "production": round(production, 2),
                "yield": round(yield_val, 2),
                "impact_factor": round(impact_factor, 3),
                "unit": latest_data["Unit Of Measure"].iat[0] if "Unit Of Measure" in latest_data.columns else "units",
                "message": f"Cultivation factor applied for {commodity}",
            }
        except Exception as e:
            logging.error(f"Error calculating cultivation factor: {e}", exc_info=True)
            return {"status": "error", "impact_factor": 1.0, "message": str(e)}

    # -------------------------------------------------------------------------
    # 3.  SEASONAL TRENDS
    # -------------------------------------------------------------------------
    def get_seasonal_trends(self, commodity: str, years: int = 5) -> Dict:
        df = self.fetch_telangana_cultivation_data()
        if df is None or df.empty:
            return {"status": "unavailable"}

        try:
            crop_col = None
            for c in df.columns:
                if c.strip().lower() == "crop":
                    crop_col = c
                    break
            if crop_col is None:
                return {"status": "error", "message": "Crop column not found"}

            df[crop_col] = df[crop_col].astype(str)
            commodity_lower = commodity.lower().strip()
            crop_data = df[df[crop_col].str.lower().str.contains(commodity_lower, na=False)]
            if crop_data.empty:
                return {"status": "not_found"}

            if "Latest Estimate" in crop_data.columns:
                crop_data = crop_data[crop_data["Latest Estimate"] == 1]

            year_col = None
            for c in df.columns:
                if c.strip().lower() == "crop year":
                    year_col = c
                    break
            if year_col is None:
                return {"status": "success", "commodity": crop_data[crop_col].iat[0],
                        "message": "Historical year data not available"}

            years_list = sorted(crop_data[year_col].unique())
            if len(years_list) < 2:
                return {"status": "insufficient_data"}

            years_to_analyze = years_list[-min(years, len(years_list)):]
            production_trend = []
            value_col = "Value" if "Value" in crop_data.columns else None
            for yr in years_to_analyze:
                yr_data = crop_data[crop_data[year_col] == yr]
                if not yr_data.empty and value_col:
                    total = float(yr_data[value_col].sum())
                    production_trend.append({"year": str(yr), "production": total})

            growth_rate = 0
            if len(production_trend) >= 2:
                first = production_trend[0]["production"]
                last = production_trend[-1]["production"]
                if first:
                    growth_rate = ((last - first) / first) * 100

            return {
                "status": "success",
                "commodity": crop_data[crop_col].iat[0],
                "years_analyzed": len(production_trend),
                "production_trend": production_trend,
                "growth_rate_pct": round(growth_rate, 2),
                "trend_direction": "increasing" if growth_rate > 5 else "decreasing" if growth_rate < -5 else "stable",
            }
        except Exception as e:
            logging.error(f"Error analyzing trends: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}


# ====================== INTEGRATION HELPERS ===================================
def enhance_prediction_with_cultivation(
    prediction_result: Dict, commodity: str, district: Optional[str] = None
) -> Dict:
    service = CultivationDataService()
    cult = service.get_cultivation_factor(commodity, district)
    if cult["status"] != "success":
        prediction_result["cultivation_data"] = cult
        return prediction_result

    factor = cult["impact_factor"]
    if "total_predicted" in prediction_result:
        for p in prediction_result["total_predicted"]:
            p["total_predicted_value"] *= factor
            p["cultivation_adjusted"] = True
    if "commodity_predictions" in prediction_result:
        for forecasts in prediction_result["commodity_predictions"].values():
            for f in forecasts:
                f["predicted_value"] *= factor
                f["cultivation_adjusted"] = True

    prediction_result["cultivation_data"] = cult
    prediction_result["cultivation_applied"] = True
    logging.info(f"✅ Cultivation factor ({factor:.3f}) applied to predictions")
    return prediction_result


def get_cultivation_summary(commodity: str, district: Optional[str] = None) -> str:
    service = CultivationDataService()
    cult = service.get_cultivation_factor(commodity, district)
    if cult["status"] != "success":
        return ""
    factor = cult["impact_factor"]
    trend = (
        "🌾 Higher cultivation area → Expect increased arrivals"
        if factor > 1.05
        else "📉 Lower cultivation area → Expect reduced arrivals"
        if factor < 0.95
        else "📊 Normal cultivation levels"
    )
    return f"\n{trend}\n📍 Cultivation: {cult['production']:.1f} lakh tonnes"


# ====================== MCP-READY FUNCTIONS ===================================
def mcp_inspect_data_structure() -> Dict:
    service = CultivationDataService()
    df = service.fetch_telangana_cultivation_data()
    if df is None:
        return {"status": "error", "message": "Failed to fetch data"}
    return {
        "status": "success",
        "total_records": len(df),
        "columns": df.columns.tolist(),
        "sample_data": df.head(5).to_dict("records"),
        "unique_crops": df["Crop"].unique().tolist()[:20],
        "data_types": {c: str(t) for c, t in df.dtypes.items()},
    }


def mcp_fetch_cultivation_data(force_refresh: bool = False) -> Dict:
    service = CultivationDataService()
    df = service.fetch_telangana_cultivation_data(force_refresh)
    if df is None:
        return {"status": "error", "message": "Failed to fetch cultivation data"}
    return {
        "status": "success",
        "records": len(df),
        "data": df.to_dict("records")[:100],
        "crops": df["Crop"].unique().tolist(),
    }


def mcp_get_cultivation_impact(commodity: str, district: Optional[str] = None) -> Dict:
    return CultivationDataService().get_cultivation_factor(commodity, district)


def mcp_get_cultivation_trends(commodity: str, years: int = 5) -> Dict:
    return CultivationDataService().get_seasonal_trends(commodity, years)


# ====================== CLI SMOKE-TEST ========================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🌾 Cultivation Data Integration Module – Test")
    print("=" * 70 + "\n")

    print("Test 0: Inspecting data structure...")
    info = mcp_inspect_data_structure()
    if info["status"] == "success":
        print(f"Total Records: {info['total_records']}")
        print(f"Columns: {info['columns']}")
        print("First 10 crops:", ", ".join(info["unique_crops"][:10]))

    print("\n" + "-" * 70 + "\n")

    print("Test 1: Fetch cultivation data...")
    data = mcp_fetch_cultivation_data()
    print(f"Status: {data['status']} | Records: {data['records']}")

    print("\n" + "-" * 70 + "\n")

    print("Test 2: Cultivation impact for Rice...")
    impact = mcp_get_cultivation_impact("Rice")
    print(json.dumps(impact, indent=2, default=str))

    print("\n" + "-" * 70 + "\n")

    print("Test 3: Cultivation trends for Rice...")
    trends = mcp_get_cultivation_trends("Rice", 5)
    print(json.dumps(trends, indent=2, default=str))

    print("\n" + "=" * 70 + "\n✅ All tests completed!\n")