import os
import re
import json
import pandas as pd
from datetime import datetime
from google import genai
from google.genai import types

def get_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)

def clean_timestamp_rule_based(val):
    """Deterministic parser for known formats."""
    if pd.isna(val) or val is None:
        return None
    val = str(val).strip()

    formats = [
        "%Y-%m-%d",
        "%m/%d/%y",
        "%m/%d/%Y",
        "%d-%m-%Y",
        "%Y.%m.%d"
    ]
    for fmt in formats:
        try:
            return datetime.strptime(val, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    sanitized = re.sub(r"/+", "/", val)
    for fmt in ["%m/%d/%y", "%m/%d/%Y"]:
        try:
            return datetime.strptime(sanitized, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass

    return "NEEDS_GEMINI"

def batch_resolve_dates(unresolved_dates: list, client) -> dict:
    """Send fat-fingered/irregular dates to Gemini Flash in a single prompt."""
    if not unresolved_dates or client is None:
        return {}

    prompt = f"""
    Convert the following messy/fat-fingered date strings into standard ISO format (YYYY-MM-DD).
    Return ONLY a JSON dictionary where keys are the input strings and values are the formatted dates (or null if unrecognizable).
    
    Inputs:
    {unresolved_dates}
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        return json.loads(response.text)
    except Exception:
        return {}

def run_pipeline(input_csv_bytes: bytes) -> bytes:
    """Processes failure logs, normalizes dates, handles nulls, and exports Parquet."""
    import io
    df = pd.read_csv(io.BytesIO(input_csv_bytes))
    client = get_gemini_client()

    # 1. Date normalization
    for col in ["event_timestamp", "resolved_timestamp"]:
        parsed = df[col].apply(clean_timestamp_rule_based)
        unresolved = df.loc[parsed == "NEEDS_GEMINI", col].dropna().unique().tolist()
        
        if unresolved and client:
            corrections = batch_resolve_dates(unresolved, client)
            mask = parsed == "NEEDS_GEMINI"
            parsed.loc[mask] = df.loc[mask, col].map(corrections).fillna("NEEDS_GEMINI")
        
        # Replace residual failure flags with null
        df[col] = parsed.replace("NEEDS_GEMINI", None)

    # 2. Impute sensor values and null metrics
    if "operating_temp_c" in df.columns:
        df["operating_temp_c"] = df.groupby("device_type")["operating_temp_c"].transform(
            lambda s: s.fillna(s.median())
        )

    df["assigned_tech"] = df["assigned_tech"].fillna("unassigned")
    df["downtime_minutes"] = df["downtime_minutes"].fillna(0.0)

    # 3. Track recurring hardware failures by Asset ID
    fail_counts = df["asset_id"].value_counts()
    df["asset_failure_frequency"] = df["asset_id"].map(fail_counts).fillna(1).astype(int)

    # Convert to Parquet buffer
    out_buffer = io.BytesIO()
    df.to_parquet(out_buffer, index=False)
    return out_buffer.getvalue()

def generate_summary_metrics(df: pd.DataFrame) -> str:
    """Calculates high-level metrics for dashboard consumption."""
    summary = {
        "metadata": {
            "total_records_processed": int(len(df)),
            "unique_assets": int(df["asset_id"].nunique()),
            "last_pipeline_run": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        },
        "severity_distribution": df["severity"].value_counts(dropna=False).to_dict(),
        "failures_by_device": df["device_type"].value_counts().head(8).to_dict(),
        "top_failing_assets": (
            df.groupby("asset_id")
            .size()
            .sort_values(ascending=False)
            .head(5)
            .to_dict()
        ),
        "avg_downtime_minutes": round(float(df["downtime_minutes"].mean()), 1)
    }
    return json.dumps(summary, indent=2)
