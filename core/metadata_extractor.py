"""
Metadata Extractor — extracts rich structural and statistical metadata
from a pandas DataFrame to feed into agent prompts.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np
import pandas as pd


def extract_metadata(df: pd.DataFrame, query: str, dataset_path: str = "") -> Dict[str, Any]:
    """
    Extract comprehensive metadata from a DataFrame.

    Returns a structured dict that agents use to understand
    the dataset before generating analysis plans or code.
    """
    metadata: Dict[str, Any] = {
        "dataset_path": dataset_path,
        "filename": os.path.basename(dataset_path) if dataset_path else "unknown",
        "query": query,
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "column_names": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "missing_values": {},
        "columns": [],
        "numeric_summary": {},
        "categorical_summary": {},
        "sample_data": df.head(5).to_dict(orient="records"),
        "is_large_dataset": df.shape[0] > 10_000,
        "sample_size": min(df.shape[0], 5_000),
        "has_datetime_columns": False,
        "inferred_domain": _infer_domain(df, query),
    }

    datetime_cols: list = []

    for col in df.columns:
        missing_count = int(df[col].isnull().sum())
        missing_pct = round(missing_count / len(df) * 100, 2) if len(df) > 0 else 0.0
        unique_count = int(df[col].nunique())

        col_info: Dict[str, Any] = {
            "name": col,
            "dtype": str(df[col].dtype),
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "unique_values": unique_count,
            "cardinality": "high" if unique_count > 50 else "medium" if unique_count > 10 else "low",
        }

        metadata["missing_values"][col] = missing_count

        # Numeric columns
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            series = df[col].dropna()
            col_info["analysis_type"] = "numeric"
            if len(series) > 0:
                q1 = float(series.quantile(0.25))
                q3 = float(series.quantile(0.75))
                iqr = q3 - q1
                metadata["numeric_summary"][col] = {
                    "mean": _safe_round(series.mean()),
                    "std": _safe_round(series.std()),
                    "min": _safe_round(series.min()),
                    "max": _safe_round(series.max()),
                    "median": _safe_round(series.median()),
                    "q1": _safe_round(q1),
                    "q3": _safe_round(q3),
                    "iqr": _safe_round(iqr),
                    "skewness": _safe_round(series.skew()),
                    "outlier_count": int(((series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))).sum()),
                }

        # Boolean columns
        elif pd.api.types.is_bool_dtype(df[col]):
            col_info["analysis_type"] = "boolean"
            vc = df[col].value_counts().to_dict()
            metadata["categorical_summary"][col] = {
                "top_values": {str(k): int(v) for k, v in list(vc.items())[:5]}
            }

        # Datetime columns
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_info["analysis_type"] = "datetime"
            metadata["has_datetime_columns"] = True
            datetime_cols.append(col)
            non_null = df[col].dropna()
            if len(non_null) > 0:
                metadata["numeric_summary"][col] = {
                    "min": str(non_null.min()),
                    "max": str(non_null.max()),
                    "range_days": (non_null.max() - non_null.min()).days,
                }

        # Categorical / object columns — try datetime parse first
        else:
            col_info["analysis_type"] = "categorical"
            # Attempt datetime inference on string columns
            if df[col].dtype == object and unique_count < df.shape[0] * 0.9:
                sample_vals = df[col].dropna().head(20)
                try:
                    pd.to_datetime(sample_vals, infer_datetime_format=True)
                    col_info["analysis_type"] = "datetime_string"
                    metadata["has_datetime_columns"] = True
                    datetime_cols.append(col)
                except Exception:
                    pass

            top_vals = df[col].value_counts().head(10).to_dict()
            metadata["categorical_summary"][col] = {
                "top_values": {str(k): int(v) for k, v in top_vals.items()}
            }

        metadata["columns"].append(col_info)

    if datetime_cols:
        metadata["datetime_columns"] = datetime_cols

    # Correlation hints (numeric only)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        try:
            corr = df[numeric_cols].corr().abs()
            # Find top 5 correlated pairs (excluding self)
            corr_pairs = []
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    c1, c2 = numeric_cols[i], numeric_cols[j]
                    val = corr.loc[c1, c2]
                    if not np.isnan(val):
                        corr_pairs.append((c1, c2, round(float(val), 4)))
            corr_pairs.sort(key=lambda x: x[2], reverse=True)
            metadata["top_correlations"] = [
                {"col1": p[0], "col2": p[1], "correlation": p[2]}
                for p in corr_pairs[:5]
            ]
        except Exception:
            metadata["top_correlations"] = []

    return metadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_round(value: Any, decimals: int = 4) -> Any:
    try:
        return round(float(value), decimals)
    except (TypeError, ValueError):
        return None


def _infer_domain(df: pd.DataFrame, query: str) -> str:
    """Heuristically guess the business domain from column names and query."""
    cols_lower = " ".join(df.columns.str.lower())
    query_lower = query.lower()
    combined = cols_lower + " " + query_lower

    domain_signals = {
        "e-commerce / sales": ["sale", "revenue", "order", "customer", "product", "price", "cart", "purchase"],
        "finance": ["stock", "price", "return", "portfolio", "dividend", "market", "trade", "equity"],
        "healthcare": ["patient", "diagnosis", "treatment", "disease", "health", "hospital", "drug", "symptom"],
        "marketing": ["campaign", "click", "impression", "conversion", "channel", "ad", "email", "ctr"],
        "hr / workforce": ["employee", "salary", "department", "hire", "attrition", "tenure", "headcount"],
        "logistics": ["shipment", "delivery", "warehouse", "route", "freight", "tracking", "supply"],
        "web / product analytics": ["session", "user", "event", "page", "funnel", "engagement", "retention"],
    }

    best_domain = "general"
    best_score = 0
    for domain, signals in domain_signals.items():
        score = sum(1 for s in signals if s in combined)
        if score > best_score:
            best_score = score
            best_domain = domain

    return best_domain
