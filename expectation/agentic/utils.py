import pandas as pd
from typing import Dict
import matplotlib.pyplot as plt
import io
import base64

def get_dataframe_summary(data: Dict[str, pd.DataFrame]) -> dict:

    summary = {}
    
    for name, df in data.items():
        num_rows, num_cols = df.shape
        
        column_types = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                column_types[col] = "numeric"
            elif pd.api.types.is_bool_dtype(df[col]):
                column_types[col] = "boolean"
            elif pd.api.types.is_categorical_dtype(df[col]) or df[col].nunique() <= 10:
                column_types[col] = "categorical"
            else:
                column_types[col] = "text"
        
        numeric_stats = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                stats = df[col].describe().to_dict()
                numeric_stats[col] = stats
        
        summary[name] = {
            "rows": num_rows,
            "columns": num_cols,
            "column_types": column_types,
            "numeric_stats": numeric_stats,
            "sample": df.head(5).to_dict(orient="records")
        }
    
    return summary

def generate_histogram(data: pd.Series) -> str:

    plt.figure(figsize=(8, 6))
    plt.hist(data.dropna(), bins=30, alpha=0.7)
    plt.title(f"Distribution of {data.name}")
    plt.xlabel(data.name)
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    plt.close()
    
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode("utf-8")
    
    return f"data:image/png;base64,{img_str}"

def generate_scatter_plot(x: pd.Series, y: pd.Series) -> str:

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.7)
    plt.title(f"Scatter plot of {x.name} vs {y.name}")
    plt.xlabel(x.name)
    plt.ylabel(y.name)
    plt.grid(True, alpha=0.3)
    
    correlation = x.corr(y)
    plt.annotate(f"Correlation: {correlation:.4f}", 
                xy=(0.05, 0.95), 
                xycoords="axes fraction",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    plt.close()
    
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode("utf-8")
    
    return f"data:image/png;base64,{img_str}"