import pandas as pd
import io
from typing import Tuple, Dict, Any

def load_and_profile_csv(uploaded_file) -> Tuple[pd.DataFrame, str, Dict[str, Any]]:
    """
    Loads a CSV file from a Streamlit UploadedFile object and generates a detailed profile.
    
    Args:
        uploaded_file: The file object from st.file_uploader

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The loaded DataFrame
            - str: A comprehensive text profile of the data
            - Dict: A dictionary of metadata about the data (e.g., column names, dtypes)

    Raises:
        ValueError: If the file is not a CSV or is empty.
        Exception: For any other pandas read error.
    """
    # Validate file type
    if uploaded_file is None:
        raise ValueError("No file uploaded.")
    if not uploaded_file.name.endswith('.csv'):
        raise ValueError("Please upload a CSV file.")
    
    # Read the file into a Pandas DataFrame
    # Using io.StringIO to convert the uploaded byte stream to a text stream for pandas
    try:
        string_io = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
        df = pd.read_csv(string_io)
    except Exception as e:
        raise Exception(f"Error reading the CSV file: {e}")
    
    # Check if DataFrame is empty
    if df.empty:
        raise ValueError("The uploaded CSV file is empty.")
    
    # Generate a comprehensive text profile
    profile = generate_data_profile(df, uploaded_file.name)
    
    # Generate metadata (will be useful later for the Vector DB)
    metadata = {
        "file_name": uploaded_file.name,
        "num_rows": df.shape[0],
        "num_columns": df.shape[1],
        "column_names": df.columns.tolist(),
        "column_dtypes": df.dtypes.astype(str).to_dict()
    }
    
    return df, profile, metadata

def generate_data_profile(df: pd.DataFrame, file_name: str) -> str:
    """
    Generates a detailed English text summary of the DataFrame's structure and content.
    This text will be the core context for the LLM to understand the data.
    """
    buffer = io.StringIO()
    
    # 1. Basic Info
    buffer.write(f"# PROFILING REPORT FOR: {file_name}\n\n")
    buffer.write(f"This dataset has **{df.shape[0]} rows** and **{df.shape[1]} columns**.\n\n")
    
    # 2. Column Names and Data Types
    buffer.write("## COLUMN SUMMARY\n")
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        buffer.write(f"{i}. `{col}` : *{dtype}*\n")
    buffer.write("\n")
    
    # 3. Preview of the Data (First 3 rows)
    buffer.write("## DATA PREVIEW (First 3 rows)\n")
    # Convert the preview to a markdown-style string for better readability
    preview_str = df.head(3).to_markdown(index=False)
    buffer.write(f"{preview_str}\n\n")
    
    # 4. Basic Statistics for Numeric Columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if not numeric_cols.empty:
        buffer.write("## BASIC STATISTICS (Numeric Columns)\n")
        stats = df[numeric_cols].describe().round(2)
        stats_str = stats.to_markdown()
        buffer.write(f"{stats_str}\n\n")
    else:
        buffer.write("## BASIC STATISTICS\n*No numeric columns found for statistical analysis.*\n\n")
    
    # 5. Check for Missing Values
    buffer.write("## MISSING VALUES\n")
    missing_values = df.isnull().sum()
    if missing_values.any():
        for col, count in missing_values.items():
            if count > 0:
                buffer.write(f"- Column `{col}` has **{count}** missing values.\n")
    else:
        buffer.write("*No missing values found in any column.*\n")
    
    # Return the entire generated profile as a single string
    return buffer.getvalue()