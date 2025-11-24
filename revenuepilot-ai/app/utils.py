def normalize_columns(df):
    df.columns = df.columns.astype(str).str.strip().str.lower().str.replace(" ", "_")
    return df


def find_contact_column(df):
    candidates = ["contact_id", "id", "user_id", "contactid"]
    for col in candidates:
        if col in df.columns:
            return col
    raise Exception(f"No contact identifier column found: {df.columns.tolist()}")
