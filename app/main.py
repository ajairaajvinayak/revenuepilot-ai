import os
from pathlib import Path
from typing import Tuple

import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import google.generativeai as genai

def fix_quoted_csv(df: pd.DataFrame) -> pd.DataFrame:
    """If the CSV was read as one quoted column, split it into real columns."""
    if len(df.columns) == 1:
        # The entire CSV is one quoted column - split it properly
        rows = []
        for idx, row in df.iterrows():
            # Remove outer quotes and split by comma
            row_str = str(row.iloc[0]).strip('"')
            values = row_str.split(',')
            rows.append(values)
        
        # Create new dataframe
        new_df = pd.DataFrame(rows)
        # First row is the header
        new_df.columns = new_df.iloc[0]
        new_df = new_df.iloc[1:].reset_index(drop=True)
        return new_df
    return df

# ================= GEMINI CONFIG =================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY and os.path.exists("GEMINI_API_KEY.txt"):
    with open("GEMINI_API_KEY.txt") as f:
        GEMINI_API_KEY = f.read().strip()

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def call_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        return "❌ Gemini API key not found. Add GEMINI_API_KEY.txt or set environment variable."

    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text


# ================= CONFIG =================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

st.set_page_config(
    page_title="RevenuePilot AI",
    page_icon="📈",
    layout="wide"
)

# ================= DATA LOADING =================

@st.cache_data
def load_data():
    # Read CSVs with quoting parameter to handle quoted rows
    contacts = pd.read_csv(DATA_DIR / "contacts.csv", quotechar='"', skipinitialspace=True)
    deals = pd.read_csv(DATA_DIR / "deals.csv", quotechar='"', skipinitialspace=True)
    email_stats = pd.read_csv(DATA_DIR / "email_stats.csv", quotechar='"', skipinitialspace=True)
    
    # If still reading as single column, manually parse
    if len(contacts.columns) == 1:
        contacts = fix_quoted_csv(contacts)
    if len(deals.columns) == 1:
        deals = fix_quoted_csv(deals)
    if len(email_stats.columns) == 1:
        email_stats = fix_quoted_csv(email_stats)

    # Clean column headers - remove quotes, whitespace, and normalize
    contacts.columns = contacts.columns.astype(str).str.strip().str.replace('"', '').str.lower().str.replace(' ', '_')
    deals.columns = deals.columns.astype(str).str.strip().str.replace('"', '').str.lower().str.replace(' ', '_')
    email_stats.columns = email_stats.columns.astype(str).str.strip().str.replace('"', '').str.lower().str.replace(' ', '_')

    # Handle duplicate columns by renaming them
    def fix_duplicates(df):
        cols = df.columns.tolist()
        seen = {}
        new_cols = []
        for col in cols:
            if col in seen:
                seen[col] += 1
                new_cols.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                new_cols.append(col)
        df.columns = new_cols
        return df
    
    contacts = fix_duplicates(contacts)
    deals = fix_duplicates(deals)
    email_stats = fix_duplicates(email_stats)

    # Debug: Print column names
    print("Contacts columns:", contacts.columns.tolist())
    print("Deals columns:", deals.columns.tolist())
    print("Email Stats columns:", email_stats.columns.tolist())

    return contacts, deals, email_stats


# ================= CORE LOGIC =================

def build_master_table(contacts, deals, email_stats):
    # Check if contact_id exists in deals
    if "contact_id" not in deals.columns:
        st.error(f"❌ 'contact_id' column not found in deals.csv. Available columns: {deals.columns.tolist()}")
        st.stop()
    
    if "contact_id" not in contacts.columns:
        st.error(f"❌ 'contact_id' column not found in contacts.csv. Available columns: {contacts.columns.tolist()}")
        st.stop()
    
    if "contact_id" not in email_stats.columns:
        st.error(f"❌ 'contact_id' column not found in email_stats.csv. Available columns: {email_stats.columns.tolist()}")
        st.stop()

    # Convert numeric columns to proper types
    deals['deal_amount'] = pd.to_numeric(deals['deal_amount'], errors='coerce')
    deals['contact_id'] = pd.to_numeric(deals['contact_id'], errors='coerce')
    contacts['contact_id'] = pd.to_numeric(contacts['contact_id'], errors='coerce')
    email_stats['contact_id'] = pd.to_numeric(email_stats['contact_id'], errors='coerce')
    email_stats['emails_sent'] = pd.to_numeric(email_stats['emails_sent'], errors='coerce')
    email_stats['opens'] = pd.to_numeric(email_stats['opens'], errors='coerce')
    email_stats['clicks'] = pd.to_numeric(email_stats['clicks'], errors='coerce')

    # Create a binary closed_won column if deal_stage exists
    if 'deal_stage' in deals.columns:
        deals['closed_won'] = deals['deal_stage'].astype(str).str.lower().str.contains('won').astype(int)
    elif 'closed_won' not in deals.columns:
        st.error(f"❌ Neither 'closed_won' nor 'deal_stage' found in deals.csv. Available columns: {deals.columns.tolist()}")
        st.stop()
    else:
        deals['closed_won'] = pd.to_numeric(deals['closed_won'], errors='coerce')

    deals_agg = deals.groupby("contact_id").agg(
        total_revenue=("deal_amount", "sum"),
        deals_won=("closed_won", "sum"),
        deals_count=("deal_id", "count")
    ).reset_index()

    df = contacts.merge(deals_agg, on="contact_id", how="left")
    df = df.merge(email_stats, on="contact_id", how="left")
    df.fillna(0, inplace=True)

    # Ensure numeric types for calculations
    df["total_revenue"] = pd.to_numeric(df["total_revenue"], errors='coerce').fillna(0)
    df["deals_won"] = pd.to_numeric(df["deals_won"], errors='coerce').fillna(0)
    df["emails_sent"] = pd.to_numeric(df["emails_sent"], errors='coerce').fillna(0)
    df["opens"] = pd.to_numeric(df["opens"], errors='coerce').fillna(0)
    df["clicks"] = pd.to_numeric(df["clicks"], errors='coerce').fillna(0)

    df["open_rate"] = df.apply(
        lambda r: r["opens"] / r["emails_sent"] if r["emails_sent"] > 0 else 0, axis=1
    )
    df["click_rate"] = df.apply(
        lambda r: r["clicks"] / r["emails_sent"] if r["emails_sent"] > 0 else 0, axis=1
    )

    return df


def build_segments(master, segment_by="industry"):
    # Ensure the segment_by column exists
    if segment_by not in master.columns:
        st.error(f"❌ Column '{segment_by}' not found. Available columns: {master.columns.tolist()}")
        st.stop()
    
    # Ensure numeric columns are actually numeric
    master['total_revenue'] = pd.to_numeric(master['total_revenue'], errors='coerce').fillna(0)
    master['deals_won'] = pd.to_numeric(master['deals_won'], errors='coerce').fillna(0)
    master['open_rate'] = pd.to_numeric(master['open_rate'], errors='coerce').fillna(0)
    master['click_rate'] = pd.to_numeric(master['click_rate'], errors='coerce').fillna(0)
    
    seg = master.groupby(segment_by).agg(
        contacts=("contact_id", "count"),
        total_revenue=("total_revenue", "sum"),
        avg_revenue=("total_revenue", "mean"),
        deals_won=("deals_won", "sum"),
        avg_open_rate=("open_rate", "mean"),
        avg_click_rate=("click_rate", "mean"),
    ).reset_index()

    scaler = MinMaxScaler()

    for col in ["total_revenue", "avg_revenue", "deals_won", "avg_open_rate", "avg_click_rate"]:
        if len(seg) > 1:
            seg[col + "_norm"] = scaler.fit_transform(seg[[col]])
        else:
            seg[col + "_norm"] = 1

    seg["opportunity_score"] = (
        0.3 * seg["total_revenue_norm"] +
        0.2 * seg["avg_revenue_norm"] +
        0.2 * seg["deals_won_norm"] +
        0.15 * seg["avg_open_rate_norm"] +
        0.15 * seg["avg_click_rate_norm"]
    ) * 100

    return seg.sort_values("opportunity_score", ascending=False)


# ================= AI STRATEGY =================

def generate_strategy_with_llm(segment_summary, goal, budget):
    prompt = f"""
You are a senior B2B SaaS revenue strategist.

Segment Summary:
{segment_summary}

Goal: {goal}
Budget: {budget}

Deliver a profit-focused campaign strategy:
- Priority segments
- Core messaging
- Channel strategy
- Revenue KPIs

Be concise, sharp and business-focused.
"""
    return call_gemini(prompt)


def generate_email_sequence_with_llm(persona_desc, offer_desc):
    prompt = f"""
You are a high-conversion B2B growth copywriter.

Persona: {persona_desc}
Offer: {offer_desc}

Generate a 3-email outreach sequence with:
- 2 subject lines each
- Strong CTA
- Revenue-focused tone
"""
    return call_gemini(prompt)


# ================= FORECAST =================

def estimate_metrics(base_open, base_click, base_conv, channel_factor, offer_strength):
    return {
        "open_rate": min(base_open * channel_factor, 0.9),
        "click_rate": min(base_click * channel_factor, 0.7),
        "conversion_rate": min(base_conv * offer_strength, 0.6)
    }


def estimate_revenue(n_leads, conv_rate, avg_deal):
    return n_leads * conv_rate * avg_deal


# ================= UI =================

def sidebar_controls():
    st.sidebar.header("⚙️ Controls")
    return (
        st.sidebar.selectbox("Segment by", ["industry", "region", "lead_source", "company_size"]),
        st.sidebar.text_input("Business Goal", "Increase qualified pipeline by 25%"),
        st.sidebar.selectbox("Budget", ["Low", "Medium", "High"]),
        st.sidebar.selectbox("Channel", ["Email", "LinkedIn + Email", "Webinar + Email"]),
        st.sidebar.slider("Offer Strength", 0.5, 1.5, 1.0, 0.1)
    )


def main():
    st.title("📈 RevenuePilot AI")
    st.caption("Gemini-Powered Autonomous Revenue Intelligence System")

    segment_by, goal, budget, channel, offer_strength = sidebar_controls()

    # Add debug expander
    with st.expander("🔍 Debug Info - Click to view loaded data columns"):
        contacts, deals, email_stats = load_data()
        st.write("**Contacts columns:**", contacts.columns.tolist())
        st.write("**Deals columns:**", deals.columns.tolist())
        st.write("**Email Stats columns:**", email_stats.columns.tolist())
        st.write("**Contacts preview:**")
        st.dataframe(contacts.head(3))
        st.write("**Deals preview:**")
        st.dataframe(deals.head(3))

    contacts, deals, email_stats = load_data()
    master = build_master_table(contacts, deals, email_stats)
    segments = build_segments(master, segment_by)

    st.subheader("Segment Intelligence")
    st.dataframe(segments.round(2))

    # ===== AI Strategy =====
    st.markdown("## 🚀 AI Strategy")
    segment_summary = segments.head(5).to_markdown(index=False)

    if st.button("Generate Campaign Strategy"):
        with st.spinner("Generating strategy..."):
            strategy = generate_strategy_with_llm(segment_summary, goal, budget)
            st.markdown(strategy)

    # ===== Email Sequence =====
    st.markdown("## ✉️ Outreach Email Sequence")
    persona = st.text_input("Persona", "B2B SaaS Founders")
    offer = st.text_area("Offer", "AI-powered campaign optimisation")

    if st.button("Generate Email Sequence"):
        with st.spinner("Generating email sequence..."):
            email_seq = generate_email_sequence_with_llm(persona, offer)
            st.markdown(email_seq)

    # ===== Revenue Forecast =====
    st.markdown("## 💰 Revenue Forecast")

    top_segment = segments.iloc[0]
    metrics = estimate_metrics(
        top_segment["avg_open_rate"],
        top_segment["avg_click_rate"],
        0.08,
        1.15 if channel != "Email" else 1.0,
        offer_strength
    )

    revenue = estimate_revenue(
        int(top_segment["contacts"]),
        metrics["conversion_rate"],
        top_segment["avg_revenue"]
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Leads", int(top_segment["contacts"]))
    col2.metric("Open Rate", f"{metrics['open_rate']*100:.1f}%")
    col3.metric("Conversion", f"{metrics['conversion_rate']*100:.1f}%")
    col4.metric("Forecast Revenue", f"${revenue:,.0f}")


if __name__ == "__main__":
    main()