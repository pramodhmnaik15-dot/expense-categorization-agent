import pandas as pd
import numpy as np

def detect_anomalies(df):
    """Detect unusual amounts using Robust Z-score (Median & MAD) and double charges."""
    if df.empty:
        df['is_alerted'] = 0
        df['alert_reason'] = ""
        return df

    alerts = []
    reasons = []

    for idx, row in df.iterrows():
        cat = row['category']
        amt = row['amount']
        date = row['date']
        desc = row['description']

        # 1. Double charge check: same merchant & amount on the same day
        same_day_dupes = df[(df['date'] == date) & (df['amount'] == amt) & (df.index != idx) & (df['description'] == desc)]
        if not same_day_dupes.empty:
            alerts.append(1)
            reasons.append("Possible double charge: same merchant and amount on same day.")
            continue

        # 2. Unusual amount check using Robust Z-score within category
        cat_data = df[df['category'] == cat]['amount']
        if len(cat_data) >= 3:
            median = cat_data.median()
            mad = ((cat_data - median).abs()).median()
            if mad > 0:
                z_score = (amt - median) / (1.4826 * mad)
                # Flag when z > 3.5 and amount >= 500
                if z_score > 3.5 and amt >= 500:
                    alerts.append(1)
                    reasons.append(f"Unusual amount: ₹{amt:,.0f} is {z_score:.1f}x your usual {cat} spend.")
                    continue

        alerts.append(0)
        reasons.append("")

    df['is_alerted'] = alerts
    df['alert_reason'] = reasons
    return df

def answer_natural_language_query(query, df):
    """Natural language router to compute exact answers using pandas."""
    query_lower = query.lower()
    
    if df.empty:
        return "No transactions loaded in the database.", None

    # Intent 1: Unusual / Alerts
    if any(k in query_lower for k in ['unusual', 'alert', 'double charge', 'flag']):
        flagged = df[df['is_alerted'] == 1]
        explanation = "Computed using Robust Z-score ($z > 3.5$) and calendar-day duplicate matching."
        return f"I found {len(flagged)} transaction(s) worth checking.", flagged, explanation

    # Intent 2: Spending by Category / Month
    # Extract month if mentioned (e.g., 'august', 'october')
    months = {'january': '01', 'february': '02', 'march': '03', 'april': '04', 'may': '05', 'june': '06',
              'july': '07', 'august': '08', 'september': '09', 'october': '10', 'november': '11', 'december': '12'}
    
    target_month = None
    for m_name, m_num in months.items():
        if m_name in query_lower:
            target_month = m_num
            break

    filtered_df = df.copy()
    if target_month:
        filtered_df['date'] = pd.to_datetime(filtered_df['date'], errors='coerce')
        filtered_df = filtered_df[filtered_df['date'].dt.strftime('%m') == target_month]

    # Category matching
    categories = ['food & dining', 'groceries', 'transport', 'shopping', 'bills & utilities', 'rent', 'entertainment']
    matched_cat = next((cat for cat in categories if cat in query_lower), None)

    if matched_cat:
        filtered_df = filtered_df[filtered_df['category'].str.lower() == matched_cat]
        total = filtered_df['amount'].sum()
        month_str = f" in the specified month" if target_month else ""
        explanation = f"Filtered transactions where category equals '{matched_cat.title()}'{month_str} and computed the sum using Pandas."
        return f"You spent **₹{total:,.0f}** on {matched_cat.title()}{month_str} across {len(filtered_df)} transaction(s).", filtered_df, explanation

    # General total spend fallback
    total = filtered_df['amount'].sum()
    explanation = "Computed total sum of all loaded transaction records using Pandas."
    return f"Total spending across {len(filtered_df)} transactions is **₹{total:,.0f}**.", filtered_df, explanation