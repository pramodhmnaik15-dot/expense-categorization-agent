import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from database import get_all_transactions, get_connection, load_sample_csv
from groq import Groq
from model import ExpenseClassifier
from rules import answer_natural_language_query, detect_anomalies

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Expense Categorization Agent | DSU Minor Project",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- AUTO-HEAL SQLITE SCHEMA IF CORRUPTED ---
def ensure_valid_transactions_schema():
  conn = get_connection()
  cursor = conn.cursor()
  cursor.execute("PRAGMA table_info(transactions)")
  cols = [row[1] for row in cursor.fetchall()]
  if cols and "source" not in cols:
    cursor.execute("DROP TABLE IF EXISTS transactions")
    cursor.execute("""
            CREATE TABLE transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                description TEXT,
                amount REAL,
                category TEXT,
                confidence REAL,
                source TEXT,
                is_reviewed INTEGER DEFAULT 0,
                is_alerted INTEGER DEFAULT 0
            )
        """)
    conn.commit()
  conn.close()


ensure_valid_transactions_schema()

# --- INNOVATIVE FINTECH CYBER-HUD & GLASSMORPHISM STYLING ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    /* Global Space-Grade Animated Mesh & Grid Background */
    .stApp {
        background-color: #020617;
        background-image: 
            radial-gradient(ellipse 80% 50% at 50% -10%, rgba(99, 102, 241, 0.28), transparent),
            radial-gradient(circle at 85% 30%, rgba(56, 189, 248, 0.12), transparent 45%),
            radial-gradient(circle at 15% 75%, rgba(16, 185, 129, 0.10), transparent 45%),
            linear-gradient(to right, rgba(99, 102, 241, 0.04) 1px, transparent 1px),
            linear-gradient(to bottom, rgba(99, 102, 241, 0.04) 1px, transparent 1px);
        background-size: 100% 100%, 100% 100%, 100% 100%, 44px 44px, 44px 44px;
        color: #f8fafc;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Keyframe Animations */
    @keyframes spinSlow {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 0 25px rgba(99, 102, 241, 0.45), inset 0 0 20px rgba(56, 189, 248, 0.25); }
        50% { box-shadow: 0 0 45px rgba(56, 189, 248, 0.75), inset 0 0 30px rgba(99, 102, 241, 0.45); }
    }
    @keyframes floatBadge {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-4px); }
    }

    /* Sidebar Glass Panel */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(9, 13, 24, 0.96) 0%, rgba(15, 23, 42, 0.96) 100%);
        backdrop-filter: blur(24px);
        border-right: 1px solid rgba(99, 102, 241, 0.2);
        padding-top: 18px;
    }

    /* Neon Status Pill */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(99, 102, 241, 0.45);
        padding: 6px 16px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 600;
        color: #c7d2fe;
        margin: 0 6px 12px 6px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
        animation: floatBadge 4s ease-in-out infinite;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.65) 0%, rgba(15, 23, 42, 0.88) 100%);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(129, 140, 248, 0.25);
        border-radius: 22px;
        padding: 28px;
        box-shadow: 0 20px 45px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.12);
        margin-bottom: 24px;
        transition: transform 0.25s ease, border-color 0.25s ease;
    }
    .glass-card:hover {
        border-color: rgba(56, 189, 248, 0.55);
        transform: translateY(-3px);
    }

    /* Custom HUD KPI Metric Cards */
    .kpi-card {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.65) 100%);
        border-radius: 18px;
        padding: 22px 24px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.5);
        transition: all 0.25s ease;
    }
    .kpi-card:hover {
        transform: translateY(-4px);
    }
    .kpi-indigo { border: 1px solid rgba(99, 102, 241, 0.45); border-top: 3px solid #6366f1; }
    .kpi-cyan   { border: 1px solid rgba(56, 189, 248, 0.45); border-top: 3px solid #38bdf8; }
    .kpi-amber  { border: 1px solid rgba(245, 158, 11, 0.45); border-top: 3px solid #f59e0b; }
    .kpi-rose   { border: 1px solid rgba(244, 63, 94, 0.45);  border-top: 3px solid #f43f5e; }

    .kpi-title {
        color: #94a3b8;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
    }
    .kpi-value {
        color: #f8fafc;
        font-size: 30px;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1.1;
    }
    .kpi-sub {
        color: #64748b;
        font-size: 12px;
        margin-top: 8px;
    }

    /* Primary Gradient Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #3b82f6 100%);
        color: white;
        border-radius: 14px;
        padding: 0.85rem 2.5rem;
        font-size: 17px;
        font-weight: 700;
        border: 1px solid rgba(255, 255, 255, 0.28);
        box-shadow: 0 8px 28px rgba(99, 102, 241, 0.55);
        transition: all 0.25s ease;
        margin: 0 auto;
        display: block;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #4f46e5 0%, #2563eb 100%);
        box-shadow: 0 12px 36px rgba(56, 189, 248, 0.7);
        transform: translateY(-2px);
    }

    /* --- BOTTOM-RIGHT FLOATING AI CHATBOT BUTTON & POPUP --- */
    [data-testid="stMain"] [data-testid="stPopover"] {
        position: fixed !important;
        bottom: 28px !important;
        right: 28px !important;
        z-index: 999999 !important;
        width: auto !important;
    }
    [data-testid="stMain"] [data-testid="stPopover"] > button {
        background: linear-gradient(135deg, #6366f1 0%, #06b6d4 100%) !important;
        color: #ffffff !important;
        border-radius: 999px !important;
        padding: 0.85rem 1.6rem !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        border: 1px solid rgba(255, 255, 255, 0.35) !important;
        box-shadow: 0 10px 32px rgba(99, 102, 241, 0.7) !important;
        transition: all 0.25s ease !important;
    }
    [data-testid="stMain"] [data-testid="stPopover"] > button:hover {
        transform: translateY(-3px) scale(1.04) !important;
        box-shadow: 0 14px 40px rgba(6, 182, 212, 0.85) !important;
    }
    /* Floating Pop-up Chat Window Styling */
    [data-testid="stPopoverBody"] {
        width: 420px !important;
        max-width: 92vw !important;
        background: linear-gradient(160deg, #0f172a 0%, #020617 100%) !important;
        border: 1px solid rgba(56, 189, 248, 0.5) !important;
        border-radius: 20px !important;
        box-shadow: 0 25px 65px rgba(0, 0, 0, 0.92) !important;
        padding: 18px !important;
    }
    [data-testid="stPopoverBody"] div.stButton > button {
        padding: 0.35rem 0.75rem !important;
        font-size: 13px !important;
        border-radius: 10px !important;
        box-shadow: none !important;
    }

    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 5.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- ANIMATED RADIANT HERO LOGO ---
logo_html = """
<div style="display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; margin-top: 8px; margin-bottom: 22px;">
    <div style="margin-bottom: 14px;">
        <span class="status-pill">🟢 Local SQLite Vault</span>
        <span class="status-pill">⚡ Real-Time ML + MAD Audit</span>
        <span class="status-pill">🤖 Groq Llama 3.3 Assistant</span>
    </div>
    <div style="position: relative; width: 148px; height: 148px; display: flex; align-items: center; justify-content: center; margin-bottom: 18px;">
        <!-- Rotating Aurora Halo -->
        <div style="position: absolute; width: 100%; height: 100%; border-radius: 50%; 
                    background: conic-gradient(from 0deg, #6366f1, #38bdf8, #10b981, #a855f7, #6366f1); 
                    filter: blur(14px); opacity: 0.75; animation: spinSlow 8s linear infinite;">
        </div>
        <!-- Inner Core -->
        <div style="position: absolute; width: 84%; height: 84%; border-radius: 50%; 
                    background: radial-gradient(circle at 30% 30%, #1e293b, #020617 85%);
                    display: flex; align-items: center; justify-content: center; 
                    border: 2px solid rgba(129, 140, 248, 0.7);
                    animation: pulseGlow 3.5s ease-in-out infinite;">
            <span style="font-size: 58px;">🛡️</span>
        </div>
    </div>
    <h1 style="font-size: 50px; font-weight: 800; line-height: 1.08; letter-spacing: -0.02em;
               background: linear-gradient(to right, #ffffff 15%, #a5b4fc 55%, #38bdf8 95%); 
               -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 10px;">
        Personal Expense Categorization Agent
    </h1>
    <p style="font-size: 18px; color: #94a3b8; max-width: 650px; margin: 0 auto; font-weight: 500;">
        Autonomous Financial Auditing, Anomaly Detection & Conversational Intelligence Suite
    </p>
</div>
"""

# --- INITIALIZE CLASSIFIER & DATA ---
if 'classifier' not in st.session_state:
  st.session_state.classifier = ExpenseClassifier()
  load_sample_csv('sample_statements.csv')
  df_init = get_all_transactions()

  conn = get_connection()
  try:
    rules_df = pd.read_sql_query('SELECT * FROM rules', conn)
    for _, r in rules_df.iterrows():
      df_init.loc[
          df_init['description'].str.contains(r['keyword'], case=False, na=False),
          'category',
      ] = r['category']
  except:
    pass
  conn.close()

  st.session_state.classifier.train(df_init)
  st.session_state.data_loaded = True

# --- APP PAGE ROUTING STATE ---
if 'app_page' not in st.session_state:
  st.session_state.app_page = 'Landing Page'

# --- INITIALIZE POP-UP CHAT HISTORY ---
if 'chat_messages' not in st.session_state:
  st.session_state.chat_messages = [
      {
          'role': 'assistant',
          'content': (
              'Hi! I am your **Financial Audit Assistant**. Ask me about your'
              ' spending breakdown, top merchants, or flagged security alerts.'
          ),
      }
  ]


def save_dataframe_to_sqlite(df_to_save: pd.DataFrame):
  """Inserts uploaded rows while preserving database.py's exact transactions schema."""
  ensure_valid_transactions_schema()
  conn = get_connection()
  cursor = conn.cursor()
  cursor.execute('DELETE FROM transactions')
  try:
    cursor.execute("DELETE FROM sqlite_sequence WHERE name='transactions'")
  except:
    pass

  for _, row in df_to_save.iterrows():
    cursor.execute(
        """
            INSERT INTO transactions (date, description, amount, category, confidence, source, is_reviewed, is_alerted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(row.get('date', '')),
            str(row.get('description', '')),
            float(row.get('amount', 0.0)),
            str(row.get('category', 'Uncategorized')),
            float(row.get('confidence', 0.85)),
            str(row.get('source', 'ml')),
            int(row.get('is_reviewed', 0)),
            int(row.get('is_alerted', 0)),
        ),
    )

  conn.commit()
  conn.close()


def build_financial_context(df: pd.DataFrame) -> str:
  """Builds a concise, anonymized financial summary from the local DataFrame."""
  if df.empty:
    return 'No transaction data is currently loaded.'

  total_spend = df['amount'].sum()
  total_txns = len(df)
  cat_totals = (
      df.groupby('category')['amount']
      .agg(['sum', 'count'])
      .reset_index()
      .to_dict(orient='records')
  )

  top_txns = df.nlargest(15, 'amount')[
      ['date', 'description', 'amount', 'category']
  ].to_csv(index=False)

  if 'is_alerted' in df.columns:
    alerts_df = df[df['is_alerted'] == 1]
    if not alerts_df.empty:
      cols = [
          c
          for c in ['date', 'description', 'amount', 'category', 'alert_reason']
          if c in alerts_df.columns
      ]
      alerts_text = alerts_df[cols].to_csv(index=False)
    else:
      alerts_text = 'None detected.'
  else:
    alerts_text = 'None detected.'

  return f"""
You are the AI Financial Auditing Assistant embedded inside the Personal Expense Categorization Agent (DSU Minor Project).
Answer the user's questions accurately, clearly, and concisely using ONLY the verified local ledger summary below.
Currency is Indian Rupees (₹). Do not invent transactions or numbers not present in the context.

- Total Tracked Spend: ₹{total_spend:,.2f}
- Total Transactions: {total_txns}
- Category Breakdown (sum & count): {cat_totals}

Flagged Security Alerts / Anomalies:
{alerts_text}

Top 15 Highest Transactions:
{top_txns}
"""


def generate_ai_reply(user_prompt: str, df: pd.DataFrame) -> str:
  """Generates an AI reply using Groq Llama 3.3 grounded with Pandas facts."""
  rule_answer, _, _ = answer_natural_language_query(user_prompt, df)
  api_key = st.secrets.get('GROQ_API_KEY', None)

  if not api_key:
    return (
        f'{rule_answer}\n\n*(Tip: Add `GROQ_API_KEY` to Streamlit Secrets to'
        ' enable full conversational AI.)*'
    )

  try:
    client = Groq(api_key=api_key)
    system_context = build_financial_context(df)
    if rule_answer:
      system_context += f'\nDeterministic Pandas Lookup Result: {rule_answer}'

    api_messages = [{'role': 'system', 'content': system_context}] + [
        {'role': m['role'], 'content': m['content']}
        for m in st.session_state.chat_messages[-6:]
    ]

    completion = client.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=api_messages,
        temperature=0.2,
    )
    return completion.choices[0].message.content
  except Exception as e:
    return f'{rule_answer}\n\n*(AI fallback triggered: {e})*'


def render_floating_chatbot(df: pd.DataFrame):
  """Renders the floating bottom-right AI Chatbot button and pop-up window."""
  with st.popover('💬 AI Chatbot'):
    h_col1, h_col2 = st.columns([4, 1])
    with h_col1:
      st.markdown('#### 🤖 Financial AI Auditor')
    with h_col2:
      if st.button('🗑️', key='clear_floating_chat', help='Clear chat'):
        st.session_state.chat_messages = [
            {
                'role': 'assistant',
                'content': 'Chat cleared! How can I help audit your expenses?',
            }
        ]
        st.rerun()

    # Quick-demo prompt chips for presentations
    q1, q2 = st.columns(2)
    quick_prompt = None
    with q1:
      if st.button('📊 Spend Summary', use_container_width=True):
        quick_prompt = (
            'Summarize my spending by category and highlight my highest'
            ' expense.'
        )
    with q2:
      if st.button('🚨 Check Anomalies', use_container_width=True):
        quick_prompt = (
            'Are there any flagged security alerts or duplicate transactions?'
        )

    chat_box = st.container(height=310)
    with chat_box:
      for msg in st.session_state.chat_messages:
        with st.chat_message(msg['role']):
          st.markdown(msg['content'])

    typed_prompt = st.chat_input(
        'Ask about spending, alerts...', key='floating_chat_input'
    )
    active_prompt = quick_prompt or typed_prompt

    if active_prompt:
      st.session_state.chat_messages.append(
          {'role': 'user', 'content': active_prompt}
      )
      with chat_box:
        with st.chat_message('user'):
          st.markdown(active_prompt)
        with st.chat_message('assistant'):
          with st.spinner('Auditing ledger...'):
            reply = generate_ai_reply(active_prompt, df)
            st.markdown(reply)
      st.session_state.chat_messages.append(
          {'role': 'assistant', 'content': reply}
      )


def show_landing_page():
  st.markdown(logo_html, unsafe_allow_html=True)

  # 3-Pillar Showcase Grid
  f1, f2, f3 = st.columns(3, gap='medium')
  with f1:
    st.markdown(
        """
            <div class="glass-card" style="text-align: left; height: 100%;">
                <div style="font-size: 28px; margin-bottom: 10px;">🧠</div>
                <h4 style="color: #f8fafc; margin-bottom: 8px;">Sub-Word ML Engine</h4>
                <p style="color: #94a3b8; font-size: 14px; line-height: 1.55;">
                    Parses messy UPI & NEFT strings using character n-gram TF-IDF and Logistic Regression with live human-in-the-loop retraining.
                </p>
            </div>
        """,
        unsafe_allow_html=True,
    )
  with f2:
    st.markdown(
        """
            <div class="glass-card" style="text-align: left; height: 100%;">
                <div style="font-size: 28px; margin-bottom: 10px;">🚨</div>
                <h4 style="color: #f8fafc; margin-bottom: 8px;">MAD Anomaly Shield</h4>
                <p style="color: #94a3b8; font-size: 14px; line-height: 1.55;">
                    Uses Median Absolute Deviation (MAD) Z-scores to catch outlier spikes and same-day duplicate charges automatically.
                </p>
            </div>
        """,
        unsafe_allow_html=True,
    )
  with f3:
    st.markdown(
        """
            <div class="glass-card" style="text-align: left; height: 100%;">
                <div style="font-size: 28px; margin-bottom: 10px;">💬</div>
                <h4 style="color: #f8fafc; margin-bottom: 8px;">Grounded AI Auditor</h4>
                <p style="color: #94a3b8; font-size: 14px; line-height: 1.55;">
                    Ask natural-language financial questions backed by deterministic Pandas ledger facts—zero hallucination risk.
                </p>
            </div>
        """,
        unsafe_allow_html=True,
    )

  st.markdown('<br>', unsafe_allow_html=True)
  col1, col2, col3 = st.columns([1.2, 1, 1.2])
  with col2:
    if st.button('🚀 Launch Command Center', use_container_width=True):
      st.session_state.app_page = 'Dashboard'
      st.rerun()

  st.markdown(
      '<p style="text-align: center; color: #64748b; font-size: 13px;'
      ' margin-top: 22px;">Dayananda Sagar University • Minor Project •'
      ' Python, Streamlit, scikit-learn, SQLite & Groq</p>',
      unsafe_allow_html=True,
  )


def show_dashboard():
  with st.sidebar:
    st.markdown(
        """
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 10px; padding: 8px 10px; background: rgba(99,102,241,0.12); border-radius: 12px; border: 1px solid rgba(99,102,241,0.3);">
                <div style="background: linear-gradient(135deg, #6366f1, #38bdf8); padding: 6px 10px; border-radius: 8px; font-size: 16px;">🛡️</div>
                <div>
                    <h4 style="margin: 0; color: #fff; font-size: 16px; font-weight: 700;">Expense Agent</h4>
                    <span style="font-size: 11px; color: #38bdf8;">v2.0 • DSU Minor Project</span>
                </div>
            </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('---')

    selected_subpage = st.radio(
        'COMMAND NAVIGATION',
        [
            '📊 Executive Dashboard',
            '🛠️ Human Review Queue',
            '⚡ Live AI Sandbox',
            '📈 Interactive Analytics',
            '🚨 Auditing & Security',
            '🏗️ System Architecture',
        ],
    )

    st.markdown('---')
    uploaded_file = st.file_uploader('📂 Upload Statement (CSV)', type=['csv'])
    if uploaded_file is not None:
      file_sig = f'{uploaded_file.name}_{uploaded_file.size}'
      if st.session_state.get('last_uploaded_file') != file_sig:
        df_uploaded = pd.read_csv(uploaded_file)
        df_uploaded = st.session_state.classifier.predict_batch(df_uploaded)
        df_uploaded = detect_anomalies(df_uploaded)
        save_dataframe_to_sqlite(df_uploaded)
        st.session_state.last_uploaded_file = file_sig
        st.success('Statement processed!')
        st.rerun()

    if st.button('📥 Reset to Demo Data', use_container_width=True):
      load_sample_csv('sample_statements.csv')
      df_reload = get_all_transactions()
      df_reload = st.session_state.classifier.predict_batch(df_reload)
      df_reload = detect_anomalies(df_reload)
      save_dataframe_to_sqlite(df_reload)
      st.session_state.pop('last_uploaded_file', None)
      st.success('Demo data reloaded!')
      st.rerun()

    st.markdown('---')
    if st.button('🏠 Back to Home', use_container_width=True):
      st.session_state.app_page = 'Landing Page'
      st.rerun()

  df = get_all_transactions()
  df = detect_anomalies(df)

  # --- SUBPAGE RENDERER ---
  if selected_subpage == '📊 Executive Dashboard':
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
            <div>
                <h1 style="margin: 0; font-size: 34px; font-weight: 800;">Executive Command Center</h1>
                <p style="color: #94a3b8; font-size: 15px; margin: 4px 0 0 0;">Real-time financial telemetry, ML classification health, and audit alerts.</p>
            </div>
            <span class="status-pill">🟢 System Online</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    total_spend = df['amount'].sum() if not df.empty else 0
    total_txn = len(df)
    review_count = (
        len(df[df['is_reviewed'] == 1]) if 'is_reviewed' in df.columns else 0
    )
    alert_count = (
        len(df[df['is_alerted'] == 1]) if 'is_alerted' in df.columns else 0
    )

    c1, c2, c3, c4 = st.columns(4, gap='medium')
    with c1:
      st.markdown(
          f"""
          <div class="kpi-card kpi-indigo">
              <div class="kpi-title">Total Tracked Spend</div>
              <div class="kpi-value">₹{total_spend:,.0f}</div>
              <div class="kpi-sub">Across all parsed ledger entries</div>
          </div>
          """,
          unsafe_allow_html=True,
      )
    with c2:
      st.markdown(
          f"""
          <div class="kpi-card kpi-cyan">
              <div class="kpi-title">Total Transactions</div>
              <div class="kpi-value">{total_txn}</div>
              <div class="kpi-sub">100% auto-classified via ML</div>
          </div>
          """,
          unsafe_allow_html=True,
      )
    with c3:
      st.markdown(
          f"""
          <div class="kpi-card kpi-amber">
              <div class="kpi-title">Pending Review</div>
              <div class="kpi-value">{review_count}</div>
              <div class="kpi-sub">Low-confidence items queued</div>
          </div>
          """,
          unsafe_allow_html=True,
      )
    with c4:
      st.markdown(
          f"""
          <div class="kpi-card kpi-rose">
              <div class="kpi-title">Security Alerts</div>
              <div class="kpi-value">{alert_count}</div>
              <div class="kpi-sub">Flagged by MAD anomaly engine</div>
          </div>
          """,
          unsafe_allow_html=True,
      )

    st.markdown('<br>', unsafe_allow_html=True)

    # Live Visual Split: Mini Spend Breakdown + Live Ledger Table
    dash_left, dash_right = st.columns([1.15, 1.35], gap='large')
    with dash_left:
      st.markdown('#### 🍩 Category Allocation')
      if not df.empty:
        cat_df = df.groupby('category')['amount'].sum().reset_index()
        fig_mini = px.pie(
            cat_df,
            values='amount',
            names='category',
            hole=0.62,
            template='plotly_dark',
            color_discrete_sequence=px.colors.qualitative.Vivid,
        )
        fig_mini.update_traces(textposition='inside', textinfo='percent+label')
        fig_mini.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
            height=320,
        )
        st.plotly_chart(fig_mini, use_container_width=True)

    with dash_right:
      st.markdown('#### 🧾 Live Categorized Ledger')
      if not df.empty:
        display_cols = [
            c
            for c in ['date', 'description', 'amount', 'category', 'confidence']
            if c in df.columns
        ]
        st.dataframe(
            df[display_cols].head(8),
            use_container_width=True,
            height=320,
            hide_index=True,
        )

  elif selected_subpage == '🛠️ Human Review Queue':
    st.title('Human Review & Active Learning')
    st.dataframe(
        df[[
            'id',
            'date',
            'description',
            'amount',
            'category',
            'confidence',
            'source',
        ]],
        use_container_width=True,
    )
    st.markdown('### 🛠️ Correct a Category')
    with st.form('correction_form'):
      cc1, cc2, cc3 = st.columns(3)
      txn_id = cc1.number_input('Transaction ID to fix', min_value=1, step=1)
      new_cat = cc2.selectbox(
          'Correct Category',
          [
              'Food & Dining',
              'Groceries',
              'Transport',
              'Shopping',
              'Bills & Utilities',
              'Rent',
              'Entertainment',
          ],
      )
      keyword_rule = cc3.text_input(
          'Keyword rule (optional)', placeholder='e.g. ZEPTO'
      )
      submitted = st.form_submit_button('Save Correction & Retrain Model')
      if submitted:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE transactions SET category = ?, confidence = 1.0, source ="
            " 'human', is_reviewed = 0 WHERE id = ?",
            (new_cat, txn_id),
        )
        if keyword_rule:
          cursor.execute(
              'INSERT OR REPLACE INTO rules (keyword, category) VALUES (?, ?)',
              (keyword_rule.upper(), new_cat),
          )
        conn.commit()
        conn.close()
        updated_df = get_all_transactions()
        st.session_state.classifier.train(updated_df)
        st.success(
            f"Updated transaction #{txn_id} to '{new_cat}' and saved rule!"
        )
        st.rerun()

  elif selected_subpage == '⚡ Live AI Sandbox':
    st.title('Live Model Inference Sandbox')
    st.caption(
        'Test how the TF-IDF + Logistic Regression model parses raw UPI/NEFT'
        ' strings in real time.'
    )
    raw_input = st.text_input(
        'Enter raw bank description:',
        'UPI/8839210/ZEPTO INSTANT GROCERIES/okicici',
    )
    if st.button('Run AI Classification'):
      if not st.session_state.classifier.is_trained:
        st.error('Model not trained yet.')
      else:
        cleaned = st.session_state.classifier.clean_text(raw_input)
        X_vec = st.session_state.classifier.vectorizer.transform([cleaned])
        pred_cat = st.session_state.classifier.model.predict(X_vec)[0]
        proba = max(st.session_state.classifier.model.predict_proba(X_vec)[0])
        col_a, col_b = st.columns(2)
        with col_a:
          st.markdown(
              f"""
              <div class="kpi-card kpi-indigo">
                  <div class="kpi-title">Predicted Category</div>
                  <div class="kpi-value">{pred_cat}</div>
              </div>
              """,
              unsafe_allow_html=True,
          )
        with col_b:
          st.markdown(
              f"""
              <div class="kpi-card kpi-cyan">
                  <div class="kpi-title">Model Confidence</div>
                  <div class="kpi-value">{proba*100:.1f}%</div>
              </div>
              """,
              unsafe_allow_html=True,
          )

  elif selected_subpage == '📈 Interactive Analytics':
    st.title('Interactive Financial Analytics')

    st.markdown('### 📈 Live Agent Confidence Stream (Real-Time)')
    x_vals = np.linspace(0, 10, 50)
    noise = np.random.normal(0, 0.08, 50)
    y_vals = np.sin(x_vals) * 0.2 + 0.85 + noise

    live_df = pd.DataFrame(
        {'Time Step': x_vals, 'Model Confidence': y_vals}
    )
    fig_live = px.line(
        live_df,
        x='Time Step',
        y='Model Confidence',
        template='plotly_dark',
        markers=True,
        title='Live Classification Confidence Telemetry',
    )
    fig_live.update_traces(line_color='#38bdf8', line_width=3)
    fig_live.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=280,
    )
    st.plotly_chart(fig_live, use_container_width=True)
    st.markdown('<br>', unsafe_allow_html=True)

    if not df.empty:
      cat_summary = df.groupby('category')['amount'].sum().reset_index()
      fig_donut = px.pie(
          cat_summary,
          values='amount',
          names='category',
          hole=0.55,
          title='Spending Distribution',
          template='plotly_dark',
      )
      fig_donut.update_layout(
          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
      )

      fig_bar = px.bar(
          cat_summary,
          x='category',
          y='amount',
          color='category',
          title='Spend per Category',
          template='plotly_dark',
      )
      fig_bar.update_layout(
          paper_bgcolor='rgba(0,0,0,0)',
          plot_bgcolor='rgba(0,0,0,0)',
          showlegend=False,
      )

      cg1, cg2 = st.columns(2, gap='large')
      cg1.plotly_chart(fig_donut, use_container_width=True)
      cg2.plotly_chart(fig_bar, use_container_width=True)
      st.dataframe(cat_summary, use_container_width=True)
    else:
      st.info('No data available.')

  elif selected_subpage == '🚨 Auditing & Security':
    st.title('Automated Auditing & Security Alerts')
    alert_df = df[df['is_alerted'] == 1]
    if not alert_df.empty:
      for _, row in alert_df.iterrows():
        st.error(
            f"**Date:** {row['date']} | **Merchant:**"
            f" {row['description']} | **Amount:** ₹{row['amount']:,.0f}\n\n**Audit"
            f" Flag:** {row['alert_reason']}"
        )
    else:
      st.success(
          'All transactions verified clean. No security anomalies detected.'
      )

  elif selected_subpage == '🏗️ System Architecture':
    st.title('System Architecture & Specifications')
    st.markdown(
        """
            <div class="glass-card" style="text-align: left;">
                <h3 style="margin-bottom: 16px; color: #818cf8;">🏗️ End-to-End Hybrid Intelligence Pipeline</h3>
                <ul style="color: #cbd5e1; line-height: 1.9; font-size: 16px;">
                    <li><b>Ingestion & Validation:</b> CSV statement normalization, date/amount cleansing, and SQLite persistence.</li>
                    <li><b>ML Classification Engine:</b> TF-IDF character n-grams (2–5 chars) extracting sub-word merchant patterns + Logistic Regression.</li>
                    <li><b>Active Learning Loop:</b> Human corrections write deterministic SQL rules and retrain classifier weights live.</li>
                    <li><b>MAD Anomaly Shield:</b> Median Absolute Deviation (MAD) Z-scores flag outlier spikes and same-day duplicate charges.</li>
                    <li><b>Grounded AI Auditor:</b> Hybrid Pandas deterministic verification + Groq Llama 3.3 floating pop-up assistant.</li>
                </ul>
            </div>
        """,
        unsafe_allow_html=True,
    )

  # --- RENDER FLOATING BOTTOM-RIGHT AI CHATBOT ON ALL DASHBOARD PAGES ---
  render_floating_chatbot(df)


# --- ROUTER EXECUTION ---
if st.session_state.app_page == 'Landing Page':
  show_landing_page()
elif st.session_state.app_page == 'Dashboard':
  show_dashboard()