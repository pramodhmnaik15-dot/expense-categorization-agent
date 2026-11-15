import sqlite3
import pandas as pd
import os

DB_NAME = "expenses.db"

def get_connection():
    """Create and return a database connection."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the SQLite database with transactions and rules tables."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Transactions table matching the agent specs
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
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
    
    # Custom correction rules table for human-in-the-loop learning
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            keyword TEXT UNIQUE,
            category TEXT
        )
    """)
    
    conn.commit()
    conn.close()

def load_sample_csv(file_path="sample_statements.csv"):
    """Load and clean the sample statement CSV into the database."""
    if not os.path.exists(file_path):
        return False
    
    df = pd.read_csv(file_path)
    conn = get_connection()
    cursor = conn.cursor()
    
    # Clear existing data to avoid duplicates on reload
    cursor.execute("DELETE FROM transactions")
    
    for _, row in df.iterrows():
        cursor.execute("""
            INSERT INTO transactions (date, description, amount, category, confidence, source, is_reviewed, is_alerted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row.get('date'),
            row.get('description'),
            row.get('amount'),
            row.get('category', 'Unassigned'),
            row.get('confidence', 0.0),
            row.get('source', 'model'),
            row.get('is_reviewed', 0),
            row.get('is_alerted', 0)
        ))
        
    conn.commit()
    conn.close()
    return True

def get_all_transactions():
    """Retrieve all transactions as a Pandas DataFrame."""
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM transactions", conn)
    conn.close()
    return df

# Initialize database tables on module load
init_db()