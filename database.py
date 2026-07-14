import sqlite3
from datetime import datetime

DB_NAME = "tb_detection.db"


def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT,
            prediction REAL,
            result TEXT,
            risk_level TEXT,
            threshold REAL,
            created_at TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_analysis(patient_name, prediction, result, risk_level, threshold):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO analyses 
        (patient_name, prediction, result, risk_level, threshold, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        patient_name,
        prediction,
        result,
        risk_level,
        threshold,
        datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    ))

    conn.commit()
    conn.close()


def get_analyses():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, patient_name, prediction, result, risk_level, threshold, created_at
        FROM analyses
        ORDER BY id DESC
    """)

    rows = cursor.fetchall()
    conn.close()
    return rows


def get_patients():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT DISTINCT patient_name
        FROM analyses
        WHERE patient_name IS NOT NULL
        ORDER BY patient_name ASC
    """)

    rows = cursor.fetchall()
    conn.close()

    return [row[0] for row in rows]


def get_analyses_by_patient(patient_name):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, patient_name, prediction, result, risk_level, threshold, created_at
        FROM analyses
        WHERE patient_name = ?
        ORDER BY id DESC
    """, (patient_name,))

    rows = cursor.fetchall()
    conn.close()
    return rows


def delete_all_analyses():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("DELETE FROM analyses")

    conn.commit()
    conn.close()