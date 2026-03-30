import sqlite3
import os
import json
from datetime import datetime
DB_PATH = os.path.join(os.path.dirname(__file__), "facialbooks.db")
class DatabaseManager:
    """Manages all database interactions for FacialBooks."""

    def __init__(self):
        self.db_path = DB_PATH

    def _get_connection(self):
        """Returns a new database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self):
        """Creates all required tables if they don't exist."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS employees (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                department  TEXT NOT NULL,
                hourly_rate REAL NOT NULL DEFAULT 0.0,
                encoding    TEXT,  -- JSON-serialized 128-d face embedding
                registered_on TEXT DEFAULT (datetime('now','localtime'))
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance_log (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id  INTEGER NOT NULL,
                log_date     TEXT NOT NULL,
                entry_time   TEXT,
                exit_time    TEXT,
                net_hours    REAL DEFAULT 0.0,
                FOREIGN KEY (employee_id) REFERENCES employees(id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS salary_records (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id   INTEGER NOT NULL,
                month         TEXT NOT NULL,  -- Format: YYYY-MM
                total_hours   REAL DEFAULT 0.0,
                gross_salary  REAL DEFAULT 0.0,
                generated_on  TEXT DEFAULT (datetime('now','localtime')),
                FOREIGN KEY (employee_id) REFERENCES employees(id)
            )
        """)

        conn.commit()
        conn.close()

    def add_employee(self, name: str, department: str, hourly_rate: float,
                     encoding: list) -> int:
        """Inserts a new employee and returns their ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        encoding_json = json.dumps(encoding)
        cursor.execute(
            "INSERT INTO employees (name, department, hourly_rate, encoding) "
            "VALUES (?, ?, ?, ?)",
            (name, department, hourly_rate, encoding_json)
        )
        employee_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return employee_id

    def get_all_employees(self) -> list:
        """Returns all employees as a list of Row objects."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM employees ORDER BY name")
        rows = cursor.fetchall()
        conn.close()
        return rows

    def get_employee_by_id(self, employee_id: int):
        """Returns a single employee row by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM employees WHERE id = ?", (employee_id,))
        row = cursor.fetchone()
        conn.close()
        return row

    def update_employee_encoding(self, employee_id: int, encoding: list):
        """Updates the face encoding for an existing employee."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE employees SET encoding = ? WHERE id = ?",
            (json.dumps(encoding), employee_id)
        )
        conn.commit()
        conn.close()

    def delete_employee(self, employee_id: int):
        """Deletes an employee and all related records."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM attendance_log WHERE employee_id = ?", (employee_id,))
        cursor.execute("DELETE FROM salary_records WHERE employee_id = ?", (employee_id,))
        cursor.execute("DELETE FROM employees WHERE id = ?", (employee_id,))
        conn.commit()
        conn.close()

    def get_all_encodings(self) -> list:
        """
        Returns list of (employee_id, name, encoding_array) tuples
        for use in face recognition matching.
        """
        import numpy as np
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, encoding FROM employees WHERE encoding IS NOT NULL")
        rows = cursor.fetchall()
        conn.close()
        result = []
        for row in rows:
            enc_list = json.loads(row["encoding"])
            result.append((row["id"], row["name"], np.array(enc_list)))
        return result
    def log_entry(self, employee_id: int) -> bool:
        """
        Logs an entry time for today. Prevents duplicate entries within 5 minutes.
        Returns True if logged, False if duplicate/already clocked in.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        today = datetime.now().strftime("%Y-%m-%d")
        now_str = datetime.now().strftime("%H:%M:%S")
        cursor.execute(
            "SELECT * FROM attendance_log WHERE employee_id=? AND log_date=? "
            "AND entry_time IS NOT NULL AND exit_time IS NULL",
            (employee_id, today)
        )
        existing = cursor.fetchone()
        if existing:
            conn.close()
            return False  

        cursor.execute(
            "INSERT INTO attendance_log (employee_id, log_date, entry_time) VALUES (?, ?, ?)",
            (employee_id, today, now_str)
        )
        conn.commit()
        conn.close()
        return True

    def log_exit(self, employee_id: int) -> bool:
        """
        Logs exit time for today's open entry record.
        Calculates net hours worked.
        Returns True if exit logged, False if no open entry found.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        today = datetime.now().strftime("%Y-%m-%d")
        now_str = datetime.now().strftime("%H:%M:%S")

        cursor.execute(
            "SELECT id, entry_time FROM attendance_log "
            "WHERE employee_id=? AND log_date=? AND exit_time IS NULL",
            (employee_id, today)
        )
        record = cursor.fetchone()
        if not record:
            conn.close()
            return False  # No open entry

        # Calculate net hours
        fmt = "%H:%M:%S"
        entry_dt = datetime.strptime(record["entry_time"], fmt)
        exit_dt = datetime.strptime(now_str, fmt)
        net_seconds = (exit_dt - entry_dt).total_seconds()
        net_hours = round(max(net_seconds / 3600, 0), 4)

        cursor.execute(
            "UPDATE attendance_log SET exit_time=?, net_hours=? WHERE id=?",
            (now_str, net_hours, record["id"])
        )
        conn.commit()
        conn.close()
        return True

    def get_today_status(self, employee_id: int) -> str:
        """Returns 'IN', 'OUT', or 'ABSENT' for an employee today."""
        conn = self._get_connection()
        cursor = conn.cursor()
        today = datetime.now().strftime("%Y-%m-%d")
        cursor.execute(
            "SELECT entry_time, exit_time FROM attendance_log "
            "WHERE employee_id=? AND log_date=?",
            (employee_id, today)
        )
        row = cursor.fetchone()
        conn.close()
        if not row:
            return "ABSENT"
        if row["exit_time"] is None:
            return "IN"
        return "OUT"

    def get_attendance_by_employee(self, employee_id: int, month: str = None) -> list:
        """
        Returns attendance logs for an employee.
        month format: 'YYYY-MM' (optional filter)
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        if month:
            cursor.execute(
                "SELECT * FROM attendance_log WHERE employee_id=? AND log_date LIKE ? "
                "ORDER BY log_date DESC",
                (employee_id, f"{month}%")
            )
        else:
            cursor.execute(
                "SELECT * FROM attendance_log WHERE employee_id=? ORDER BY log_date DESC",
                (employee_id,)
            )
        rows = cursor.fetchall()
        conn.close()
        return rows

    def get_all_attendance_for_month(self, month: str) -> list:
        """Returns all attendance logs for a given month (YYYY-MM)."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT al.*, e.name, e.hourly_rate, e.department "
            "FROM attendance_log al "
            "JOIN employees e ON al.employee_id = e.id "
            "WHERE al.log_date LIKE ? ORDER BY al.log_date, e.name",
            (f"{month}%",)
        )
        rows = cursor.fetchall()
        conn.close()
        return rows
    def generate_salary_records(self, month: str) -> list:
        """
        Computes salary for all employees for a given month.
        Returns list of dicts with salary details.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT e.id, e.name, e.department, e.hourly_rate,
                   COALESCE(SUM(al.net_hours), 0) AS total_hours
            FROM employees e
            LEFT JOIN attendance_log al
                ON e.id = al.employee_id AND al.log_date LIKE ?
            GROUP BY e.id
        """, (f"{month}%",))

        rows = cursor.fetchall()
        results = []
        for row in rows:
            gross = round(row["total_hours"] * row["hourly_rate"], 2)
            cursor.execute(
                "SELECT id FROM salary_records WHERE employee_id=? AND month=?",
                (row["id"], month)
            )
            existing = cursor.fetchone()
            if existing:
                cursor.execute(
                    "UPDATE salary_records SET total_hours=?, gross_salary=?, "
                    "generated_on=datetime('now','localtime') WHERE id=?",
                    (row["total_hours"], gross, existing["id"])
                )
            else:
                cursor.execute(
                    "INSERT INTO salary_records (employee_id, month, total_hours, gross_salary) "
                    "VALUES (?, ?, ?, ?)",
                    (row["id"], month, row["total_hours"], gross)
                )

            results.append({
                "id": row["id"],
                "name": row["name"],
                "department": row["department"],
                "hourly_rate": row["hourly_rate"],
                "total_hours": round(row["total_hours"], 2),
                "gross_salary": gross
            })

        conn.commit()
        conn.close()
        return results
