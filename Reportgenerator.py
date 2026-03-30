"""
Reportgenerator.py

Handles payroll computation and export of reports to CSV/Excel.
"""

import os
import pandas as pd
from datetime import datetime


# reports folder will be created next to this file
REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


class ReportGenerator:
    """Generates salary and attendance reports."""

    def __init__(self, db_manager):
        self.db = db_manager

    def generate_payroll_report(self, month: str) -> str:
        """
        Computes payroll for all employees for the given month and exports to Excel.
        month format: 'YYYY-MM'
        Returns absolute path to the generated file.
        """
        salary_data = self.db.generate_salary_records(month)

        if not salary_data:
            raise ValueError(f"No employee data found for month: {month}")

        df = pd.DataFrame(salary_data)
        df.columns = ["Employee ID", "Name", "Department",
                      "Hourly Rate (Rs)", "Total Hours", "Gross Salary (Rs)"]
        df["Month"] = month

        # Summary row
        total_row = pd.DataFrame([{
            "Employee ID": "-",
            "Name": "TOTAL",
            "Department": "-",
            "Hourly Rate (Rs)": "-",
            "Total Hours": df["Total Hours"].sum(),
            "Gross Salary (Rs)": df["Gross Salary (Rs)"].sum(),
            "Month": month
        }])
        df = pd.concat([df, total_row], ignore_index=True)

        filename = f"Payroll_{month}.xlsx"
        filepath = os.path.join(REPORTS_DIR, filename)

        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Payroll Report")
            ws = writer.sheets["Payroll Report"]
            for col in ws.columns:
                max_len = max(len(str(cell.value)) for cell in col if cell.value)
                ws.column_dimensions[col[0].column_letter].width = max_len + 4

        print(f"[Report] Payroll report saved to: {filepath}")
        return filepath

    def generate_attendance_report(self, month: str) -> str:
        """
        Exports a full attendance log for the given month to Excel.
        month format: 'YYYY-MM'
        Returns absolute path to the generated file.
        """
        rows = self.db.get_all_attendance_for_month(month)

        if not rows:
            raise ValueError(f"No attendance data found for month: {month}")

        data = []
        for r in rows:
            data.append({
                "Employee": r["name"],
                "Department": r["department"],
                "Date": r["log_date"],
                "Entry Time": r["entry_time"] or "-",
                "Exit Time": r["exit_time"] or "Still In",
                "Net Hours": r["net_hours"],
                "Earnings (Rs)": round(r["net_hours"] * r["hourly_rate"], 2)
            })

        df = pd.DataFrame(data)
        filename = f"Attendance_{month}.xlsx"
        filepath = os.path.join(REPORTS_DIR, filename)

        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Attendance Log")
            ws = writer.sheets["Attendance Log"]
            for col in ws.columns:
                max_len = max(len(str(cell.value)) for cell in col if cell.value)
                ws.column_dimensions[col[0].column_letter].width = max_len + 4

        print(f"[Report] Attendance report saved to: {filepath}")
        return filepath

    def export_csv(self, month: str, report_type: str = "attendance") -> str:
        """
        Exports attendance or payroll data to CSV.
        report_type: 'attendance' or 'payroll'
        Returns absolute path to the generated CSV file.
        """
        if report_type == "payroll":
            data = self.db.generate_salary_records(month)
            df = pd.DataFrame(data)
            filename = f"Payroll_{month}.csv"
        else:
            rows = self.db.get_all_attendance_for_month(month)
            data = [dict(r) for r in rows]
            df = pd.DataFrame(data)
            filename = f"Attendance_{month}.csv"

        filepath = os.path.join(REPORTS_DIR, filename)
        df.to_csv(filepath, index=False)
        print(f"[Report] CSV saved to: {filepath}")
        return filepath