"""
Application.py

Main GUI Application using CustomTkinter.
Camera feed is displayed INSIDE the Tkinter window (no cv2.imshow).
This is required on macOS to avoid SIGABRT crashes.
"""

import customtkinter as ctk
from tkinter import messagebox, ttk
import tkinter as tk
import time
import os
import subprocess
import sys
import cv2
from PIL import Image, ImageTk
from datetime import datetime

from Faceengine import FaceEngine
from Reportgenerator import ReportGenerator


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class FacialBooksApp:

    def __init__(self, db_manager):
        self.db = db_manager
        self.face_engine = FaceEngine(db_manager)
        self.report_gen = ReportGenerator(db_manager)

        self.root = ctk.CTk()
        self.root.title("FacialBooks — Automated Attendance & Payroll")
        self.root.geometry("1200x750")
        self.root.minsize(1000, 650)

        # Camera state
        self._cap = None
        self._camera_running = False
        self._last_action = {}
        self._photo_ref = None   # keep ImageTk reference alive

        self._build_ui()

    # ─────────────────────────── UI BUILD ───────────────────────────────

    def _build_ui(self):
        # Header
        header = ctk.CTkFrame(self.root, height=55, corner_radius=0,
                               fg_color="#1a1a2e")
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        ctk.CTkLabel(header, text="FacialBooks",
                     font=ctk.CTkFont(size=22, weight="bold"),
                     text_color="#e94560").pack(side="left", padx=20, pady=10)

        ctk.CTkLabel(header,
                     text="Automated Employee Time-Tracking & Payroll",
                     font=ctk.CTkFont(size=12),
                     text_color="#aaaaaa").pack(side="left", padx=5)

        self.time_label = ctk.CTkLabel(header, text="",
                                        font=ctk.CTkFont(size=13),
                                        text_color="#ffffff")
        self.time_label.pack(side="right", padx=20)
        self._update_clock()

        # Tabs
        self.tabs = ctk.CTkTabview(self.root, anchor="nw")
        self.tabs.pack(fill="both", expand=True, padx=10, pady=10)

        for tab in ["Dashboard", "Employees", "Attendance", "Payroll"]:
            self.tabs.add(tab)

        self._build_dashboard_tab(self.tabs.tab("Dashboard"))
        self._build_employees_tab(self.tabs.tab("Employees"))
        self._build_attendance_tab(self.tabs.tab("Attendance"))
        self._build_payroll_tab(self.tabs.tab("Payroll"))

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _update_clock(self):
        self.time_label.configure(
            text=datetime.now().strftime("%A, %d %b %Y   %H:%M:%S"))
        self.root.after(1000, self._update_clock)

    def _on_close(self):
        self._stop_camera()
        self.root.destroy()

    # ─────────────────────── DASHBOARD TAB ──────────────────────────────

    def _build_dashboard_tab(self, parent):
        parent.grid_columnconfigure(0, weight=0)
        parent.grid_columnconfigure(1, weight=2)
        parent.grid_columnconfigure(2, weight=1)
        parent.grid_rowconfigure(0, weight=1)

        # Left controls
        left = ctk.CTkFrame(parent, width=195)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        left.grid_propagate(False)

        ctk.CTkLabel(left, text="Attendance Camera",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(18, 4))

        ctk.CTkLabel(left,
                     text="Live feed shown in\nthe centre panel.\n\n"
                          "Faces are recognized\n& logged automatically.",
                     font=ctk.CTkFont(size=11), text_color="#aaaaaa",
                     justify="center").pack(padx=12, pady=6)

        self.cam_status_label = ctk.CTkLabel(
            left, text="● Camera OFF",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#e74c3c")
        self.cam_status_label.pack(pady=4)

        ctk.CTkButton(left, text="▶  Start Camera",
                      font=ctk.CTkFont(size=13, weight="bold"),
                      fg_color="#27ae60", hover_color="#1e8449",
                      command=self._start_camera).pack(pady=5, padx=12, fill="x")

        ctk.CTkButton(left, text="⏹  Stop Camera",
                      fg_color="#c0392b", hover_color="#922b21",
                      command=self._stop_camera).pack(pady=5, padx=12, fill="x")

        ctk.CTkLabel(left, text="Manual Override",
                     font=ctk.CTkFont(size=12, weight="bold")).pack(pady=(16, 4))

        self.manual_emp_var = ctk.StringVar()
        self.manual_emp_menu = ctk.CTkOptionMenu(
            left, variable=self.manual_emp_var,
            values=["-- select --"], width=170)
        self.manual_emp_menu.pack(pady=4, padx=12)
        self._refresh_employee_dropdown(self.manual_emp_var, self.manual_emp_menu)

        ctk.CTkButton(left, text="Log Entry",
                      command=lambda: self._manual_log("entry")).pack(
            pady=3, padx=12, fill="x")
        ctk.CTkButton(left, text="Log Exit",
                      command=lambda: self._manual_log("exit")).pack(
            pady=3, padx=12, fill="x")

        # Centre: camera feed inside Tkinter label widget
        cam_frame = ctk.CTkFrame(parent, fg_color="#0d0d0d")
        cam_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 5))
        cam_frame.grid_rowconfigure(1, weight=1)
        cam_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(cam_frame, text="Live Camera Feed",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color="#666666").grid(row=0, column=0, pady=(8, 2))

        self.cam_canvas = tk.Label(cam_frame, bg="#0d0d0d",
                                    text="[ Camera not started ]",
                                    fg="#444444",
                                    font=("Courier New", 12))
        self.cam_canvas.grid(row=1, column=0, sticky="nsew",
                              padx=6, pady=(0, 6))

        # Right: activity log
        right = ctk.CTkFrame(parent)
        right.grid(row=0, column=2, sticky="nsew")

        ctk.CTkLabel(right, text="Today's Activity Log",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(15, 5))

        self.activity_log = ctk.CTkTextbox(
            right, font=ctk.CTkFont(family="Courier New", size=11),
            state="disabled")
        self.activity_log.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        ctk.CTkButton(right, text="Refresh Log",
                      command=self._refresh_activity_log).pack(pady=(0, 10))
        self._refresh_activity_log()

    # ─────────────────────── CAMERA LOGIC ───────────────────────────────

    def _start_camera(self):
        if self._camera_running:
            messagebox.showinfo("Camera", "Camera is already running.")
            return

        self.face_engine.refresh_encodings()
        self._cap = cv2.VideoCapture(0)

        if not self._cap.isOpened():
            messagebox.showerror(
                "Camera Error",
                "Cannot open webcam.\n\n"
                "Allow camera access:\n"
                "System Settings → Privacy & Security → Camera\n"
                "→ enable for Terminal or your IDE")
            self._cap = None
            return

        self._camera_running = True
        self._last_action = {}
        self.cam_status_label.configure(
            text="● Camera ON", text_color="#27ae60")
        self._append_activity("Camera started — watching for faces...\n")
        self._update_camera_frame()

    def _update_camera_frame(self):
        """
        Reads one frame from webcam, runs face recognition,
        draws results, then shows it inside the Tkinter label widget.
        No cv2.imshow or cv2.waitKey used — safe on macOS.
        """
        if not self._camera_running or self._cap is None:
            return

        ret, frame = self._cap.read()
        if not ret:
            self._stop_camera()
            return

        frame = cv2.flip(frame, 1)                          # mirror
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        recognized = self.face_engine.recognize_face(rgb)

        for info in recognized:
            top, right_c, bottom, left = info["location"]
            name = info["name"]
            emp_id = info["id"]

            color = (0, 220, 80) if emp_id else (220, 50, 50)
            cv2.rectangle(rgb, (left, top), (right_c, bottom), color, 2)

            # Label background
            (tw, th), _ = cv2.getTextSize(
                name, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(rgb, (left, top - th - 10),
                          (left + tw + 6, top), color, -1)
            cv2.putText(rgb, name, (left + 3, top - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            # Log with cooldown
            if emp_id:
                now = time.time()
                if now - self._last_action.get(emp_id, 0) > 30:
                    self._last_action[emp_id] = now
                    self._on_face_recognized(emp_id, name)

        # Timestamp
        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(rgb, ts, (10, rgb.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # Resize to fit widget
        w = self.cam_canvas.winfo_width()
        h = self.cam_canvas.winfo_height()
        if w > 20 and h > 20:
            rgb = cv2.resize(rgb, (w, h))

        # Display inside Tkinter label — NO cv2.imshow
        img = Image.fromarray(rgb)
        self._photo_ref = ImageTk.PhotoImage(image=img)
        self.cam_canvas.configure(image=self._photo_ref, text="")

        # Schedule next frame (~25 fps)
        self.root.after(40, self._update_camera_frame)

    def _stop_camera(self):
        self._camera_running = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self.cam_canvas.configure(
            image="", text="[ Camera stopped ]", fg="#444444")
        self._photo_ref = None
        if hasattr(self, "cam_status_label"):
            self.cam_status_label.configure(
                text="● Camera OFF", text_color="#e74c3c")
        self._append_activity("Camera stopped.\n")
        self._refresh_activity_log()

    def _on_face_recognized(self, employee_id: int, name: str):
        status = self.db.get_today_status(employee_id)
        if status in ("ABSENT", "OUT"):
            success = self.db.log_entry(employee_id)
            action = "ENTRY logged" if success else "Already IN"
        else:
            success = self.db.log_exit(employee_id)
            action = "EXIT logged" if success else "Already OUT"

        ts = datetime.now().strftime("%H:%M:%S")
        self._append_activity(f"[{ts}]  {name:<20}  ->  {action}\n")

    def _append_activity(self, text: str):
        self.activity_log.configure(state="normal")
        self.activity_log.insert("end", text)
        self.activity_log.see("end")
        self.activity_log.configure(state="disabled")

    def _refresh_activity_log(self):
        today = datetime.now().strftime("%Y-%m-%d")
        self.activity_log.configure(state="normal")
        self.activity_log.delete("1.0", "end")

        employees = self.db.get_all_employees()
        lines = []
        for emp in employees:
            for log in self.db.get_attendance_by_employee(emp["id"]):
                if log["log_date"] == today:
                    entry = log["entry_time"] or "--"
                    exit_ = log["exit_time"] or "Still In"
                    hours = (f"{log['net_hours']:.2f}h"
                             if log["exit_time"] else "--")
                    lines.append(
                        f"  {emp['name']:<20} In:{entry}  "
                        f"Out:{exit_:<10}  {hours}\n")

        if lines:
            self.activity_log.insert(
                "end", f"-- {today} -----------------------\n")
            for l in lines:
                self.activity_log.insert("end", l)
        else:
            self.activity_log.insert(
                "end",
                f"No records yet for {today}.\nStart the camera to begin.")

        self.activity_log.configure(state="disabled")

    def _manual_log(self, log_type: str):
        selected = self.manual_emp_var.get()
        if "--" in selected:
            messagebox.showwarning("Manual Log", "Select an employee first.")
            return
        emp_id = int(selected.split("[")[1].rstrip("]"))
        emp_name = selected.split(" [")[0]

        if log_type == "entry":
            ok = self.db.log_entry(emp_id)
            msg = (f"Entry logged for {emp_name}."
                   if ok else f"{emp_name} is already clocked in.")
        else:
            ok = self.db.log_exit(emp_id)
            msg = (f"Exit logged for {emp_name}."
                   if ok else f"No open entry for {emp_name}.")

        messagebox.showinfo("Manual Log", msg)
        self._refresh_activity_log()

    def _refresh_employee_dropdown(self, var, menu):
        employees = self.db.get_all_employees()
        if employees:
            options = [f"{e['name']} [{e['id']}]" for e in employees]
            menu.configure(values=options)
            var.set(options[0])
        else:
            menu.configure(values=["-- no employees --"])
            var.set("-- no employees --")

    # ─────────────────────── EMPLOYEES TAB ──────────────────────────────

    def _build_employees_tab(self, parent):
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_columnconfigure(1, weight=2)
        parent.grid_rowconfigure(0, weight=1)

        form = ctk.CTkFrame(parent)
        form.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        ctk.CTkLabel(form, text="Register New Employee",
                     font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(20, 10))

        ctk.CTkLabel(form, text="Full Name").pack(anchor="w", padx=20)
        self.reg_name = ctk.CTkEntry(
            form, placeholder_text="e.g. Shifa Parveen")
        self.reg_name.pack(fill="x", padx=20, pady=(0, 10))

        ctk.CTkLabel(form, text="Department").pack(anchor="w", padx=20)
        self.reg_dept = ctk.CTkEntry(
            form, placeholder_text="e.g. Engineering")
        self.reg_dept.pack(fill="x", padx=20, pady=(0, 10))

        ctk.CTkLabel(form, text="Hourly Rate (Rs)").pack(anchor="w", padx=20)
        self.reg_rate = ctk.CTkEntry(form, placeholder_text="e.g. 500")
        self.reg_rate.pack(fill="x", padx=20, pady=(0, 10))

        self.reg_status = ctk.CTkLabel(
            form, text="", font=ctk.CTkFont(size=12),
            text_color="#aaaaaa", wraplength=220)
        self.reg_status.pack(pady=5, padx=20)

        ctk.CTkButton(form, text="Capture Face & Register",
                      font=ctk.CTkFont(size=13, weight="bold"),
                      command=self._register_employee).pack(
            pady=10, padx=20, fill="x")

        ctk.CTkLabel(form, text="Remove Employee",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(20, 5))
        self.del_emp_var = ctk.StringVar()
        self.del_emp_menu = ctk.CTkOptionMenu(
            form, variable=self.del_emp_var, values=["-- select --"])
        self.del_emp_menu.pack(fill="x", padx=20, pady=5)
        self._refresh_employee_dropdown(self.del_emp_var, self.del_emp_menu)

        ctk.CTkButton(form, text="Delete Employee",
                      fg_color="#c0392b", hover_color="#922b21",
                      command=self._delete_employee).pack(
            pady=5, padx=20, fill="x")

        right = ctk.CTkFrame(parent)
        right.grid(row=0, column=1, sticky="nsew")

        ctk.CTkLabel(right, text="Registered Employees",
                     font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(20, 5))

        cols = ("ID", "Name", "Department", "Rate (Rs/hr)", "Registered On")
        self.emp_tree = ttk.Treeview(
            right, columns=cols, show="headings", selectmode="browse")
        for col in cols:
            self.emp_tree.heading(col, text=col)
            self.emp_tree.column(col, width=130, anchor="center")

        sb = ttk.Scrollbar(right, orient="vertical",
                           command=self.emp_tree.yview)
        self.emp_tree.configure(yscroll=sb.set)
        self.emp_tree.pack(fill="both", expand=True, padx=10,
                           pady=(0, 5), side="left")
        sb.pack(side="right", fill="y", pady=(0, 5))

        ctk.CTkButton(right, text="Refresh List",
                      command=self._refresh_employee_list).pack(pady=(0, 10))
        self._refresh_employee_list()

    def _register_employee(self):
        name = self.reg_name.get().strip()
        dept = self.reg_dept.get().strip()
        rate_str = self.reg_rate.get().strip()

        if not name or not dept or not rate_str:
            messagebox.showwarning("Registration", "Please fill in all fields.")
            return
        try:
            rate = float(rate_str)
        except ValueError:
            messagebox.showerror("Registration",
                                  "Hourly rate must be a number.")
            return

        self.reg_status.configure(
            text="Opening camera — look at the lens...")
        self.root.update()

        encoding = self.face_engine.capture_face_images(
            name, num_samples=5,
            progress_callback=lambda s, t: (
                self.reg_status.configure(
                    text=f"Capturing sample {s}/{t}..."),
                self.root.update()
            )
        )

        if not encoding:
            self.reg_status.configure(
                text="No face detected. Try again in better lighting.")
            messagebox.showerror(
                "Registration",
                "No face detected.\n"
                "Ensure good lighting and face the camera directly.")
            return

        emp_id = self.db.add_employee(name, dept, rate, encoding)
        self.face_engine.refresh_encodings()
        self.reg_status.configure(text=f"{name} registered! (ID: {emp_id})")
        self.reg_name.delete(0, "end")
        self.reg_dept.delete(0, "end")
        self.reg_rate.delete(0, "end")
        self._refresh_employee_list()
        self._refresh_employee_dropdown(self.del_emp_var, self.del_emp_menu)
        self._refresh_employee_dropdown(self.manual_emp_var, self.manual_emp_menu)

    def _delete_employee(self):
        selected = self.del_emp_var.get()
        if "--" in selected:
            messagebox.showwarning("Delete", "Select an employee to delete.")
            return
        emp_id = int(selected.split("[")[1].rstrip("]"))
        emp_name = selected.split(" [")[0]

        if messagebox.askyesno("Confirm Delete",
                               f"Delete '{emp_name}' and all their records?"):
            self.db.delete_employee(emp_id)
            self.face_engine.refresh_encodings()
            messagebox.showinfo("Deleted", f"'{emp_name}' removed.")
            self._refresh_employee_list()
            self._refresh_employee_dropdown(
                self.del_emp_var, self.del_emp_menu)
            self._refresh_employee_dropdown(
                self.manual_emp_var, self.manual_emp_menu)

    def _refresh_employee_list(self):
        for item in self.emp_tree.get_children():
            self.emp_tree.delete(item)
        for emp in self.db.get_all_employees():
            self.emp_tree.insert("", "end", values=(
                emp["id"], emp["name"], emp["department"],
                emp["hourly_rate"], emp["registered_on"]
            ))

    # ─────────────────────── ATTENDANCE TAB ─────────────────────────────

    def _build_attendance_tab(self, parent):
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(1, weight=1)

        ctrl = ctk.CTkFrame(parent)
        ctrl.grid(row=0, column=0, sticky="ew", pady=(0, 5))

        ctk.CTkLabel(ctrl, text="Month (YYYY-MM):").pack(
            side="left", padx=10)
        self.att_month_var = ctk.StringVar(
            value=datetime.now().strftime("%Y-%m"))
        ctk.CTkEntry(ctrl, textvariable=self.att_month_var,
                     width=100).pack(side="left", padx=5)

        ctk.CTkLabel(ctrl, text="Employee:").pack(side="left", padx=10)
        self.att_emp_var = ctk.StringVar(value="All")
        self.att_emp_filter = ctk.CTkOptionMenu(
            ctrl, variable=self.att_emp_var, values=["All"])
        self.att_emp_filter.pack(side="left", padx=5)
        self._populate_att_filter()

        ctk.CTkButton(ctrl, text="Search",
                      command=self._load_attendance).pack(
            side="left", padx=10)
        ctk.CTkButton(ctrl, text="Export Excel",
                      command=self._export_attendance_excel).pack(
            side="right", padx=10)

        cols = ("Employee", "Department", "Date", "Entry",
                "Exit", "Net Hours", "Earnings (Rs)")
        self.att_tree = ttk.Treeview(parent, columns=cols, show="headings")
        for col in cols:
            self.att_tree.heading(col, text=col)
            self.att_tree.column(col, width=120, anchor="center")

        sb = ttk.Scrollbar(parent, orient="vertical",
                           command=self.att_tree.yview)
        self.att_tree.configure(yscroll=sb.set)
        self.att_tree.grid(row=1, column=0, sticky="nsew", padx=(10, 0))
        sb.grid(row=1, column=1, sticky="ns")
        self._load_attendance()

    def _populate_att_filter(self):
        employees = self.db.get_all_employees()
        options = ["All"] + [
            f"{e['name']} [{e['id']}]" for e in employees]
        self.att_emp_filter.configure(values=options)

    def _load_attendance(self):
        for item in self.att_tree.get_children():
            self.att_tree.delete(item)

        month = self.att_month_var.get().strip()
        selected = self.att_emp_var.get()

        if selected == "All":
            rows = self.db.get_all_attendance_for_month(month)
            for r in rows:
                earnings = round(r["net_hours"] * r["hourly_rate"], 2)
                self.att_tree.insert("", "end", values=(
                    r["name"], r["department"], r["log_date"],
                    r["entry_time"] or "--", r["exit_time"] or "Active",
                    f"{r['net_hours']:.2f}", f"Rs {earnings:.2f}"
                ))
        else:
            emp_id = int(selected.split("[")[1].rstrip("]"))
            emp = self.db.get_employee_by_id(emp_id)
            for r in self.db.get_attendance_by_employee(emp_id, month):
                earnings = round(r["net_hours"] * emp["hourly_rate"], 2)
                self.att_tree.insert("", "end", values=(
                    emp["name"], emp["department"], r["log_date"],
                    r["entry_time"] or "--", r["exit_time"] or "Active",
                    f"{r['net_hours']:.2f}", f"Rs {earnings:.2f}"
                ))

    def _export_attendance_excel(self):
        month = self.att_month_var.get().strip()
        try:
            path = self.report_gen.generate_attendance_report(month)
            messagebox.showinfo("Export", f"Saved:\n{path}")
            self._open_file(path)
        except ValueError as e:
            messagebox.showwarning("Export", str(e))

    # ─────────────────────── PAYROLL TAB ────────────────────────────────

    def _build_payroll_tab(self, parent):
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(1, weight=1)

        ctrl = ctk.CTkFrame(parent)
        ctrl.grid(row=0, column=0, sticky="ew", pady=(0, 5))

        ctk.CTkLabel(ctrl, text="Month (YYYY-MM):").pack(
            side="left", padx=10)
        self.pay_month_var = ctk.StringVar(
            value=datetime.now().strftime("%Y-%m"))
        ctk.CTkEntry(ctrl, textvariable=self.pay_month_var,
                     width=100).pack(side="left", padx=5)

        ctk.CTkButton(ctrl, text="Generate Payroll",
                      fg_color="#1a6b3c", hover_color="#145a32",
                      command=self._generate_payroll).pack(
            side="left", padx=10)
        ctk.CTkButton(ctrl, text="Export Excel",
                      command=self._export_payroll_excel).pack(
            side="right", padx=10)
        ctk.CTkButton(ctrl, text="Export CSV",
                      command=self._export_payroll_csv).pack(
            side="right", padx=5)

        cols = ("ID", "Name", "Department",
                "Hourly Rate", "Total Hours", "Gross Salary")
        self.pay_tree = ttk.Treeview(parent, columns=cols, show="headings")
        for col in cols:
            self.pay_tree.heading(col, text=col)
            self.pay_tree.column(col, width=140, anchor="center")

        sb = ttk.Scrollbar(parent, orient="vertical",
                           command=self.pay_tree.yview)
        self.pay_tree.configure(yscroll=sb.set)
        self.pay_tree.grid(row=1, column=0, sticky="nsew", padx=(10, 0))
        sb.grid(row=1, column=1, sticky="ns")

        self.pay_summary = ctk.CTkLabel(
            parent, text="",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#27ae60")
        self.pay_summary.grid(row=2, column=0, pady=8)

    def _generate_payroll(self):
        month = self.pay_month_var.get().strip()
        for item in self.pay_tree.get_children():
            self.pay_tree.delete(item)

        data = self.db.generate_salary_records(month)
        if not data:
            messagebox.showwarning("Payroll", "No employee data found.")
            return

        total = 0
        for rec in data:
            self.pay_tree.insert("", "end", values=(
                rec["id"], rec["name"], rec["department"],
                f"Rs {rec['hourly_rate']:.2f}",
                f"{rec['total_hours']:.2f} hrs",
                f"Rs {rec['gross_salary']:.2f}"
            ))
            total += rec["gross_salary"]

        self.pay_summary.configure(
            text=f"Total Payroll for {month}:  Rs {total:,.2f}"
                 f"  ({len(data)} employees)")

    def _export_payroll_excel(self):
        month = self.pay_month_var.get().strip()
        try:
            path = self.report_gen.generate_payroll_report(month)
            messagebox.showinfo("Export", f"Saved:\n{path}")
            self._open_file(path)
        except ValueError as e:
            messagebox.showwarning("Export", str(e))

    def _export_payroll_csv(self):
        month = self.pay_month_var.get().strip()
        try:
            path = self.report_gen.export_csv(month, "payroll")
            messagebox.showinfo("Export", f"CSV saved:\n{path}")
        except Exception as e:
            messagebox.showerror("Export", str(e))

    # ─────────────────────── HELPERS ────────────────────────────────────

    def _open_file(self, path: str):
        try:
            if sys.platform == "darwin":
                subprocess.call(["open", path])
            elif sys.platform == "win32":
                os.startfile(path)
            else:
                subprocess.call(["xdg-open", path])
        except Exception:
            pass

    def run(self):
        self.root.mainloop()