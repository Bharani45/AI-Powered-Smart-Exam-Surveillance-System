"""
exam_gui.py
--------------------------------------------------
Unified Smart Exam Surveillance GUI
- Takes subject name and attendance duration
- Runs attendance for the specified time
- Automatically switches to YOLO-based cheating detection
"""

import tkinter as tk
from tkinter import messagebox
import logging
import time
import os
from threading import Thread

# Import your existing modules
from Attendance.attendance_taker import FaceRecognizer
from exam import CheatingDetector


class ExamGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Exam Surveillance System")
        self.root.geometry("550x350")
        self.root.configure(bg="#f8f8f8")

        # --- UI Variables ---
        self.subject_var = tk.StringVar()
        self.time_var = tk.StringVar()

        # --- Title ---
        tk.Label(root, text="🎓 Smart Exam Surveillance System",
                 font=("Arial", 18, "bold"), bg="#f8f8f8").pack(pady=20)

        # --- Subject Input ---
        tk.Label(root, text="Enter Subject Name:", bg="#f8f8f8", font=("Arial", 12)).pack()
        tk.Entry(root, textvariable=self.subject_var, font=("Arial", 12), width=30, justify='center').pack(pady=5)

        # --- Time Limit Input ---
        tk.Label(root, text="Enter Attendance Duration (seconds):", bg="#f8f8f8", font=("Arial", 12)).pack()
        tk.Entry(root, textvariable=self.time_var, font=("Arial", 12), width=30, justify='center').pack(pady=5)

        # --- Buttons ---
        tk.Button(root, text="Start Process", command=self.start_process, bg="#4caf50", fg="white",
                  font=("Arial", 12), width=20).pack(pady=10)

        tk.Button(root, text="Quit", command=root.destroy, bg="#f44336", fg="white",
                  font=("Arial", 12), width=20).pack(pady=5)

        # --- Status Display ---
        self.status_label = tk.Label(root, text="", bg="#f8f8f8", font=("Arial", 11, "italic"), fg="#333")
        self.status_label.pack(pady=10)

    # ---------------------------------------------------------------------
    def start_process(self):
        subject = self.subject_var.get().strip().capitalize()
        duration = self.time_var.get().strip()

        if not subject:
            messagebox.showwarning("Missing Input", "Please enter a subject name.")
            return

        if not duration.isdigit():
            messagebox.showwarning("Invalid Input", "Please enter a valid numeric time limit (in seconds).")
            return

        duration = int(duration)
        self.status_label.config(text=f"Starting attendance for '{subject}' ({duration} sec)...")

        # Run the attendance + cheating detection in a separate thread
        Thread(target=self.run_full_process, args=(subject, duration), daemon=True).start()

    # ---------------------------------------------------------------------
    def run_full_process(self, subject, duration):
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            subject_folder = os.path.join(base_dir, "Attendance", "students", subject)

            if not os.path.exists(subject_folder):
                messagebox.showerror("Error", f"Subject folder '{subject}' not found.\nExpected at:\n{subject_folder}")
                return

            logging.info(f"🕒 Starting Attendance for '{subject}' ({duration} seconds)...")

            # Step 1: Attendance Mode
            recognizer = FaceRecognizer(subject_name=subject)
            cap = recognizer.start_camera()

            start_time = time.time()
            while True:
                elapsed = time.time() - start_time
                if elapsed > duration:
                    logging.info("✅ Attendance phase complete.")
                    break
                recognizer.process_frame(cap)

            recognizer.stop_camera(cap)
            self.status_label.config(text="✅ Attendance complete. Starting cheating detection...")

            # Step 2: Cheating Detection
            logging.info(f"🧠 Starting Cheating Detection for '{subject}'...")
            detector = CheatingDetector(subject_name=subject)
            detector.run()

            self.status_label.config(text="🏁 Process Completed Successfully.")
            messagebox.showinfo("Process Complete", f"Attendance and cheating detection for '{subject}' completed.")

        except Exception as e:
            logging.error(f"❌ Error: {e}")
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")


# -------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    root = tk.Tk()
    app = ExamGUI(root)
    root.mainloop()
