import cv2
import dlib
import os
import datetime
import sqlite3
import logging
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk


class FaceRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("📸 Face Registration System")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")

        # Variables
        self.subjects_var = tk.StringVar()
        self.student_var = tk.StringVar()
        self.cap = None
        self.running = False

        # === UI SETUP ===
        tk.Label(root, text="📚 Smart Attendance Register", font=("Arial", 18, "bold"), bg="#f0f0f0").pack(pady=10)

        tk.Label(root, text="Enter Subject Names (space-separated):", bg="#f0f0f0").pack()
        tk.Entry(root, textvariable=self.subjects_var, width=50).pack(pady=5)

        tk.Label(root, text="Enter Student Name:", bg="#f0f0f0").pack()
        tk.Entry(root, textvariable=self.student_var, width=50).pack(pady=5)

        tk.Button(root, text="Start Camera", command=self.start_camera, bg="#4caf50", fg="white", width=15).pack(pady=10)
        tk.Button(root, text="Save Photo", command=self.save_photo, bg="#2196f3", fg="white", width=15).pack(pady=5)
        tk.Button(root, text="Quit", command=self.quit_app, bg="#f44336", fg="white", width=15).pack(pady=5)

        self.video_label = tk.Label(root, bg="#000000")
        self.video_label.pack(pady=10)

        # === Face detection setup ===
        base_dir = os.path.dirname(os.path.abspath(__file__))
        haar_path = os.path.join(base_dir, 'haarcascade_frontalface_default.xml')
        self.face_cascade = cv2.CascadeClassifier(haar_path)

    # ---------------------------------------------------------------------
    def start_camera(self):
        subjects = self.subjects_var.get().strip()
        student = self.student_var.get().strip()

        if not subjects or not student:
            messagebox.showwarning("Input Required", "Please enter both subject(s) and student name.")
            return

        self.subject_names = [s.strip().capitalize() for s in subjects.split()]
        self.student_name = student.strip().capitalize()

        # Prepare folders
        base_dir = os.path.dirname(os.path.abspath(__file__))
        students_base = os.path.join(base_dir, "students")
        os.makedirs(students_base, exist_ok=True)

        self.subject_folders = []
        for subject in self.subject_names:
            subject_folder = os.path.join(students_base, subject)
            student_folder = os.path.join(subject_folder, self.student_name)
            os.makedirs(student_folder, exist_ok=True)
            self.subject_folders.append((subject, student_folder))

        # Start camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Unable to access webcam.")
            return

        self.running = True
        self.show_frame()

    # ---------------------------------------------------------------------
    def show_frame(self):
        if self.running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, self.student_name, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Convert to Tkinter image
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

            self.root.after(10, self.show_frame)

    # ---------------------------------------------------------------------
    def save_photo(self):
        if not self.cap or not self.running:
            messagebox.showwarning("Warning", "Camera not running.")
            return

        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture frame.")
            return

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        for subject, folder in self.subject_folders:
            filename = os.path.join(folder, f"{self.student_name}_{subject}_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            print(f"✅ Saved: {filename}")

        messagebox.showinfo("Saved", f"Photo saved for {self.student_name} in all subjects.")

    # ---------------------------------------------------------------------
    def quit_app(self):
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()


# -------------------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognizerApp(root)
    root.mainloop()
