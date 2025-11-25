# ExamGuard AI  
### Smart Exam Surveillance & Automated Attendance System  

---

## 📌 Overview  
**ExamGuard AI** is an embedded AI-powered proctoring system that automates exam hall surveillance using **real-time face recognition** and **cheating detection**.  
The system marks attendance using **DLIB**, switches automatically to cheating-monitoring mode, and detects prohibited activities with **YOLOv11**.  
All violations are captured and sent to the administrator instantly with evidence.

This project ensures high exam integrity while reducing the need for continuous human monitoring.

---

## ✨ Key Features  
- Automated **face-based attendance**  
- Subject-wise student registration and folder-level result management  
- Continuous identity verification to prevent impersonation  
- Real-time **cheating detection** using YOLOv11  
- Detects **phone**, **earphone**, **smartwatch**, **head turning**  
- RTSP-based live streaming from IP camera  
- Automatic **email alerts with image evidence**  
- Runs completely on **Raspberry Pi** (no cloud required)

---

## 🧠 Tech Stack  
**Hardware:** Raspberry Pi 4B, Imou Ranger 2 IP Camera  
**AI Models:** YOLOv11, DLIB  
**Libraries:** OpenCV, SMTP, NumPy  
**Tools:** Roboflow (dataset), Google Colab (training)  
**Languages:** Python  

---

## 📂 Dataset  
Custom dataset built in Roboflow containing:

**Cheating Classes:**  
- Mobile phone  
- Earphone  
- Smartwatch  
- Head turning  

**Normal Class:**  
- Reading / writing (non-cheating)

Augmentations include: brightness variation, rotation, blur, cropping, scaling — improving robustness for real exam conditions.

---

## 🚀 How the System Works  
1. Students are registered subject-wise with DLIB face encodings.  
2. System begins in **attendance mode** and marks present students automatically.  
3. After the attendance window closes, the system switches to **cheating detection mode**.  
4. YOLOv11 identifies any suspicious activity in real time.  
5. When cheating is detected, an email alert is sent with:  
   - Student name  
   - Roll number  
   - Cheating type  
   - Timestamp  
   - Proof image  
6. All data and images are stored in structured subject folders.

---

## 📨 Alert Format  
Each alert email contains:  
- **Student Identity**  
- **Cheating Event Description**  
- **Timestamp**  
- **Attached Evidence Image**

---

## 📦 Future Enhancements  
- Multi-camera support  
- Faster inference using Coral TPU / Jetson  
- Web-based admin dashboard  
- Voice-based anomaly detection  
- Multi-person tracking

---



## 🔧 System Architecture  

