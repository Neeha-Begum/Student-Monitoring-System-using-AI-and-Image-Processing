from flask import Flask, render_template
import sqlite3
import cv2
import face_recognition
import pickle
import datetime
import pyttsx3
import json
from collections import defaultdict

app = Flask(_name_)

# Initialize TTS
try:
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 150)
except Exception as e:
    print(f"TTS Engine Error: {e}")
    tts_engine = None

def speak(message):
    print(message)
    if tts_engine:
        try:
            tts_engine.say(message)
            tts_engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")

# Load encodings
try:
    with open("face_encodings.pkl", "rb") as f:
        data = pickle.load(f)
    known_faces = data["encodings"]
    known_names = data["names"]
except Exception as e:
    print(f"Encoding load error: {e}")
    known_faces = []
    known_names = []

# Initialize DB
def initialize_database():
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS face_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            date TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

initialize_database()

def mark_attendance(name):
    today = datetime.date.today().strftime("%Y-%m-%d")
    now = datetime.datetime.now().strftime("%H:%M:%S")
    try:
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM attendance WHERE name=? AND date=?", (name, today))
        result = cursor.fetchone()
        if result:
            cursor.execute("UPDATE attendance SET time=? WHERE name=? AND date=?", (now, name, today))
        else:
            cursor.execute("INSERT INTO attendance (name, date, time) VALUES (?, ?, ?)", (name, today, now))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Attendance error: {e}")

def log_face(name):
    today = datetime.date.today().strftime("%Y-%m-%d")
    try:
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        cursor.execute("INSERT INTO face_log (name, date) VALUES (?, ?)", (name, today))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Log error: {e}")

@app.route('/')
def index():
    try:
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM attendance ORDER BY date DESC, time DESC")
        records = cursor.fetchall()
        conn.close()

        grouped_records = defaultdict(list)
        for row in records:
            grouped_records[row[2]].append(row)

        return render_template("index.html", grouped_records=grouped_records)
    except Exception as e:
        print(f"Fetch error: {e}")
        return render_template("index.html", grouped_records={})

@app.route('/scan')
def scan_faces():
    cam = None
    try:
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            raise Exception("Could not open camera.")

        speak("Face scanning has started. Please look at the camera.")

        while True:
            ret, frame = cam.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.5)
                name = "Unknown"

                if True in matches:
                    match_index = matches.index(True)
                    name = known_names[match_index]
                    mark_attendance(name)
                    log_face(name)
                    speak(f"{name}, marked present today.")

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                    cv2.imshow("Face Recognition", frame)
                    cv2.waitKey(1000)  # Wait 1 second before closing
                    return render_template("scan_complete.html", name=name)

                else:
                    speak("Unknown face detected. Please register.")
                    log_face("Unknown")

            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        return render_template("scan_complete.html", name="No face detected")

    except Exception as e:
        print(f"Scan error: {e}")
        return f"Error occurred: {e}"

    finally:
        if cam:
            cam.release()
        cv2.destroyAllWindows()

@app.route('/charts')
def charts():
    try:
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        cursor.execute("SELECT name, date FROM face_log")
        data = cursor.fetchall()
        conn.close()

        daily_counts = defaultdict(lambda: {"Known": 0, "Unknown": 0})
        for name, date in data:
            if name == "Unknown":
                daily_counts[date]["Unknown"] += 1
            else:
                daily_counts[date]["Known"] += 1

        labels = list(daily_counts.keys())
        known_counts = [daily_counts[date]["Known"] for date in labels]
        unknown_counts = [daily_counts[date]["Unknown"] for date in labels]

        return render_template("charts.html", labels=json.dumps(labels),
                               known=json.dumps(known_counts),
                               unknown=json.dumps(unknown_counts))
    except Exception as e:
        return f"Chart loading error: {e}"

if _name_ == '_main_':
    app.run(debug=True)
