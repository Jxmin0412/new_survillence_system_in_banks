# smart_surveillance_ui.py

import cv2
import numpy as np
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import os

# === CONFIGURATION ===
USE_VIDEO_FILE = True  # False → webcam
VIDEO_PATH = "fire_sample2.mp4"  # your fire clip
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
UPDATE_DELAY_MS = 30  # approx 33 FPS

# Fire detection params
LOWER_FIRE = np.array([0, 120, 150])  # HSV lower bound
UPPER_FIRE = np.array([35, 255, 255])  # HSV upper bound
FIRE_PIXEL_THRESHOLD = 5000  # total mask pixels to flip “active”
CONTOUR_AREA_THRESHOLD = 2000  # min area of each contour to include


class SurveillanceApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Smart Surveillance System – Bank Prototype")

        # Video display
        self.video_label = Label(window, bg="black")
        self.video_label.pack()

        # Status labels
        self.fire_status = Label(
            window,
            text="🔥 Fire Detection: Inactive",
            fg="gray",
            bg="black",
            font=("Helvetica", 14),
        )
        self.weapon_status = Label(
            window,
            text="🔫 Weapon Detection: Inactive",
            fg="gray",
            bg="black",
            font=("Helvetica", 14),
        )
        self.face_status = Label(
            window,
            text="🧑 Face Recognition: Inactive",
            fg="gray",
            bg="black",
            font=("Helvetica", 14),
        )
        for lbl in (self.fire_status, self.weapon_status, self.face_status):
            lbl.pack(pady=2, fill="x")

        # Video source
        if USE_VIDEO_FILE and os.path.exists(VIDEO_PATH):
            self.capture = cv2.VideoCapture(VIDEO_PATH)
        else:
            if USE_VIDEO_FILE and not os.path.exists(VIDEO_PATH):
                print(f"[WARNING] '{VIDEO_PATH}' not found, falling back to webcam.")
            self.capture = cv2.VideoCapture(0)

        # Start loop
        self.window.after(0, self.update_frame)
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

    def update_frame(self):
        ret, frame = self.capture.read()
        if not ret:
            # loop video file
            if USE_VIDEO_FILE:
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.capture.read()
            if not ret:
                return

        # Resize for consistent processing
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # ——— Fire Detection ———
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_FIRE, UPPER_FIRE)
        # clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

        fire_pixels = cv2.countNonZero(mask)
        fire_detected = fire_pixels > FIRE_PIXEL_THRESHOLD

        # find and merge contours into one union box
        if fire_detected:
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            boxes = []
            for cnt in contours:
                if cv2.contourArea(cnt) > CONTOUR_AREA_THRESHOLD:
                    x, y, w, h = cv2.boundingRect(cnt)
                    boxes.append((x, y, x + w, y + h))

            if boxes:
                # compute union of all boxes
                x1 = min(b[0] for b in boxes)
                y1 = min(b[1] for b in boxes)
                x2 = max(b[2] for b in boxes)
                y2 = max(b[3] for b in boxes)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Update fire status label
        if fire_detected:
            self.fire_status.configure(text="🔥 Fire Detection: ACTIVE", fg="red")
        else:
            self.fire_status.configure(text="🔥 Fire Detection: Inactive", fg="gray")

        # ——— Convert to Tk image & display ———
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        # Schedule next frame
        self.window.after(UPDATE_DELAY_MS, self.update_frame)

    def on_close(self):
        self.capture.release()
        self.window.destroy()


# Add this at the bottom of smart_surveillance_ui.py


def get_processed_frame():
    if USE_VIDEO_FILE and os.path.exists(VIDEO_PATH):
        capture = cv2.VideoCapture(VIDEO_PATH)
    else:
        capture = cv2.VideoCapture(0)

    while True:
        ret, frame = capture.read()
        if not ret:
            if USE_VIDEO_FILE:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_FIRE, UPPER_FIRE)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

        fire_pixels = cv2.countNonZero(mask)
        fire_detected = fire_pixels > FIRE_PIXEL_THRESHOLD

        if fire_detected:
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            boxes = []
            for cnt in contours:
                if cv2.contourArea(cnt) > CONTOUR_AREA_THRESHOLD:
                    x, y, w, h = cv2.boundingRect(cnt)
                    boxes.append((x, y, x + w, y + h))

            if boxes:
                x1 = min(b[0] for b in boxes)
                y1 = min(b[1] for b in boxes)
                x2 = max(b[2] for b in boxes)
                y2 = max(b[3] for b in boxes)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        yield frame, fire_detected

    capture.release()


if __name__ == "__main__":
    root = tk.Tk()
    app = SurveillanceApp(root)
    root.mainloop()
