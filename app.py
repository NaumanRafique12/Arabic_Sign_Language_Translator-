# app.py - FAST & STABLE VERSION
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'





import cv2
import numpy as np
import threading
import time
import mediapipe as mp
from tensorflow.keras.models import load_model

from src.config import ACTIONS, ARABIC_MAPPING, THRESHOLD
from src.keypoints import mediapipe_detection, extract_keypoints, draw_styled_landmarks
from src.llm_grammar import format_arabic_sentence
from src.utils import draw_info_text

# 1. SETUP
SEQ_LENGTH = 15  # Your model's sequence length

try:
    modelLSTM = load_model("models/action3.h5", compile=False)
    print("✅ LSTM Model loaded")
except Exception as e:
    print(f"❌ LSTM Error: {e}")
    exit()

# 2. SIMPLE GENDER THREAD (No complex logic)
class GenderThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.gender = "Male"
        self.manual = False
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        self.start()

    def run(self):
        # Load inside thread
        try:
            face_net = cv2.dnn.readNet("models/res10_300x300_ssd_iter_140000.caffemodel", "models/deploy.prototxt")
            gender_net = cv2.dnn.readNet("models/gender_net.caffemodel", "models/gender_deploy.prototxt")
            
            while self.running:
                if self.frame is not None and not self.manual:
                    self._process(face_net, gender_net)
                time.sleep(0.2)
        except: pass

    def _process(self, f_net, g_net):
        try:
            with self.lock: img = self.frame.copy()
            h, w = img.shape[:2]
            blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123])
            f_net.setInput(blob)
            dets = f_net.forward()
            
            # Simple "Biggest Face" logic
            best_idx = -1
            max_area = 0
            
            for i in range(dets.shape[2]):
                conf = dets[0,0,i,2]
                if conf > 0.6:
                    box = (dets[0,0,i,3:7] * np.array([w,h,w,h])).astype(int)
                    area = (box[2]-box[0]) * (box[3]-box[1])
                    if area > max_area:
                        max_area = area
                        best_idx = i
            
            if best_idx >= 0:
                box = (dets[0,0,best_idx,3:7] * np.array([w,h,w,h])).astype(int)
                face = img[box[1]:box[3], box[0]:box[2]]
                blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4, 87.8, 114.9), swapRB=False)
                g_net.setInput(blob)
                preds = g_net.forward()[0]
                # High threshold for female to prevent errors
                g = "Female" if preds[1] > 0.7 else "Male"
                with self.lock: self.gender = g
        except: pass

    def update(self, frame):
        with self.lock: self.frame = frame
    
    def get(self):
        with self.lock: return self.gender
        
    def toggle(self):
        with self.lock:
            self.manual = True
            self.gender = "Female" if self.gender == "Male" else "Male"
            return self.gender

# 3. LLM WORKER (Prevents Overlapping)
class LLMWorker:
    def __init__(self):
        self.display_text = "..."
        self.raw_words = []
        self.lock = threading.Lock()
        self.last_req_time = 0
    
    def add_word(self, word):
        """Add word only if it's different from the last word"""
        if self.raw_words and self.raw_words[-1] == word:
            return False  # Don't add duplicate
        
        self.raw_words.append(word)
        with self.lock:
            self.display_text = " ".join(self.raw_words)
        return True  # Word was added
    
    def trigger(self, gender):
        # Only trigger if we have words
        if not self.raw_words: return
        
        req_time = time.time()
        self.last_req_time = req_time
        
        threading.Thread(target=self._run, args=(self.raw_words.copy(), gender, req_time), daemon=True).start()
        
    def _run(self, words, gender, req_time):
        res = format_arabic_sentence(words, gender)
        
        with self.lock:
            # IMPORTANT: Only update if this request is still the newest one
            if req_time >= self.last_req_time:
                self.display_text = res
    
    def get_text(self):
        with self.lock: return self.display_text
    
    def clear(self):
        self.raw_words = []
        with self.lock: self.display_text = "..."

# 4. MAIN LOOP
def main():
    cap = cv2.VideoCapture(0)
    
    # Init
    gender_thread = GenderThread()
    llm_worker = LLMWorker()
    
    sequence = []
    
    # Simple Stabilizer Variables
    current_prediction = None
    frame_counter = 0
    STABILITY_FRAMES = 5  # Require 8 consecutive identical frames to confirm
    COOLDOWN = 0
    last_confirmed_word = None

    
    with mp.solutions.hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2, static_image_mode=True) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            
            # Gender
            gender_thread.update(frame)
            gender = gender_thread.get()
            
            # Hands
            image, results = mediapipe_detection(frame, hands)
            draw_styled_landmarks(image, results)
            
            if results.multi_hand_landmarks:
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-SEQ_LENGTH:]
                
                if len(sequence) == SEQ_LENGTH:
                    # PREDICT EVERY FRAME
                    res = modelLSTM.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                    idx = np.argmax(res)
                    conf = res[idx]
                    
                    if conf > THRESHOLD:
                        predicted_action = ACTIONS[idx]
                        
                        # --- SIMPLE STABILIZER ---
                        if predicted_action == current_prediction:
                            frame_counter += 1
                        else:
                            # Reset if prediction changes
                            current_prediction = predicted_action
                            frame_counter = 0
                            
                        # Confirm if held for X frames
                        if frame_counter > STABILITY_FRAMES and COOLDOWN == 0:
                            ar_word = ARABIC_MAPPING.get(predicted_action, predicted_action)
                            
                            if ar_word != last_confirmed_word:
                                added = llm_worker.add_word(ar_word)
                                
                                if added:
                                    llm_worker.trigger(gender)
                                    print(f"✅ Confirmed: {ar_word}")
                                    last_confirmed_word = ar_word  # Update last word
                                else:
                                    print(f"⏭️ Skipped duplicate: {ar_word}")
                            else:
                                print(f"⏭️ Skipped (same as last): {ar_word}")
                            
                            # Reset
                            frame_counter = 0
                            COOLDOWN = 8 # Wait 0.5s before next word
                            
            else:
                # If hands leave, reset sequence
                if len(sequence) > 0:
                    sequence = []
                    frame_counter = 0
                    
            # Handle cooldown
            if COOLDOWN > 0: COOLDOWN -= 1
            
            # Draw UI
            text = llm_worker.get_text()
            image = draw_info_text(image, text, gender)
            
            cv2.putText(image, "'G': Toggle Gender | 'C': Clear | 'Q': Quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 1)
            cv2.imshow('ArSL Simple', image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            if key == ord('c'): 
                llm_worker.clear()
                sequence = []
                last_confirmed_word = None
            if key == ord('g'): 
                gender = gender_thread.toggle()
                llm_worker.trigger(gender)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()