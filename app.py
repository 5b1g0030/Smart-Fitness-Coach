from flask import Flask, render_template, Response, request, jsonify, session
import cv2
import json
import os
import threading
import time
import secrets
from pose_detector import StandardSquatAnalyzer, SquatDetectorWithStandard

app = Flask(__name__)

# 金鑰
app.secret_key = secrets.token_hex(16)  # 生成32字元的隨機金鑰

class FlaskSquatDetector:
    def __init__(self):
        self.camera = None          # 鏡頭
        self.video_capture = None   # 影片檔案
        self.video_path = None      # 影片檔案路徑
        self.detector = None        # 深蹲檢測
        self.standard_sequence = None # 標準動作序列
        self.is_running = False       # 檢查是否在執行
        self.is_paused = False        # 檢查是否暫停
        self.current_frame = None     # 目前處理的畫面
        self.detection_mode = None  # 檢測模式：'camera' 或 'video'
        
        # 載入標準動作序列
        self.load_standard_sequence()
    
    # ===== 載入標準動作序列 =====
    def load_standard_sequence(self):
        
        try:
            if os.path.exists("standard_squat_sequence.json"): # 檢查檔案是否存在
                analyzer = StandardSquatAnalyzer() # 引入類別
                # 呼叫載入序列的函式，如果成功載入則儲存該序列
                if analyzer.load_standard_sequence("standard_squat_sequence.json"): 
                    self.standard_sequence = analyzer.standard_sequence
                    print("標準動作序列載入成功 by app")
                analyzer.cleanup() # 釋放資源
        except Exception as e:
            print(f"載入標準序列失敗: {e} by app")
    
    # ===== 啟動攝像頭檢測 =====
    def start_camera(self):
        """啟動攝像頭檢測"""
        if self.standard_sequence is None:
            return False, "沒有標準動作資料 by app"
        
        try:
            self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 920)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
            
            if not self.camera.isOpened():
                return False, "無法開啟攝像頭"
            
            self.detector = SquatDetectorWithStandard(
                standard_sequence=self.standard_sequence,
                squat_threshold=120,
                similarity_threshold=0.6
            )
            
            self.detection_mode = 'camera'
            self.is_running = True
            self.is_paused = False
            return True, "攝像頭啟動成功"
            
        except Exception as e:
            return False, f"啟動攝像頭失敗: {str(e)}"
    
    # ===== 啟動影片檢測 =====
    def start_video(self, video_path):
        
        # ----- 檢查有無標準序列 -----
        if self.standard_sequence is None:
            return False, "沒有標準動作資料 by app"
        
        # ----- 檢查檔案是否存在 -----
        if not os.path.exists(video_path):
            return False, "影片檔案不存在 by app"
        
        try:
            self.video_capture = cv2.VideoCapture(video_path)
            if not self.video_capture.isOpened():
                return False, "無法開啟影片檔案 by app"
            
            self.detector = SquatDetectorWithStandard(
                standard_sequence=self.standard_sequence,
                squat_threshold=120,
                similarity_threshold=0.6
            )
            
            self.detection_mode = 'video'
            self.video_path = video_path
            self.is_running = True
            self.is_paused = False
            return True, "影片載入成功"
            
        except Exception as e:
            return False, f"載入影片失敗: {str(e)}"
    
    def get_frame(self):
        """獲取處理後的影格"""
        if not self.is_running:
            return None
        
        if self.is_paused and self.current_frame is not None:
            return self.current_frame
        
        frame = None
        
        if self.detection_mode == 'camera' and self.camera:
            ret, frame = self.camera.read()
            if not ret:
                return None
            processed_frame = self.detector.process_frame(frame, mirror=True)
            
        elif self.detection_mode == 'video' and self.video_capture:
            if not self.is_paused:
                ret, frame = self.video_capture.read()
                if not ret:
                    self.stop()
                    return None
                processed_frame = self.detector.process_frame(frame, mirror=False)
            else:
                processed_frame = self.current_frame
        
        if frame is not None:
            self.current_frame = processed_frame
            
        return processed_frame
    
    def pause(self):
        """暫停"""
        self.is_paused = True
    
    def resume(self):
        """繼續"""
        self.is_paused = False
    
    def stop(self):
        """停止檢測"""
        self.is_running = False
        self.is_paused = False
        
        if self.camera:
            self.camera.release()
            self.camera = None
            
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
            
        if self.detector:
            self.detector.cleanup()
            self.detector = None
            
        self.current_frame = None
        self.detection_mode = None
    
    def get_stats(self):
        """獲取統計資料"""
        if self.detector:
            return {
                'squat_count': self.detector.squat_count,
                'correct_squat_count': self.detector.correct_squat_count,
                'accuracy': (self.detector.correct_squat_count / self.detector.squat_count * 100) if self.detector.squat_count > 0 else 0
            }
        return {'squat_count': 0, 'correct_squat_count': 0, 'accuracy': 0}

# 全域檢測器實例
flask_detector = FlaskSquatDetector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_detection', methods=['POST'])
def start_detection():
    """啟動檢測"""
    data = request.json
    mode = data.get('mode')
    
    if mode == 'camera':
        success, message = flask_detector.start_camera()
    elif mode == 'video':
        video_path = data.get('video_path', '')
        success, message = flask_detector.start_video(video_path)
    else:
        success, message = False, "無效的模式"
    
    return jsonify({'success': success, 'message': message})

@app.route('/control_playback', methods=['POST'])
def control_playback():
    """控制播放"""
    data = request.json
    action = data.get('action')
    
    if action == 'pause':
        flask_detector.pause()
    elif action == 'resume':
        flask_detector.resume()
    elif action == 'stop':
        flask_detector.stop()
    
    return jsonify({'success': True})

@app.route('/get_stats')
def get_stats():
    """獲取統計資料"""
    return jsonify(flask_detector.get_stats())

@app.route('/video_feed')
def video_feed():
    """視訊串流"""
    def generate_frames():
        while True:
            frame = flask_detector.get_frame()
            if frame is None:
                break
                
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                break
                
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.033)  # 約30fps
    
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
