from flask import Flask, render_template, Response, request, jsonify, session
import cv2
import json
import os
import threading
import time
import secrets
import atexit
import signal
import shutil
import glob
from werkzeug.utils import secure_filename
from squat_detector import StandardSquatAnalyzer, SquatDetectorWithStandard

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

# 設定上傳檔案的目錄
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv'}

# 確保上傳目錄存在
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 限制檔案大小為 100MB

def allowed_file(filename):
    """檢查檔案副檔名是否允許"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_upload_folder():
    """清理上傳資料夾中的所有檔案"""
    try:
        if os.path.exists(UPLOAD_FOLDER):
            # 獲取資料夾中的所有檔案
            files = glob.glob(os.path.join(UPLOAD_FOLDER, '*'))
            for file_path in files:
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        print(f"已刪除檔案: {file_path}")
                except Exception as e:
                    print(f"刪除檔案 {file_path} 失敗: {e}")
            print("Upload 資料夾清理完成")
    except Exception as e:
        print(f"清理 upload 資料夾失敗: {e}")

def signal_handler(signum, frame):
    """處理程式終止信號"""
    print(f"接收到信號 {signum}，正在清理...")
    cleanup_upload_folder()
    flask_detector.stop()
    exit(0)

# 註冊程式退出時的清理函數
atexit.register(cleanup_upload_folder)

# 註冊信號處理器（處理 Ctrl+C 等強制終止）
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@app.teardown_appcontext
def cleanup_on_teardown(error):
    """Flask 應用關閉時的清理"""
    if error:
        print(f"應用關閉時發生錯誤: {error}")

# 在應用關閉時執行清理
@atexit.register
def cleanup_on_exit():
    """程式退出時的最終清理"""
    cleanup_upload_folder()
    flask_detector.stop()

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

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """上傳影片檔案"""
    try:
        # 檢查是否有檔案
        if 'video' not in request.files:
            return jsonify({'success': False, 'message': '沒有檔案被上傳'})
        
        file = request.files['video']
        
        # 檢查檔案名稱
        if file.filename == '':
            return jsonify({'success': False, 'message': '沒有選擇檔案'})
        
        # 檢查檔案類型
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'message': '不支援的檔案格式'})
        
        # 生成安全的檔案名稱
        filename = secure_filename(file.filename)
        # 加上時間戳避免檔名衝突
        timestamp = secrets.token_hex(4)
        name, ext = os.path.splitext(filename)
        filename = f"{name}_{timestamp}{ext}"
        
        # 儲存檔案
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # 返回完整路徑
        full_path = os.path.abspath(file_path)
        
        return jsonify({
            'success': True, 
            'message': '檔案上傳成功',
            'file_path': full_path,
            'filename': filename
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'上傳失敗: {str(e)}'})

@app.route('/calculate_accuracy_stream')
def calculate_accuracy_stream():
    """使用 EventSource 進行影片準確度計算的串流回應"""
    video_path = request.args.get('video_path', '')
    
    def generate_progress():
        if not video_path or not os.path.exists(video_path):
            yield f"data: {json.dumps({'type': 'error', 'message': '影片檔案不存在'})}\n\n"
            return
        
        if flask_detector.standard_sequence is None:
            yield f"data: {json.dumps({'type': 'error', 'message': '沒有標準動作資料'})}\n\n"
            return
        
        try:
            # 初始化影片讀取
            yield f"data: {json.dumps({'type': 'progress', 'progress': 5, 'message': '正在初始化影片讀取...'})}\n\n"
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                yield f"data: {json.dumps({'type': 'error', 'message': '無法開啟影片檔案'})}\n\n"
                return
            
            # 獲取影片資訊
            yield f"data: {json.dumps({'type': 'progress', 'progress': 10, 'message': '正在獲取影片資訊...'})}\n\n"
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            print(f"影片資訊: 總幀數={total_frames}, FPS={fps}")
            
            # 使用與影片播放模式相同的檢測器
            yield f"data: {json.dumps({'type': 'progress', 'progress': 15, 'message': '正在初始化檢測器...'})}\n\n"
            
            detector = SquatDetectorWithStandard(
                standard_sequence=flask_detector.standard_sequence,
                squat_threshold=120,
                similarity_threshold=0.6
            )
            
            frame_count = 0
            similarity_scores = []
            squat_similarities = []  # 只記錄深蹲動作時的相似度
            last_progress_update = 0
            
            yield f"data: {json.dumps({'type': 'progress', 'progress': 20, 'message': f'開始分析 {total_frames} 幀影片'})}\n\n"
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 轉換顏色格式
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 使用與播放模式相同的姿勢檢測邏輯
                results = detector.pose.process(rgb_frame)
                
                if results.pose_landmarks:
                    h, w, _ = frame.shape
                    
                    # 使用檢測器的完整判斷邏輯
                    is_squat, left_angle, right_angle, similarity, feedback = detector.is_squat_pose(
                        results.pose_landmarks.landmark, w, h
                    )
                    
                    # 記錄所有有效幀的相似度
                    similarity_scores.append(similarity)
                    
                    # 只在深蹲動作時記錄相似度（與播放模式一致）
                    if is_squat:
                        squat_similarities.append(similarity)
                    
                    # 更新檢測器的計數（模擬播放模式的行為）
                    detector.update_squat_count(is_squat, similarity)
                
                # 更新進度 - 從20%到90%
                progress = 20 + (frame_count / total_frames) * 70
                
                # 減少進度更新頻率
                if frame_count % 100 == 0 or progress - last_progress_update >= 2:
                    last_progress_update = progress
                    message = f'已處理 {frame_count}/{total_frames} 幀'
                    if similarity_scores:
                        current_avg = sum(similarity_scores) / len(similarity_scores) * 100
                        message += f' (當前平均準確度: {current_avg:.1f}%)'
                    
                    # 顯示深蹲檢測統計
                    if detector.squat_count > 0:
                        squat_accuracy = (detector.correct_squat_count / detector.squat_count) * 100
                        message += f' (深蹲準確率: {squat_accuracy:.1f}%)'
                    
                    yield f"data: {json.dumps({'type': 'progress', 'progress': progress, 'message': message})}\n\n"
            
            yield f"data: {json.dumps({'type': 'progress', 'progress': 95, 'message': '正在計算最終結果...'})}\n\n"
            
            cap.release()
            detector.cleanup()
            
            # 計算最終結果 - 提供多種準確度計算方式
            if similarity_scores:
                # 方式1: 所有幀的平均相似度
                average_accuracy = sum(similarity_scores) / len(similarity_scores) * 100
                
                # 方式2: 只計算深蹲動作幀的平均相似度
                squat_average_accuracy = (sum(squat_similarities) / len(squat_similarities) * 100) if squat_similarities else 0
                
                # 方式3: 基於深蹲計數的準確率（與播放模式一致）
                count_based_accuracy = (detector.correct_squat_count / detector.squat_count * 100) if detector.squat_count > 0 else 0
                
                result = {
                    'type': 'result',
                    'average_accuracy': average_accuracy,  # 所有幀的平均
                    'squat_accuracy': squat_average_accuracy,  # 深蹲幀的平均
                    'count_based_accuracy': count_based_accuracy,  # 計數式準確率
                    'total_frames': total_frames,
                    'valid_frames': len(similarity_scores),
                    'squat_frames': len(squat_similarities),
                    'squat_count': detector.squat_count,
                    'correct_squat_count': detector.correct_squat_count
                }
                print(f"計算完成:")
                print(f"  - 所有幀平均準確度: {average_accuracy:.2f}%")
                print(f"  - 深蹲幀平均準確度: {squat_average_accuracy:.2f}%") 
                print(f"  - 計數式準確率: {count_based_accuracy:.2f}%")
                print(f"  - 深蹲次數: {detector.squat_count}, 正確次數: {detector.correct_squat_count}")
            else:
                result = {
                    'type': 'result',
                    'average_accuracy': 0,
                    'squat_accuracy': 0,
                    'count_based_accuracy': 0,
                    'total_frames': total_frames,
                    'valid_frames': 0,
                    'squat_frames': 0,
                    'squat_count': 0,
                    'correct_squat_count': 0
                }
                print("警告: 影片中未檢測到有效的人體姿勢")
            
            yield f"data: {json.dumps(result)}\n\n"
            
        except Exception as e:
            error_msg = f'計算過程發生錯誤: {str(e)}'
            print(f"錯誤: {error_msg}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
    
    # 使用 EventSource 的標準回應格式
    response = Response(generate_progress(), mimetype='text/event-stream')
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Connection'] = 'keep-alive'
    response.headers['X-Accel-Buffering'] = 'no'
    return response

# 保留原有的 POST 路由作為備用
@app.route('/calculate_accuracy', methods=['POST'])
def calculate_accuracy():
    """計算影片準確度 (備用方法)"""
    def generate_progress():
        data = request.json
        video_path = data.get('video_path', '')
        
        if not video_path or not os.path.exists(video_path):
            yield f"data: {json.dumps({'type': 'error', 'message': '影片檔案不存在'})}\n\n"
            return
        
        if flask_detector.standard_sequence is None:
            yield f"data: {json.dumps({'type': 'error', 'message': '沒有標準動作資料'})}\n\n"
            return
        
        try:
            # 初始化影片讀取
            yield f"data: {json.dumps({'type': 'progress', 'progress': 5, 'message': '正在初始化影片讀取...'})}\n\n"
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                yield f"data: {json.dumps({'type': 'error', 'message': '無法開啟影片檔案'})}\n\n"
                return
            
            # 獲取影片資訊
            yield f"data: {json.dumps({'type': 'progress', 'progress': 10, 'message': '正在獲取影片資訊...'})}\n\n"
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            print(f"影片資訊: 總幀數={total_frames}, FPS={fps}")
            
            # 初始化檢測器（用於計算模式）
            yield f"data: {json.dumps({'type': 'progress', 'progress': 15, 'message': '正在初始化檢測器...'})}\n\n"
            
            detector = SquatDetectorWithStandard(
                standard_sequence=flask_detector.standard_sequence,
                squat_threshold=120,
                similarity_threshold=0.6
            )
            
            frame_count = 0
            similarity_scores = []
            
            yield f"data: {json.dumps({'type': 'progress', 'progress': 20, 'message': f'開始分析 {total_frames} 幀影片'})}\n\n"
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 轉換顏色格式
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 進行姿勢檢測
                results = detector.pose.process(rgb_frame)
                
                if results.pose_landmarks:
                    h, w, _ = frame.shape
                    # 提取角度
                    angles = detector.analyzer.extract_key_angles(
                        results.pose_landmarks.landmark, w, h
                    )
                    
                    if angles:
                        # 計算與標準動作的相似度
                        similarity, _ = detector.compare_with_standard(angles)
                        similarity_scores.append(similarity)
                
                # 更新進度 - 從20%到90%
                progress = 20 + (frame_count / total_frames) * 70
                
                # 更頻繁地發送進度更新
                if frame_count % 10 == 0:  # 改為每10幀發送一次
                    message = f'已處理 {frame_count}/{total_frames} 幀'
                    if similarity_scores:
                        current_avg = sum(similarity_scores) / len(similarity_scores) * 100
                        message += f' (當前平均準確度: {current_avg:.1f}%)'
                    
                    yield f"data: {json.dumps({'type': 'progress', 'progress': progress, 'message': message})}\n\n"
                    
                    # 強制刷新輸出緩衝區
                    import sys
                    sys.stdout.flush()
            
            yield f"data: {json.dumps({'type': 'progress', 'progress': 95, 'message': '正在計算最終結果...'})}\n\n"
            
            cap.release()
            detector.cleanup()
            
            # 計算結果
            if similarity_scores:
                average_accuracy = sum(similarity_scores) / len(similarity_scores) * 100
                result = {
                    'type': 'result',
                    'average_accuracy': average_accuracy,
                    'total_frames': total_frames,
                    'valid_frames': len(similarity_scores)
                }
                print(f"計算完成: 平均準確度={average_accuracy:.2f}%, 有效幀數={len(similarity_scores)}/{total_frames}")
            else:
                result = {
                    'type': 'result',
                    'average_accuracy': 0,
                    'total_frames': total_frames,
                    'valid_frames': 0
                }
                print("警告: 影片中未檢測到有效的人體姿勢")
            
            yield f"data: {json.dumps(result)}\n\n"
            
        except Exception as e:
            error_msg = f'計算過程發生錯誤: {str(e)}'
            print(f"錯誤: {error_msg}")
            yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
    
    # 設置適當的響應頭
    response = Response(generate_progress(), mimetype='text/event-stream')
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'
    return response

if __name__ == '__main__':
    try:
        app.run(debug=True, port=5000)
    except KeyboardInterrupt:
        print("\n程式被中斷，正在清理...")
        cleanup_upload_folder()
        flask_detector.stop()
    finally:
        cleanup_upload_folder()
