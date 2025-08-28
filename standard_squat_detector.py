import cv2              # 存取攝像頭、讀取和顯示影像、處理影像（如翻轉、加文字等）
import mediapipe as mp  # 進行人體姿勢偵測，提供關鍵點座標、繪製骨架等功能 
import numpy as np      # 數值運算、陣列處理，方便影像或座標資料的計算與操作
import math             # 數學函數，用於角度計算
import json             # 用於儲存和讀取標準動作資料
import os               # 用於檔案和路徑操作
from scipy.spatial.distance import euclidean
from dtw import dtw  # 需要安裝: pip install dtw-python

class StandardSquatAnalyzer:
    """標準深蹲動作分析器"""
    
    def __init__(self):
        """初始化標準動作分析器"""
        # ===== 初始化 MediaPipe =====
        # mp_pose => 姿勢偵測模組
        # mp_drawing => 繪圖工具模組
        # ============================ 
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # ===== 設定姿勢檢測 =====
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,  # 最小檢測信心度
            min_tracking_confidence=0.7    # 最小追蹤信心度
        )
        
        # 儲存標準動作序列
        self.standard_sequence = []
        
    def extract_key_angles(self, landmarks, frame_width, frame_height):
        """
        提取關鍵角度
        返回包含關鍵角度的字典
        """
        try:
            # 轉換座標
            def get_coords(landmark):
                return [landmark.x * frame_width, landmark.y * frame_height]
            
            # 獲取關鍵點
            left_shoulder = get_coords(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value])
            right_shoulder = get_coords(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
            left_hip = get_coords(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value])
            right_hip = get_coords(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value])
            left_knee = get_coords(landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value])
            right_knee = get_coords(landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value])
            left_ankle = get_coords(landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value])
            right_ankle = get_coords(landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value])
            
            # 計算關鍵角度
            angles = {
                'left_knee_angle': self.calculate_angle(left_hip, left_knee, left_ankle),
                'right_knee_angle': self.calculate_angle(right_hip, right_knee, right_ankle),
                'left_hip_angle': self.calculate_angle(left_shoulder, left_hip, left_knee),
                'right_hip_angle': self.calculate_angle(right_shoulder, right_hip, right_knee),
                # 身體前傾角度（肩膀到臀部的角度）
                'body_lean': self.calculate_vertical_angle(
                    [(left_shoulder[0] + right_shoulder[0])/2, (left_shoulder[1] + right_shoulder[1])/2],
                    [(left_hip[0] + right_hip[0])/2, (left_hip[1] + right_hip[1])/2]
                ),
                # 膝蓋內扣檢測（膝蓋與臀部的 x 座標比較）
                'knee_valgus': self.calculate_knee_valgus(left_hip, left_knee, right_hip, right_knee)
            }
            
            return angles
            
        except Exception as e:
            print(f"提取角度時發生錯誤: {e}")
            return None
    
    # ===== 計算角度的函數 =====
    # 計算三個點形成的角度
    # a => 第一個點座標 (x, y)
    # b => 頂點座標 (x, y) 
    # c => 第三個點座標 (x, y)
    # 返回角度值（度數）
    # =========================  
    def calculate_angle(self, a, b, c):
        # 將座標轉換為numpy陣列
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        # 計算向量
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        # 確保角度在0-180度之間
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
    
    def calculate_vertical_angle(self, point1, point2):
        """計算兩點連線與垂直線的夾角"""
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        angle = np.abs(np.arctan2(dx, dy) * 180.0 / np.pi)
        return angle
    
    def calculate_knee_valgus(self, left_hip, left_knee, right_hip, right_knee):
        """計算膝蓋內扣程度"""
        # 計算膝蓋間距與臀部間距的比例
        knee_distance = abs(right_knee[0] - left_knee[0])
        hip_distance = abs(right_hip[0] - left_hip[0])
        
        if hip_distance > 0:
            valgus_ratio = knee_distance / hip_distance
        else:
            valgus_ratio = 1.0
            
        return valgus_ratio
    
    def analyze_standard_video(self, video_path):
        """
        分析標準深蹲影片，提取動作序列
        """
        print(f"正在分析標準影片: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"無法開啟影片文件: {video_path}")
        
        frame_count = 0
        sequence = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # 轉換顏色格式
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 進行姿勢檢測
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                h, w, _ = frame.shape
                angles = self.extract_key_angles(results.pose_landmarks.landmark, w, h)
                
                if angles:
                    sequence.append(angles)
                    
                    # 顯示分析進度（可選）
                    if frame_count % 10 == 0:
                        print(f"已處理 {frame_count} 幀")
        
        cap.release()
        
        if not sequence:
            raise Exception("影片中未檢測到有效的人體姿勢")
        
        # 儲存標準序列
        self.standard_sequence = sequence
        print(f"標準動作分析完成，共 {len(sequence)} 幀")
        
        # 儲存到文件（可選）
        self.save_standard_sequence("standard_squat_sequence.json")
        
        return sequence
    
    def save_standard_sequence(self, filename):
        """儲存標準動作序列到文件"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.standard_sequence, f, indent=2, ensure_ascii=False)
            print(f"標準動作序列已儲存到: {filename}")
        except Exception as e:
            print(f"儲存標準序列時發生錯誤: {e}")
    
    def load_standard_sequence(self, filename):
        """從文件載入標準動作序列"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self.standard_sequence = json.load(f)
            print(f"標準動作序列已從 {filename} 載入，共 {len(self.standard_sequence)} 幀")
            return True
        except Exception as e:
            print(f"載入標準序列時發生錯誤: {e}")
            return False
    
    def cleanup(self):
        """清理資源"""
        self.pose.close()


class SquatDetectorWithStandard:
    """結合標準動作的深蹲檢測器"""
    
    def __init__(self, standard_sequence, squat_threshold=120, similarity_threshold=0.8):
        """
        初始化檢測器
        
        Args:
            standard_sequence: 標準動作序列
            squat_threshold: 深蹲判斷的膝蓋角度閾值
            similarity_threshold: 與標準動作的相似度閾值
        """
        self.standard_sequence = standard_sequence
        self.squat_threshold = squat_threshold
        self.similarity_threshold = similarity_threshold
        
        # 計數器
        self.squat_count = 0
        self.correct_squat_count = 0
        self.in_squat = False
        
        # 當前動作序列記錄
        self.current_sequence = []
        self.max_sequence_length = 60  # 最大記錄幀數
        
        # ===== 初始化 MediaPipe =====
        # mp_pose => 姿勢偵測模組
        # mp_drawing => 繪圖工具模組
        # ============================
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # ===== 設定姿勢檢測 =====
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,  # 最小檢測信心度
            min_tracking_confidence=0.5    # 最小追蹤信心度
        )
        
        # 初始化標準動作分析器（用於角度計算）
        self.analyzer = StandardSquatAnalyzer()
    
    def compare_with_standard(self, current_angles):
        """
        比較當前姿勢與標準動作的相似度
        """
        if not self.standard_sequence or not current_angles:
            return 0.0, "無標準資料"
        
        # 記錄當前動作序列
        self.current_sequence.append(current_angles)
        
        # 限制序列長度
        if len(self.current_sequence) > self.max_sequence_length:
            self.current_sequence.pop(0)
        
        # 如果序列太短，無法比較
        if len(self.current_sequence) < 10:
            return 0.5, "動作序列太短"
        
        # 使用 DTW 比較角度序列
        try:
            # 提取關鍵角度序列進行比較
            key_angles = ['left_knee_angle', 'right_knee_angle', 'left_hip_angle', 'right_hip_angle']
            
            similarities = []
            for angle_type in key_angles:
                # 當前序列的角度
                current_angle_seq = [frame.get(angle_type, 0) for frame in self.current_sequence]
                
                # 標準序列的角度（取平均長度的片段）
                if len(self.standard_sequence) > len(current_angle_seq):
                    # 從標準序列中選取相應長度的片段
                    start_idx = len(self.standard_sequence) // 4  # 從1/4處開始
                    end_idx = start_idx + len(current_angle_seq)
                    standard_angle_seq = [frame.get(angle_type, 0) 
                                        for frame in self.standard_sequence[start_idx:end_idx]]
                else:
                    standard_angle_seq = [frame.get(angle_type, 0) for frame in self.standard_sequence]
                
                if standard_angle_seq and current_angle_seq:
                    # 計算 DTW 距離
                    alignment = dtw(np.array(current_angle_seq), np.array(standard_angle_seq))
                    normalized_distance = alignment.distance / max(len(current_angle_seq), len(standard_angle_seq))
                    
                    # 轉換為相似度 (0-1)
                    similarity = max(0, 1 - normalized_distance / 100)  # 調整歸一化係數
                    similarities.append(similarity)
            
            if similarities:
                avg_similarity = np.mean(similarities)
                return avg_similarity, self.get_feedback(current_angles, avg_similarity)
            else:
                return 0.0, "計算相似度失敗"
                
        except Exception as e:
            print(f"DTW 比較時發生錯誤: {e}")
            return 0.0, f"比較錯誤: {str(e)}"
    
    def get_feedback(self, angles, similarity):
        """根據角度和相似度生成回饋"""
        feedback_parts = []
        
        # 檢查深度
        left_knee = angles.get('left_knee_angle', 180)
        right_knee = angles.get('right_knee_angle', 180)
        if left_knee > 100 or right_knee > 100:
            feedback_parts.append("深度不足")
        
        # 檢查膝蓋內扣
        knee_valgus = angles.get('knee_valgus', 1.0)
        if knee_valgus < 0.8:  # 膝蓋過於內扣
            feedback_parts.append("膝蓋內扣")
        
        # 檢查身體前傾
        body_lean = angles.get('body_lean', 0)
        if body_lean > 20:  # 過度前傾
            feedback_parts.append("身體過於前傾")
        
        # 根據相似度給出總體評價
        if similarity > 0.8:
            overall = "動作標準"
        elif similarity > 0.6:
            overall = "動作良好"
        elif similarity > 0.4:
            overall = "需要改善"
        else:
            overall = "動作不正確"
        
        if feedback_parts:
            return f"{overall}: {', '.join(feedback_parts)}"
        else:
            return overall
    
    def is_squat_pose(self, landmarks, frame_width, frame_height):
        """判斷是否為深蹲姿勢並分析品質"""
        # 提取角度
        angles = self.analyzer.extract_key_angles(landmarks, frame_width, frame_height)
        if not angles:
            return False, 0, 0, 0.0, "無法分析"
        
        # 基本深蹲判斷
        left_knee_angle = angles['left_knee_angle']
        right_knee_angle = angles['right_knee_angle']
        
        is_squat = (left_knee_angle < self.squat_threshold and 
                   right_knee_angle < self.squat_threshold)
        
        # 與標準動作比較
        similarity, feedback = self.compare_with_standard(angles)
        
        return is_squat, left_knee_angle, right_knee_angle, similarity, feedback
    
    # ===== 更新深蹲計數 =====
    # is_squat => 當前是否為深蹲姿勢
    # similarity => 與標準動作的相似度
    # 避免重複計數的邏輯
    # ========================
    def update_squat_count(self, is_squat, similarity):
        """更新深蹲計數"""
        if is_squat and not self.in_squat:
            # 從非深蹲狀態進入深蹲狀態，計數+1
            self.squat_count += 1
            self.in_squat = True
            
            # 如果相似度夠高，記為正確深蹲
            if similarity > self.similarity_threshold:
                self.correct_squat_count += 1
                
        elif not is_squat and self.in_squat:
            # 從深蹲狀態回到非深蹲狀態
            self.in_squat = False
            # 清空當前序列，準備記錄下一次動作
            self.current_sequence = []
    
    # ----- 繪製狀態資訊 -----
    # frame => 原始影像
    # is_squat => 是否為深蹲姿勢
    # left_angle, right_angle => 左右膝蓋角度
    # similarity => 與標準動作的相似度
    # feedback => 動作回饋訊息
    # has_pose => 是否檢測到姿勢
    # --------------------------
    def draw_status_info(self, frame, is_squat, left_angle, right_angle, similarity, feedback, has_pose):
        """繪製狀態資訊"""
        h, w, _ = frame.shape # 影像高度、影像寬度、色彩通道數(不會用到)
        
        if has_pose:
            # 顯示姿勢狀態
            if is_squat:
                # 在畫面左上角顯示「深蹲」
                cv2.putText(frame, "squats", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            else:
                # 在畫面左上角顯示「站立」
                cv2.putText(frame, "Standing", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
            
            # 顯示深蹲計數
            cv2.putText(frame, f"Count: {self.squat_count}", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame, f"Correct: {self.correct_squat_count}", (10, 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 顯示相似度
            similarity_color = (0, 255, 0) if similarity > 0.8 else (0, 255, 255) if similarity > 0.6 else (0, 0, 255)
            cv2.putText(frame, f"Similarity: {similarity:.2f}", (10, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, similarity_color, 2)
            
            # 顯示回饋
            cv2.putText(frame, feedback, (10, 220), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 顯示角度資訊
            cv2.putText(frame, f"L: {int(left_angle)}° R: {int(right_angle)}°", (10, 260), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            # 沒有檢測到姿勢
            cv2.putText(frame, "No Pose Detected", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    
    # ===== 處理單一影格 =====
    # frame => 原始影像
    # mirror => 是否鏡像翻轉（攝像頭模式需要，影片模式不需要）
    # 返回 => 處理後的影像
    # 流程:
    # 1. 畫面左右顛倒（可選）
    # 2. 轉換顏色格式
    # 3. 進行姿勢檢測
    # 4. 判斷深蹲姿勢並分析品質
    # 5. 更新計數
    # 6. 繪製資訊
    # ========================
    def process_frame(self, frame, mirror=True):
        """處理單一影格"""
        # ----- 畫面左右顛倒 -----
        # 鏡像效果，符合使用者視角（僅在攝像頭模式需要）
        # 參數補充:
        # 1 => 水平翻轉
        # 0 => 垂直翻轉
        # -1 => 水平垂直翻轉
        # ----------------------- 
        if mirror:
            frame = cv2.flip(frame, 1)
        
        # ----- 轉換顏色格式 -----
        # BGR -> RGB，MediaPipe需要RGB格式
        # -----------------------
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # ----- 進行姿勢檢測 -----
        results = self.pose.process(rgb_frame)
        
        # 預設值
        is_squat, left_angle, right_angle, similarity, feedback = False, 0, 0, 0.0, "無姿勢"
        has_pose = results.pose_landmarks is not None
        
        # ----- 檢查是否檢測到姿勢並分析動作品質 -----
        if has_pose:
            # 取得關鍵點座標 (正規化座標，需要轉換為像素座標)
            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame.shape # 影像高度、影像寬度、色彩通道數(不會用到)
            
            # ----- 判斷深蹲姿勢並分析品質 -----
            is_squat, left_angle, right_angle, similarity, feedback = self.is_squat_pose(landmarks, w, h)
            
            # 更新深蹲計數
            self.update_squat_count(is_squat, similarity)
            
            # ----- 繪製骨架 -----
            # 繪製關鍵點和骨架
            self.mp_drawing.draw_landmarks(
                frame, # 原始影像
                results.pose_landmarks, # 人體關鍵點資料
                self.mp_pose.POSE_CONNECTIONS, # 定義關鍵點之間的連線(骨架)
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2), # 關鍵點樣式
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2) # 連接線樣式
            )
        
        # 繪製狀態資訊
        self.draw_status_info(frame, is_squat, left_angle, right_angle, similarity, feedback, has_pose)
        return frame
    
    def reset_counters(self):
        """重設計數器"""
        self.squat_count = 0
        self.correct_squat_count = 0
        self.in_squat = False
        self.current_sequence = []
        print("計數已重設")
    
    # ===== 清理資源 =====
    def cleanup(self):
        """清理資源"""
        self.pose.close() # 關閉 MediaPipe 的姿勢偵測器，釋放相關資源
        self.analyzer.cleanup()

# ===== 顯示功能選單 ======
# 顯示系統操作提示
# ======================== 
def show_menu():
    print("="*60)
    print("        深蹲姿勢檢測程式 - 功能選單")
    print("="*60)
    print("1. 即時攝像頭檢測 (需要先有標準動作資料)")
    print("2. 分析標準深蹲影片")
    print("3. 測試影片分析")
    print("4. 載入現有的標準動作資料")
    print("5. 退出程式")
    print("="*60)

# ===== 攝像頭即時檢測模式 =====
# 1. 初始化、開啟鏡頭
# 2. 檢查攝像頭是否正常開啟
# 3. 初始化檢測器
# 4. 顯示操作提示
# 5. 主要檢測迴圈(...)
# =============================  
def camera_detection_mode(standard_sequence):
    
    print("\n=== 攝像頭即時檢測模式 ===")
    
    # ----- 初始化攝像頭 -----
    # cv2.VideoCapture(0) => 使用預設後端(有可能使用到不適合的系統)
    # cv2.VideoCapture(0, cv2.【系統參數】) => 可以指定適合的系統 
    # Windows => CAP_DSHOW(推薦), CAP_MSMF
    # macOS => CAP_AVFOUNDATION(推薦)
    # Linux => CAP_V4L2(推薦), CAP_GSTREAMER
    # ----------------------- 
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    # ----- 設定攝像頭解析度 (可選) -----
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 920) # 畫面寬度
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540) # 畫面高度
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) # 用 MJPG 編碼，減少延遲
    
    # ----- 檢查攝像頭是否正常開啟 -----
    if not cap.isOpened():
        raise Exception("無法開啟攝像頭")
    
    # ----- 初始化檢測器 -----
    detector = SquatDetectorWithStandard(
        standard_sequence=standard_sequence,
        squat_threshold=120,
        similarity_threshold=0.6
    )
    
    # ----- 顯示操作提示 -----
    print("\n使用說明：")
    print("- 站在攝像頭前方，確保全身都在畫面內")
    print("- 程式會比較您的動作與標準動作")
    print("- 綠色相似度 > 0.8，黃色 0.6-0.8，紅色 < 0.6")
    print("- 按 'q' 鍵退出程式")
    print("- 按 'r' 鍵重設計數")
    print("-" * 50)
    
    try:
        # ----- 主要檢測迴圈 -----
        # detector => 深蹲檢測器實例
        # cap => 攝像頭物件
        # 流程:
        # 1. 讀取鏡頭畫面
        # 2. 檢查有無讀取到畫面
        # 3. 處理影格
        # 4. 顯示畫面
        # 5. 處理按鍵事件
        # -----------------------
        while cap.isOpened():
            # ----- 讀取攝像頭畫面 -----
            # ret => 布林值，表示是否成功讀取到影像
            # frame => Numpy 陣列，讀取到的影像資料(如果 ret 是 False，frame 會是 None)
            # ------------------------- 
            ret, frame = cap.read()
            
            # ----- 檢查有無讀取到畫面 -----
            if not ret:
                print("無法讀取攝像頭")
                break # 跳出迴圈(不繼續以下流程)
            
            # ----- 處理影格 -----
            # 攝像頭模式需要鏡像
            # ------------------- 
            processed_frame = detector.process_frame(frame, mirror=True)
            
            # ----- 顯示畫面 -----
            # 視窗名稱、影像
            # ------------------- 
            cv2.imshow('Camera Squat Detection', processed_frame)
            
            # ------ 處理按鍵事件 -----
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                # 按 'q' 鍵退出 
                print("使用者按下 'q' 鍵，退出攝像頭模式")
                break
            elif key == ord('r'):
                # 按 'r' 鍵重設計數
                detector.reset_counters()
    
    finally:
        # ===== 釋放資源並顯示結果 =====
        print(f"攝像頭模式結束")
        print(f"總深蹲次數: {detector.squat_count}")
        print(f"正確深蹲次數: {detector.correct_squat_count}")
        if detector.squat_count > 0:
            accuracy = detector.correct_squat_count / detector.squat_count * 100
            print(f"正確率: {accuracy:.1f}%")
        
        cap.release() # 釋放攝像頭資源，讓攝像頭可以被其他程式使用
        detector.cleanup() # 關閉 MediaPipe 的姿勢偵測器，釋放相關資源
        cv2.destroyAllWindows() # 關閉所有由 OpenCV 開啟的視窗，清理顯示資源


def analyze_standard_video_mode():
    """分析標準深蹲影片模式"""
    print("\n=== 分析標準深蹲影片模式 ===")
    
    video_path = input("請輸入標準深蹲影片路徑: ").strip()
    if not video_path:
        print("未提供影片路徑，返回主選單")
        return None
    
    if not os.path.exists(video_path):
        print(f"影片文件不存在: {video_path}")
        return None
    
    try:
        # 分析標準影片
        analyzer = StandardSquatAnalyzer()
        standard_sequence = analyzer.analyze_standard_video(video_path)
        analyzer.cleanup()
        
        print("標準影片分析完成！")
        return standard_sequence
        
    except Exception as e:
        print(f"分析標準影片時發生錯誤: {e}")
        return None


def test_video_analysis_mode(standard_sequence):
    """測試影片分析模式"""
    print("\n=== 測試影片分析模式 ===")
    
    if not standard_sequence:
        print("錯誤：沒有標準動作資料，請先分析標準影片或載入標準動作資料")
        return
    
    video_path = input("請輸入測試影片路徑: ").strip()
    if not video_path:
        print("未提供影片路徑，返回主選單")
        return
    
    if not os.path.exists(video_path):
        print(f"影片文件不存在: {video_path}")
        return
    
    # ===== 初始化影片讀取 =====
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"無法開啟測試影片: {video_path}")
        return
    
    # 初始化檢測器
    detector = SquatDetectorWithStandard(
        standard_sequence=standard_sequence,
        squat_threshold=120,
        similarity_threshold=0.6
    )
    
    print("\n使用說明：")
    print("- 程式將分析測試影片中的深蹲動作")
    print("- 會比較測試動作與標準動作的相似度")
    print("- 按 'q' 鍵退出分析")
    print("- 按 'r' 鍵重設計數")
    print("- 按 'p' 鍵暫停/繼續播放")
    print("- 按空白鍵暫停/繼續播放")
    print("-" * 50)
    
    # 獲取影片資訊
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"影片資訊：")
    print(f"- FPS: {fps:.2f}")
    print(f"- 總幀數: {total_frames}")
    print(f"- 時長: {duration:.2f}秒")
    print("-" * 50)
    
    frame_count = 0
    paused = False
    
    try:
        # ===== 主要分析迴圈 =====
        # detector => 深蹲檢測器實例
        # cap => 影片物件
        # 流程:
        # 1. 讀取影片畫面
        # 2. 檢查有無讀取到畫面
        # 3. 處理影格
        # 4. 顯示畫面和進度
        # 5. 處理按鍵事件
        # ========================
        while cap.isOpened():
            if not paused:
                # ----- 讀取影片畫面 -----
                # ret => 布林值，表示是否成功讀取到影像
                # frame => Numpy 陣列，讀取到的影像資料(如果 ret 是 False，frame 會是 None)
                # -------------------------
                ret, frame = cap.read()
                
                # ----- 檢查有無讀取到畫面 -----
                if not ret:
                    print("影片分析完成")
                    break # 跳出迴圈(不繼續以下流程)
                
                frame_count += 1
                
                # 處理影格（影片模式不需要鏡像）
                processed_frame = detector.process_frame(frame, mirror=False)
                
                # 在畫面上顯示進度資訊
                progress = frame_count / total_frames * 100 if total_frames > 0 else 0
                current_time = frame_count / fps if fps > 0 else 0
                
                cv2.putText(processed_frame, f"Progress: {progress:.1f}% ({current_time:.1f}s/{duration:.1f}s)", 
                           (10, processed_frame.shape[0] - 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.putText(processed_frame, "Press 'p' or SPACE to pause, 'q' to quit", 
                           (10, processed_frame.shape[0] - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # ----- 顯示畫面 -----
            # 視窗名稱、影像
            # -------------------
            cv2.imshow('Video Squat Analysis', processed_frame)
            
            # 處理按鍵事件
            wait_time = int(1000 / fps) if fps > 0 and not paused else 1
            key = cv2.waitKey(wait_time) & 0xFF
            
            if key == ord('q'):
                # 按 'q' 鍵退出
                print("使用者按下 'q' 鍵，退出影片分析")
                break
            elif key == ord('r'):
                # 按 'r' 鍵重設計數
                detector.reset_counters()
            elif key == ord('p') or key == ord(' '):
                # 按 'p' 鍵或空白鍵暫停/繼續
                paused = not paused
                if paused:
                    print("影片已暫停，按 'p' 或空白鍵繼續")
                else:
                    print("影片繼續播放")
    
    finally:
        # ===== 釋放資源並顯示結果 =====
        print(f"影片分析結束")
        print(f"總深蹲次數: {detector.squat_count}")
        print(f"正確深蹲次數: {detector.correct_squat_count}")
        if detector.squat_count > 0:
            accuracy = detector.correct_squat_count / detector.squat_count * 100
            print(f"正確率: {accuracy:.1f}%")
        
        cap.release() # 釋放影片資源
        detector.cleanup() # 關閉 MediaPipe 的姿勢偵測器，釋放相關資源
        cv2.destroyAllWindows() # 關閉所有由 OpenCV 開啟的視窗，清理顯示資源


def load_standard_sequence_mode():
    """載入現有標準動作資料模式"""
    print("\n=== 載入標準動作資料模式 ===")
    
    filename = input("請輸入標準動作資料檔案路徑 (預設: standard_squat_sequence.json): ").strip()
    if not filename:
        filename = "standard_squat_sequence.json"
    
    if not os.path.exists(filename):
        print(f"檔案不存在: {filename}")
        return None
    
    try:
        analyzer = StandardSquatAnalyzer()
        if analyzer.load_standard_sequence(filename):
            standard_sequence = analyzer.standard_sequence
            analyzer.cleanup()
            return standard_sequence
        else:
            return None
    except Exception as e:
        print(f"載入標準序列時發生錯誤: {e}")
        return None

# ===== 主函式 =====
def main():
    
    print("歡迎使用深蹲姿勢檢測程式！")
    
    # 儲存標準動作序列
    standard_sequence = None
    
    while True:
        try:
            show_menu() # 呼叫「顯示功能選單函式」
            choice = input("\n請選擇功能 (1-5): ").strip() # 使用者輸入(去空白)
            
            # 1. 即時攝像頭檢測
            if choice == '1':
                # 檢查「標準動作序列」是否有資料
                if not standard_sequence:
                    print("\n錯誤：沒有標準動作資料！")
                    print("請先選擇功能 2 分析標準影片，或選擇功能 4 載入現有資料")
                    continue
                
                camera_detection_mode(standard_sequence) # 呼叫「」
                
            elif choice == '2':
                # 分析標準深蹲影片
                result = analyze_standard_video_mode()
                if result:
                    standard_sequence = result
                    print("標準動作資料已更新，現在可以使用其他功能了！")
                
            elif choice == '3':
                # 測試影片分析
                test_video_analysis_mode(standard_sequence)
                
            elif choice == '4':
                # 載入現有的標準動作資料
                result = load_standard_sequence_mode()
                if result:
                    standard_sequence = result
                    print("標準動作資料載入成功，現在可以使用其他功能了！")
                
            elif choice == '5':
                # 退出程式
                print("感謝使用深蹲姿勢檢測程式！")
                break
                
            else:
                print("無效選擇，請輸入 1-5 的數字")
                
        except KeyboardInterrupt:
            print("\n\n程式被使用者中斷")
            break
        except Exception as e:
            print(f"\n程式執行時發生錯誤: {e}")
            print("請重新選擇功能或聯絡開發者")
    
    # ===== 程式結束清理 =====
    cv2.destroyAllWindows() # 關閉所有可能還開啟的 OpenCV 視窗


if __name__ == '__main__':
    main()