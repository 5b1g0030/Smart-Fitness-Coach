import cv2              # 存取攝像頭、讀取和顯示影像、處理影像（如翻轉、加文字等）
import mediapipe as mp  # 進行人體姿勢偵測，提供關鍵點座標、繪製骨架等功能 
import numpy as np      # 數值運算、陣列處理，方便影像或座標資料的計算與操作
import math             # 數學函數，用於角度計算
import json             # 用於儲存和讀取標準動作資料
import os               # 用於檔案和路徑操作
from scipy.spatial.distance import euclidean
from dtw import dtw  # 需要安裝: pip install dtw-python

# ===== 標準深蹲動作分析器(類別) =====
# 初始化標準動作分析器(__init__)
# 提取關鍵角度；返回包含關鍵角度的字典(extract_key_angles)
# 計算角度的函數(calculate_angle)
# 計算兩點連線與垂直線的夾角(calculate_vertical_angle)
# 計算膝蓋內扣程度(calculate_knee_valgus) 
# 分析標準深蹲影片，提取動作序列(analyze_standard_video)
# 儲存標準動作序列到文件(save_standard_sequence)
# 從文件載入標準動作序列(load_standard_sequence)
# 清理資源(cleanup)
# ===================================   
class StandardSquatAnalyzer:
    
    # ===== 初始化標準動作分析器 =====
    # 初始化 MediaPipe
    # 設定姿勢檢測: 最小檢測信心度 & 最小追蹤信心度
    # 初始化標準動作序列
    # ===============================  
    def __init__(self):
        # ----- 初始化 MediaPipe -----
        # mp_pose => 姿勢偵測模組
        # mp_drawing => 繪圖工具模組
        # ---------------------------- 
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # ----- 設定姿勢檢測 -----
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,  # 最小檢測信心度
            min_tracking_confidence=0.7    # 最小追蹤信心度
        )
        
        # 儲存標準動作序列
        self.standard_sequence = []
    
    # ===== 提取關鍵角度；返回包含關鍵角度的字典 =====
    # 1. 獲取關鍵點
    # 2. 轉換座標(關鍵點 => 實際像素位置)
    # 3. 計算關鍵角度
    # ============================================= 
    def extract_key_angles(self, landmarks, frame_width, frame_height):
        
        try:
            # ----- 轉換座標 -----
            # 關鍵點座標（0~1）轉換成影像上的實際像素位置，方便後續角度計算或繪圖
            # ------------------- 
            def get_coords(landmark):
                return [landmark.x * frame_width, landmark.y * frame_height]
            
            # ----- 獲取關鍵點 -----
            # 呼叫「轉換座標函式」
            # --------------------- 
            left_shoulder = get_coords(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value])
            right_shoulder = get_coords(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
            left_hip = get_coords(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value])
            right_hip = get_coords(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value])
            left_knee = get_coords(landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value])
            right_knee = get_coords(landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value])
            left_ankle = get_coords(landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value])
            right_ankle = get_coords(landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value])
            
            # ----- 計算關鍵角度 -----
            # 這個字典會被用來分析每一幀的深蹲動作品質，並累積成「標準動作序列」或即時比對用的資料
            # ----------------------- 
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
        # 例外處理 
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
    
    # ===== 計算兩點連線與垂直線的夾角 =====
    # 計算兩點的連線與垂直線的夾角 
    # EX: 用來判斷身體前傾角度，分析深蹲時身體是否過度前傾
    # ==================================== 
    def calculate_vertical_angle(self, point1, point2):
        dx = point2[0] - point1[0] # x 座標
        dy = point2[1] - point1[1] # y 座標
        # ----- 計算身體前傾程度 -----
        # np.arctan2(dx, dy)：計算連線與垂直方向（y軸）的夾角（單位：弧度）。
        # * 180.0 / np.pi：將弧度轉換為角度（度數）。
        # np.abs(...)：取絕對值，確保角度為正。
        # --------------------------- 
        angle = np.abs(np.arctan2(dx, dy) * 180.0 / np.pi)

        return angle # 回傳「兩點連線與垂直線的夾角度數」
    
    # ===== 計算膝蓋內扣程度 =====
    # 膝蓋距離/臀部距離，比例越小代表膝蓋越內扣
    # ==========================  
    def calculate_knee_valgus(self, left_hip, left_knee, right_hip, right_knee):
        # 計算膝蓋間距與臀部間距的比例
        knee_distance = abs(right_knee[0] - left_knee[0])   # 左右膝蓋的 x 座標距離
        hip_distance = abs(right_hip[0] - left_hip[0])      # 左右臀部的 x 座標距離
        
        # 檢查左右臀部的 x 座標距離是否大於 0，確保不會除以零
        if hip_distance > 0:
            valgus_ratio = knee_distance / hip_distance # 膝蓋距離/臀部距離，比例越小代表膝蓋越內扣
        else:
            valgus_ratio = 1.0 # 避免除以零，預設為 1.0
            
        return valgus_ratio # 回傳「膝蓋內扣比例」
    
    # ===== 分析標準深蹲影片，提取動作序列 =====
    # 1. 檢查影片是否能正確讀取
    # 2. 讀取影片
    # 3. 轉換顏色格式
    # 4. 進行姿勢檢測
    # 5. 檢查是否偵測到人體
    # 6. 儲存標準序列
    # ======================================= 
    def analyze_standard_video(self, video_path):

        print(f"正在分析標準影片: {video_path}")
        
        # ----- 檢查影片是否能正確讀取 -----
        cap = cv2.VideoCapture(video_path) # 取指定路徑的影片檔案
        # 檢查影片是否成功開啟
        if not cap.isOpened():
            raise Exception(f"無法開啟影片文件: {video_path}") # 丟出例外（Exception），並顯示錯誤訊息
        
        frame_count = 0 # 初始使化影格計數器，從 0 開始 
        sequence = [] # 儲存每一幀分析得到的標準動作角度資料
        
        while cap.isOpened():
            # ----- 讀取影片 -----
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # ----- 轉換顏色格式 -----
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # ----- 進行姿勢檢測 -----
            results = self.pose.process(rgb_frame)
            
            # ----- 檢查是否偵測到人體 -----
            if results.pose_landmarks:
                # 讀取影像
                h, w, _ = frame.shape
                # 根據偵測到的關鍵點座標計算各種關鍵角度（如膝蓋、髖關節、身體前傾等）
                angles = self.extract_key_angles(results.pose_landmarks.landmark, w, h) # 呼叫「」
                
                if angles:
                    sequence.append(angles)
                    
                    # ----- 顯示分析進度（可選） -----
                    if frame_count % 10 == 0:
                        print(f"已處理 {frame_count} 幀")
        
        cap.release()
        
        if not sequence:
            raise Exception("影片中未檢測到有效的人體姿勢")
        
        # ----- 儲存標準序列 -----
        self.standard_sequence = sequence
        print(f"標準動作分析完成，共 {len(sequence)} 幀")
        # 儲存到文件（可選）
        self.save_standard_sequence("standard_squat_sequence.json")
        
        return sequence
    
    # ===== 儲存標準動作序列到文件 =====
    # 1. 以JSON格式寫入檔案
    # 2. 例外處理
    # ================================   
    def save_standard_sequence(self, filename): 
        try:
            # 以JSON格式寫入檔案
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.standard_sequence, f, indent=2, ensure_ascii=False)
            print(f"標準動作序列已儲存到: {filename}")
        # 例外處理
        except Exception as e: 
            print(f"儲存標準序列時發生錯誤: {e}")
    
    # ===== 從文件載入標準動作序列 =====
    # 1. 載入標準序列
    # 2. 例外處理
    # ================================  
    def load_standard_sequence(self, filename):
        
        try:
            # 載入標準序列: 正確動作的角度變化流程
            with open(filename, 'r', encoding='utf-8') as f:
                self.standard_sequence = json.load(f)
            print(f"標準動作序列已從 {filename} 載入，共 {len(self.standard_sequence)} 幀")
            return True
            # 例外處理
        except Exception as e:
            print(f"載入標準序列時發生錯誤: {e}")
            return False
    
    # ===== 清理資源 =====
    def cleanup(self):
        self.pose.close() # 關閉 MediaPipe 的姿勢偵測器

# ===== 結合標準動作的深蹲檢測器(類別) =====
class SquatDetectorWithStandard:
    
    # ===== 初始化檢測器 =====
    # 1. 設定基礎參數
    # 2. 設定計數器
    # 3. 設定當前動作序列記錄
    # 4. 初始化 MediaPipe
    # 5. 設定姿勢檢測
    # 6. 初始化標準動作分析器
    # ======================= 
    def __init__(self, standard_sequence, squat_threshold=120, similarity_threshold=0.8):
        
        # ----- 設定基礎參數 -----
        self.standard_sequence = standard_sequence          # 標準動作序列
        self.squat_threshold = squat_threshold              # 深蹲判斷的膝蓋角度閾值
        self.similarity_threshold = similarity_threshold    # 與標準動作的相似度閾值
        
        # ----- 設定計數器 -----
        self.squat_count = 0            # 偵測到的深蹲動作總數，從 0 開始
        self.correct_squat_count = 0    # 動作標準（相似度達標）的深蹲次數，從 0 開始
        self.in_squat = False           # 是否處於深蹲姿勢，預設為「否」
        
        # ----- 設定當前動作序列記錄 -----
        self.current_sequence = []
        self.max_sequence_length = 60  # 最大記錄幀數
        
        # ----- 初始化 MediaPipe -----
        # mp_pose => 姿勢偵測模組
        # mp_drawing => 繪圖工具模組
        # ----------------------------
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # ----- 設定姿勢檢測 -----
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,  # 最小檢測信心度
            min_tracking_confidence=0.5    # 最小追蹤信心度
        )
        
        # ----- 初始化標準動作分析器 -----
        # 用於角度計算
        # ------------------------------  
        self.analyzer = StandardSquatAnalyzer() # 引入「標準深蹲動作分析器(類別)」
    
    # ===== 比較當前姿勢與標準動作的相似度 =====
    # 1. 檢查「標準動作序列&目前角度資料」是否有資料
    # 2. 記錄當前動作序列
    # 3. 限制序列長度
    # 4. 檢查序列是否太短
    # 5. 提取關鍵角度序列進行比較
    # 6. 讀取當前序列的角度
    # 7. 檢查「標準角度序列&目前角度序列」是否有資料
    # 8.  
    def compare_with_standard(self, current_angles):
        
        # ----- 檢查「標準動作序列&目前角度資料」是否有資料 -----
        if not self.standard_sequence or not current_angles:
            return 0.0, # 回傳「相似度 0.0」，代表無資料
        
        # ----- 記錄當前動作序列 -----
        # 用來記錄最近一段時間（多幀）的動作角度序列
        # ---------------------------  
        self.current_sequence.append(current_angles)
        
        # ----- 限制序列長度 -----
        # 如果超過最大長度，就移除最前面（最舊）的一幀資料，確保只保留最近一段時間的動作序列
        # 讓比對只針對「最近的動作」，避免序列過長造成記憶體浪費或比對不準確
        # ----------------------- 
        if len(self.current_sequence) > self.max_sequence_length:
            self.current_sequence.pop(0)
        
        # ----- 檢查序列是否太短 -----
        # 剛開始偵測時，self.current_sequence 只累積了幾幀（少於 10 幀），還沒記錄到足夠的動作過程。
        # 使用者動作太快或剛進入深蹲狀態，還沒累積到 10 幀資料
        # -------------------------------- 
        if len(self.current_sequence) < 10:
            return 0.5, # 0.5 是暫時性、保留的相似度，避免誤判為完全錯誤
        
        # 使用 DTW 比較角度序列
        try:
            # ----- 提取關鍵角度序列進行比較 -----
            key_angles = ['left_knee_angle', 'right_knee_angle', 'left_hip_angle', 'right_hip_angle']
            
            similarities = [] # 儲存「每種關鍵角度」與標準動作比對後的相似度分數

            for angle_type in key_angles:
                # ----- 讀取當前序列的角度 -----
                # 將最近一段時間的某種角度（如左膝蓋角度）提取出來，形成一個序列
                # ------------------------- 
                current_angle_seq = [frame.get(angle_type, 0) for frame in self.current_sequence]
                
                # ----- 標準序列的角度（取平均長度的片段） -----
                # DTW（動態時間校正）概念說明(與實際有差別): 
                # 把標準影片的撥放速度加速或放慢到接近使用者做動作的速度再去比較相似度
                # 而 DTW 的實際做法，會把你做的每個動作（每一幀的角度）和標準影片裡最接近的動作配對
                # ------------------------------------------- 
                if len(self.standard_sequence) > len(current_angle_seq): # 如果標準動作序列 > 目前動作序列
                    # 從標準序列中選取相同長度的片段
                    start_idx = len(self.standard_sequence) // 4  # 片段起始位置: 設在標準序列的 1/4 處
                    end_idx = start_idx + len(current_angle_seq)  # 片段結束位置: 確保片段長度和目前動作序列一致
                    # 標準序列的片段中，提取指定角度（如左膝蓋角度）形成一個序列
                    standard_angle_seq = [frame.get(angle_type, 0) 
                                        for frame in self.standard_sequence[start_idx:end_idx]]
                
                else: # 如果標準動作序列 > 目前動作序列，則直接用全部標準序列的角度資料
                    standard_angle_seq = [frame.get(angle_type, 0) for frame in self.standard_sequence]
                
                # ----- 檢查「標準角度序列&目前角度序列」是否有資料 -----
                if standard_angle_seq and current_angle_seq:
                    # --- 計算 DTW 距離 ---
                    # 比對使用者的動作和標準動作的角度變化，計算兩個序列的「距離」。（距離越小，代表越相似）
                    # --------------------
                    alignment = dtw(np.array(current_angle_seq), np.array(standard_angle_seq))
                    normalized_distance = alignment.distance / max(len(current_angle_seq), len(standard_angle_seq))
                    
                    # --- 轉換為相似度 (0-1) ---
                    # 把距離轉換成「相似度分數」（0~1），距離越小，相似度越高
                    # 如果距離很大，相似度最低為 0
                    # ------------------------- 
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

# ===== 分析標準深蹲影片模式 =====
# 1. 輸入影片
# 2. 檢查影片是否存在
# 3. 分析標準影片
# ============================== 
def analyze_standard_video_mode():
    
    print("\n=== 分析標準深蹲影片模式 ===")
    
    # ----- 輸入影片 -----
    # strip() => 去除使用者輸入內容前後的空白或換行，確保取得乾淨的路徑字串
    # ------------------- 
    video_path = input("請輸入標準深蹲影片路徑: ").strip()
    if not video_path:
        print("未提供影片路徑，返回主選單")
        return None
    
    # ----- 檢查影片是否存在 -----
    if not os.path.exists(video_path):
        print(f"影片文件不存在: {video_path}")
        return None
    
    try:
        # ----- 分析標準影片 -----
        analyzer = StandardSquatAnalyzer() # 引入「標準深蹲動作分析器(類別)」
        standard_sequence = analyzer.analyze_standard_video(video_path) # 呼叫「提取關鍵角度；返回包含關鍵角度的字典函式」
        analyzer.cleanup() # 呼叫「清理資源函式」
        
        print("標準影片分析完成！")
        return standard_sequence # 回傳「標準動作序列」
    # 例外處理 
    except Exception as e:
        print(f"分析標準影片時發生錯誤: {e}")
        return None

# ===== 測試影片分析模式 =====
# 1. 檢查有沒有標準動作資料 
# 2. 輸入影片
# 3. 檢查影片是否存在
# 4. 初始化影片讀取
# 5. 檢查影片是否能讀取
# 6. 初始化檢測器
# 7. 獲取影片資訊
# 8. 主要分析迴圈(...)
# =========================== 
def test_video_analysis_mode(standard_sequence):
    
    print("\n=== 測試影片分析模式 ===")
    
    # ----- 檢查有沒有標準動作資料 -----
    if not standard_sequence:
        print("錯誤：沒有標準動作資料，請先分析標準影片或載入標準動作資料")
        return
    
    # ----- 輸入影片 -----
    # strip() => 去除使用者輸入內容前後的空白或換行，確保取得乾淨的路徑字串
    # ------------------- 
    video_path = input("請輸入測試影片路徑: ").strip()
    if not video_path:
        print("未提供影片路徑，返回主選單")
        return
    
    # ----- 檢查影片是否存在 -----
    if not os.path.exists(video_path):
        print(f"影片文件不存在: {video_path}")
        return
    
    # ----- 初始化影片讀取 -----
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"無法開啟測試影片: {video_path}")
        return
    
    # ----- 初始化檢測器 -----
    detector = SquatDetectorWithStandard(
        standard_sequence=standard_sequence,
        squat_threshold=120,
        similarity_threshold=0.6
    )
    
    # 文字提示
    print("\n使用說明：")
    print("- 程式將分析測試影片中的深蹲動作")
    print("- 會比較測試動作與標準動作的相似度")
    print("- 按 'q' 鍵退出分析")
    print("- 按 'r' 鍵重設計數")
    print("- 按 'p' 鍵暫停/繼續播放")
    print("- 按空白鍵暫停/繼續播放")
    print("-" * 50)
    
    # ----- 獲取影片資訊 -----
    fps = cap.get(cv2.CAP_PROP_FPS) # 影片幀數
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # 影片總幀數
    duration = total_frames / fps if fps > 0 else 0 # 計算影片總時長（秒）
    
    # 顯示影片資訊
    print(f"影片資訊：")
    print(f"- FPS: {fps:.2f}")
    print(f"- 總幀數: {total_frames}")
    print(f"- 時長: {duration:.2f}秒")
    print("-" * 50)
    
    frame_count = 0 # 初始化影格計時數器，已處理的影片影格數，從 0 開始
    paused = False # 初始化暫停狀態，預設「未暫停」
    
    try:
        # ----- 主要分析迴圈 -----
        # detector => 深蹲檢測器實例
        # cap => 影片物件
        # 流程:
        # 1. 讀取影片畫面
        # 2. 檢查有無讀取到畫面
        # 3. 處理影格
        # 4. 顯示畫面和進度
        # 5. 處理按鍵事件
        # 6. 釋放資源並顯示結果 
        # ------------------------
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
                
                # ----- 處理影格（影片模式不需要鏡像） -----
                processed_frame = detector.process_frame(frame, mirror=False)
                
                # ----- 在畫面上顯示進度資訊 -----
                progress = frame_count / total_frames * 100 if total_frames > 0 else 0
                current_time = frame_count / fps if fps > 0 else 0
                
                # 顯示目前分析進度（百分比和秒數），方便使用者了解影片播放狀態
                cv2.putText(processed_frame, f"Progress: {progress:.1f}% ({current_time:.1f}s/{duration:.1f}s)", 
                           (10, processed_frame.shape[0] - 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (131,131,131), 2)
                
                # 顯示的提示文字，告訴使用者如何暫停或退出影片分析
                cv2.putText(processed_frame, "Press 'p' or SPACE to pause, 'q' to quit", 
                           (10, processed_frame.shape[0] - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (131,131,131), 2)
            
            # ----- 縮放畫面 -----
            # None => 目標尺寸（dsize），設為 None 代表用比例縮放，不直接指定寬高
            # fx => 水平方向縮放比例(寬度)
            # fy => 垂直方向縮放比例(高度)
            # ------------------- 
            scale = 0.6 # 畫面比例(原本的60%)
            processed_frame = cv2.resize(processed_frame, None, fx=scale, fy=scale)
            cv2.namedWindow('Video Squat Analysis', cv2.WINDOW_NORMAL)  # 讓視窗可調整大小
            
            # ----- 顯示畫面 -----
            # 視窗名稱、影像
            # -------------------
            cv2.imshow('Video Squat Analysis', processed_frame)
            
            # ----- 處理按鍵事件 -----
            # fps => 影片每秒幀數
            # if fps > 0 and not paused => 只有在影片有 fps 且沒暫停時，才用正常速度播放
            # else 1：如果暫停或 fps 不正確，則每次只等待 1 毫秒（讓程式能即時處理按鍵）
            # ----------------------- 
            wait_time = int(1000 / fps) if fps > 0 and not paused else 1

            # ----- 等待使用者按鍵 -----
            # wait_time => 暫停指定毫秒數 
            # & 0xFF => 只取按鍵的低 8 位元（確保跨平台一致）
            # ------------------------- 
            key = cv2.waitKey(wait_time) & 0xFF
            
            if key == ord('q'): # 按 'q' 鍵退出
                print("使用者按下 'q' 鍵，退出影片分析")
                break
            elif key == ord('r'): # 按 'r' 鍵重設計數
                detector.reset_counters()
            elif key == ord('p') or key == ord(' '): # 按 'p' 鍵或空白鍵暫停/繼續
                paused = not paused
                if paused:
                    print("影片已暫停，按 'p' 或空白鍵繼續")
                else:
                    print("影片繼續播放")
    
    finally:
        # ------ 釋放資源並顯示結果 -----
        print(f"影片分析結束")
        print(f"總深蹲次數: {detector.squat_count}")
        print(f"正確深蹲次數: {detector.correct_squat_count}")
        if detector.squat_count > 0:
            accuracy = detector.correct_squat_count / detector.squat_count * 100
            print(f"正確率: {accuracy:.1f}%")
        
        cap.release() # 釋放影片資源
        detector.cleanup() # 關閉 MediaPipe 的姿勢偵測器，釋放相關資源
        cv2.destroyAllWindows() # 關閉所有由 OpenCV 開啟的視窗，清理顯示資源

# ===== 載入現有標準動作資料模式 =====
# 1. 輸入標準動作資料檔案路徑
# 2. 檢查檔案是否存在
# 3. 檢查標準序列檔案是否正常讀取
# ================================== 
def load_standard_sequence_mode():

    print("\n=== 載入標準動作資料模式 ===")
    
    # ----- 輸入標準動作資料檔案路徑 -----
    # 預設 standard_squat_sequence.json
    # ---------------------------------- 
    filename = input("請輸入標準動作資料檔案路徑 (預設: standard_squat_sequence.json): ").strip()
    if not filename: # 如果使用者沒有輸入，則使用預設路徑
        filename = "standard_squat_sequence.json"
    
    # ----- 檢查檔案是否存在 -----
    if not os.path.exists(filename):
        print(f"檔案不存在: {filename}")
        return None
    
    try:
        analyzer = StandardSquatAnalyzer() # 引入「標準深蹲動作分析器(類別)」
        # ----- 檢查標準序列檔案是否正常讀取 -----
        if analyzer.load_standard_sequence(filename): # 呼叫「從文件載入標準動作序列函式」
            standard_sequence = analyzer.standard_sequence # 取得標準動作序列資料
            analyzer.cleanup() # 呼叫「清理資源」
            return standard_sequence # 回傳「標準動作序列」
        else:
            return None # 如果載入失敗，則回傳 None，表示沒有取得標準動作資料
    # 例外處理
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
                
                camera_detection_mode(standard_sequence) # 呼叫「攝像頭即時檢測模式函式」

            # 2. 分析標準深蹲影片   
            elif choice == '2':
                result = analyze_standard_video_mode() # 呼叫「分析標準深蹲影片模式函式」
                # 檢查分析標準影片或載入標準資料是否成功
                if result:
                    standard_sequence = result # 如果成功，將分析或載入得到的標準動作序列資料儲存
                    print("標準動作資料已更新，現在可以使用其他功能了！")
                
            # 3. 輸入影片進行分析
            elif choice == '3':
                # 測試影片分析
                test_video_analysis_mode(standard_sequence) # 呼叫「測試影片分析模式函式」
                
            # 4. 載入現有的標準動作資料
            elif choice == '4':
                result = load_standard_sequence_mode() # 呼叫「載入現有標準動作資料模式函式」
                # 檢查分析標準影片或載入標準資料是否成功
                if result:
                    standard_sequence = result # 如果成功，將分析或載入得到的標準動作序列資料儲存
                    print("標準動作資料載入成功，現在可以使用其他功能了！")
            
            # 5. 退出程式
            elif choice == '5':
                print("感謝使用深蹲姿勢檢測程式！")
                break # 跳出無限迴圈
            
            # 例外輸入處理  
            else:
                print("無效選擇，請輸入 1-5 的數字")
        # 例外: 使用者在終端機按下 Ctrl+C（或其他中斷鍵），主動中斷程式執行        
        except KeyboardInterrupt:
            print("\n\n程式被使用者中斷")
            break
        # 程式執行過程中發生任何未預期的錯誤
        except Exception as e:
            print(f"\n程式執行時發生錯誤: {e}")
            print("請重新選擇功能或聯絡開發者")
    
    # ===== 程式結束清理 =====
    cv2.destroyAllWindows() # 關閉所有可能還開啟的 OpenCV 視窗


if __name__ == '__main__':
    main()