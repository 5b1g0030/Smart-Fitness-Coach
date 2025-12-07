import cv2              # 存取攝像頭、讀取和顯示影像、處理影像（如翻轉、加文字等）
import mediapipe as mp  # 進行人體姿勢偵測，提供關鍵點座標、繪製骨架等功能 
import numpy as np      # 數值運算、陣列處理，方便影像或座標資料的計算與操作
import json             # 用於儲存和讀取標準動作資料
from dtw import dtw  # 需要安裝: pip install dtw-python
from PIL import Image, ImageDraw, ImageFont

# ===== 標準深蹲動作分析器(類別) =====
# ===================================   
class StandardSquatAnalyzer:
    
    # ===== 初始化標準動作分析器 ===== 
    # 引用類別時自動執行
    # ===============================
    def __init__(self):
        '''
            初始化 MediaPipe
            設定姿勢檢測: 最小檢測信心度 & 最小追蹤信心度
            初始化標準動作序列
        '''
        # ----- 初始化 MediaPipe -----
        self.mp_pose = mp.solutions.pose # 姿勢偵測模組
        self.mp_drawing = mp.solutions.drawing_utils # 繪圖工具模組
        
        # ----- 設定姿勢檢測 ----- 
        # 最小檢測信心度: 模型判定畫面中是否有人體的置信度門檻
        # 最小追蹤信心度: 已建立追蹤後，每幀維持關鍵點追蹤的置信度門檻，低於此值，系統會回到偵測流程以重新定位人體
        # -----------------------
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,  
            min_tracking_confidence=0.7    
        )
        
        # 儲存標準動作序列
        self.standard_sequence = []
    
    
    # ===== 提取關鍵角度；返回包含關鍵角度的字典 =====
    # 輸入 人體關鍵點、畫面寬度、畫面高度 
    # 輸出 關鍵角度&膝蓋間距/臀部間距的比例(字典)
    # =============================================
    def extract_key_angles(self, landmarks, frame_width, frame_height):
        """
            1. 獲取關鍵點
            2. 轉換座標(關鍵點 => 實際像素位置)
            3. 計算關鍵角度
        """
        
        try:
            # ----- 轉換座標 -----
            # 輸入 關鍵點
            # 輸出 此點在畫面上的真實座標
            # -------------------
            def get_coords(landmark):
                """
                    關鍵點座標（0~1）轉換成影像上的實際像素位置，方便後續角度計算或繪圖
                    landmark.x , landmark.y => 正規化座標「比例（0~1）」，形容此物體位於畫面多少%的位置
                    frame_width , frame_height => 畫面解析度「寬 , 高」，整個畫面大小
                    物體位置比例*整個畫面 = 物體在畫面上的實際位置
                """
                return [landmark.x * frame_width, landmark.y * frame_height]
            
            # ----- 計算關節角度的函數 -----
            # 輸入 髖部、膝蓋、腳踝
            # 輸出 角度值（度數）
            # ----------------------------- 
            def calculate_angle(a, b, c):
                # 將座標轉換為numpy陣列
                a = np.array(a) # 髖部（hip）
                b = np.array(b) # 膝蓋（knee） <- 頂點
                c = np.array(c) # 腳踝（ankle）
            
                # ---計算三個點形成的角度---
                # 計算向量
                # np.arctan2(c[1] - b[1], c[0] - b[0]) => 向量 b→c 的角度
                # np.arctan2(a[1] - b[1], a[0] - b[0]) => 向量 b→a 的角度
                # -------------------------
                radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
                angle = np.abs(radians * 180.0 / np.pi)
                
                # 確保角度在0-180度之間
                if angle > 180.0:
                    angle = 360 - angle # 如果大於 180 則減去 360 讓數值保持在 180 以內
                    
                return angle
            
            # ----- 計算兩點連線與垂直線的夾角 -----
            # 輸入 點1、點2
            # 輸出 兩點連線與垂直線的夾角度數
            # ------------------------------------ 
            def calculate_vertical_angle(point1, point2):
                """
                    計算兩點的連線與垂直線的夾角 
                    EX: 用來判斷身體前傾角度，分析深蹲時身體是否過度前
                """

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
            # 輸入 左、右髖部；左、右膝蓋
            # 輸出 膝蓋內扣比例
            # ==========================  
            def calculate_knee_valgus(left_hip, left_knee, right_hip, right_knee):
                """ 膝蓋距離/臀部距離，比例越小代表膝蓋越內扣 """

                # 計算膝蓋間距與臀部間距的比例
                knee_distance = abs(right_knee[0] - left_knee[0])   # 左右膝蓋的 x 座標距離
                hip_distance = abs(right_hip[0] - left_hip[0])      # 左右臀部的 x 座標距離
                
                # 檢查左右臀部的 x 座標距離是否大於 0，確保不會除以零
                if hip_distance > 0:
                    valgus_ratio = knee_distance / hip_distance # 膝蓋距離/臀部距離，比例越小代表膝蓋越內扣
                else:
                    valgus_ratio = 1.0 # 避免除以零，預設為 1.0
                    
                return valgus_ratio # 回傳「膝蓋內扣比例」

            # ----- 獲取關鍵點 -----
            # 呼叫「轉換座標函式」
            # PoseLandmark = 在 pose 模組裡定義的一個列舉（enum），列出所有人體關鍵點的名稱
            # 取得的 landmark 物件包含欄位：x, y, z, visibility 
            # (x,y: 正規化座標（範圍約 0~1）, z: 深度值, visibility: 關鍵點信任度(0~1)) 
            # --------------------- 
            left_shoulder = get_coords(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value])    # 左肩膀
            right_shoulder = get_coords(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value])  # 右肩膀
            left_hip = get_coords(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value])              # 左髖部
            right_hip = get_coords(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value])            # 右髖部
            left_knee = get_coords(landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value])            # 左膝蓋
            right_knee = get_coords(landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value])          # 右膝蓋
            left_ankle = get_coords(landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value])          # 左腳踝 
            right_ankle = get_coords(landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value])        # 右腳踝
            
            # ----- 計算關鍵角度 ----- 
            # 這個字典會被用來分析每一幀的深蹲動作品質，並累積成「標準動作序列」或即時比對用的資料
            # ----------------------- 
            angles = {
                'left_knee_angle': calculate_angle(left_hip, left_knee, left_ankle),       # 左膝蓋角度 
                'right_knee_angle': calculate_angle(right_hip, right_knee, right_ankle),   # 右膝蓋角度
                'left_hip_angle': calculate_angle(left_shoulder, left_hip, left_knee),     # 左髖部角度
                'right_hip_angle': calculate_angle(right_shoulder, right_hip, right_knee), # 右髖部角度
                # 身體前傾角度（肩膀到臀部的角度）
                'body_lean': calculate_vertical_angle(
                    [(left_shoulder[0] + right_shoulder[0])/2, (left_shoulder[1] + right_shoulder[1])/2],
                    [(left_hip[0] + right_hip[0])/2, (left_hip[1] + right_hip[1])/2]
                ),
                # 膝蓋內扣檢測（膝蓋與臀部的 x 座標比較）
                'knee_valgus': calculate_knee_valgus(left_hip, left_knee, right_hip, right_knee)
            }
            
            return angles
        # 例外處理 
        except Exception as e:
            print(f"提取角度時發生錯誤: {e}")
            return None
    
    
    # ===== 分析標準深蹲影片，提取動作序列 =====
    # ======================================= 
    def analyze_standard_video(self, video_path):
        """
            1. 檢查影片是否能正確讀取
            2. 讀取影片
            3. 轉換顏色格式
            4. 進行姿勢檢測
            5. 檢查是否偵測到人體
            6. 儲存標準序列
        """

        print(f"正在分析標準影片: {video_path}")
        
        # ----- 檢查影片是否能正確讀取 -----
        cap = cv2.VideoCapture(video_path) # 取指定路徑的影片檔案

        # 檢查影片是否成功開啟
        if not cap.isOpened():
            raise Exception(f"無法開啟影片文件: {video_path}") # 丟出例外（Exception），並顯示錯誤訊息
        
        frame_count = 0 # 初始使化影格計數器，從 0 開始 
        sequence = [] # 儲存每一幀分析得到的標準動作角度資料
        
        # ----- 處理畫面 -----
        while cap.isOpened():
            # ----- 讀取影片 -----
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # ----- 轉換顏色格式 -----
            # RGB 格式的 NumPy 陣列（shape: H x W x 3，通常 dtype 為 uint8）
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ----- 進行姿勢檢測 -----
            # 呼叫 MediaPipe Pose 的 process 方法（由 self.pose = mp.solutions.pose.Pose(...) 建立的物件）
            results = self.pose.process(rgb_frame) 
            
            # ----- 檢查是否偵測到人體 -----
            if results.pose_landmarks:
                # 讀取影像
                h, w, _ = frame.shape # frame.shape 回傳一個包含三個值的 tuple（height, width, channels）
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
            print(f"標準動作序列已從 {filename} 載入，共 {len(self.standard_sequence)} 幀 by pose_detector")
            return True
            # 例外處理
        except Exception as e:
            print(f"載入標準序列時發生錯誤: {e} by pose_detector")
            return False
    
    # ===== 清理資源 =====
    def cleanup(self):
        self.pose.close() # 關閉 MediaPipe 的姿勢偵測器，避免資源占用
        print("關閉 MediaPipe 的姿勢偵測器，資源已釋放 by pose_detector")


# ===== 結合標準動作的深蹲檢測器(類別) =====
# ======================================== 
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
    # 輸入 關鍵點角度
    # 輸出 
    # =======================================  
    def compare_with_standard(self, current_angles):
        
        """
            1. 檢查「標準動作序列&目前角度資料」是否有資料
            2. 記錄當前動作序列
            3. 限制序列長度
            4. 檢查序列是否太短
            5. 提取關鍵角度序列進行比較
            6. 讀取當前序列的角度
            7. 檢查「標準角度序列&目前角度序列」是否有資料
            8. 
        """

        # ----- 檢查「標準動作序列&目前角度資料」是否有資料 -----
        if not self.standard_sequence or not current_angles:
            return 0.0, "無標準資料" # 回傳「相似度 0.0」，代表無資料
        
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
            return 0.5, "動作序列太短" # 0.5 是暫時性、保留的相似度，避免誤判為完全錯誤
        
        # 使用 DTW 比較角度序列
        try:
            # ----- 提取關鍵角度序列進行比較 -----
            key_angles = ['left_knee_angle', 'right_knee_angle', 'left_hip_angle', 'right_hip_angle']
            
            similarities = [] # 儲存「每種關鍵角度」與標準動作比对後的相似度分數

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
    
    # ===== 根據角度和相似度生成回饋 =====
    # 1. 檢查下蹲深度
    # 2. 檢查膝蓋內扣
    # 3. 檢查身體前傾
    # 4. 根據相似度給出總體評價
    # 5. 顯示動作評價
    # ================================== 
    def get_feedback(self, angles, similarity):
        
        feedback_parts = [] # 動作評價
        
        # ----- 檢查下蹲深度 -----
        # 左右膝蓋角度 > 100 度
        # ----------------------- 
        left_knee = angles.get('left_knee_angle', 180) # 預設角度 180 度，代表「完全伸直」的膝蓋
        right_knee = angles.get('right_knee_angle', 180)
        if left_knee > 100 or right_knee > 100:
            feedback_parts.append("深度不足")
        
        # ----- 檢查膝蓋內扣 -----
        # 膝蓋距離和臀部距離的比例小於 0.8
        # ----------------------- 
        knee_valgus = angles.get('knee_valgus', 1.0)
        if knee_valgus < 0.8:  # 膝蓋過於內扣
            feedback_parts.append("膝蓋內扣")
        
        # ----- 檢查身體前傾 -----
        # 深蹲時身體前傾角度超過 20 度
        # ----------------------- 
        body_lean = angles.get('body_lean', 0)
        if body_lean > 20:  # 過度前傾
            feedback_parts.append("身體過於前傾")
        
        # ----- 根據相似度給出總體評價 -----
        # 最近一段深蹲動作的關鍵角度變化，和標準深蹲動作的關鍵角度變化之間的相似程度
        # -------------------------------- 
        if similarity > 0.8:
            overall = "動作標準"
        elif similarity > 0.6:
            overall = "動作良好"
        elif similarity > 0.4:
            overall = "需要改善"
        else:
            overall = "動作不正確"
        
        # ----- 顯示動作評價 -----
        if feedback_parts:
            return f"{overall}: {', '.join(feedback_parts)}"
        else:
            return overall
    
    # ===== 判斷是否為深蹲姿勢並分析品質 =====
    # 1. 提取角度 
    def is_squat_pose(self, landmarks, frame_width, frame_height):
        
        # 提取角度(左右膝蓋、左右髖關節、肩膀到臀部、膝蓋距離/臀部距離)
        angles = self.analyzer.extract_key_angles(landmarks, frame_width, frame_height)
        if not angles:
            return False, 0, 0, 0.0, "無法分析"
        
        # 基本深蹲判斷
        left_knee_angle = angles['left_knee_angle'] # 左膝蓋
        right_knee_angle = angles['right_knee_angle'] # 右膝蓋
        
        # 判斷是否為「深蹲」 => 左右膝蓋都小於120度
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
    
    # ===== 繪製中文字 =====
    def draw_chineese_text(self, frame, x_y, text, font_size=32, color=(255,255,255)):
        # ----- 參數說明 -----
        # 在 frame 上指定位置繪製中文字。
        # frame: OpenCV 影像 (numpy array)
        # x_y: (x, y) 座標
        # text: 要顯示的文字（可含中文）
        # font_size: 字體大小
        # color: 文字顏色 (B, G, R)
        # -------------------- 
        # 1. 轉成 PIL 影像
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # 2. 指定中文字型（Windows 可用微軟正黑體 msjh.ttc）
        font_path = "msjh.ttc"  # 請確認這個字型檔案在你的系統上存在
        font = ImageFont.truetype(font_path, font_size)

        # 3. 畫字
        draw.text(x_y, text, font=font, fill=(color[2], color[1], color[0]))  # PIL 用 RGB
        
        # 4. 轉回 OpenCV 格式
        frame[:,:,:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # ===== 繪製狀態資訊 =====
    # frame => 原始影像
    # is_squat => 是否為深蹲姿勢
    # left_angle, right_angle => 左右膝蓋角度
    # similarity => 與標準動作的相似度
    # feedback => 動作回饋訊息
    # has_pose => 是否檢測到姿勢
    # ========================
    def draw_status_info(self, frame, is_squat, left_angle, right_angle, similarity, feedback, has_pose):
        
        h, w, _ = frame.shape # 影像高度、影像寬度、色彩通道數(不會用到)
        

        # ----- (補充)cv2.putText參數說明: 
        if has_pose:
            # 顯示姿勢狀態
            if is_squat:
                # 在畫面左上角顯示「深蹲」
                # cv2.putText(frame, "Squats", (10, 50), 
                #            cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 3)
                self.draw_chineese_text(frame, (10, 0), "深蹲"
                                        , font_size=100, color=(0, 255, 0))
            else:
                # 在畫面左上角顯示「站立」
                # cv2.putText(frame, "Standing", (10, 50), 
                #            cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 0, 0), 3)
                self.draw_chineese_text(frame, (10, 0), "站立"
                                        , font_size=100, color=(255, 0, 0))
            
            # 顯示深蹲計數
            # cv2.putText(frame, f"Count: {self.squat_count}", (10, 100), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            self.draw_chineese_text(frame, (10, 120), f"總次數: {self.squat_count}"
                                    , font_size=40, color=(255, 255, 0))
            # cv2.putText(frame, f"Correct: {self.correct_squat_count}", (10, 140), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            self.draw_chineese_text(frame, (10, 160), f"總次數: {self.correct_squat_count}"
                                    , font_size=40, color=(0, 255, 0))
            
            # 顯示相似度
            similarity_color = (0, 255, 0) if similarity > 0.8 else (0, 255, 255) if similarity > 0.6 else (0, 0, 255)
            # cv2.putText(frame, f"Similarity: {similarity:.2f}", (10, 180), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.8, similarity_color, 2)
            self.draw_chineese_text(frame, (10, 200), f"準確率: {similarity:.2f}"
                                    , font_size=40, color=similarity_color)
            
            # 顯示回饋
            # cv2.putText(frame, feedback, (10, 220), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            self.draw_chineese_text(frame, (10, 300), f"{feedback}"
                                    , font_size=40, color=(10, 10, 10))
            
            # 顯示角度資訊
            cv2.putText(frame, f"L: {int(left_angle)} deg  R: {int(right_angle)} deg", (10, 300), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 10, 10), 2)
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
    def process_frame(self, frame, mirror=True, show_info=True):
        
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
        # 只有在 show_info=True 時才繪製狀態資訊
        if show_info:
            self.draw_status_info(frame, is_squat, left_angle, right_angle, similarity, feedback, has_pose)

        return frame

    # ===== 重設計數器 =====     
    def reset_counters(self):
        self.squat_count = 0            # 總深蹲次數
        self.correct_squat_count = 0    # 正確深蹲次數
        self.in_squat = False           # 深蹲狀態
        self.current_sequence = []      # 動作角度序列
        print("計數已重設")
    
    # ===== 清理資源 =====
    def cleanup(self):
        self.pose.close() # 關閉 MediaPipe 的姿勢偵測器，釋放相關資源
        self.analyzer.cleanup()