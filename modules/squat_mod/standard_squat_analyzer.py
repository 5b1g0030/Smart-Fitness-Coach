import cv2              # 存取攝像頭、讀取和顯示影像、處理影像（如翻轉、加文字等）
import mediapipe as mp  # 進行人體姿勢偵測，提供關鍵點座標、繪製骨架等功能 
import numpy as np      # 數值運算、陣列處理，方便影像或座標資料的計算與操作
import json             # 用於儲存和讀取標準動作資料

# ===== 標準深蹲動作分析器(類別) =====
# ===================================   
class StandardSquatAnalyzer:
    
    # ===== 初始化標準動作分析器 ===== 
    def __init__(self):
        '''
            初始化 MediaPipe
            設定姿勢檢測: 最小檢測信心度 & 最小追蹤信心度
            初始化標準動作序列
            ***引用類別時自動執行***
        '''
        # ----- 初始化 MediaPipe -----
        self.mp_pose = mp.solutions.pose # 姿勢偵測模組
        self.mp_drawing = mp.solutions.drawing_utils # 繪圖工具模組
        
        # ----- 設定姿勢檢測&追蹤 ----- 
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
