import cv2              # 存取攝像頭、讀取和顯示影像、處理影像（如翻轉、加文字等）
import mediapipe as mp  # 進行人體姿勢偵測，提供關鍵點座標、繪製骨架等功能 
import numpy as np      # 數值運算、陣列處理，方便影像或座標資料的計算與操作
from dtw import dtw  # 需要安裝: pip install dtw-python
# 新增：匯入已分離的類別
from modules.squat_mod.draw import draw_status_info
from modules.squat_mod.standard_squat_analyzer import StandardSquatAnalyzer  # 新增：匯入已分離的類別

# ===== 結合標準動作的深蹲檢測器(類別) =====
# ======================================== 
class SquatDetectorWithStandard:
    
    # ===== 初始化檢測器 =====
    def __init__(self, standard_sequence, squat_threshold=120, similarity_threshold=0.8):

        """
            1. 設定基礎參數
            2. 設定計數器
            3. 設定當前動作序列記錄
            4. 初始化 MediaPipe
            5. 設定姿勢檢測
            6. 初始化標準動作分析器
        """
        
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
    def get_feedback(self, angles, similarity):
        
        """
            1. 檢查下蹲深度
            2. 檢查膝蓋內扣
            3. 檢查身體前傾
            4. 根據相似度給出總體評價
            5. 顯示動作評價
        """
        
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
    
    # ===== 處理單一影格 =====
    def process_frame(self, frame, mirror=True, show_info=True):
        
        """
            frame => 原始影像
            mirror => 是否鏡像翻轉（攝像頭模式需要，影片模式不需要）
            返回 => 處理後的影像
            流程:
            1. 畫面左右顛倒（可選）
            2. 轉換顏色格式
            3. 進行姿勢檢測
            4. 判斷深蹲姿勢並分析品質
            5. 更新計數
            6. 繪製資訊
        """
        
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
        
        # 繪製狀態資訊（改用外部 draw 模組）
        if show_info:
            # 傳入必要數值給 draw_status_info
            draw_status_info(frame, is_squat, left_angle, right_angle, similarity, feedback, has_pose,
                             squat_count=self.squat_count, correct_squat_count=self.correct_squat_count)

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