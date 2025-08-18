import cv2              # 存取攝像頭、讀取和顯示影像、處理影像（如翻轉、加文字等）
import mediapipe as mp  # 進行人體姿勢偵測，提供關鍵點座標、繪製骨架等功能 
import numpy as np      # 數值運算、陣列處理，方便影像或座標資料的計算與操作
import math             # 數學函數，用於角度計算

class SquatDetector:
    """深蹲檢測器類別"""
    
    def __init__(self, squat_threshold=120, detection_confidence=0.5, tracking_confidence=0.5):
        """
        初始化深蹲檢測器
        
        Args:
            squat_threshold: 深蹲判斷的膝蓋角度閾值
            detection_confidence: 姿勢檢測信心度
            tracking_confidence: 姿勢追蹤信心度
        """
        self.squat_threshold = squat_threshold
        self.squat_count = 0
        self.in_squat = False
        
        # ===== 初始化 MediaPipe =====
        # mp_pose => 姿勢偵測模組
        # mp_drawing => 繪圖工具模組
        # ============================ 
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # ===== 設定姿勢檢測 =====
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=detection_confidence,  # 最小檢測信心度
            min_tracking_confidence=tracking_confidence     # 最小追蹤信心度
        )
    
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
    
    # ===== 提取關鍵身體部位座標 =====
    # landmarks => MediaPipe檢測到的關鍵點
    # frame_width, frame_height => 影像寬度和高度
    # 返回 => 包含各身體部位座標的字典
    # ================================
    def extract_landmarks(self, landmarks, frame_width, frame_height):
        try:
            # 取得關鍵點座標（轉換為像素座標）
            # 左側關鍵點
            left_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x * frame_width,
                       landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y * frame_height]
            left_knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x * frame_width,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y * frame_height]
            left_ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x * frame_width,
                         landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y * frame_height]
            
            # 右側關鍵點
            right_hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x * frame_width,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y * frame_height]
            right_knee = [landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x * frame_width,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y * frame_height]
            right_ankle = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * frame_width,
                          landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * frame_height]
            
            return {
                'left_hip': left_hip,
                'left_knee': left_knee,
                'left_ankle': left_ankle,
                'right_hip': right_hip,
                'right_knee': right_knee,
                'right_ankle': right_ankle
            }
        except:
            # 如果計算過程出錯，返回None
            return None
    
    # ===== 判斷是否為深蹲姿勢 =====
    # landmarks => MediaPipe檢測到的關鍵點
    # frame_width, frame_height => 影像寬度和高度
    # 返回 => True(深蹲) 或 False(非深蹲)
    # 流程:
    # 1. 取得關鍵點座標（轉換為像素座標，包括臀部、膝蓋、腳踝）
    # 2. 計算膝蓋角度
    # 3. 深蹲條件判斷
    # 4. 返回結果 
    # ============================= 
    def is_squat_pose(self, landmarks, frame_width, frame_height):
        # 提取關鍵點座標
        coords = self.extract_landmarks(landmarks, frame_width, frame_height)
        if coords is None:
            return False, 0, 0
        
        # 計算膝蓋角度
        left_knee_angle = self.calculate_angle(coords['left_hip'], coords['left_knee'], coords['left_ankle'])
        right_knee_angle = self.calculate_angle(coords['right_hip'], coords['right_knee'], coords['right_ankle'])
        
        # 深蹲判斷條件：
        # 1. 膝蓋彎曲角度小於120度（可調整）
        # 2. 左右膝蓋角度都符合條件
        squat_threshold = self.squat_threshold  # 深蹲閾值角度
        
        if left_knee_angle < squat_threshold and right_knee_angle < squat_threshold:
            return True, left_knee_angle, right_knee_angle
        else:
            return False, left_knee_angle, right_knee_angle
    
    # ===== 更新深蹲計數 =====
    # is_squat => 當前是否為深蹲姿勢
    # 避免重複計數的邏輯
    # ========================
    def update_squat_count(self, is_squat):
        if is_squat and not self.in_squat:
            # 從非深蹲狀態進入深蹲狀態，計數+1
            self.squat_count += 1
            self.in_squat = True
        elif not is_squat and self.in_squat:
            # 從深蹲狀態回到非深蹲狀態
            self.in_squat = False
    
    # ----- 繪製人體骨架 -----
    # frame => 原始影像
    # results => MediaPipe檢測結果
    # -----------------------
    def draw_pose_skeleton(self, frame, results):
        if results.pose_landmarks:
            # 繪製關鍵點和骨架
            self.mp_drawing.draw_landmarks(
                frame, # 原始影像
                results.pose_landmarks, # 人體關鍵點資料
                self.mp_pose.POSE_CONNECTIONS, # 定義關鍵點之間的連線(骨架)
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),  # 關鍵點樣式
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)  # 連接線樣式
            )
    
    # ----- 在畫面上繪製狀態資訊 -----
    # frame => 原始影像
    # is_squat => 是否為深蹲姿勢
    # left_angle, right_angle => 左右膝蓋角度
    # has_pose => 是否檢測到姿勢
    # --------------------------------
    def draw_status_info(self, frame, is_squat, left_angle, right_angle, has_pose):
        h, w, _ = frame.shape # 影像高度、影像寬度、色彩通道數(不會用到)
        
        if has_pose:
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
            
            # 顯示角度資訊（除錯用）
            cv2.putText(frame, f"Left knee angle: {int(left_angle)} angle", (10, 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Right knee angle: {int(right_angle)} angle", (10, 170), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # --- 在畫面上顯示檢測狀態 ---
            # 參數說明如下：
            # frame：要在其上顯示文字的影像。
            # "Pose Detected"：要顯示的文字內容。
            # (10, h-20)：文字左下角的位置（x=10, y=h-20，單位是像素）。
            # cv2.FONT_HERSHEY_SIMPLEX：字型樣式（OpenCV 內建字型）。
            # 0.7：文字大小（縮放比例）。
            # (0, 255, 0)：文字顏色（BGR格式，這裡是綠色）。
            # 2：文字線條粗細。
            # -------------------------- 
            cv2.putText(frame, "Pose Detected", (10, h-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 顯示關鍵點數量
            cv2.putText(frame, f"Landmarks: {len(landmarks) if 'landmarks' in locals() else 33}", (10, h-50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            # 沒有檢測到姿勢
            cv2.putText(frame, "No Pose Detected", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    
    # ===== 處理單一影格 =====
    # frame => 原始影像
    # 返回 => 處理後的影像
    # 流程:
    # 1. 畫面左右顛倒
    # 2. 轉換顏色格式
    # 3. 進行姿勢檢測
    # 4. 判斷深蹲姿勢
    # 5. 更新計數
    # 6. 繪製資訊
    # ========================
    def process_frame(self, frame):
        # ----- 畫面左右顛倒 -----
        # 鏡像效果，符合使用者視角
        # 參數補充:
        # 1 => 水平翻轉
        # 0 => 垂直翻轉
        # -1 => 水平垂直翻轉
        # ----------------------- 
        frame = cv2.flip(frame, 1)
        
        # ----- 轉換顏色格式 -----
        # BGR -> RGB，MediaPipe需要RGB格式
        # -----------------------
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # ----- 進行姿勢檢測 -----
        results = self.pose.process(rgb_frame)
        
        # 預設值
        is_squat, left_angle, right_angle = False, 0, 0
        has_pose = results.pose_landmarks is not None
        
        # ----- 檢查是否檢測到姿勢並繪製骨架 -----
        if has_pose:
            # 取得關鍵點座標 (正規化座標，需要轉換為像素座標)
            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame.shape # 影像高度、影像寬度、色彩通道數(不會用到)
            
            # ----- 判斷深蹲姿勢 -----
            is_squat, left_angle, right_angle = self.is_squat_pose(landmarks, w, h)
            
            # 更新深蹲計數
            self.update_squat_count(is_squat)
            
            # 繪製骨架
            self.draw_pose_skeleton(frame, results)
        
        # 繪製狀態資訊
        self.draw_status_info(frame, is_squat, left_angle, right_angle, has_pose)
        
        return frame
    
    # ===== 清理資源 =====
    def cleanup(self):
        self.pose.close() # 關閉 MediaPipe 的姿勢偵測器，釋放相關資源


# ===== 設定攝像頭 =====
# camera_index => 攝像頭索引
# width, height => 影像寬度和高度
# 返回 => 攝像頭物件
# ======================
def setup_camera(camera_index=0, width=920, height=540):
    # ===== 初始化攝像頭 ======
    # cv2.VideoCapture(0) => 使用預設後端(有可能使用到不適合的系統)
    # cv2.VideoCapture(0, cv2.【系統參數】) => 可以指定適合的系統 
    # Windows => CAP_DSHOW(推薦), CAP_MSMF
    # macOS => CAP_AVFOUNDATION(推薦)
    # Linux => CAP_V4L2(推薦), CAP_GSTREAMER
    # ======================== 
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    
    # ===== 設定攝像頭解析度 (可選) ======
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width) # 畫面寬度
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height) # 畫面高度
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) # 用 MJPG 編碼，減少延遲
    
    # 檢查攝像頭是否正常開啟
    if not cap.isOpened():
        raise Exception("無法開啟攝像頭")
    
    return cap


# ===== 顯示使用說明 =====
def display_instructions():
    print("="*50)
    print("深蹲姿勢檢測程式")
    print("="*50)
    print("使用說明：")
    print("- 站在攝像頭前方，確保全身都在畫面內")
    print("- 做深蹲動作時，膝蓋角度需小於120度才會被識別")
    print("- 程式會自動計算深蹲次數")
    print("- 按 'q' 鍵退出程式")
    print("- 按 'r' 鍵重設計數")
    print("="*50)


# ===== 主要檢測迴圈 =====
# detector => 深蹲檢測器實例
# cap => 攝像頭物件
# 流程:
# 1. 讀取鏡頭畫面
# 2. 檢查有無讀取到畫面
# 3. 處理影格
# 4. 顯示畫面
# 5. 處理按鍵事件
# ========================
def main_detection_loop(detector, cap):
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
        
        # 處理影格
        processed_frame = detector.process_frame(frame)
        
        # ----- 顯示畫面 -----
        # 視窗名稱、影像
        # ------------------- 
        cv2.imshow('Squat posture detection', processed_frame)
        
        # 處理按鍵事件
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # 按 'q' 鍵退出 
            print("使用者按下 'q' 鍵，退出程式")
            break
        elif key == ord('r'):
            # 按 'r' 鍵重設計數
            detector.squat_count = 0
            detector.in_squat = False
            print("深蹲計數已重設")


# ===== 清理程式資源 =====
# cap => 攝像頭物件
# detector => 深蹲檢測器實例
# ========================
def cleanup_resources(cap, detector):
    # ===== 釋放資源 =====
    cap.release() # 釋放攝像頭資源，讓攝像頭可以被其他程式使用
    cv2.destroyAllWindows() # 關閉所有由 OpenCV 開啟的視窗，清理顯示資源
    detector.cleanup() # 關閉 MediaPipe 的姿勢偵測器，釋放相關資源
    
    # 結束訊息
    print(f"程式結束，總計完成 {detector.squat_count} 次深蹲")


def main():
    """主函式"""
    try:
        # 顯示使用說明
        display_instructions()
        
        # 設定攝像頭
        cap = setup_camera(camera_index=0, width=920, height=540)
        
        # 初始化深蹲檢測器
        detector = SquatDetector(
            squat_threshold=120,
            detection_confidence=0.5,
            tracking_confidence=0.5
        )
        
        print("按 'q' 鍵退出程式") # 提示字
        print("深蹲時膝蓋角度需小於120度才會顯示'深蹲'") # 深蹲判斷說明
        
        # ===== 開始偵測 =====
        main_detection_loop(detector, cap)
        
    except Exception as e:
        print(f"程式執行時發生錯誤: {e}")
    
    finally:
        # 清理資源
        if 'cap' in locals() and 'detector' in locals():
            cleanup_resources(cap, detector)


if __name__ == '__main__':
    main()