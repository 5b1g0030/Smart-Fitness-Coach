import cv2              # 存取攝像頭、讀取和顯示影像、處理影像（如翻轉、加文字等）
import mediapipe as mp  # 進行人體姿勢偵測，提供關鍵點座標、繪製骨架等功能 
import numpy as np      # 數值運算、陣列處理，方便影像或座標資料的計算與操作

# ===== 初始化 MediaPipe =====
# mp_pose => 姿勢偵測模組
# mp_drawing => 繪圖工具模組
# ============================ 
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ===== 設定姿勢檢測 =====
pose = mp_pose.Pose(
    min_detection_confidence=0.5,  # 最小檢測信心度
    min_tracking_confidence=0.5    # 最小追蹤信心度
)

# ===== 初始化攝像頭 ======
# cv2.VideoCapture(0) => 使用預設後端(有可能使用到不適合的系統)
# cv2.VideoCapture(0, cv2.【系統參數】) => 可以指定適合的系統 
# Windows => CAP_DSHOW(推薦), CAP_MSMF
# macOS => CAP_AVFOUNDATION(推薦)
# Linux => CAP_V4L2(推薦), CAP_GSTREAMER
# ======================== 
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# ===== 設定攝像頭解析度 (可選) ======
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 920) # 畫面寬度
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540) # 畫面高度
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) # 用 MJPG 編碼，減少延遲

print("按 'q' 鍵退出程式") # 提示字

# ===== 開始偵測 =====
# 1. 讀取鏡頭畫面
# 2. 檢查有無讀取到畫面
# 3. 畫面左右顛倒
# 4. 轉換顏色格式
# 5. 進行姿勢檢測
# 6. 檢查是否檢測到姿勢
# 7. 繪製骨架和關鍵點
# 8. 顯示畫面
# ==================== 
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
    results = pose.process(rgb_frame)
    
    # ----- 檢查是否檢測到姿勢 -----
    if results.pose_landmarks:
        # 繪製關鍵點和骨架
        mp_drawing.draw_landmarks(
            frame, # 原始影像
            results.pose_landmarks, # 人體關鍵點資料
            mp_pose.POSE_CONNECTIONS, # 定義關鍵點之間的連線(骨架)
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),  # 關鍵點樣式
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)  # 連接線樣式
        )
        
        # 取得關鍵點座標 (正規化座標，需要轉換為像素座標)
        landmarks = results.pose_landmarks.landmark
        h, w, _ = frame.shape # 影像高度、影像寬度、色彩通道數(不會用到)
        
        # --- 顯示一些重要關鍵點的座標資訊 ---
        # 左右肩膀
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        
        # 左右臀部
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        # 左右膝蓋
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        
        # --- 在畫面上顯示檢測狀態 ---
        # 參數說明如下：
        # frame：要在其上顯示文字的影像。
        # "Pose Detected"：要顯示的文字內容。
        # (10, 30)：文字左下角的位置（x=10, y=30，單位是像素）。
        # cv2.FONT_HERSHEY_SIMPLEX：字型樣式（OpenCV 內建字型）。
        # 1：文字大小（縮放比例）。
        # (0, 255, 0)：文字顏色（BGR格式，這裡是綠色）。
        # 2：文字線條粗細。
        # -------------------------- 
        cv2.putText(frame, "Pose Detected", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 顯示關鍵點數量
        cv2.putText(frame, f"Landmarks: {len(landmarks)}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    else:
        # 沒有檢測到姿勢
        cv2.putText(frame, "No Pose Detected", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # ----- 顯示畫面 -----
    # 視窗名稱、影像
    # ------------------- 
    cv2.imshow('MediaPipe Pose Detection', frame)
    
    # 按 'q' 鍵退出 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源
cap.release() # 釋放攝像頭資源，讓攝像頭可以被其他程式使用
cv2.destroyAllWindows() # 關閉所有由 OpenCV 開啟的視窗，清理顯示資源
pose.close() # 關閉 MediaPipe 的姿勢偵測器，釋放相關資源

# 結束訊息
print("程式結束")