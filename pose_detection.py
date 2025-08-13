import cv2
import mediapipe as mp
import numpy as np

# 初始化 MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 設定姿勢檢測
pose = mp_pose.Pose(
    min_detection_confidence=0.5,  # 最小檢測信心度
    min_tracking_confidence=0.5    # 最小追蹤信心度
)

# 初始化攝像頭
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# 設定攝像頭解析度 (可選)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("按 'q' 鍵退出程式")

while cap.isOpened():
    # 讀取攝像頭畫面
    ret, frame = cap.read()
    
    if not ret:
        print("無法讀取攝像頭")
        break
    
    # 翻轉畫面 (鏡像效果)
    frame = cv2.flip(frame, 1)
    
    # 轉換顏色格式 (BGR -> RGB，MediaPipe需要RGB格式)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 進行姿勢檢測
    results = pose.process(rgb_frame)
    
    # 如果檢測到姿勢
    if results.pose_landmarks:
        # 繪製關鍵點和骨架
        mp_drawing.draw_landmarks(
            frame, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),  # 關鍵點樣式
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)  # 連接線樣式
        )
        
        # 取得關鍵點座標 (正規化座標，需要轉換為像素座標)
        landmarks = results.pose_landmarks.landmark
        h, w, _ = frame.shape
        
        # 顯示一些重要關鍵點的座標資訊
        # 左右肩膀
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        
        # 左右臀部
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        # 左右膝蓋
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        
        # 在畫面上顯示檢測狀態
        cv2.putText(frame, "Pose Detected", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 顯示關鍵點數量
        cv2.putText(frame, f"Landmarks: {len(landmarks)}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    else:
        # 沒有檢測到姿勢
        cv2.putText(frame, "No Pose Detected", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # 顯示畫面
    cv2.imshow('MediaPipe Pose Detection', frame)
    
    # 按 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源
cap.release()
cv2.destroyAllWindows()
pose.close()

print("程式結束")