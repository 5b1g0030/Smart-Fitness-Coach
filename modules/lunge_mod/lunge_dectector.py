import cv2
import time
import os
import math
from datetime import datetime

try:
	import mediapipe as mp
except Exception as e:
	raise ImportError("Please install mediapipe (pip install mediapipe).") from e

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ===== 計算膝蓋彎曲角度 =====
# 計算「以 b 為頂點」，由點 a → b → c 形成的夾角（角度制）
def angle_between(a, b, c):
	# 把每個點的 x、y 拿出來，方便後面計算
	(ax, ay) = a
	(bx, by) = b
	(cx, cy) = c
	# v1：從 b 指向 a 的向量, v2：從 b 指向 c 的向量
	v1 = (ax - bx, ay - by)
	v2 = (cx - bx, cy - by)
	# 向量內積: 越大 → 越接近 0°, 越小（負）→ 越接近 180°
	dot = v1[0]*v2[0] + v1[1]*v2[1]
	# math.hypot(x, y)  = sqrt(x*x + y*y)
	# n1：向量 v1 的長度, n2：向量 v2 的長度
	n1 = math.hypot(v1[0], v1[1])
	n2 = math.hypot(v2[0], v2[1])
	# 沒有方向，就無法定義角度
	if n1 * n2 == 0:
		return 0.0
	# 計算 cos(θ)，並「夾住」範圍
	cosang = max(-1.0, min(1.0, dot / (n1 * n2)))
	# 反推角度（弧度 → 度）
	return math.degrees(math.acos(cosang))

def landmarks_to_point(lm, width, height):
	return (lm.x * width, lm.y * height)

# ===== 使用左右腿資訊判斷弓箭步 =====
def is_lunge(landmarks, image_w, image_h):
	# 使用左右腿資訊判斷弓箭步
	# 參考：前膝 60-100°, 後膝 >150°, 軀幹與垂直角度小於 30°
	try:
		# PoseLandmark enums
		LH = mp_pose.PoseLandmark.LEFT_HIP
		RH = mp_pose.PoseLandmark.RIGHT_HIP
		LK = mp_pose.PoseLandmark.LEFT_KNEE
		RK = mp_pose.PoseLandmark.RIGHT_KNEE
		LA = mp_pose.PoseLandmark.LEFT_ANKLE
		RA = mp_pose.PoseLandmark.RIGHT_ANKLE
		LS = mp_pose.PoseLandmark.LEFT_SHOULDER
		RS = mp_pose.PoseLandmark.RIGHT_SHOULDER

		left_hip = landmarks[LH]
		right_hip = landmarks[RH]
		left_knee = landmarks[LK]
		right_knee = landmarks[RK]
		left_ankle = landmarks[LA]
		right_ankle = landmarks[RA]
		left_sho = landmarks[LS]
		right_sho = landmarks[RS]
	except Exception:
		print("發生錯誤!")
		return (False, None)

	# convert to pixel coords
	LH_pt = landmarks_to_point(left_hip, image_w, image_h)
	RH_pt = landmarks_to_point(right_hip, image_w, image_h)
	LK_pt = landmarks_to_point(left_knee, image_w, image_h)
	RK_pt = landmarks_to_point(right_knee, image_w, image_h)
	LA_pt = landmarks_to_point(left_ankle, image_w, image_h)
	RA_pt = landmarks_to_point(right_ankle, image_w, image_h)
	# LS_pt = landmarks_to_point(left_sho, image_w, image_h)
	# RS_pt = landmarks_to_point(right_sho, image_w, image_h)

	# angles at knees
	left_knee_angle = angle_between(LH_pt, LK_pt, LA_pt)
	right_knee_angle = angle_between(RH_pt, RK_pt, RA_pt)

	# torso: angle between shoulders and vertical (use midpoint)
	# shoulder_mid = ((LS_pt[0] + RS_pt[0]) / 2, (LS_pt[1] + RS_pt[1]) / 2)
	# hip_mid = ((LH_pt[0] + RH_pt[0]) / 2, (LH_pt[1] + RH_pt[1]) / 2)
	# # vector hip_mid -> shoulder_mid, compare with vertical vector (0,-1)
	# torso_vec = (shoulder_mid[0] - hip_mid[0], shoulder_mid[1] - hip_mid[1])
	# vertical_vec = (0, -1)
	# compute torso tilt angle
	# dot = torso_vec[0]*vertical_vec[0] + torso_vec[1]*vertical_vec[1]
	# n1 = math.hypot(torso_vec[0], torso_vec[1])
	# if n1 == 0:
	# 	torso_tilt = 90.0
	# else:
	# 	cosang = max(-1.0, min(1.0, dot / n1))
	# 	torso_tilt = math.degrees(math.acos(cosang))

	# Check left-front / right-front possibilities
	# left front: left_knee 60-100, right_knee >150 and torso_tilt <= 35
	if 60 <= left_knee_angle <= 120 and right_knee_angle >= 120:
		return (True, "left")
	# right front: right_knee 60-100, left_knee >150
	if 60 <= right_knee_angle <= 120 and left_knee_angle >= 120:
		return (True, "right")
	
	# print(f"{left_knee_angle}, {right_knee_angle}") # 除錯

	return (False, None)

# ===== 繪製左右腳骨架 =====
# 輸入 畫面、左右腿骨架座標、 
def draw_lags(image, left_idxs, right_idxs, lm, w, h):
	for idx in left_idxs:
		p = lm[idx]
		vis = getattr(p, "visibility", 1.0)
		if vis is None or vis >= 0.5:
			x, y = int(p.x * w), int(p.y * h)
			cv2.circle(image, (x, y), 6, (0, 255, 0), -1)  # 左綠 (B,G,R)
	for idx in right_idxs:
		p = lm[idx]
		vis = getattr(p, "visibility", 1.0)
		if vis is None or vis >= 0.5:
			x, y = int(p.x * w), int(p.y * h)
			cv2.circle(image, (x, y), 6, (255, 0, 0), -1)  # 右藍 (B,G,R)

# ===== 繪製判定結果 =====
# 輸入 畫面、左右弓箭步次數
def draw_result(image, side, count_left, count_right):
	cv2.putText(image, f"Lunge L:{count_left} R:{count_right}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
	if side:
		print("side is Ture!")
		cv2.putText(image, f"Detected: {side}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,200,255), 5)
	else:
		cv2.putText(image, f"Detected: None", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,200,255), 5)

# ===== 弓箭步記數 =====
# 傳入 判定結果、左/右側當前是否處於「偵測到弓箭步」(bool) 
def lunge_count(side, prev_state_left, prev_state_right, count_left, count_right):
	# 左弓箭步
	if side == "left":
		# when left lunge detected now
		if not prev_state_left:
			count_left += 1
		prev_state_left = True
		prev_state_right = False
	# 右弓箭步
	elif side == "right":
		if not prev_state_right:
			count_right += 1
		prev_state_right = True
		prev_state_left = False
	else:
		prev_state_left = False
		prev_state_right = False
	
	return count_left, count_right


# ===== 處理動作影格 =====
def process_stream(cap, delete_input_after=None):
	pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) # 建立偵測器
	# 左/右側弓箭步累計計數
	count_left = 0	
	count_right = 0
	# 左/右側當前是否處於「偵測到弓箭步」(bool)
	prev_state_left = False  
	prev_state_right = False 

	# ===== 影像串流 =====
	cv2.namedWindow("Lunge Detector", cv2.WINDOW_NORMAL)
	cv2.resizeWindow("Lunge Detector", 800, 800)
	while cap.isOpened():
		# 讀取影像
		ret, frame = cap.read()
		if not ret:
			break

		h, w = frame.shape[:2] # 取影像寬高
		image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # BGR => RGB
		image.flags.writeable = False # 將陣列標記為不可寫，讓 MediaPipe 可以避免不必要的複製與優化處理
		results = pose.process(image) # 執行 MediaPipe Pose 偵測，回傳 pose_landmarks
		image.flags.writeable = True  # 重新允許對影像寫入，以便後續在影像上畫標記或文字
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # RBG => BGR (方便 OpenCV 取用)

		lunge_detected = False
		side = None
		if results.pose_landmarks:
			mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
			lunge_detected, side = is_lunge(results.pose_landmarks.landmark, w, h)
			# print(lunge_detected) # 除錯

			# Debug: 用顏色標示左右腿的 hip/knee/ankle（左紅、右藍）
			try:
				lm = results.pose_landmarks.landmark
				# 左側 landmark idx
				left_idxs = [mp_pose.PoseLandmark.LEFT_HIP,
							mp_pose.PoseLandmark.LEFT_KNEE,
							mp_pose.PoseLandmark.LEFT_ANKLE]
				# 右側 landmark idx
				right_idxs = [mp_pose.PoseLandmark.RIGHT_HIP,
							mp_pose.PoseLandmark.RIGHT_KNEE,
							mp_pose.PoseLandmark.RIGHT_ANKLE]
				
				# 繪製左右腿骨架(點位顏色不同)
				draw_lags(image, left_idxs, right_idxs, lm, w, h)
				
			except Exception:
				pass
			
			# 弓箭步記數 
			count_left, count_right = lunge_count(side, prev_state_left, prev_state_right, count_left, count_right)


		# 顯示判定結果
		draw_result(image, side, count_left, count_right)
		# cv2.putText(image, f"Lunge L:{count_left} R:{count_right}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
		# if side:
		# 	print("side is Ture!")
		# 	cv2.putText(image, f"Detected: {side}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,200,255), 5)
		# else:
		# 	cv2.putText(image, f"Detected: None", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,200,255), 5)

		# 顯示畫面
		cv2.imshow("Lunge Detector", image)
		key = cv2.waitKey(1) & 0xFF
		if key == ord('q'):
			break

	pose.close()
	cap.release()
	cv2.destroyAllWindows()

	# optional cleanup: delete input file if asked (path provided)
	if delete_input_after:
		try:
			os.remove(delete_input_after)
		except Exception:
			pass

def main():
	# 互動式選單：1 相機模式、2 影片模式
	print("選擇模式：")
	print("1. 相機模式")
	print("2. 影片輸入模式")
	choice = input("輸入數字 (1 或 2)：").strip()

	if choice == "1":
		cap = cv2.VideoCapture(0) # 開啟相機
		if not cap.isOpened():
			print("無法開啟相機")
			return
		process_stream(cap)
	elif choice == "2":
		video_path = input("請輸入影片檔案路徑：").strip().strip('"')
		if not video_path:
			print("未提供影片路徑，結束。")
			return
		if not os.path.exists(video_path):
			print("找不到影片檔：", video_path)
			return
		# 使用預設 recent_seconds = 300s 判斷是否刪除
		recent_seconds = 300
		mtime = os.path.getmtime(video_path)
		age_seconds = time.time() - mtime
		remove_after = None
		if age_seconds <= recent_seconds:
			remove_after = video_path
		cap = cv2.VideoCapture(video_path)
		if not cap.isOpened():
			print("無法開啟影片：", video_path)
			return
		process_stream(cap, delete_input_after=remove_after)
		# 再次嘗試刪除（若先前被鎖住）
		if remove_after and os.path.exists(remove_after):
			try:
				os.remove(remove_after)
			except Exception:
				pass
	else:
		print("不正確的選項，請輸入 1 或 2。")

if __name__ == "__main__":
	main()