import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ===== 繪製中文字（保留原名稱）=====
def draw_chineese_text(frame, x_y, text, font_size=32, color=(255,255,255)):
	# 1. 轉成 PIL 影像
	img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
	draw = ImageDraw.Draw(img_pil)

	# 2. 指定中文字型（請確認字型檔案存在）
	font_path = "msjh.ttc"
	try:
		font = ImageFont.truetype(font_path, font_size)
	except Exception:
		font = ImageFont.load_default()

	# 3. 畫字 (PIL 用 RGB)
	draw.text(x_y, text, font=font, fill=(color[2], color[1], color[0]))

	# 4. 轉回 OpenCV 格式
	frame[:,:,:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ===== 繪製狀態資訊（由 detector 傳入必要數值）=====
def draw_status_info(frame, is_squat, left_angle, right_angle, similarity, feedback, has_pose, squat_count=0, correct_squat_count=0):
	h, w, _ = frame.shape

	if has_pose:
		if is_squat:
			draw_chineese_text(frame, (10, 0), "深蹲", font_size=100, color=(0, 255, 0))
		else:
			draw_chineese_text(frame, (10, 0), "站立", font_size=100, color=(255, 0, 0))

		# 計數資訊
		draw_chineese_text(frame, (10, 120), f"總次數: {squat_count}", font_size=40, color=(255, 255, 0))
		draw_chineese_text(frame, (10, 160), f"正確: {correct_squat_count}", font_size=40, color=(0, 255, 0))

		# 相似度顏色
		sim_color = (0, 255, 0) if similarity > 0.8 else (0, 255, 255) if similarity > 0.6 else (0, 0, 255)
		draw_chineese_text(frame, (10, 200), f"準確率: {similarity:.2f}", font_size=40, color=sim_color)

		# 回饋
		draw_chineese_text(frame, (10, 300), f"{feedback}", font_size=40, color=(10,10,10))

		# 角度資訊（保留使用 cv2.putText）
		cv2.putText(frame, f"L: {int(left_angle)} deg  R: {int(right_angle)} deg", (10, 360),
					cv2.FONT_HERSHEY_SIMPLEX, 1, (10,10,10), 2)
	else:
		cv2.putText(frame, "No Pose Detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)


# ===== 畫面上顯示「偵測到深蹲」等結果（影片/測試模式）=====
def draw_squat_detected(frame, elapsed_time=None):
	if elapsed_time is None:
		elapsed_text = ""
	else:
		elapsed_text = f"Time: {elapsed_time:.2f} seconds"
	cv2.putText(frame, "SQUAT DETECTED!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)
	if elapsed_text:
		cv2.putText(frame, elapsed_text, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)
	cv2.putText(frame, "Press any key to exit", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)


# ===== 畫面底部進度與提示 =====
def draw_progress(frame, progress, current_time, duration):
	h = frame.shape[0]
	cv2.putText(frame, f"Progress: {progress:.1f}% ({current_time:.1f}s/{duration:.1f}s)", 
				(10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (131,131,131), 2)
	cv2.putText(frame, "Press 'p' or SPACE to pause, 'q' to quit", 
				(10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (131,131,131), 2)


# ===== 計時中顯示 =====
def draw_timing_overlay(frame, current_elapsed_time):
	cv2.putText(frame, f"TIMING... {current_elapsed_time:.1f}s", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,0), 3)
	cv2.putText(frame, "Perform a squat!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2)


# ===== 等待開始提示 =====
def draw_press_start_hint(frame):
	cv2.putText(frame, "Press 's' to start timing test", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2)
	cv2.putText(frame, "Stand ready for squat", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200,200,200), 2)


# ===== 底部退出提示 =====
def draw_quit_hint(frame):
	h = frame.shape[0]
	cv2.putText(frame, "Press 'q' to quit", (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100,100,100), 2)
