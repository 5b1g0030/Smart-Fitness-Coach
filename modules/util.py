import os
import cv2

def open_camera(device=0, width=920, height=540, fourcc='MJPG', backend=cv2.CAP_DSHOW):
    """開啟並設定攝像頭，失敗則拋例外"""
    cap = cv2.VideoCapture(device, backend)
    # 設定解析度與編碼
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    if not cap.isOpened():
        raise Exception("無法開啟攝像頭")
    return cap

def release_camera(cap):
    """釋放攝像頭並關閉視窗（安全呼叫）"""
    try:
        if cap is not None and hasattr(cap, "release"):
            cap.release()
    except Exception:
        pass
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

# 新增：影片讀取輔助
def open_video(video_path):
    """檢查檔案存在並開啟影片，回傳 cv2.VideoCapture，失敗拋例外或回傳 None"""
    if not video_path:
        raise ValueError("未提供影片路徑")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"影片文件不存在: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"無法開啟影片: {video_path}")
    return cap

def get_video_info(cap):
    """由已開啟的 cv2.VideoCapture 取得 fps, total_frames, duration"""
    if cap is None:
        return 0.0, 0, 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = (total_frames / fps) if fps > 0 else 0.0
    return fps, total_frames, duration

def release_video(cap):
    """安全釋放影片資源"""
    try:
        if cap is not None and hasattr(cap, "release"):
            cap.release()
    except Exception:
        pass
