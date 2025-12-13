from models.squat_detector_with_standard import SquatDetectorWithStandard

_detector = None

def init_detector(standard_sequence, squat_threshold=100, similarity_threshold=0.6):
	"""初始化或重建全域偵測器，回傳該偵測器"""
	global _detector
	if _detector is not None:
		try:
			_detector.cleanup()
		except Exception:
			pass
	_detector = SquatDetectorWithStandard(
		standard_sequence=standard_sequence,
		squat_threshold=squat_threshold,
		similarity_threshold=similarity_threshold
	)
	return _detector

def get_detector():
	"""回傳目前的偵測器實例（若尚未初始化回傳 None）"""
	return _detector

def reset_detector():
	"""重設偵測器計數器"""
	if _detector is not None:
		try:
			_detector.reset_counters()
		except Exception:
			pass

def cleanup_detector():
	"""清理並移除偵測器實例"""
	global _detector
	if _detector is not None:
		try:
			_detector.cleanup()
		except Exception:
			pass
		finally:
			_detector = None
