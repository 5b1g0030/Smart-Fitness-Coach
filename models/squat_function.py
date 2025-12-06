import cv2      # 存取攝像頭、讀取和顯示影像、處理影像（如翻轉、加文字等）
import os       # 檢查檔案路徑、建立、儲存檔案
import csv      # 儲存csv檔案
import datetime # 計時器
from models.squat import SquatDetectorWithStandard, StandardSquatAnalyzer # 引入深蹲偵測器

# ===== 攝像頭即時檢測模式 =====
def camera_detection_mode(standard_sequence):
    
    print("\n=== 攝像頭即時檢測模式 ===")
    
    # ----- 初始化攝像頭 -----
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    # ----- 設定攝像頭解析度 (可選) -----
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    
    # ----- 檢查攝像頭是否正常開啟 -----
    if not cap.isOpened():
        raise Exception("無法開啟攝像頭")
    
    # ----- 初始化檢測器 -----
    detector = SquatDetectorWithStandard(
        standard_sequence=standard_sequence,
        squat_threshold=120,
        similarity_threshold=0.6
    )
    
    # ----- 顯示操作提示 -----
    print("\n使用說明：")
    print("- 站在攝像頭前方，確保全身都在畫面內")
    print("- 程式會比較您的動作與標準動作")
    print("- 綠色相似度 > 0.8，黃色 0.6-0.8，紅色 < 0.6")
    print("- 按 'q' 鍵退出程式")
    print("- 按 'r' 鍵重設計數")
    print("-" * 50)
    
    try:
        # ----- 主要檢測迴圈 -----
        while cap.isOpened():
            # ----- 讀取攝像頭畫面 -----
            ret, frame = cap.read()
            
            # ----- 檢查有無讀取到畫面 -----
            if not ret:
                print("無法讀取攝像頭")
                break
            
            # ----- 處理影格 ----- 
            processed_frame = detector.process_frame(frame, mirror=True)
            
            # ----- 設定視窗可調整大小並顯示畫面 -----
            cv2.namedWindow('Camera Squat Detection', cv2.WINDOW_NORMAL)
            cv2.imshow('Camera Squat Detection', processed_frame)
            
            # ------ 處理按鍵事件 -----
            key = cv2.waitKey(1) & 0xFF 
            if key == ord('q'):
                print("使用者按下 'q' 鍵，退出攝像頭模式")
                break
            elif key == ord('r'):
                detector.reset_counters()
    
    finally:
        # ===== 釋放資源並顯示結果 =====
        print(f"攝像頭模式結束")
        print(f"總深蹲次數: {detector.squat_count}")
        print(f"正確深蹲次數: {detector.correct_squat_count}")
        if detector.squat_count > 0:
            accuracy = detector.correct_squat_count / detector.squat_count * 100
            print(f"正確率: {accuracy:.1f}%")
        
        cap.release()
        detector.cleanup()
        cv2.destroyAllWindows()

# ===== 分析標準深蹲影片模式 =====
# 1. 輸入影片
# 2. 檢查是否有路徑 -> none
# 2. 檢查影片是否存在 -> none
# 3. 分析標準影片
# ============================== 
def analyze_standard_video_mode():
    
    print("\n=== 分析標準深蹲影片模式 ===")
    
    # ----- 輸入影片 -----
    # strip() => 去除使用者輸入內容前後的空白或換行，確保取得乾淨的路徑字串
    # ------------------- 
    video_path = input("請輸入標準深蹲影片路徑: ").strip()

    # ----- 檢查是否有路徑 -----
    if not video_path:
        print("未提供影片路徑，返回主選單...")
        return None # 停止函式
    
    # ----- 檢查影片是否存在 -----
    if not os.path.exists(video_path): # 根據路徑去驗證檔案或資料是否存在
        print(f"影片文件不存在: {video_path}")
        return None # 停止函式
    
    try:
        # ----- 分析標準影片 -----
        analyzer = StandardSquatAnalyzer() # 引入「標準深蹲動作分析器(類別)」
        standard_sequence = analyzer.analyze_standard_video(video_path) # 呼叫「提取關鍵角度；返回包含關鍵角度的字典函式」
        analyzer.cleanup() # 呼叫「清理資源函式」
        
        print("標準影片分析完成！")
        return standard_sequence # 回傳「標準動作序列」
    # 例外處理 
    except Exception as e:
        print(f"分析標準影片時發生錯誤: {e}")
        return None
    

# ===== 測試影片分析模式 =====
# 1. 檢查有沒有標準動作資料 
# 2. 輸入影片
# 3. 檢查影片是否存在
# 4. 初始化影片讀取
# 5. 檢查影片是否能讀取
# 6. 初始化檢測器
# 7. 獲取影片資訊
# 8. 主要分析迴圈(...)
# =========================== 
def test_video_analysis_mode(standard_sequence):
    
    print("\n=== 測試影片分析模式 ===")
    
    # ----- 檢查有沒有標準動作資料 -----
    if not standard_sequence:
        print("錯誤：沒有標準動作資料，請先分析標準影片或載入標準動作資料")
        return
    
    # ----- 輸入影片 -----
    # strip() => 去除使用者輸入內容前後的空白或換行，確保取得乾淨的路徑字串
    # ------------------- 
    video_path = input("請輸入測試影片路徑: ").strip()
    # ----- 輸入測試者資料 -----
    name = input("請輸入測試者名稱(中英文皆可): ").strip()
    gender = input("請輸入測試者性別(男/女): ").strip()
    age = int(input("請輸入測試者年齡(數字): "))
    if not video_path:
        print("未提供影片路徑，返回主選單")
        return
    
    # ----- 檢查影片是否存在 -----
    if not os.path.exists(video_path):
        print(f"影片文件不存在: {video_path}")
        return
    
    # ----- 初始化影片讀取 -----
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"無法開啟測試影片: {video_path}")
        return
    
    # ----- 初始化檢測器 -----
    detector = SquatDetectorWithStandard(
        standard_sequence=standard_sequence,
        squat_threshold=120,
        similarity_threshold=0.6
    )
    
    # 文字提示
    print("\n使用說明：")
    print("- 程式將分析測試影片中的深蹲動作")
    print("- 會比較測試動作與標準動作的相似度")
    print("- 按 'q' 鍵退出分析")
    print("- 按 'r' 鍵重設計數")
    print("- 按 'p' 鍵暫停/繼續播放")
    print("- 按空白鍵暫停/繼續播放")
    print("-" * 50)
    
    # ----- 獲取影片資訊 -----
    fps = cap.get(cv2.CAP_PROP_FPS) # 影片幀數
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # 影片總幀數
    duration = total_frames / fps if fps > 0 else 0 # 計算影片總時長（秒）
    
    # 顯示影片資訊
    print(f"影片資訊：")
    print(f"- FPS: {fps:.2f}")
    print(f"- 總幀數: {total_frames}")
    print(f"- 時長: {duration:.2f}秒")
    print("-" * 50)
    
    frame_count = 0 # 初始化影格計時數器，已處理的影片影格數，從 0 開始
    paused = False # 初始化暫停狀態，預設「未暫停」
    
    try:
        # ----- 主要分析迴圈 -----
        # detector => 深蹲檢測器實例
        # cap => 影片物件
        # 流程:
        # 1. 讀取影片畫面
        # 2. 檢查有無讀取到畫面
        # 3. 處理影格
        # 4. 顯示畫面和進度
        # 5. 處理按鍵事件
        # 6. 釋放資源並顯示結果 
        # ------------------------
        while cap.isOpened():
            if not paused:
                # ----- 讀取影片畫面 -----
                # ret => 布林值，表示是否成功讀取到影像
                # frame => Numpy 陣列，讀取到的影像資料(如果 ret 是 False，frame 會是 None)
                # -------------------------
                ret, frame = cap.read()
                
                # ----- 檢查有無讀取到畫面 -----
                if not ret:
                    print("影片分析完成")
                    break # 跳出迴圈(不繼續以下流程)
                
                frame_count += 1
                current_time = frame_count / fps if fps > 0 else 0
                
                # 記錄開始時間（第一幀）
                if start_time is None:
                    start_time = current_time
                    print("開始分析...")
                
                # ----- 處理影格並檢測深蹲 -----
                processed_frame = detector.process_frame(frame, mirror=False, show_info=False)
                
                # ----- 檢查是否檢測到深蹲 -----
                if detector.squat_count > 0:
                    squat_detected = True
                    elapsed_time = current_time - start_time
                    
                    # 在畫面上顯示結果
                    cv2.putText(processed_frame, f"SQUAT DETECTED!", 
                               (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                    cv2.putText(processed_frame, f"Time: {elapsed_time:.2f} seconds", 
                               (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    cv2.putText(processed_frame, "Press any key to exit", 
                               (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    # 設定視窗可調整大小並顯示最終畫面
                    cv2.namedWindow('Squat Timing Test - Video Mode', cv2.WINDOW_NORMAL)
                    cv2.imshow('Squat Timing Test - Video Mode', processed_frame)
                    
                    # 在終端顯示結果
                    print(f"\n🎉 深蹲動作檢測成功！")
                    print(f"⏱️  完成時間: {elapsed_time:.2f} 秒")
                    print(f"📊 總處理幀數: {frame_count}")
                    
                    # 儲存測試結果
                    save_test_result(name, gender, age, elapsed_time, "影片模式")
                    
                    print("按任意鍵關閉視窗...")
                    
                    # 等待使用者按鍵後退出
                    cv2.waitKey(0)
                    break
                
                # ----- 在畫面上顯示進度資訊 -----
                progress = frame_count / total_frames * 100 if total_frames > 0 else 0
                current_time = frame_count / fps if fps > 0 else 0
                
                # 顯示目前分析進度（百分比和秒數），方便使用者了解影片播放狀態
                cv2.putText(processed_frame, f"Progress: {progress:.1f}% ({current_time:.1f}s/{duration:.1f}s)", 
                           (10, processed_frame.shape[0] - 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (131,131,131), 2)
                
                # 顯示的提示文字，告訴使用者如何暫停或退出影片分析
                cv2.putText(processed_frame, "Press 'p' or SPACE to pause, 'q' to quit", 
                           (10, processed_frame.shape[0] - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (131,131,131), 2)
            
            # ----- 縮放畫面 -----
            # None => 目標尺寸（dsize），設為 None 代表用比例縮放，不直接指定寬高
            # fx => 水平方向縮放比例(寬度)
            # fy => 垂直方向縮放比例(高度)
            # ------------------- 
            scale = 0.6 # 畫面比例(原本的60%)
            processed_frame = cv2.resize(processed_frame, None, fx=scale, fy=scale)
            cv2.namedWindow('Video Squat Analysis', cv2.WINDOW_NORMAL)  # 讓視窗可調整大小
            
            # ----- 顯示畫面 -----
            # 視窗名稱、影像
            # -------------------
            cv2.imshow('Video Squat Analysis', processed_frame)
            
            # ----- 處理按鍵事件 -----
            # fps => 影片每秒幀數
            # if fps > 0 and not paused => 只有在影片有 fps 且沒暫停時，才用正常速度播放
            # else 1：如果暫停或 fps 不正確，則每次只等待 1 毫秒（讓程式能即時處理按鍵）
            # ----------------------- 
            wait_time = int(1000 / fps) if fps > 0 and not paused else 1

            # ----- 等待使用者按鍵 -----
            # wait_time => 暫停指定毫秒數 
            # & 0xFF => 只取按鍵的低 8 位元（確保跨平台一致）
            # ------------------------- 
            key = cv2.waitKey(wait_time) & 0xFF
            
            if key == ord('q'): # 按 'q' 鍵退出
                print("使用者按下 'q' 鍵，退出影片分析")
                break
            elif key == ord('r'): # 按 'r' 鍵重設計數
                detector.reset_counters()
            elif key == ord('p') or key == ord(' '): # 按 'p' 鍵或空白鍵暫停/繼續
                paused = not paused
                if paused:
                    print("影片已暫停，按 'p' 或空白鍵繼續")
                else:
                    print("影片繼續播放")
    
    finally:
        # ------ 釋放資源並顯示結果 -----
        print(f"影片分析結束")
        print(f"總深蹲次數: {detector.squat_count}")
        print(f"正確深蹲次數: {detector.correct_squat_count}")
        if detector.squat_count > 0:
            accuracy = detector.correct_squat_count / detector.squat_count * 100
            print(f"正確率: {accuracy:.1f}%")
        
        cap.release()
        detector.cleanup()
        cv2.destroyAllWindows()

# ===== 載入現有標準動作資料模式 =====
# 1. 輸入標準動作資料檔案路徑
# 2. 檢查檔案是否存在
# 3. 檢查標準序列檔案是否正常讀取
# ================================== 
def load_standard_sequence_mode():

    print("\n=== 載入標準動作資料模式 ===")
    
    # ----- 輸入標準動作資料檔案路徑 -----
    # 預設 standard_squat_sequence.json
    # ---------------------------------- 
    filename = input("請輸入標準動作資料檔案路徑 (預設: standard_squat_sequence.json): ").strip()
    if not filename: # 如果使用者沒有輸入，則使用預設路徑
        filename = "standard_squat_sequence.json"
    
    # ----- 檢查檔案是否存在 -----
    if not os.path.exists(filename):
        print(f"檔案不存在: {filename}")
        return None
    
    try:
        analyzer = StandardSquatAnalyzer() # 引入「標準深蹲動作分析器(類別)」
        # ----- 檢查標準序列檔案是否正常讀取 -----
        if analyzer.load_standard_sequence(filename): # 呼叫「從文件載入標準動作序列函式」
            standard_sequence = analyzer.standard_sequence # 取得標準動作序列資料
            analyzer.cleanup() # 呼叫「清理資源」
            return standard_sequence # 回傳「標準動作序列」
        else:
            return None # 如果載入失敗，則回傳 None，表示沒有取得標準動作資料
    # 例外處理
    except Exception as e:
        print(f"載入標準序列時發生錯誤: {e}")
        return None

# ===== 測試深蹲完成時間模式 =====
# 1. 檢查有沒有標準動作資料
# 2. 選擇測試模式（影片或即時鏡頭）
# 3. 根據選擇執行對應模式
# ================================
def squat_timing_mode(standard_sequence):
    
    print("\n=== 測試深蹲完成時間模式 ===")
    
    # ----- 檢查有沒有標準動作資料 -----
    if not standard_sequence:
        print("錯誤：沒有標準動作資料，請先分析標準影片或載入標準動作資料")
        return
    
    # ----- 收集用戶資料 -----
    name, gender, age = collect_user_data()
    print(f"\n用戶資料已登記：{name} ({gender}，{age}歲)")
    
    # ----- 顯示模式選擇 -----
    print("\n請選擇測試模式：")
    print("1. 影片輸入模式")
    print("2. 即時鏡頭模式")
    print("-" * 30)
    
    while True:
        choice = input("請選擇模式 (1-2): ").strip()
        
        if choice == '1':
            squat_timing_video_mode(standard_sequence, name, gender, age)
            break
        elif choice == '2':
            squat_timing_camera_mode(standard_sequence, name, gender, age)
            break
        else:
            print("無效選擇，請輸入 1 或 2")

# ===== 收集用戶資料 =====
def collect_user_data():
    """收集用戶基本資料"""
    print("\n=== 用戶資料登記 ===")
    
    # 收集姓名
    while True:
        name = input("請輸入姓名: ").strip()
        if name:
            break
        print("姓名不能為空，請重新輸入")
    
    # 收集性別
    while True:
        gender = input("請輸入性別 (男/女): ").strip()
        if gender in ['男', '女']:
            break
        print("請輸入 '男' 或 '女'")
    
    # 收集年齡
    while True:
        try:
            age = int(input("請輸入年齡: ").strip())
            if 1 <= age <= 120:
                break
            else:
                print("年齡請輸入 1-120 之間的數字")
        except ValueError:
            print("請輸入有效的數字")
    
    return name, gender, age

# ===== 儲存測試結果到CSV =====
def save_test_result(name, gender, age, completion_time, mode):
    """儲存測試結果到CSV檔案"""
    
    # 確保data資料夾存在
    data_folder = "data"
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        print(f"已創建 {data_folder} 資料夾")
    
    # CSV檔案路徑
    csv_file_path = os.path.join(data_folder, "squat_timing_results.csv")
    
    # 檢查檔案是否存在，決定是否需要寫入標題列
    file_exists = os.path.exists(csv_file_path)
    
    try:
        # 準備要寫入的資料
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row_data = [current_time, name, gender, age, f"{completion_time:.2f}", mode]
        
        # 寫入CSV檔案
        with open(csv_file_path, 'a', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)
            
            # 如果檔案不存在，先寫入標題列
            if not file_exists:
                headers = ['測試時間', '姓名', '性別', '年齡', '完成秒數', '測試模式']
                writer.writerow(headers)
            
            # 寫入測試結果
            writer.writerow(row_data)
        
        print(f"✅ 測試結果已儲存到: {csv_file_path}")
        return True
        
    except Exception as e:
        print(f"❌ 儲存測試結果時發生錯誤: {e}")
        return False

# ===== 深蹲計時 - 影片模式 =====
def squat_timing_video_mode(standard_sequence, name, gender, age):
    
    print("\n=== 深蹲計時 - 影片模式 ===")
    
    # ----- 輸入影片 -----
    video_path = input("請輸入測試影片路徑: ").strip()
    if not video_path:
        print("未提供影片路徑，返回主選單")
        return
    
    # ----- 檢查影片是否存在 -----
    if not os.path.exists(video_path):
        print(f"影片文件不存在: {video_path}")
        return
    
    # ----- 初始化影片讀取 -----
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"無法開啟測試影片: {video_path}")
        return
    
    # ----- 初始化檢測器 -----
    detector = SquatDetectorWithStandard(
        standard_sequence=standard_sequence,
        squat_threshold=100,
        similarity_threshold=0.6
    )
    
    # ----- 獲取影片資訊 -----
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"\n測試者：{name} ({gender}，{age}歲)")
    print("\n使用說明：")
    print("- 程式將分析影片直到檢測到第一次深蹲動作")
    print("- 檢測到深蹲後會自動停止並顯示耗時")
    print("- 按 'q' 鍵可提前退出分析")
    print(f"- 影片總時長: {duration:.2f}秒")
    print("-" * 50)
    
    frame_count = 0
    start_time = None
    squat_detected = False
    
    try:
        # ----- 主要分析迴圈 -----
        while cap.isOpened() and not squat_detected:
            # ----- 讀取影片畫面 -----
            ret, frame = cap.read()
            if not ret:
                print("影片播放完畢，未檢測到深蹲動作")
                break
            
            frame_count += 1
            current_time = frame_count / fps if fps > 0 else 0
            
            # 記錄開始時間（第一幀）
            if start_time is None:
                start_time = current_time
                print("開始分析...")
            
            # ----- 處理影格並檢測深蹲 -----
            processed_frame = detector.process_frame(frame, mirror=False, show_info=False)
            
            # ----- 檢查是否檢測到深蹲 -----
            if detector.squat_count > 0:
                squat_detected = True
                elapsed_time = current_time - start_time
                
                # 在畫面上顯示結果
                cv2.putText(processed_frame, f"SQUAT DETECTED!", 
                           (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                cv2.putText(processed_frame, f"Time: {elapsed_time:.2f} seconds", 
                           (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cv2.putText(processed_frame, "Press any key to exit", 
                           (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # 設定視窗可調整大小並顯示最終畫面
                cv2.namedWindow('Squat Timing Test - Video Mode', cv2.WINDOW_NORMAL)
                cv2.imshow('Squat Timing Test - Video Mode', processed_frame)
                
                # 在終端顯示結果
                print(f"\n🎉 深蹲動作檢測成功！")
                print(f"👤 測試者：{name} ({gender}，{age}歲)")
                print(f"⏱️  完成時間: {elapsed_time:.2f} 秒")
                print(f"📊 總處理幀數: {frame_count}")
                
                # 儲存測試結果
                save_test_result(name, gender, age, elapsed_time, "影片模式")
                
                print("按任意鍵關閉視窗...")
                
                # 等待使用者按鍵後退出
                cv2.waitKey(0)
                break
            
            # ----- 在畫面上顯示進度資訊 -----
            progress = frame_count / total_frames * 100 if total_frames > 0 else 0
            current_time = frame_count / fps if fps > 0 else 0
            
            # 顯示目前分析進度（百分比和秒數），方便使用者了解影片播放狀態
            cv2.putText(processed_frame, f"Progress: {progress:.1f}% ({current_time:.1f}s/{duration:.1f}s)", 
                       (10, processed_frame.shape[0] - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (131,131,131), 2)
            
            # 顯示的提示文字，告訴使用者如何暫停或退出影片分析
            cv2.putText(processed_frame, "Press 'p' or SPACE to pause, 'q' to quit", 
                       (10, processed_frame.shape[0] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (131,131,131), 2)
            
            # ----- 縮放畫面 -----
            # None => 目標尺寸（dsize），設為 None 代表用比例縮放，不直接指定寬高
            # fx => 水平方向縮放比例(寬度)
            # fy => 垂直方向縮放比例(高度)
            # ------------------- 
            scale = 0.6 # 畫面比例(原本的60%)
            processed_frame = cv2.resize(processed_frame, None, fx=scale, fy=scale)
            cv2.namedWindow('Video Squat Analysis', cv2.WINDOW_NORMAL)  # 讓視窗可調整大小
            
            # ----- 顯示畫面 -----
            # 視窗名稱、影像
            # -------------------
            cv2.imshow('Video Squat Analysis', processed_frame)
            
            # ----- 處理按鍵事件 -----
            # fps => 影片每秒幀數
            # if fps > 0 and not paused => 只有在影片有 fps 且沒暫停時，才用正常速度播放
            # else 1：如果暫停或 fps 不正確，則每次只等待 1 毫秒（讓程式能即時處理按鍵）
            # ----------------------- 
            wait_time = int(1000 / fps) if fps > 0 and not paused else 1

            # ----- 等待使用者按鍵 -----
            # wait_time => 暫停指定毫秒數 
            # & 0xFF => 只取按鍵的低 8 位元（確保跨平台一致）
            # ------------------------- 
            key = cv2.waitKey(wait_time) & 0xFF
            
            if key == ord('q'): # 按 'q' 鍵退出
                print("使用者按下 'q' 鍵，退出影片分析")
                break
            elif key == ord('r'): # 按 'r' 鍵重設計數
                detector.reset_counters()
            elif key == ord('p') or key == ord(' '): # 按 'p' 鍵或空白鍵暫停/繼續
                paused = not paused
                if paused:
                    print("影片已暫停，按 'p' 或空白鍵繼續")
                else:
                    print("影片繼續播放")
    
    finally:
        # ----- 釋放資源 -----
        cap.release()
        detector.cleanup()
        cv2.destroyAllWindows()
        
        if not squat_detected and frame_count > 0:
            total_analyzed_time = frame_count / fps if fps > 0 else 0
            print(f"\n分析結束，未檢測到深蹲動作")
            print(f"已分析時長: {total_analyzed_time:.2f}秒")

# ===== 深蹲計時 - 即時鏡頭模式 =====
def squat_timing_camera_mode(standard_sequence, name, gender, age):
    
    print("\n=== 深蹲計時 - 即時鏡頭模式 ===")
    
    # ----- 初始化攝像頭 -----
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    # ----- 設定攝像頭解析度 -----
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    
    # ----- 檢查攝像頭是否正常開啟 -----
    if not cap.isOpened():
        raise Exception("無法開啟攝像頭")
    
    # ----- 初始化檢測器 -----
    detector = SquatDetectorWithStandard(
        standard_sequence=standard_sequence,
        squat_threshold=100,
        similarity_threshold=0.6
    )
    
    print(f"\n測試者：{name} ({gender}，{age}歲)")
    print("\n使用說明：")
    print("- 站在攝像頭前方，確保全身都在畫面內")
    print("- 按 's' 鍵開始計時測試")
    print("- 檢測到深蹲後會顯示結果1秒，然後自動重設")
    print("- 按 'q' 鍵退出程式")
    print("-" * 50)
    
    import time
    
    timing_started = False
    start_time = None
    squat_detected = False
    result_display_time = None  # 結果顯示開始時間
    elapsed_time = 0  # 儲存完成時間
    
    try:
        # ----- 主要檢測迴圈 -----
        while cap.isOpened():
            # ----- 讀取攝像頭畫面 -----
            ret, frame = cap.read()
            if not ret:
                print("無法讀取攝像頭")
                break
            
            # ----- 處理影格 -----
            processed_frame = detector.process_frame(frame, mirror=True, show_info=True)
            
            # ----- 檢查是否在顯示結果狀態 -----
            if result_display_time is not None:
                current_time = time.time()
                # 檢查是否已顯示1秒
                if current_time - result_display_time >= 1.0:
                    # 1秒後自動重設
                    timing_started = False
                    squat_detected = False
                    start_time = None
                    result_display_time = None
                    detector.reset_counters()
                    print("🔄 測試已自動重設，按 's' 鍵開始新的計時測試")

                    # ----- 收集用戶資料 -----
                    name, gender, age = collect_user_data()
                    print(f"\n用戶資料已登記：{name} ({gender}，{age}歲)")
                else:
                    # 顯示結果畫面
                    print("SQUAT DETECTED!")
                    # cv2.putText(processed_frame, f"SQUAT DETECTED!", 
                    #            (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                    # cv2.putText(processed_frame, f"Time: {elapsed_time:.2f} seconds", 
                    #            (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            
            # ----- 檢查計時狀態 -----
            elif timing_started and not squat_detected:
                current_time = time.time()
                current_elapsed_time = current_time - start_time
                
                # ----- 檢查是否檢測到深蹲 -----
                if detector.squat_count > 0:
                    squat_detected = True
                    elapsed_time = current_elapsed_time  # 儲存完成時間
                    result_display_time = time.time()  # 開始顯示結果
                    
                    # 在終端顯示結果
                    print(f"\n🎉 深蹲動作檢測成功！")
                    print(f"👤 測試者：{name} ({gender}，{age}歲)")
                    print(f"⏱️  完成時間: {elapsed_time:.2f} 秒")
                    
                    # 儲存測試結果
                    save_test_result(name, gender, age, elapsed_time, "即時鏡頭模式")
                    
                    print("結果將顯示1秒後自動重設...")
                    
                    timing_started = False  # 停止計時
                else:
                    # 顯示計時中的狀態
                    cv2.putText(processed_frame, f"TIMING... {current_elapsed_time:.1f}s", 
                               (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3)
                    cv2.putText(processed_frame, "Perform a squat!", 
                               (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            else:
                # 等待開始狀態
                cv2.putText(processed_frame, "Press 's' to start timing test", 
                           (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
                cv2.putText(processed_frame, "Stand ready for squat", 
                           (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)
            
            # ----- 顯示操作提示 -----
            cv2.putText(processed_frame, "Press 'q' to quit", 
                       (10, processed_frame.shape[0] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
            
            # ----- 設定視窗可調整大小並顯示畫面 -----
            cv2.namedWindow('Squat Timing Test - Camera Mode', cv2.WINDOW_NORMAL)
            cv2.imshow('Squat Timing Test - Camera Mode', processed_frame)
            
            # ----- 處理按鍵事件 -----
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("使用者按下 'q' 鍵，退出計時模式")
                break
            elif key == ord('s') and not timing_started and not squat_detected and result_display_time is None:
                # 開始計時測試（只有在非計時、非顯示結果狀態才能開始）
                timing_started = True
                start_time = time.time()
                detector.reset_counters()
                print("⏱️  計時開始！請執行深蹲動作...")
    
    finally:
        # ----- 釋放資源 -----
        cap.release()
        detector.cleanup()
        cv2.destroyAllWindows()
        print("計時測試結束")
