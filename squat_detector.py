import cv2              # 存取攝像頭、讀取和顯示影像、處理影像（如翻轉、加文字等）
# 引入所有模式
from modules.squat_mod.squat_modes import (load_standard_sequence_mode, camera_detection_mode,
                                   analyze_standard_video_mode, test_video_analysis_mode,
                                   squat_timing_mode) 



# ===== 顯示功能選單 ====== 
# 顯示系統操作提示
# ======================== 
def show_menu():
    print("="*60)
    print("        深蹲姿勢檢測程式 - 功能選單")
    print("="*60)
    print("1. 即時攝像頭檢測 (需要先有標準動作資料)")
    print("2. 分析標準深蹲影片")
    print("3. 測試影片分析")
    print("5. 測試幾秒完成深蹲")
    print("6. 退出程式")
    print("="*60)

# ===== 主函式 =====
def main():
    
    print("歡迎使用深蹲姿勢檢測程式！")
    
    # 儲存標準動作序列
    standard_sequence = None

    # 載入現有的標準動作資料
    result = load_standard_sequence_mode() # 呼叫「載入現有標準動作資料模式函式」
    # 檢查分析標準影片或載入標準資料是否成功
    if result:
        standard_sequence = result # 如果成功，將分析或載入得到的標準動作序列資料儲存
        print("標準動作資料載入成功")
    else:
        print("標準動作資料載入失敗...")
            
    
    while True:
        try:
            show_menu() # 呼叫「顯示功能選單函式」
            choice = input("\n請選擇功能 (1-6): ").strip() # 使用者輸入(去空白)
            
            # 1. 即時攝像頭檢測
            if choice == '1':
                # 檢查「標準動作序列」是否有資料
                if not standard_sequence:
                    print("\n錯誤：沒有標準動作資料！")
                    print("請先選擇功能 2 分析標準影片，或選擇功能 4 載入現有資料")
                    continue # 跳出 if 避免誤執行後續函式導致錯誤
                
                camera_detection_mode(standard_sequence) # 呼叫「攝像頭即時檢測模式函式」

            # 2. 分析標準深蹲影片   
            elif choice == '2':
                result = analyze_standard_video_mode() # 呼叫「分析標準深蹲影片模式函式」
                # 檢查分析標準影片或載入標準資料是否成功
                if result:
                    standard_sequence = result # 如果成功，將分析或載入得到的標準動作序列資料儲存
                    print("標準動作資料已更新，現在可以使用其他功能了！")
                
            # 3. 輸入影片進行分析
            elif choice == '3':
                # 測試影片分析
                test_video_analysis_mode(standard_sequence) # 呼叫「測試影片分析模式函式」
                
            # 5. 測試幾秒完成深蹲
            elif choice == '5':
                squat_timing_mode(standard_sequence) # 呼叫「測試深蹲完成時間模式函式」
            
            # 6. 退出程式
            elif choice == '6':
                print("感謝使用深蹲姿勢檢測程式！")
                break # 跳出無限迴圈
            
            # 例外輸入處理  
            else:
                print("無效選擇，請輸入 1-6 的數字")
        # 例外: 使用者在終端機按下 Ctrl+C（或其他中斷鍵），主動中斷程式執行        
        except KeyboardInterrupt:
            print("\n\n程式被使用者中斷")
            break
        # 程式執行過程中發生任何未預期的錯誤
        except Exception as e:
            print(f"\n程式執行時發生錯誤: {e}")
            print("請重新選擇功能或聯絡開發者")
    
    # ===== 程式結束清理 =====
    cv2.destroyAllWindows() # 關閉所有可能還開啟的 OpenCV 視窗


if __name__ == '__main__':
    main()