import os
import csv
from datetime import datetime

# ===== 收集用戶基本資料（互動輸入）=====
def collect_user_data():
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

# ===== 儲存測試結果到 CSV 檔案（會自動建立 data 資料夾）=====
def save_test_result(name, gender, age, completion_time, mode):
    data_folder = "data"
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        print(f"已創建 {data_folder} 資料夾")
    csv_file_path = os.path.join(data_folder, "squat_timing_results.csv")
    file_exists = os.path.exists(csv_file_path)
    try:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row_data = [current_time, name, gender, age, f"{completion_time:.2f}", mode]
        with open(csv_file_path, 'a', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                headers = ['測試時間', '姓名', '性別', '年齡', '完成秒數', '測試模式']
                writer.writerow(headers)
            writer.writerow(row_data)
        print(f"✅ 測試結果已儲存到: {csv_file_path}")
        return True
    except Exception as e:
        print(f"❌ 儲存測試結果時發生錯誤: {e}")
        return False
