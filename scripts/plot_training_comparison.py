import pandas as pd
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def print_training_comparison():
    csv_path = "../results/phase3_2_training_comparison.csv"
    try:
        df = pd.read_csv(csv_path)
        print("=== Bảng so sánh dữ liệu huấn luyện và thực tế (Validation) ===")
        # Selecting relevant columns for comparison
        columns_to_show = ['Training Size', 'Train Loss', 'Val Loss', 'F1', 'EM', 'Time (s)']
        print(df[columns_to_show].to_markdown(index=False))
        
        print("\n=== Phân tích ===")
        print("- Train Loss và Val Loss: Cho thấy mức độ lỗi trên tập huấn luyện so với tập thực tế (Validation).")
        print("- F1 và EM (Exact Match): Là các chỉ số đo lường hiệu suất thực tế của mô hình.")
        print("- Với số lượng mẫu tăng từ 100 lên 200, ta có thể thấy sự thay đổi về mặt thời gian và hiệu suất.")
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")

if __name__ == "__main__":
    print_training_comparison()
