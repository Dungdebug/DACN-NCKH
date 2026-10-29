"""
Document Stream Definition and Milestone Configurations for RQ4.
Specifies 21 sequential documents (D0 through D20) and milestone evaluation points:
- Milestone 0: D0 (Initial Ingestion)
- Milestone 1: D0+5 (After 5 additional documents)
- Milestone 2: D0+10 (After 10 additional documents)
- Milestone 3: D0+20 (After 20 additional documents)
"""

from typing import Dict, Any, List

MILESTONES = [0, 5, 10, 20]

STREAM_TOPICS = [
    "Quy hoạch Hạ tầng Đô thị và Giao thông Thông minh",
    "Kiến trúc Bộ nhớ Tham số Liên tục cho LLM",
    "Báo cáo Tài chính và Phân tích Rủi ro Tín dụng",
    "Hồ sơ Y khoa và Phác đồ Điều trị Tim mạch",
    "Quy chuẩn Kỹ thuật Xây dựng và Tiêu chuẩn An toàn",
    "Động lực học Dòng chảy và Khí động học Ứng dụng",
    "Luật Doanh nghiệp và Hợp đồng Thương mại Quốc tế",
    "Phân tích Chuỗi Cung ứng và Tối ưu Hóa Logistics",
    "Hệ Thống Điện Lưới Thông Minh và Năng lượng Tái tạo",
    "Bảo mật Mạng và Mật mã học Kháng Lượng tử",
    "Xử lý Tín hiệu Số và Thị giác Máy tính Công nghiệp",
    "Dược động học và Thử nghiệm Lâm sàng Giai đoạn III",
    "Quản trị Nhân sự và Văn hóa Doanh nghiệp Số",
    "Mô hình Hóa Khí hậu và Biến đổi Môi trường Biển",
    "Chế tạo Robot Tự hành và Điều khiển Thích nghi",
    "Kinh tế học Vi mô và Hành vi Tiêu dùng Cá nhân",
    "Vật liệu Bán dẫn Tiên tiến và Thiết kế Vi mạch",
    "Địa chất Thủy văn và Quản lý Tài nguyên Nước",
    "Hệ thống Thông tin Không dây 6G và Vệ tinh Tầm thấp",
    "Lý thuyết Trò chơi và Đấu thầu Điện tử Thuật toán",
    "Đạo đức Trí tuệ Nhân tạo và Quản trị Dữ liệu Lớn"
]

def get_document_stream() -> List[Dict[str, Any]]:
    docs = []
    for idx, topic in enumerate(STREAM_TOPICS):
        doc_id = f"STREAM_DOC_{idx:02d}"
        key_fact = f"FACT_CODE_{idx:02d}_TARGET_{1000 + idx * 7}"
        docs.append({
            "doc_index": idx,
            "doc_id": doc_id,
            "topic": topic,
            "target_fact": key_fact,
            "probe_question": f"Mã dữ kiện chính của tài liệu '{topic}' là gì?",
            "probe_ground_truth": key_fact
        })
    return docs
