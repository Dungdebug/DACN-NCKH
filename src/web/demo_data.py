"""
Demo Corpus Generator for Phase 5.1 Document Intelligence Web Platform.
Creates 4 distinct, rich multi-section documents with corresponding answerable
and unanswerable questions to test retrieval, memory, citations, and refusal.

Zero data leakage: Independent from Phase 4 official test benchmarks.
"""

import json
from pathlib import Path
from typing import Dict, Any, List


DEMO_DOCUMENTS = [
    {
        "id": "DEMO_DOC_001",
        "title": "Cơ chế Bộ nhớ Đa quy mô và Khối Giao tiếp Liên cấp trong Mô hình Ngôn ngữ",
        "category": "Computer Science & AI",
        "text": (
            "# 1. Giới thiệu Kiến trúc Bộ nhớ Đa quy mô\n\n"
            "Trong các mô hình ngôn ngữ hiện đại, việc xử lý văn bản dài thường gặp thách thức lớn về chi phí bộ nhớ "
            "và sự suy giảm chú ý (attention dilution). Hệ thống bộ nhớ liên tục (Continuum Memory System - CMS) "
            "giải quyết vấn đề này bằng cách thiết lập nhiều tầng bộ nhớ tham số hoạt động ở các chu kỳ thời gian khác nhau.\n\n"
            "Tầng 1 (Level 1) chịu trách nhiệm thích ứng ở cấp độ đoạn văn (paragraph), bắt giữ các chuyển tiếp cục bộ "
            "và từ vựng ngắn hạn. Tầng 2 (Level 2) cập nhật ở ranh giới phần mục (section), duy trì mạch lập luận "
            "và ngữ cảnh chủ đề. Tầng 3 (Level 3) đồng bộ ở ranh giới toàn bộ tài liệu (document), lưu trữ các thực thể cốt lõi.\n\n"
            "# 2. Cơ chế Cổng Hòa trộn Phần dư (Gated Residual Blending)\n\n"
            "Không giống các cơ chế cộng dồn đơn giản, SA-CMS áp dụng công thức cổng thích ứng để kết hợp biểu diễn chú ý "
            "của backbone và đầu ra bộ nhớ tham số. Cổng g_t nhận đầu vào là vector ẩn của tầng chú ý và điều tiết tỷ lệ "
            "thông tin bộ nhớ được đưa vào dòng biểu diễn chính. Điều này giúp ngăn ngừa hiện tượng trôi dạt biểu diễn (representation drift).\n\n"
            "# 3. Hiệu quả Tính toán và Giảm Chi phí Token\n\n"
            "Bằng cách nén thông tin lịch sử tài liệu vào các ma trận bộ nhớ, hệ thống cho phép mô hình trả lời các truy vấn "
            "mà không cần nhồi toàn bộ văn bản 10.000 token vào context window. Chi phí token đầu vào giảm hơn 80%, "
            "trong khi độ trễ phản hồi được duy trì dưới 50 mili-giây trên phần cứng tiêu chuẩn."
        ),
        "questions": [
            {
                "id": "DEMO_Q_001_1",
                "question": "Tầng 1 của hệ thống CMS đảm nhiệm vai trò gì?",
                "answerable": True,
                "expected_answer": "Tầng 1 thích ứng ở cấp độ đoạn văn, bắt giữ các chuyển tiếp cục bộ và từ vựng ngắn hạn.",
            },
            {
                "id": "DEMO_Q_001_2",
                "question": "Cơ chế cổng hòa trộn phần dư (Gated Residual Blending) giúp ngăn ngừa hiện tượng gì?",
                "answerable": True,
                "expected_answer": "Giúp ngăn ngừa hiện tượng trôi dạt biểu diễn (representation drift).",
            },
            {
                "id": "DEMO_Q_001_3",
                "question": "Hệ thống CMS sử dụng vi xử lý lượng tử loại nào để mã hóa bộ nhớ?",
                "answerable": False,
                "expected_answer": "Tài liệu không cung cấp thông tin về vi xử lý lượng tử (Refusal required).",
            }
        ]
    },
    {
        "id": "DEMO_DOC_002",
        "title": "Quy hoạch Hạ tầng Đường sắt Tốc độ cao Bắc - Nam và Giao thông Đô thị 2030",
        "category": "Infrastructure & Transportation",
        "text": (
            "# 1. Tổng quan Dự án Đường sắt Tốc độ cao\n\n"
            "Dự án đường sắt tốc độ cao trục Bắc - Nam có tổng chiều dài tuyến khoảng 1.541 km, đi qua 20 tỉnh và thành phố. "
            "Tuyến bắt đầu từ ga Ngọc Hồi (Hà Nội) và kết thúc tại ga Thủ Thiêm (Thành phố Hồ Chí Minh). "
            "Toàn tuyến được thiết kế theo tiêu chuẩn đường đôi, khổ đường tiêu chuẩn 1.435 mm, điện khí hóa với tốc độ thiết kế 350 km/h.\n\n"
            "# 2. Phân kỳ Đầu tư và Phương án Vận hành\n\n"
            "Dự án được phân kỳ thực hiện với mục tiêu khởi công các đoạn ưu tiên trước năm 2027. "
            "Đoạn Hà Nội - Vinh và đoạn Nha Trang - Thành phố Hồ Chí Minh sẽ được triển khai thi công đồng loạt. "
            "Mục tiêu đến năm 2035 sẽ hoàn thành và đưa vào khai thác toàn bộ tuyến đường sắt cao tốc.\n\n"
            "# 3. Tác động Kinh tế - Xã hội và Kết nối Đa phương thức\n\n"
            "Hệ thống đường sắt tốc độ cao sẽ kết nối trực tiếp với 23 ga hành khách và 5 ga hàng hóa chiến lược. "
            "Thời gian di chuyển giữa Hà Nội và TP.HCM dự kiến rút ngắn xuống còn khoảng 5 giờ 30 phút. "
            "Công trình dự kiến đóng góp tăng trưởng GDP khoảng 1% mỗi năm trong giai đoạn thi công cao điểm."
        ),
        "questions": [
            {
                "id": "DEMO_Q_002_1",
                "question": "Tuyến đường sắt tốc độ cao Bắc - Nam có chiều dài bao nhiêu km và tốc độ thiết kế là bao nhiêu?",
                "answerable": True,
                "expected_answer": "Chiều dài tuyến khoảng 1.541 km, tốc độ thiết kế là 350 km/h.",
            },
            {
                "id": "DEMO_Q_002_2",
                "question": "Ga đầu và ga cuối của tuyến đường sắt tốc độ cao là những ga nào?",
                "answerable": True,
                "expected_answer": "Bắt đầu từ ga Ngọc Hồi (Hà Nội) và kết thúc tại ga Thủ Thiêm (TP.HCM).",
            },
            {
                "id": "DEMO_Q_002_3",
                "question": "Mức giá vé hạng thương gia từ Hà Nội đi Đà Nẵng là bao nhiêu triệu đồng?",
                "answerable": False,
                "expected_answer": "Tài liệu không đề cập đến mức giá vé cụ thể (Refusal required).",
            }
        ]
    },
    {
        "id": "DEMO_DOC_003",
        "title": "Phác đồ Quản lý Đái tháo đường Tuýp 2 và Kiểm soát Biến chứng Tim mạch",
        "category": "Biomedical & Clinical",
        "text": (
            "# 1. Tiêu chuẩn Chẩn đoán và Mục tiêu HbA1c\n\n"
            "Chẩn đoán đái tháo đường tuýp 2 dựa trên nồng độ glucose huyết tương lúc đói từ 126 mg/dL (7.0 mmol/L) trở lên, "
            "hoặc chỉ số HbA1c từ 6.5% trở lên qua xét nghiệm chuẩn hóa NGSP. "
            "Mục tiêu điều trị chung cho người trưởng thành không mang thai là duy trì mức HbA1c dưới 7.0% nhằm giảm thiểu "
            "nguy cơ biến chứng mạch máu nhỏ và mạch máu lớn.\n\n"
            "# 2. Lựa chọn Thuốc Điều trị Ban đầu và Phối hợp\n\n"
            "Metformin vẫn là lựa chọn đầu tay trong đơn trị liệu nếu bệnh nhân không có chống chỉ định suy thận nặng (eGFR < 30 mL/phút). "
            "Đối với bệnh nhân có bệnh tim mạch xơ vữa đã xác định hoặc có nguy cơ tim mạch rất cao, "
            "khuyến cáo phối hợp sớm với nhóm thuốc ức chế SGLT2 (SGLT2i) hoặc đồng vận thụ thể GLP-1 (GLP-1 RA).\n\n"
            "# 3. Theo dõi Chức năng Thận và Đáy mắt Định kỳ\n\n"
            "Bệnh nhân cần được định lượng tỷ lệ Albumin/Creatinine niệu (uACR) và ước tính eGFR ít nhất 1 lần mỗi năm. "
            "Khám chuyên khoa mắt soi đáy mắt cần được thực hiện ngay tại thời điểm chẩn đoán và định kỳ hàng năm "
            "để tầm soát sớm bệnh võng mạc đái tháo đường."
        ),
        "questions": [
            {
                "id": "DEMO_Q_003_1",
                "question": "Chỉ số HbA1c từ bao nhiêu phần trăm trở lên thì đạt tiêu chuẩn chẩn đoán đái tháo đường?",
                "answerable": True,
                "expected_answer": "Chỉ số HbA1c từ 6.5% trở lên.",
            },
            {
                "id": "DEMO_Q_003_2",
                "question": "Thuốc nào là lựa chọn đầu tay trong đơn trị liệu đái tháo đường tuýp 2?",
                "answerable": True,
                "expected_answer": "Metformin là lựa chọn đầu tay nếu không có chống chỉ định suy thận nặng.",
            },
            {
                "id": "DEMO_Q_003_3",
                "question": "Bệnh nhân có được khuyến nghị phẫu thuật ghép thận nhân tạo sinh học ngay trong tuần đầu tiên không?",
                "answerable": False,
                "expected_answer": "Tài liệu không đề cập đến ghép thận nhân tạo sinh học (Refusal required).",
            }
        ]
    },
    {
        "id": "DEMO_DOC_004",
        "title": "Khung Quản trị An toàn Thông tin Số và Tuân thủ Bảo vệ Dữ liệu Cá nhân",
        "category": "Cybersecurity & Law",
        "text": (
            "# 1. Nguyên tắc Xử lý Dữ liệu Cá nhân Hợp pháp\n\n"
            "Dữ liệu cá nhân phải được xử lý trên cơ sở có sự đồng thuận rõ ràng của chủ thể dữ liệu, "
            "trừ các trường hợp khẩn cấp đe dọa đến tính mạng hoặc an ninh quốc gia. "
            "Bên kiểm soát dữ liệu có trách nhiệm áp dụng các biện pháp kỹ thuật và tổ chức phù hợp "
            "để đảm bảo tính bảo mật, tính toàn vẹn và tính khả dụng của dữ liệu.\n\n"
            "# 2. Yêu cầu Báo cáo Sự cố An ninh Mạng\n\n"
            "Khi phát hiện sự cố rò rỉ dữ liệu hoặc xâm nhập trái phép, bên xử lý dữ liệu phải thông báo cho cơ quan "
            "quản lý có thẩm quyền trong thời hạn tối đa 72 giờ kể từ thời điểm phát hiện sự cố. "
            "Báo cáo phải mô tả bản chất của sự cố, số lượng chủ thể dữ liệu bị ảnh hưởng và các biện pháp khắc phục tức thời.\n\n"
            "# 3. Đánh giá Tác động Xử lý Dữ liệu (DPIA)\n\n"
            "Hồ sơ đánh giá tác động xử lý dữ liệu cá nhân phải được lập và lưu trữ thường xuyên. "
            "Doanh nghiệp phải chỉ định Bộ phận hoặc Nhân sự chuyên trách bảo vệ dữ liệu (DPO) "
            "để giám sát việc tuân thủ các quy định pháp luật và phối hợp với thanh tra chuyên ngành."
        ),
        "questions": [
            {
                "id": "DEMO_Q_004_1",
                "question": "Thời hạn tối đa để thông báo cho cơ quan quản lý khi phát hiện sự cố rò rỉ dữ liệu là bao lâu?",
                "answerable": True,
                "expected_answer": "Thời hạn tối đa là 72 giờ kể từ thời điểm phát hiện sự cố.",
            },
            {
                "id": "DEMO_Q_004_2",
                "question": "Doanh nghiệp cần chỉ định chức danh nào để giám sát việc tuân thủ bảo vệ dữ liệu?",
                "answerable": True,
                "expected_answer": "Nhân sự chuyên trách bảo vệ dữ liệu (DPO).",
            },
            {
                "id": "DEMO_Q_004_3",
                "question": "Mức phạt tiền tối đa là bao nhiêu tỷ đồng đối với vi phạm của ngân hàng quốc tế?",
                "answerable": False,
                "expected_answer": "Tài liệu không quy định mức phạt tiền cụ thể (Refusal required).",
            }
        ]
    }
]


def seed_demo_documents(store_dir: str = "data/document_store") -> List[str]:
    """
    Seeds the demo documents into the DocumentStore and chunker index.
    Returns list of seeded document IDs.
    """
    from src.hybrid_qa.document_store import DocumentStore
    from src.hybrid_qa.chunker import DocumentChunker

    store = DocumentStore(store_dir=store_dir)
    chunker = DocumentChunker(chunk_size=256, chunk_overlap=32)
    seeded_ids = []

    for item in DEMO_DOCUMENTS:
        doc_id = item["id"]
        title = item["title"]
        raw_text = item["text"]
        metadata = {
            "category": item["category"],
            "is_demo": True,
            "questions": item["questions"],
        }

        # Check if already present
        if doc_id in store._index:
            doc_record = store.get_document(doc_id)
        else:
            doc_record = store.add_document(
                title=title,
                raw_text=raw_text,
                document_id=doc_id,
                metadata=metadata,
            )

        # Ensure passages are chunked
        if not doc_record.passages:
            passages = chunker.chunk_document(doc_record)
            doc_record.passages = passages
            doc_file = store.docs_dir / doc_id / f"v{doc_record.version}.json"
            with open(doc_file, "w", encoding="utf-8") as f:
                json.dump(doc_record.to_dict(), f, indent=2, ensure_ascii=False)

        seeded_ids.append(doc_id)

    return seeded_ids
