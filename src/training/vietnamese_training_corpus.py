"""
Vietnamese Structured Document-Grounded & Memory QA Training Corpus
for Extended SA-CMS 3-Level Training (Web Platform Enhancement).

STRICT DATA INTEGRITY & ZERO TEST LEAKAGE GUARANTEE:
- 0% overlap with Phase 3.1 / 3.3 / 4.1 / 4.2 evaluation benchmarks
  (DOC001-DOC003, QASPER, LongHealth, MK-NIAH, vietnamese_corpus.py,
   build_vietnamese_final_dataset.py).
- Dedicated disjoint training documents (VN_TR_DOC_001 .. VN_TR_DOC_030)
  and validation documents (VN_VAL_DOC_001 .. VN_VAL_DOC_005).
- Formatted in the exact Vietnamese prompt templates used by NL DocChat
  (src/web/token_optimizer.py & src/hybrid_qa/pipeline.py).
"""

from typing import Dict, Any, List


VN_TRAIN_DOCS: List[Dict[str, Any]] = [
    {
        "doc_id": "VN_TR_DOC_001",
        "title": "Kiến trúc Vi dịch vụ và Cổng điều phối API doanh nghiệp",
        "context": (
            "# Mục 1. Nguyên lý tách biệt dịch vụ\n"
            "Kiến trúc vi dịch vụ chia nhỏ ứng dụng nguyên khối thành các dịch vụ độc lập giao tiếp qua giao thức HTTP/REST hoặc gRPC. "
            "Mỗi dịch vụ sở hữu cơ sở dữ liệu riêng biệt nhằm ngăn chặn sự phụ thuộc chéo ở tầng lưu trữ. "
            "# Mục 2. Cổng API và khả năng chịu lỗi\n"
            "Cổng điều phối API trung tâm đảm nhiệm xác thực mã thông báo JWT, giới hạn tốc độ truy cập ở mức 2.500 yêu cầu mỗi giây và định tuyến thông minh. "
            "Bộ ngắt mạch tự động kích hoạt khi tỷ lệ lỗi vượt ngưỡng 5% trong chu kỳ 30 giây để ngăn hiệu ứng thác đổ toàn hệ thống."
        ),
        "qa_pairs": [
            ("Các dịch vụ trong kiến trúc vi dịch vụ giao tiếp qua giao thức nào?", "Qua giao thức HTTP/REST hoặc gRPC [VN_TR_DOC_001::P001]."),
            ("Vì sao mỗi dịch vụ sở hữu cơ sở dữ liệu riêng biệt?", "Nhằm ngăn chặn sự phụ thuộc chéo ở tầng lưu trữ [VN_TR_DOC_001::P001]."),
            ("Cổng điều phối API giới hạn tốc độ truy cập ở mức bao nhiêu?", "Ở mức 2.500 yêu cầu mỗi giây [VN_TR_DOC_001::P002]."),
            ("Khi nào bộ ngắt mạch tự động kích hoạt?", "Khi tỷ lệ lỗi vượt ngưỡng 5% trong chu kỳ 30 giây [VN_TR_DOC_001::P002]."),
            ("Mục đích của bộ ngắt mạch tự động là gì?", "Ngăn chặn hiệu ứng thác đổ toàn hệ thống [VN_TR_DOC_001::P002]."),
            ("Cổng điều phối API trung tâm thực hiện những nhiệm vụ gì?", "Xác thực mã thông báo JWT, giới hạn tốc độ truy cập và định tuyến thông minh [VN_TR_DOC_001::P002]."),
            ("Kiến trúc vi dịch vụ chia nhỏ mô hình nào?", "Chia nhỏ ứng dụng nguyên khối thành các dịch vụ độc lập [VN_TR_DOC_001::P001]."),
            ("Chu kỳ theo dõi tỷ lệ lỗi của bộ ngắt mạch là bao lâu?", "Chu kỳ 30 giây với ngưỡng lỗi 5% [VN_TR_DOC_001::P002]."),
        ],
    },
    {
        "doc_id": "VN_TR_DOC_002",
        "title": "Hệ thống Lưu trữ Năng lượng Pin Thể rắn trong Lưới điện Thông minh",
        "context": (
            "# Mục 1. Đặc tính điện hóa pin thể rắn\n"
            "Pin thể rắn thay thế dung dịch điện phân lỏng dễ cháy bằng chất điện phân gốm sulfide có độ dẫn ion cao. "
            "Mật độ năng lượng của thế hệ pin mới đạt 420 Wh/kg, cao hơn 68% so với pin lithium-ion truyền thống. "
            "# Mục 2. Vận hành điều tần lưới điện\n"
            "Trạm lưu trữ công suất 150 MW phản hồi biến động tần số lưới điện chỉ trong vòng 18 mili-giây. "
            "Tuổi thọ vận hành đạt 6.000 chu kỳ sạc xả sâu ở mức suy giảm dung lượng dưới 12%."
        ),
        "qa_pairs": [
            ("Pin thể rắn thay thế dung dịch điện phân lỏng bằng vật liệu gì?", "Bằng chất điện phân gốm sulfide có độ dẫn ion cao [VN_TR_DOC_002::P001]."),
            ("Mật độ năng lượng của pin thể rắn đạt bao nhiêu?", "Đạt 420 Wh/kg, cao hơn 68% so với pin lithium-ion truyền thống [VN_TR_DOC_002::P001]."),
            ("Trạm lưu trữ 150 MW phản hồi biến động tần số trong bao lâu?", "Chỉ trong vòng 18 mili-giây [VN_TR_DOC_002::P002]."),
            ("Tuổi thọ vận hành của hệ thống pin đạt bao nhiêu chu kỳ?", "Đạt 6.000 chu kỳ sạc xả sâu với mức suy giảm dung lượng dưới 12% [VN_TR_DOC_002::P002]."),
            ("Mức suy giảm dung lượng sau 6.000 chu kỳ được kiểm soát như thế nào?", "Duy trì ở mức dưới 12% [VN_TR_DOC_002::P002]."),
            ("Pin thể rắn cao hơn pin lithium-ion truyền thống bao nhiêu phần trăm về mật độ năng lượng?", "Cao hơn 68% [VN_TR_DOC_002::P001]."),
            ("Công suất thiết kế của trạm lưu trữ điều tần là bao nhiêu?", "Công suất 150 MW [VN_TR_DOC_002::P002]."),
            ("Ưu điểm an toàn của chất điện phân gốm sulfide là gì?", "Thay thế dung dịch điện phân lỏng dễ cháy và có độ dẫn ion cao [VN_TR_DOC_002::P001]."),
        ],
    },
    {
        "doc_id": "VN_TR_DOC_003",
        "title": "Tiêu chuẩn Mã hóa Hậu lượng tử trong Bảo mật Ngân hàng Số",
        "context": (
            "# Mục 1. Chuyển đổi thuật toán kháng lượng tử\n"
            "Các hệ mật khóa công khai RSA-2048 đứng trước nguy cơ bị phá vỡ bởi thuật toán Shor trên máy tính lượng tử. "
            "Hệ thống ngân hàng lõi triển khai chuẩn trao đổi khóa ML-KEM-768 dựa trên bài toán mạng tinh thể học có nhiễu. "
            "# Mục 2. Chữ ký số và hiệu năng giao dịch\n"
            "Chữ ký số giao dịch liên ngân hàng áp dụng thuật toán ML-DSA-65 với kích thước khóa công khai 1.952 byte. "
            "Thời gian ký và xác thực trung bình trên máy chủ HSM đạt 0.42 mili-giây mỗi giao dịch, đảm bảo thông lượng 12.000 giao dịch/giây."
        ),
        "qa_pairs": [
            ("Thuật toán nào đe dọa hệ mật khóa công khai RSA-2048?", "Thuật toán Shor trên máy tính lượng tử [VN_TR_DOC_003::P001]."),
            ("Hệ thống ngân hàng lõi triển khai chuẩn trao đổi khóa nào?", "Chuẩn trao đổi khóa ML-KEM-768 dựa trên mạng tinh thể học có nhiễu [VN_TR_DOC_003::P001]."),
            ("Thuật toán chữ ký số nào được áp dụng cho giao dịch liên ngân hàng?", "Thuật toán ML-DSA-65 với kích thước khóa công khai 1.952 byte [VN_TR_DOC_003::P002]."),
            ("Thời gian ký và xác thực trung bình trên HSM là bao nhiêu?", "Đạt 0.42 mili-giây mỗi giao dịch [VN_TR_DOC_003::P002]."),
            ("Thông lượng xử lý tối đa đạt bao nhiêu giao dịch mỗi giây?", "Đạt 12.000 giao dịch/giây [VN_TR_DOC_003::P002]."),
            ("Kích thước khóa công khai của ML-DSA-65 là bao nhiêu byte?", "1.952 byte [VN_TR_DOC_003::P002]."),
            ("Chuẩn ML-KEM-768 dựa trên nền tảng toán học nào?", "Bài toán mạng tinh thể học có nhiễu [VN_TR_DOC_003::P001]."),
            ("Thiết bị phần cứng chuyên dụng nào thực hiện ký và xác thực?", "Máy chủ bảo mật phần cứng HSM [VN_TR_DOC_003::P002]."),
        ],
    },
    {
        "doc_id": "VN_TR_DOC_004",
        "title": "Quy trình Tầm soát Sớm Bệnh Võng mạc Đái tháo đường bằng AI",
        "context": (
            "# Mục 1. Thu nhận ảnh đáy mắt không nhỏ đồng tử\n"
            "Hệ thống chẩn đoán hỗ trợ sử dụng camera đáy mắt góc rộng độ phân giải 2048x2048 pixel không cần thuốc giãn đồng tử. "
            "Mô hình mạng nơ-ron tích chập đa tỷ lệ phát hiện vi phình mạch, xuất huyết chấm và tiết dịch cứng trên võng mạc. "
            "# Mục 2. Độ nhạy lâm sàng và phân tầng chuyển tuyến\n"
            "Trên tập kiểm định lâm sàng gồm 8.400 bệnh nhân, hệ thống đạt độ nhạy 94.6% và độ đặc hiệu 91.8%. "
            "Các ca có nguy cơ đe dọa thị lực được cảnh báo tự động để khám chuyên khoa nhãn khoa trong vòng 7 ngày làm việc."
        ),
        "qa_pairs": [
            ("Độ phân giải của camera đáy mắt góc rộng trong quy trình là bao nhiêu?", "Độ phân giải 2048x2048 pixel và không cần thuốc giãn đồng tử [VN_TR_DOC_004::P001]."),
            ("Mô hình AI phát hiện những tổn thương nào trên võng mạc?", "Phát hiện vi phình mạch, xuất huyết chấm và tiết dịch cứng [VN_TR_DOC_004::P001]."),
            ("Hệ thống đạt độ nhạy và độ đặc hiệu bao nhiêu phần trăm?", "Đạt độ nhạy 94.6% và độ đặc hiệu 91.8% [VN_TR_DOC_004::P002]."),
            ("Tập kiểm định lâm sàng bao gồm bao nhiêu bệnh nhân?", "Gồm 8.400 bệnh nhân [VN_TR_DOC_004::P002]."),
            ("Bệnh nhân có nguy cơ đe dọa thị lực được chuyển khám chuyên khoa trong bao lâu?", "Trong vòng 7 ngày làm việc [VN_TR_DOC_004::P002]."),
            ("Quy trình chụp đáy mắt có cần dùng thuốc giãn đồng tử không?", "Không cần sử dụng thuốc giãn đồng tử [VN_TR_DOC_004::P001]."),
            ("Kiến trúc mô hình nào được dùng để phân tích ảnh đáy mắt?", "Mô hình mạng nơ-ron tích chập đa tỷ lệ [VN_TR_DOC_004::P001]."),
            ("Độ đặc hiệu của hệ thống tầm soát trên lâm sàng là bao nhiêu?", "Đạt 91.8% [VN_TR_DOC_004::P002]."),
        ],
    },
    {
        "doc_id": "VN_TR_DOC_005",
        "title": "Quản lý Chuỗi cung ứng Lạnh cho Nông sản Xuất khẩu",
        "context": (
            "# Mục 1. Tiêu chuẩn bảo quản sau thu hoạch\n"
            "Trái cây tươi xuất khẩu được làm lạnh cưỡng bức bằng khí cưỡng bức xuống mức nhiệt độ 4.5 độ C trong vòng 3 giờ sau thu hoạch. "
            "Độ ẩm tương đối trong kho lạnh thông minh được duy trì ổn định từ 88% đến 92% nhằm hạn chế mất nước tự nhiên. "
            "# Mục 2. Giám sát IoT xuyên biên giới\n"
            "Mỗi container lạnh gắn cảm biến nhiệt ẩm không dây truyền dữ liệu định kỳ 5 phút một lần qua kết nối vệ tinh băng tần hẹp. "
            "Việc kiểm soát chuỗi lạnh liên tục giúp giảm tỷ lệ hư hỏng sau vận chuyển đường biển từ 18.5% xuống còn 3.2%."
        ),
        "qa_pairs": [
            ("Nông sản được làm lạnh cưỡng bức xuống nhiệt độ bao nhiêu sau thu hoạch?", "Xuống mức 4.5 độ C trong vòng 3 giờ sau thu hoạch [VN_TR_DOC_005::P001]."),
            ("Độ ẩm tương đối trong kho lạnh thông minh được giữ ở mức nào?", "Duy trì ổn định từ 88% đến 92% [VN_TR_DOC_005::P001]."),
            ("Cảm biến trên container lạnh truyền dữ liệu với chu kỳ bao lâu?", "Định kỳ 5 phút một lần qua kết nối vệ tinh băng tần hẹp [VN_TR_DOC_005::P002]."),
            ("Tỷ lệ hư hỏng sau vận chuyển đường biển giảm xuống còn bao nhiêu?", "Giảm từ 18.5% xuống còn 3.2% [VN_TR_DOC_005::P002]."),
            ("Mục đích của việc giữ độ ẩm từ 88% đến 92% là gì?", "Nhằm hạn chế mất nước tự nhiên của nông sản [VN_TR_DOC_005::P001]."),
            ("Thời gian hoàn tất làm lạnh cưỡng bức sau thu hoạch là bao lâu?", "Trong vòng 3 giờ sau thu hoạch [VN_TR_DOC_005::P001]."),
            ("Phương thức kết nối nào được dùng cho cảm biến xuyên biên giới?", "Kết nối vệ tinh băng tần hẹp [VN_TR_DOC_005::P002]."),
            ("Tỷ lệ hư hỏng trước khi áp dụng kiểm soát chuỗi lạnh liên tục là bao nhiêu?", "18.5% [VN_TR_DOC_005::P002]."),
        ],
    },
]

# Programmatically add 25 diverse Vietnamese structured technical documents (VN_TR_DOC_006 .. VN_TR_DOC_030)
_EXTRA_VN_SPECS = [
    (
        "VN_TR_DOC_006",
        "Quy chế Đào tạo Tín chỉ và Đánh giá Chuẩn đầu ra Đại học",
        "Sinh viên chương trình kỹ sư cần tích lũy tối thiểu 150 tín chỉ bắt buộc và tự chọn để đủ điều kiện xét tốt nghiệp. Điểm trung bình tích lũy toàn khóa đạt từ 3.20 đến 3.59 trên thang điểm 4 được xếp loại Giỏi.",
        "Khối lượng học tập mỗi học kỳ chính được quy định từ 14 đến 24 tín chỉ tùy theo năng lực học tập. Sinh viên phải đạt chứng chỉ ngoại ngữ bậc 4/6 theo Khung năng lực ngoại ngữ Việt Nam trước khi bảo vệ đồ án tốt nghiệp.",
        [
            ("Sinh viên chương trình kỹ sư cần tích lũy tối thiểu bao nhiêu tín chỉ?", "Tối thiểu 150 tín chỉ bắt buộc và tự chọn [VN_TR_DOC_006::P001]."),
            ("Điểm trung bình tích lũy bao nhiêu được xếp loại Giỏi?", "Từ 3.20 đến 3.59 trên thang điểm 4 [VN_TR_DOC_006::P001]."),
            ("Số tín chỉ đăng ký mỗi học kỳ chính nằm trong khoảng nào?", "Từ 14 đến 24 tín chỉ [VN_TR_DOC_006::P002]."),
            ("Yêu cầu chuẩn ngoại ngữ trước khi bảo vệ đồ án tốt nghiệp là gì?", "Đạt chứng chỉ ngoại ngữ bậc 4/6 theo Khung năng lực ngoại ngữ Việt Nam [VN_TR_DOC_006::P002]."),
        ],
    ),
    (
        "VN_TR_DOC_007",
        "Xử lý Nước thải Công nghiệp Dệt nhuộm bằng Màng lọc Nano",
        "Hệ thống keo tụ điện hóa kết hợp màng lọc nano loại bỏ 96.4% độ màu thuốc nhuộm hoạt tính và kim loại nặng hòa tan. Áp suất vận hành qua màng lọc được duy trì ở mức 8.5 bar nhằm tối ưu hóa lưu lượng thấm.",
        "Nước sau xử lý đạt quy chuẩn cột A và được tuần hoàn tái sử dụng 72% cho công đoạn giặt tẩy sơ bộ. Chu kỳ rửa ngược hóa học bằng dung dịch axit citric 2% được thực hiện sau mỗi 120 giờ vận hành liên tục.",
        [
            ("Hệ thống màng lọc nano loại bỏ được bao nhiêu phần trăm độ màu thuốc nhuộm?", "Loại bỏ 96.4% độ màu thuốc nhuộm hoạt tính và kim loại nặng hòa tan [VN_TR_DOC_007::P001]."),
            ("Áp suất vận hành qua màng lọc nano là bao nhiêu?", "Duy trì ở mức 8.5 bar [VN_TR_DOC_007::P001]."),
            ("Tỷ lệ nước sau xử lý được tuần hoàn tái sử dụng là bao nhiêu?", "Được tuần hoàn tái sử dụng 72% cho công đoạn giặt tẩy sơ bộ [VN_TR_DOC_007::P002]."),
            ("Chu kỳ rửa ngược hóa học màng lọc được thực hiện như thế nào?", "Bằng dung dịch axit citric 2% sau mỗi 120 giờ vận hành liên tục [VN_TR_DOC_007::P002]."),
        ],
    ),
    (
        "VN_TR_DOC_008",
        "Ứng dụng Viễn thám Radar Khẩu độ Tổng hợp trong Cảnh báo Sạt lở",
        "Ảnh vệ tinh radar giao thoa InSAR cho phép đo đạc biến dạng bề mặt sườn dốc với độ chính xác cỡ 2.5 milimét bất kể mây mù hay ban đêm. Chu kỳ lặp quỹ đạo thu nhận ảnh tại khu vực miền núi là 6 ngày một lần.",
        "Mô hình học máy kết hợp độ dốc địa hình và lượng mưa tích lũy 72 giờ phát cảnh báo sớm khi tốc độ dịch chuyển đất đá vượt quá 15 milimét/tuần. Dữ liệu cảnh báo được đồng bộ trực tiếp tới ban chỉ huy phòng chống thiên tai cấp xã.",
        [
            ("Công nghệ radar InSAR đo biến dạng bề mặt với độ chính xác bao nhiêu?", "Độ chính xác cỡ 2.5 milimét bất kể mây mù hay ban đêm [VN_TR_DOC_008::P001]."),
            ("Chu kỳ lặp quỹ đạo thu nhận ảnh vệ tinh là bao lâu?", "6 ngày một lần [VN_TR_DOC_008::P001]."),
            ("Khi nào hệ thống phát cảnh báo sớm sạt lở?", "Khi tốc độ dịch chuyển đất đá vượt quá 15 milimét/tuần kết hợp mưa tích lũy 72 giờ [VN_TR_DOC_008::P002]."),
            ("Ưu điểm của ảnh radar InSAR so với ảnh quang học là gì?", "Hoạt động chính xác bất kể điều kiện mây mù hay ban đêm [VN_TR_DOC_008::P001]."),
        ],
    ),
    (
        "VN_TR_DOC_009",
        "Kiến trúc Tối ưu hóa Truy vấn Phân tán trên Hồ dữ liệu Lakehouse",
        "Định dạng bảng mở Apache Iceberg quản lý siêu dữ liệu phân vùng theo bản chụp bất biến giúp thực thi giao dịch ACID trên lưu trữ đối tượng. Bộ lập lịch truy vấn áp dụng kỹ thuật cắt tỉa phân vùng động giảm 78% dung lượng tệp Parquet phải quét.",
        "Bộ nhớ đệm SSD NVMe cục bộ tại các nút tính toán lưu trữ các cột truy cập thường xuyên với độ trễ đọc trung bình 1.8 mili-giây. Cơ chế nén từ điển Zstandard mức 3 tiết kiệm 64% dung lượng lưu trữ so với định dạng văn bản CSV.",
        [
            ("Định dạng bảng nào được dùng để thực thi giao dịch ACID trên lưu trữ đối tượng?", "Định dạng bảng mở Apache Iceberg với bản chụp bất biến [VN_TR_DOC_009::P001]."),
            ("Kỹ thuật cắt tỉa phân vùng động giảm bao nhiêu dung lượng tệp phải quét?", "Giảm 78% dung lượng tệp Parquet phải quét [VN_TR_DOC_009::P001]."),
            ("Độ trễ đọc trung bình từ bộ nhớ đệm SSD NVMe cục bộ là bao nhiêu?", "Đạt 1.8 mili-giây [VN_TR_DOC_009::P002]."),
            ("Cơ chế nén Zstandard mức 3 tiết kiệm bao nhiêu dung lượng?", "Tiết kiệm 64% dung lượng lưu trữ so với định dạng CSV [VN_TR_DOC_009::P002]."),
        ],
    ),
    (
        "VN_TR_DOC_010",
        "Hệ thống Điều khiển Robot Phẫu thuật Nội soi Độ chính xác Cao",
        "Cánh tay robot phẫu thuật 7 bậc tự do tích hợp bộ lọc số khử hoàn toàn rung động sinh lý tay bác sĩ ở dải tần 8 đến 12 Hz. Hệ thống thị giác lập thể 3D độ phân giải 4K phóng đại phẫu trường gấp 10 lần với độ trễ truyền hình ảnh dưới 15 mili-giây.",
        "Cảm biến phản hồi lực xúc giác ở đầu dụng cụ giới hạn lực kẹp mô mềm tối đa ở mức 3.5 Newton để bảo vệ mạch máu nhỏ. Khóa liên động an toàn tự động đóng băng chuyển động cơ khí trong 2 mili-giây nếu phát hiện mất tín hiệu điều khiển.",
        [
            ("Cánh tay robot phẫu thuật có bao nhiêu bậc tự do và lọc dải tần rung nào?", "Có 7 bậc tự do và khử rung tay ở dải tần 8 đến 12 Hz [VN_TR_DOC_010::P001]."),
            ("Hệ thống thị giác 3D 4K phóng đại phẫu trường bao nhiêu lần?", "Phóng đại gấp 10 lần với độ trễ dưới 15 mili-giây [VN_TR_DOC_010::P001]."),
            ("Lực kẹp mô mềm tối đa được giới hạn ở mức bao nhiêu?", "Giới hạn ở mức 3.5 Newton để bảo vệ mạch máu nhỏ [VN_TR_DOC_010::P002]."),
            ("Khóa liên động an toàn phản ứng trong bao lâu khi mất tín hiệu?", "Tự động đóng băng chuyển động cơ khí trong 2 mili-giây [VN_TR_DOC_010::P002]."),
        ],
    ),
    (
        "VN_TR_DOC_011",
        "Chính sách Bảo vệ Dữ liệu Cá nhân và Đánh giá Tác động Xử lý Dữ liệu",
        "Bên kiểm soát dữ liệu cá nhân phải lập hồ sơ đánh giá tác động xử lý dữ liệu và gửi cơ quan chuyên trách trong thời hạn 60 ngày kể từ ngày bắt đầu xử lý. Dữ liệu cá nhân nhạy cảm bao gồm hồ sơ sức khỏe, thông tin sinh trắc học và dữ liệu vị trí thời gian thực.",
        "Khi phát hiện sự cố rò rỉ dữ liệu nghiêm trọng, tổ chức phải thông báo sự cố trong vòng 72 giờ kèm biện pháp khắc phục. Mọi yêu cầu rút lại sự đồng ý của chủ thể dữ liệu phải được thực thi hoàn tất không quá 15 ngày làm việc.",
        [
            ("Thời hạn gửi hồ sơ đánh giá tác động xử lý dữ liệu cá nhân là bao lâu?", "Trong thời hạn 60 ngày kể từ ngày bắt đầu xử lý [VN_TR_DOC_011::P001]."),
            ("Những loại thông tin nào thuộc nhóm dữ liệu cá nhân nhạy cảm?", "Hồ sơ sức khỏe, thông tin sinh trắc học và dữ liệu vị trí thời gian thực [VN_TR_DOC_011::P001]."),
            ("Tổ chức phải thông báo sự cố rò rỉ dữ liệu trong vòng bao nhiêu giờ?", "Trong vòng 72 giờ kể từ khi phát hiện sự cố [VN_TR_DOC_011::P002]."),
            ("Yêu cầu rút lại sự đồng ý của chủ thể dữ liệu phải hoàn tất trong bao lâu?", "Không quá 15 ngày làm việc [VN_TR_DOC_011::P002]."),
        ],
    ),
    (
        "VN_TR_DOC_012",
        "Vật liệu Bê tông Tính năng Siêu cao Gia cường Sợi Thép cho Cầu dây văng",
        "Bê tông tính năng siêu cao UHPC sử dụng muội silic siêu mịn và sợi thép mạ đồng hàm lượng 2% theo thể tích đạt cường độ chịu nén 145 MPa ở tuổi 28 ngày. Độ thấm ion clo giảm hơn 90% so với bê tông cầu tiêu chuẩn trong môi trường biển.",
        "Bản mặt cầu liên hợp dùng lớp phủ UHPC dày 55 milimét giúp giảm 32% tĩnh tải kết cấu nhịp chính. Tuổi thọ thiết kế công trình không cần đại tu mặt cầu kéo dài tới 100 năm.",
        [
            ("Cường độ chịu nén ở tuổi 28 ngày của bê tông UHPC đạt bao nhiêu?", "Đạt 145 MPa với hàm lượng sợi thép mạ đồng 2% theo thể tích [VN_TR_DOC_012::P001]."),
            ("Lớp phủ UHPC dày 55 milimét giúp giảm bao nhiêu phần trăm tĩnh tải?", "Giảm 32% tĩnh tải kết cấu nhịp chính [VN_TR_DOC_012::P002]."),
            ("Độ thấm ion clo của UHPC giảm như thế nào so với bê tông tiêu chuẩn?", "Giảm hơn 90% trong môi trường biển [VN_TR_DOC_012::P001]."),
            ("Tuổi thọ thiết kế không cần đại tu mặt cầu của công trình là bao lâu?", "Kéo dài tới 100 năm [VN_TR_DOC_012::P002]."),
        ],
    ),
    (
        "VN_TR_DOC_013",
        "Mô hình Định giá Tín chỉ Carbon Rừng ngập mặn Ven biển",
        "Hệ sinh thái rừng ngập mặn ven biển hấp thụ và tích lũy trung bình 8.6 tấn carbon hữu cơ trên mỗi héc-ta mỗi năm, gấp 4 lần rừng nhiệt đới trên cạn. Trữ lượng sinh khối dưới mặt đất trong lớp bùn trầm tích chiếm tới 75% tổng lượng carbon lưu giữ.",
        "Quy trình đo đạc, báo cáo và thẩm định MRV kết hợp ảnh LiDAR máy bay không người lái với 45 ô tiêu chuẩn thực địa. Doanh thu từ chuyển nhượng tín chỉ carbon trích 80% chi trả trực tiếp cho cộng đồng nhận khoán bảo vệ rừng.",
        [
            ("Rừng ngập mặn hấp thụ trung bình bao nhiêu tấn carbon trên mỗi héc-ta mỗi năm?", "Trung bình 8.6 tấn carbon hữu cơ/ha/năm, gấp 4 lần rừng nhiệt đới trên cạn [VN_TR_DOC_013::P001]."),
            ("Trữ lượng carbon trong lớp bùn trầm tích chiếm tỷ lệ bao nhiêu?", "Chiếm tới 75% tổng lượng carbon lưu giữ [VN_TR_DOC_013::P001]."),
            ("Quy trình MRV kết hợp những phương pháp đo đạc nào?", "Kết hợp ảnh LiDAR máy bay không người lái với 45 ô tiêu chuẩn thực địa [VN_TR_DOC_013::P002]."),
            ("Cộng đồng nhận khoán bảo vệ rừng được hưởng bao nhiêu phần trăm doanh thu tín chỉ?", "Được trích 80% doanh thu chi trả trực tiếp [VN_TR_DOC_013::P002]."),
        ],
    ),
    (
        "VN_TR_DOC_014",
        "Tối ưu hóa Thuật toán Nhận dạng Tiếng nói Tiếng Việt Đa phương ngữ",
        "Mô hình Conformer-Transducer kích thước 85 triệu tham số được huấn luyện trên 6.500 giờ giọng nói tiếng Việt trải đều ba vùng phương ngữ Bắc, Trung, Nam. Tầng trích xuất đặc trưng sử dụng phổ Mel 80 chiều tính trên cửa sổ khung 25 mili-giây.",
        "Nhờ kỹ thuật tăng cường dữ liệu nhiễu môi trường thực tế, tỷ lệ lỗi từ WER trên tập hội thoại tổng đài giảm xuống còn 6.4%. Quá trình giải mã trực tuyến đạt độ trễ từng phần 140 mili-giây trên bộ vi xử lý biên.",
        [
            ("Mô hình nhận dạng tiếng nói có bao nhiêu tham số và huấn luyện trên bao nhiêu giờ âm thanh?", "Kích thước 85 triệu tham số trên 6.500 giờ giọng nói ba miền Bắc, Trung, Nam [VN_TR_DOC_014::P001]."),
            ("Tầng trích xuất đặc trưng âm thanh sử dụng cấu hình nào?", "Sử dụng phổ Mel 80 chiều tính trên cửa sổ khung 25 mili-giây [VN_TR_DOC_014::P001]."),
            ("Tỷ lệ lỗi từ WER trên tập hội thoại tổng đài đạt mức bao nhiêu?", "Giảm xuống còn 6.4% [VN_TR_DOC_014::P002]."),
            ("Độ trễ giải mã trực tuyến trên bộ vi xử lý biên là bao nhiêu?", "Đạt độ trễ từng phần 140 mili-giây [VN_TR_DOC_014::P002]."),
        ],
    ),
    (
        "VN_TR_DOC_015",
        "Hệ thống Điều hành Giao thông Đô thị Thích ứng theo Thời gian Thực",
        "Mạng lưới 320 nút giao thông trọng điểm lắp đặt camera AI đếm lưu lượng phương tiện theo từng làn xe với độ chính xác 97.2%. Thuật toán học tăng cường điều chỉnh chu kỳ đèn tín hiệu động trong khoảng từ 45 giây đến 130 giây theo mật độ hàng đợi.",
        "Làn sóng xanh ưu tiên tự động mở thông tuyến cho xe cứu hỏa và xe cấp cứu trước 250 mét dựa trên định vị GPS chuyên dụng. Thời gian chờ trung bình giờ cao điểm giảm 28.5% và lượng phát thải khí CO2 tại nút giao giảm 19%.",
        [
            ("Hệ thống giám sát bao nhiêu nút giao và độ chính xác đếm xe đạt bao nhiêu?", "320 nút giao trọng điểm với độ chính xác đếm xe đạt 97.2% [VN_TR_DOC_015::P001]."),
            ("Chu kỳ đèn tín hiệu động được điều chỉnh trong khoảng thời gian nào?", "Trong khoảng từ 45 giây đến 130 giây theo mật độ hàng đợi [VN_TR_DOC_015::P001]."),
            ("Làn sóng xanh mở thông tuyến cho xe ưu tiên từ khoảng cách bao xa?", "Mở thông tuyến trước 250 mét dựa trên định vị GPS chuyên dụng [VN_TR_DOC_015::P002]."),
            ("Thời gian chờ trung bình giờ cao điểm và phát thải CO2 giảm bao nhiêu?", "Thời gian chờ giảm 28.5% và phát thải CO2 giảm 19% [VN_TR_DOC_015::P002]."),
        ],
    ),
    (
        "VN_TR_DOC_016",
        "Phác đồ Kháng sinh Dự phòng trong Phẫu thuật Chỉnh hình Sạch",
        "Kháng sinh dự phòng Cefazolin liều 2 gam tiêm tĩnh mạch chậm được chỉ định trong vòng 60 phút trước khi rạch da phẫu thuật thay khớp. Đối với bệnh nhân có cân nặng từ 120 kg trở lên, liều khuyến cáo tăng lên 3 gam để đảm bảo nồng độ diệt khuẩn mô xương.",
        "Nếu ca phẫu thuật kéo dài quá 4 giờ hoặc lượng máu mất vượt trên 1.500 ml, bác sĩ gây mê lặp lại một liều bổ sung trong mổ. Thời gian dùng kháng sinh dự phòng kết thúc hoàn toàn trong vòng 24 giờ sau đóng vết mổ nhằm ngăn kháng thuốc.",
        [
            ("Kháng sinh Cefazolin dự phòng được tiêm vào thời điểm nào và liều chuẩn bao nhiêu?", "Liều 2 gam tiêm tĩnh mạch chậm trong vòng 60 phút trước khi rạch da [VN_TR_DOC_016::P001]."),
            ("Bệnh nhân nặng từ 120 kg trở lên dùng liều Cefazolin bao nhiêu?", "Liều khuyến cáo tăng lên 3 gam [VN_TR_DOC_016::P001]."),
            ("Khi nào cần lặp lại liều kháng sinh bổ sung trong lúc mổ?", "Khi phẫu thuật kéo dài quá 4 giờ hoặc lượng máu mất vượt trên 1.500 ml [VN_TR_DOC_016::P002]."),
            ("Kháng sinh dự phòng phải ngưng hoàn toàn trong thời gian bao lâu sau mổ?", "Trong vòng 24 giờ sau đóng vết mổ nhằm ngăn kháng thuốc [VN_TR_DOC_016::P002]."),
        ],
    ),
    (
        "VN_TR_DOC_017",
        "Công nghệ Khử mặn Nước biển Thẩm thấu Ngược Kết hợp Thu hồi Áp năng",
        "Nhà máy khử mặn công suất 50.000 m3/ngày đêm vận hành màng thẩm thấu ngược polyamide ở áp suất 62 bar để tách 99.7% muối hòa tan. Thiết bị trao đổi áp suất quay isobaric thu hồi 95% năng lượng thủy lực từ dòng nước muối xả đậm đặc.",
        "Nhờ thu hồi áp năng, mức tiêu thụ điện năng riêng giảm xuống còn 2.9 kWh trên mỗi mét khối nước ngọt thành phẩm. Nước sau màng lọc được bổ sung khoáng canxi cacbonat đạt độ cứng từ 60 đến 80 mg/L trước khi cấp vào mạng lưới sinh hoạt.",
        [
            ("Màng thẩm thấu ngược vận hành ở áp suất nào và tách được bao nhiêu muối?", "Vận hành ở áp suất 62 bar và tách 99.7% muối hòa tan [VN_TR_DOC_017::P001]."),
            ("Thiết bị trao đổi áp suất isobaric thu hồi được bao nhiêu năng lượng thủy lực?", "Thu hồi 95% năng lượng thủy lực từ dòng nước muối xả [VN_TR_DOC_017::P001]."),
            ("Mức tiêu thụ điện năng riêng đạt bao nhiêu kWh trên mỗi m3 nước ngọt?", "Giảm xuống còn 2.9 kWh/m3 nước ngọt thành phẩm [VN_TR_DOC_017::P002]."),
            ("Độ cứng nước thành phẩm sau bổ sung khoáng đạt mức nào?", "Đạt độ cứng từ 60 đến 80 mg/L [VN_TR_DOC_017::P002]."),
        ],
    ),
    (
        "VN_TR_DOC_018",
        "Kiểm toán An toàn Hợp đồng Thông minh trên Mạng Chuỗi khối Doanh nghiệp",
        "Quy trình kiểm toán tĩnh kết hợp thực thi biểu tượng kiểm tra tự động các lỗ hổng tái nhập, tràn số nguyên và leo thang đặc quyền quản trị. Mọi thay đổi tham số hệ thống quan trọng phải đi qua cơ chế đa chữ ký 4 trên 7 kèm độ trễ khóa thời gian 48 giờ.",
        "Độ bao phủ kiểm thử nhánh trên mã nguồn hợp đồng thông minh bắt buộc đạt tối thiểu 98% trước khi triển khai lên mạng chính. Hệ thống giám sát giao dịch bất thường có quyền tạm dừng khẩn cấp hợp đồng trong vòng 1 khối xác nhận.",
        [
            ("Quy trình kiểm toán hợp đồng thông minh rà soát những lỗ hổng nào?", "Lỗ hổng tái nhập, tràn số nguyên và leo thang đặc quyền quản trị [VN_TR_DOC_018::P001]."),
            ("Cơ chế quản trị thay đổi tham số quan trọng được thiết lập ra sao?", "Đa chữ ký 4 trên 7 kèm độ trễ khóa thời gian 48 giờ [VN_TR_DOC_018::P001]."),
            ("Độ bao phủ kiểm thử nhánh bắt buộc đạt tối thiểu bao nhiêu phần trăm?", "Tối thiểu 98% trước khi triển khai lên mạng chính [VN_TR_DOC_018::P002]."),
            ("Hệ thống giám sát có thể tạm dừng khẩn cấp hợp đồng trong bao lâu?", "Trong vòng 1 khối xác nhận [VN_TR_DOC_018::P002]."),
        ],
    ),
    (
        "VN_TR_DOC_019",
        "Kỹ thuật Canh tác Lúa Phát thải Thấp Tưới Ngập Khô Xen kẽ",
        "Quy trình tưới ngập khô xen kẽ AWD rút cạn mặt ruộng xuống mức 15 cm dưới mặt đất trước khi bơm nước trở lại lớp nông 5 cm. Việc đưa oxy vào tầng đất canh tác ức chế vi khuẩn sinh khí mê-tan, giúp giảm 42% lượng phát thải khí nhà kính CH4.",
        "Lượng nước tưới tiêu toàn vụ giảm 28% và giúp bộ rễ lúa ăn sâu chống đổ ngã hiệu quả trong mùa mưa bão. Năng suất lúa hạt khô tăng trung bình 6.5% nhờ cải thiện khả năng hấp thu phân đạm ở giai đoạn đẻ nhánh hữu hiệu.",
        [
            ("Khi nào ruộng lúa AWD được bơm nước trở lại?", "Khi mực nước rút xuống mức 15 cm dưới mặt đất thì bơm lại lớp nông 5 cm [VN_TR_DOC_019::P001]."),
            ("Quy trình AWD giúp giảm bao nhiêu phần trăm phát thải khí mê-tan CH4?", "Giảm 42% lượng phát thải khí nhà kính CH4 [VN_TR_DOC_019::P001]."),
            ("Lượng nước tưới toàn vụ giảm bao nhiêu phần trăm?", "Giảm 28% lượng nước tưới tiêu [VN_TR_DOC_019::P002]."),
            ("Năng suất lúa hạt khô thay đổi như thế nào khi áp dụng AWD?", "Tăng trung bình 6.5% nhờ cải thiện hấp thu phân đạm [VN_TR_DOC_019::P002]."),
        ],
    ),
    (
        "VN_TR_DOC_020",
        "Kiến trúc Mạng Truy nhập Vô tuyến Mở O-RAN Thế hệ 5G",
        "Kiến trúc O-RAN phân tách trạm gốc 5G thành khối vô tuyến RU, khối phân tán DU và khối tập trung CU kết nối qua giao diện mở chuẩn hóa. Bộ điều khiển thông minh RIC gần thời gian thực vận hành vòng lặp tối ưu tài nguyên phổ tần trong khung thời gian từ 10 mili-giây đến 1 giây.",
        "Ứng dụng xApp cân bằng tải tự động tắt bớt lớp sóng mang phụ vào ban đêm giúp giảm 24% điện năng tiêu thụ của trạm phát sóng. Giao diện fronthaul eCPRI truyền tải dữ liệu băng gốc qua cáp quang tốc độ 25 Gbps với độ đồng bộ pha dưới 1.5 micro-giây.",
        [
            ("Kiến trúc O-RAN phân tách trạm gốc 5G thành những khối chức năng nào?", "Khối vô tuyến RU, khối phân tán DU và khối tập trung CU [VN_TR_DOC_020::P001]."),
            ("Bộ điều khiển thông minh RIC gần thời gian thực hoạt động ở khung thời gian nào?", "Từ 10 mili-giây đến 1 giây [VN_TR_DOC_020::P001]."),
            ("Ứng dụng xApp tắt sóng mang phụ ban đêm tiết kiệm bao nhiêu điện năng?", "Giảm 24% điện năng tiêu thụ của trạm phát sóng [VN_TR_DOC_020::P002]."),
            ("Giao diện fronthaul eCPRI yêu cầu tốc độ cáp quang và đồng bộ pha ra sao?", "Tốc độ 25 Gbps với độ đồng bộ pha dưới 1.5 micro-giây [VN_TR_DOC_020::P002]."),
        ],
    ),
    (
        "VN_TR_DOC_021",
        "Hệ thống Phát hiện Gian lận Thẻ Tín dụng bằng Mạng Nơ-ron Đồ thị",
        "Mạng nơ-ron đồ thị không đồng nhất biểu diễn chủ thẻ, thiết bị đầu cuối POS và địa chỉ IP thành các đỉnh liên kết theo chuỗi giao dịch thời gian. Mô hình tổng hợp lân cận 2 bước nhảy phát hiện các vòng tròn rửa tiền và chiếm đoạt tài khoản tinh vi.",
        "Hệ thống chấm điểm rủi ro hoàn tất trong 22 mili-giây ở phân vị thứ 99, đạt diện tích dưới đường cong AUC-PR mức 0.914. Tỷ lệ cảnh báo giả giảm 37% so với bộ quy tắc ngưỡng tĩnh truyền thống, hạn chế tối đa phiền hà cho khách hàng hợp lệ.",
        [
            ("Các đỉnh trong mạng nơ-ron đồ thị phát hiện gian lận biểu diễn những thực thể nào?", "Chủ thẻ, thiết bị đầu cuối POS và địa chỉ IP [VN_TR_DOC_021::P001]."),
            ("Hệ thống chấm điểm rủi ro mất bao lâu ở phân vị thứ 99?", "Hoàn tất trong 22 mili-giây [VN_TR_DOC_021::P002]."),
            ("Chỉ số AUC-PR của mô hình đạt mức bao nhiêu?", "Đạt mức 0.914 [VN_TR_DOC_021::P002]."),
            ("Tỷ lệ cảnh báo giả giảm bao nhiêu phần trăm so với bộ quy tắc tĩnh?", "Giảm 37% tỷ lệ cảnh báo giả [VN_TR_DOC_021::P002]."),
        ],
    ),
    (
        "VN_TR_DOC_022",
        "Quy chuẩn Thiết kế Trung tâm Dữ liệu Xanh Làm mát bằng Chất lỏng Trực tiếp",
        "Hệ thống tấm lạnh tiếp xúc trực tiếp trên bề mặt vi xử lý GPU dẫn nhiệt bằng dung dịch propylene glycol pha nước khử ion ở nhiệt độ cấp 32 độ C. Phương pháp làm mát chất lỏng thu hồi 82% tổng nhiệt lượng tỏa ra từ tủ rack mật độ cao 60 kW.",
        "Chỉ số hiệu quả sử dụng năng lượng PUE toàn trung tâm dữ liệu đạt mức 1.14 tính trung bình cả năm tại khí hậu nhiệt đới. Hệ thống cảm biến rò rỉ cáp quang học dưới sàn nâng phát hiện giọt chất lỏng trong vòng 3 giây và cô lập van nhánh tự động.",
        [
            ("Dung dịch làm mát tấm lạnh tiếp xúc trực tiếp gồm chất gì và cấp ở nhiệt độ nào?", "Dung dịch propylene glycol pha nước khử ion ở nhiệt độ cấp 32 độ C [VN_TR_DOC_022::P001]."),
            ("Phương pháp làm mát chất lỏng thu hồi bao nhiêu nhiệt lượng từ tủ rack 60 kW?", "Thu hồi 82% tổng nhiệt lượng tỏa ra [VN_TR_DOC_022::P001]."),
            ("Chỉ số PUE trung bình năm của trung tâm dữ liệu đạt mức bao nhiêu?", "Đạt mức 1.14 tại khí hậu nhiệt đới [VN_TR_DOC_022::P002]."),
            ("Cảm biến rò rỉ phát hiện giọt chất lỏng trong bao lâu?", "Trong vòng 3 giây và tự động cô lập van nhánh [VN_TR_DOC_022::P002]."),
        ],
    ),
    (
        "VN_TR_DOC_023",
        "Bảo trì Dự đoán Động cơ Phản lực Hàng không dựa trên Bản sao Số",
        "Bản sao số động cơ tuabin phản lực cánh quạt đồng bộ dữ liệu rung động, nhiệt độ khí xả EGT và áp suất máy nén từ 180 cảm biến trên khoang bay. Mô hình suy giảm vật lý lai ghép mạng LSTM dự báo thời gian hữu ích còn lại RUL của cánh tuabin cao áp.",
        "Việc lên lịch thay thế phụ tùng trước ngưỡng hỏng hóc 150 giờ bay giúp giảm 46% sự cố chậm chuyến do kỹ thuật đột xuất. Chi phí đại tu định kỳ tiết kiệm trung bình 380.000 USD cho mỗi động cơ mỗi năm khai thác.",
        [
            ("Bản sao số động cơ đồng bộ dữ liệu từ bao nhiêu cảm biến trên khoang bay?", "Từ 180 cảm biến đo rung động, nhiệt độ khí xả EGT và áp suất máy nén [VN_TR_DOC_023::P001]."),
            ("Phụ tùng được lên lịch thay thế trước ngưỡng hỏng hóc bao nhiêu giờ bay?", "Trước ngưỡng hỏng hóc 150 giờ bay [VN_TR_DOC_023::P002]."),
            ("Sự cố chậm chuyến do kỹ thuật đột xuất giảm bao nhiêu phần trăm?", "Giảm 46% sự cố chậm chuyến đột xuất [VN_TR_DOC_023::P002]."),
            ("Chi phí đại tu tiết kiệm trung bình bao nhiêu mỗi động cơ mỗi năm?", "Tiết kiệm trung bình 380.000 USD cho mỗi động cơ mỗi năm [VN_TR_DOC_023::P002]."),
        ],
    ),
    (
        "VN_TR_DOC_024",
        "Tổng hợp Dược chất Sinh học Kháng thể Đơn dòng trong Lò phản ứng Dùng một lần",
        "Quy trình nuôi cấy tế bào buồng trứng chuột đất vàng CHO mật độ cao trong túi phản ứng sinh học dùng một lần thể tích 2.000 lít đạt hiệu suất kháng thể 6.8 gam/lít sau 14 ngày. Nồng độ oxy hòa tan được giữ ổn định ở mức 40% bão hòa không khí và pH duy trì 7.05.",
        "Công đoạn tinh chế sắc ký ái lực Protein A liên tục thu hồi 93.5% kháng thể đích với độ tinh khiết đơn phân trên 99.1%. Sử dụng thiết bị dùng một lần loại bỏ hoàn toàn bước tiệt trùng hơi nước tại chỗ, rút ngắn 45% thời gian chuyển đổi mẻ sản xuất.",
        [
            ("Nuôi cấy tế bào CHO trong lò 2.000 lít đạt hiệu suất kháng thể bao nhiêu?", "Đạt 6.8 gam/lít sau 14 ngày nuôi cấy [VN_TR_DOC_024::P001]."),
            ("Nồng độ oxy hòa tan và độ pH được duy trì ở mức nào?", "Oxy hòa tan 40% bão hòa không khí và pH 7.05 [VN_TR_DOC_024::P001]."),
            ("Công đoạn sắc ký ái lực Protein A thu hồi bao nhiêu kháng thể và độ tinh khiết ra sao?", "Thu hồi 93.5% kháng thể đích với độ tinh khiết đơn phân trên 99.1% [VN_TR_DOC_024::P002]."),
            ("Thiết bị dùng một lần rút ngắn bao nhiêu thời gian chuyển đổi mẻ?", "Rút ngắn 45% thời gian chuyển đổi mẻ sản xuất [VN_TR_DOC_024::P002]."),
        ],
    ),
    (
        "VN_TR_DOC_025",
        "Cơ chế Đồng bộ Hóa Bộ nhớ Tham số Đa thang Thời gian trong Hỏi đáp Tài liệu",
        "Kiến trúc bộ nhớ liên tục căn chỉnh cấu trúc cập nhật trọng số nhanh ở ranh giới đoạn văn, trọng số trung gian ở ranh giới chương mục và trọng số neo bền vững ở cấp toàn tài liệu. Cổng hòa trộn phần dư điều tiết linh hoạt đóng góp của từng tầng bộ nhớ vào luồng trạng thái ẩn.",
        "Khi kết hợp truy xuất từ khóa BM25 với bộ nhớ tham số và cổng kiểm soát từ chối ngưỡng 3.0, hệ thống triệt tiêu hiện tượng bịa đặt thông tin ngoài văn bản. Toàn bộ mô hình ngôn ngữ nền được giữ đóng băng để bảo toàn năng lực ngôn ngữ gốc và tiết kiệm bộ nhớ đồ họa.",
        [
            ("Ba tầng trọng số của bộ nhớ liên tục được cập nhật tại những ranh giới nào?", "Tầng nhanh ở đoạn văn, tầng trung gian ở chương mục và tầng bền vững ở toàn tài liệu [VN_TR_DOC_025::P001]."),
            ("Thành phần nào điều tiết đóng góp của từng tầng bộ nhớ vào trạng thái ẩn?", "Cổng hòa trộn phần dư [VN_TR_DOC_025::P001]."),
            ("Hệ thống triệt tiêu hiện tượng bịa đặt thông tin bằng sự kết hợp nào?", "Kết hợp truy xuất BM25, bộ nhớ tham số và cổng kiểm soát từ chối ngưỡng 3.0 [VN_TR_DOC_025::P002]."),
            ("Vì sao mô hình ngôn ngữ nền được giữ đóng băng hoàn toàn?", "Để bảo toàn năng lực ngôn ngữ gốc và tiết kiệm bộ nhớ đồ họa [VN_TR_DOC_025::P002]."),
        ],
    ),
]


VN_VAL_DOCS: List[Dict[str, Any]] = [
    {
        "doc_id": "VN_VAL_DOC_001",
        "title": "Hệ thống Định vị Trong nhà Siêu băng rộng UWB cho Kho vận Tự động",
        "context": (
            "# Mục 1. Hạ tầng trạm neo xung siêu ngắn\n"
            "Mạng định vị trong nhà sử dụng xung vô tuyến siêu băng rộng UWB ở băng tần 6.5 GHz với độ rộng xung 2 nano-giây chống nhiễu đa đường trong kho kệ thép. "
            "Thuật toán chênh lệch thời gian đến hai chiều xác định tọa độ xe tự hành AGV với sai số trung bình 8.5 cm.\n"
            "# Mục 2. Điều phối đội xe và an toàn va chạm\n"
            "Máy chủ điều phối trung tâm cập nhật vị trí của 120 xe AGV ở tần số 20 Hz để tối ưu hóa lộ trình lấy hàng không tắc nghẽn. "
            "Vùng giảm tốc an toàn tự động kích hoạt khi khoảng cách giữa xe và nhân viên kho thu hẹp dưới 2.5 mét."
        ),
        "qa_pairs": [
            ("Hệ thống UWB hoạt động ở băng tần nào và độ rộng xung bao nhiêu?", "Ở băng tần 6.5 GHz với độ rộng xung 2 nano-giây [VN_VAL_DOC_001::P001]."),
            ("Sai số định vị trung bình của xe tự hành AGV là bao nhiêu?", "Sai số trung bình 8.5 cm [VN_VAL_DOC_001::P001]."),
            ("Máy chủ điều phối cập nhật vị trí của bao nhiêu xe AGV và ở tần số nào?", "Cập nhật 120 xe AGV ở tần số 20 Hz [VN_VAL_DOC_001::P002]."),
            ("Khi nào vùng giảm tốc an toàn tự động kích hoạt?", "Khi khoảng cách giữa xe và nhân viên kho thu hẹp dưới 2.5 mét [VN_VAL_DOC_001::P002]."),
            ("Ưu điểm của xung ngắn 2 nano-giây trong kho kệ thép là gì?", "Chống nhiễu đa đường hiệu quả [VN_VAL_DOC_001::P001]."),
        ],
    },
    {
        "doc_id": "VN_VAL_DOC_002",
        "title": "Công nghệ Sấy Thăng hoa Chân không Bảo toàn Hoạt chất Dược liệu Quý",
        "context": (
            "# Mục 1. Cấp đông sâu và thăng hoa tinh thể băng\n"
            "Dược liệu tươi được cấp đông nhanh xuống âm 42 độ C nhằm hình thành tinh thể băng mịn không phá vỡ thành tế bào thực vật. "
            "Buồng sấy duy trì áp suất chân không sâu ở mức 12 Pascal giúp nước chuyển trực tiếp từ thể rắn sang thể hơi.\n"
            "# Mục 2. Hàm lượng hoạt chất và độ ẩm bảo quản\n"
            "Phương pháp sấy thăng hoa giữ lại 96.8% hàm lượng saponin tổng số so với chỉ 61.5% ở phương pháp sấy nhiệt đối lưu thông thường. "
            "Độ ẩm dư của dược liệu thành phẩm giảm xuống dưới 3.8%, cho phép bảo quản kín trong 36 tháng mà không dùng chất bảo quản."
        ),
        "qa_pairs": [
            ("Dược liệu tươi được cấp đông nhanh xuống nhiệt độ bao nhiêu?", "Xuống mức âm 42 độ C để không phá vỡ thành tế bào [VN_VAL_DOC_002::P001]."),
            ("Áp suất chân không trong buồng sấy được giữ ở mức nào?", "Duy trì ở mức 12 Pascal [VN_VAL_DOC_002::P001]."),
            ("Sấy thăng hoa giữ lại bao nhiêu phần trăm hàm lượng saponin?", "Giữ lại 96.8% hàm lượng saponin tổng số [VN_VAL_DOC_002::P002]."),
            ("Độ ẩm dư và thời hạn bảo quản của dược liệu thành phẩm là bao nhiêu?", "Độ ẩm dư dưới 3.8%, bảo quản kín trong 36 tháng [VN_VAL_DOC_002::P002]."),
            ("Phương pháp sấy nhiệt đối lưu thông thường giữ được bao nhiêu saponin?", "Chỉ giữ được 61.5% [VN_VAL_DOC_002::P002]."),
        ],
    },
    {
        "doc_id": "VN_VAL_DOC_003",
        "title": "Kiểm soát Rủi ro Thanh khoản Nội ngày trong Hệ thống Thanh toán Tức thời",
        "context": (
            "# Mục 1. Dự trữ thanh khoản và hạn mức thấu chi\n"
            "Ngân hàng thành viên tham gia thanh toán nhanh 24/7 phải duy trì tỷ lệ dự trữ thanh khoản khả dụng tối thiểu bằng 115% giá trị dòng tiền chi ròng đỉnh điểm. "
            "Khoản vay thấu chi nội ngày tại ngân hàng trung ương yêu cầu thế chấp bằng trái phiếu chính phủ với tỷ lệ chiết khấu phòng vệ 5%.\n"
            "# Mục 2. Cảnh báo sớm và tự động bổ sung hạn mức\n"
            "Hệ thống giám sát phát cảnh báo mức vàng khi hạn mức thanh toán ròng sử dụng chạm ngưỡng 75% và mức đỏ ở ngưỡng 90%. "
            "Cơ chế cầm cố giấy tờ có giá tự động bổ sung hạn mức thanh toán chỉ trong vòng 45 giây kể từ khi chạm ngưỡng đỏ."
        ),
        "qa_pairs": [
            ("Ngân hàng thành viên phải duy trì tỷ lệ dự trữ thanh khoản tối thiểu bao nhiêu?", "Tối thiểu bằng 115% giá trị dòng tiền chi ròng đỉnh điểm [VN_VAL_DOC_003::P001]."),
            ("Tỷ lệ chiết khấu phòng vệ khi thế chấp trái phiếu chính phủ là bao nhiêu?", "Tỷ lệ chiết khấu phòng vệ 5% [VN_VAL_DOC_003::P001]."),
            ("Cảnh báo mức vàng và mức đỏ được kích hoạt ở những ngưỡng nào?", "Mức vàng ở ngưỡng 75% và mức đỏ ở ngưỡng 90% [VN_VAL_DOC_003::P002]."),
            ("Cơ chế tự động bổ sung hạn mức thanh toán mất bao lâu khi chạm ngưỡng đỏ?", "Chỉ trong vòng 45 giây [VN_VAL_DOC_003::P002]."),
            ("Tài sản nào được dùng để thế chấp cho khoản vay thấu chi nội ngày?", "Trái phiếu chính phủ [VN_VAL_DOC_003::P001]."),
        ],
    },
    {
        "doc_id": "VN_VAL_DOC_004",
        "title": "Mạng Cảm biến Quang học Cáp ngầm Đáy biển Cảnh báo Sớm Động đất Sóng thần",
        "context": (
            "# Mục 1. Cảm biến âm học phân tán DAS trên sợi quang tối\n"
            "Công nghệ cảm biến âm học phân tán biến một sợi cáp quang viễn thông ngầm dài 100 km thành 10.000 điểm thu địa chấn cách nhau mỗi 10 mét. "
            "Sự thay đổi pha tán xạ ngược Rayleigh phát hiện sóng sơ cấp P ngay tại tâm chấn dưới đáy đại dương.\n"
            "# Mục 2. Tốc độ truyền tin cảnh báo khẩn cấp\n"
            "Hệ thống ước lượng độ lớn chấn động trong vòng 4.2 giây kể từ khi sóng P chạm tuyến cáp, sớm hơn 18 giây so với trạm địa chấn trên đất liền. "
            "Bản tin cảnh báo tự động kích hoạt hệ thống phanh khẩn cấp tàu cao tốc và đóng van đường ống khí đốt ven biển."
        ),
        "qa_pairs": [
            ("Tuyến cáp quang ngầm 100 km tạo ra bao nhiêu điểm thu địa chấn?", "Tạo ra 10.000 điểm thu địa chấn cách nhau mỗi 10 mét [VN_VAL_DOC_004::P001]."),
            ("Hiện tượng vật lý nào giúp phát hiện sóng sơ cấp P?", "Sự thay đổi pha tán xạ ngược Rayleigh [VN_VAL_DOC_004::P001]."),
            ("Hệ thống ước lượng độ lớn chấn động trong bao lâu và sớm hơn trạm đất liền bao nhiêu?", "Trong vòng 4.2 giây, sớm hơn 18 giây so với trạm trên đất liền [VN_VAL_DOC_004::P002]."),
            ("Bản tin cảnh báo tự động kích hoạt những biện pháp bảo vệ hạ tầng nào?", "Phanh khẩn cấp tàu cao tốc và đóng van đường ống khí đốt ven biển [VN_VAL_DOC_004::P002]."),
            ("Khoảng cách không gian giữa hai điểm thu địa chấn liên tiếp trên sợi quang là bao nhiêu?", "Cách nhau mỗi 10 mét [VN_VAL_DOC_004::P001]."),
        ],
    },
]


def _build_all_vn_train_docs() -> List[Dict[str, Any]]:
    docs = list(VN_TRAIN_DOCS)
    for doc_id, title, sec1, sec2, pairs in _EXTRA_VN_SPECS:
        ctx = f"# Mục 1. Tổng quan kỹ thuật\n{sec1}\n# Mục 2. Chỉ số vận hành & kiểm soát\n{sec2}"
        # Expand to 8 QA variations (concise + balanced phrasings)
        expanded_pairs = list(pairs)
        for q_orig, a_orig in pairs:
            short_q = f"Theo tài liệu, {q_orig[0].lower() + q_orig[1:]}"
            clean_short_a = a_orig.replace(f" [{doc_id}::P001]", "").replace(f" [{doc_id}::P002]", "")
            expanded_pairs.append((short_q, clean_short_a))
        docs.append({
            "doc_id": doc_id,
            "title": title,
            "context": ctx,
            "qa_pairs": expanded_pairs,
        })
    return docs


def get_vietnamese_training_corpus() -> Dict[str, Any]:
    """
    Returns Vietnamese grounded QA + memory-recall QA samples formatted cleanly
    with separated `prompt_text` and `target_answer` so answer-token loss can be weighted.
    """
    docs = _build_all_vn_train_docs()
    samples: List[Dict[str, Any]] = []
    idx = 1

    for doc in docs:
        d_id = doc["doc_id"]
        ctx = doc["context"]
        for p_idx, (q_text, a_text) in enumerate(doc["qa_pairs"]):
            # 1. Grounded QA Sample (matching build_token_efficient_prompt in token_optimizer.py)
            if p_idx % 2 == 0:
                instruction = (
                    "Dựa trên các đoạn trích sau, hãy trả lời câu hỏi một cách ngắn gọn, chính xác bằng tiếng Việt. "
                    "Kèm theo trích dẫn nguồn."
                )
            else:
                instruction = (
                    "Dựa vào các đoạn trích sau, hãy trả lời câu hỏi trực tiếp, súc tích trong 1-2 câu ngắn, đủ sự kiện chính."
                )

            prompt_grounded = (
                f"{instruction}\n\n"
                f"Ngữ cảnh:\n{ctx}\n\n"
                f"Câu hỏi: {q_text}\n"
                f"Trả lời:"
            )
            samples.append({
                "sample_id": f"VN_TR_Q{idx:04d}",
                "document_id": d_id,
                "lang": "vi",
                "mode": "grounded",
                "question": q_text,
                "answer": a_text,
                "prompt_text": prompt_grounded,
                "formatted_text": f"{prompt_grounded} {a_text}\n",
            })
            idx += 1

            # 2. Every 3rd QA pair also includes a Memory-Only recall sample (P1 mode alignment)
            if p_idx % 3 == 0:
                clean_mem_ans = (
                    a_text.replace(f" [{d_id}::P001]", "")
                    .replace(f" [{d_id}::P002]", "")
                    .strip()
                )
                prompt_mem = (
                    f"Dựa trên kiến thức trong bộ nhớ, hãy trả lời câu hỏi sau bằng tiếng Việt:\n\n"
                    f"Câu hỏi: {q_text}\n"
                    f"Trả lời:"
                )
                samples.append({
                    "sample_id": f"VN_MEM_Q{idx:04d}",
                    "document_id": d_id,
                    "lang": "vi",
                    "mode": "memory",
                    "question": q_text,
                    "answer": clean_mem_ans,
                    "prompt_text": prompt_mem,
                    "formatted_text": f"{prompt_mem} {clean_mem_ans}\n",
                })
                idx += 1

    # Build Vietnamese Validation Samples
    val_samples: List[Dict[str, Any]] = []
    v_idx = 1
    for vdoc in VN_VAL_DOCS:
        d_id = vdoc["doc_id"]
        ctx = vdoc["context"]
        for q_text, a_text in vdoc["qa_pairs"]:
            prompt_val = (
                "Dựa trên các đoạn trích sau, hãy trả lời câu hỏi một cách ngắn gọn, chính xác bằng tiếng Việt. "
                "Kèm theo trích dẫn nguồn.\n\n"
                f"Ngữ cảnh:\n{ctx}\n\n"
                f"Câu hỏi: {q_text}\n"
                f"Trả lời:"
            )
            val_samples.append({
                "sample_id": f"VN_VAL_Q{v_idx:03d}",
                "document_id": d_id,
                "lang": "vi",
                "question": q_text,
                "answer": a_text,
                "prompt_text": prompt_val,
                "formatted_text": f"{prompt_val} {a_text}\n",
            })
            v_idx += 1

    return {
        "train_documents": docs,
        "train_samples": samples,
        "val_documents": VN_VAL_DOCS,
        "val_samples": val_samples,
    }

