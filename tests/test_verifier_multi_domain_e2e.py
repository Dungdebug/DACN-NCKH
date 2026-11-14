"""
Multi-domain end-to-end integration test suite.
Validates end-to-end pipeline across 7 core computer science subjects:
1. Cryptography and Information Security
2. Computer Networks and Telecommunications
3. Database Systems
4. Operating Systems
5. Artificial Intelligence and Machine Learning
6. Data Structures and Algorithms
7. Software Engineering

Covers AcademicDomainRegistry, expand_all_academic_synonyms,
SelfStudyKnowledgeBank, and DocumentAnalyzer.
"""

import time
import pytest
from typing import Dict, Any, List

from src.self_study.datasets.domain_router_registry import (
    AcademicDomain,
    AcademicDomainRegistry,
    DomainRouteResult,
    default_academic_domain_registry,
    route_academic_query,
    route_academic_document,
    expand_all_academic_synonyms,
    get_academic_domain_metadata,
    DOMAIN_METADATA,
)
from src.self_study.knowledge_bank import (
    SelfStudyKnowledgeBank,
    ANSWER_INSTRUCTION_BLUEPRINTS,
    get_knowledge_bank,
)
from src.web.document_analyzer import DocumentAnalyzer


SAMPLE_CORPUS_7_DOMAINS: Dict[str, Dict[str, Any]] = {
    AcademicDomain.CRYPTOGRAPHY.value: {
        "title": "Mật mã học & Tiêu chuẩn AES",
        "doc_id": "DOC_E2E_CRYPTO",
        "raw_text": """=== Page 1 ===
1. GIỚI THIỆU HỆ MÃ KHÓA ĐỐI XỨNG:
- Mã hóa đối xứng là phương pháp mã hóa sử dụng cùng một khóa bí mật K cho cả quá trình mã hóa và giải mã.
- Hai phương pháp mã hóa cơ bản:
* Mã hóa thay thế (Substitution cipher).
* Mã hóa hoán vị (Permutation/Transposition cipher).
2. TIÊU CHUẨN MÃ HÓA TIÊN TIẾN AES:
- Thuật toán mã hóa khối với kích thước khối 128 bit và các độ dài khóa 128, 192 hoặc 256 bit.
- Bốn bước biến đổi trong mỗi vòng lặp:
* SubBytes: Thay thế phi tuyến các byte bằng bảng S-box.
* ShiftRows: Dịch chuyển vòng các hàng của ma trận trạng thái.
* MixColumns: Trộn các cột bằng phép nhân ma trận trong trường Galois GF(2^8).
* AddRoundKey: Cộng khóa vòng bằng phép toán XOR.
3. PHÁ MÃ VÀ AN NINH HỆ MÃ:
- An ninh vô điều kiện và an ninh tính toán.
- Tấn công thám mã vi sai và thám mã tuyến tính.
""",
        "sample_query": "Hệ mã hóa đối xứng và thuật toán AES hoạt động như thế nào?",
        "unaccented_query": "he ma hoa doi xung va thuat toan aes hoat dong nhu the nao",
        "synonym_seed": "mã hóa đối xứng",
        "expected_metric": "128 bit",
    },
    AcademicDomain.NETWORKS.value: {
        "title": "Mạng máy tính & Giao thức TCP/IP",
        "doc_id": "DOC_E2E_NETWORKS",
        "raw_text": """=== Page 1 ===
1. MÔ HÌNH THAM CHIẾU OSI VÀ TCP/IP:
- Mô hình OSI gồm 7 tầng từ Tầng vật lý đến Tầng ứng dụng.
- Mô hình TCP/IP gồm 4 tầng: Tầng ứng dụng, Tầng giao vận, Tầng mạng, Tầng truy cập mạng.
- Giao thức tầng giao vận:
* Giao thức TCP: Hướng kết nối, đảm bảo độ tin cậy và kiểm soát luồng.
* Giao thức UDP: Không hướng kết nối, tốc độ truyền nhanh, chấp nhận mất mát gói tin.
2. QUÁ TRÌNH BẮT TAY BA BƯỚC TCP:
- Quá trình thiết lập kết nối tin cậy giữa máy khách client và máy chủ server:
* Bước 1 (SYN): Máy khách gửi gói tin SYN với số thứ tự khởi tạo seq = x.
* Bước 2 (SYN-ACK): Máy chủ phản hồi gói tin SYN-ACK với ack = x + 1 và seq = y.
* Bước 3 (ACK): Máy khách gửi gói tin ACK xác nhận với ack = y + 1.
3. ĐỊNH TUYẾN GÓI TIN VÀ CHIA MẠNG CON:
- Thuật toán định tuyến Dijkstra OSPF và thuật toán Bellman-Ford RIP.
- Mặt nạ mạng con Subnet Mask và địa chỉ IPv4, IPv6.
""",
        "sample_query": "Quá trình bắt tay 3 bước trong giao thức TCP diễn ra như thế nào?",
        "unaccented_query": "qua trinh bat tay 3 buoc trong giao thuc tcp dien ra nhu the nao",
        "synonym_seed": "bắt tay 3 bước tcp",
        "expected_metric": "4 tầng",
    },
    AcademicDomain.DATABASES.value: {
        "title": "Hệ cơ sở dữ liệu quan hệ & Chuẩn hóa",
        "doc_id": "DOC_E2E_DATABASES",
        "raw_text": """=== Page 1 ===
1. CƠ SỞ DỮ LIỆU QUAN HỆ VÀ CHUẨN HÓA:
- Chuẩn hóa dữ liệu là quá trình phân rã lược đồ quan hệ nhằm loại bỏ dư thừa dữ liệu và dị thường cập nhật.
- Các dạng chuẩn cơ bản:
* Dạng chuẩn 1NF: Tất cả các thuộc tính đều chứa giá trị nguyên tử (atomic values).
* Dạng chuẩn 2NF: Đạt chuẩn 1NF và mọi thuộc tính không khóa đều phụ thuộc đầy đủ vào khóa chính.
* Dạng chuẩn 3NF: Đạt chuẩn 2NF và không có phụ thuộc bắc cầu vào khóa chính.
* Dạng chuẩn BCNF: Mọi phụ thuộc hàm X -> Y thì X phải là một siêu khóa (superkey).
2. QUẢN LÝ GIAO DỊCH VÀ TÍNH CHẤT ACID:
- Bốn tính chất sống còn của giao dịch cơ sở dữ liệu:
* Tính nguyên tử (Atomicity): Giao dịch hoàn thành trọn vẹn hoặc bị hủy bỏ hoàn toàn.
* Tính nhất quán (Consistency): Dữ liệu chuyển từ trạng thái hợp lệ này sang trạng thái hợp lệ khác.
* Tính cô lập (Isolation): Các giao dịch đồng thời không can thiệp lẫn nhau.
* Tính bền vững (Durability): Kết quả giao dịch được lưu trữ vĩnh viễn dù có sự cố hệ thống.
3. TỐI ƯU HÓA TRUY VẤN VÀ CHỈ MỤC B-TREE:
- Chỉ mục cây B-Tree và B+Tree giúp giảm số lần truy xuất đĩa I/O từ O(n) xuống O(log n).
""",
        "sample_query": "Các dạng chuẩn 1NF 2NF 3NF BCNF và tính chất ACID trong cơ sở dữ liệu là gì?",
        "unaccented_query": "cac dang chuan 1nf 2nf 3nf bcnf va tinh chat acid trong co so du lieu la gi",
        "synonym_seed": "chuẩn hóa cơ sở dữ liệu bcnf",
        "expected_metric": "3NF",
    },
    AcademicDomain.OPERATING_SYSTEMS.value: {
        "title": "Hệ điều hành & Quản lý tiến trình",
        "doc_id": "DOC_E2E_OPERATING_SYSTEMS",
        "raw_text": """=== Page 1 ===
1. QUẢN LÝ TIẾN TRÌNH VÀ TIẾN TRÌNH CON:
- Tiến trình (Process) là một thực thể đang thực thi bao gồm bộ đếm chương trình, thanh ghi và không gian ngăn xếp stack.
- Luồng (Thread) là đơn vị cơ bản sử dụng CPU bên trong tiến trình và chia sẻ không gian địa chỉ.
- Các thuật toán lập lịch CPU:
* Lập lịch FCFS (First-Come, First-Served): Tiến trình đến trước được phục vụ trước.
* Lập lịch SJF (Shortest Job First): Ưu tiên tiến trình có thời gian thực thi ngắn nhất.
* Lập lịch Round Robin (RR): Phân bổ thời gian thực thi theo từng lượng tử thời gian time quantum.
2. HIỆN TƯỢNG BẾ TẮC DEADLOCK:
- Bế tắc xảy ra khi tập hợp các tiến trình bị chặn vĩnh viễn do chờ đợi tài nguyên lẫn nhau.
- Bốn điều kiện Coffman:
* Loại trừ tương hỗ (Mutual Exclusion).
* Giữ và chờ (Hold and Wait).
* Không chiếm đoạt (No Preemption).
* Chu trình chờ (Circular Wait).
3. HỆ THỐNG BỘ NHỚ ẢO VÀ PHÂN TRANG:
- Cơ chế dịch địa chỉ ảo sang địa chỉ vật lý thông qua bảng trang Page Table và bộ đệm TLB.
- Thuật toán thay thế trang LRU (Least Recently Used) và FIFO.
""",
        "sample_query": "Các thuật toán lập lịch CPU và 4 điều kiện gây bế tắc deadlock trong hệ điều hành",
        "unaccented_query": "cac thuat toan lap lich cpu va 4 dieu kien gay be tac deadlock trong he dieu hanh",
        "synonym_seed": "lập lịch cpu round robin",
        "expected_metric": "4 điều kiện",
    },
    AcademicDomain.AI_ML.value: {
        "title": "Trí tuệ nhân tạo & Mạng nơ-ron sâu",
        "doc_id": "DOC_E2E_AI_ML",
        "raw_text": """=== Page 1 ===
1. HỌC MÁY VÀ CÁC MÔ HÌNH HỌC CÓ GIÁM SÁT:
- Học máy (Machine Learning) nghiên cứu các thuật toán cho phép máy tính tự học từ dữ liệu.
- Phân loại học máy:
* Học có giám sát (Supervised Learning): Dữ liệu huấn luyện có nhãn đầu vào và đầu ra.
* Học không giám sát (Unsupervised Learning): Khám phá cấu trúc tiềm ẩn từ dữ liệu không gán nhãn.
* Học tăng cường (Reinforcement Learning): Tác tử học hành động tối ưu thông qua phần thưởng và phạt.
2. MẠNG NƠ-RON NHÂN TẠO VÀ HỌC SÂU:
- Mạng nơ-ron truyền thẳng nhiều tầng MLP với hàm kích hoạt ReLU, Sigmoid.
- Thuật toán lan truyền ngược (Backpropagation):
* Tính toán đạo hàm riêng của hàm mất mát loss function theo từng trọng số.
* Cập nhật trọng số thông qua thuật toán hạ độ dốc ngẫu nhiên SGD hoặc Adam optimizer.
3. MÔ HÌNH TRANSFORMER VÀ CƠ CHẾ TỰ CHÚ Ý:
- Kiến trúc Transformer loại bỏ mạng hồi quy RNN nhờ cơ chế tự chú ý nhiều đầu Multi-Head Attention.
""",
        "sample_query": "Mô hình mạng nơ-ron sâu và thuật toán lan truyền ngược trong học máy hoạt động ra sao?",
        "unaccented_query": "mo hinh mang no-ron sau va thuat toan lan truyen nguoc trong hoc may hoat dong ra sao",
        "synonym_seed": "mạng nơ-ron lan truyền ngược",
        "expected_metric": "Multi-Head Attention",
    },
    AcademicDomain.DATA_STRUCTURES_ALGORITHMS.value: {
        "title": "Cấu trúc dữ liệu & Thuật toán",
        "doc_id": "DOC_E2E_DSA",
        "raw_text": """=== Page 1 ===
1. CẤU TRÚC DỮ LIỆU CÂY VÀ ĐỒ THỊ:
- Cây nhị phân tìm kiếm (Binary Search Tree - BST):
* Nút con bên trái luôn có giá trị nhỏ hơn nút gốc.
* Nút con bên phải luôn có giá trị lớn hơn nút gốc.
- Đồ thị (Graph) gồm tập đỉnh V và tập cạnh E:
* Biểu diễn đồ thị bằng ma trận kề (Adjacency Matrix) hoặc danh sách kề (Adjacency List).
* Thuật toán duyệt BFS: Sử dụng hàng đợi FIFO Queue để duyệt theo từng mức độ sâu.
* Thuật toán duyệt DFS: Sử dụng ngăn xếp LIFO Stack hoặc đệ quy để đi sâu nhất có thể.
2. THUẬT TOÁN SẮP XẾP VÀ ĐỘ PHỨC TẠP THUẬT TOÁN:
- Ký hiệu Big-O biểu diễn chặn trên của thời gian thực thi:
* Thuật toán QuickSort: Chia để trị bằng cách chọn phần tử chốt pivot, độ phức tạp trung bình O(n log n).
* Thuật toán MergeSort: Chia mảng làm đôi, đệ quy sắp xếp và trộn lại, độ phức tạp luôn là O(n log n).
3. PHƯƠNG PHÁP QUY HOẠCH ĐỘNG:
- Giải bài toán tối ưu bằng cách chia thành các bài toán con gối nhau và lưu trữ kết quả memoization.
""",
        "sample_query": "Cây nhị phân tìm kiếm BST và thuật toán sắp xếp QuickSort có độ phức tạp như thế nào?",
        "unaccented_query": "cay nhi phan tim kiem bst va thuat toan sap xep quicksort co do phuc tap nhu the nao",
        "synonym_seed": "cây nhị phân tìm kiếm bst",
        "expected_metric": "O(n log n)",
    },
    AcademicDomain.SOFTWARE_ENGINEERING.value: {
        "title": "Công nghệ phần mềm & Mô hình Agile",
        "doc_id": "DOC_E2E_SE",
        "raw_text": """=== Page 1 ===
1. QUY TRÌNH PHÁT TRIỂN PHẦN MỀM VÀ MÔ HÌNH AGILE:
- Kỹ nghệ phần mềm là ứng dụng có hệ thống các nguyên lý công nghệ vào phát triển phần mềm.
- So sánh quy trình phát triển:
* Mô hình Thác nước (Waterfall): Tuần tự, nghiêm ngặt qua từng giai đoạn phân tích, thiết kế, cài đặt, kiểm thử.
* Khung làm việc Scrum: Lặp và tăng trưởng qua các chu kỳ Sprint từ 1 đến 4 tuần.
- Các vai trò trong Scrum:
* Product Owner: Quản lý danh mục sản phẩm Product Backlog.
* Scrum Master: Hướng dẫn đội ngũ tuân thủ nguyên tắc Scrum và loại bỏ rào cản.
* Development Team: Đội phát triển liên chức năng tự quản.
2. NGUYÊN LÝ THIẾT KẾ PHẦN MỀM SOLID:
- Nguyên lý thiết kế hướng đối tượng:
* Single Responsibility Principle (SRP): Mỗi lớp chỉ có một lý do duy nhất để thay đổi.
* Open/Closed Principle (OCP): Mở rộng cho phát triển, đóng cho sửa đổi.
* Liskov Substitution Principle (LSP): Lớp con có thể thay thế lớp cha mà không làm hỏng tính đúng đắn.
3. KIỂM THỬ PHẦN MỀM VÀ QUY TRÌNH CI/CD:
- Kiểm thử đơn vị (Unit Test), kiểm thử tích hợp (Integration Test) và kiểm thử hệ thống (System Test).
- Tích hợp liên tục CI và triển khai liên tục CD tự động hóa quy trình xây dựng và phát hành.
""",
        "sample_query": "Khung làm việc Scrum và các nguyên lý SOLID trong công nghệ phần mềm",
        "unaccented_query": "khung lam viec scrum va cac nguyen ly solid trong cong nghe phan mem",
        "synonym_seed": "quy trình phát triển agile scrum",
        "expected_metric": "4 tuần",
    },
}


class TestAcademicDomainRegistryMultiDomain:
    """Validates multi-domain registry and router performance across 7 computer science domains."""

    @pytest.fixture
    def registry(self) -> AcademicDomainRegistry:
        return AcademicDomainRegistry()

    def test_all_seven_domains_registered_and_metadata(self, registry: AcademicDomainRegistry):
        domains = registry.list_supported_domains()
        assert len(domains) == 7

        domain_ids = [d["domain_id"] for d in domains]
        expected_ids = [
            AcademicDomain.CRYPTOGRAPHY.value,
            AcademicDomain.NETWORKS.value,
            AcademicDomain.DATABASES.value,
            AcademicDomain.OPERATING_SYSTEMS.value,
            AcademicDomain.AI_ML.value,
            AcademicDomain.DATA_STRUCTURES_ALGORITHMS.value,
            AcademicDomain.SOFTWARE_ENGINEERING.value,
        ]
        for eid in expected_ids:
            assert eid in domain_ids
            meta = get_academic_domain_metadata(eid)
            assert meta["name_vi"] != ""
            assert meta["name_en"] != ""
            assert len(meta["aliases"]) > 0

    def test_alias_resolution_all_seven_domains(self, registry: AcademicDomainRegistry):
        alias_test_cases = {
            AcademicDomain.CRYPTOGRAPHY.value: ["crypto", "mật mã", "mật mã học", "attt", "an ninh mạng"],
            AcademicDomain.NETWORKS.value: ["mạng", "mạng máy tính", "networking", "telecom", "viễn thông"],
            AcademicDomain.DATABASES.value: ["csdl", "cơ sở dữ liệu", "database", "rdbms", "sql"],
            AcademicDomain.OPERATING_SYSTEMS.value: ["hdh", "hệ điều hành", "os", "operating_systems"],
            AcademicDomain.AI_ML.value: ["ai", "trí tuệ nhân tạo", "học máy", "ml", "ai/ml"],
            AcademicDomain.DATA_STRUCTURES_ALGORITHMS.value: ["ctdl&gt", "cấu trúc dữ liệu", "giải thuật", "dsa", "thuật toán"],
            AcademicDomain.SOFTWARE_ENGINEERING.value: ["cnpm", "công nghệ phần mềm", "se", "kỹ nghệ phần mềm"],
        }
        for expected_domain, aliases in alias_test_cases.items():
            for alias in aliases:
                resolved = registry.get_domain_by_alias(alias)
                assert resolved == expected_domain, f"Alias '{alias}' failed to resolve to '{expected_domain}'"

    def test_query_routing_all_seven_domains(self, registry: AcademicDomainRegistry):
        for domain_id, data in SAMPLE_CORPUS_7_DOMAINS.items():
            q = data["sample_query"]
            t0 = time.perf_counter()
            res = registry.route_query(q)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            assert res.primary_domain == domain_id, (
                f"Query '{q}' expected {domain_id}, got {res.primary_domain}"
            )
            assert res.confidence >= 0.70, (
                f"Confidence for '{q}' too low: {res.confidence}"
            )
            assert latency_ms < 15.0

    def test_unaccented_query_routing_resilience(self, registry: AcademicDomainRegistry):
        for domain_id, data in SAMPLE_CORPUS_7_DOMAINS.items():
            uq = data["unaccented_query"]
            res = registry.route_query(uq)
            assert res.primary_domain == domain_id, (
                f"Unaccented query '{uq}' expected {domain_id}, got {res.primary_domain}"
            )
            assert res.confidence >= 0.65

    def test_document_routing_all_seven_domains(self, registry: AcademicDomainRegistry):
        for domain_id, data in SAMPLE_CORPUS_7_DOMAINS.items():
            doc_text = data["raw_text"]
            res = registry.route_document(doc_text)
            assert res.primary_domain == domain_id, (
                f"Document '{data['title']}' expected {domain_id}, got {res.primary_domain}"
            )
            assert res.confidence >= 0.80

    def test_cross_domain_query_routing(self, registry: AcademicDomainRegistry):
        cross_query = "Giao thức bảo mật TLS và mã hóa AES trong truyền thông mạng máy tính"
        res = registry.route_query(cross_query)
        assert res.primary_domain in [AcademicDomain.CRYPTOGRAPHY.value, AcademicDomain.NETWORKS.value]
        assert (
            AcademicDomain.CRYPTOGRAPHY.value in res.secondary_domains
            or AcademicDomain.NETWORKS.value in res.secondary_domains
            or res.scores.get(AcademicDomain.CRYPTOGRAPHY.value, 0) > 0
            and res.scores.get(AcademicDomain.NETWORKS.value, 0) > 0
        )

    def test_custom_keyword_registration_runtime(self, registry: AcademicDomainRegistry):
        unique_term = "quantum-key-distribution-bb84"
        registry.register_custom_keyword(AcademicDomain.CRYPTOGRAPHY.value, unique_term, weight=5.0)
        res = registry.route_query(f"Nghiên cứu giao thức {unique_term} trong bảo mật")
        assert res.primary_domain == AcademicDomain.CRYPTOGRAPHY.value
        assert res.confidence > 0.85

    def test_empty_and_whitespace_query_routing(self, registry: AcademicDomainRegistry):
        empty_res = registry.route_query("")
        assert empty_res.primary_domain == AcademicDomain.GENERAL.value
        assert empty_res.confidence == 0.0

        ws_res = registry.route_query("   \t\n  ")
        assert ws_res.primary_domain == AcademicDomain.GENERAL.value
        assert ws_res.confidence == 0.0


class TestMultiDomainSynonymExpansion:
    """Validates multi-domain academic synonym expansion across all 7 subjects."""

    def test_synonym_expansion_all_seven_domains(self):
        for domain_id, data in SAMPLE_CORPUS_7_DOMAINS.items():
            seed = data["synonym_seed"]
            synonyms = expand_all_academic_synonyms(seed)

            assert isinstance(synonyms, list)
            assert len(synonyms) >= 5, f"Domain {domain_id} with seed '{seed}' returned too few synonyms: {synonyms}"
            # Verify uniqueness
            assert len(synonyms) == len(dict.fromkeys(synonyms))

    def test_unaccented_synonym_expansion(self):
        unaccented_seeds = [
            ("ma hoa doi xung", AcademicDomain.CRYPTOGRAPHY.value),
            ("bat tay 3 buoc tcp", AcademicDomain.NETWORKS.value),
            ("chuan hoa du lieu", AcademicDomain.DATABASES.value),
            ("lap lich cpu", AcademicDomain.OPERATING_SYSTEMS.value),
        ]
        for seed, _ in unaccented_seeds:
            synonyms = expand_all_academic_synonyms(seed)
            assert isinstance(synonyms, list)
            assert len(synonyms) >= 10

    def test_empty_and_unknown_query_expansion(self):
        assert expand_all_academic_synonyms("") == []
        assert expand_all_academic_synonyms("   ") == []

        random_term = "khái niệm phi học thuật ngẫu nhiên zzz123"
        res = expand_all_academic_synonyms(random_term)
        assert isinstance(res, list)

    def test_domain_specific_synonym_getter(self):
        reg = AcademicDomainRegistry()
        crypto_syns = reg.get_domain_synonyms(AcademicDomain.CRYPTOGRAPHY.value, "aes")
        assert isinstance(crypto_syns, list)
        assert len(crypto_syns) > 0


class TestSelfStudyKnowledgeBankMultiDomain:
    """Validates knowledge extraction, blueprint classification, and learned QA across 7 subjects."""

    @pytest.fixture
    def bank(self) -> SelfStudyKnowledgeBank:
        b = SelfStudyKnowledgeBank()
        b.clear()
        return b

    def test_knowledge_bank_learn_all_seven_domains(self, bank: SelfStudyKnowledgeBank):
        for domain_id, data in SAMPLE_CORPUS_7_DOMAINS.items():
            res = bank.learn_document(
                doc_id=data["doc_id"],
                title=data["title"],
                raw_text=data["raw_text"],
            )
            assert res["status"] == "learned"
            assert res["sections_count"] >= 2
            assert res["doc_id"] == data["doc_id"]

    def test_blueprint_classification_across_domains(self, bank: SelfStudyKnowledgeBank):
        test_queries = {
            "Tổng quan toàn bộ nội dung tài liệu này là gì?": "SUMMARY_OVERVIEW",
            "Mã hóa đối xứng là gì, có bao nhiêu loại, cho ví dụ minh họa": "DEFINITION_AND_TYPES",
            "Nguyên lý hoạt động và công thức tính toán của thuật toán": "MECHANISM_AND_FORMULA",
            "So sánh an ninh vô điều kiện và an ninh tính toán, ưu nhược điểm": "COMPARISON_AND_SECURITY",
            "Cho ví dụ minh họa từng bước thực hiện": "EXAMPLE_WALKTHROUGH",
            "1. GIỚI THIỆU HỆ MÃ KHÓA ĐỐI XỨNG": "SECTION_FULL_EXPLANATION",
        }
        for query, expected_bp in test_queries.items():
            bp = bank.classify_answer_blueprint(query)
            assert bp == expected_bp, f"Query '{query}' expected '{expected_bp}', got '{bp}'"

    def test_synthesize_learned_answer_across_domains(self, bank: SelfStudyKnowledgeBank):
        # Learn all 7 documents
        for data in SAMPLE_CORPUS_7_DOMAINS.values():
            bank.learn_document(data["doc_id"], data["title"], data["raw_text"])

        # Query OS document
        os_ans = bank.synthesize_learned_answer("tiến trình là gì, có các thuật toán nào")
        assert os_ans is not None
        assert "answer" in os_ans
        assert len(os_ans["answer"]) > 20
        assert os_ans.get("grounded", False) is True

        # Query Cryptography document
        crypto_ans = bank.synthesize_learned_answer("1. GIỚI THIỆU HỆ MÃ KHÓA ĐỐI XỨNG")
        assert crypto_ans is not None
        assert "answer" in crypto_ans

    def test_clear_and_relearn(self, bank: SelfStudyKnowledgeBank):
        bank.learn_document("DOC_TEMP", "Temp Title", "1. MỤC 1\n- Nội dung tạm thời")
        assert len(bank._documents) == 1
        bank.clear()
        assert len(bank._documents) == 0
        assert bank.synthesize_learned_answer("nội dung") is None

    def test_singleton_instance_consistency(self):
        kb1 = get_knowledge_bank()
        kb2 = get_knowledge_bank()
        assert kb1 is kb2


class TestDocumentAnalyzerMultiDomain:
    """Validates document blueprint generation, key takeaways, and suggested questions across 7 subjects."""

    def test_extract_key_points_and_summary_seven_domains(self):
        for domain_id, data in SAMPLE_CORPUS_7_DOMAINS.items():
            res = DocumentAnalyzer.extract_key_points_and_summary(
                raw_text=data["raw_text"],
                title=data["title"],
                max_points=4,
            )
            assert "overview" in res
            assert len(res["overview"]) > 10
            assert "key_points" in res
            assert len(res["key_points"]) > 0

    def test_generate_suggested_questions_seven_domains(self):
        for domain_id, data in SAMPLE_CORPUS_7_DOMAINS.items():
            questions = DocumentAnalyzer.generate_suggested_questions(
                raw_text=data["raw_text"],
                title=data["title"],
                max_questions=4,
            )
            assert isinstance(questions, list)
            assert len(questions) >= 1
            for q in questions:
                assert "question" in q
                assert "category" in q
                assert len(q["question"]) > 5

    def test_build_knowledge_blueprint_seven_domains(self):
        for domain_id, data in SAMPLE_CORPUS_7_DOMAINS.items():
            blueprint = DocumentAnalyzer.build_document_knowledge_blueprint(
                raw_text=data["raw_text"],
                title=data["title"],
            )
            assert isinstance(blueprint, str)
            assert "BẢN ĐỒ TRI THỨC TÀI LIỆU" in blueprint
            assert data["title"] in blueprint
            assert "Tóm lược cốt lõi" in blueprint


class TestEndToEndMultiDomainPipeline:
    """Validates unified pipeline integration across all 7 computer science domains."""

    def test_complete_e2e_pipeline_seven_domains(self):
        registry = AcademicDomainRegistry()
        bank = SelfStudyKnowledgeBank()
        bank.clear()

        pipeline_results = {}

        for domain_id, data in SAMPLE_CORPUS_7_DOMAINS.items():
            # Step 1: Automatic Domain Routing
            route_res: DomainRouteResult = registry.route_document(data["raw_text"])
            assert route_res.primary_domain == domain_id
            assert route_res.confidence >= 0.80

            # Step 2: Academic Synonym Expansion
            synonyms = expand_all_academic_synonyms(data["synonym_seed"])
            assert len(synonyms) >= 5

            # Step 3: Fast Structural Knowledge Learning
            learn_res = bank.learn_document(
                doc_id=data["doc_id"],
                title=data["title"],
                raw_text=data["raw_text"],
            )
            assert learn_res["status"] == "learned"
            assert learn_res["sections_count"] >= 2

            # Step 4: Executive Blueprint Synthesis
            blueprint = DocumentAnalyzer.build_document_knowledge_blueprint(
                raw_text=data["raw_text"],
                title=data["title"],
            )
            assert len(blueprint) > 100

            # Step 5: Document-Grounded Question Generation
            suggested_q = DocumentAnalyzer.generate_suggested_questions(
                raw_text=data["raw_text"],
                title=data["title"],
                max_questions=3,
            )
            assert len(suggested_q) >= 1

            # Step 6: Grounded QA Synthesis
            synthesized = bank.synthesize_learned_answer(data["sample_query"], doc_ids=[data["doc_id"]])
            assert synthesized is not None
            assert len(synthesized.get("answer", "")) > 10

            pipeline_results[domain_id] = {
                "route": route_res.primary_domain,
                "confidence": route_res.confidence,
                "synonyms_count": len(synonyms),
                "sections": learn_res["sections_count"],
                "questions_generated": len(suggested_q),
            }

        # Assert all 7 domains completed the full pipeline
        assert len(pipeline_results) == 7
        for dom, stats in pipeline_results.items():
            assert stats["route"] == dom
            assert stats["confidence"] >= 0.80
            assert stats["synonyms_count"] >= 5
            assert stats["sections"] >= 2
            assert stats["questions_generated"] >= 1

    def test_cross_domain_isolation_no_contamination(self):
        bank = SelfStudyKnowledgeBank()
        bank.clear()

        # Learn only Databases and Networks
        db_data = SAMPLE_CORPUS_7_DOMAINS[AcademicDomain.DATABASES.value]
        net_data = SAMPLE_CORPUS_7_DOMAINS[AcademicDomain.NETWORKS.value]

        bank.learn_document(db_data["doc_id"], db_data["title"], db_data["raw_text"])
        bank.learn_document(net_data["doc_id"], net_data["title"], net_data["raw_text"])

        # Targeted query to DB only
        db_res = bank.synthesize_learned_answer(db_data["sample_query"], doc_ids=[db_data["doc_id"]])
        assert db_res is not None
        assert "TCP" not in db_res.get("answer", "")

        # Targeted query to Networks only
        net_res = bank.synthesize_learned_answer(net_data["sample_query"], doc_ids=[net_data["doc_id"]])
        assert net_res is not None
        assert "BCNF" not in net_res.get("answer", "")

    def test_sub_millisecond_routing_throughput(self):
        registry = AcademicDomainRegistry()
        queries = [d["sample_query"] for d in SAMPLE_CORPUS_7_DOMAINS.values()] * 10  # 70 queries

        t0 = time.perf_counter()
        for q in queries:
            res = registry.route_query(q)
            assert res.primary_domain != AcademicDomain.GENERAL.value
        total_time_ms = (time.perf_counter() - t0) * 1000.0
        avg_time_ms = total_time_ms / len(queries)

        assert avg_time_ms < 2.0  # Ultra-fast sub-millisecond or low-millisecond average
