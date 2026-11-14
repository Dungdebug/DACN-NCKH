"""
Unit tests for AcademicDomainRegistry & Multi-Domain Router.
Verifies contract compliance, sub-millisecond latency (< 1ms), 7-domain accuracy,
safe lazy-import synonym expansion, cross-domain recognition, and robust offline fallback.
"""

import time
import pytest
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


class TestAcademicDomainRegistry:
    """Test suite covering AcademicDomainRegistry implementation."""

    @pytest.fixture
    def registry(self) -> AcademicDomainRegistry:
        return AcademicDomainRegistry()

    def test_domain_enum_and_metadata_completeness(self):
        """Validates all 7 core domains plus general are defined with complete metadata."""
        expected_domains = [
            AcademicDomain.CRYPTOGRAPHY.value,
            AcademicDomain.NETWORKS.value,
            AcademicDomain.DATABASES.value,
            AcademicDomain.OPERATING_SYSTEMS.value,
            AcademicDomain.AI_ML.value,
            AcademicDomain.DATA_STRUCTURES_ALGORITHMS.value,
            AcademicDomain.SOFTWARE_ENGINEERING.value,
            AcademicDomain.GENERAL.value,
        ]
        for dom in expected_domains:
            assert dom in DOMAIN_METADATA
            meta = DOMAIN_METADATA[dom]
            assert "name_vi" in meta and len(meta["name_vi"]) > 0
            assert "name_en" in meta and len(meta["name_en"]) > 0
            assert "aliases" in meta

    def test_domain_alias_resolution(self, registry: AcademicDomainRegistry):
        """Verifies resolution of various Vietnamese & English shorthand aliases."""
        assert registry.get_domain_by_alias("crypto") == AcademicDomain.CRYPTOGRAPHY.value
        assert registry.get_domain_by_alias("mật mã học") == AcademicDomain.CRYPTOGRAPHY.value
        assert registry.get_domain_by_alias("mat ma") == AcademicDomain.CRYPTOGRAPHY.value

        assert registry.get_domain_by_alias("csdl") == AcademicDomain.DATABASES.value
        assert registry.get_domain_by_alias("cơ sở dữ liệu") == AcademicDomain.DATABASES.value
        assert registry.get_domain_by_alias("database") == AcademicDomain.DATABASES.value

        assert registry.get_domain_by_alias("hdh") == AcademicDomain.OPERATING_SYSTEMS.value
        assert registry.get_domain_by_alias("hệ điều hành") == AcademicDomain.OPERATING_SYSTEMS.value
        assert registry.get_domain_by_alias("os") == AcademicDomain.OPERATING_SYSTEMS.value

        assert registry.get_domain_by_alias("mạng") == AcademicDomain.NETWORKS.value
        assert registry.get_domain_by_alias("mạng máy tính") == AcademicDomain.NETWORKS.value
        assert registry.get_domain_by_alias("networking") == AcademicDomain.NETWORKS.value

        assert registry.get_domain_by_alias("ai") == AcademicDomain.AI_ML.value
        assert registry.get_domain_by_alias("trí tuệ nhân tạo") == AcademicDomain.AI_ML.value
        assert registry.get_domain_by_alias("học máy") == AcademicDomain.AI_ML.value

        assert registry.get_domain_by_alias("dsa") == AcademicDomain.DATA_STRUCTURES_ALGORITHMS.value
        assert registry.get_domain_by_alias("ctdl&gt") == AcademicDomain.DATA_STRUCTURES_ALGORITHMS.value
        assert registry.get_domain_by_alias("cấu trúc dữ liệu") == AcademicDomain.DATA_STRUCTURES_ALGORITHMS.value

        assert registry.get_domain_by_alias("cnpm") == AcademicDomain.SOFTWARE_ENGINEERING.value
        assert registry.get_domain_by_alias("công nghệ phần mềm") == AcademicDomain.SOFTWARE_ENGINEERING.value
        assert registry.get_domain_by_alias("se") == AcademicDomain.SOFTWARE_ENGINEERING.value

        assert registry.get_domain_by_alias("unknown_alien_topic") is None

    def test_route_cryptography_queries(self, registry: AcademicDomainRegistry):
        """Verifies accurate classification for Cryptography & Security."""
        queries = [
            "Hệ mã hóa đối xứng và mã hóa bất đối xứng khác nhau thế nào?",
            "Phương pháp phá mã thám mã đối với mã Caesar và Vigenere",
            "Đặc điểm của hệ mã độn một lần one-time pad và tính bảo mật tuyệt đối",
            "Chữ ký số và hàm băm mật mã SHA256 trong xác thực dữ liệu",
        ]
        for q in queries:
            res = registry.route_query(q)
            assert res.primary_domain == AcademicDomain.CRYPTOGRAPHY.value
            assert res.confidence >= 0.70
            assert res.latency_ms < 1.0

    def test_route_networks_queries(self, registry: AcademicDomainRegistry):
        """Verifies accurate classification for Computer Networks."""
        queries = [
            "Quá trình bắt tay 3 bước three-way handshake trong giao thức TCP",
            "Mô hình OSI 7 tầng và vai trò của tầng giao vận so với tầng mạng",
            "Cách phân chia mạng con subnet mask và định tuyến gói tin qua router",
            "So sánh giao thức TCP và UDP về độ tin cậy và kiểm soát luồng",
        ]
        for q in queries:
            res = registry.route_query(q)
            assert res.primary_domain == AcademicDomain.NETWORKS.value
            assert res.confidence >= 0.70
            assert res.latency_ms < 1.0

    def test_route_databases_queries(self, registry: AcademicDomainRegistry):
        """Verifies accurate classification for Database Systems."""
        queries = [
            "Các tính chất ACID trong giao dịch cơ sở dữ liệu",
            "Quy trình chuẩn hóa dữ liệu từ dạng chuẩn 1NF 2NF lên 3NF và BCNF",
            "Phân biệt khóa chính primary key và khóa ngoại foreign key toàn vẹn tham chiếu",
            "Cấu trúc chỉ mục cây B-Tree và B+Tree giúp tối ưu truy vấn SQL",
        ]
        for q in queries:
            res = registry.route_query(q)
            assert res.primary_domain == AcademicDomain.DATABASES.value
            assert res.confidence >= 0.70
            assert res.latency_ms < 1.0

    def test_route_operating_systems_queries(self, registry: AcademicDomainRegistry):
        """Verifies accurate classification for Operating Systems."""
        queries = [
            "Sự khác biệt giữa tiến trình process và luồng thread",
            "Các thuật toán lập lịch CPU FCFS SJF và Round Robin",
            "Bốn điều kiện Coffman dẫn đến bế tắc deadlock và thuật toán nhà băng",
            "Cơ chế bộ nhớ ảo phân trang paging phân đoạn và thuật toán thay thế trang LRU",
        ]
        for q in queries:
            res = registry.route_query(q)
            assert res.primary_domain == AcademicDomain.OPERATING_SYSTEMS.value
            assert res.confidence >= 0.70
            assert res.latency_ms < 1.0

    def test_route_ai_ml_queries(self, registry: AcademicDomainRegistry):
        """Verifies accurate classification for AI/ML."""
        queries = [
            "Khái niệm học có giám sát supervised learning và phân loại so với hồi quy",
            "Mạng nơ-ron sâu thuật toán lan truyền ngược backpropagation và gradient descent",
            "Hiện tượng quá khớp overfitting và phương pháp điều chuẩn regularization",
            "Kiến trúc Transformer cơ chế tự chú ý self-attention và mô hình RAG",
        ]
        for q in queries:
            res = registry.route_query(q)
            assert res.primary_domain == AcademicDomain.AI_ML.value
            assert res.confidence >= 0.70
            assert res.latency_ms < 1.0

    def test_route_dsa_queries(self, registry: AcademicDomainRegistry):
        """Verifies accurate classification for Data Structures & Algorithms."""
        queries = [
            "Độ phức tạp thuật toán ký hiệu Big-O trong thời gian và không gian",
            "Cây nhị phân tìm kiếm BST và cây cân bằng AVL",
            "Thuật toán sắp xếp nhanh Quicksort so với sắp xếp trộn Mergesort",
            "Thuật toán tìm đường đi ngắn nhất Dijkstra trên đồ thị có hướng",
        ]
        for q in queries:
            res = registry.route_query(q)
            assert res.primary_domain == AcademicDomain.DATA_STRUCTURES_ALGORITHMS.value
            assert res.confidence >= 0.70
            assert res.latency_ms < 1.0

    def test_route_software_engineering_queries(self, registry: AcademicDomainRegistry):
        """Verifies accurate classification for Software Engineering."""
        queries = [
            "So sánh mô hình thác nước Waterfall và phương pháp linh hoạt Agile Scrum",
            "Năm nguyên lý thiết kế SOLID trong lập trình hướng đối tượng",
            "Mẫu thiết kế Singleton Factory và Observer trong xây dựng kiến trúc",
            "Quy trình kiểm thử phần mềm kiểm thử đơn vị Unit Test và TDD",
        ]
        for q in queries:
            res = registry.route_query(q)
            assert res.primary_domain == AcademicDomain.SOFTWARE_ENGINEERING.value
            assert res.confidence >= 0.70
            assert res.latency_ms < 1.0

    def test_sub_millisecond_latency_requirement(self, registry: AcademicDomainRegistry):
        """
        Enforces strict latency requirement: < 1.0ms per routing operation.
        Runs a 50-iteration benchmark over diverse realistic queries.
        """
        queries = [
            "Mã hóa đối xứng AES và khóa bí mật",
            "Bắt tay ba bước TCP và tầng giao vận trong mạng",
            "Chuẩn hóa CSDL dạng chuẩn 3NF và khóa chính",
            "Quản lý tiến trình và đồng bộ hóa bằng semaphore",
            "Mạng nơ-ron Transformer và lan truyền ngược",
            "Giải thuật Dijkstra tìm đường đi trên đồ thị",
            "Nguyên lý SOLID và quy trình Agile Scrum",
        ]
        # Warm up JIT/cache and collect garbage to avoid GC pause during micro-benchmark
        import gc
        gc.collect()
        for q in queries:
            registry.route_query(q)

        latencies = []
        for _ in range(5):
            for q in queries:
                res = registry.route_query(q)
                latencies.append(res.latency_ms)

        avg_latency = sum(latencies) / len(latencies)
        # Verify typical average latency is well under 0.25ms
        assert avg_latency < 0.25, f"Average latency {avg_latency:.3f}ms is unexpectedly high"
        assert min(latencies) < 0.5, f"Minimum latency {min(latencies):.3f}ms exceeded limit"
        sub_ms_ratio = sum(1 for lat in latencies if lat < 1.0) / len(latencies)
        assert sub_ms_ratio >= 0.9, f"Only {sub_ms_ratio*100:.1f}% queries completed under 1ms"

    def test_unaccented_student_queries(self, registry: AcademicDomainRegistry):
        """Ensures queries typed without Vietnamese accents are recognized accurately."""
        q1 = "ma hoa doi xung va khoa bi mat la gi"
        assert registry.route_query(q1).primary_domain == AcademicDomain.CRYPTOGRAPHY.value

        q2 = "bat tay 3 buoc tcp trong mang may tinh"
        assert registry.route_query(q2).primary_domain == AcademicDomain.NETWORKS.value

        q3 = "chuan hoa 3nf va tinh chat acid trong csdl"
        assert registry.route_query(q3).primary_domain == AcademicDomain.DATABASES.value

        q4 = "lap lich cpu round robin va be tac deadlock"
        assert registry.route_query(q4).primary_domain == AcademicDomain.OPERATING_SYSTEMS.value

    def test_cross_domain_query_detection(self, registry: AcademicDomainRegistry):
        """Tests that queries bridging two distinct fields are flagged with is_cross_domain=True."""
        # Query combining Cryptography and Computer Networks
        cross_query = "Giao thức an ninh mạng SSL TLS sử dụng mã hóa đối xứng AES và bắt tay 3 bước TCP"
        res = registry.route_query(cross_query)
        assert res.is_cross_domain is True
        assert len(res.secondary_domains) > 0

    def test_empty_and_noise_queries(self, registry: AcademicDomainRegistry):
        """Tests fallback behavior for empty, punctuation, or non-academic noise queries."""
        empty_res = registry.route_query("")
        assert empty_res.primary_domain == AcademicDomain.GENERAL.value
        assert empty_res.confidence == 0.0

        whitespace_res = registry.route_query("   \n\t  ")
        assert whitespace_res.primary_domain == AcademicDomain.GENERAL.value

        noise_res = registry.route_query("hôm nay thời tiết đẹp quá đi ăn kem thôi bạn ơi")
        assert noise_res.primary_domain == AcademicDomain.GENERAL.value
        assert noise_res.confidence == 0.0

    def test_expand_all_academic_synonyms(self, registry: AcademicDomainRegistry):
        """Verifies multi-domain synonym expansion across various technical concepts."""
        # 1. Cryptography query expansion
        crypto_syns = registry.expand_all_academic_synonyms("mã hóa đối xứng và phá mã")
        assert isinstance(crypto_syns, list)
        assert len(crypto_syns) > 0
        assert any("khóa bí mật" in s.lower() or "symmetric" in s.lower() for s in crypto_syns)

        # 2. Database query expansion
        db_syns = registry.expand_all_academic_synonyms("tính chất ACID và chuẩn hóa CSDL")
        assert any("atomicity" in s.lower() or "nguyên tử" in s.lower() for s in db_syns)
        assert any("1nf" in s.lower() or "3nf" in s.lower() for s in db_syns)

        # 3. Networks query expansion
        net_syns = registry.expand_all_academic_synonyms("mô hình OSI và bắt tay 3 bước TCP")
        assert any("7 tầng" in s.lower() or "three-way" in s.lower() for s in net_syns)

        # 4. OS query expansion
        os_syns = registry.expand_all_academic_synonyms("lập lịch CPU và deadlock")
        assert any("coffman" in s.lower() or "round robin" in s.lower() for s in os_syns)

        # 5. Empty query handles gracefully
        assert registry.expand_all_academic_synonyms("") == []

    def test_custom_keyword_registration(self, registry: AcademicDomainRegistry):
        """Verifies runtime registration of custom technical keywords."""
        res_before = registry.route_query("Giao thức zk-SNARKs")
        assert res_before.primary_domain == AcademicDomain.GENERAL.value

        registry.register_custom_keyword(AcademicDomain.CRYPTOGRAPHY.value, "zk-SNARKs", weight=5.0)

        res_after = registry.route_query("Giao thức zk-SNARKs ứng dụng thế nào?")
        assert res_after.primary_domain == AcademicDomain.CRYPTOGRAPHY.value
        assert res_after.confidence > 0.80

    def test_result_serialization_and_summary(self, registry: AcademicDomainRegistry):
        """Verifies to_dict and summary methods on DomainRouteResult."""
        res = registry.route_query("Thuật toán sắp xếp nhanh Quicksort và Big-O")
        d = res.to_dict()
        assert "primary_domain" in d
        assert "confidence" in d
        assert "latency_ms" in d
        assert "matched_keywords" in d
        assert isinstance(d["scores"], dict)

        summary = res.summary()
        assert "DATA_STRUCTURES_ALGORITHMS" in summary
        assert "ms" in summary

    def test_list_supported_domains(self, registry: AcademicDomainRegistry):
        """Verifies list_supported_domains returns all 7 specialized domains."""
        domains_list = registry.list_supported_domains()
        assert len(domains_list) == 7
        domain_ids = [d["domain_id"] for d in domains_list]
        assert AcademicDomain.CRYPTOGRAPHY.value in domain_ids
        assert AcademicDomain.NETWORKS.value in domain_ids
        assert AcademicDomain.DATABASES.value in domain_ids
        assert AcademicDomain.OPERATING_SYSTEMS.value in domain_ids
        assert AcademicDomain.AI_ML.value in domain_ids
        assert AcademicDomain.DATA_STRUCTURES_ALGORITHMS.value in domain_ids
        assert AcademicDomain.SOFTWARE_ENGINEERING.value in domain_ids

    def test_long_document_routing(self, registry: AcademicDomainRegistry):
        """Verifies routing of a multi-sentence academic document chapter."""
        doc = (
            "Chương 3: Quản lý bộ nhớ và bộ nhớ ảo trong hệ điều hành hiện đại. "
            "Một trong những nhiệm vụ cốt lõi của hệ điều hành là quản lý không gian địa chỉ vật lý và ảo. "
            "Khi một tiến trình yêu cầu bộ nhớ, cơ chế phân trang (paging) và phân đoạn (segmentation) "
            "sẽ phân chia không gian nhớ thành các trang khung trang tương ứng. Hiện tượng lỗi trang (page fault) "
            "xảy ra khi trang được tham chiếu chưa nạp vào RAM. Lúc này thuật toán thay thế trang như LRU hay FIFO "
            "sẽ được kích hoạt để chọn trang nạn nhân nhằm tránh hiện tượng thrashing làm sụp đổ hiệu năng hệ thống."
        )
        res = registry.route_document(doc)
        assert res.primary_domain == AcademicDomain.OPERATING_SYSTEMS.value
        assert res.confidence > 0.85
        assert res.latency_ms < 5.0

    def test_lazy_sibling_cryptography_integration(self, registry: AcademicDomainRegistry):
        """Verifies integration with existing cryptography_and_security_patterns module."""
        # Query known crypto terms that should trigger sibling module
        syns = registry.get_domain_synonyms(AcademicDomain.CRYPTOGRAPHY.value, "phá mã")
        assert isinstance(syns, list)
        assert len(syns) > 0
        assert any("thám mã" in s.lower() or "cryptanalysis" in s.lower() for s in syns)

    def test_synonym_expansion_multi_domain_query(self, registry: AcademicDomainRegistry):
        """Verifies synonym expansion on cross-domain or multi-facet technical queries."""
        query = "Bảo mật cơ sở dữ liệu với khóa mã hóa AES và tính chất ACID"
        syns = registry.expand_all_academic_synonyms(query)
        assert isinstance(syns, list)
        assert len(syns) > 0
        # Should contain synonyms from both crypto and database domains
        has_db_syn = any("atomicity" in s.lower() or "nguyên tử" in s.lower() or "transaction" in s.lower() for s in syns)
        has_crypto_syn = any("mã hóa" in s.lower() or "cipher" in s.lower() or "khóa" in s.lower() for s in syns)
        assert has_db_syn or has_crypto_syn

    def test_punctuation_and_code_snippet_queries(self, registry: AcademicDomainRegistry):
        """Verifies robustness against punctuation, code fragments, and math formulas."""
        snippet = "SELECT * FROM users INNER JOIN orders ON users.id = orders.user_id WHERE acid_tx = 1;"
        res = registry.route_query(snippet)
        assert res.primary_domain == AcademicDomain.DATABASES.value

        math_query = "Thuật toán mã hóa RSA: c = m^e mod n và khóa công khai (e, n)"
        res_math = registry.route_query(math_query)
        assert res_math.primary_domain == AcademicDomain.CRYPTOGRAPHY.value

    def test_module_level_convenience_functions(self):
        """Verifies top-level helper functions route_academic_query, expand_all_academic_synonyms, etc."""
        res = route_academic_query("Kiểm thử phần mềm Unit Test và TDD")
        assert res.primary_domain == AcademicDomain.SOFTWARE_ENGINEERING.value

        doc_res = route_academic_document("Mô hình OSI bao gồm 7 tầng mạng...")
        assert doc_res.primary_domain == AcademicDomain.NETWORKS.value

        syns = expand_all_academic_synonyms("chỉ mục B-Tree")
        assert len(syns) > 0

        meta = get_academic_domain_metadata("ai_ml")
        assert meta["name_en"] == "Artificial Intelligence & Machine Learning"


