"""
Domain-Specific Software Engineering & Architecture Semantic Patterns.
Provides bidirectional synonym mapping, domain query expansion, and few-shot QA blueprints
for authoritative Software Engineering, SDLC, Agile/Scrum, SOLID Principles, Design Patterns,
Testing (Unit/Integration/TDD), and CI/CD/DevOps education.
100% Offline, Pure Python stdlib, Zero External Dependencies.
"""

import re
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple


def strip_accents(text: str) -> str:
    """Removes Vietnamese tone marks for accent-insensitive canonical alignment."""
    if not text:
        return ""
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D")


def normalize_term(term: str) -> str:
    """Standardizes casing and whitespace for consistent dictionary lookup."""
    return " ".join(term.strip().lower().split())


# Core Software Engineering domain synonym clusters
SE_SYNONYM_CLUSTERS: Dict[str, List[str]] = {
    # 1. SDLC & Process Models
    "sdlc": [
        "sdlc",
        "vòng đời phát triển phần mềm",
        "chu kỳ phát triển phần mềm",
        "quy trình phát triển phần mềm",
        "tiến trình phần mềm",
        "software development life cycle",
        "software development lifecycle",
        "software life cycle",
    ],
    "waterfall": [
        "waterfall",
        "mô hình thác nước",
        "thác nước",
        "mô hình tuần tự",
        "tuần tự tuyến tính",
        "waterfall model",
        "linear sequential model",
        "traditional sdlc",
    ],
    "agile": [
        "agile",
        "phương pháp linh hoạt",
        "phương pháp agile",
        "mô hình agile",
        "phát triển linh hoạt",
        "agile software development",
        "agile methodology",
        "tuyên ngôn agile",
        "agile manifesto",
    ],
    "scrum": [
        "scrum",
        "khung làm việc scrum",
        "phương pháp scrum",
        "scrum framework",
        "sprint",
        "chu kỳ sprint",
        "product backlog",
        "sprint backlog",
        "scrum master",
        "product owner",
        "development team",
        "daily scrum",
        "sprint review",
        "sprint retrospective",
    ],
    "spiral_and_incremental": [
        "spiral model",
        "mô hình xoắn ốc",
        "xoắn ốc",
        "incremental model",
        "mô hình tăng trưởng",
        "lặp và tăng dần",
        "iterative and incremental",
    ],

    # 2. Requirements & Modeling
    "requirements_engineering": [
        "kỹ nghệ yêu cầu",
        "phân tích yêu cầu",
        "quản lý yêu cầu",
        "thu thập yêu cầu",
        "requirements engineering",
        "requirements analysis",
        "srs",
        "software requirements specification",
        "user story",
        "câu chuyện người dùng",
        "use case",
        "trường hợp sử dụng",
    ],
    "uml": [
        "uml",
        "ngôn ngữ mô hình hóa thống nhất",
        "unified modeling language",
        "class diagram",
        "biểu đồ lớp",
        "sequence diagram",
        "biểu đồ tuần tự",
        "use case diagram",
        "biểu đồ ca sử dụng",
        "activity diagram",
        "biểu đồ hoạt động",
    ],

    # 3. SOLID Principles
    "solid": [
        "solid",
        "nguyên lý solid",
        "5 nguyên lý solid",
        "solid principles",
        "nguyên tắc solid",
        "thiết kế hướng đối tượng solid",
    ],
    "srp": [
        "srp",
        "single responsibility",
        "single responsibility principle",
        "nguyên lý đơn trách nhiệm",
        "nguyên tắc đơn trách nhiệm",
        "một trách nhiệm duy nhất",
        "đơn trách nhiệm",
    ],
    "ocp": [
        "ocp",
        "open closed",
        "open closed principle",
        "open-closed principle",
        "nguyên lý đóng mở",
        "nguyên tắc đóng mở",
        "mở để mở rộng đóng để sửa đổi",
        "đóng mở",
    ],
    "lsp": [
        "lsp",
        "liskov substitution",
        "liskov substitution principle",
        "nguyên lý thay thế liskov",
        "nguyên tắc thay thế liskov",
        "thay thế liskov",
        "lớp con thay thế lớp cha",
    ],
    "isp": [
        "isp",
        "interface segregation",
        "interface segregation principle",
        "nguyên lý phân tách giao diện",
        "nguyên tắc phân tách interface",
        "phân tách giao diện",
        "chia nhỏ interface",
    ],
    "dip": [
        "dip",
        "dependency inversion",
        "dependency inversion principle",
        "nguyên lý đảo ngược phụ thuộc",
        "nguyên tắc đảo ngược phụ thuộc",
        "đảo ngược phụ thuộc",
        "phụ thuộc vào trừu tượng",
        "dependency injection",
        "tiêm phụ thuộc",
        "inversion of control",
        "ioc",
    ],

    # 4. Design Patterns
    "design_patterns": [
        "design patterns",
        "mẫu thiết kế",
        "mẫu thiết kế phần mềm",
        "software design patterns",
        "gof patterns",
        "gang of four",
    ],
    "singleton": [
        "singleton",
        "singleton pattern",
        "mẫu singleton",
        "mẫu đơn thể",
        "thể hiện duy nhất",
        "single instance",
        "private constructor",
        "double-checked locking",
    ],
    "factory": [
        "factory",
        "factory pattern",
        "factory method",
        "abstract factory",
        "mẫu nhà máy",
        "mẫu factory",
        "phương thức nhà máy",
        "nhà máy trừu tượng",
    ],
    "observer": [
        "observer",
        "observer pattern",
        "mẫu quan sát",
        "mẫu observer",
        "publish subscribe",
        "pub sub",
        "người quan sát",
        "event listener",
        "subject observer",
    ],
    "builder": [
        "builder",
        "builder pattern",
        "mẫu builder",
        "mẫu xây dựng",
        "xây dựng đối tượng phức tạp",
    ],
    "adapter": [
        "adapter",
        "adapter pattern",
        "mẫu adapter",
        "mẫu bộ điều hợp",
        "bộ chuyển đổi giao diện",
        "wrapper pattern",
    ],
    "strategy": [
        "strategy",
        "strategy pattern",
        "mẫu chiến lược",
        "mẫu strategy",
        "thuật toán linh hoạt",
    ],
    "decorator": [
        "decorator",
        "decorator pattern",
        "mẫu decorator",
        "mẫu trang trí",
        "bọc đối tượng động",
    ],

    # 5. Software Testing & Quality Assurance
    "software_testing": [
        "software testing",
        "kiểm thử phần mềm",
        "kiểm thử",
        "đảm bảo chất lượng phần mềm",
        "qa",
        "quality assurance",
        "qc",
        "test plan",
        "kế hoạch kiểm thử",
        "test case",
        "trường hợp kiểm thử",
    ],
    "unit_testing": [
        "unit test",
        "unit testing",
        "kiểm thử đơn vị",
        "kiểm thử mức đơn vị",
        "kiểm tra đơn vị",
        "test đơn vị",
        "mock",
        "stub",
        "pytest",
        "junit",
    ],
    "integration_testing": [
        "integration test",
        "integration testing",
        "kiểm thử tích hợp",
        "kiểm thử mức tích hợp",
        "kiểm tra tích hợp",
        "api testing",
    ],
    "system_and_acceptance_testing": [
        "system test",
        "system testing",
        "kiểm thử hệ thống",
        "acceptance test",
        "acceptance testing",
        "kiểm thử nghiệm thu",
        "uat",
        "user acceptance testing",
    ],
    "tdd": [
        "tdd",
        "test-driven development",
        "test driven development",
        "phát triển hướng kiểm thử",
        "phát triển hướng test",
        "red green refactor",
        "red-green-refactor",
        "viết test trước khi viết code",
    ],
    "regression_testing": [
        "regression test",
        "regression testing",
        "kiểm thử hồi quy",
        "kiểm thử lặp lại",
        "hồi quy",
    ],

    # 6. CI/CD & DevOps
    "ci_cd": [
        "ci/cd",
        "ci cd",
        "ci-cd",
        "continuous integration continuous deployment",
        "continuous integration continuous delivery",
        "tích hợp liên tục và triển khai liên tục",
        "quy trình ci cd",
        "ci/cd pipeline",
    ],
    "continuous_integration": [
        "ci",
        "continuous integration",
        "tích hợp liên tục",
        "tự động build",
        "tự động chạy test",
        "automated build",
        "automated testing",
    ],
    "continuous_delivery_deployment": [
        "cd",
        "continuous delivery",
        "continuous deployment",
        "chuyển giao liên tục",
        "triển khai liên tục",
        "tự động phát hành",
        "automated release",
        "deploy staging production",
    ],
    "version_control": [
        "git",
        "vcs",
        "version control",
        "quản lý phiên bản",
        "hệ thống kiểm soát phiên bản",
        "github",
        "gitlab",
        "pull request",
        "merge request",
        "commit",
        "branch",
    ],
    "devops": [
        "devops",
        "văn hóa devops",
        "iac",
        "infrastructure as code",
        "hạ tầng dưới dạng mã",
        "docker",
        "containerization",
        "container",
        "kubernetes",
        "k8s",
    ],

    # 7. Code Quality & Architecture
    "clean_code_and_refactoring": [
        "clean code",
        "mã sạch",
        "viết code sạch",
        "refactoring",
        "tái cấu trúc mã",
        "tái cấu trúc code",
        "code smell",
        "mùi mã nguồn",
        "technical debt",
        "nợ kỹ thuật",
    ],
    "software_architecture": [
        "software architecture",
        "kiến trúc phần mềm",
        "monolithic",
        "kiến trúc nguyên khối",
        "nguyên khối",
        "microservices",
        "kiến trúc vi dịch vụ",
        "vi dịch vụ",
        "mvc",
        "model view controller",
        "clean architecture",
        "hexagonal architecture",
    ],
}


class SEThesaurusRegistry:
    """
    Bidirectional thesaurus registry for Software Engineering terminology.
    Supports case-insensitive, tone-insensitive normalization and query term expansion.
    """

    def __init__(self, clusters: Optional[Dict[str, List[str]]] = None):
        self.clusters = clusters or SE_SYNONYM_CLUSTERS
        self._norm_to_cluster: Dict[str, str] = {}
        self._stripped_to_cluster: Dict[str, str] = {}
        self._bidirectional_map: Dict[str, Set[str]] = {}
        self._compiled_keywords: List[Tuple[str, str]] = []  # (raw_term, cluster_name)
        self._build_indexes()

    def _build_indexes(self) -> None:
        """Constructs forward, reverse, and accent-insensitive indexes."""
        all_terms_with_len: List[Tuple[str, str, int]] = []

        for cluster_name, terms in self.clusters.items():
            for term in terms:
                n_term = normalize_term(term)
                s_term = strip_accents(n_term)

                self._norm_to_cluster[n_term] = cluster_name
                self._stripped_to_cluster[s_term] = cluster_name

                if n_term not in self._bidirectional_map:
                    self._bidirectional_map[n_term] = set()
                self._bidirectional_map[n_term].update(terms)

                all_terms_with_len.append((term, cluster_name, len(n_term)))

        # Sort terms from longest to shortest for greedy n-gram / substring matching
        all_terms_with_len.sort(key=lambda item: item[2], reverse=True)
        self._compiled_keywords = [(item[0], item[1]) for item in all_terms_with_len]

    def get_cluster_name(self, term: str) -> Optional[str]:
        """Finds the cluster identifier for a given term (accent & case-insensitive)."""
        n_term = normalize_term(term)
        if not n_term:
            return None
        if n_term in self._norm_to_cluster:
            return self._norm_to_cluster[n_term]

        s_term = strip_accents(n_term)
        return self._stripped_to_cluster.get(s_term)

    def get_synonyms(self, term: str) -> List[str]:
        """Retrieves all synonymous terms belonging to the same semantic cluster."""
        cluster_name = self.get_cluster_name(term)
        if not cluster_name:
            return []
        return list(self.clusters.get(cluster_name, []))

    def expand_query(self, query: str) -> List[str]:
        """
        Extracts software engineering terms from the query and expands them into synonyms.
        Maintains order, preserves query-specific primary terms, and removes duplicates.
        """
        if not query or not query.strip():
            return []

        norm_q = normalize_term(query)
        stripped_q = strip_accents(norm_q)

        matched_clusters: Set[str] = set()
        matched_original_terms: List[str] = []

        # Greedy match from longest known domain term to shortest
        for term, cluster_name in self._compiled_keywords:
            n_t = normalize_term(term)
            s_t = strip_accents(n_t)

            pattern_raw = r"(?:\b|_|^)" + re.escape(n_t) + r"(?:\b|_|$)"
            pattern_stripped = r"(?:\b|_|^)" + re.escape(s_t) + r"(?:\b|_|$)"

            if re.search(pattern_raw, norm_q, flags=re.IGNORECASE) or re.search(
                pattern_stripped, stripped_q, flags=re.IGNORECASE
            ):
                if cluster_name not in matched_clusters:
                    matched_clusters.add(cluster_name)
                    matched_original_terms.append(term)

        result: List[str] = []
        seen: Set[str] = set()

        def _add_term(t: str) -> None:
            normalized = normalize_term(t)
            if normalized not in seen:
                seen.add(normalized)
                result.append(t)

        for term in matched_original_terms:
            _add_term(term)

        for cluster_name in matched_clusters:
            for syn in self.clusters.get(cluster_name, []):
                _add_term(syn)

        return result

    def identify_se_concepts(self, text: str) -> List[str]:
        """Identifies which SE concept clusters are mentioned in a body of text."""
        if not text or not text.strip():
            return []
        norm_t = normalize_term(text)
        stripped_t = strip_accents(norm_t)

        matched: List[str] = []
        for term, cluster_name in self._compiled_keywords:
            n_t = normalize_term(term)
            s_t = strip_accents(n_t)

            pattern_raw = r"(?:\b|_|^)" + re.escape(n_t) + r"(?:\b|_|$)"
            pattern_stripped = r"(?:\b|_|^)" + re.escape(s_t) + r"(?:\b|_|$)"

            if re.search(pattern_raw, norm_t, flags=re.IGNORECASE) or re.search(
                pattern_stripped, stripped_t, flags=re.IGNORECASE
            ):
                if cluster_name not in matched:
                    matched.append(cluster_name)
        return matched


# Default Singleton Registry
_DEFAULT_SE_REGISTRY = SEThesaurusRegistry()


def get_se_synonyms(term: str) -> List[str]:
    """Public helper function to get bidirectional software engineering synonyms."""
    return _DEFAULT_SE_REGISTRY.get_synonyms(term)


def expand_se_query(query: str) -> List[str]:
    """Expands conversational user queries with authoritative software engineering technical keywords."""
    return _DEFAULT_SE_REGISTRY.expand_query(query)


def find_matching_se_topics(text: str) -> List[str]:
    """Finds list of software engineering domain topics detected in text."""
    return _DEFAULT_SE_REGISTRY.identify_se_concepts(text)


# Authoritative, Pedagogical Few-Shot QA Examples for Software Engineering
SE_FEW_SHOT_EXAMPLES: List[Dict[str, Any]] = [
    {
        "id": "se_qa_01",
        "topic": "Vòng đời phát triển phần mềm (SDLC)",
        "category": "sdlc",
        "canonical_terms": [
            "sdlc",
            "vòng đời phát triển phần mềm",
            "quy trình phát triển phần mềm",
            "tiến trình phần mềm",
        ],
        "question": "Vòng đời phát triển phần mềm (SDLC) là gì và bao gồm các giai đoạn cơ bản nào?",
        "intent": "concept_and_stages",
        "expected_action": "explain_sdlc_and_detail_phases",
        "grounded_answer": (
            "Vòng đời phát triển phần mềm (SDLC - Software Development Life Cycle) là quy trình chuẩn hóa "
            "bao gồm một chuỗi các giai đoạn kế tiếp nhau nhằm định hướng việc lập kế hoạch, xây dựng, kiểm thử "
            "và chuyển giao phần mềm với chất lượng cao nhất và chi phí tối ưu nhất.\n\n"
            "Các giai đoạn cơ bản của SDLC bao gồm:\n"
            "1. Phân tích yêu cầu (Requirements Analysis): Thu thập, làm rõ và tài liệu hóa nhu cầu của các bên liên quan (stakeholders) "
            "thành tài liệu đặc tả yêu cầu phần mềm (SRS - Software Requirements Specification).\n"
            "2. Thiết kế hệ thống (System & Software Design): Chuyển hóa yêu cầu thành kiến trúc tổng thể (High-Level Design - HLD) "
            "và thiết kế chi tiết từng module/hàm/cơ sở dữ liệu (Low-Level Design - LLD).\n"
            "3. Triển khai & Lập trình (Implementation / Coding): Các kỹ sư phần mềm hiện thực hóa thiết kế thành mã nguồn có thể thực thi.\n"
            "4. Kiểm thử phần mềm (Testing & Quality Assurance): Đánh giá tính đúng đắn, phát hiện khiếm khuyết (bugs/defects) "
            "thông qua Unit Test, Integration Test, System Test và UAT.\n"
            "5. Phát hành & Triển khai (Deployment): Đưa sản phẩm lên môi trường thực tế (staging/production) cho người dùng cuối.\n"
            "6. Vận hành & Bảo trì (Maintenance & Operations): Khắc phục lỗi phát sinh, cập nhật phiên bản, tối ưu hiệu năng và thích ứng với môi trường mới."
        ),
        "bullet_types": [
            "- 1. Phân tích yêu cầu (SRS, User Stories).",
            "- 2. Thiết kế kiến trúc & chi tiết (HLD, LLD, Database Schema).",
            "- 3. Triển khai lập trình (Coding & Source Code).",
            "- 4. Kiểm thử toàn diện (Unit, Integration, System, UAT).",
            "- 5. Triển khai sản phẩm (Staging, Production Deployment).",
            "- 6. Bảo trì & vận hành (Bug fixing, Patching, Evolution).",
        ],
        "example": "Quy trình xây dựng hệ thống thương mại điện tử từ thu thập yêu cầu giỏ hàng đến triển khai máy chủ đám mây.",
    },
    {
        "id": "se_qa_02",
        "topic": "Mô hình Thác nước (Waterfall Model)",
        "category": "sdlc",
        "canonical_terms": [
            "waterfall",
            "mô hình thác nước",
            "tuần tự tuyến tính",
            "waterfall model",
        ],
        "question": "Mô hình Thác nước (Waterfall) là gì? Nêu ưu điểm, nhược điểm và trường hợp áp dụng phù hợp?",
        "intent": "model_overview_pros_cons",
        "expected_action": "explain_waterfall_and_pros_cons",
        "grounded_answer": (
            "Mô hình Thác nước (Waterfall Model) do Winston W. Royce đề xuất là mô hình phát triển phần mềm tuần tự tuyến tính "
            "(linear sequential), trong đó mỗi giai đoạn phát triển (Yêu cầu -> Thiết kế -> Lập trình -> Kiểm thử -> Triển khai -> Bảo trì) "
            "chỉ bắt đầu khi và chỉ khi giai đoạn trước đó đã hoàn tất 100% và được phê duyệt chặt chẽ.\n\n"
            "Ưu điểm:\n"
            "- Cấu trúc rõ ràng, kỷ luật cao, dễ quản lý tiến độ và phân công công việc.\n"
            "- Tài liệu kỹ thuật chi tiết và đầy đủ qua từng giai đoạn.\n"
            "- Phù hợp với các dự án có yêu cầu cố định, ít biến động.\n\n"
            "Nhược điểm:\n"
            "- Tính linh hoạt kém: Rất khó khăn và tốn kém khi thay đổi yêu cầu ở các giai đoạn sau.\n"
            "- Sản phẩm thực thi chỉ xuất hiện ở cuối vòng đời, tiềm ẩn rủi ro phát hiện lỗi kiến trúc muộn.\n"
            "- Khách hàng chỉ có thể đánh giá sản phẩm khi đã gần hoàn tất, dễ dẫn đến bất đồng mong đợi.\n\n"
            "Trường hợp áp dụng:\n"
            "- Dự án có yêu cầu rõ ràng, ổn định, công nghệ đã quen thuộc và rủi ro thay đổi thấp (ví dụ: phần mềm nhúng hàng không, y tế, hệ thống ngân hàng truyền thống)."
        ),
        "bullet_types": [
            "- Bản chất: Tiếp cận tuần tự tuyến tính nghiêm ngặt.",
            "- Ưu điểm: Dễ quản lý, tài liệu hóa kỹ càng, quy trình minh bạch.",
            "- Nhược điểm: Kém thích nghi với thay đổi, rủi ro tích tụ ở cuối chu kỳ.",
            "- Ứng dụng: Dự án miền kỹ thuật cố định, tiêu chuẩn an toàn bắt buộc.",
        ],
        "example": "Dự án chế tạo phần mềm điều khiển thiết bị phần cứng quân sự với đặc tả phần cứng không đổi.",
    },
    {
        "id": "se_qa_03",
        "topic": "Phương pháp Agile và Khung làm việc Scrum",
        "category": "agile_scrum",
        "canonical_terms": [
            "agile",
            "scrum",
            "sprint",
            "scrum master",
            "product owner",
            "product backlog",
        ],
        "question": "Agile là gì và khung làm việc Scrum vận hành như thế nào?",
        "intent": "methodology_and_framework",
        "expected_action": "explain_agile_and_scrum_details",
        "grounded_answer": (
            "Agile là triết lý phát triển phần mềm linh hoạt dựa trên phương pháp lặp (iterative) và tăng dần (incremental), "
            "ưu tiên con người và sự tương tác, phần mềm chạy tốt, sự cộng tác với khách hàng, và khả năng thích ứng linh hoạt trước thay đổi.\n\n"
            "Scrum là khung làm việc (framework) phổ biến nhất hiện thực hóa triết lý Agile:\n"
            "1. Ba vai trò cốt lõi (Scrum Roles):\n"
            "- Product Owner (PO): Đại diện cho khách hàng, chịu trách nhiệm định nghĩa và tối ưu hóa giá trị sản phẩm thông qua Product Backlog.\n"
            "- Scrum Master (SM): Đảm bảo đội ngũ tuân thủ các nguyên lý Scrum, loại bỏ các trở ngại (impediments) cản trở tiến độ của đội.\n"
            "- Development Team: Đội ngũ phát triển liên chức năng (cross-functional), tự tổ chức (self-organizing) trực tiếp xây dựng sản phẩm.\n\n"
            "2. Năm sự kiện trong Scrum (Scrum Events):\n"
            "- Sprint: Chu kỳ phát triển cố định (thường từ 1 đến 4 tuần) nhằm tạo ra phần tăng trưởng sản phẩm (Increment) dùng được ngay.\n"
            "- Sprint Planning: Họp lập kế hoạch xác định mục tiêu Sprint và các hạng mục công việc từ Backlog được chọn vào Sprint Backlog.\n"
            "- Daily Scrum: Họp đứng hằng ngày (15 phút) đồng bộ tiến độ: hôm qua làm gì, hôm nay làm gì, có khó khăn gì không.\n"
            "- Sprint Review: Đánh giá và demo sản phẩm tăng trưởng cùng các bên liên quan ở cuối Sprint.\n"
            "- Sprint Retrospective: Họp cải tiến nội bộ đội ngũ nhìn nhận điểm làm tốt và điểm cần cải tiến cho Sprint tiếp theo.\n\n"
            "3. Ba đồ tác phẩm (Scrum Artifacts):\n"
            "- Product Backlog: Danh sách ưu tiên các tính năng, yêu cầu và sửa lỗi của toàn bộ sản phẩm.\n"
            "- Sprint Backlog: Tập hợp các hạng mục được chọn cho Sprint hiện tại cùng kế hoạch hiện thực hóa.\n"
            "- Increment: Sản phẩm phần mềm có thể chuyển giao (shippable product) đáp ứng Định nghĩa Hoàn thành (Definition of Done - DoD)."
        ),
        "bullet_types": [
            "- Triết lý: Lặp, tăng dần, phản hồi nhanh, giá trị chuyển giao sớm.",
            "- 3 Vai trò: Product Owner, Scrum Master, Development Team.",
            "- 5 Sự kiện: Sprint, Planning, Daily Scrum, Review, Retrospective.",
            "- 3 Đồ tác phẩm: Product Backlog, Sprint Backlog, Potentially Shippable Increment.",
        ],
        "example": "Một đội ngũ phát triển ứng dụng di động phân chia chu kỳ 2 tuần một Sprint để liên tục đưa tính năng mới đến người dùng.",
    },
    {
        "id": "se_qa_04",
        "topic": "So sánh Waterfall và Agile",
        "category": "comparison",
        "canonical_terms": [
            "waterfall",
            "agile",
            "so sánh waterfall và agile",
            "mô hình thác nước",
            "phương pháp linh hoạt",
        ],
        "question": "So sánh sự khác biệt cơ bản giữa mô hình Thác nước (Waterfall) và phương pháp Agile?",
        "intent": "comparative_analysis",
        "expected_action": "compare_waterfall_vs_agile",
        "grounded_answer": (
            "So sánh chi tiết giữa Mô hình Thác nước (Waterfall) và Phương pháp Linh hoạt (Agile):\n\n"
            "1. Cách tiếp cận & Chu kỳ:\n"
            "- Waterfall: Tuyến tính tuần tự. Toàn bộ phạm vi dự án được lên kế hoạch và thực hiện một lần duy nhất.\n"
            "- Agile: Lặp và tăng dần (Iterative & Incremental). Dự án được chia thành nhiều chu kỳ ngắn (1-4 tuần).\n\n"
            "2. Xử lý sự thay đổi yêu cầu:\n"
            "- Waterfall: Kháng cự thay đổi. Yêu cầu phải được đóng băng sớm; thay đổi cần quy trình Change Request phức tạp.\n"
            "- Agile: Chào đón thay đổi. Yêu cầu có thể được cập nhật, tái ưu tiên liên tục theo phản hồi thực tế của người dùng.\n\n"
            "3. Chuyển giao sản phẩm:\n"
            "- Waterfall: Sản phẩm chỉ được chuyển giao và thấy được ở cuối toàn bộ dự án.\n"
            "- Agile: Mỗi Sprint đều tạo ra một phần tăng trưởng (Increment) có thể sử dụng và chạy thử nghiệm được.\n\n"
            "4. Kiểm thử phần mềm:\n"
            "- Waterfall: Kiểm thử là giai đoạn riêng biệt diễn ra sau khi lập trình toàn bộ xong.\n"
            "- Agile: Kiểm thử diễn ra liên tục, song hành cùng quá trình phát triển trong từng ngày của Sprint.\n\n"
            "5. Sự tham gia của khách hàng:\n"
            "- Waterfall: Chủ yếu ở đầu giai đoạn yêu cầu và cuối giai đoạn nghiệm thu.\n"
            "- Agile: Khách hàng (thông qua Product Owner) tham gia liên tục vào quá trình phản hồi và lập kế hoạch."
        ),
        "bullet_types": [
            "- Waterfall: Kế hoạch cố định, tiếp cận tuần tự, bàn giao muộn một lần, kiểm thử sau cùng.",
            "- Agile: Kế hoạch linh hoạt, tiếp cận lặp tăng dần, bàn giao liên tục từng Sprint, kiểm thử liên tục.",
        ],
        "example": "Waterfall thích hợp làm hợp đồng thầu trọn gói nhà nước; Agile thích hợp làm sản phẩm Startup công nghệ biến động nhanh.",
    },
    {
        "id": "se_qa_05",
        "topic": "Năm nguyên lý SOLID trong Thiết kế hướng đối tượng",
        "category": "solid_principles",
        "canonical_terms": [
            "solid",
            "nguyên lý solid",
            "srp",
            "ocp",
            "lsp",
            "isp",
            "dip",
        ],
        "question": "Trình bày 5 nguyên lý SOLID trong thiết kế phần mềm hướng đối tượng?",
        "intent": "principles_overview_and_breakdown",
        "expected_action": "detail_each_solid_principle",
        "grounded_answer": (
            "SOLID là 5 nguyên lý thiết kế hướng đối tượng do Robert C. Martin (Uncle Bob) khởi xướng, "
            "giúp mã nguồn dễ bảo trì, dễ mở rộng và giảm thiểu rủi ro kỹ thuật:\n\n"
            "1. S - Single Responsibility Principle (SRP - Nguyên lý đơn trách nhiệm):\n"
            "- Mỗi lớp chỉ nên có một lý do duy nhất để thay đổi, nghĩa là chỉ đảm nhận một trách nhiệm cụ thể.\n"
            "- Tránh tạo ra các lớp thần thánh (God Object) vừa xử lý logic nghiệp vụ, vừa truy vấn CSDL, vừa định dạng UI.\n\n"
            "2. O - Open/Closed Principle (OCP - Nguyên lý đóng/mở):\n"
            "- Các thực thể phần mềm (class, module, function) nên mở cho việc mở rộng (open for extension) nhưng đóng cho việc sửa đổi (closed for modification).\n"
            "- Đạt được thông qua kế thừa, đa hình và các giao diện trừu tượng (Interface/Polymorphism) thay vì sửa trực tiếp code cũ.\n\n"
            "3. L - Liskov Substitution Principle (LSP - Nguyên lý thay thế Liskov):\n"
            "- Các đối tượng của lớp con phải có khả năng thay thế hoàn toàn cho đối tượng của lớp cha mà không làm thay đổi tính đúng đắn của chương trình.\n"
            "- Lớp con không được vi phạm các tiền điều kiện (preconditions) hay hậu điều kiện (postconditions) của lớp cha (ví dụ kinh điển: Square không nên kế thừa Rectangle nếu làm biến đổi logic setWidth/setHeight).\n\n"
            "4. I - Interface Segregation Principle (ISP - Nguyên lý phân tách giao diện):\n"
            "- Khách hàng không nên bị ép buộc phải phụ thuộc vào các giao diện mà họ không sử dụng.\n"
            "- Nên chia nhỏ các interface đồ sộ thành nhiều interface tinh gọn, tập trung vào từng hành vi cụ thể.\n\n"
            "5. D - Dependency Inversion Principle (DIP - Nguyên lý đảo ngược phụ thuộc):\n"
            "- Các module cấp cao không nên phụ thuộc trực tiếp vào các module cấp thấp; cả hai nên phụ thuộc vào các trừu tượng (abstraction).\n"
            "- Các trừu tượng không phụ thuộc vào chi tiết; chi tiết phải phụ thuộc vào trừu tượng. Thường hiện thực bằng Dependency Injection (DI)."
        ),
        "bullet_types": [
            "- S (SRP): 1 lý do duy nhất để thay đổi.",
            "- O (OCP): Mở để mở rộng, đóng để sửa đổi.",
            "- L (LSP): Lớp con thay thế hoàn hảo cho lớp cha.",
            "- I (ISP): Giao diện nhỏ gọn, không ép dùng hàm thừa.",
            "- D (DIP): Phụ thuộc vào trừu tượng (Interface/Abstract), không phụ thuộc cụ thể.",
        ],
        "example": "Hệ thống thanh toán dùng interface `PaymentProcessor` thay vì gọi trực tiếp `PaypalService` cụ thể trong lớp `Order`.",
    },
    {
        "id": "se_qa_06",
        "topic": "Mẫu thiết kế Singleton (Singleton Design Pattern)",
        "category": "design_patterns",
        "canonical_terms": [
            "singleton",
            "singleton pattern",
            "mẫu đơn thể",
            "thể hiện duy nhất",
        ],
        "question": "Mẫu thiết kế Singleton là gì? Cơ chế hoạt động, ưu nhược điểm và cách hiện thực hóa an toàn luồng?",
        "intent": "pattern_concept_and_implementation",
        "expected_action": "explain_singleton_and_thread_safety",
        "grounded_answer": (
            "Singleton là mẫu thiết kế khởi tạo (Creational Pattern) đảm bảo rằng một lớp chỉ có duy nhất một thể hiện (instance) "
            "trong suốt vòng đời của ứng dụng và cung cấp một điểm truy cập toàn cục (global access point) tới thể hiện đó.\n\n"
            "Cơ chế cốt lõi:\n"
            "- Hàm khởi tạo (Constructor) ở chế độ riêng tư (private) để ngăn chặn việc tạo đối tượng tùy ý từ bên ngoài bằng toán tử new.\n"
            "- Một biến tĩnh riêng tư (private static instance) lưu giữ thể hiện duy nhất.\n"
            "- Một phương thức công khai tĩnh (public static getInstance()) trả về thể hiện duy nhất, tự khởi tạo khi được gọi lần đầu (lazy initialization).\n\n"
            "Cơ chế an toàn đa luồng (Thread-Safety):\n"
            "Trong môi trường đa luồng, hai luồng có thể cùng lúc kiểm tra instance == null và tạo ra 2 đối tượng khác nhau. "
            "Giải pháp là sử dụng Double-Checked Locking kết hợp từ khóa volatile (Java/C++) hoặc sử dụng cơ chế nạp module tích hợp sẵn (Python) "
            "hoặc synchronized block.\n\n"
            "Ưu điểm:\n"
            "- Kiểm soát truy cập chặt chẽ tới tài nguyên chia sẻ duy nhất (như kết nối cơ sở dữ liệu, bộ quản lý tệp cấu hình, logger).\n"
            "- Tiết kiệm bộ nhớ vì không tạo ra các bản sao thừa thãi.\n\n"
            "Nhược điểm:\n"
            "- Gây khó khăn cho việc viết Unit Test do che giấu sự phụ thuộc và duy trì trạng thái toàn cục (global state).\n"
            "- Vi phạm nguyên lý SRP (lớp vừa xử lý nghiệp vụ vừa quản lý vòng đời khởi tạo chính nó)."
        ),
        "bullet_types": [
            "- Bản chất: Đảm bảo duy nhất 1 thể hiện, điểm truy cập toàn cục.",
            "- Cấu trúc: Private constructor, private static instance, public static getInstance().",
            "- An toàn đa luồng: Double-Checked Locking, thread synchronization.",
            "- Thận trọng: Tránh biến Singleton thành Anti-pattern do phụ thuộc ẩn.",
        ],
        "example": "Lớp `DatabaseConnectionPool` chia sẻ chung trong toàn bộ hệ thống web backend.",
    },
    {
        "id": "se_qa_07",
        "topic": "Mẫu thiết kế Factory (Factory Method & Abstract Factory)",
        "category": "design_patterns",
        "canonical_terms": [
            "factory",
            "factory pattern",
            "factory method",
            "abstract factory",
            "mẫu nhà máy",
        ],
        "question": "Phân biệt Factory Method và Abstract Factory trong Design Patterns?",
        "intent": "pattern_comparison_and_usage",
        "expected_action": "compare_factory_method_and_abstract_factory",
        "grounded_answer": (
            "Cả Factory Method và Abstract Factory đều là các mẫu thiết kế khởi tạo (Creational Patterns) giúp tách rời logic "
            "tạo đối tượng ra khỏi mã nguồn sử dụng đối tượng, tuân thủ nguyên lý Đóng/Mở (OCP).\n\n"
            "1. Factory Method (Phương thức nhà máy):\n"
            "- Định nghĩa một giao diện (hoặc lớp trừu tượng) để tạo một đối tượng, nhưng để các lớp con quyết định lớp cụ thể nào sẽ được khởi tạo.\n"
            "- Sử dụng cơ chế kế thừa (Inheritance).\n"
            "- Tập trung vào việc tạo ra MỘT sản phẩm (single product family).\n"
            "- Ví dụ: Lớp `Dialog` có phương thức trừu tượng `createButton()`, lớp con `WindowsDialog` tạo `WindowsButton`, lớp `WebDialog` tạo `HTMLButton`.\n\n"
            "2. Abstract Factory (Nhà máy trừu tượng):\n"
            "- Cung cấp một giao diện để tạo ra một HỌ các đối tượng có liên quan hoặc phụ thuộc lẫn nhau mà không chỉ định các lớp cụ thể của chúng.\n"
            "- Sử dụng cơ chế ủy thác/kết hợp (Composition/Delegation).\n"
            "- Tập trung vào việc tạo ra MỘT BỘ SẢN PHẨM đồng bộ (product families).\n"
            "- Ví dụ: `GUIFactory` tạo ra cả một bộ giao diện gồm `createButton()`, `createCheckbox()`, `createScrollBar()`. Lớp `MacFactory` tạo đồng bộ các widget phong cách MacOS, còn `WinFactory` tạo widget phong cách Windows."
        ),
        "bullet_types": [
            "- Factory Method: Dựa trên kế thừa, tạo một sản phẩm đơn lẻ thông qua lớp con.",
            "- Abstract Factory: Dựa trên đối tượng ủy thác, tạo ra cả một họ sản phẩm tương thích.",
            "- Lợi ích chung: Giảm thiểu ghép nối chặt (loose coupling), dễ mở rộng loại đối tượng mới.",
        ],
        "example": "Factory Method tạo loại thông báo (Email, SMS, Push); Abstract Factory tạo bộ giao diện theo từng hệ điều hành.",
    },
    {
        "id": "se_qa_08",
        "topic": "Mẫu thiết kế Observer (Observer Design Pattern)",
        "category": "design_patterns",
        "canonical_terms": [
            "observer",
            "observer pattern",
            "mẫu quan sát",
            "publish subscribe",
            "pub sub",
            "event listener",
        ],
        "question": "Mẫu Observer hoạt động theo nguyên lý nào và được ứng dụng ở đâu?",
        "intent": "pattern_concept_and_application",
        "expected_action": "explain_observer_and_real_world_use",
        "grounded_answer": (
            "Observer (Mẫu quan sát) là mẫu thiết kế hành vi (Behavioral Pattern) định nghĩa mối phụ thuộc một-nhiều (one-to-many) "
            "giữa các đối tượng, sao cho khi một đối tượng (Subject / Publisher) thay đổi trạng thái, tất cả các đối tượng phụ thuộc "
            "(Observers / Subscribers) vào nó sẽ được tự động thông báo và cập nhật kịp thời.\n\n"
            "Cấu trúc thành phần:\n"
            "1. Subject (Chủ thể): Quản lý danh sách các Observer thông qua các hàm `attach()`, `detach()`, và phương thức `notify()` "
            "lặp qua danh sách để gọi hàm `update()` của từng Observer.\n"
            "2. Observer (Đối tượng quan sát): Khai báo interface với phương thức chuẩn `update(data)`.\n"
            "3. ConcreteSubject & ConcreteObserver: Hiện thực hóa cụ thể dữ liệu và hành động phản ứng.\n\n"
            "Ứng dụng thực tế:\n"
            "- Hệ thống kiến trúc hướng sự kiện (Event-Driven Architecture) và mô hình MVC (Model cập nhật dữ liệu, tự động báo các View vẽ lại).\n"
            "- Cơ chế lắng nghe sự kiện (Event Listeners) trong UI như click chuột, gõ phím.\n"
            "- Hệ thống Message Broker / Publish-Subscribe (Kafka, RabbitMQ, Redis Pub/Sub).\n"
            "- Reactive Programming (RxJS, Combine, State Management như Redux, Zustand)."
        ),
        "bullet_types": [
            "- Bản chất: Mối quan hệ 1-N lỏng lẻo (loose coupling) giữa Subject và Observers.",
            "- Cơ chế: Subject duy trì danh sách đăng ký và tự động broadcast khi có thay đổi.",
            "- Ứng dụng: Giao diện người dùng (UI Event Handlers), MVC pattern, Kiến trúc Pub/Sub.",
        ],
        "example": "Người dùng đăng ký nhận bản tin sản phẩm; khi sản phẩm hạ giá, hệ thống gửi thông báo cho tất cả người đã đăng ký.",
    },
    {
        "id": "se_qa_09",
        "topic": "Kiểm thử phần mềm: Unit Test vs Integration Test",
        "category": "software_testing",
        "canonical_terms": [
            "unit test",
            "integration test",
            "kiểm thử đơn vị",
            "kiểm thử tích hợp",
            "kim tự tháp kiểm thử",
        ],
        "question": "Phân biệt Unit Test và Integration Test? Vai trò của Kim tự tháp kiểm thử (Test Pyramid)?",
        "intent": "testing_levels_and_pyramid",
        "expected_action": "compare_unit_vs_integration_test",
        "grounded_answer": (
            "Unit Test (Kiểm thử đơn vị) và Integration Test (Kiểm thử tích hợp) là hai cấp độ kiểm thử cốt lõi "
            "trong chiến lược đảm bảo chất lượng phần mềm:\n\n"
            "1. Unit Test (Kiểm thử đơn vị):\n"
            "- Phạm vi: Kiểm tra thành phần nhỏ nhất của mã nguồn có thể cô lập được (thường là một hàm, phương thức hoặc lớp riêng biệt).\n"
            "- Môi trường: Hoàn toàn độc lập; các phụ thuộc bên ngoài (CSDL, mạng, tệp tin, API bên thứ ba) đều được giả lập bằng Mock / Stub / Fake.\n"
            "- Tốc độ: Cực nhanh (mỗi test tính bằng mili-giây).\n"
            "- Mục đích: Chứng minh logic thuật toán xử lý chính xác theo từng nhánh điều kiện.\n\n"
            "2. Integration Test (Kiểm thử tích hợp):\n"
            "- Phạm vi: Kiểm tra sự tương tác, phối hợp và luồng truyền dữ liệu giữa hai hoặc nhiều module, thành phần với nhau.\n"
            "- Môi trường: Thường kết nối tới các dịch vụ thực hoặc môi trường giả lập cao cấp (như cơ sở dữ liệu test, container Docker).\n"
            "- Tốc độ: Chậm hơn đáng kể so với Unit Test.\n"
            "- Mục đích: Phát hiện các lỗi không tương thích giao diện, sai lệch định dạng dữ liệu hoặc lỗi giao tiếp mạng/CSDL.\n\n"
            "3. Kim tự tháp kiểm thử (Test Pyramid - Mike Cohn):\n"
            "- Đáy tháp (Phần lớn nhất): Unit Tests (số lượng nhiều nhất, thực thi nhanh nhất, chi phí bảo trì thấp nhất).\n"
            "- Thân tháp: Integration Tests / Service Tests (số lượng vừa phải, kiểm tra luồng tích hợp).\n"
            "- Đỉnh tháp (Phần nhỏ nhất): End-to-End (E2E) / UI Tests (số lượng ít nhất, chạy chậm nhất, dễ vỡ - flaky nhất nhưng bao quát trải nghiệm người dùng)."
        ),
        "bullet_types": [
            "- Unit Test: Kiểm tra hàm/lớp đơn lẻ, cô lập hoàn toàn bằng Mock, chạy tính bằng mili-giây.",
            "- Integration Test: Kiểm tra sự kết nối giữa các module/CSDL/API, xác minh tính tương thích.",
            "- Test Pyramid: Đáy tháp Unit Test chiếm 70%, giữa tháp Integration 20%, đỉnh tháp E2E 10%.",
        ],
        "example": "Unit Test kiểm tra hàm tính thuế `calc_tax(100) == 10`; Integration Test kiểm tra `create_order()` lưu đơn hàng thành công vào CSDL Postgres.",
    },
    {
        "id": "se_qa_10",
        "topic": "Phát triển hướng kiểm thử (TDD - Test-Driven Development)",
        "category": "software_testing",
        "canonical_terms": [
            "tdd",
            "test-driven development",
            "red green refactor",
            "phát triển hướng kiểm thử",
        ],
        "question": "TDD (Test-Driven Development) là gì? Trình bày chu trình Red-Green-Refactor và lợi ích cốt lõi?",
        "intent": "tdd_cycle_and_benefits",
        "expected_action": "explain_tdd_and_red_green_refactor",
        "grounded_answer": (
            "TDD (Test-Driven Development - Phát triển hướng kiểm thử) là một kỹ thuật công nghệ phần mềm trong đó "
            "các trường hợp kiểm thử (test cases) được viết trước khi viết mã nguồn chức năng thực tế.\n\n"
            "Chu trình Red - Green - Refactor:\n"
            "1. Red (Đỏ): Viết một kiểm thử đơn vị mô tả hành vi mong muốn của tính năng mới. Chạy kiểm thử và xác nhận nó thất bại (Red), "
            "bởi vì mã chức năng chưa hề tồn tại hoặc chưa thỏa mãn yêu cầu.\n"
            "2. Green (Xanh): Viết một lượng mã nguồn tối thiểu (minimal code) chỉ vừa đủ để làm cho kiểm thử vượt qua thành công (Green). "
            "Ở bước này, giải pháp có thể đơn giản, thậm chí chưa tối ưu.\n"
            "3. Refactor (Tái cấu trúc): Cải tiến và tối ưu hóa thiết kế mã nguồn, loại bỏ trùng lặp (DRY), cải thiện tính dễ đọc và kiến trúc "
            "trong khi liên tục chạy lại test suite để đảm bảo mã vẫn luôn xanh (không làm gãy tính năng cũ).\n\n"
            "Lợi ích cốt lõi:\n"
            "- Đảm bảo độ bao phủ mã nguồn (code coverage) rất cao ngay từ đầu.\n"
            "- Giúp lập trình viên tư duy về giao diện và cách dùng của hàm/lớp trước khi lao vào cài đặt chi tiết.\n"
            "- Giảm đáng kể số lượng lỗi phát sinh khi bàn giao và tạo mạng lưới an toàn vững chắc cho việc bảo trì, tái cấu trúc sau này."
        ),
        "bullet_types": [
            "- 1. Red: Viết test trước -> Xác nhận test fail.",
            "- 2. Green: Viết mã tối thiểu -> Xác nhận test pass.",
            "- 3. Refactor: Làm sạch mã, tối ưu thiết kế mà không làm vỡ test.",
            "- Lợi ích: Thiết kế module lỏng lẻo, giảm bug, an tâm tái cấu trúc.",
        ],
        "example": "Viết test `assert divide(10, 2) == 5` trước; sau đó mới khai báo hàm `def divide(a, b): return a / b`.",
    },
    {
        "id": "se_qa_11",
        "topic": "CI/CD (Continuous Integration & Continuous Delivery/Deployment)",
        "category": "ci_cd_devops",
        "canonical_terms": [
            "ci/cd",
            "ci cd",
            "continuous integration",
            "continuous delivery",
            "continuous deployment",
            "tích hợp liên tục",
        ],
        "question": "CI/CD là gì? Phân biệt Continuous Integration, Continuous Delivery và Continuous Deployment?",
        "intent": "ci_cd_concept_and_distinction",
        "expected_action": "explain_and_distinguish_ci_cd",
        "grounded_answer": (
            "CI/CD là bộ thực hành tự động hóa then chốt trong kỹ nghệ phần mềm hiện đại và văn hóa DevOps, "
            "nhằm rút ngắn chu kỳ phát hành phần mềm và nâng cao chất lượng mã nguồn:\n\n"
            "1. Continuous Integration (CI - Tích hợp liên tục):\n"
            "- Các nhà phát triển thường xuyên đẩy (commit/merge) mã nguồn mới vào kho mã chung nhiều lần trong ngày.\n"
            "- Mỗi lần đẩy mã sẽ kích hoạt một hệ thống tự động build mã nguồn và chạy toàn bộ bộ kiểm thử tự động (Unit test, Linter, Security scan).\n"
            "- Mục tiêu: Phát hiện sớm lỗi xung đột mã nguồn và lỗi logic ngay khi mã vừa được viết ra.\n\n"
            "2. Continuous Delivery (CD - Chuyển giao liên tục):\n"
            "- Mở rộng từ CI: Sau khi các bài kiểm tra tự động vượt qua, mã nguồn được tự động đóng gói và sẵn sàng phát hành "
            "lên các môi trường thử nghiệm (Staging/Pre-production).\n"
            "- Tuy nhiên, bước triển khai thực tế lên môi trường Production cho khách hàng cuối vẫn cần một quyết định phê duyệt thủ công (Manual approval) "
            "từ nhà quản lý hoặc Product Owner.\n\n"
            "3. Continuous Deployment (CD - Triển khai liên tục):\n"
            "- Mức độ tự động hóa tối đa: Mọi thay đổi vượt qua thành công toàn bộ các cổng kiểm định CI/CD "
            "sẽ tự động được triển khai trực tiếp lên môi trường Production mà hoàn toàn không cần sự can thiệp thủ công của con người.\n\n"
            "Lợi ích tổng thể:\n"
            "- Giảm thiểu 'nỗi sợ phát hành' (deployment anxiety).\n"
            "- Rút ngắn thời gian đưa tính năng ra thị trường (Time-to-Market).\n"
            "- Khắc phục lỗi nhanh chóng thông qua các chu kỳ phát hành siêu nhỏ (micro-releases)."
        ),
        "bullet_types": [
            "- CI: Tự động hóa build và test ngay khi lập trình viên commit mã.",
            "- Continuous Delivery: Tự động phát hành sẵn sàng đến môi trường Staging; cần duyệt tay để lên Production.",
            "- Continuous Deployment: Tự động hóa 100% thẳng lên Production không cần duyệt tay.",
        ],
        "example": "Quy trình GitHub Actions tự động chạy `pytest` khi mở Pull Request và kích hoạt Terraform deploy lên AWS khi merge vào nhánh `main`.",
    },
    {
        "id": "se_qa_12",
        "topic": "Kiến trúc Phần mềm: Monolithic so với Microservices",
        "category": "software_architecture",
        "canonical_terms": [
            "monolithic",
            "microservices",
            "kiến trúc nguyên khối",
            "kiến trúc vi dịch vụ",
            "kiến trúc phần mềm",
        ],
        "question": "So sánh kiến trúc nguyên khối (Monolith) và kiến trúc vi dịch vụ (Microservices)?",
        "intent": "architecture_comparison",
        "expected_action": "compare_monolith_vs_microservices",
        "grounded_answer": (
            "So sánh hai phong cách kiến trúc phần mềm phổ biến trong hệ thống doanh nghiệp:\n\n"
            "1. Kiến trúc nguyên khối (Monolithic Architecture):\n"
            "- Khái niệm: Toàn bộ thành phần chức năng (UI, Business Logic, Data Access, Database) được đóng gói và triển khai thành một khối duy nhất (single deployable unit).\n"
            "- Ưu điểm: Đơn giản trong giai đoạn đầu, dễ phát triển, dễ kiểm thử đầu-cuối và dễ triển khai; không có độ trễ mạng giữa các module nội bộ.\n"
            "- Nhược điểm: Khó mở rộng linh hoạt từng tính năng; một module bị lỗi bộ nhớ có thể kéo sập toàn bộ ứng dụng; rào cản công nghệ (toàn bộ codebase bị bó cứng vào một ngôn ngữ/framework duy nhất).\n\n"
            "2. Kiến trúc vi dịch vụ (Microservices Architecture):\n"
            "- Khái niệm: Hệ thống được phân tách thành nhiều dịch vụ độc lập, quy mô nhỏ, tập trung vào một miền nghiệp vụ duy nhất (Bounded Context), giao tiếp với nhau qua API nhẹ (REST, gRPC, Message Queue).\n"
            "- Ưu điểm: Khả năng mở rộng độc lập từng dịch vụ (scale theo nhu cầu tải); khả năng cô lập lỗi cao (fault isolation); linh hoạt sử dụng các công nghệ, cơ sở dữ liệu khác nhau cho từng dịch vụ.\n"
            "- Nhược điểm: Độ phức tạp vận hành rất cao (phân tán dữ liệu, eventual consistency, phân tích log phân tán qua Jaeger/OpenTelemetry, chi phí quản trị hạ tầng DevOps và mạng)."
        ),
        "bullet_types": [
            "- Monolith: Triển khai 1 khối duy nhất, dễ làm lúc đầu, khó mở rộng độc lập khi hệ thống lớn.",
            "- Microservices: Phân tán nhiều dịch vụ độc lập, mở rộng linh hoạt, đòi hỏi năng lực DevOps cao.",
        ],
        "example": "Amazon và Netflix chuyển đổi từ hệ thống Monolith khổng lồ sang hàng ngàn Microservices độc lập để phục vụ hàng triệu truy cập đồng thời.",
    },
]
