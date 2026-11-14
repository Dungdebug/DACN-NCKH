"""
Domain-Specific Database & SQL Semantic Patterns.
Provides bidirectional synonym mapping, domain query expansion, and few-shot QA blueprints
for authoritative Database Systems (CSDL) educational and technical documents.
Covers: ACID, Primary/Foreign Keys, Normalization (1NF/2NF/3NF/BCNF), B-Tree/B+Tree Indexing,
Transactions & Concurrency Control, SQL JOINs.
100% Offline, Pure Python stdlib, Zero External Dependencies.
"""

import re
import unicodedata
from typing import Dict, List, Set, Optional, Tuple, Any


def strip_accents(text: str) -> str:
    """Removes Vietnamese tone marks for accent-insensitive canonical alignment."""
    if not text:
        return ""
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D")


def normalize_term(term: str) -> str:
    """Standardizes casing and whitespace for consistent lookup."""
    return " ".join(term.strip().lower().split())


# Core Relational Database & SQL Domain Synonym Clusters
DATABASE_SYNONYM_CLUSTERS: Dict[str, List[str]] = {
    "acid_properties": [
        "acid",
        "thuộc tính acid",
        "tính chất acid",
        "tính acid",
        "acid properties",
        "tính nguyên tử, nhất quán, cô lập, bền vững",
        "nguyên tắc acid",
        "acid transaction",
    ],
    "atomicity": [
        "tính nguyên tử",
        "nguyên tử",
        "atomicity",
        "tất cả hoặc không",
        "tất cả hoặc không gì cả",
        "all-or-nothing",
        "tính toàn vẹn nguyên tử",
    ],
    "consistency": [
        "tính nhất quán",
        "nhất quán",
        "consistency",
        "bảo toàn tính nhất quán",
        "ràng buộc toàn vẹn",
        "trạng thái hợp lệ",
    ],
    "isolation": [
        "tính cô lập",
        "cô lập",
        "isolation",
        "cô lập giao dịch",
        "mức cô lập",
        "cấp độ cô lập",
        "isolation level",
        "độ cô lập giao dịch",
    ],
    "durability": [
        "tính bền vững",
        "bền vững",
        "tính bền",
        "durability",
        "lưu trữ bền vững",
        "khả năng phục hồi dữ liệu",
        "bền vững sau sự cố",
    ],
    "primary_key": [
        "khóa chính",
        "primary key",
        "pk",
        "khóa nguyên phát",
        "thuộc tính khóa chính",
        "khóa chính bảng",
        "định danh duy nhất bản ghi",
    ],
    "foreign_key": [
        "khóa ngoại",
        "foreign key",
        "fk",
        "khóa liên kết",
        "khóa tham chiếu",
        "toàn vẹn tham chiếu",
        "referential integrity",
        "ràng buộc khóa ngoại",
    ],
    "candidate_key": [
        "khóa ứng viên",
        "khóa ứng cử viên",
        "candidate key",
        "siêu khóa tối thiểu",
        "minimal superkey",
        "khóa tiềm năng",
    ],
    "super_key": [
        "siêu khóa",
        "super key",
        "superkey",
        "tập thuộc tính xác định duy nhất",
        "siêu khóa quan hệ",
    ],
    "composite_key": [
        "khóa kết hợp",
        "khóa phức hợp",
        "composite key",
        "compound key",
        "khóa ghép",
        "khóa gồm nhiều thuộc tính",
    ],
    "normalization": [
        "chuẩn hóa",
        "chuẩn hóa dữ liệu",
        "chuẩn hóa cơ sở dữ liệu",
        "database normalization",
        "dạng chuẩn",
        "các dạng chuẩn",
        "normal forms",
        "chuẩn hóa quan hệ",
    ],
    "first_normal_form": [
        "1nf",
        "dạng chuẩn 1",
        "dạng chuẩn 1nf",
        "first normal form",
        "thuộc tính nguyên tử",
        "giá trị nguyên tử",
        "atomic values",
        "chuẩn 1nf",
    ],
    "second_normal_form": [
        "2nf",
        "dạng chuẩn 2",
        "dạng chuẩn 2nf",
        "second normal form",
        "phụ thuộc hàm đầy đủ",
        "loại bỏ phụ thuộc bộ phận",
        "full functional dependency",
        "chuẩn 2nf",
    ],
    "third_normal_form": [
        "3nf",
        "dạng chuẩn 3",
        "dạng chuẩn 3nf",
        "third normal form",
        "loại bỏ phụ thuộc bắc cầu",
        "phụ thuộc bắc cầu",
        "transitive dependency",
        "chuẩn 3nf",
    ],
    "bcnf": [
        "bcnf",
        "dạng chuẩn boyce-codd",
        "boyce-codd normal form",
        "dạng chuẩn bcnf",
        "chuẩn bcnf",
        "boyce-codd",
        "chuẩn boyce codd",
    ],
    "functional_dependency": [
        "phụ thuộc hàm",
        "functional dependency",
        "fd",
        "x xác định y",
        "phụ thuộc dữ liệu",
        "x suy ra y",
    ],
    "btree_index": [
        "b-tree",
        "chỉ mục b-tree",
        "b-tree index",
        "b tree",
        "cây b-tree",
        "cây b",
        "chỉ mục cây b",
        "chỉ mục cây b-tree",
    ],
    "bplus_tree_index": [
        "b+tree",
        "chỉ mục b+tree",
        "b+ tree index",
        "b+ tree",
        "cây b+tree",
        "b plus tree",
        "cây b+ tree",
        "chỉ mục b+ tree",
    ],
    "clustered_index": [
        "chỉ mục phân cụm",
        "clustered index",
        "chỉ mục gom cụm",
        "chỉ mục vật lý",
        "chỉ mục cụm",
        "clustered b-tree",
    ],
    "non_clustered_index": [
        "chỉ mục không phân cụm",
        "non-clustered index",
        "secondary index",
        "chỉ mục phụ",
        "chỉ mục phi phân cụm",
        "chỉ mục logic",
    ],
    "transaction": [
        "giao dịch",
        "giao dịch cơ sở dữ liệu",
        "transaction",
        "db transaction",
        "quản lý giao dịch",
        "chuỗi thao tác cơ sở dữ liệu",
    ],
    "transaction_control": [
        "commit",
        "rollback",
        "savepoint",
        "xác nhận giao dịch",
        "hoàn tác giao dịch",
        "lệnh commit",
        "lệnh rollback",
        "điểm lưu giao dịch",
    ],
    "concurrency_control": [
        "kiểm soát tương tranh",
        "concurrency control",
        "khóa hai pha",
        "2pl",
        "two-phase locking",
        "deadlock",
        "khóa chết",
        "tranh chấp dữ liệu",
        "hiện tượng khóa chết",
    ],
    "isolation_phenomena": [
        "dirty read",
        "đọc rác",
        "non-repeatable read",
        "đọc không lặp lại",
        "phantom read",
        "đọc bóng ma",
        "hiện tượng đọc dữ liệu",
    ],
    "sql_join": [
        "kết nối bảng",
        "phép kết nối",
        "join",
        "sql join",
        "ghép bảng",
        "phép kết nối bảng",
        "truy vấn nhiều bảng",
    ],
    "inner_join": [
        "inner join",
        "kết nối trong",
        "kết nối nội",
        "kết trong",
        "equi join",
        "kết nối bằng",
        "phép kết trong",
    ],
    "left_join": [
        "left join",
        "left outer join",
        "kết nối trái",
        "kết nối ngoài bên trái",
        "kết ngoài trái",
        "phép kết trái",
    ],
    "right_join": [
        "right join",
        "right outer join",
        "kết nối phải",
        "kết nối ngoài bên phải",
        "kết ngoài phải",
        "phép kết phải",
    ],
    "full_join": [
        "full join",
        "full outer join",
        "kết nối đầy đủ",
        "kết nối ngoài toàn phần",
        "kết ngoài toàn phần",
        "phép kết ngoài đầy đủ",
    ],
    "cross_join": [
        "cross join",
        "tích descartes",
        "cartesian product",
        "kết chéo",
        "phép nhân quan hệ",
        "tích đề-các",
    ],
    "self_join": [
        "self join",
        "tự kết nối",
        "kết nối chính nó",
        "tự kết",
        "phép tự kết nối",
    ],
}


class DatabaseThesaurusRegistry:
    """
    Registry for bidirectional database and SQL terminology mappings.
    Supports exact matching, tone-insensitive matching, and n-gram term discovery.
    """

    def __init__(self, clusters: Optional[Dict[str, List[str]]] = None):
        self.clusters = clusters or DATABASE_SYNONYM_CLUSTERS
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
        Extracts database domain terms from the query and expands them into synonyms.
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


# Default Singleton Instance
_DEFAULT_DB_REGISTRY = DatabaseThesaurusRegistry()


def get_database_synonyms(term: str) -> List[str]:
    """
    Public helper to retrieve bidirectional database and SQL synonyms.
    Guaranteed tone-insensitive and casing-insensitive.
    """
    return _DEFAULT_DB_REGISTRY.get_synonyms(term)


def expand_database_query_terms(query: str) -> List[str]:
    """
    Expands conversational user queries into authoritative technical Database keywords.
    E.g.: 'khóa chính và khóa ngoại' -> ['khóa chính', 'primary key', 'khóa ngoại', 'foreign key', ...]
    """
    return _DEFAULT_DB_REGISTRY.expand_query(query)


def is_sql_or_relational_query(text: str) -> bool:
    """Quick check whether a given text contains database or SQL domain concepts."""
    return len(_DEFAULT_DB_REGISTRY.expand_query(text)) > 0


# Few-shot QA examples for Database Systems & SQL Curriculum
DATABASE_FEW_SHOT_EXAMPLES: List[Dict[str, Any]] = [
    {
        "id": "db_qa_01",
        "topic": "Thuộc tính ACID trong Giao dịch Cơ sở dữ liệu",
        "category": "transactions_and_acid",
        "canonical_terms": [
            "acid",
            "atomicity",
            "consistency",
            "isolation",
            "durability",
            "giao dịch",
            "transaction",
        ],
        "question": "Thuộc tính ACID trong hệ quản trị cơ sở dữ liệu là gì và ý nghĩa của từng chữ cái?",
        "intent": "concept_definition_and_breakdown",
        "expected_action": "explain_acid_properties_in_detail",
        "grounded_answer": (
            "ACID là 4 thuộc tính nền tảng bắt buộc của một giao dịch (transaction) trong Hệ quản trị cơ sở dữ liệu quan hệ (RDBMS), "
            "nhằm đảm bảo tính toàn vẹn và độ tin cậy của dữ liệu ngay cả khi xảy ra lỗi hệ thống hoặc xung đột đồng thời:\n\n"
            "1. A - Atomicity (Tính nguyên tử):\n"
            "Đảm bảo nguyên tắc 'tất cả hoặc không gì cả' (all-or-nothing). Toàn bộ các thao tác trong giao dịch phải hoàn tất thành công "
            "(COMMIT). Nếu bất kỳ thao tác nào thất bại, toàn bộ giao dịch sẽ bị hủy bỏ và phục hồi lại trạng thái ban đầu (ROLLBACK).\n\n"
            "2. C - Consistency (Tính nhất quán):\n"
            "Dữ liệu trước và sau khi hoàn tất giao dịch phải luôn thỏa mãn tất cả các quy tắc toàn vẹn định sẵn "
            "(khóa chính, khóa ngoại, kiểu dữ liệu, ràng buộc CHECK, trigger, v.v.).\n\n"
            "3. I - Isolation (Tính cô lập):\n"
            "Các giao dịch thực thi đồng thời không được can thiệp lẫn nhau. Kết quả của một giao dịch đang chạy phải ẩn đối với các giao dịch khác "
            "cho đến khi nó được COMMIT, tránh các hiện tượng như Dirty Read hay Lost Update.\n\n"
            "4. D - Durability (Tính bền vững):\n"
            "Khi một giao dịch đã COMMIT thành công, các thay đổi dữ liệu sẽ được ghi nhận vĩnh viễn vào bộ lưu trữ không khả biến (đĩa cứng/WAL), "
            "không bị mất đi ngay cả khi hệ thống sập nguồn đột ngột."
        ),
        "bullet_types": [
            "- Atomicity: Tất cả hoặc không gì cả (All-or-Nothing), Rollback khi lỗi.",
            "- Consistency: Duy trì toàn vẹn ràng buộc trước và sau giao dịch.",
            "- Isolation: Các giao dịch đồng thời không can thiệp lẫn nhau.",
            "- Durability: Dữ liệu đã commit được lưu trữ vĩnh viễn bất chấp sự cố.",
        ],
        "example": (
            "Chuyển 500k từ tài khoản A sang B: Trừ tiền A thành công nhưng cộng tiền B gặp sự cố mạng -> "
            "Tính Atomicity lập tức Rollback, tiền của A được hoàn lại 500k."
        ),
    },
    {
        "id": "db_qa_02",
        "topic": "Khóa chính (Primary Key) và Khóa ngoại (Foreign Key)",
        "category": "keys_and_relational_integrity",
        "canonical_terms": [
            "khóa chính",
            "primary key",
            "pk",
            "khóa ngoại",
            "foreign key",
            "fk",
            "toàn vẹn tham chiếu",
            "referential integrity",
        ],
        "question": "Phân biệt Khóa chính (Primary Key) và Khóa ngoại (Foreign Key) trong mô hình dữ liệu quan hệ?",
        "intent": "comparative_concept_analysis",
        "expected_action": "compare_primary_key_vs_foreign_key",
        "grounded_answer": (
            "So sánh Khóa chính (Primary Key) và Khóa ngoại (Foreign Key) trong cơ sở dữ liệu quan hệ:\n\n"
            "1. Khái niệm và mục đích:\n"
            "- Khóa chính (PK): Là một hoặc một tập hợp thuộc tính dùng để định danh duy nhất (uniquely identify) mỗi hàng (tuple) trong một bảng quan hệ.\n"
            "- Khóa ngoại (FK): Là một hoặc một tập hợp thuộc tính trong một bảng tham chiếu đến Khóa chính (hoặc khóa ứng viên duy nhất) của bảng khác, "
            "dùng để thiết lập liên kết giữa hai bảng.\n\n"
            "2. Giá trị NULL và tính duy nhất:\n"
            "- Khóa chính: Tuyệt đối KHÔNG được chứa giá trị NULL (ràng buộc toàn vẹn thực thể - Entity Integrity) và giá trị phải là duy nhất (UNIQUE).\n"
            "- Khóa ngoại: CÓ THỂ chấp nhận giá trị NULL (nếu mối quan hệ là tùy chọn) và CÓ THỂ chứa các giá trị trùng lặp (quan hệ 1-N).\n\n"
            "3. Số lượng trên một bảng:\n"
            "- Một bảng chỉ có DUY NHẤT một Khóa chính (dù có thể là khóa kết hợp gồm nhiều cột).\n"
            "- Một bảng có thể có NHIỀU Khóa ngoại tham chiếu đến nhiều bảng khác nhau.\n\n"
            "4. Ràng buộc toàn vẹn:\n"
            "- Khóa chính đảm bảo toàn vẹn thực thể (Entity Integrity).\n"
            "- Khóa ngoại đảm bảo toàn vẹn tham chiếu (Referential Integrity) kèm theo các hành động khi xóa/sửa như CASCADE, SET NULL, RESTRICT."
        ),
        "bullet_types": [
            "- Khóa chính: Duy nhất, không NULL, định danh dòng, tối đa 1 PK trên mỗi bảng.",
            "- Khóa ngoại: Tham chiếu đến PK bảng khác, có thể trùng lặp, có thể NULL, tạo liên kết.",
            "- Ràng buộc: PK bảo vệ Entity Integrity; FK bảo vệ Referential Integrity.",
        ],
        "example": (
            "Bảng SinhVien có MaSV là Primary Key. Bảng DangKyHoc có MaSV là Foreign Key "
            "tham chiếu đến SinhVien(MaSV)."
        ),
    },
    {
        "id": "db_qa_03",
        "topic": "Siêu khóa (Super Key), Khóa ứng viên (Candidate Key) và Khóa chính (Primary Key)",
        "category": "keys_and_relational_integrity",
        "canonical_terms": [
            "siêu khóa",
            "super key",
            "khóa ứng viên",
            "candidate key",
            "khóa chính",
            "primary key",
            "minimal superkey",
        ],
        "question": "Thế nào là Siêu khóa, Khóa ứng viên và Khóa chính? Mối quan hệ giữa chúng là gì?",
        "intent": "taxonomy_and_relationship_hierarchy",
        "expected_action": "explain_key_hierarchy",
        "grounded_answer": (
            "Mối quan hệ phân cấp giữa Siêu khóa, Khóa ứng viên và Khóa chính:\n\n"
            "1. Siêu khóa (Super Key - SK):\n"
            "Là một tập hợp gồm một hoặc nhiều thuộc tính mà giá trị của chúng có thể phân biệt duy nhất từng bộ (bản ghi) trong quan hệ. "
            "Siêu khóa có thể chứa các thuộc tính dư thừa không cần thiết cho tính duy nhất.\n\n"
            "2. Khóa ứng viên (Candidate Key - CK):\n"
            "Là một Siêu khóa tối thiểu (Minimal Super Key). Nghĩa là nếu loại bỏ bất kỳ một thuộc tính nào ra khỏi CK thì tập thuộc tính còn lại "
            "sẽ mất đi tính duy nhất. Một quan hệ có thể có nhiều khóa ứng viên.\n\n"
            "3. Khóa chính (Primary Key - PK):\n"
            "Là một Khóa ứng viên được người thiết kế cơ sở dữ liệu lựa chọn chính thức làm bộ định danh chính cho bảng.\n\n"
            "Tóm tắt mối quan hệ bao hàm:\n"
            "Khóa chính ⊆ Khóa ứng viên ⊆ Siêu khóa.\n"
            "(Mọi Khóa chính đều là Khóa ứng viên; mọi Khóa ứng viên đều là Siêu khóa; nhưng chiều ngược lại không nhất thiết đúng)."
        ),
        "bullet_types": [
            "- Siêu khóa: Bất kỳ tập thuộc tính nào định danh duy nhất bản ghi (có thể thừa thuộc tính).",
            "- Khóa ứng viên: Siêu khóa tối thiểu (bỏ bớt 1 thuộc tính là mất tính duy nhất).",
            "- Khóa chính: Khóa ứng viên được chỉ định làm định danh đại diện.",
        ],
        "example": (
            "Bảng NhanVien {CCCD, MaNV, HoTen, Email}:\n"
            "- Siêu khóa: {CCCD, HoTen}, {MaNV, Email}, {CCCD}, {MaNV}.\n"
            "- Khóa ứng viên: {CCCD}, {MaNV}.\n"
            "- Khóa chính: Chọn {MaNV} làm Primary Key."
        ),
    },
    {
        "id": "db_qa_04",
        "topic": "Chuẩn hóa Cơ sở dữ liệu và Dạng chuẩn 1NF (First Normal Form)",
        "category": "normalization",
        "canonical_terms": [
            "chuẩn hóa",
            "1nf",
            "dạng chuẩn 1",
            "thuộc tính nguyên tử",
            "atomic values",
            "database normalization",
        ],
        "question": "Chuẩn hóa cơ sở dữ liệu là gì và điều kiện để một bảng đạt dạng chuẩn 1 (1NF)?",
        "intent": "concept_and_condition_explanation",
        "expected_action": "define_normalization_and_1nf_rules",
        "grounded_answer": (
            "Chuẩn hóa cơ sở dữ liệu (Database Normalization) là quá trình tổ chức lại các bảng và thuộc tính "
            "trong mô hình quan hệ nhằm giảm thiểu dư thừa dữ liệu (data redundancy) và loại bỏ các bất thường khi thao tác dữ liệu "
            "(Insertion, Update, Deletion Anomalies).\n\n"
            "Điều kiện để đạt Dạng chuẩn 1 (1NF - First Normal Form):\n"
            "Một quan hệ đạt chuẩn 1NF khi và chỉ khi thỏa mãn các điều kiện sau:\n"
            "1. Tính nguyên tử (Atomicity): Mỗi ô dữ liệu (giao giữa hàng và cột) chỉ được chứa đúng MỘT giá trị đơn nhất (giá trị nguyên tử), "
            "không được chứa danh sách, tập hợp mảng, hay đối tượng lồng nhau.\n"
            "2. Không có nhóm lặp (No Repeating Groups): Không được phép tạo các cột lặp lại để lưu nhiều giá trị cùng loại (ví dụ: SoDienThoai_1, SoDienThoai_2).\n"
            "3. Mỗi hàng phải được phân biệt duy nhất (có khóa chính).\n"
            "4. Thứ tự của các hàng và các cột không ảnh hưởng đến ý nghĩa dữ liệu."
        ),
        "bullet_types": [
            "- Mục tiêu chuẩn hóa: Giảm dư thừa, tránh dị thường Thêm/Sửa/Xóa.",
            "- 1NF: Mọi giá trị thuộc tính phải nguyên tử (atomic), không mảng, không danh sách.",
            "- 1NF: Không chứa các cột nhóm lặp (repeating groups).",
        ],
        "example": (
            "Vi phạm 1NF: Cột SoDienThoai chứa '0901234567, 0987654321'. "
            "Đưa về 1NF: Tách thành 2 dòng riêng biệt cho từng số điện thoại hoặc tách sang bảng liên kết."
        ),
    },
    {
        "id": "db_qa_05",
        "topic": "Dạng chuẩn 2NF (Second Normal Form) và Phụ thuộc hàm đầy đủ",
        "category": "normalization",
        "canonical_terms": [
            "2nf",
            "dạng chuẩn 2",
            "phụ thuộc hàm đầy đủ",
            "phụ thuộc bộ phận",
            "partial dependency",
            "full functional dependency",
        ],
        "question": "Điều kiện để một quan hệ đạt dạng chuẩn 2 (2NF) là gì? Thế nào là phụ thuộc bộ phận?",
        "intent": "normalization_rule_and_partial_dependency",
        "expected_action": "explain_2nf_and_partial_dependency",
        "grounded_answer": (
            "Dạng chuẩn 2 (2NF - Second Normal Form) giải quyết vấn đề phụ thuộc hàm bộ phận vào khóa chính.\n\n"
            "Điều kiện để quan hệ đạt chuẩn 2NF:\n"
            "1. Quan hệ đó đã đạt dạng chuẩn 1 (1NF).\n"
            "2. Mọi thuộc tính không khóa (non-prime attribute) phải phụ thuộc hàm đầy đủ (fully functionally dependent) vào khóa chính, "
            "nghĩa là không có bất kỳ thuộc tính không khóa nào phụ thuộc vào một phần của khóa chính (không có phụ thuộc bộ phận).\n\n"
            "Phụ thuộc bộ phận (Partial Dependency) là gì?\n"
            "Xảy ra khi khóa chính là khóa kết hợp (gồm từ 2 thuộc tính trở lên, ví dụ {A, B}), nhưng tồn tại thuộc tính không khóa C "
            "mà chỉ phụ thuộc vào một thuộc tính thành phần của khóa (ví dụ A -> C mà không cần B).\n\n"
            "Lưu ý đặc biệt:\n"
            "Nếu một bảng đã ở 1NF và Khóa chính của bảng chỉ gồm DUY NHẤT MỘT thuộc tính đơn, thì bảng đó TỰ ĐỘNG đạt 2NF "
            "(vì không thể chia nhỏ khóa chính đơn)."
        ),
        "bullet_types": [
            "- Điều kiện 2NF: Đã ở 1NF + Mọi thuộc tính không khóa phụ thuộc đầy đủ vào khóa chính.",
            "- Loại bỏ phụ thuộc bộ phận: Thuộc tính không khóa không được chỉ phụ thuộc 1 phần khóa ghép.",
            "- Bảng có khóa chính đơn đã đạt 1NF thì nghiễm nhiên đạt 2NF.",
        ],
        "example": (
            "Bảng {MaSV, MaMH, Diem, TenMH} có Khóa chính là {MaSV, MaMH}.\n"
            "Phụ thuộc hàm MaMH -> TenMH: TenMH chỉ phụ thuộc vào MaMH (một phần của khóa) => Vi phạm 2NF.\n"
            "Khắc phục: Tách thành bảng DiemThi{MaSV, MaMH, Diem} và MonHoc{MaMH, TenMH}."
        ),
    },
    {
        "id": "db_qa_06",
        "topic": "Dạng chuẩn 3NF (Third Normal Form) và Phụ thuộc hàm bắc cầu",
        "category": "normalization",
        "canonical_terms": [
            "3nf",
            "dạng chuẩn 3",
            "phụ thuộc bắc cầu",
            "transitive dependency",
            "thuộc tính không khóa",
            "non-prime attribute",
        ],
        "question": "Điều kiện để quan hệ đạt dạng chuẩn 3 (3NF) là gì? Thế nào là phụ thuộc bắc cầu?",
        "intent": "normalization_rule_and_transitive_dependency",
        "expected_action": "explain_3nf_and_transitive_dependency",
        "grounded_answer": (
            "Dạng chuẩn 3 (3NF - Third Normal Form) giải quyết triệt để sự phụ thuộc lẫn nhau giữa các thuộc tính không khóa.\n\n"
            "Điều kiện để quan hệ đạt chuẩn 3NF:\n"
            "1. Quan hệ đó đã đạt dạng chuẩn 2 (2NF).\n"
            "2. Không tồn tại phụ thuộc hàm bắc cầu (Transitive Dependency) giữa các thuộc tính không khóa vào khóa chính.\n\n"
            "Phát biểu hình thức theo lý thuyết phụ thuộc hàm:\n"
            "Với mọi phụ thuộc hàm không tầm thường X -> Y trong lược đồ R, ít nhất một trong hai điều kiện sau phải thỏa mãn:\n"
            "- X là một siêu khóa (Superkey) của R, HOẶC\n"
            "- Mỗi thuộc tính trong Y là một thuộc tính khóa (Prime attribute - thuộc về một khóa ứng viên nào đó).\n\n"
            "Thế nào là phụ thuộc bắc cầu?\n"
            "Nếu A là khóa chính, tồn tại thuộc tính không khóa B sao cho A -> B, và thuộc tính không khóa C sao cho B -> C, "
            "dẫn đến A -> C một cách gián tiếp. Khi đó C phụ thuộc bắc cầu vào A qua B. Cần tách B và C ra bảng riêng."
        ),
        "bullet_types": [
            "- Điều kiện 3NF: Đạt 2NF + Không có phụ thuộc bắc cầu giữa các thuộc tính không khóa.",
            "- Quy tắc X -> Y: Hoặc X là siêu khóa, hoặc Y là thuộc tính khóa.",
            "- Khắc phục: Tách các thuộc tính phụ thuộc bắc cầu sang bảng mới.",
        ],
        "example": (
            "Bảng NhanVien {MaNV, HoTen, MaPhong, TenPhong, TruongPhong}.\n"
            "Khóa chính là MaNV. Ta có: MaNV -> MaPhong và MaPhong -> TenPhong.\n"
            "TenPhong phụ thuộc bắc cầu vào MaNV qua MaPhong => Vi phạm 3NF.\n"
            "Khắc phục: Tách thành NhanVien{MaNV, HoTen, MaPhong} và PhongBan{MaPhong, TenPhong, TruongPhong}."
        ),
    },
    {
        "id": "db_qa_07",
        "topic": "Dạng chuẩn Boyce-Codd (BCNF) so với 3NF",
        "category": "normalization",
        "canonical_terms": [
            "bcnf",
            "dạng chuẩn boyce-codd",
            "boyce-codd",
            "3nf",
            "siêu khóa",
            "superkey",
        ],
        "question": "Dạng chuẩn BCNF là gì và tại sao nó nghiêm ngặt hơn dạng chuẩn 3NF?",
        "intent": "advanced_normalization_comparison",
        "expected_action": "compare_bcnf_and_3nf",
        "grounded_answer": (
            "Dạng chuẩn Boyce-Codd (BCNF - Boyce-Codd Normal Form) là dạng chuẩn mở rộng nghiêm ngặt hơn của 3NF (thường gọi là chuẩn 3.5NF).\n\n"
            "Định nghĩa BCNF:\n"
            "Một lược đồ quan hệ R đạt chuẩn BCNF khi và chỉ khi với MỌI phụ thuộc hàm không tầm thường X -> Y tồn tại trong R, "
            "vế trái X BẮT BUỘC PHẢI LÀ MỘT SIÊU KHÓA (Superkey).\n\n"
            "So sánh với 3NF:\n"
            "- Trong 3NF, phụ thuộc hàm X -> Y được phép tồn tại nếu X là siêu khóa HOẶC Y là thuộc tính khóa (prime attribute).\n"
            "- Trong BCNF, điều kiện nới lỏng 'Y là thuộc tính khóa' hoàn toàn bị BÃI BỎ. Mọi determinant (vế trái xác định) đều phải là siêu khóa.\n\n"
            "Đánh đổi khi đưa về BCNF:\n"
            "Mọi phân rã về 3NF đều đảm bảo bảo toàn thông tin (Lossless Join) và bảo toàn phụ thuộc hàm (Dependency Preservation). "
            "Tuy nhiên, khi phân rã một số lược đồ về BCNF, ta có thể bị mất mát khả năng bảo toàn phụ thuộc hàm."
        ),
        "bullet_types": [
            "- BCNF: Với mọi X -> Y không tầm thường, X bắt buộc phải là siêu khóa.",
            "- BCNF nghiêm ngặt hơn 3NF vì không cho phép ngoại lệ 'Y là thuộc tính khóa'.",
            "- Đánh đổi: BCNF bảo toàn kết nối nhưng đôi khi không bảo toàn được tất cả phụ thuộc hàm.",
        ],
        "example": (
            "Quan hệ {SinhVien, MonHoc, GiangVien}. Khóa ứng viên: {SinhVien, GiangVien} và {SinhVien, MonHoc}.\n"
            "Phụ thuộc hàm: GiangVien -> MonHoc. MonHoc là thuộc tính khóa nên đạt 3NF, "
            "nhưng GiangVien không phải siêu khóa nên vi phạm BCNF."
        ),
    },
    {
        "id": "db_qa_08",
        "topic": "Chỉ mục B-Tree và B+Tree (B-Tree / B+Tree Index)",
        "category": "indexing_and_storage",
        "canonical_terms": [
            "b-tree",
            "b+tree",
            "chỉ mục b-tree",
            "chỉ mục b+tree",
            "index",
            "cây b-tree",
            "độ phức tạp o(log n)",
        ],
        "question": "Chỉ mục B-Tree và B+Tree hoạt động như thế nào trong CSDL? Tại sao B+Tree được ưa chuộng hơn cho RDBMS?",
        "intent": "data_structure_and_index_mechanism",
        "expected_action": "explain_btree_and_bplus_tree",
        "grounded_answer": (
            "B-Tree và B+Tree là các cấu trúc dữ liệu cây tìm kiếm tự cân bằng nhiều nhánh (self-balancing multi-way search trees), "
            "được thiết kế tối ưu cho các hệ thống lưu trữ khối trên đĩa cứng (disk-block I/O):\n\n"
            "1. Đặc điểm cấu trúc B-Tree:\n"
            "- Mỗi node có thể chứa nhiều khóa và nhiều con trỏ.\n"
            "- Cây luôn tự cân bằng, đảm bảo chiều cao h ở mức tối thiểu. Tìm kiếm, thêm, xóa đạt độ phức tạp thời gian O(log N).\n"
            "- Trong B-Tree gốc, dữ liệu (con trỏ bản ghi) được lưu trữ ở CẢ node trong lẫn node lá.\n\n"
            "2. Cải tiến của B+Tree:\n"
            "- Node trung gian (Internal nodes): CHỈ lưu trữ khóa định tuyến và con trỏ điều hướng, KHÔNG lưu con trỏ dữ liệu thực tế.\n"
            "- Node lá (Leaf nodes): Lưu trữ toàn bộ khóa và con trỏ dữ liệu thực tế.\n"
            "- Danh sách liên kết kép ở lá: Tất cả các node lá được liên kết tuần tự với nhau (doubly linked list).\n\n"
            "3. Tại sao RDBMS (như MySQL InnoDB, PostgreSQL) ưa chuộng B+Tree hơn B-Tree?\n"
            "- Fan-out lớn hơn, cây thấp hơn: Do node trung gian không chứa con trỏ dữ liệu nên mỗi trang đĩa (page 16KB) chứa được nhiều khóa hơn, giảm số lần đọc I/O đĩa.\n"
            "- Truy vấn theo dải (Range Scan) vượt trội: Khi tìm kiếm theo khoảng (BETWEEN, >=, <=), B+Tree chỉ cần tìm đến node lá đầu tiên rồi duyệt tuyến tính dọc theo danh sách liên kết của các node lá, không cần duyệt ngược lên các node cha như B-Tree."
        ),
        "bullet_types": [
            "- Cây tự cân bằng nhiều nhánh, độ phức tạp tìm kiếm O(log N).",
            "- B+Tree: Node trung gian chỉ chứa khóa điều hướng, node lá chứa trọn vẹn dữ liệu.",
            "- B+Tree: Các node lá liên kết tuần tự hai chiều, tối ưu hóa triệt để Range Query và Scan.",
        ],
        "example": (
            "Câu lệnh: SELECT * FROM DonHang WHERE NgayTao BETWEEN '2026-01-01' AND '2026-01-31'; "
            "B+Tree tìm node lá chứa '2026-01-01' rồi duyệt tuần tự đến '2026-01-31' với chi phí I/O cực thấp."
        ),
    },
    {
        "id": "db_qa_09",
        "topic": "Chỉ mục Phân cụm (Clustered Index) và Chỉ mục Không phân cụm (Non-Clustered Index)",
        "category": "indexing_and_storage",
        "canonical_terms": [
            "clustered index",
            "chỉ mục phân cụm",
            "non-clustered index",
            "chỉ mục không phân cụm",
            "secondary index",
            "chỉ mục phụ",
        ],
        "question": "Phân biệt Clustered Index (Chỉ mục phân cụm) và Non-Clustered Index (Chỉ mục thứ cấp)?",
        "intent": "index_architecture_comparison",
        "expected_action": "compare_clustered_vs_non_clustered",
        "grounded_answer": (
            "So sánh Clustered Index và Non-Clustered (Secondary) Index trong hệ quản trị CSDL:\n\n"
            "1. Cấu trúc lưu trữ vật lý:\n"
            "- Clustered Index: Quyết định thứ tự sắp xếp vật lý thực tế của các dòng dữ liệu trên đĩa. Node lá của Clustered Index chính là các trang dữ liệu thực tế (Data Pages).\n"
            "- Non-Clustered Index: Cấu trúc cây chỉ mục nằm tách rời hoàn toàn với dữ liệu vật lý. Node lá của Non-Clustered Index chỉ chứa khóa chỉ mục và con trỏ tham chiếu trỏ về dòng dữ liệu thực tế (Row ID hoặc Khóa chính Clustered Key).\n\n"
            "2. Số lượng chỉ mục trên một bảng:\n"
            "- Clustered Index: Chỉ có DUY NHẤT 1 chỉ mục trên mỗi bảng (vì dữ liệu vật lý chỉ có thể sắp xếp theo một thứ tự duy nhất trên đĩa).\n"
            "- Non-Clustered Index: Có thể tạo NHIỀU chỉ mục trên cùng một bảng (ví dụ tối đa 999 trong MS SQL Server) để tối ưu các truy vấn trên nhiều cột khác nhau.\n\n"
            "3. Hiệu năng truy vấn:\n"
            "- Clustered Index: Cực nhanh khi quét dải (Range Scan), sắp xếp ORDER BY theo khóa chỉ mục vì dữ liệu nằm kề nhau trên đĩa.\n"
            "- Non-Clustered Index: Tốt cho tìm kiếm giá trị chính xác (Point Lookup); tuy nhiên nếu truy vấn chọn các cột không nằm trong chỉ mục, hệ thống phải thực hiện thêm bước tra cứu lại bảng gốc (Key Lookup / Bookmark Lookup)."
        ),
        "bullet_types": [
            "- Clustered Index: Sắp xếp vật lý bảng, node lá là Data Page, tối đa 1 index/bảng.",
            "- Non-Clustered Index: Cấu trúc phụ trợ độc lập, node lá trỏ về Clustered Key/RID, nhiều index/bảng.",
            "- Lookup: Clustered đọc trực tiếp; Non-clustered có thể cần thêm bước Key Lookup.",
        ],
        "example": (
            "Cuốn từ điển: Toàn bộ từ được in xếp thứ tự A-Z là Clustered Index. "
            "Mục lục tra cứu theo chủ đề ở cuối sách kèm số trang là Non-Clustered Index."
        ),
    },
    {
        "id": "db_qa_10",
        "topic": "Quản lý Giao dịch và Kiểm soát Tương tranh (Concurrency Control, 2PL, Deadlock)",
        "category": "transactions_and_acid",
        "canonical_terms": [
            "giao dịch",
            "concurrency control",
            "khóa hai pha",
            "2pl",
            "deadlock",
            "khóa chết",
            "commit",
            "rollback",
        ],
        "question": "Kiểm soát tương tranh trong cơ sở dữ liệu là gì? Trình bày cơ chế Khóa hai pha (2PL) và hiện tượng Deadlock?",
        "intent": "concurrency_control_mechanism_and_deadlock",
        "expected_action": "explain_concurrency_control_and_deadlock",
        "grounded_answer": (
            "Kiểm soát tương tranh (Concurrency Control) là cơ chế của DBMS nhằm điều phối các giao dịch đồng thời "
            "sao cho chúng không ghi đè dữ liệu lên nhau, đảm bảo lịch trình thực thi tương đương với một lịch trình tuần tự (Serializability).\n\n"
            "1. Giao thức Khóa hai pha (Two-Phase Locking - 2PL):\n"
            "Là giao thức đảm bảo tính tuần tự hóa thông qua việc sử dụng các khóa (Shared Lock - đọc, Exclusive Lock - ghi) gồm 2 giai đoạn:\n"
            "- Giai đoạn Mở rộng (Growing Phase): Giao dịch có thể xin cấp thêm khóa mới nhưng TUYỆT ĐỐI KHÔNG ĐƯỢC giải phóng bất kỳ khóa nào.\n"
            "- Giai đoạn Thu hẹp (Shrinking Phase): Giao dịch bắt đầu giải phóng các khóa đã giữ và TUYỆT ĐỐI KHÔNG ĐƯỢC xin cấp thêm bất kỳ khóa mới nào.\n"
            "Cột mốc chuyển tiếp giữa 2 giai đoạn gọi là Điểm khóa (Lock Point).\n\n"
            "2. Hiện tượng Khóa chết (Deadlock):\n"
            "Deadlock xảy ra khi hai hoặc nhiều giao dịch rơi vào trạng thái chờ đợi lẫn nhau giải phóng tài nguyên tạo thành một chu trình đóng (Wait-For Graph có chu trình).\n"
            "Ví dụ: Giao dịch T1 giữ khóa tài nguyên A và đợi khóa tài nguyên B; trong khi Giao dịch T2 giữ khóa tài nguyên B và đợi khóa tài nguyên A.\n"
            "Cách xử lý của DBMS: Định kỳ quét đồ thị chờ đợi (Wait-For Graph), phát hiện chu trình, chọn một giao dịch làm 'nạn nhân' (victim) "
            "để thực hiện ROLLBACK, giải phóng khóa cho các giao dịch còn lại tiếp tục."
        ),
        "bullet_types": [
            "- Mục tiêu Concurrency Control: Đảm bảo Serializability khi nhiều transaction chạy đồng thời.",
            "- 2PL gồm 2 pha: Pha mở rộng (chỉ xin thêm khóa) và Pha thu hẹp (chỉ nhả khóa).",
            "- Deadlock: Chờ đợi vòng tròn tài nguyên, giải quyết bằng Wait-For Graph và Rollback transaction nạn nhân.",
        ],
        "example": (
            "T1: Lock(AccountA), Wait Lock(AccountB).\n"
            "T2: Lock(AccountB), Wait Lock(AccountA).\n"
            "=> DBMS kích hoạt Deadlock Detector sau timeout, rollback T1."
        ),
    },
    {
        "id": "db_qa_11",
        "topic": "Các Cấp độ Cô lập Giao dịch và Hiện tượng Bất thường Đọc Dữ liệu",
        "category": "transactions_and_acid",
        "canonical_terms": [
            "isolation level",
            "mức cô lập",
            "dirty read",
            "non-repeatable read",
            "phantom read",
            "read committed",
            "repeatable read",
            "serializable",
        ],
        "question": "Các cấp độ cô lập (Isolation Levels) chuẩn SQL là gì và chúng ngăn chặn các hiện tượng đọc dị thường nào?",
        "intent": "isolation_levels_and_read_phenomena",
        "expected_action": "explain_isolation_levels_matrix",
        "grounded_answer": (
            "Chuẩn SQL-92 định nghĩa 4 cấp độ cô lập (Isolation Levels) dựa trên khả năng ngăn chặn 3 hiện tượng dị thường đọc dữ liệu:\n\n"
            "1. Ba hiện tượng dị thường đọc:\n"
            "- Dirty Read (Đọc rác): Giao dịch T1 đọc dữ liệu chưa được COMMIT của giao dịch T2; nếu T2 sau đó ROLLBACK, dữ liệu T1 đã đọc trở nên vô giá trị.\n"
            "- Non-repeatable Read (Đọc không lặp lại): Giao dịch T1 đọc một dòng; T2 sửa hoặc xóa dòng đó và COMMIT; T1 đọc lại dòng đó thì nhận được giá trị khác ban đầu.\n"
            "- Phantom Read (Đọc bóng ma): Giao dịch T1 thực hiện truy vấn theo điều kiện (ví dụ WHERE Tuoi > 18); T2 INSERT thêm các dòng mới thỏa mãn điều kiện đó và COMMIT; T1 chạy lại truy vấn thì thấy xuất hiện thêm các dòng mới.\n\n"
            "2. Bốn cấp độ cô lập theo thứ tự từ thấp đến cao:\n"
            "- Read Uncommitted: Cho phép Dirty Read, Non-repeatable Read, Phantom Read (hiệu năng cao nhất, an toàn thấp nhất).\n"
            "- Read Committed: Ngăn chặn Dirty Read; vẫn bị Non-repeatable Read và Phantom Read (mức mặc định của PostgreSQL, Oracle, SQL Server).\n"
            "- Repeatable Read: Ngăn chặn Dirty Read và Non-repeatable Read; vẫn có thể bị Phantom Read (mức mặc định của MySQL InnoDB).\n"
            "- Serializable: Ngăn chặn hoàn toàn cả 3 hiện tượng Dirty Read, Non-repeatable Read và Phantom Read (tương đương chạy tuần tự, chi phí khóa cao nhất)."
        ),
        "bullet_types": [
            "- Read Uncommitted: Cho phép tất cả dị thường (chỉ dùng khi không quan tâm tính đúng đắn).",
            "- Read Committed: Chặn Dirty Read, phổ biến trong hệ thống OLTP.",
            "- Repeatable Read: Chặn Dirty Read và Non-repeatable Read.",
            "- Serializable: Mức cao nhất, chặn toàn bộ dị thường đọc.",
        ],
        "example": (
            "Trong Read Committed, lệnh SELECT lặp lại có thể thấy giá trị cột đã đổi nếu transaction khác vừa commit UPDATE; "
            "trong Repeatable Read, giá trị cột được giữ nguyên như snapshot ban đầu."
        ),
    },
    {
        "id": "db_qa_12",
        "topic": "Các loại SQL JOIN (INNER, LEFT, RIGHT, FULL OUTER JOIN)",
        "category": "sql_joins",
        "canonical_terms": [
            "sql join",
            "inner join",
            "left join",
            "right join",
            "full outer join",
            "kết nối bảng",
        ],
        "question": "Phân biệt các loại SQL JOIN: INNER JOIN, LEFT JOIN, RIGHT JOIN và FULL OUTER JOIN?",
        "intent": "sql_join_classification_and_syntax",
        "expected_action": "compare_all_sql_joins",
        "grounded_answer": (
            "SQL JOIN được sử dụng để kết hợp các dòng từ hai hoặc nhiều bảng dựa trên điều kiện liên kết (thường là giữa Khóa ngoại và Khóa chính):\n\n"
            "1. INNER JOIN (Kết nối trong):\n"
            "Chỉ trả về các dòng có giá trị khớp nhau ở CẢ HAI bảng thỏa mãn điều kiện ON. Các dòng không có cặp tương ứng ở bảng kia sẽ bị loại bỏ hoàn toàn.\n\n"
            "2. LEFT JOIN / LEFT OUTER JOIN (Kết nối ngoài bên trái):\n"
            "Trả về TẤT CẢ các dòng từ bảng bên trái (Left table), cùng với các dòng khớp từ bảng bên phải. Nếu không có dòng khớp ở bảng phải, các cột của bảng phải sẽ nhận giá trị NULL.\n\n"
            "3. RIGHT JOIN / RIGHT OUTER JOIN (Kết nối ngoài bên phải):\n"
            "Trả về TẤT CẢ các dòng từ bảng bên phải (Right table), cùng với các dòng khớp từ bảng bên trái. Nếu không có dòng khớp ở bảng trái, các cột của bảng trái sẽ nhận giá trị NULL.\n\n"
            "4. FULL JOIN / FULL OUTER JOIN (Kết nối ngoài toàn phần):\n"
            "Trả về TẤT CẢ các dòng khi có sự xuất hiện ở bảng trái HOẶC bảng phải. Điền giá trị NULL cho các cột của bảng thiếu dữ liệu ghép cặp."
        ),
        "bullet_types": [
            "- INNER JOIN: Giao của 2 bảng (chỉ lấy phần thỏa mãn điều kiện ở cả hai bên).",
            "- LEFT JOIN: Lấy toàn bộ bảng trái, điền NULL nếu bảng phải không có dòng khớp.",
            "- RIGHT JOIN: Lấy toàn bộ bảng phải, điền NULL nếu bảng trái không có dòng khớp.",
            "- FULL OUTER JOIN: Hợp của cả 2 bảng, điền NULL cho bên thiếu.",
        ],
        "example": (
            "SELECT nv.Ten, pb.TenPhong FROM NhanVien nv "
            "LEFT JOIN PhongBan pb ON nv.MaPhong = pb.MaPhong; "
            "-- Nhân viên chưa được gán phòng ban vẫn hiển thị với PhongBan NULL."
        ),
    },
    {
        "id": "db_qa_13",
        "topic": "CROSS JOIN (Tích Descartes) và SELF JOIN (Tự kết nối)",
        "category": "sql_joins",
        "canonical_terms": [
            "cross join",
            "self join",
            "tích descartes",
            "cartesian product",
            "tự kết nối",
        ],
        "question": "Thế nào là CROSS JOIN và SELF JOIN trong SQL? Khi nào nên sử dụng chúng?",
        "intent": "special_join_types_and_use_cases",
        "expected_action": "explain_cross_join_and_self_join",
        "grounded_answer": (
            "CROSS JOIN và SELF JOIN là hai dạng kết nối đặc biệt trong SQL:\n\n"
            "1. CROSS JOIN (Tích Descartes / Cartesian Product):\n"
            "- Cơ chế: Ghép từng dòng của bảng thứ nhất với MỌI dòng của bảng thứ hai mà không cần bất kỳ điều kiện kết nối ON nào.\n"
            "- Kích thước kết quả: Nếu bảng A có N dòng và bảng B có M dòng, kết quả CROSS JOIN sẽ có chính xác N * M dòng.\n"
            "- Ứng dụng thực tế: Sinh dữ liệu tổ hợp (ví dụ ghép mọi kích cỡ áo {S, M, L} với mọi màu sắc {Đỏ, Xanh, Vàng}), "
            "hoặc tạo ma trận ngày tháng cho báo cáo thống kê.\n\n"
            "2. SELF JOIN (Tự kết nối một bảng với chính nó):\n"
            "- Cơ chế: Thực hiện phép JOIN một bảng với chính bản thân nó bằng cách sử dụng hai bí danh (alias) khác nhau cho cùng một bảng.\n"
            "- Ứng dụng thực tế:\n"
            "  + Dữ liệu dạng cây phân cấp (Hierarchical data): Tìm người quản lý của từng nhân viên khi bảng NhanVien có cột MaQuanLy trỏ về MaNV của cùng bảng.\n"
            "  + So sánh các dòng trong cùng một bảng: Tìm các cặp bản ghi có cùng một thuộc tính (như hai sinh viên sống cùng thành phố)."
        ),
        "bullet_types": [
            "- CROSS JOIN: Tích Descartes, không cần ON, số dòng trả về = N * M.",
            "- SELF JOIN: Bảng kết nối với chính nó thông qua bí danh (alias).",
            "- Ứng dụng: CROSS JOIN để sinh tổ hợp; SELF JOIN để duyệt cây phân cấp quản lý / cha con.",
        ],
        "example": (
            "SELF JOIN tìm tên quản lý:\n"
            "SELECT e.HoTen AS NhanVien, m.HoTen AS QuanLy "
            "FROM NhanVien e LEFT JOIN NhanVien m ON e.MaQuanLy = m.MaNV;"
        ),
    },
]

# Alias to guarantee compatibility with callers expecting either identifier
DATABASE_FEW_SHOT_QA_EXAMPLES = DATABASE_FEW_SHOT_EXAMPLES
