"""
Unit Tests for Relational Database & SQL Semantic Patterns.
Validates bidirectional synonym mapping, domain query expansion, and few-shot QA datasets
for ACID, Primary/Foreign Keys, Normalization (1NF/2NF/3NF/BCNF), B-Tree Index,
Transactions/Concurrency, and SQL JOINs.
100% Offline, Pure Python stdlib, Zero External Dependencies.
"""

import pytest
from src.self_study.datasets.database_and_sql_patterns import (
    strip_accents,
    normalize_term,
    get_database_synonyms,
    expand_database_query_terms,
    is_sql_or_relational_query,
    DatabaseThesaurusRegistry,
    DATABASE_SYNONYM_CLUSTERS,
    DATABASE_FEW_SHOT_EXAMPLES,
    DATABASE_FEW_SHOT_QA_EXAMPLES,
)


class TestDatabaseTextNormalization:
    """Tests Vietnamese text normalization and diacritic removal."""

    def test_strip_accents_vietnamese(self):
        assert strip_accents("Khóa chính và khóa ngoại") == "Khoa chinh va khoa ngoai"
        assert strip_accents("Chuẩn hóa dạng 1NF, 2NF, 3NF, BCNF") == "Chuan hoa dang 1NF, 2NF, 3NF, BCNF"
        assert strip_accents("Tính nguyên tử, nhất quán, cô lập, bền vững") == "Tinh nguyen tu, nhat quan, co lap, ben vung"
        assert strip_accents("ĐỘ CÔ LẬP GIAO DỊCH") == "DO CO LAP GIAO DICH"

    def test_strip_accents_empty_and_ascii(self):
        assert strip_accents("") == ""
        assert strip_accents("SELECT * FROM users") == "SELECT * FROM users"
        assert strip_accents("B-Tree Indexing 12345!@#") == "B-Tree Indexing 12345!@#"

    def test_normalize_term(self):
        assert normalize_term("   KHÓA   CHÍNH   ") == "khóa chính"
        assert normalize_term("\n\tINNER   JOIN\t") == "inner join"
        assert normalize_term("  b+tree   index  ") == "b+tree index"


class TestDatabaseSynonymMapping:
    """Verifies bidirectional synonym mapping across core Relational Database topics."""

    def test_acid_cluster_mapping(self):
        syns_acid = get_database_synonyms("acid")
        assert len(syns_acid) > 0
        assert "thuộc tính acid" in syns_acid
        assert "acid properties" in syns_acid

        # Atomicity mapping
        syns_atom = get_database_synonyms("tính nguyên tử")
        assert "atomicity" in syns_atom
        assert "all-or-nothing" in syns_atom

        # Consistency mapping
        syns_cons = get_database_synonyms("tính nhất quán")
        assert "consistency" in syns_cons
        assert "ràng buộc toàn vẹn" in syns_cons

        # Isolation mapping
        syns_iso = get_database_synonyms("tính cô lập")
        assert "isolation" in syns_iso
        assert "mức cô lập" in syns_iso

        # Durability mapping
        syns_dur = get_database_synonyms("tính bền vững")
        assert "durability" in syns_dur
        assert "khả năng phục hồi dữ liệu" in syns_dur

    def test_primary_and_foreign_keys_mapping(self):
        syns_pk = get_database_synonyms("khóa chính")
        assert "primary key" in syns_pk
        assert "pk" in syns_pk

        syns_fk = get_database_synonyms("khóa ngoại")
        assert "foreign key" in syns_fk
        assert "toàn vẹn tham chiếu" in syns_fk
        assert "referential integrity" in syns_fk

        syns_ck = get_database_synonyms("khóa ứng viên")
        assert "candidate key" in syns_ck
        assert "siêu khóa tối thiểu" in syns_ck

        syns_sk = get_database_synonyms("siêu khóa")
        assert "super key" in syns_sk
        assert "superkey" in syns_sk

    def test_normalization_forms_mapping(self):
        # 1NF
        syns_1nf = get_database_synonyms("1nf")
        assert "first normal form" in syns_1nf
        assert "dạng chuẩn 1" in syns_1nf
        assert "thuộc tính nguyên tử" in syns_1nf

        # 2NF
        syns_2nf = get_database_synonyms("2nf")
        assert "second normal form" in syns_2nf
        assert "dạng chuẩn 2" in syns_2nf
        assert "phụ thuộc hàm đầy đủ" in syns_2nf

        # 3NF
        syns_3nf = get_database_synonyms("3nf")
        assert "third normal form" in syns_3nf
        assert "dạng chuẩn 3" in syns_3nf
        assert "loại bỏ phụ thuộc bắc cầu" in syns_3nf

        # BCNF
        syns_bcnf = get_database_synonyms("bcnf")
        assert "boyce-codd normal form" in syns_bcnf
        assert "dạng chuẩn boyce-codd" in syns_bcnf

        # Functional Dependency
        syns_fd = get_database_synonyms("phụ thuộc hàm")
        assert "functional dependency" in syns_fd
        assert "fd" in syns_fd

    def test_btree_and_indexes_mapping(self):
        syns_btree = get_database_synonyms("b-tree")
        assert "chỉ mục b-tree" in syns_btree
        assert "b-tree index" in syns_btree

        syns_bplus = get_database_synonyms("b+tree")
        assert "chỉ mục b+tree" in syns_bplus
        assert "b plus tree" in syns_bplus

        syns_clustered = get_database_synonyms("chỉ mục phân cụm")
        assert "clustered index" in syns_clustered

        syns_non_clustered = get_database_synonyms("non-clustered index")
        assert "chỉ mục không phân cụm" in syns_non_clustered
        assert "secondary index" in syns_non_clustered

    def test_transactions_and_concurrency_mapping(self):
        syns_tx = get_database_synonyms("giao dịch")
        assert "transaction" in syns_tx
        assert "quản lý giao dịch" in syns_tx

        syns_ctrl = get_database_synonyms("commit")
        assert "rollback" in syns_ctrl
        assert "xác nhận giao dịch" in syns_ctrl

        syns_cc = get_database_synonyms("kiểm soát tương tranh")
        assert "concurrency control" in syns_cc
        assert "khóa hai pha" in syns_cc
        assert "2pl" in syns_cc
        assert "deadlock" in syns_cc
        assert "khóa chết" in syns_cc

        syns_phenomena = get_database_synonyms("dirty read")
        assert "đọc rác" in syns_phenomena
        assert "non-repeatable read" in syns_phenomena
        assert "phantom read" in syns_phenomena

    def test_sql_joins_mapping(self):
        syns_join = get_database_synonyms("sql join")
        assert "kết nối bảng" in syns_join
        assert "phép kết nối" in syns_join

        syns_inner = get_database_synonyms("inner join")
        assert "kết nối trong" in syns_inner
        assert "equi join" in syns_inner

        syns_left = get_database_synonyms("left join")
        assert "left outer join" in syns_left
        assert "kết nối ngoài bên trái" in syns_left

        syns_right = get_database_synonyms("right join")
        assert "right outer join" in syns_right
        assert "kết nối ngoài bên phải" in syns_right

        syns_full = get_database_synonyms("full outer join")
        assert "full join" in syns_full
        assert "kết nối đầy đủ" in syns_full

        syns_cross = get_database_synonyms("cross join")
        assert "tích descartes" in syns_cross
        assert "cartesian product" in syns_cross

        syns_self = get_database_synonyms("self join")
        assert "tự kết nối" in syns_self

    def test_tone_insensitive_and_unknown_terms(self):
        syns_unaccented = get_database_synonyms("khoa chinh")
        assert "primary key" in syns_unaccented
        assert "khóa chính" in syns_unaccented

        syns_unaccented_acid = get_database_synonyms("tinh nguyen tu")
        assert "atomicity" in syns_unaccented_acid
        assert "tính nguyên tử" in syns_unaccented_acid

        assert get_database_synonyms("unknown_term_xyz_123") == []
        assert get_database_synonyms("") == []


class TestExpandDatabaseQueryTerms:
    """Verifies domain query expansion for student questions and exam revision."""

    def test_acid_query_expansion(self):
        query = "Trình bày 4 thuộc tính ACID trong giao dịch cơ sở dữ liệu"
        expanded = expand_database_query_terms(query)
        assert any("acid" in t for t in expanded)
        assert any("giao dịch" in t or "transaction" in t for t in expanded)

    def test_primary_and_foreign_key_query(self):
        query = "Phân biệt khóa chính và khóa ngoại trong RDBMS"
        expanded = expand_database_query_terms(query)
        assert any("khóa chính" in t or "primary key" in t for t in expanded)
        assert any("khóa ngoại" in t or "foreign key" in t for t in expanded)

    def test_normalization_query(self):
        query = "Điều kiện để một bảng đạt dạng chuẩn 1NF, 2NF, 3NF và BCNF là gì?"
        expanded = expand_database_query_terms(query)
        assert any("1nf" in t for t in expanded)
        assert any("2nf" in t for t in expanded)
        assert any("3nf" in t for t in expanded)
        assert any("bcnf" in t for t in expanded)

    def test_indexing_query(self):
        query = "So sánh chỉ mục B-Tree và B+Tree, clustered index và secondary index"
        expanded = expand_database_query_terms(query)
        assert any("b-tree" in t for t in expanded)
        assert any("b+tree" in t for t in expanded)
        assert any("clustered index" in t or "chỉ mục phân cụm" in t for t in expanded)
        assert any("non-clustered index" in t or "secondary index" in t for t in expanded)

    def test_join_query(self):
        query = "Khác nhau giữa INNER JOIN, LEFT JOIN và CROSS JOIN (tích Descartes)?"
        expanded = expand_database_query_terms(query)
        assert any("inner join" in t or "kết nối trong" in t for t in expanded)
        assert any("left join" in t for t in expanded)
        assert any("cross join" in t or "tích descartes" in t for t in expanded)

    def test_empty_query(self):
        assert expand_database_query_terms("") == []
        assert expand_database_query_terms("   ") == []

    def test_is_sql_or_relational_query(self):
        assert is_sql_or_relational_query("Làm thế nào để tạo chỉ mục b-tree cho bảng?") is True
        assert is_sql_or_relational_query("Giải thích tính chất ACID của transaction") is True
        assert is_sql_or_relational_query("Hôm nay thời tiết thế nào?") is False


class TestDatabaseFewShotQAExamples:
    """Verifies integrity, completeness, and educational rigor of DATABASE_FEW_SHOT_EXAMPLES."""

    def test_examples_exist_and_non_empty(self):
        assert isinstance(DATABASE_FEW_SHOT_EXAMPLES, list)
        assert len(DATABASE_FEW_SHOT_EXAMPLES) >= 12
        # Check alias
        assert DATABASE_FEW_SHOT_QA_EXAMPLES == DATABASE_FEW_SHOT_EXAMPLES

    def test_example_schema_and_keys(self):
        required_keys = {
            "id",
            "topic",
            "category",
            "canonical_terms",
            "question",
            "intent",
            "expected_action",
            "grounded_answer",
            "bullet_types",
            "example",
        }
        seen_ids = set()
        for idx, item in enumerate(DATABASE_FEW_SHOT_EXAMPLES):
            assert isinstance(item, dict), f"Item {idx} is not a dict"
            missing = required_keys - set(item.keys())
            assert not missing, f"Item {item.get('id', idx)} missing keys: {missing}"

            # Check id uniqueness
            item_id = item["id"]
            assert item_id not in seen_ids, f"Duplicate ID: {item_id}"
            seen_ids.add(item_id)

            # Check content validity
            assert len(item["topic"].strip()) > 3
            assert len(item["question"].strip()) > 10
            assert len(item["canonical_terms"]) > 0
            assert len(item["grounded_answer"].strip()) > 50
            assert len(item["bullet_types"]) >= 2
            assert len(item["example"].strip()) > 10

    def test_covers_all_mandatory_curriculum_topics(self):
        all_topics_text = " ".join(
            (item["topic"] + " " + item["question"] + " " + " ".join(item["canonical_terms"])).lower()
            for item in DATABASE_FEW_SHOT_EXAMPLES
        )

        mandatory_tokens = [
            "acid",
            "khóa chính",
            "khóa ngoại",
            "1nf",
            "2nf",
            "3nf",
            "bcnf",
            "b-tree",
            "giao dịch",
            "join",
        ]
        for token in mandatory_tokens:
            assert token in all_topics_text, f"Mandatory curriculum topic '{token}' not found in QA examples"

    def test_custom_registry_injection(self):
        custom_clusters = {
            "custom_cluster": ["term_a", "term_b", "thuật ngữ c"],
        }
        reg = DatabaseThesaurusRegistry(clusters=custom_clusters)
        assert "term_b" in reg.get_synonyms("term_a")
        assert "term_a" in reg.get_synonyms("thuat ngu c")
        assert reg.get_cluster_name("term_b") == "custom_cluster"
