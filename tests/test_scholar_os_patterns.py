"""
Unit Tests for Operating Systems Semantic Patterns and Knowledge Base.
Tests text normalization, bidirectional synonym mappings, query expansion,
and few-shot QA dataset integrity for Operating Systems curriculum.
100% Offline, Pure Python stdlib, Zero External Dependencies.
"""

import pytest
from src.self_study.datasets.operating_systems_patterns import (
    strip_accents,
    normalize_term,
    get_os_synonyms,
    expand_os_query_terms,
    OSThesaurusRegistry,
    OS_SYNONYM_CLUSTERS,
    OS_FEW_SHOT_EXAMPLES,
)


class TestOSTextNormalization:
    """Verifies Vietnamese diacritics stripping and term normalization."""

    def test_strip_accents_vietnamese(self):
        assert strip_accents("Tiến trình và Luồng") == "Tien trinh va Luong"
        assert strip_accents("Lập lịch CPU xoay vòng") == "Lap lich CPU xoay vong"
        assert strip_accents("Bế tắc Deadlock 4 điều kiện") == "Be tac Deadlock 4 dieu kien"
        assert strip_accents("Bộ nhớ ảo và phân trang") == "Bo nho ao va phan trang"
        assert strip_accents("Đèn hiệu Semaphore") == "Den hieu Semaphore"

    def test_strip_accents_edge_cases(self):
        assert strip_accents("") == ""
        assert strip_accents("round robin") == "round robin"
        assert strip_accents("12345!@#$%^&*()") == "12345!@#$%^&*()"

    def test_normalize_term(self):
        assert normalize_term("   Tiến   Trình   ") == "tiến trình"
        assert normalize_term("\n\tROUND   ROBIN\t\n") == "round robin"
        assert normalize_term("") == ""


class TestOSBidirectionalSynonyms:
    """Verifies bidirectional mapping across required Operating Systems topics."""

    def test_process_and_thread_synonyms(self):
        syns_process = get_os_synonyms("tiến trình")
        syns_thread = get_os_synonyms("luồng")
        syns_pcb = get_os_synonyms("pcb")
        syns_switch = get_os_synonyms("context switch")

        assert "process" in syns_process
        assert "thread" in syns_process
        assert "tiến trình" in syns_thread
        assert "tcb" in syns_pcb
        assert "chuyển ngữ cảnh" in syns_switch
        assert "process control block" in syns_pcb

    def test_process_lifecycle_states(self):
        syns_state = get_os_synonyms("trạng thái tiến trình")
        assert "ready" in syns_state
        assert "running" in syns_state
        assert "waiting" in syns_state
        assert "terminated" in syns_state
        assert "blocked" in syns_state

    def test_cpu_scheduling_fcfs(self):
        syns_fcfs = get_os_synonyms("fcfs")
        syns_first_come = get_os_synonyms("đến trước phục vụ trước")

        assert "first come first served" in syns_fcfs
        assert "convoy effect" in syns_fcfs
        assert "hiệu ứng đoàn tàu" in syns_fcfs
        assert "fcfs" in syns_first_come

    def test_cpu_scheduling_sjf_srtf(self):
        syns_sjf = get_os_synonyms("sjf")
        syns_srtf = get_os_synonyms("srtf")

        assert "shortest job first" in syns_sjf
        assert "tiến trình ngắn nhất trước" in syns_sjf
        assert "shortest remaining time first" in syns_srtf
        assert "sjf" in syns_srtf

    def test_cpu_scheduling_round_robin(self):
        syns_rr = get_os_synonyms("round robin")
        syns_quantum = get_os_synonyms("time quantum")

        assert "rr" in syns_rr
        assert "lát cắt thời gian" in syns_rr
        assert "time slice" in syns_rr
        assert "lập lịch xoay vòng" in syns_rr
        assert "round robin" in syns_quantum

    def test_deadlock_four_conditions(self):
        syns_deadlock = get_os_synonyms("deadlock")
        syns_be_tac = get_os_synonyms("bế tắc")
        syns_mutex = get_os_synonyms("mutual exclusion")
        syns_hold = get_os_synonyms("hold and wait")
        syns_circ = get_os_synonyms("circular wait")

        assert "bế tắc" in syns_deadlock
        assert "deadlock" in syns_be_tac
        assert "4 điều kiện coffman" in syns_deadlock
        assert "loại trừ lẫn nhau" in syns_mutex
        assert "giữ và chờ" in syns_hold
        assert "chờ đợi vòng tròn" in syns_circ
        assert "no preemption" in syns_deadlock

    def test_deadlock_handling_and_banker(self):
        syns_banker = get_os_synonyms("thuật toán banker")
        syns_avoid = get_os_synonyms("deadlock avoidance")

        assert "banker's algorithm" in syns_banker
        assert "safe state" in syns_banker
        assert "trạng thái an toàn" in syns_banker
        assert "thuật toán banker" in syns_avoid

    def test_virtual_memory_and_page_fault(self):
        syns_vm = get_os_synonyms("bộ nhớ ảo")
        syns_fault = get_os_synonyms("page fault")
        syns_thrash = get_os_synonyms("thrashing")

        assert "virtual memory" in syns_vm
        assert "demand paging" in syns_vm
        assert "lỗi trang" in syns_fault
        assert "ngắt trang" in syns_fault
        assert "trì trệ bộ nhớ" in syns_thrash
        assert "đảo trang liên tục" in syns_thrash

    def test_paging_and_segmentation(self):
        syns_page = get_os_synonyms("phân trang")
        syns_seg = get_os_synonyms("phân đoạn")

        assert "paging" in syns_page
        assert "khung trang" in syns_page
        assert "internal fragmentation" in syns_page
        assert "tlb" in syns_page
        assert "segmentation" in syns_seg
        assert "external fragmentation" in syns_seg
        assert "bảng phân đoạn" in syns_seg

    def test_page_replacement_algorithms(self):
        syns_replace = get_os_synonyms("thay thế trang")
        syns_lru = get_os_synonyms("lru")
        syns_belady = get_os_synonyms("belady anomaly")

        assert "fifo page replacement" in syns_replace
        assert "least recently used" in syns_lru
        assert "optimal page replacement" in syns_replace
        assert "bất thường belady" in syns_belady

    def test_synchronization_mutex_and_semaphore(self):
        syns_sync = get_os_synonyms("miền găng")
        syns_sem = get_os_synonyms("semaphore")
        syns_mutex = get_os_synonyms("mutex")

        assert "critical section" in syns_sync
        assert "race condition" in syns_sync
        assert "loại trừ lẫn nhau" in syns_sync
        assert "mutex lock" in syns_mutex
        assert "counting semaphore" in syns_sem
        assert "binary semaphore" in syns_sem
        assert "wait" in syns_sem
        assert "signal" in syns_sem

    def test_classic_concurrency_problems(self):
        syns_prod = get_os_synonyms("producer consumer")
        syns_philosophers = get_os_synonyms("dining philosophers")

        assert "nhà sản xuất người tiêu dùng" in syns_prod
        assert "bounded buffer" in syns_prod
        assert "bữa ăn triết gia" in syns_philosophers
        assert "readers writers" in syns_prod

    def test_unaccented_lookup(self):
        # Searching with unaccented terms correctly resolves to the accented cluster
        syns_tien_trinh = get_os_synonyms("tien trinh")
        assert "tiến trình" in syns_tien_trinh
        assert "process" in syns_tien_trinh

        syns_be_tac = get_os_synonyms("be tac")
        assert "deadlock" in syns_be_tac
        assert "mutual exclusion" in syns_be_tac

        syns_phan_trang = get_os_synonyms("phan trang")
        assert "khung trang" in syns_phan_trang
        assert "internal fragmentation" in syns_phan_trang

        syns_dien_hieu = get_os_synonyms("semaphore dem")
        assert "counting semaphore" in syns_dien_hieu

    def test_unknown_term_returns_empty(self):
        assert get_os_synonyms("thuật toán lượng tử shor xyz") == []
        assert get_os_synonyms("") == []
        assert get_os_synonyms("   ") == []


class TestOSQueryExpansion:
    """Verifies domain query expansion into relevant technical keywords."""

    def test_expand_query_round_robin(self):
        query = "Thuật toán Round Robin phân bổ time quantum như thế nào?"
        expanded = expand_os_query_terms(query)

        assert "round robin" in expanded or "time quantum" in expanded
        assert any("lát cắt thời gian" in term for term in expanded)
        assert any("rr" in term for term in expanded)

    def test_expand_query_deadlock(self):
        query = "Giải thích hiện tượng bế tắc deadlock và 4 điều kiện Coffman"
        expanded = expand_os_query_terms(query)

        assert any("deadlock" in term for term in expanded)
        assert any("mutual exclusion" in term for term in expanded)
        assert any("hold and wait" in term for term in expanded)
        assert any("circular wait" in term for term in expanded)

    def test_expand_query_unaccented(self):
        query = "so sanh phan trang va phan doan trong bo nho ao"
        expanded = expand_os_query_terms(query)

        assert any("phân trang" in term for term in expanded)
        assert any("phân đoạn" in term for term in expanded)
        assert any("bộ nhớ ảo" in term for term in expanded)

    def test_expand_empty_or_whitespace_query(self):
        assert expand_os_query_terms("") == []
        assert expand_os_query_terms("   \t\n ") == []


class TestOSFewShotExamplesIntegrity:
    """Verifies few-shot QA examples completeness, schema compliance and academic rigor."""

    REQUIRED_KEYS = {
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

    def test_few_shot_dataset_not_empty(self):
        assert len(OS_FEW_SHOT_EXAMPLES) >= 10

    def test_schema_compliance_all_items(self):
        seen_ids = set()
        for idx, item in enumerate(OS_FEW_SHOT_EXAMPLES):
            assert isinstance(item, dict), f"Item {idx} is not a dict"
            for key in self.REQUIRED_KEYS:
                assert key in item, f"Item {item.get('id', idx)} missing required key: {key}"

            # Ensure unique IDs
            qa_id = item["id"]
            assert qa_id not in seen_ids, f"Duplicate ID: {qa_id}"
            seen_ids.add(qa_id)

            # Ensure non-empty fields
            assert len(item["canonical_terms"]) > 0
            assert len(item["question"].strip()) > 10
            assert len(item["grounded_answer"].strip()) > 100
            assert len(item["bullet_types"]) >= 3
            assert len(item["example"].strip()) > 10

    def test_coverage_of_mandatory_topics(self):
        """Verifies presence of all core required topics from brief."""
        all_text = " ".join(
            (item["topic"] + " " + item["question"] + " " + " ".join(item["canonical_terms"])).lower()
            for item in OS_FEW_SHOT_EXAMPLES
        )

        assert "tiến trình" in all_text and "luồng" in all_text
        assert "lập lịch" in all_text or "scheduling" in all_text
        assert "fcfs" in all_text or "sjf" in all_text or "round robin" in all_text
        assert "deadlock" in all_text or "bế tắc" in all_text
        assert "bộ nhớ ảo" in all_text or "virtual memory" in all_text
        assert "phân trang" in all_text and "phân đoạn" in all_text
        assert "semaphore" in all_text or "mutex" in all_text


class TestCustomRegistry:
    """Tests custom instantiation and isolation of the registry."""

    def test_custom_cluster_registry(self):
        custom_clusters = {
            "kernel_types": ["microkernel", "monolithic kernel", "nhân nguyên khối", "vi nhân"],
        }
        reg = OSThesaurusRegistry(clusters=custom_clusters)
        assert "microkernel" in reg.get_synonyms("vi nhân")
        assert "monolithic kernel" in reg.get_synonyms("nhân nguyên khối")
        assert reg.get_synonyms("deadlock") == []
