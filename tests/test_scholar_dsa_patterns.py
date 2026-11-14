"""
Unit Tests for Data Structures & Algorithms (DSA) Semantic Patterns.
Tests bidirectional synonym mapping, domain query expansion, algorithmic complexity tables,
and few-shot QA datasets for academic self-study.

100% Offline, Pure Python stdlib, Zero External Dependencies.
"""

import pytest
from src.self_study.datasets.data_structures_algorithms_patterns import (
    strip_accents,
    normalize_term,
    get_dsa_synonyms,
    expand_dsa_query_terms,
    get_dsa_cluster_name,
    DsaThesaurusRegistry,
    DSA_SYNONYM_CLUSTERS,
    DSA_FEW_SHOT_EXAMPLES,
    DSA_FEW_SHOT_QA_EXAMPLES,
    DSA_COMPLEXITY_REFERENCE,
)


class TestDsaTextNormalization:
    """Verifies text normalization and tone mark removal on DSA terminology."""

    def test_strip_accents_vietnamese(self):
        assert strip_accents("Độ phức tạp Big-O") == "Do phuc tap Big-O"
        assert strip_accents("Cây nhị phân tìm kiếm") == "Cay nhi phan tim kiem"
        assert strip_accents("Quy hoạch động") == "Quy hoach dong"
        assert strip_accents("Bảng băm và đụng độ") == "Bang bam va dung do"
        assert strip_accents("ĐƯỜNG ĐI NGẮN NHẤT") == "DUONG DI NGAN NHAT"

    def test_strip_accents_empty_and_ascii(self):
        assert strip_accents("") == ""
        assert strip_accents("quicksort mergesort") == "quicksort mergesort"
        assert strip_accents("O(n log n)") == "O(n log n)"

    def test_normalize_term(self):
        assert normalize_term("   CÂY   Nhị  Phân   ") == "cây nhị phân"
        assert normalize_term("\n\tQuicksort\tPartition\n") == "quicksort partition"
        assert normalize_term("") == ""


class TestBidirectionalDsaSynonyms:
    """Verifies bidirectional synonym mapping across core DSA domains."""

    def test_big_o_complexity_mapping(self):
        terms = ["độ phức tạp big-o", "time complexity", "space complexity", "ký hiệu big-o"]
        for term in terms:
            syns = get_dsa_synonyms(term)
            assert len(syns) > 0, f"Expected synonyms for '{term}'"
            for other in terms:
                assert other in syns, f"'{other}' not found in synonyms of '{term}'"

    def test_array_and_linked_list_mapping(self):
        syns_arr = get_dsa_synonyms("mảng")
        syns_ll = get_dsa_synonyms("danh sách liên kết")
        syns_sll = get_dsa_synonyms("singly linked list")

        assert "array" in syns_arr
        assert "linked list" in syns_ll
        assert "danh sách liên kết đơn" in syns_sll
        assert "mảng" in syns_sll

    def test_stack_and_queue_mapping(self):
        syns_stack = get_dsa_synonyms("ngăn xếp")
        syns_queue = get_dsa_synonyms("hàng đợi")
        syns_lifo = get_dsa_synonyms("lifo")
        syns_fifo = get_dsa_synonyms("fifo")

        assert "stack" in syns_stack
        assert "queue" in syns_queue
        assert "push" in syns_lifo
        assert "pop" in syns_fifo
        assert "enqueue" in syns_queue
        assert "dequeue" in syns_stack

    def test_binary_search_tree_mapping(self):
        syns_bst = get_dsa_synonyms("bst")
        syns_tree = get_dsa_synonyms("cây nhị phân")
        syns_inorder = get_dsa_synonyms("inorder")

        assert "binary search tree" in syns_bst
        assert "cây tìm kiếm nhị phân" in syns_tree
        assert "duyệt trung thứ tự" in syns_inorder

    def test_avl_balanced_tree_mapping(self):
        syns_avl = get_dsa_synonyms("cây avl")
        syns_bf = get_dsa_synonyms("balance factor")
        syns_rot = get_dsa_synonyms("quay cây")

        assert "avl tree" in syns_avl
        assert "hệ số cân bằng" in syns_bf
        assert "left rotation" in syns_rot
        assert "right rotation" in syns_rot

    def test_hash_table_mapping(self):
        syns_hash = get_dsa_synonyms("bảng băm")
        syns_col = get_dsa_synonyms("xung đột băm")
        syns_chaining = get_dsa_synonyms("chaining")
        syns_open = get_dsa_synonyms("open addressing")

        assert "hash table" in syns_hash
        assert "hash map" in syns_hash
        assert "đụng độ băm" in syns_col
        assert "phương pháp nối kết" in syns_chaining
        assert "dò tuyến tính" in syns_open
        assert "load factor" in syns_hash

    def test_graph_traversal_mapping(self):
        syns_bfs = get_dsa_synonyms("bfs")
        syns_dfs = get_dsa_synonyms("dfs")
        syns_graph = get_dsa_synonyms("đồ thị")

        assert "duyệt theo chiều rộng" in syns_bfs
        assert "duyệt theo chiều sâu" in syns_dfs
        assert "ma trận kề" in syns_graph
        assert "danh sách kề" in syns_graph

    def test_dijkstra_shortest_path_mapping(self):
        syns_dijkstra = get_dsa_synonyms("dijkstra")
        syns_path = get_dsa_synonyms("đường đi ngắn nhất")

        assert "thuật toán dijkstra" in syns_dijkstra
        assert "shortest path" in syns_path
        assert "thư giãn cạnh" in syns_dijkstra

    def test_dynamic_programming_mapping(self):
        syns_dp = get_dsa_synonyms("quy hoạch động")
        syns_memo = get_dsa_synonyms("memoization")
        syns_tab = get_dsa_synonyms("tabulation")

        assert "dynamic programming" in syns_dp
        assert "dp" in syns_dp
        assert "bài toán con gối nhau" in syns_memo
        assert "cấu trúc con tối ưu" in syns_tab
        assert "0/1 knapsack" in syns_dp

    def test_sorting_quicksort_and_mergesort_mapping(self):
        syns_quick = get_dsa_synonyms("quicksort")
        syns_merge = get_dsa_synonyms("mergesort")

        assert "sắp xếp nhanh" in syns_quick
        assert "pivot" in syns_quick
        assert "phân hoạch" in syns_quick

        assert "sắp xếp trộn" in syns_merge
        assert "hợp nhất" in syns_merge
        assert "stable sort" in syns_merge

    def test_minimum_spanning_tree_mapping(self):
        syns_mst = get_dsa_synonyms("cây khung nhỏ nhất")
        assert "minimum spanning tree" in syns_mst
        assert "thuật toán kruskal" in syns_mst
        assert "thuật toán prim" in syns_mst
        assert "union find" in syns_mst

    def test_heap_mapping(self):
        syns_heap = get_dsa_synonyms("heap")
        assert "min heap" in syns_heap
        assert "max heap" in syns_heap
        assert "heapify" in syns_heap
        assert "heapsort" in syns_heap

    def test_unaccented_lookups(self):
        # Tone-insensitive queries must find canonical accented terms
        assert "quy hoạch động" in get_dsa_synonyms("quy hoach dong")
        assert "cây avl" in get_dsa_synonyms("cay avl")
        assert "bảng băm" in get_dsa_synonyms("bang bam")
        assert "ngăn xếp" in get_dsa_synonyms("ngan xep")
        assert "đồ thị" in get_dsa_synonyms("do thi")
        assert "độ phức tạp big-o" in get_dsa_synonyms("do phuc tap big-o")

    def test_unknown_term_returns_empty(self):
        assert get_dsa_synonyms("random_unknown_algorithm_xyz") == []
        assert get_dsa_synonyms("") == []


class TestExpandDsaQueryTerms:
    """Verifies conversational query keyword expansion for DSA concepts."""

    def test_single_term_expansion(self):
        expanded = expand_dsa_query_terms("quicksort")
        assert "quicksort" in expanded
        assert "sắp xếp nhanh" in expanded
        assert "pivot" in expanded
        assert "phân hoạch" in expanded

    def test_multi_topic_conversational_query(self):
        query = "Phân biệt ngăn xếp stack và hàng đợi queue trong giải thuật duyệt BFS đồ thị"
        expanded = expand_dsa_query_terms(query)

        # Must capture stack_and_queue and graph_traversal clusters
        assert any("ngăn xếp" in t or "stack" in t for t in expanded)
        assert any("hàng đợi" in t or "queue" in t for t in expanded)
        assert any("bfs" in t or "đồ thị" in t for t in expanded)

    def test_tree_and_avl_comparison_query(self):
        query = "Khi nào cây tìm kiếm nhị phân bst cần cân bằng thành cây avl?"
        expanded = expand_dsa_query_terms(query)

        assert any("cây tìm kiếm nhị phân" in t or "bst" in t for t in expanded)
        assert any("cây avl" in t or "hệ số cân bằng" in t for t in expanded)

    def test_sorting_comparison_query(self):
        query = "So sánh độ phức tạp giữa mergesort và quicksort"
        expanded = expand_dsa_query_terms(query)

        assert any("mergesort" in t or "sắp xếp trộn" in t for t in expanded)
        assert any("quicksort" in t or "sắp xếp nhanh" in t for t in expanded)

    def test_unaccented_queries(self):
        queries = [
            ("bang bam xu ly xung dot chaining nhu the nao", "bảng băm"),
            ("quy hoach dong memoization va tabulation", "quy hoạch động"),
            ("thuat toan dijkstra tim duong di ngan nhat", "dijkstra"),
            ("do phuc tap big-o worst case", "độ phức tạp big-o"),
        ]
        for q, expected in queries:
            res = expand_dsa_query_terms(q)
            assert any(expected in item for item in res), (
                f"Query '{q}' failed to expand into '{expected}'"
            )

    def test_no_duplicate_terms_in_expansion(self):
        expanded = expand_dsa_query_terms(
            "quicksort quicksort sắp xếp nhanh quick sort pivot phân hoạch"
        )
        assert len(expanded) == len(set(normalize_term(t) for t in expanded))

    def test_empty_and_irrelevant_queries(self):
        assert expand_dsa_query_terms("") == []
        assert expand_dsa_query_terms("      ") == []
        assert expand_dsa_query_terms("trời hôm nay nhiều mây có mưa rào nhẹ") == []


class TestDsaFewShotQaDataset:
    """Verifies schema integrity, topic completeness, and academic quality of few-shot QA dataset."""

    def test_alias_equivalence(self):
        assert DSA_FEW_SHOT_EXAMPLES is DSA_FEW_SHOT_QA_EXAMPLES
        assert len(DSA_FEW_SHOT_EXAMPLES) >= 12

    def test_mandatory_fields_presence(self):
        required_keys = {
            "id",
            "topic",
            "category",
            "canonical_terms",
            "question",
            "intent",
            "expected_action",
            "grounded_answer",
        }
        for item in DSA_FEW_SHOT_EXAMPLES:
            missing = required_keys - set(item.keys())
            assert not missing, f"Item {item.get('id')} is missing keys: {missing}"
            assert len(item["canonical_terms"]) >= 2
            assert len(item["grounded_answer"]) > 120, (
                f"Grounded answer in {item['id']} is too terse"
            )

    def test_topic_breadth_coverage(self):
        topics = {item["topic"] for item in DSA_FEW_SHOT_EXAMPLES}

        # Verify Big-O, Array/LL, Stack/Queue, BST, AVL, Hash, BFS/DFS, Dijkstra, DP, QuickSort/MergeSort, MST, Heap
        assert any("Big-O" in t for t in topics)
        assert any("Mảng" in t and "Danh sách liên kết" in t for t in topics)
        assert any("Ngăn xếp" in t and "Hàng đợi" in t for t in topics)
        assert any("BST" in t for t in topics)
        assert any("AVL" in t for t in topics)
        assert any("Bảng băm" in t for t in topics)
        assert any("BFS" in t and "DFS" in t for t in topics)
        assert any("Dijkstra" in t for t in topics)
        assert any("Quy hoạch động" in t for t in topics)
        assert any("QuickSort" in t and "MergeSort" in t for t in topics)
        assert any("MST" in t or "Kruskal" in t for t in topics)
        assert any("Heap" in t for t in topics)

    def test_complexity_reference_table(self):
        assert "QuickSort" in DSA_COMPLEXITY_REFERENCE
        assert "MergeSort" in DSA_COMPLEXITY_REFERENCE
        assert "Hash Table (Search)" in DSA_COMPLEXITY_REFERENCE
        assert "AVL Tree (Search)" in DSA_COMPLEXITY_REFERENCE

        qs = DSA_COMPLEXITY_REFERENCE["QuickSort"]
        assert qs["time_worst"] == "O(n^2)"
        assert qs["time_avg"] == "O(n log n)"

        ms = DSA_COMPLEXITY_REFERENCE["MergeSort"]
        assert ms["time_worst"] == "O(n log n)"
        assert ms["space_worst"] == "O(n)"


class TestDsaThesaurusRegistryExtensibility:
    """Verifies standalone instantiation and customization of DsaThesaurusRegistry."""

    def test_custom_cluster_override(self):
        custom_clusters = {
            "trie_structure": ["cây tiền tố", "trie", "prefix tree", "trie node"],
            "suffix_tree": ["cây hậu tố", "suffix tree", "suffix array"],
        }
        reg = DsaThesaurusRegistry(clusters=custom_clusters)

        assert reg.get_cluster_name("cây tiền tố") == "trie_structure"
        assert "prefix tree" in reg.get_synonyms("trie")
        assert reg.get_synonyms("quicksort") == []

    def test_cluster_name_resolution(self):
        assert get_dsa_cluster_name("quicksort") == "sorting_quicksort"
        assert get_dsa_cluster_name("sap xep tron") == "sorting_mergesort"
        assert get_dsa_cluster_name("dijkstra") == "shortest_path_dijkstra"
        assert get_dsa_cluster_name("non_existent_key") is None
