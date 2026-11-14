"""
Domain-Specific Data Structures & Algorithms (DSA) Semantic Patterns.
Provides bidirectional synonym mapping, domain query expansion, algorithmic complexity tables,
and few-shot QA blueprints for authoritative Computer Science & Software Engineering educational documents.

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


# Core Data Structures & Algorithms domain synonym clusters
DSA_SYNONYM_CLUSTERS: Dict[str, List[str]] = {
    "big_o_complexity": [
        "độ phức tạp big-o",
        "độ phức tạp thuật toán",
        "độ phức tạp thời gian",
        "độ phức tạp không gian",
        "big-o",
        "big o",
        "big-o notation",
        "time complexity",
        "space complexity",
        "độ phức tạp tính toán",
        "ký hiệu big-o",
        "o(1)",
        "o(n)",
        "o(log n)",
        "o(n log n)",
        "o(n^2)",
        "o(2^n)",
        "o(n!)",
        "trường hợp xấu nhất",
        "trường hợp tốt nhất",
        "trường hợp trung bình",
        "worst-case",
        "best-case",
        "average-case",
        "asymptotic analysis",
        "đánh giá tiệm cận",
    ],
    "array_and_list": [
        "mảng",
        "mảng tĩnh",
        "mảng động",
        "danh sách liên kết",
        "danh sách liên kết đơn",
        "danh sách liên kết đôi",
        "danh sách liên kết vòng",
        "array",
        "dynamic array",
        "linked list",
        "singly linked list",
        "doubly linked list",
        "circular linked list",
        "node danh sách",
        "nút danh sách",
        "con trỏ next",
        "con trỏ prev",
        "mảng một chiều",
        "mảng hai chiều",
        "danh sách tuyến tính",
    ],
    "stack_and_queue": [
        "ngăn xếp",
        "hàng đợi",
        "stack",
        "queue",
        "lifo",
        "fifo",
        "hàng đợi hai đầu",
        "deque",
        "hàng đợi ưu tiên",
        "priority queue",
        "push",
        "pop",
        "enqueue",
        "dequeue",
        "đỉnh ngăn xếp",
        "đầu hàng đợi",
        "cuối hàng đợi",
        "vào trước ra trước",
        "vào sau ra trước",
    ],
    "binary_tree": [
        "cây nhị phân",
        "cây tìm kiếm nhị phân",
        "cây nhị phân tìm kiếm",
        "binary tree",
        "binary search tree",
        "bst",
        "nút gốc",
        "nút lá",
        "nút con",
        "nút cha",
        "root node",
        "leaf node",
        "duyệt cây",
        "duyệt tiền thứ tự",
        "duyệt trung thứ tự",
        "duyệt hậu thứ tự",
        "duyệt theo mức",
        "preorder",
        "inorder",
        "postorder",
        "level order",
        "chiều cao cây",
        "độ sâu cây",
        "tree traversal",
    ],
    "balanced_avl_tree": [
        "cây avl",
        "cây cân bằng",
        "cây nhị phân cân bằng",
        "cây nhị phân tìm kiếm cân bằng",
        "avl tree",
        "self-balancing bst",
        "self-balancing binary search tree",
        "hệ số cân bằng",
        "balance factor",
        "quay cây",
        "quay trái",
        "quay phải",
        "quay kép",
        "tree rotation",
        "left rotation",
        "right rotation",
        "left-right rotation",
        "right-left rotation",
    ],
    "hash_table": [
        "bảng băm",
        "hàm băm",
        "xung đột băm",
        "đụng độ băm",
        "xử lý xung đột",
        "hash table",
        "hash map",
        "hash function",
        "hash collision",
        "collision resolution",
        "chaining",
        "phương pháp nối kết",
        "địa chỉ mở",
        "open addressing",
        "dò tuyến tính",
        "linear probing",
        "dò bậc hai",
        "quadratic probing",
        "băm kép",
        "double hashing",
        "hệ số tải",
        "load factor",
    ],
    "graph_traversal": [
        "đồ thị",
        "đồ thị có hướng",
        "đồ thị vô hướng",
        "đồ thị có trọng số",
        "ma trận kề",
        "danh sách kề",
        "graph",
        "directed graph",
        "undirected graph",
        "weighted graph",
        "adjacency matrix",
        "adjacency list",
        "duyệt đồ thị",
        "duyệt theo chiều rộng",
        "duyệt theo chiều sâu",
        "bfs",
        "dfs",
        "breadth first search",
        "depth first search",
        "thành phần liên thông",
        "connected components",
    ],
    "shortest_path_dijkstra": [
        "dijkstra",
        "thuật toán dijkstra",
        "đường đi ngắn nhất",
        "đường đi ngắn nhất một nguồn",
        "shortest path",
        "single-source shortest path",
        "dijkstra algorithm",
        "thuật toán bellman-ford",
        "bellman-ford",
        "thuật toán floyd-warshall",
        "floyd-warshall",
        "khoảng cách ngắn nhất",
        "thư giãn cạnh",
        "edge relaxation",
    ],
    "dynamic_programming": [
        "quy hoạch động",
        "dynamic programming",
        "dp",
        "bài toán con gối nhau",
        "overlapping subproblems",
        "cấu trúc con tối ưu",
        "optimal substructure",
        "ghi nhớ",
        "memoization",
        "bảng phương án",
        "tabulation",
        "top-down",
        "bottom-up",
        "bài toán cái ba lô",
        "knapsack",
        "0/1 knapsack",
        "xâu con chung dài nhất",
        "lcs",
        "longest common subsequence",
        "dãy con tăng dài nhất",
        "lis",
        "longest increasing subsequence",
    ],
    "sorting_quicksort": [
        "quicksort",
        "sắp xếp nhanh",
        "quick sort",
        "thuật toán quicksort",
        "thuật toán sắp xếp nhanh",
        "phân hoạch",
        "partition",
        "phần tử chốt",
        "pivot",
        "chia để trị quicksort",
        "lomuto partition",
        "hoare partition",
        "độ phức tạp quicksort",
    ],
    "sorting_mergesort": [
        "mergesort",
        "sắp xếp trộn",
        "merge sort",
        "thuật toán mergesort",
        "thuật toán sắp xếp trộn",
        "trộn hai danh sách",
        "hợp nhất",
        "merge",
        "chia để trị mergesort",
        "sắp xếp ổn định",
        "stable sort",
        "độ phức tạp mergesort",
    ],
    "minimum_spanning_tree": [
        "cây khung nhỏ nhất",
        "cây bao trùm nhỏ nhất",
        "minimum spanning tree",
        "mst",
        "thuật toán kruskal",
        "thuật toán prim",
        "kruskal",
        "prim",
        "tập hợp rời rạc",
        "disjoint set",
        "disjoint set union",
        "dsu",
        "union find",
    ],
    "heap_and_priority_queue": [
        "heap",
        "đống",
        "cấu trúc heap",
        "min heap",
        "max heap",
        "đống cực tiểu",
        "đống cực đại",
        "heapsort",
        "sắp xếp vun đống",
        "vun đống",
        "heapify",
        "sift-up",
        "sift-down",
    ],
}


class DsaThesaurusRegistry:
    """
    Registry for bidirectional Data Structures & Algorithms terminology mappings.
    Supports exact matching, tone-insensitive matching, and greedy n-gram term discovery.
    """

    def __init__(self, clusters: Optional[Dict[str, List[str]]] = None):
        self.clusters = clusters or DSA_SYNONYM_CLUSTERS
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

        # Sort terms from longest to shortest for greedy n-gram / phrase matching
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
        Extracts DSA domain terms from the query and expands them into synonyms.
        Maintains order, preserves query-specific primary terms, and removes duplicates.
        """
        if not query or not query.strip():
            return []

        norm_q = normalize_term(query)
        stripped_q = strip_accents(norm_q)

        matched_clusters: Set[str] = set()
        matched_original_terms: List[str] = []

        for term, cluster_name in self._compiled_keywords:
            n_t = normalize_term(term)
            s_t = strip_accents(n_t)

            # Check boundary match
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


# Singleton instance
_DEFAULT_REGISTRY = DsaThesaurusRegistry()


def get_dsa_synonyms(term: str) -> List[str]:
    """Public helper to get bidirectional DSA synonyms."""
    return _DEFAULT_REGISTRY.get_synonyms(term)


def expand_dsa_query_terms(query: str) -> List[str]:
    """
    Expands conversational user queries into authoritative technical DSA keywords.
    E.g.: 'quicksort' -> ['quicksort', 'sắp xếp nhanh', 'pivot', 'phân hoạch', ...]
    """
    return _DEFAULT_REGISTRY.expand_query(query)


def get_dsa_cluster_name(term: str) -> Optional[str]:
    """Returns semantic cluster name for a given term."""
    return _DEFAULT_REGISTRY.get_cluster_name(term)


# Algorithmic Complexity Reference Table (Time & Space)
DSA_COMPLEXITY_REFERENCE: Dict[str, Dict[str, str]] = {
    "Array (Access)": {"time_best": "O(1)", "time_avg": "O(1)", "time_worst": "O(1)", "space_worst": "O(1)"},
    "Array (Search)": {"time_best": "O(1)", "time_avg": "O(n)", "time_worst": "O(n)", "space_worst": "O(1)"},
    "Array (Insertion)": {"time_best": "O(1)", "time_avg": "O(n)", "time_worst": "O(n)", "space_worst": "O(1)"},
    "Array (Deletion)": {"time_best": "O(1)", "time_avg": "O(n)", "time_worst": "O(n)", "space_worst": "O(1)"},
    "Singly Linked List (Access)": {"time_best": "O(1)", "time_avg": "O(n)", "time_worst": "O(n)", "space_worst": "O(1)"},
    "Singly Linked List (Insertion at Head)": {"time_best": "O(1)", "time_avg": "O(1)", "time_worst": "O(1)", "space_worst": "O(1)"},
    "Stack (Push/Pop)": {"time_best": "O(1)", "time_avg": "O(1)", "time_worst": "O(1)", "space_worst": "O(1)"},
    "Queue (Enqueue/Dequeue)": {"time_best": "O(1)", "time_avg": "O(1)", "time_worst": "O(1)", "space_worst": "O(1)"},
    "BST (Search)": {"time_best": "O(1)", "time_avg": "O(log n)", "time_worst": "O(n)", "space_worst": "O(1)"},
    "BST (Insertion)": {"time_best": "O(1)", "time_avg": "O(log n)", "time_worst": "O(n)", "space_worst": "O(1)"},
    "AVL Tree (Search)": {"time_best": "O(1)", "time_avg": "O(log n)", "time_worst": "O(log n)", "space_worst": "O(1)"},
    "AVL Tree (Insertion)": {"time_best": "O(log n)", "time_avg": "O(log n)", "time_worst": "O(log n)", "space_worst": "O(1)"},
    "Hash Table (Search)": {"time_best": "O(1)", "time_avg": "O(1)", "time_worst": "O(n)", "space_worst": "O(n)"},
    "Hash Table (Insertion)": {"time_best": "O(1)", "time_avg": "O(1)", "time_worst": "O(n)", "space_worst": "O(n)"},
    "QuickSort": {"time_best": "O(n log n)", "time_avg": "O(n log n)", "time_worst": "O(n^2)", "space_worst": "O(log n)"},
    "MergeSort": {"time_best": "O(n log n)", "time_avg": "O(n log n)", "time_worst": "O(n log n)", "space_worst": "O(n)"},
    "HeapSort": {"time_best": "O(n log n)", "time_avg": "O(n log n)", "time_worst": "O(n log n)", "space_worst": "O(1)"},
    "BFS / DFS": {"time_best": "O(V + E)", "time_avg": "O(V + E)", "time_worst": "O(V + E)", "space_worst": "O(V)"},
    "Dijkstra (with Binary Heap)": {"time_best": "O(E log V)", "time_avg": "O(E log V)", "time_worst": "O(E log V)", "space_worst": "O(V)"},
}


# Few-shot QA examples for Data Structures & Algorithms
DSA_FEW_SHOT_QA_EXAMPLES: List[Dict[str, Any]] = [
    {
        "id": "dsa_qa_01",
        "topic": "Độ phức tạp thuật toán và Ký hiệu Big-O",
        "category": "complexity_analysis",
        "canonical_terms": [
            "độ phức tạp big-o",
            "độ phức tạp thời gian",
            "độ phức tạp không gian",
            "worst-case",
            "best-case",
            "average-case",
            "asymptotic analysis",
        ],
        "question": "Độ phức tạp thuật toán Big-O là gì? Phân biệt trường hợp tốt nhất, trung bình và xấu nhất?",
        "intent": "concept_and_classification",
        "expected_action": "explain_big_o_and_bounds",
        "grounded_answer": (
            "Ký hiệu Big-O (Big-O Notation) mô tả cận trên tiệm cận của hàm thời gian thực thi hoặc dung lượng bộ nhớ "
            "mà thuật toán yêu cầu khi kích thước đầu vào n tiến tới vô cùng (n -> ∞).\n\n"
            "Các trường hợp phân tích độ phức tạp:\n"
            "1. Trường hợp tốt nhất (Best-case, thường ký hiệu Big-Omega Ω): Kịch bản dữ liệu đầu vào thuận lợi nhất, đòi hỏi số bước tính toán ít nhất.\n"
            "2. Trường hợp trung bình (Average-case, thường ký hiệu Big-Theta Θ): Kỳ vọng toán học về số bước tính trên mọi phân bố dữ liệu đầu vào ngẫu nhiên.\n"
            "3. Trường hợp xấu nhất (Worst-case, Big-O): Cận trên bảo đảm chắc chắn; thời gian thực thi trong mọi trường hợp không bao giờ vượt quá ngưỡng này.\n\n"
            "Các cấp độ phức tạp phổ biến từ nhanh nhất đến chậm nhất:\n"
            "- O(1): Thời gian hằng số (truy xuất phần tử mảng theo chỉ số, push/pop stack).\n"
            "- O(log n): Thời gian logarit (tìm kiếm nhị phân, thao tác trên cây AVL).\n"
            "- O(n): Thời gian tuyến tính (duyệt mảng, tìm kiếm tuần tự).\n"
            "- O(n log n): Tuyến tính nhân logarit (MergeSort, HeapSort, QuickSort trung bình).\n"
            "- O(n^2): Đa thức bậc hai (Bubble Sort, Insertion Sort hai vòng lặp lồng nhau).\n"
            "- O(2^n): Số mũ (thuật toán quay lui tháp Hà Nội, duyệt toàn bộ tập con).\n"
            "- O(n!): Giai thừa (bài toán người giao hàng TSP vét cạn hoán vị)."
        ),
        "bullet_types": [
            "- Trường hợp tốt nhất (Best-case, Ω)",
            "- Trường hợp trung bình (Average-case, Θ)",
            "- Trường hợp xấu nhất (Worst-case, O)",
        ],
        "example": "Truy xuất mảng A[i] là O(1); tìm kiếm nhị phân trên mảng đã sắp xếp là O(log n).",
    },
    {
        "id": "dsa_qa_02",
        "topic": "Mảng và Danh sách liên kết",
        "category": "linear_data_structures",
        "canonical_terms": [
            "mảng",
            "danh sách liên kết",
            "danh sách liên kết đơn",
            "danh sách liên kết đôi",
            "array",
            "linked list",
        ],
        "question": "So sánh mảng (Array) và danh sách liên kết (Linked List) về cấu trúc lưu trữ và chi phí thao tác?",
        "intent": "comparison_and_tradeoffs",
        "expected_action": "compare_array_and_linked_list",
        "grounded_answer": (
            "Mảng (Array) và Danh sách liên kết (Linked List) là hai cấu trúc dữ liệu tuyến tính cơ bản nhưng có nguyên lý lưu trữ khác nhau hoàn toàn:\n\n"
            "1. Cấu trúc lưu trữ bộ nhớ:\n"
            "- Mảng: Các phần tử được lưu trữ liên tiếp trong các ô nhớ kế tiếp nhau (contiguous memory). Kích thước mảng tĩnh cố định lúc khai báo.\n"
            "- Danh sách liên kết: Các nút (node) được cấp phát rời rạc trên vùng nhớ heap, liên kết với nhau qua các con trỏ (pointer/reference).\n\n"
            "2. So sánh chi phí thao tác (Time Complexity):\n"
            "- Truy xuất theo chỉ số (Access by index): Mảng đạt O(1) nhờ công thức Base_Address + i * size. Danh sách liên kết phải duyệt tuần tự từ đầu (Head) mất O(n).\n"
            "- Chèn/xóa ở đầu danh sách: Danh sách liên kết chỉ cần đổi con trỏ Head mất O(1). Mảng tĩnh/động phải dời toàn bộ phần tử phía sau mất O(n).\n"
            "- Chèn/xóa ở giữa: Cả hai đều mất O(n) (mảng mất thời gian dời phần tử, danh sách liên kết mất thời gian tìm nút vị trí chèn).\n"
            "- Bộ nhớ phụ trợ: Mảng không tốn bộ nhớ con trỏ. Danh sách liên kết tốn thêm 4 hoặc 8 byte cho mỗi con trỏ (next, prev) ở mỗi nút."
        ),
        "bullet_types": [
            "- Mảng lưu trữ liên tiếp, truy xuất ngẫu nhiên O(1).",
            "- Danh sách liên kết cấp phát động, chèn/xóa đầu O(1), truy xuất O(n).",
        ],
        "example": "Mảng số nguyên `int a[100]` vs `Node* head` trỏ tới các nút `(data, next)`.",
    },
    {
        "id": "dsa_qa_03",
        "topic": "Ngăn xếp và Hàng đợi",
        "category": "abstract_data_types",
        "canonical_terms": [
            "ngăn xếp",
            "hàng đợi",
            "stack",
            "queue",
            "lifo",
            "fifo",
            "push",
            "pop",
            "enqueue",
            "dequeue",
        ],
        "question": "Ngăn xếp (Stack) và Hàng đợi (Queue) hoạt động theo nguyên lý nào và có những ứng dụng thực tế nào?",
        "intent": "mechanism_and_application",
        "expected_action": "detail_stack_and_queue_properties",
        "grounded_answer": (
            "Ngăn xếp (Stack) và Hàng đợi (Queue) là hai kiểu dữ liệu trừu tượng tuyến tính bị ràng buộc bởi vị trí thêm và xóa phần tử:\n\n"
            "1. Ngăn xếp (Stack) - Nguyên lý LIFO (Last In, First Out - Vào sau ra trước):\n"
            "- Thao tác: `push` thêm phần tử vào đỉnh (Top), `pop` lấy phần tử ra khỏi đỉnh, `peek`/`top` xem phần tử đỉnh mà không xóa. Tất cả thao tác đều đạt O(1).\n"
            "- Ứng dụng thực tế: Quản lý ngăn xếp lời gọi hàm (Call Stack, Recursion), cơ chế Undo/Redo trong trình soạn thảo, kiểm tra đóng mở ngoặc hợp lệ, chuyển đổi biểu thức Trung tố sang Hậu tố (Infix to Postfix), duyệt DFS.\n\n"
            "2. Hàng đợi (Queue) - Nguyên lý FIFO (First In, First Out - Vào trước ra trước):\n"
            "- Thao tác: `enqueue` thêm phần tử vào cuối (Rear/Tail), `dequeue` lấy phần tử ra khỏi đầu (Front/Head), đều đạt O(1) khi cài đặt bằng mảng tròn hoặc DSLK.\n"
            "- Ứng dụng thực tế: Hàng đợi in ấn máy in (Print Spooler), điều phối tiến trình CPU (CPU Task Scheduling), truyền gói tin qua bộ đệm mạng (Router Buffer), thuật toán tìm kiếm theo chiều rộng (BFS)."
        ),
        "bullet_types": [
            "- Stack: LIFO, push/pop ở đỉnh (Top).",
            "- Queue: FIFO, enqueue ở cuối (Rear), dequeue ở đầu (Front).",
        ],
        "example": "Stack kiểm tra ngoặc `{[()]}` hợp lệ; Queue điều phối hàng đợi tác vụ máy in FIFO.",
    },
    {
        "id": "dsa_qa_04",
        "topic": "Cây tìm kiếm nhị phân (BST) và Duyệt cây",
        "category": "tree_data_structures",
        "canonical_terms": [
            "cây nhị phân",
            "cây tìm kiếm nhị phân",
            "bst",
            "binary search tree",
            "duyệt cây",
            "inorder",
            "preorder",
            "postorder",
        ],
        "question": "Trình bày tính chất của Cây tìm kiếm nhị phân (BST) và các phương pháp duyệt cây nhị phân?",
        "intent": "definition_and_traversal",
        "expected_action": "explain_bst_properties_and_traversals",
        "grounded_answer": (
            "Cây tìm kiếm nhị phân (Binary Search Tree - BST) là một cây nhị phân trong đó mỗi nút đều tuân thủ tính chất thứ tự nghiêm ngặt:\n"
            "- Mọi khóa thuộc cây con bên trái (Left Subtree) đều nhỏ hơn khóa của nút hiện tại.\n"
            "- Mọi khóa thuộc cây con bên phải (Right Subtree) đều lớn hơn khóa của nút hiện tại.\n"
            "- Cả cây con bên trái và bên phải đều phải là các cây tìm kiếm nhị phân.\n\n"
            "Độ phức tạp thao tác tìm kiếm, chèn, xóa trên BST:\n"
            "- Trường hợp trung bình (cây cân đối): O(log n).\n"
            "- Trường hợp suy biến (cây thoái hóa thành danh sách liên kết lệch một bên): O(n).\n\n"
            "Bốn phương pháp duyệt cây nhị phân kinh điển (Độ phức tạp O(n)):\n"
            "1. Duyệt trung thứ tự (In-order: Left -> Root -> Right): Đặc biệt trên BST, duyệt In-order sinh ra dãy các khóa theo thứ tự tăng dần hoàn hảo.\n"
            "2. Duyệt tiền thứ tự (Pre-order: Root -> Left -> Right): Thường dùng để sao chép cấu trúc cây hoặc tuần tự hóa (serialize) cây.\n"
            "3. Duyệt hậu thứ tự (Post-order: Left -> Right -> Root): Thường dùng để giải phóng bộ nhớ của cây từ lá lên gốc, hoặc tính kích thước thư mục.\n"
            "4. Duyệt theo mức (Level-order / BFS): Duyệt qua các tầng từ trên xuống dưới, từ trái sang phải bằng Hàng đợi (Queue)."
        ),
        "bullet_types": [
            "- In-order: Left -> Node -> Right (sinh chuỗi tăng dần trên BST).",
            "- Pre-order: Node -> Left -> Right.",
            "- Post-order: Left -> Right -> Node.",
            "- Level-order: Duyệt theo tầng từ gốc xuống lá dùng Queue.",
        ],
        "example": "Cho BST có gốc 8, con trái 3, con phải 10. Duyệt In-order: 3 -> 8 -> 10.",
    },
    {
        "id": "dsa_qa_05",
        "topic": "Cây cân bằng AVL (AVL Tree) và Phép quay",
        "category": "balanced_trees",
        "canonical_terms": [
            "cây avl",
            "cây cân bằng",
            "hệ số cân bằng",
            "balance factor",
            "quay cây",
            "quay trái",
            "quay phải",
        ],
        "question": "Cây AVL là gì? Hệ số cân bằng được tính như thế nào và các phép quay cây giải quyết mất cân bằng ra sao?",
        "intent": "mechanism_and_rotation",
        "expected_action": "explain_avl_balance_and_rotations",
        "grounded_answer": (
            "Cây AVL (đặt theo tên Georgy Adelson-Velsky và Evgenii Landis, 1962) là cây tìm kiếm nhị phân tự cân bằng (Self-balancing BST), "
            "đảm bảo chiều cao của cây luôn duy trì ở mức O(log n), khắc phục hoàn toàn hiện tượng suy biến về O(n) của BST thông thường.\n\n"
            "Hệ số cân bằng (Balance Factor - BF):\n"
            "- BF(node) = Height(Left_Subtree) - Height(Right_Subtree).\n"
            "- Điều kiện cân bằng AVL: Với mọi nút trên cây, BF(node) ∈ {-1, 0, +1}.\n"
            "- Nếu |BF(node)| > 1, cây bị mất cân bằng tại nút đó và phải thực hiện tái cân bằng thông qua phép quay.\n\n"
            "Bốn trường hợp mất cân bằng và phép quay tương ứng:\n"
            "1. Mất cân bằng Left-Left (LL): Nhánh lệch bên trái của con bên trái (BF > 1 và BF(con trái) >= 0) -> Thực hiện 1 phép quay đơn sang phải (Right Rotation).\n"
            "2. Mất cân bằng Right-Right (RR): Nhánh lệch bên phải của con bên phải (BF < -1 và BF(con phải) <= 0) -> Thực hiện 1 phép quay đơn sang trái (Left Rotation).\n"
            "3. Mất cân bằng Left-Right (LR): Con bên phải của con bên trái gây lệch -> Thực hiện quay kép: quay trái tại nút con, sau đó quay phải tại nút gốc bị lệch.\n"
            "4. Mất cân bằng Right-Left (RL): Con bên trái của con bên phải gây lệch -> Thực hiện quay kép: quay phải tại nút con, sau đó quay trái tại nút gốc bị lệch.\n\n"
            "Độ phức tạp tìm kiếm, chèn, xóa trên cây AVL luôn được đảm bảo nghiêm ngặt là O(log n)."
        ),
        "bullet_types": [
            "- Hệ số cân bằng: BF = h(Left) - h(Right) ∈ {-1, 0, 1}.",
            "- Quay đơn: Quay phải (Right Rotation) cho LL; Quay trái (Left Rotation) cho RR.",
            "- Quay kép: Left-Right (LR) và Right-Left (RL).",
        ],
        "example": "Chèn liên tiếp 30, 20, 10 tạo thành đường thẳng lệch trái (LL). Quay phải tại 30 -> 20 lên làm gốc, 10 và 30 là hai con.",
    },
    {
        "id": "dsa_qa_06",
        "topic": "Bảng băm và Xử lý xung đột",
        "category": "hashing_structures",
        "canonical_terms": [
            "bảng băm",
            "hàm băm",
            "xung đột băm",
            "đụng độ băm",
            "chaining",
            "địa chỉ mở",
            "linear probing",
            "load factor",
        ],
        "question": "Bảng băm (Hash Table) là gì? Xung đột băm xảy ra khi nào và các phương pháp xử lý xung đột phổ biến?",
        "intent": "concept_and_collision_handling",
        "expected_action": "explain_hash_table_and_collision_methods",
        "grounded_answer": (
            "Bảng băm (Hash Table / Hash Map) là cấu trúc dữ liệu lưu trữ các cặp khóa - giá trị (Key - Value), "
            "sử dụng hàm băm (Hash Function) h(key) để ánh xạ khóa thành chỉ số ô nhớ trong mảng. Trong điều kiện lý tưởng, thời gian tìm kiếm, chèn, xóa đạt O(1).\n\n"
            "Xung đột băm (Hash Collision):\n"
            "- Xảy ra khi hai khóa khác nhau k1 ≠ k2 nhưng hàm băm cho ra cùng một chỉ số h(k1) = h(k2) do nguyên lý Dirichlet (ngăn kéo).\n"
            "- Hệ số tải (Load Factor) α = n / m (n: số phần tử, m: kích thước bảng). Khi α vượt quá ngưỡng (thường là 0.7 - 0.75), bảng cần được băm lại (Rehashing) nhân đôi kích thước.\n\n"
            "Các phương pháp xử lý xung đột kinh điển:\n"
            "1. Phương pháp nối kết (Separate Chaining):\n"
            "- Mỗi ô (bucket) của bảng băm là một danh sách liên kết. Các phần tử bị trùng chỉ số băm được chèn nối tiếp vào danh sách đó.\n"
            "- Ưu điểm: Đơn giản, bảng không bao giờ bị đầy hoàn toàn, chịu tải tốt.\n"
            "2. Phương pháp địa chỉ mở (Open Addressing):\n"
            "- Tất cả các phần tử được lưu trực tiếp trong các ô của mảng. Khi xảy ra xung đột, thuật toán dò tìm ô trống kế tiếp:\n"
            "  * Dò tuyến tính (Linear Probing): Thử lần lượt h(k, i) = (h(k) + i) mod m. Dễ gây hiện tượng tụm cụm chính (Primary Clustering).\n"
            "  * Dò bậc hai (Quadratic Probing): Thử h(k, i) = (h(k) + c1*i + c2*i^2) mod m nhằm giảm bớt tụm cụm.\n"
            "  * Băm kép (Double Hashing): Dùng hai hàm băm h(k, i) = (h1(k) + i * h2(k)) mod m, phân tán xung đột tốt nhất."
        ),
        "bullet_types": [
            "- Nối kết (Separate Chaining): Dùng DSLK tại mỗi bucket.",
            "- Địa chỉ mở (Open Addressing): Dò tuyến tính, dò bậc hai, băm kép.",
            "- Hệ số tải α = n / m.",
        ],
        "example": "Bảng m = 10, h(k) = k mod 10. Chèn 15 và 25 đều có h = 5. Với Chaining: ô 5 chứa 15 -> 25.",
    },
    {
        "id": "dsa_qa_07",
        "topic": "Thuật toán duyệt đồ thị BFS và DFS",
        "category": "graph_algorithms",
        "canonical_terms": [
            "đồ thị",
            "bfs",
            "dfs",
            "duyệt theo chiều rộng",
            "duyệt theo chiều sâu",
            "breadth first search",
            "depth first search",
        ],
        "question": "Trình bày thuật toán duyệt đồ thị BFS và DFS? So sánh cấu trúc dữ liệu, độ phức tạp và ứng dụng?",
        "intent": "comparison_and_algorithm_details",
        "expected_action": "compare_bfs_and_dfs",
        "grounded_answer": (
            "Duyệt đồ thị là quá trình ghé thăm tất cả các đỉnh của đồ thị một cách có hệ thống sao cho mỗi đỉnh chỉ được thăm đúng 1 lần:\n\n"
            "1. Duyệt theo chiều rộng (BFS - Breadth-First Search):\n"
            "- Nguyên lý: Xuất phát từ đỉnh nguồn, thăm tất cả các đỉnh láng giềng kề trước khi đi xuống các đỉnh ở mức sâu hơn (lan tỏa theo từng lớp sóng).\n"
            "- Cấu trúc dữ liệu: Sử dụng Hàng đợi (Queue FIFO).\n"
            "- Ứng dụng: Tìm đường đi ngắn nhất trên đồ thị không trọng số (hoặc trọng số bằng nhau), tìm thành phần liên thông, thuật toán kiểm tra đồ thị hai phía (Bipartite Graph).\n\n"
            "2. Duyệt theo chiều sâu (DFS - Depth-First Search):\n"
            "- Nguyên lý: Đi xa nhất có thể dọc theo mỗi nhánh trước khi quay lui (Backtracking).\n"
            "- Cấu trúc dữ liệu: Sử dụng Ngăn xếp (Stack LIFO) hoặc Đệ quy (Recursion Call Stack).\n"
            "- Ứng dụng: Sắp xếp tô-pô (Topological Sort), phát hiện chu trình trên đồ thị, tìm các khớp và cầu (Bridges/Articulation Points), tìm thành phần liên thông mạnh (Tarjan / Kosaraju).\n\n"
            "3. Độ phức tạp thời gian và không gian:\n"
            "- Thời gian: Cả BFS và DFS đều có độ phức tạp O(V + E) với biểu diễn danh sách kề (V: số đỉnh, E: số cạnh), hoặc O(V^2) với ma trận kề.\n"
            "- Không gian bộ nhớ: O(V) để lưu mảng đánh dấu `visited` và hàng đợi/ngăn xếp."
        ),
        "bullet_types": [
            "- BFS: Dùng Queue (FIFO), duyệt theo từng tầng, tìm đường ngắn nhất đồ thị không trọng số.",
            "- DFS: Dùng Stack/Đệ quy, đi sâu tối đa rồi quay lui, phát hiện chu trình và sắp xếp tô-pô.",
            "- Độ phức tạp: O(V + E) với danh sách kề.",
        ],
        "example": "Đồ thị 1 -> 2, 1 -> 3. BFS từ 1 thăm: 1, 2, 3; DFS từ 1 thăm 1, 2 rồi quay lại thăm 3.",
    },
    {
        "id": "dsa_qa_08",
        "topic": "Thuật toán Dijkstra tìm đường đi ngắn nhất",
        "category": "shortest_path_algorithms",
        "canonical_terms": [
            "dijkstra",
            "thuật toán dijkstra",
            "đường đi ngắn nhất",
            "shortest path",
            "thư giãn cạnh",
            "hàng đợi ưu tiên",
        ],
        "question": "Thuật toán Dijkstra giải quyết bài toán gì, hoạt động theo nguyên lý nào và tại sao không áp dụng được cho cạnh có trọng số âm?",
        "intent": "algorithm_mechanism_and_limitation",
        "expected_action": "explain_dijkstra_principles_and_constraints",
        "grounded_answer": (
            "Thuật toán Dijkstra (Edsger W. Dijkstra, 1956) giải quyết bài toán tìm đường đi ngắn nhất từ một đỉnh nguồn duy nhất "
            "(Single-Source Shortest Path) đến tất cả các đỉnh còn lại trên đồ thị có hướng hoặc vô hướng với trọng số các cạnh không âm (w(u, v) >= 0).\n\n"
            "Nguyên lý hoạt động (Chiến lược Tham lam - Greedy):\n"
            "1. Khởi tạo: Đặt khoảng cách d[s] = 0 đối với đỉnh nguồn s; d[v] = ∞ đối với mọi đỉnh v khác. Đưa tất cả các đỉnh vào hàng đợi ưu tiên Min-Heap.\n"
            "2. Lặp: Tại mỗi bước, trích xuất đỉnh u có d[u] nhỏ nhất chưa được cố định khỏi hàng đợi ưu tiên.\n"
            "3. Thư giãn cạnh (Edge Relaxation): Với mỗi đỉnh v kề với u:\n"
            "   Nếu d[u] + w(u, v) < d[v] thì cập nhật: d[v] = d[u] + w(u, v) và ghi nhận cha của v là u.\n"
            "4. Kết thúc khi tất cả các đỉnh có thể đến được đã được chốt nhãn khoảng cách ngắn nhất.\n\n"
            "Tại sao không áp dụng được cho trọng số âm?\n"
            "- Thuật toán Dijkstra giả định rằng khi một đỉnh u được rút ra khỏi Min-Heap với d[u] nhỏ nhất, nhãn d[u] đó đã là khoảng cách tối ưu vĩnh viễn và không bao giờ giảm thêm được nữa.\n"
            "- Nếu tồn tại cạnh có trọng số âm, một đường đi vòng qua cạnh âm sau đó có thể làm giảm chi phí của đỉnh đã chốt, khiến tính đúng đắn của tham lam bị phá vỡ hoàn toàn.\n"
            "- Để xử lý đồ thị có trọng số âm, ta phải sử dụng thuật toán Bellman-Ford (độ phức tạp O(V * E)).\n\n"
            "Độ phức tạp của Dijkstra:\n"
            "- Cài đặt bằng Min-Heap / Priority Queue nhị phân: O((V + E) log V) hay O(E log V).\n"
            "- Cài đặt bằng mảng thô sơ: O(V^2)."
        ),
        "bullet_types": [
            "- Bài toán: Tìm đường đi ngắn nhất 1 nguồn với trọng số không âm.",
            "- Thao tác cốt lõi: Thư giãn cạnh d[v] = min(d[v], d[u] + w(u, v)).",
            "- Hạn chế: Thất bại khi có cạnh trọng số âm (phải dùng Bellman-Ford).",
            "- Độ phức tạp: O(E log V) với Min-Heap.",
        ],
        "example": "Đỉnh A đến B trọng số 4, A đến C trọng số 2, C đến B trọng số 1. Dijkstra chọn A->C (2) rồi relax C->B (2+1=3 < 4) => Đường đi ngắn nhất A->B là 3.",
    },
    {
        "id": "dsa_qa_09",
        "topic": "Quy hoạch động (Dynamic Programming)",
        "category": "algorithmic_paradigms",
        "canonical_terms": [
            "quy hoạch động",
            "dynamic programming",
            "dp",
            "bài toán con gối nhau",
            "cấu trúc con tối ưu",
            "memoization",
            "tabulation",
            "knapsack",
        ],
        "question": "Quy hoạch động (Dynamic Programming) là gì? Nêu hai đặc trưng cốt lõi và phân biệt phương pháp Top-down với Bottom-up?",
        "intent": "concept_and_methodology",
        "expected_action": "explain_dp_properties_and_approaches",
        "grounded_answer": (
            "Quy hoạch động (Dynamic Programming - DP) là phương pháp thiết kế thuật toán giải quyết các bài toán tối ưu bằng cách chia nhỏ bài toán lớn "
            "thành các bài toán con, giải từng bài toán con đúng một lần và lưu lại kết quả để tái sử dụng mà không bao giờ phải tính toán lại.\n\n"
            "Hai đặc trưng bắt buộc của bài toán áp dụng được Quy hoạch động:\n"
            "1. Cấu trúc con tối ưu (Optimal Substructure): Nghiệm tối ưu của bài toán lớn được cấu thành từ các nghiệm tối ưu của các bài toán con cấp thấp hơn.\n"
            "2. Các bài toán con gối nhau (Overlapping Subproblems): Quá trình phân rã bài toán sinh ra cùng những bài toán con giống nhau lặp đi lặp lại nhiều lần (khác với Chia để trị vốn giải các bài toán con độc lập không gối nhau).\n\n"
            "Hai hướng tiếp cận triển khai Quy hoạch động:\n"
            "- Tiếp cận từ trên xuống (Top-down with Memoization):\n"
            "  * Viết theo dạng đệ quy tự nhiên theo công thức truy hồi.\n"
            "  * Sử dụng bảng ghi nhớ (mảng hoặc từ điển/hash map) để lưu kết quả. Trước khi tính f(n), kiểm tra nếu đã có trong bảng thì trả về ngay.\n"
            "- Tiếp cận từ dưới lên (Bottom-up with Tabulation):\n"
            "  * Khởi tạo bảng phương án (mảng 1D hoặc 2D) từ các trường hợp cơ sở (base cases).\n"
            "  * Dùng vòng lặp tuần tự điền từng ô trong bảng phương án từ bài toán nhỏ nhất đến bài toán lớn nhất.\n"
            "  * Không tốn chi phí ngăn xếp đệ quy, dễ tối ưu hóa không gian bộ nhớ (Space Optimization)."
        ),
        "bullet_types": [
            "- Hai đặc trưng: Cấu trúc con tối ưu & Bài toán con gối nhau.",
            "- Top-down (Memoization): Đệ quy kết hợp bảng ghi nhớ.",
            "- Bottom-up (Tabulation): Vòng lặp điền bảng từ bài toán cơ sở lên.",
        ],
        "example": "Tính số Fibonacci: f(n) = f(n-1) + f(n-2). Đệ quy thường O(2^n); Quy hoạch động lưu mảng giảm xuống O(n) thời gian và O(1) bộ nhớ.",
    },
    {
        "id": "dsa_qa_10",
        "topic": "Sắp xếp QuickSort vs MergeSort",
        "category": "sorting_algorithms",
        "canonical_terms": [
            "quicksort",
            "mergesort",
            "sắp xếp nhanh",
            "sắp xếp trộn",
            "pivot",
            "phân hoạch",
            "stable sort",
        ],
        "question": "So sánh chi tiết hai thuật toán sắp xếp chia để trị QuickSort và MergeSort về nguyên lý, độ phức tạp và tính ổn định?",
        "intent": "comparison_and_complexity",
        "expected_action": "compare_quicksort_and_mergesort",
        "grounded_answer": (
            "QuickSort và MergeSort là hai thuật toán sắp xếp kinh điển dựa trên mô hình thiết kế thuật toán Chia để trị (Divide and Conquer):\n\n"
            "1. Nguyên lý hoạt động:\n"
            "- QuickSort:\n"
            "  * Chọn một phần tử làm chốt (Pivot).\n"
            "  * Phân hoạch (Partition): Sắp đặt các phần tử nhỏ hơn chốt về bên trái và các phần tử lớn hơn chốt về bên phải.\n"
            "  * Đệ quy áp dụng QuickSort cho hai nửa bên trái và bên phải chốt.\n"
            "- MergeSort:\n"
            "  * Chia mảng thành hai nửa bằng nhau tại vị trí chính giữa (Mid).\n"
            "  * Đệ quy sắp xếp từng nửa mảng con.\n"
            "  * Hợp nhất (Merge): Trộn hai nửa mảng con đã có thứ tự thành mảng kết quả hoàn chỉnh.\n\n"
            "2. So sánh độ phức tạp (Complexity):\n"
            "- Thời gian QuickSort: Tốt nhất O(n log n), Trung bình O(n log n), Xấu nhất O(n^2) (khi pivot chọn vào phần tử lớn nhất hoặc nhỏ nhất trên mảng đã sắp xếp).\n"
            "- Thời gian MergeSort: Luôn luôn đảm bảo O(n log n) trong mọi trường hợp (tốt nhất, trung bình, xấu nhất).\n"
            "- Không gian phụ trợ (Space Complexity): QuickSort sắp xếp tại chỗ (In-place), chỉ tốn O(log n) ngăn xếp đệ quy. MergeSort cần thêm mảng phụ trợ kích thước O(n) để thực hiện thao tác trộn.\n\n"
            "3. Tính ổn định (Stability):\n"
            "- MergeSort là thuật toán sắp xếp ổn định (Stable Sort): các phần tử có giá trị bằng nhau luôn giữ nguyên thứ tự xuất hiện ban đầu.\n"
            "- QuickSort cơ bản không ổn định (Unstable) do các phép hoán đổi vị trí qua pivot có thể làm đảo lộn thứ tự tương đối."
        ),
        "bullet_types": [
            "- QuickSort: Chọn Pivot -> Phân hoạch -> Đệ quy; O(n log n) trung bình, O(n^2) xấu nhất; In-place O(log n) bộ nhớ; Không ổn định.",
            "- MergeSort: Chia đôi -> Đệ quy -> Trộn; O(n log n) mọi trường hợp; Cần O(n) bộ nhớ phụ; Ổn định.",
        ],
        "example": "Sắp xếp danh sách sinh viên theo Điểm số nhưng muốn giữ thứ tự Tên theo vần ABC khi điểm bằng nhau: Bắt buộc dùng MergeSort (hoặc TimSort) vì tính ổn định.",
    },
    {
        "id": "dsa_qa_11",
        "topic": "Cây khung nhỏ nhất (MST) - Thuật toán Kruskal và Prim",
        "category": "graph_algorithms",
        "canonical_terms": [
            "cây khung nhỏ nhất",
            "minimum spanning tree",
            "mst",
            "thuật toán kruskal",
            "thuật toán prim",
            "disjoint set",
        ],
        "question": "Cây khung nhỏ nhất (MST) là gì? Trình bày và so sánh hai thuật toán Kruskal và Prim?",
        "intent": "concept_and_comparison",
        "expected_action": "explain_mst_kruskal_and_prim",
        "grounded_answer": (
            "Cây khung nhỏ nhất (Minimum Spanning Tree - MST) của đồ thị vô hướng liên thông có trọng số là một đồ thị con chứa tất cả các đỉnh của đồ thị, "
            "không chứa chu trình (là một cây gồm đúng V - 1 cạnh) và có tổng trọng số của các cạnh là nhỏ nhất có thể.\n\n"
            "1. Thuật toán Kruskal (Tiếp cận theo Cạnh):\n"
            "- Các bước: Sắp xếp tất cả các cạnh theo trọng số tăng dần. Khởi tạo mỗi đỉnh là một tập hợp rời rạc. Duyệt qua từng cạnh: nếu cạnh nối hai đỉnh thuộc hai tập hợp khác nhau (không tạo chu trình), nạp cạnh vào MST và hợp nhất hai tập hợp (Union-Find / DSU).\n"
            "- Độ phức tạp: O(E log E) hay O(E log V) chủ yếu do chi phí sắp xếp cạnh.\n"
            "- Ưu thế: Hoạt động rất nhanh trên đồ thị thưa (Sparse Graph, E << V^2).\n\n"
            "2. Thuật toán Prim (Tiếp cận theo Đỉnh):\n"
            "- Các bước: Bắt đầu từ một đỉnh tùy ý. Tại mỗi bước, chọn cạnh có trọng số nhỏ nhất nối giữa một đỉnh đã thuộc cây với một đỉnh chưa thuộc cây để mở rộng cây, lặp lại cho đến khi đủ V đỉnh.\n"
            "- Độ phức tạp: O(E log V) khi dùng Min-Heap / Binary Heap; O(V^2) khi dùng ma trận kề.\n"
            "- Ưu thế: Hoạt động rất hiệu quả trên đồ thị dày (Dense Graph, E xấp xỉ V^2)."
        ),
        "bullet_types": [
            "- Kruskal: Phát triển theo cạnh, sắp xếp cạnh + Disjoint Set (DSU), tối ưu cho đồ thị thưa.",
            "- Prim: Phát triển theo đỉnh từ gốc, dùng Min-Heap, tối ưu cho đồ thị dày.",
        ],
        "example": "Mạng cáp quang nối 5 thành phố với chi phí tối thiểu tạo thành MST gồm 4 đường truyền kết nối tất cả các thành phố không có chu trình.",
    },
    {
        "id": "dsa_qa_12",
        "topic": "Cấu trúc Heap và Hàng đợi ưu tiên",
        "category": "heap_structures",
        "canonical_terms": [
            "heap",
            "đống",
            "min heap",
            "max heap",
            "priority queue",
            "heapsort",
            "heapify",
        ],
        "question": "Cấu trúc dữ liệu Heap là gì? Phân biệt Min-Heap và Max-Heap cùng thao tác Heapify?",
        "intent": "definition_and_operations",
        "expected_action": "explain_heap_structure_and_operations",
        "grounded_answer": (
            "Heap (Đống) là cấu trúc cây nhị phân gần hoàn chỉnh (Complete Binary Tree) thỏa mãn tính chất đống (Heap Property). "
            "Heap thường được biểu diễn gọn gàng trong mảng một chiều mà không cần con trỏ:\n"
            "- Nút cha ở chỉ số i -> Nút con trái ở `2*i + 1`, nút con phải ở `2*i + 2`, nút cha của i ở `(i - 1) // 2`.\n\n"
            "Phân loại Heap:\n"
            "1. Max-Heap: Giá trị của nút cha luôn lớn hơn hoặc bằng giá trị của các nút con. Phần tử lớn nhất luôn nằm ở gốc (Root, chỉ số 0).\n"
            "2. Min-Heap: Giá trị của nút cha luôn nhỏ hơn hoặc bằng giá trị của các nút con. Phần tử nhỏ nhất luôn nằm ở gốc.\n\n"
            "Các thao tác then chốt:\n"
            "- Heapify (Vun đống): Duyệt chỉnh một nút bị vi phạm tính chất heap xuống dưới (Sift-Down / Sink) đến khi thỏa mãn tính chất heap, chi phí O(log n).\n"
            "- Xây dựng Heap từ mảng ban đầu (Build-Heap): Thực hiện heapify từ nút cha cuối cùng ngược về gốc, có độ phức tạp thời gian tổng cộng là O(n).\n"
            "- Trích xuất cực trị (Extract Max/Min): Lấy phần tử gốc O(1), chuyển phần tử cuối cùng lên gốc và gọi heapify mất O(log n).\n"
            "- Ứng dụng: Cài đặt Hàng đợi ưu tiên (Priority Queue), Thuật toán sắp xếp HeapSort (O(n log n) tại chỗ), Thuật toán Dijkstra và Prim."
        ),
        "bullet_types": [
            "- Cây nhị phân hoàn chỉnh lưu trong mảng: Trái = 2i+1, Phải = 2i+2, Cha = (i-1)//2.",
            "- Max-Heap: Gốc lớn nhất; Min-Heap: Gốc nhỏ nhất.",
            "- Thao tác: Heapify O(log n), Build-Heap O(n), Extract Min/Max O(log n).",
        ],
        "example": "Mảng [90, 80, 70, 50, 60] là một Max-Heap hợp lệ với gốc 90 có hai con 80 và 70.",
    },
]

# Alias to fulfill both DSA_FEW_SHOT_EXAMPLES and DSA_FEW_SHOT_QA_EXAMPLES
DSA_FEW_SHOT_EXAMPLES = DSA_FEW_SHOT_QA_EXAMPLES
