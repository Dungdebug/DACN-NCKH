"""
Domain-Specific Operating Systems Semantic Patterns and Knowledge Base.
Provides bidirectional synonym mapping, domain query expansion, and few-shot QA blueprints
for authoritative Operating Systems educational curricula (Silberschatz, Tanenbaum, Stallings).
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


# Core Operating Systems domain synonym clusters
OS_SYNONYM_CLUSTERS: Dict[str, List[str]] = {
    "process_vs_thread": [
        "tiến trình và luồng",
        "tiến trình vs luồng",
        "so sánh tiến trình và luồng",
        "process vs thread",
        "processes and threads",
        "tiến trình",
        "luồng",
        "process",
        "thread",
        "tiểu trình",
        "đa luồng",
        "multithreading",
        "đa tiến trình",
        "multiprocessing",
        "pcb",
        "tcb",
        "process control block",
        "thread control block",
        "chuyển ngữ cảnh",
        "context switch",
        "context switching",
    ],
    "process_lifecycle": [
        "trạng thái tiến trình",
        "vòng đời tiến trình",
        "process state",
        "process states",
        "process lifecycle",
        "new",
        "ready",
        "running",
        "waiting",
        "blocked",
        "terminated",
        "sẵn sàng",
        "thực thi",
        "chờ",
        "kết thúc",
        "khởi tạo",
        "chuyển đổi trạng thái tiến trình",
        "state transition",
    ],
    "cpu_scheduling_general": [
        "lập lịch cpu",
        "điều phối cpu",
        "lập lịch tiến trình",
        "cpu scheduling",
        "process scheduling",
        "thuật toán lập lịch",
        "scheduling algorithms",
        "scheduler",
        "bộ điều phối",
        "bộ lập lịch",
        "thời gian chờ",
        "waiting time",
        "thời gian lưu lại",
        "thời gian hoàn thành",
        "turnaround time",
        "thời gian đáp ứng",
        "response time",
        "thông lượng cpu",
        "cpu throughput",
        "độc quyền",
        "không độc quyền",
        "trưng dụng",
        "không trưng dụng",
        "preemptive",
        "non-preemptive",
    ],
    "scheduling_fcfs": [
        "fcfs",
        "first come first served",
        "đến trước phục vụ trước",
        "thuật toán fcfs",
        "lập lịch fcfs",
        "hiệu ứng đoàn tàu",
        "convoy effect",
        "first-in first-out cpu",
    ],
    "scheduling_sjf_srtf": [
        "sjf",
        "shortest job first",
        "tiến trình ngắn nhất trước",
        "công việc ngắn nhất trước",
        "thuật toán sjf",
        "srtf",
        "shortest remaining time first",
        "thời gian còn lại ngắn nhất trước",
        "thuật toán srtf",
        "sjf không trưng dụng",
        "sjf trưng dụng",
    ],
    "scheduling_round_robin": [
        "round robin",
        "rr",
        "thuật toán round robin",
        "lập lịch round robin",
        "lập lịch xoay vòng",
        "xoay vòng",
        "thời lượng định mức",
        "lát cắt thời gian",
        "time quantum",
        "time slice",
        "định ngạch thời gian",
    ],
    "scheduling_priority": [
        "lập lịch ưu tiên",
        "priority scheduling",
        "độ ưu tiên",
        "priority",
        "nạn đói tài nguyên",
        "nạn đói",
        "starvation",
        "kỹ thuật già hóa",
        "kỹ thuật lão hóa",
        "aging",
        "multilevel queue",
        "hàng đợi đa cấp",
    ],
    "deadlock_conditions": [
        "deadlock",
        "bế tắc",
        "tắc nghẽn",
        "khóa chết",
        "điều kiện deadlock",
        "4 điều kiện bế tắc",
        "4 điều kiện coffman",
        "coffman conditions",
        "deadlock conditions",
        "mutual exclusion",
        "loại trừ lẫn nhau",
        "hold and wait",
        "giữ và chờ",
        "no preemption",
        "không giải tỏa",
        "không trưng dụng tài nguyên",
        "circular wait",
        "chờ đợi vòng tròn",
        "vòng tròn chờ đợi",
    ],
    "deadlock_handling": [
        "xử lý deadlock",
        "phòng ngừa deadlock",
        "deadlock prevention",
        "tránh deadlock",
        "deadlock avoidance",
        "thuật toán banker",
        "banker's algorithm",
        "thuật toán người chia bài",
        "trạng thái an toàn",
        "safe state",
        "chuỗi an toàn",
        "safe sequence",
        "đồ thị cấp phát tài nguyên",
        "resource allocation graph",
        "rag",
        "phát hiện deadlock",
        "deadlock detection",
        "phục hồi sau deadlock",
        "deadlock recovery",
    ],
    "virtual_memory": [
        "bộ nhớ ảo",
        "virtual memory",
        "vm",
        "không gian địa chỉ ảo",
        "virtual address space",
        "bộ nhớ vật lý",
        "physical memory",
        "lỗi trang",
        "ngắt trang",
        "page fault",
        "page fault handler",
        "nạp trang theo yêu cầu",
        "demand paging",
        "tráo đổi trang",
        "swapping",
        "swap space",
        "vùng swap",
        "trì trệ bộ nhớ",
        "đảo trang liên tục",
        "thrashing",
        "working-set",
        "mô hình working set",
    ],
    "paging_system": [
        "phân trang",
        "paging",
        "trang nhớ",
        "trang logic",
        "page",
        "khung trang",
        "frame",
        "page frame",
        "bảng trang",
        "page table",
        "địa chỉ ảo",
        "địa chỉ logic",
        "logical address",
        "địa chỉ vật lý",
        "physical address",
        "page offset",
        "độ dời",
        "phân mảnh nội vi",
        "internal fragmentation",
        "tlb",
        "translation lookaside buffer",
        "bộ đệm tlb",
        "hit ratio",
    ],
    "segmentation_system": [
        "phân đoạn",
        "segmentation",
        "đoạn nhớ",
        "segment",
        "bảng phân đoạn",
        "segment table",
        "base limit",
        "phân mảnh ngoại vi",
        "external fragmentation",
        "kết hợp phân trang phân đoạn",
        "segmented paging",
        "kết đặc bộ nhớ",
        "compaction",
    ],
    "page_replacement": [
        "thay thế trang",
        "page replacement",
        "thuật toán thay thế trang",
        "page replacement algorithm",
        "fifo page replacement",
        "lru",
        "least recently used",
        "trang ít dùng gần đây nhất",
        "opt",
        "optimal page replacement",
        "thuật toán tối ưu",
        "bất thường belady",
        "belady anomaly",
        "thuật toán đồng hồ",
        "clock algorithm",
        "second chance",
        "cơ hội thứ hai",
    ],
    "critical_section_sync": [
        "đồng bộ hóa tiến trình",
        "process synchronization",
        "đồng bộ hóa",
        "đồng bộ",
        "miền găng",
        "đoạn găng",
        "vùng găng",
        "critical section",
        "critical section problem",
        "race condition",
        "điều kiện tranh đua",
        "tình trạng tương tranh",
        "loại trừ lẫn nhau",
        "tiến triển",
        "progress",
        "chờ đợi có giới hạn",
        "bounded waiting",
        "thuật toán peterson",
        "peterson's algorithm",
        "test-and-set",
        "atomic operation",
        "thao tác nguyên tử",
    ],
    "semaphore_and_mutex": [
        "semaphore",
        "mutex",
        "mutex lock",
        "khóa mutex",
        "đèn hiệu",
        "counting semaphore",
        "semaphore đếm",
        "binary semaphore",
        "semaphore nhị phân",
        "wait",
        "signal",
        "thao tác wait",
        "thao tác signal",
        "p và v",
        "p operation",
        "v operation",
        "busy waiting",
        "chờ đợi bận",
        "spinlock",
    ],
    "classic_concurrency_problems": [
        "bài toán đồng bộ kinh điển",
        "classical synchronization problems",
        "producer consumer",
        "nhà sản xuất người tiêu dùng",
        "bài toán sản xuất tiêu dùng",
        "bộ đệm có giới hạn",
        "bounded buffer",
        "dining philosophers",
        "bữa ăn triết gia",
        "bài toán các triết gia ăn tối",
        "triết gia ăn tối",
        "readers writers",
        "người đọc người ghi",
        "bài toán người đọc người ghi",
        "sleeping barber",
        "thợ cắt tóc ngủ quên",
    ],
    "file_system_and_io": [
        "hệ thống tệp",
        "file system",
        "hệ thống tập tin",
        "inode",
        "directory",
        "thư mục",
        "lập lịch truy cập đĩa",
        "disk scheduling",
        "sstf",
        "scan",
        "c-scan",
        "look",
        "c-look",
        "vfs",
        "virtual file system",
        "dma",
        "direct memory access",
        "truy cập bộ nhớ trực tiếp",
        "ngắt i/o",
        "interrupt",
    ],
}


class OSThesaurusRegistry:
    """
    Registry for bidirectional Operating Systems terminology mappings.
    Supports exact matching, tone-insensitive matching, and greedy multi-word extraction.
    """

    def __init__(self, clusters: Optional[Dict[str, List[str]]] = None):
        self.clusters = clusters or OS_SYNONYM_CLUSTERS
        self._norm_to_cluster: Dict[str, str] = {}
        self._stripped_to_cluster: Dict[str, str] = {}
        self._bidirectional_map: Dict[str, Set[str]] = {}
        self._compiled_keywords: List[Tuple[str, str]] = []  # (raw_term, cluster_name)
        self._build_indexes()

    def _build_indexes(self) -> None:
        """Constructs forward, reverse, and tone-insensitive indexes."""
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

        # Sort terms from longest to shortest for greedy n-gram matching
        all_terms_with_len.sort(key=lambda item: item[2], reverse=True)
        self._compiled_keywords = [(item[0], item[1]) for item in all_terms_with_len]

    def get_cluster_name(self, term: str) -> Optional[str]:
        """Finds cluster identifier for a given term (accent & case insensitive)."""
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
        Extracts operating system terms from query and expands them into domain synonyms.
        Preserves original query matched terms, avoids duplicates, maintains order.
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


# Singleton instance for module-level usage
_DEFAULT_REGISTRY = OSThesaurusRegistry()


def get_os_synonyms(term: str) -> List[str]:
    """
    Public API helper to get bidirectional Operating Systems synonyms.
    E.g.: 'process' -> ['tiến trình và luồng', 'process vs thread', 'tiến trình', 'luồng', ...]
    """
    return _DEFAULT_REGISTRY.get_synonyms(term)


def expand_os_query_terms(query: str) -> List[str]:
    """
    Expands conversational user queries into authoritative technical OS keywords.
    E.g.: 'lập lịch round robin' -> ['round robin', 'rr', 'time quantum', 'time slice', ...]
    """
    return _DEFAULT_REGISTRY.expand_query(query)


# Few-shot QA examples for Operating Systems academic curriculum
OS_FEW_SHOT_EXAMPLES: List[Dict[str, Any]] = [
    {
        "id": "os_qa_01",
        "topic": "Tiến trình (Process) và Luồng (Thread)",
        "category": "comparison",
        "canonical_terms": [
            "tiến trình",
            "luồng",
            "process",
            "thread",
            "pcb",
            "tcb",
            "chuyển ngữ cảnh",
            "context switch",
        ],
        "question": "Phân biệt sự khác nhau cơ bản giữa Tiến trình (Process) và Luồng (Thread)?",
        "intent": "concept_comparison",
        "expected_action": "compare_process_and_thread",
        "grounded_answer": (
            "Tiến trình (Process) và Luồng (Thread) là hai đơn vị thực thi cơ bản trong hệ điều hành với các điểm khác biệt cốt lõi:\n\n"
            "1. Định nghĩa và Bản chất:\n"
            "- Tiến trình (Process): Là một chương trình đang thực thi, sở hữu một không gian địa chỉ độc lập (Address Space gồm Code, Data, Heap, Stack) "
            "và tài nguyên hệ thống riêng (files, sockets). Được quản lý bởi Khối điều khiển tiến trình PCB (Process Control Block).\n"
            "- Luồng (Thread): Là đơn vị thực thi nhỏ nhất bên trong một tiến trình (thường gọi là tiến trình nhẹ - Lightweight Process). "
            "Nhiều luồng trong cùng một tiến trình chia sẻ chung không gian địa chỉ, biến toàn cục, mã nguồn và tài nguyên mở, nhưng mỗi luồng có ngăn xếp (Stack), "
            "con trỏ lệnh (Program Counter) và tập thanh ghi riêng, quản lý bởi TCB (Thread Control Block).\n\n"
            "2. Chi phí tạo lập và Chuyển ngữ cảnh (Context Switching):\n"
            "- Tiến trình: Chi phí tạo lập và hủy bỏ rất lớn; chuyển ngữ cảnh giữa các tiến trình tốn thời gian do phải hoán đổi bảng trang (Page Table), xóa bộ đệm TLB.\n"
            "- Luồng: Chi phí tạo lập rất nhẹ; chuyển ngữ cảnh giữa các luồng trong cùng tiến trình diễn ra nhanh chóng vì không cần thay đổi không gian bộ nhớ ảo.\n\n"
            "3. Tính độc lập và An toàn:\n"
            "- Một tiến trình bị lỗi (crash) thường không gây sập các tiến trình khác nhờ cơ chế bảo vệ bộ nhớ.\n"
            "- Một luồng bị lỗi bộ nhớ (segmentation fault) có thể kéo theo toàn bộ tiến trình và các luồng anh em bị sụp đổ."
        ),
        "bullet_types": [
            "- Tiến trình sở hữu không gian địa chỉ độc lập; Luồng chia sẻ chung bộ nhớ tiến trình mẹ.",
            "- Luồng có Program Counter, thanh ghi và Stack riêng (TCB); Tiến trình quản lý bằng PCB.",
            "- Chuyển ngữ cảnh luồng nhanh và tiết kiệm tài nguyên hơn chuyển ngữ cảnh tiến trình.",
            "- Tiến trình cô lập an toàn; lỗi ở một luồng có thể làm sập cả tiến trình.",
        ],
        "example": "Trình duyệt Chrome chạy mỗi tab là một Tiến trình độc lập (cô lập an toàn), trong khi trình xử lý văn bản Word dùng một Tiến trình có nhiều Luồng (1 luồng gõ văn bản, 1 luồng kiểm tra chính tả, 1 luồng tự động lưu).",
    },
    {
        "id": "os_qa_02",
        "topic": "Vòng đời và Các trạng thái của Tiến trình",
        "category": "process_lifecycle",
        "canonical_terms": [
            "trạng thái tiến trình",
            "vòng đời tiến trình",
            "process state",
            "ready",
            "running",
            "waiting",
            "blocked",
            "pcb",
        ],
        "question": "Trình bày các trạng thái cơ bản trong vòng đời của một tiến trình và điều kiện chuyển trạng thái?",
        "intent": "lifecycle_and_state_transition",
        "expected_action": "explain_process_states_and_transitions",
        "grounded_answer": (
            "Trong suốt vòng đời từ khi sinh ra đến khi kết thúc, một tiến trình thường trải qua mô hình 5 trạng thái kinh điển:\n\n"
            "1. New (Khởi tạo): Tiến trình đang được tạo lập và khởi tạo cấu trúc dữ liệu PCB trong bộ nhớ.\n"
            "2. Ready (Sẵn sàng): Tiến trình đã nạp vào bộ nhớ chính, có đầy đủ tài nguyên cần thiết và đang nằm trong hàng đợi Ready Queue để chờ CPU phân bổ thời gian thực thi.\n"
            "3. Running (Thực thi): Các chỉ thị của tiến trình đang được CPU trực tiếp xử lý (trên một lõi CPU tại một thời điểm chỉ có tối đa 1 tiến trình Running).\n"
            "4. Waiting / Blocked (Chờ / Khóa): Tiến trình tạm ngưng thực thi để chờ một sự kiện hoàn thành, chẳng hạn hoàn thành thao tác I/O hoặc chờ nhận tín hiệu đồng bộ từ tiến trình khác.\n"
            "5. Terminated (Kết thúc): Tiến trình đã hoàn thành công việc hoặc bị hủy bởi hệ điều hành, tài nguyên được thu hồi và PCB được giải phóng.\n\n"
            "Các bước chuyển tiếp trạng thái chính:\n"
            "- New -> Ready: Bộ điều phối dài hạn (Long-term scheduler) chấp thuận tiến trình vào bộ nhớ.\n"
            "- Ready -> Running: Bộ điều phối ngắn hạn (Short-term scheduler / Dispatcher) cấp phát CPU.\n"
            "- Running -> Ready: Hết thời lượng định mức (Time Quantum) trong thuật toán chia sẻ thời gian, hoặc bị tiến trình có độ ưu tiên cao hơn chiếm dụng.\n"
            "- Running -> Waiting: Tiến trình yêu cầu I/O hoặc tài nguyên chưa sẵn sàng.\n"
            "- Waiting -> Ready: Sự kiện hoặc thao tác I/O hoàn tất."
        ),
        "bullet_types": [
            "- 5 trạng thái kinh điển: New, Ready, Running, Waiting (Blocked), Terminated.",
            "- Dispatcher chuyển tiến trình từ Ready sang Running.",
            "- Hết lát cắt thời gian CPU: Running chuyển về Ready.",
            "- Đợi thao tác I/O: Running chuyển sang Waiting; I/O hoàn tất chuyển về Ready.",
        ],
        "example": "Khi một chương trình C gọi lệnh `scanf()`, tiến trình chuyển từ Running sang Waiting để đợi người dùng gõ phím từ bàn phím. Khi nhấn Enter, ngắt I/O đưa nó về Ready.",
    },
    {
        "id": "os_qa_03",
        "topic": "Lập lịch CPU (CPU Scheduling)",
        "category": "scheduling_algorithm",
        "canonical_terms": [
            "lập lịch cpu",
            "cpu scheduling",
            "fcfs",
            "sjf",
            "srtf",
            "turnaround time",
            "waiting time",
            "preemptive",
        ],
        "question": "Các tiêu chí đánh giá thuật toán lập lịch CPU là gì? So sánh FCFS và SJF/SRTF?",
        "intent": "scheduling_criteria_and_comparison",
        "expected_action": "compare_fcfs_and_sjf_scheduling",
        "grounded_answer": (
            "Lập lịch CPU là cơ chế của hệ điều hành nhằm phân bổ thời gian thực thi của CPU cho các tiến trình trong hàng đợi Ready.\n\n"
            "1. Các tiêu chuẩn đánh giá hiệu năng lập lịch:\n"
            "- Hiệu suất sử dụng CPU (CPU Utilization): Tỷ lệ phần trăm thời gian CPU bận rộn (mục tiêu: tối đa hóa).\n"
            "- Thông lượng (Throughput): Số lượng tiến trình hoàn thành trong một đơn vị thời gian (mục tiêu: tối đa hóa).\n"
            "- Thời gian hoàn thành (Turnaround Time): Khoảng thời gian từ lúc tiến trình được gửi đến cho tới khi hoàn tất (mục tiêu: tối thiểu hóa).\n"
            "- Thời gian chờ (Waiting Time): Tổng thời gian tiến trình phải nằm chờ trong hàng đợi Ready (mục tiêu: tối thiểu hóa).\n"
            "- Thời gian đáp ứng (Response Time): Khoảng thời gian từ lúc gửi yêu cầu đến khi có phản hồi đầu tiên (mục tiêu: tối thiểu hóa).\n\n"
            "2. So sánh thuật toán FCFS và SJF/SRTF:\n"
            "- FCFS (First-Come, First-Served): Tiến trình đến trước được cấp phát CPU trước (không trưng dụng - non-preemptive). Đơn giản, dễ cài đặt nhưng dễ gặp hiệu ứng đoàn tàu (Convoy Effect) khiến thời gian chờ trung bình tăng vọt nếu một tiến trình dài chạy trước các tiến trình ngắn.\n"
            "- SJF (Shortest Job First): Chọn tiến trình có thời gian thực thi CPU Burst ngắn nhất để chạy trước. Đạt thời gian chờ trung bình tối ưu nhất về mặt lý thuyết. Phiên bản trưng dụng (Preemptive) của SJF được gọi là SRTF (Shortest Remaining Time First). Nhược điểm của SJF là khó xác định chính xác độ dài CPU Burst tiếp theo và có thể gây nạn đói (Starvation) cho các tiến trình dài."
        ),
        "bullet_types": [
            "- Tiêu chuẩn đo lường: CPU Utilization, Throughput, Turnaround Time, Waiting Time, Response Time.",
            "- FCFS: Không trưng dụng, đơn giản, dễ mắc hiệu ứng đoàn tàu (Convoy Effect).",
            "- SJF/SRTF: Tối ưu thời gian chờ trung bình, nhưng có nguy cơ gây đói tiến trình dài và khó ước tính CPU Burst.",
        ],
        "example": "Có 3 tiến trình P1(burst=24ms), P2(burst=3ms), P3(burst=3ms). Với FCFS thứ tự P1,P2,P3 thì thời gian chờ TB là (0 + 24 + 27)/3 = 17ms. Với SJF thứ tự P2,P3,P1 thì thời gian chờ TB chỉ là (0 + 3 + 6)/3 = 3ms.",
    },
    {
        "id": "os_qa_04",
        "topic": "Thuật toán Round Robin (RR) và Lát cắt thời gian",
        "category": "scheduling_algorithm",
        "canonical_terms": [
            "round robin",
            "rr",
            "time quantum",
            "time slice",
            "lát cắt thời gian",
            "context switch",
        ],
        "question": "Thuật toán lập lịch Round Robin (RR) hoạt động như thế nào và việc lựa chọn kích thước Time Quantum ảnh hưởng gì?",
        "intent": "round_robin_mechanism_and_quantum_tradeoff",
        "expected_action": "explain_round_robin_and_quantum_impact",
        "grounded_answer": (
            "Thuật toán Round Robin (RR) là thuật toán lập lịch CPU trưng dụng (preemptive) được thiết kế đặc biệt cho các hệ thống chia sẻ thời gian (Time-sharing systems).\n\n"
            "Nguyên lý hoạt động:\n"
            "- Hàng đợi Ready được quản lý như một hàng đợi vòng tròn (FIFO circular queue).\n"
            "- Bộ lập lịch cấp phát cho mỗi tiến trình một định ngạch thời gian nhất định gọi là Lát cắt thời gian (Time Quantum hoặc Time Slice, thường từ 10ms đến 100ms).\n"
            "- Tiến trình chạy liên tục cho đến khi: (a) Hoàn thành hoặc tự giải phóng CPU, hoặc (b) Hết thời lượng Time Quantum. Khi hết quantum, bộ đếm ngắt đồng hồ phát tín hiệu, hệ điều hành thực hiện ngắt, đưa tiến trình trở lại cuối hàng đợi Ready và cấp CPU cho tiến trình kế tiếp.\n\n"
            "Ảnh hưởng của việc chọn kích thước Time Quantum (q):\n"
            "1. Nếu q quá lớn (q -> ∞): Thuật toán Round Robin suy biến thành FCFS, đánh mất tính tương tác nhanh nhạy của hệ thống đa nhiệm.\n"
            "2. Nếu q quá nhỏ: Tần suất chuyển ngữ cảnh (Context Switching Overhead) tăng vọt, tiêu tốn phần lớn năng lực xử lý của CPU chỉ để lưu và khôi phục trạng thái thanh ghi/PCB thay vì làm việc có ích.\n"
            "3. Quy tắc thiết kế chuẩn: Kích thước Time Quantum nên đủ lớn sao cho khoảng 80% các đợt bùng nổ CPU (CPU Burst) có thể hoàn thành trong vòng một lát cắt đơn lẻ, đồng thời thời gian chuyển ngữ cảnh chỉ chiếm dưới 1% thời lượng quantum."
        ),
        "bullet_types": [
            "- Hoạt động theo cơ chế hàng đợi xoay vòng với lát cắt thời gian cố định (Time Quantum).",
            "- Hết Time Quantum sẽ phát sinh ngắt đồng hồ để điều phối tiến trình tiếp theo.",
            "- Quantum quá lớn: Suy biến thành thuật toán FCFS.",
            "- Quantum quá nhỏ: Gánh nặng chuyển ngữ cảnh (Context Switch Overhead) làm giảm hiệu suất CPU.",
        ],
        "example": "Nếu một tiến trình cần 12ms và Time Quantum q = 10ms, tiến trình sẽ chạy 10ms trong lượt 1, bị ngắt đưa về cuối hàng đợi, và chạy nốt 2ms ở lượt 2.",
    },
    {
        "id": "os_qa_05",
        "topic": "Hiện tượng Bế tắc (Deadlock) và 4 Điều kiện Coffman",
        "category": "deadlock_conditions",
        "canonical_terms": [
            "deadlock",
            "bế tắc",
            "4 điều kiện coffman",
            "mutual exclusion",
            "hold and wait",
            "no preemption",
            "circular wait",
        ],
        "question": "Deadlock (Bế tắc) là gì và nêu rõ 4 điều kiện Coffman dẫn đến Deadlock?",
        "intent": "deadlock_definition_and_coffman_conditions",
        "expected_action": "explain_deadlock_and_four_conditions",
        "grounded_answer": (
            "Deadlock (Bế tắc) là trạng thái trong hệ thống đa chương trình mà một tập hợp các tiến trình bị ngừng trệ vĩnh viễn, "
            "bởi vì mỗi tiến trình đều đang nắm giữ một tài nguyên và đồng thời đang chờ cấp phát tài nguyên khác mà một tiến trình khác trong chính tập hợp đó đang chiếm giữ.\n\n"
            "Theo công bố của E. G. Coffman (1971), Deadlock chỉ có thể xảy ra khi và chỉ khi cả 4 điều kiện sau đồng thời xuất hiện:\n"
            "1. Loại trừ lẫn nhau (Mutual Exclusion): Ít nhất một tài nguyên phải ở trạng thái không thể chia sẻ (Non-shareable), nghĩa là tại một thời điểm chỉ có duy nhất một tiến trình được sử dụng tài nguyên đó.\n"
            "2. Giữ và chờ (Hold and Wait): Tiến trình đang nắm giữ ít nhất một tài nguyên nhưng lại gửi yêu cầu xin cấp phát thêm tài nguyên mới đang bị tiến trình khác nắm giữ.\n"
            "3. Không giải tỏa / Không trưng dụng (No Preemption): Tài nguyên không thể bị tước đoạt cưỡng chế khỏi tiến trình đang giữ nó; tài nguyên chỉ được giải phóng tự nguyện khi tiến trình đã hoàn thành tác vụ.\n"
            "4. Chờ đợi vòng tròn (Circular Wait): Tồn tại một chuỗi khép kín các tiến trình {P0, P1, ..., Pn} sao cho P0 đang chờ tài nguyên do P1 giữ, P1 chờ tài nguyên do P2 giữ, ..., và Pn lại đang chờ tài nguyên do P0 nắm giữ."
        ),
        "bullet_types": [
            "- Deadlock: Hiện tượng các tiến trình chờ đợi lẫn nhau vô hạn định.",
            "- Điều kiện 1: Loại trừ tương hỗ (Mutual Exclusion).",
            "- Điều kiện 2: Giữ và chờ (Hold and Wait).",
            "- Điều kiện 3: Không thể trưng dụng cưỡng chế (No Preemption).",
            "- Điều kiện 4: Chờ đợi vòng tròn khép kín (Circular Wait).",
        ],
        "example": "Tiến trình P1 đang giữ Ổ ghi đĩa và yêu cầu Máy in, trong khi Tiến trình P2 đang giữ Máy in và yêu cầu Ổ ghi đĩa. Cả hai chờ nhau mãi mãi tạo thành chuỗi vòng tròn.",
    },
    {
        "id": "os_qa_06",
        "topic": "Xử lý Deadlock: Phòng ngừa, Tránh né (Banker) và Phát hiện",
        "category": "deadlock_handling",
        "canonical_terms": [
            "xử lý deadlock",
            "deadlock prevention",
            "deadlock avoidance",
            "thuật toán banker",
            "banker's algorithm",
            "safe state",
            "resource allocation graph",
        ],
        "question": "Trình bày các phương pháp xử lý Deadlock trong hệ điều hành và nguyên lý thuật toán Banker?",
        "intent": "deadlock_strategies_and_banker_algorithm",
        "expected_action": "explain_deadlock_prevention_avoidance_banker",
        "grounded_answer": (
            "Hệ điều hành có 4 chiến lược chính để ứng phó với Deadlock:\n\n"
            "1. Bỏ qua (Ostrich Algorithm - Đà điểu vùi đầu vào cát): Giả định deadlock hiếm khi xảy ra và chi phí giải quyết đắt hơn thiệt hại. Được nhiều hệ điều hành phổ thông (Unix/Windows) áp dụng trong một số trường hợp người dùng.\n"
            "2. Phòng ngừa Deadlock (Deadlock Prevention): Đưa ra các ràng buộc nghiêm ngặt để triệt tiêu ít nhất 1 trong 4 điều kiện Coffman (ví dụ: gán số thứ tự cho mọi tài nguyên và bắt buộc xin tài nguyên theo thứ tự tăng dần để phá vỡ Circular Wait).\n"
            "3. Tránh Deadlock (Deadlock Avoidance) & Thuật toán Banker:\n"
            "- Hệ điều hành chỉ cấp phát tài nguyên khi kiểm tra trạng thái sau cấp phát vẫn là Trạng thái An toàn (Safe State - tồn tại một Chuỗi An toàn để mọi tiến trình đều có thể hoàn thành).\n"
            "- Thuật toán Banker (Dijkstra): Quản lý các ma trận Available (tài nguyên sẵn có), Max (nhu cầu tối đa), Allocation (tài nguyên đã cấp), và Need = Max - Allocation. Khi tiến trình yêu cầu tài nguyên, thuật toán giả lập cấp phát; nếu hệ thống vẫn tìm được chuỗi an toàn thì mới cấp thật, ngược lại bắt tiến trình phải chờ.\n"
            "4. Phát hiện và Phục hồi (Deadlock Detection & Recovery): Định kỳ xây dựng đồ thị phân bổ tài nguyên (RAG) hoặc đồ thị Wait-For; nếu phát hiện chu trình, hệ điều hành tiến hành hủy tiến trình (kill process) hoặc tước đoạt tài nguyên (resource preemption) để phá vỡ bế tắc."
        ),
        "bullet_types": [
            "- 4 chiến lược: Bỏ qua (Ostrich), Phòng ngừa (Prevention), Tránh né (Avoidance), Phát hiện & Phục hồi.",
            "- Phòng ngừa: Triệt tiêu ít nhất 1 trong 4 điều kiện Coffman.",
            "- Thuật toán Banker: Đảm bảo hệ thống luôn nằm trong Trạng thái An toàn (Safe State).",
            "- Công thức kiểm tra Banker: Need = Max - Allocation <= Available.",
        ],
        "example": "Hệ thống có 10 ổ băng từ (Available=3, cấp cho P0=5, P1=2). Khi P1 xin thêm 2 ổ, hệ thống kiểm tra nếu cấp thì P1 hoàn thành xong sẽ trả lại đủ 4 ổ cho P0 hoàn tất -> Thuộc Safe State nên chấp thuận.",
    },
    {
        "id": "os_qa_07",
        "topic": "Bộ nhớ ảo (Virtual Memory) và Lỗi trang (Page Fault)",
        "category": "virtual_memory",
        "canonical_terms": [
            "bộ nhớ ảo",
            "virtual memory",
            "page fault",
            "lỗi trang",
            "demand paging",
            "swap space",
            "tlb",
        ],
        "question": "Bộ nhớ ảo (Virtual Memory) là gì và quy trình xử lý khi xảy ra Lỗi trang (Page Fault)?",
        "intent": "virtual_memory_concept_and_page_fault_handling",
        "expected_action": "explain_virtual_memory_and_page_fault_routine",
        "grounded_answer": (
            "Bộ nhớ ảo (Virtual Memory) là kỹ thuật cho phép tiến trình thực thi mà không cần toàn bộ không gian địa chỉ phải nằm liên tục trong bộ nhớ vật lý RAM. "
            "Nhờ đó, kích thước bộ nhớ logic của chương trình có thể vượt xa dung lượng RAM vật lý hiện có của máy tính.\n\n"
            "Cơ chế Nạp trang theo yêu cầu (Demand Paging):\n"
            "Các trang nhớ chỉ được nạp từ đĩa cứng (Swap Space / Page File) vào RAM khi CPU thực sự có chỉ thị truy cập tới chúng. "
            "Bảng trang sử dụng một bit hợp lệ (Valid/Invalid Bit) để đánh dấu trang đang nằm trong RAM (v) hay còn nằm trên đĩa (i).\n\n"
            "Quy trình xử lý Lỗi trang (Page Fault Handling Routine):\n"
            "1. CPU truy cập địa chỉ ảo; phần cứng MMU tra cứu bảng trang và phát hiện bit 'invalid' (trang chưa có trong RAM), lập tức phát sinh một tín hiệu ngắt phần mềm (Page Fault Trap) chuyển quyền điều khiển cho Hệ điều hành.\n"
            "2. Hệ điều hành kiểm tra tính hợp lệ của địa chỉ truy cập trong cấu trúc quản lý tiến trình:\n"
            "   - Nếu địa chỉ bất hợp pháp (ngoài vùng nhớ được cấp): Hủy tiến trình (báo lỗi Segmentation Fault).\n"
            "   - Nếu địa chỉ hợp lệ: Chuẩn bị nạp trang vào RAM.\n"
            "3. Hệ điều hành tìm một Khung trang trống (Free Frame) trong RAM. Nếu không còn khung trống, thực thi Thuật toán thay thế trang (Page Replacement) để chọn trang nạn nhân (Victim Page) ghi về đĩa.\n"
            "4. Hệ điều hành phát lệnh I/O đọc dữ liệu trang từ đĩa (Swap Space) nạp vào khung trang vừa chọn.\n"
            "5. Cập nhật lại Bảng trang: Ghi chỉ số frame mới và chuyển bit từ 'invalid' sang 'valid'.\n"
            "6. Khôi phục trạng thái tiến trình và thực thi lại đúng chỉ thị lệnh vừa bị ngắt trước đó."
        ),
        "bullet_types": [
            "- Bộ nhớ ảo cho phép mở rộng không gian địa chỉ vượt qua dung lượng RAM vật lý.",
            "- Bit Valid/Invalid trong bảng trang đánh dấu vị trí trang.",
            "- Page Fault là ngắt bẫy (trap) xảy ra khi truy cập trang mang bit invalid.",
            "- Quy trình 6 bước: Trap -> Kiểm tra -> Tìm Free Frame -> Đọc I/O đĩa -> Cập nhật Page Table -> Khởi động lại lệnh.",
        ],
        "example": "Một máy tính có 4GB RAM vẫn có thể mở một file đồ họa 16GB nhờ bộ nhớ ảo chỉ nạp những mảng pixel đang hiển thị trên màn hình vào RAM.",
    },
    {
        "id": "os_qa_08",
        "topic": "Phân trang (Paging) và Phân đoạn (Segmentation)",
        "category": "memory_management",
        "canonical_terms": [
            "phân trang",
            "phân đoạn",
            "paging",
            "segmentation",
            "internal fragmentation",
            "external fragmentation",
            "frame",
        ],
        "question": "So sánh phương pháp Phân trang (Paging) và Phân đoạn (Segmentation)?",
        "intent": "memory_allocation_comparison",
        "expected_action": "compare_paging_and_segmentation",
        "grounded_answer": (
            "Phân trang và Phân đoạn là hai kỹ thuật quản lý bộ nhớ phi liên tục cốt lõi trong hệ điều hành hiện đại:\n\n"
            "1. Kỹ thuật Phân trang (Paging):\n"
            "- Cơ chế: Chia không gian địa chỉ logic thành các khối kích thước cố định bằng nhau gọi là Trang (Pages, ví dụ 4KB). Bộ nhớ vật lý cũng được chia thành các khối kích thước tương đương gọi là Khung trang (Frames).\n"
            "- Cấu trúc địa chỉ: Gồm số hiệu trang (Page number - p) và độ dời (Offset - d).\n"
            "- Phân mảnh: Hoàn toàn loại bỏ phân mảnh ngoại vi (External Fragmentation), nhưng vẫn tồn tại Phân mảnh nội vi (Internal Fragmentation) ở trang cuối cùng của tiến trình.\n"
            "- Quan điểm: Thuần túy là cơ chế kỹ thuật phần cứng, trong suốt đối với lập trình viên.\n\n"
            "2. Kỹ thuật Phân đoạn (Segmentation):\n"
            "- Cơ chế: Chia bộ nhớ theo các đơn vị logic có ý nghĩa ngữ nghĩa đối với lập trình viên (Code segment, Data segment, Stack segment, Symbol table) với kích thước biến thiên không cố định.\n"
            "- Cấu trúc địa chỉ: Gồm số hiệu đoạn (Segment number - s) và độ dời (Offset - d), đối chiếu với Segment Table qua Base và Limit.\n"
            "- Phân mảnh: Không có phân mảnh nội vi, nhưng dễ gây ra Phân mảnh ngoại vi (External Fragmentation) nghiêm trọng khi các đoạn được nạp/giải phóng liên tục.\n"
            "- Quan điểm: Phù hợp với góc nhìn logic và cấu trúc module của người lập trình.\n\n"
            "Hệ thống hiện đại (x86/x64) thường kết hợp cả hai mô hình (Segmented Paging): Phân đoạn ở tầng logic và Phân trang ở tầng vật lý bên dưới."
        ),
        "bullet_types": [
            "- Phân trang chia khối kích thước cố định; Phân đoạn chia theo khối logic kích thước biến thiên.",
            "- Phân trang bị Phân mảnh nội vi (Internal); Phân đoạn bị Phân mảnh ngoại vi (External).",
            "- Ánh xạ địa chỉ: Paging dùng Page Table (p, d); Segmentation dùng Segment Table (Base, Limit).",
            "- Hệ thống hiện đại kết hợp Phân đoạn có Phân trang (Segmented Paging).",
        ],
        "example": "Trang 4KB cấp cho dữ liệu 1KB sẽ lãng phí 3KB bên trong khung (Phân mảnh nội vi). Phân đoạn chia đoạn Code 20KB và Stack 5KB liên tiếp có thể để lại khoảng trống 2KB giữa chừng không dùng được cho tiến trình khác (Phân mảnh ngoại vi).",
    },
    {
        "id": "os_qa_09",
        "topic": "Các thuật toán thay thế trang và Hiện tượng Belady",
        "category": "page_replacement",
        "canonical_terms": [
            "thay thế trang",
            "page replacement",
            "fifo",
            "lru",
            "optimal",
            "bất thường belady",
            "belady anomaly",
        ],
        "question": "Trình bày các thuật toán thay thế trang FIFO, LRU, Optimal và hiện tượng dị thường Belady?",
        "intent": "page_replacement_algorithms_and_belady_anomaly",
        "expected_action": "explain_page_replacement_and_belady",
        "grounded_answer": (
            "Khi xảy ra lỗi trang và bộ nhớ RAM không còn khung trang trống, hệ điều hành phải áp dụng thuật toán thay thế trang để chọn một trang làm nạn nhân (Victim Page) đưa về đĩa:\n\n"
            "1. Thuật toán FIFO (First-In, First-Out):\n"
            "- Nguyên lý: Trang nào được nạp vào bộ nhớ trước tiên sẽ bị chọn để thay thế đầu tiên (dùng cấu trúc hàng đợi Queue).\n"
            "- Ưu điểm: Đơn giản, chi phí tính toán cực thấp.\n"
            "- Nhược điểm: Có thể thay thế nhầm các trang khởi tạo dùng thường xuyên, và mắc phải Hiện tượng dị thường Belady.\n\n"
            "2. Thuật toán Tối ưu (Optimal - OPT / MIN):\n"
            "- Nguyên lý: Thay thế trang mà trong tương lai xa nhất sẽ không còn được sử dụng tới nữa.\n"
            "- Ưu điểm: Đảm bảo số lần lỗi trang thấp nhất tuyệt đối trong mọi trường hợp.\n"
            "- Nhược điểm: Không thể hiện thực trong thực tế vì hệ điều hành không thể biết trước tương lai chuỗi truy cập bộ nhớ của chương trình. Thường chỉ dùng làm chuẩn đối sánh (benchmark).\n\n"
            "3. Thuật toán LRU (Least Recently Used):\n"
            "- Nguyên lý: Xấp xỉ thuật toán Optimal bằng cách nhìn về quá khứ: Thay thế trang có thời gian chưa được truy cập lâu nhất.\n"
            "- Ưu điểm: Hiệu năng cao, tiệm cận tối ưu, không mắc dị thường Belady.\n"
            "- Nhược điểm: Chi phí phần cứng cao (cần bộ đếm thời gian hoặc danh sách liên kết kép cập nhật trên mỗi lần truy cập bộ nhớ).\n\n"
            "Hiện tượng dị thường Belady (Belady's Anomaly):\n"
            "Là hiện tượng nghịch lý xảy ra trong thuật toán FIFO: Khi cấp thêm số khung trang RAM cho tiến trình, số lần xảy ra lỗi trang (Page Fault) lại TĂNG LÊN thay vì giảm đi. "
            "Nguyên nhân vì FIFO không thuộc nhóm Thuật toán ngăn xếp (Stack Algorithms) như LRU hay Optimal."
        ),
        "bullet_types": [
            "- FIFO: Thay trang vào trước tiên; đơn giản nhưng dễ mắc dị thường Belady.",
            "- Optimal: Thay trang xa nhất trong tương lai; tối ưu tuyệt đối nhưng chỉ dùng làm chuẩn lý thuyết.",
            "- LRU: Thay trang lâu nhất chưa được dùng trong quá khứ; hiệu năng cao, không bị Belady.",
            "- Dị thường Belady: Tăng số khung trang nhưng lại làm tăng số lần lỗi trang trong thuật toán FIFO.",
        ],
        "example": "Với chuỗi truy cập: 1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5. Với 3 khung trang FIFO sinh ra 9 lỗi trang, nhưng khi tăng lên 4 khung trang FIFO lại sinh ra 10 lỗi trang (Dị thường Belady).",
    },
    {
        "id": "os_qa_10",
        "topic": "Vấn đề Miền găng (Critical Section) và Điều kiện Đồng bộ",
        "category": "synchronization",
        "canonical_terms": [
            "miền găng",
            "đoạn găng",
            "critical section",
            "race condition",
            "mutual exclusion",
            "progress",
            "bounded waiting",
        ],
        "question": "Miền găng (Critical Section) là gì và nêu rõ 3 yêu cầu bắt buộc để giải quyết bài toán miền găng?",
        "intent": "critical_section_requirements",
        "expected_action": "explain_critical_section_and_three_requirements",
        "grounded_answer": (
            "Trong môi trường đa luồng / đa tiến trình chia sẻ dữ liệu chung:\n\n"
            "1. Tình trạng chạy đua (Race Condition):\n"
            "Là tình trạng khi nhiều tiến trình hoặc luồng cùng truy cập và thao tác đồng thời trên dữ liệu chia sẻ, "
            "khiến kết quả cuối cùng phụ thuộc vào thứ tự thực thi ngẫu nhiên của các lệnh. Điều này dẫn đến dữ liệu bị sai lệch, không nhất quán.\n\n"
            "2. Miền găng (Critical Section):\n"
            "Là đoạn mã lệnh trong chương trình có thực hiện truy cập và thay đổi tài nguyên chia sẻ (biến chung, tệp tin, bảng dữ liệu). "
            "Để ngăn ngừa Race Condition, tại một thời điểm chỉ cho phép duy nhất một tiến trình được thực thi trong miền găng của nó.\n\n"
            "3. Ba yêu cầu bắt buộc cho mọi giải pháp giải quyết bài toán miền găng:\n"
            "- Loại trừ tương hỗ (Mutual Exclusion): Nếu tiến trình Pi đang thực thi trong miền găng của nó, thì không có tiến trình Pj nào khác được phép thực thi trong miền găng.\n"
            "- Tiến triển (Progress): Nếu không có tiến trình nào đang ở trong miền găng và có một số tiến trình muốn vào miền găng, thì chỉ những tiến trình không nằm trong phần còn lại (Remainder Section) mới được tham gia quyết định tiến trình tiếp theo được vào, và quyết định này không được trì hoãn vô hạn.\n"
            "- Chờ đợi có giới hạn (Bounded Waiting): Phải tồn tại một giới hạn xác định về số lần các tiến trình khác được phép vào miền găng sau khi một tiến trình đã đưa ra yêu cầu vào miền găng, nhằm đảm bảo không tiến trình nào bị đói tài nguyên (Starvation) vĩnh viễn."
        ),
        "bullet_types": [
            "- Race Condition: Sự không nhất quán dữ liệu do thực thi đồng thời.",
            "- Miền găng: Khối lệnh thao tác trên tài nguyên chia sẻ chung.",
            "- Yêu cầu 1: Loại trừ lẫn nhau (Mutual Exclusion) - chỉ 1 tiến trình vào cùng lúc.",
            "- Yêu cầu 2: Tiến triển (Progress) - việc chọn tiến trình tiếp theo không bị bế tắc vô hạn.",
            "- Yêu cầu 3: Chờ đợi có giới hạn (Bounded Waiting) - ngăn chặn nạn đói tài nguyên.",
        ],
        "example": "Hai luồng cùng thực hiện lệnh `balance = balance + 100` tại cùng một thời điểm mà không đồng bộ miền găng có thể ghi đè kết quả của nhau dẫn đến mất tiền trong tài khoản.",
    },
    {
        "id": "os_qa_11",
        "topic": "Khóa Mutex và Đèn hiệu Semaphore",
        "category": "synchronization_primitives",
        "canonical_terms": [
            "mutex",
            "semaphore",
            "mutex lock",
            "counting semaphore",
            "binary semaphore",
            "wait",
            "signal",
            "p và v",
        ],
        "question": "Phân biệt Khóa Mutex và Semaphore (Binary & Counting)? Nguyên lý hoạt động của Wait() và Signal()?",
        "intent": "mutex_semaphore_comparison_and_operations",
        "expected_action": "explain_mutex_and_semaphore_mechanisms",
        "grounded_answer": (
            "Khóa Mutex và Đèn hiệu Semaphore là các công cụ đồng bộ hóa mức hệ điều hành nhằm giải quyết bài toán miền găng:\n\n"
            "1. Khóa Mutex (Mutual Exclusion Lock):\n"
            "- Bản chất: Là biến khóa nhị phân (0 hoặc 1) có khái niệm Quyền sở hữu (Ownership). Tiến trình nào khóa Mutex (acquire / lock) thì bắt buộc chỉ chính tiến trình đó mới có quyền mở khóa (release / unlock).\n"
            "- Mục đích: Dùng thuần túy để bảo vệ một vùng găng đơn lẻ chống Race Condition.\n\n"
            "2. Đèn hiệu Semaphore (Dijkstra đề xuất):\n"
            "- Bản chất: Là một biến nguyên S được khởi tạo giá trị ban đầu và chỉ có thể truy cập qua 2 thao tác nguyên tử chuẩn là `wait(S)` (hay `P(S)`) và `signal(S)` (hay `V(S)`).\n"
            "- Hai phân loại Semaphore:\n"
            "  * Binary Semaphore: Biến S chỉ nhận giá trị 0 hoặc 1 (hoạt động tương tự Mutex nhưng không bắt buộc ràng buộc sở hữu).\n"
            "  * Counting Semaphore: Biến S có thể nhận giá trị nguyên không âm bất kỳ, dùng để quản lý một nhóm tài nguyên hữu hạn gồm n bản sao giống nhau.\n\n"
            "3. Cơ chế hoạt động của Wait() và Signal():\n"
            "- Thao tác `wait(S)`: Giảm giá trị S đi 1. Nếu giá trị sau khi giảm S < 0, tiến trình gọi lệnh sẽ bị chặn và đưa vào hàng đợi chờ (Waiting Queue) của Semaphore.\n"
            "- Thao tác `signal(S)`: Tăng giá trị S lên 1. Nếu giá trị S <= 0 (nghĩa là vẫn còn ít nhất một tiến trình đang bị khóa trong hàng đợi), hệ điều hành đánh thức (Wakeup) một tiến trình đang chờ đưa về hàng đợi Ready."
        ),
        "bullet_types": [
            "- Mutex: Khóa nhị phân có ràng buộc sở hữu (ai khóa người đó phải mở).",
            "- Binary Semaphore: Giá trị 0 hoặc 1, không ràng buộc người giải phóng.",
            "- Counting Semaphore: Quản lý pool tài nguyên gồm n thực thể giống nhau.",
            "- Wait() giảm biến đếm, chặn nếu thiếu tài nguyên; Signal() tăng biến đếm và đánh thức tiến trình chờ.",
        ],
        "example": "Bãi đỗ xe có 5 chỗ trống dùng Counting Semaphore S = 5. Mỗi xe vào gọi wait(S) làm S giảm. Khi có 5 xe S=0, xe thứ 6 gọi wait(S) sẽ bị chặn chờ. Khi 1 xe ra gọi signal(S), xe thứ 6 được vào.",
    },
    {
        "id": "os_qa_12",
        "topic": "Các bài toán Đồng bộ kinh điển",
        "category": "classic_problems",
        "canonical_terms": [
            "bài toán đồng bộ kinh điển",
            "producer consumer",
            "nhà sản xuất người tiêu dùng",
            "dining philosophers",
            "bữa ăn triết gia",
            "readers writers",
        ],
        "question": "Trình bày Bài toán Nhà sản xuất - Người tiêu dùng (Producer-Consumer) và cách giải quyết bằng Semaphore?",
        "intent": "classic_problem_producer_consumer",
        "expected_action": "explain_producer_consumer_problem_and_semaphore_solution",
        "grounded_answer": (
            "Bài toán Nhà sản xuất - Người tiêu dùng (Producer-Consumer / Bounded-Buffer Problem) là một trong những bài toán đồng bộ tiến trình kinh điển nhất trong hệ điều hành:\n\n"
            "1. Mô tả bài toán:\n"
            "- Hai tiến trình chia sẻ chung một bộ đệm có kích thước cố định N phần tử.\n"
            "- Producer (Nhà sản xuất): Liên tục tạo dữ liệu và đưa vào bộ đệm. Nếu bộ đệm đã đầy (Full), Producer phải chờ cho đến khi có ít nhất một ô trống.\n"
            "- Consumer (Người tiêu dùng): Liên tục lấy dữ liệu ra khỏi bộ đệm để xử lý. Nếu bộ đệm đang rỗng (Empty), Consumer phải chờ cho đến khi có ít nhất một phần tử dữ liệu.\n"
            "- Cả hai không được phép đồng thời truy cập thao tác trên cùng một con trỏ bộ đệm (tránh Race Condition).\n\n"
            "2. Giải pháp sử dụng Semaphore chuẩn:\n"
            "Cần sử dụng 3 Semaphore:\n"
            "- `mutex` (Binary Semaphore, khởi tạo = 1): Đảm bảo loại trừ tương hỗ khi thêm hoặc lấy phần tử khỏi bộ đệm.\n"
            "- `empty` (Counting Semaphore, khởi tạo = N): Đếm số lượng ô nhớ còn trống trong bộ đệm.\n"
            "- `full` (Counting Semaphore, khởi tạo = 0): Đếm số lượng ô nhớ đã có dữ liệu trong bộ đệm.\n\n"
            "3. Quy trình thực thi mã giả:\n"
            "Producer:\n"
            "  wait(empty);\n"
            "  wait(mutex);\n"
            "  // thêm dữ liệu vào buffer\n"
            "  signal(mutex);\n"
            "  signal(full);\n\n"
            "Consumer:\n"
            "  wait(full);\n"
            "  wait(mutex);\n"
            "  // lấy dữ liệu ra khỏi buffer\n"
            "  signal(mutex);\n"
            "  signal(empty);\n\n"
            "Lưu ý: Thứ tự các lệnh wait là cực kỳ quan trọng; nếu đảo ngược `wait(mutex)` lên trước `wait(empty)` hoặc `wait(full)` có thể dẫn đến bế tắc (Deadlock)."
        ),
        "bullet_types": [
            "- Bài toán Bounded-Buffer: Đồng bộ bộ đệm giới hạn N phần tử giữa Producer và Consumer.",
            "- 3 Semaphore: mutex=1 (bảo vệ miền găng), empty=N (đếm ô trống), full=0 (đếm ô có dữ liệu).",
            "- Producer: wait(empty) -> wait(mutex) -> ghi -> signal(mutex) -> signal(full).",
            "- Consumer: wait(full) -> wait(mutex) -> đọc -> signal(mutex) -> signal(empty).",
        ],
        "example": "Hàng đợi Message Queue giữa Web Server (Producer sinh request) và Worker Task (Consumer xử lý job) giao tiếp qua bộ đệm RAM giới hạn 1000 items.",
    },
    {
        "id": "os_qa_13",
        "topic": "Hiện tượng Thrashing (Trì trệ bộ nhớ) và Mô hình Working-Set",
        "category": "virtual_memory_performance",
        "canonical_terms": [
            "thrashing",
            "trì trệ bộ nhớ",
            "đảo trang liên tục",
            "working-set",
            "bộ nhớ ảo",
            "page fault",
        ],
        "question": "Hiện tượng Thrashing là gì, nguyên nhân gây ra và hệ điều hành khắc phục bằng mô hình Working-Set như thế nào?",
        "intent": "thrashing_concept_and_working_set_solution",
        "expected_action": "explain_thrashing_and_working_set_model",
        "grounded_answer": (
            "Hiện tượng Thrashing (Trì trệ bộ nhớ do đảo trang liên tục):\n\n"
            "1. Định nghĩa và Biểu hiện:\n"
            "Thrashing là tình trạng hệ thống dành phần lớn thời gian để thực hiện thao tác tráo đổi trang (Paging / Swapping) vào ra giữa RAM và đĩa cứng, "
            "thay vì thực thi các chỉ thị hữu ích của chương trình. Khi Thrashing xảy ra, tỷ lệ lỗi trang (Page Fault Rate) tăng vọt, hiệu suất CPU (CPU Utilization) sụt giảm thê thảm, và hệ thống rơi vào trạng thái đơ cứng (freeze/hang).\n\n"
            "2. Nguyên nhân cốt lõi:\n"
            "- Hệ điều hành nhận thấy CPU Utilization thấp nên tiếp tục tăng mức độ đa chương (Degree of Multiprogramming) bằng cách nạp thêm tiến trình mới vào RAM.\n"
            "- Tổng số trang tích cực cần thiết của tất cả các tiến trình vượt quá số khung trang vật lý hiện có của RAM vật lý.\n"
            "- Mỗi tiến trình không được cấp đủ số khung trang tối thiểu cần thiết để chứa Tập làm việc (Locality) của nó, dẫn đến việc vừa nạp trang mới vào thì đã phải đẩy trang cũ ra và lập tức lại lỗi trang ngay sau đó.\n\n"
            "3. Giải pháp khắc phục bằng Mô hình Working-Set (Peter Denning):\n"
            "- Dựa trên Nguyên lý cục bộ (Locality of Reference): Trong một khoảng thời gian nhất định, tiến trình chỉ tập trung truy cập một tập con các trang nhất định.\n"
            "- Hệ điều hành định nghĩa một cửa sổ làm việc (Working-Set Window - Δ), tính toán tập hợp các trang được truy cập trong Δ thời gian qua gọi là Working-Set WSSi.\n"
            "- Hệ điều hành tính tổng nhu cầu: D = Σ WSSi.\n"
            "- Chiến lược điều phối:\n"
            "  * Nếu D <= Tổng số khung RAM khả dụng: Hệ thống hoạt động an toàn, không có Thrashing.\n"
            "  * Nếu D > Tổng số khung RAM khả dụng: Hệ điều hành chủ động tạm dừng (Suspend / Swap-out) một hoặc nhiều tiến trình đưa về đĩa cứng, thu hồi toàn bộ khung trang của chúng phân bổ cho các tiến trình còn lại để đảm bảo mỗi tiến trình đang chạy đều đủ trang làm việc."
        ),
        "bullet_types": [
            "- Thrashing: Hiện tượng hệ thống chỉ bận rộn tráo đổi trang nhớ qua lại ổ đĩa khiến CPU bị nghẽn.",
            "- Nguyên nhân: Mức độ đa chương quá cao khiến tổng nhu cầu bộ nhớ vượt dung lượng RAM vật lý.",
            "- Giải pháp Working-Set: Xác định tập trang cục bộ WSSi trong cửa sổ thời gian Δ.",
            "- Khi tổng nhu cầu D > RAM: Tạm ngưng (suspend) một số tiến trình để giải phóng RAM cho tiến trình còn lại.",
        ],
        "example": "Mở đồng thời 50 tab nặng trên máy tính 4GB RAM khiến đèn ổ cứng sáng liên tục, chuột giật lag không phản hồi, CPU chạy dưới 5% vì nghẽn ngắt I/O ổ đĩa.",
    },
]
