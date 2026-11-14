"""
Domain-Specific Computer Networking & Protocols Semantic Patterns.
Provides bidirectional synonym mapping, domain query expansion, and few-shot QA blueprints
for authoritative computer networking educational slides and curricula (OSI 7 Layers, TCP/IP, UDP,
HTTP/HTTPS, DNS, DHCP, OSPF/BGP Routing, Firewalls & NAT).
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


# Core Computer Networking & Protocols Synonym Clusters
NETWORKING_SYNONYM_CLUSTERS: Dict[str, List[str]] = {
    "osi_model": [
        "mô hình osi",
        "osi 7 tầng",
        "mô hình 7 tầng osi",
        "mô hình osi 7 lớp",
        "osi model",
        "7 layer osi",
        "open systems interconnection",
        "mô hình tham chiếu osi",
        "7 tầng osi",
        "7 lớp osi",
        "chồng giao thức osi",
    ],
    "osi_physical_layer": [
        "tầng vật lý",
        "lớp vật lý",
        "physical layer",
        "layer 1",
        "tầng 1",
        "lớp 1",
        "bit stream",
        "dòng bit",
        "tín hiệu điện",
        "cáp mạng",
        "cáp xoắn đôi",
        "cáp quang",
        "repeater",
        "hub",
    ],
    "osi_data_link_layer": [
        "tầng liên kết dữ liệu",
        "lớp liên kết dữ liệu",
        "data link layer",
        "layer 2",
        "tầng 2",
        "lớp 2",
        "khung dữ liệu",
        "frame",
        "địa chỉ mac",
        "mac address",
        "switch",
        "bộ chuyển mạch",
        "ethernet",
        "csma/cd",
        "arp",
        "giao thức arp",
    ],
    "osi_network_layer": [
        "tầng mạng",
        "lớp mạng",
        "network layer",
        "layer 3",
        "tầng 3",
        "lớp 3",
        "gói tin",
        "packet",
        "địa chỉ ip",
        "ip address",
        "router",
        "bộ định tuyến",
        "định tuyến",
        "routing",
        "icmp",
        "giao thức icmp",
    ],
    "osi_transport_layer": [
        "tầng giao vận",
        "lớp giao vận",
        "tầng vận chuyển",
        "lớp vận chuyển",
        "transport layer",
        "layer 4",
        "tầng 4",
        "lớp 4",
        "phân đoạn dữ liệu",
        "segment",
        "tcp",
        "udp",
        "port",
        "cổng dịch vụ",
        "port number",
        "số hiệu cổng",
    ],
    "osi_session_layer": [
        "tầng phiên",
        "lớp phiên",
        "session layer",
        "layer 5",
        "tầng 5",
        "lớp 5",
        "quản lý phiên",
        "thiết lập phiên",
        "session management",
        "rpc",
        "duplex",
        "half-duplex",
        "full-duplex",
    ],
    "osi_presentation_layer": [
        "tầng trình diễn",
        "lớp trình bày",
        "tầng trình bày",
        "presentation layer",
        "layer 6",
        "tầng 6",
        "lớp 6",
        "mã hóa dữ liệu",
        "nén dữ liệu",
        "định dạng dữ liệu",
        "chuyển đổi dữ liệu",
        "ascii",
        "jpeg",
        "tls presentation",
    ],
    "osi_application_layer": [
        "tầng ứng dụng",
        "lớp ứng dụng",
        "application layer",
        "layer 7",
        "tầng 7",
        "lớp 7",
        "giao diện người dùng",
        "dịch vụ mạng người dùng",
        "http",
        "https",
        "ftp",
        "smtp",
        "dns",
        "dhcp",
        "telnet",
        "ssh",
    ],
    "tcp_ip_model": [
        "mô hình tcp/ip",
        "kiến trúc tcp/ip",
        "bộ giao thức tcp/ip",
        "tcp/ip model",
        "tcp/ip protocol suite",
        "chồng giao thức tcp/ip",
        "internet protocol suite",
    ],
    "tcp_protocol": [
        "giao thức tcp",
        "tcp",
        "transmission control protocol",
        "hướng kết nối",
        "connection-oriented",
        "vận chuyển tin cậy",
        "reliable transport",
        "bảo đảm phân phát",
        "kiểm soát lỗi",
        "error control",
        "truyền tin cậy",
    ],
    "three_way_handshake": [
        "bắt tay 3 bước",
        "bắt tay ba bước",
        "three-way handshake",
        "3-way handshake",
        "tcp handshake",
        "syn",
        "syn-ack",
        "ack",
        "thiết lập kết nối tcp",
        "kết nối 3 bước",
    ],
    "tcp_flow_and_congestion_control": [
        "kiểm soát luồng",
        "flow control",
        "cửa sổ trượt",
        "sliding window",
        "receiver window",
        "rwnd",
        "kiểm soát tắc nghẽn",
        "congestion control",
        "congestion window",
        "cwnd",
        "slow start",
        "khởi động chậm",
        "congestion avoidance",
        "né tránh tắc nghẽn",
        "fast retransmit",
        "phát lại nhanh",
        "fast recovery",
        "phục hồi nhanh",
    ],
    "udp_protocol": [
        "giao thức udp",
        "udp",
        "user datagram protocol",
        "phi kết nối",
        "connectionless",
        "vận chuyển không tin cậy",
        "unreliable transport",
        "datagram",
        "best effort",
        "overhead thấp",
        "độ trễ thấp",
        "low latency",
        "không cần bắt tay",
    ],
    "http_protocol": [
        "giao thức http",
        "http",
        "hypertext transfer protocol",
        "giao thức truyền siêu văn bản",
        "http request",
        "http response",
        "http/1.1",
        "http/2",
        "http/3",
        "cổng 80",
        "port 80",
    ],
    "https_protocol": [
        "giao thức https",
        "https",
        "hypertext transfer protocol secure",
        "http an toàn",
        "http bảo mật",
        "ssl/tls",
        "tls handshake",
        "chứng chỉ ssl",
        "cổng 443",
        "port 443",
        "ma hoa https",
    ],
    "http_methods_and_status": [
        "phương thức http",
        "http methods",
        "http verbs",
        "get",
        "post",
        "put",
        "delete",
        "patch",
        "head",
        "options",
        "mã trạng thái http",
        "http status codes",
        "200 ok",
        "201 created",
        "301 moved permanently",
        "400 bad request",
        "401 unauthorized",
        "403 forbidden",
        "404 not found",
        "500 internal server error",
        "502 bad gateway",
    ],
    "dns_system": [
        "hệ thống phân giải tên miền",
        "dịch vụ phân giải tên miền",
        "dns",
        "domain name system",
        "máy chủ tên miền",
        "name server",
        "phân giải tên miền",
        "cổng 53",
        "port 53",
        "dns resolver",
        "recursive query",
        "truy vấn đệ quy",
        "iterative query",
        "truy vấn lặp",
        "root dns",
        "tld dns",
    ],
    "dns_records": [
        "bản ghi dns",
        "dns records",
        "resource record",
        "bản ghi a",
        "a record",
        "bản ghi aaaa",
        "aaaa record",
        "bản ghi cname",
        "cname record",
        "bản ghi mx",
        "mx record",
        "bản ghi ns",
        "ns record",
        "bản ghi ptr",
        "ptr record",
        "bản ghi soa",
        "bản ghi txt",
    ],
    "dhcp_service": [
        "giao thức dhcp",
        "dhcp",
        "dynamic host configuration protocol",
        "cấp phát ip động",
        "tiến trình dora",
        "dora process",
        "dora",
        "dhcp discover",
        "dhcp offer",
        "dhcp request",
        "dhcp ack",
        "ip lease",
        "thời gian thuê ip",
        "cổng 67",
        "cổng 68",
    ],
    "routing_ospf": [
        "định tuyến ospf",
        "ospf",
        "open shortest path first",
        "giao thức ospf",
        "trạng thái liên kết",
        "link-state",
        "thuật toán dijkstra",
        "dijkstra algorithm",
        "shortest path first",
        "spf",
        "area 0",
        "backbone area",
        "lsa",
        "link state advertisement",
        "igp",
    ],
    "routing_bgp": [
        "định tuyến bgp",
        "bgp",
        "border gateway protocol",
        "giao thức bgp",
        "định tuyến vector đường dẫn",
        "path vector",
        "autonomous system",
        "hệ tự trị",
        "as",
        "asn",
        "as number",
        "định tuyến ngoài",
        "egp",
        "ebgp",
        "ibgp",
        "cổng 179",
    ],
    "firewall_security": [
        "tường lửa",
        "firewall",
        "firewalls",
        "lọc gói tin",
        "packet filtering",
        "kiểm tra trạng thái",
        "stateful inspection",
        "stateful firewall",
        "ngfw",
        "tường lửa thế hệ mới",
        "next-generation firewall",
        "dmz",
        "vùng phi quân sự",
        "demilitarized zone",
        "bảng luật tường lửa",
        "firewall rules",
        "access control list",
        "acl",
    ],
    "nat_and_pat": [
        "nat",
        "pat",
        "network address translation",
        "port address translation",
        "dịch địa chỉ mạng",
        "dịch địa chỉ cổng",
        "nat overload",
        "ip private",
        "địa chỉ ip riêng",
        "ip public",
        "địa chỉ ip công cộng",
        "rfc 1918",
    ],
    "subnetting_and_cidr": [
        "chia mạng con",
        "subnetting",
        "cidr",
        "classless inter-domain routing",
        "subnet mask",
        "mặt nạ mạng con",
        "địa chỉ mạng",
        "network address",
        "địa chỉ broadcast",
        "broadcast address",
        "prefix length",
        "tiền tố mạng",
    ],
}


class NetworkingThesaurusRegistry:
    """
    Registry for bidirectional computer networking terminology mappings.
    Supports exact matching, tone-insensitive matching, and greedy phrase extraction.
    """

    def __init__(self, clusters: Optional[Dict[str, List[str]]] = None):
        self.clusters = clusters or NETWORKING_SYNONYM_CLUSTERS
        self._norm_to_cluster: Dict[str, str] = {}
        self._stripped_to_cluster: Dict[str, str] = {}
        self._bidirectional_map: Dict[str, Set[str]] = {}
        self._compiled_keywords: List[Tuple[str, str]] = []  # (raw_term, cluster_name)
        self._build_indexes()

    def _build_indexes(self) -> None:
        """Constructs forward, reverse, and accent-insensitive lookup indexes."""
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

        # Sort terms from longest to shortest for greedy n-gram / substring extraction
        all_terms_with_len.sort(key=lambda item: item[2], reverse=True)
        self._compiled_keywords = [(item[0], item[1]) for item in all_terms_with_len]

    def all_clusters(self) -> Dict[str, List[str]]:
        """Returns the raw synonym dictionary."""
        return self.clusters

    def contains_term(self, term: str) -> bool:
        """Checks if a term exists in any known networking cluster."""
        return self.get_cluster_name(term) is not None

    def get_cluster_name(self, term: str) -> Optional[str]:
        """Finds cluster identifier for a given term (accent & case-insensitive)."""
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
        Extracts networking domain terms from a user query and expands them into synonyms.
        Maintains order, preserves primary terms first, and removes duplicate normalized items.
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
_DEFAULT_NETWORKING_REGISTRY = NetworkingThesaurusRegistry()


def get_networking_synonyms(term: str) -> List[str]:
    """Public helper to get bidirectional computer networking synonyms."""
    return _DEFAULT_NETWORKING_REGISTRY.get_synonyms(term)


def expand_networking_query_terms(query: str) -> List[str]:
    """
    Expands conversational user queries into authoritative technical networking keywords.
    E.g.: 'osi 7 tầng' -> ['mô hình osi', 'osi 7 tầng', 'mô hình 7 tầng osi', 'open systems interconnection', ...]
    """
    return _DEFAULT_NETWORKING_REGISTRY.expand_query(query)


# Authoritative Few-Shot QA Examples for Computer Networks & Protocols
NETWORKING_FEW_SHOT_EXAMPLES: List[Dict[str, Any]] = [
    {
        "id": "net_qa_01",
        "topic": "Mô hình OSI 7 tầng",
        "category": "osi_model",
        "canonical_terms": [
            "mô hình osi",
            "osi 7 tầng",
            "open systems interconnection",
            "pdu",
            "đóng gói dữ liệu",
        ],
        "question": "Mô hình OSI 7 tầng là gì, liệt kê thứ tự các tầng từ 1 đến 7 kèm đơn vị dữ liệu (PDU) và chức năng chính của từng tầng?",
        "intent": "concept_architecture_and_breakdown",
        "expected_action": "explain_osi_layers_and_pdus",
        "grounded_answer": (
            "Mô hình OSI (Open Systems Interconnection) do ISO ban hành năm 1984 là mô hình tham chiếu lý thuyết gồm 7 tầng, "
            "chuẩn hóa cách thức các hệ thống máy tính giao tiếp qua mạng dựa trên nguyên lý đóng gói dữ liệu (Encapsulation).\n\n"
            "Danh sách 7 tầng từ Tầng 1 đến Tầng 7:\n"
            "1. Tầng Vật lý (Physical Layer - Layer 1): PDU là Bit. Truyền tải chuỗi bit thô (0 và 1) qua môi trường vật lý (cáp đồng xoắn đôi, cáp quang, sóng vô tuyến). Thiết bị: Repeater, Hub.\n"
            "2. Tầng Liên kết dữ liệu (Data Link Layer - Layer 2): PDU là Khung dữ liệu (Frame). Truyền dữ liệu tin cậy node-to-node qua cùng liên kết vật lý, định địa chỉ MAC vật lý, phát hiện/sửa lỗi cục bộ (CRC), điều khiển truy cập môi trường (CSMA/CD). Thiết bị: Switch, Bridge, NIC.\n"
            "3. Tầng Mạng (Network Layer - Layer 3): PDU là Gói tin (Packet). Định tuyến đường đi từ nguồn tới đích qua nhiều mạng khác nhau (Routing), định địa chỉ logic (IP Address), phân mảnh gói tin. Giao thức & thiết bị: IPv4, IPv6, ICMP, Router.\n"
            "4. Tầng Giao vận (Transport Layer - Layer 4): PDU là Phân đoạn (Segment đối với TCP) hoặc Datagram (đối với UDP). Cung cấp giao tiếp tiến trình-đến-tiến trình (Process-to-Process) qua số hiệu cổng (Port), kiểm soát luồng (Flow Control), kiểm soát tắc nghẽn (Congestion Control) và truyền tin cậy.\n"
            "5. Tầng Phiên (Session Layer - Layer 5): PDU là Dữ liệu (Data). Thiết lập, duy trì, đồng bộ hóa và giải phóng các phiên làm việc giữa các ứng dụng đối thoại (Full-duplex, Half-duplex), quản lý checkpoint phục hồi.\n"
            "6. Tầng Trình diễn (Presentation Layer - Layer 6): PDU là Dữ liệu (Data). Đảm bảo tính tương thích ngữ nghĩa và cú pháp giữa các hệ thống: định dạng dữ liệu (MIME, ASCII, JPEG), nén dữ liệu và mã hóa/giải mã (Encryption/Decryption, TLS/SSL).\n"
            "7. Tầng Ứng dụng (Application Layer - Layer 7): PDU là Dữ liệu (Data). Cung cấp giao diện trực tiếp cho người dùng và các phần mềm ứng dụng mạng tương tác. Giao thức tiêu biểu: HTTP, HTTPS, DNS, DHCP, FTP, SMTP, SSH."
        ),
        "bullet_types": [
            "- Tầng 1: Physical (Bit) - Truyền tín hiệu vật lý.",
            "- Tầng 2: Data Link (Frame) - Chuyển tiếp MAC, Switch.",
            "- Tầng 3: Network (Packet) - Định tuyến IP, Router.",
            "- Tầng 4: Transport (Segment/Datagram) - TCP/UDP, Port.",
            "- Tầng 5: Session (Data) - Quản lý phiên giao tiếp.",
            "- Tầng 6: Presentation (Data) - Mã hóa, nén, định dạng.",
            "- Tầng 7: Application (Data) - HTTP, DNS, DHCP, FTP.",
        ],
        "example": "Khi duyệt web 'https://example.com', dữ liệu HTML (L7) được mã hóa TLS (L6), đóng thành TCP segment với port 443 (L4), gói IP (L3), khung Ethernet với địa chỉ MAC (L2) và phát thành tín hiệu điện/quang (L1).",
    },
    {
        "id": "net_qa_02",
        "topic": "So sánh TCP và UDP",
        "category": "transport_protocols",
        "canonical_terms": [
            "tcp",
            "udp",
            "transmission control protocol",
            "user datagram protocol",
            "hướng kết nối",
            "phi kết nối",
        ],
        "question": "So sánh giao thức TCP và UDP về cơ chế hoạt động, độ tin cậy, overhead và các trường hợp ứng dụng thực tế?",
        "intent": "protocol_comparison",
        "expected_action": "compare_tcp_and_udp",
        "grounded_answer": (
            "So sánh toàn diện giữa hai giao thức tầng giao vận cốt lõi TCP và UDP:\n\n"
            "1. Thiết lập kết nối:\n"
            "- TCP: Hướng kết nối (Connection-oriented). Phải thiết lập kết nối thông qua tiến trình bắt tay 3 bước (3-Way Handshake: SYN -> SYN-ACK -> ACK) trước khi gửi dữ liệu và giải phóng kết nối bằng 4 bước (FIN-ACK).\n"
            "- UDP: Phi kết nối (Connectionless). Gửi dữ liệu ngay lập tức mà không cần thỏa thuận hay thiết lập phiên trước.\n\n"
            "2. Độ tin cậy & Kiểm soát lỗi:\n"
            "- TCP: Tin cậy tuyệt đối (Reliable). Đảm bảo dữ liệu đến đích đầy đủ, đúng thứ tự tuần tự (Sequence Number), tự động phát hiện mất mát và truyền lại (Retransmission/ACK), kiểm soát luồng (Flow Control) và kiểm soát tắc nghẽn (Congestion Control).\n"
            "- UDP: Không tin cậy (Unreliable / Best-effort). Không cam kết gói tin đến đích, không truyền lại gói tin mất, không kiểm soát thứ tự phân phát, không kiểm soát tắc nghẽn.\n\n"
            "3. Overhead và Tốc độ:\n"
            "- TCP: Overhead lớn. Kích thước tiêu đề (Header) tối thiểu 20 bytes (có thể lên tới 60 bytes với Options). Tốc độ chậm hơn do kiểm tra lỗi và đồng bộ trạng thái.\n"
            "- UDP: Overhead rất nhẹ. Kích thước tiêu đề cố định chỉ 8 bytes (Source Port, Destination Port, Length, Checksum). Tốc độ xử lý tối đa, độ trễ cực thấp.\n\n"
            "4. Ứng dụng thực tế:\n"
            "- TCP: Dành cho các dịch vụ yêu cầu tính toàn vẹn dữ liệu chính xác 100%: Web (HTTP/HTTPS), chuyển file (FTP, SFTP), thư điện tử (SMTP, IMAP, POP3), truy cập dòng lệnh từ xa (SSH).\n"
            "- UDP: Dành cho các ứng dụng ưu tiên thời gian thực, chấp nhận mất mát gói nhỏ mà không ảnh hưởng toàn cục: Streaming video/audio trực tiếp, thoại qua IP (VoIP), game đối kháng online thời gian thực, DNS query, DHCP, giao thức SNMP."
        ),
        "bullet_types": [
            "- TCP: Hướng kết nối, tin cậy, sắp xếp thứ tự, overhead 20B+, dùng cho HTTP/SSH/FTP.",
            "- UDP: Phi kết nối, best-effort, không đảm bảo thứ tự, overhead chỉ 8B, dùng cho VoIP/DNS/Streaming.",
        ],
        "example": "Tải một file nhị phân bắt buộc dùng TCP để không bị hỏng byte; cuộc gọi video Discord/Zoom ưu tiên UDP để khung hình không bị giật lag chờ truyền lại.",
    },
    {
        "id": "net_qa_03",
        "topic": "Tiến trình bắt tay 3 bước của TCP",
        "category": "tcp_handshake",
        "canonical_terms": [
            "bắt tay 3 bước",
            "three-way handshake",
            "tcp handshake",
            "syn",
            "syn-ack",
            "ack",
        ],
        "question": "Trình bày chi tiết tiến trình bắt tay 3 bước (Three-way Handshake) trong TCP để thiết lập kết nối và giải thích ý nghĩa các cờ điều khiển?",
        "intent": "connection_establishment_mechanism",
        "expected_action": "explain_three_way_handshake",
        "grounded_answer": (
            "Tiến trình bắt tay 3 bước (TCP Three-way Handshake) là cơ chế bắt buộc để Client và Server đồng bộ số thứ tự tuần tự ban đầu (ISN - Initial Sequence Number) "
            "và cấp phát bộ đệm tài nguyên trước khi trao đổi dữ liệu.\n\n"
            "3 Bước thực hiện:\n"
            "Bước 1 (SYN từ Client -> Server):\n"
            "- Client gửi gói TCP với cờ SYN = 1, ACK = 0.\n"
            "- Chọn một số tuần tự ngẫu nhiên ban đầu: Sequence Number = x (ISN của Client).\n"
            "- Client chuyển sang trạng thái SYN-SENT.\n\n"
            "Bước 2 (SYN-ACK từ Server -> Client):\n"
            "- Server nhận được SYN, cấp phát tài nguyên và phản hồi gói tin có cờ SYN = 1, ACK = 1.\n"
            "- Chọn số tuần tự ban đầu của Server: Sequence Number = y (ISN của Server).\n"
            "- Xác nhận đã nhận được byte của Client: Acknowledgment Number = x + 1.\n"
            "- Server chuyển sang trạng thái SYN-RECEIVED.\n\n"
            "Bước 3 (ACK từ Client -> Server):\n"
            "- Client nhận được SYN-ACK, gửi lại gói tin xác nhận với cờ ACK = 1, SYN = 0.\n"
            "- Sequence Number = x + 1, Acknowledgment Number = y + 1.\n"
            "- Cả Client và Server chuyển sang trạng thái ESTABLISHED. Tại bước này, gói tin ACK có thể bắt đầu mang kèm dữ liệu ứng dụng (Piggybacking).\n\n"
            "Ý nghĩa các cờ:\n"
            "- SYN (Synchronize): Yêu cầu đồng bộ số thứ tự tuần tự phiên kết nối mới.\n"
            "- ACK (Acknowledgment): Báo hiệu trường Acknowledgment Number có giá trị hợp lệ.\n"
            "- FIN (Finish) & RST (Reset): Dùng để kết thúc kết nối bình thường (4 bước FIN-ACK) hoặc chấm dứt kết nối cưỡng bức."
        ),
        "bullet_types": [
            "- Bước 1: Client -> Server: SYN=1, Seq=x (Client SYN-SENT).",
            "- Bước 2: Server -> Client: SYN=1, ACK=1, Seq=y, Ack=x+1 (Server SYN-RECEIVED).",
            "- Bước 3: Client -> Server: ACK=1, Seq=x+1, Ack=y+1 (Cả hai ESTABLISHED).",
        ],
        "example": "Client chọn Seq=100 gửi SYN. Server nhận được, gửi SYN-ACK với Seq=300, Ack=101. Client gửi ACK Seq=101, Ack=301 kết thúc bắt tay.",
    },
    {
        "id": "net_qa_04",
        "topic": "Kiểm soát luồng và Kiểm soát tắc nghẽn trong TCP",
        "category": "traffic_control",
        "canonical_terms": [
            "kiểm soát luồng",
            "flow control",
            "kiểm soát tắc nghẽn",
            "congestion control",
            "sliding window",
            "cwnd",
            "rwnd",
        ],
        "question": "Phân biệt cơ chế kiểm soát luồng (Flow Control) và kiểm soát tắc nghẽn (Congestion Control) trong giao thức TCP?",
        "intent": "mechanism_differentiation",
        "expected_action": "distinguish_flow_and_congestion_control",
        "grounded_answer": (
            "Kiểm soát luồng và Kiểm soát tắc nghẽn là hai cơ chế độc lập nhưng phối hợp nhịp nhàng trong TCP để bảo vệ cả bên nhận lẫn toàn bộ hạ tầng mạng trung gian.\n\n"
            "1. Kiểm soát luồng (Flow Control):\n"
            "- Mục đích: Ngăn bên gửi (Sender) truyền dữ liệu quá nhanh khiến bộ đệm của bên nhận (Receiver Buffer) bị tràn (bảo vệ bên nhận).\n"
            "- Cơ chế: Cơ chế Cửa sổ trượt (Sliding Window). Bên nhận thông báo dung lượng bộ đệm còn trống của mình qua trường Window Size trong tiêu đề TCP, gọi là Receiver Window (rwnd).\n"
            "- Bên gửi giới hạn số byte dữ liệu chưa được xác nhận không vượt quá rwnd. Nếu rwnd = 0, bên gửi tạm ngừng truyền và gửi định kỳ các gói Zero Window Probe để kiểm tra khi nào bộ đệm được giải phóng.\n\n"
            "2. Kiểm soát tắc nghẽn (Congestion Control):\n"
            "- Mục đích: Ngăn bên gửi bơm quá nhiều dữ liệu vào mạng khiến các thiết bị định tuyến trung gian (Routers/Switches) bị tràn hàng đợi (bảo vệ mạng).\n"
            "- Cơ chế: Bên gửi duy trì biến Cửa sổ tắc nghẽn (Congestion Window - cwnd) dựa trên các thuật toán thích ứng mạng:\n"
            "  * Khởi động chậm (Slow Start): cwnd bắt đầu từ 1-10 MSS, tăng theo hàm mũ (nhân đôi sau mỗi RTT) cho tới khi chạm ngưỡng ssthresh (Slow Start Threshold).\n"
            "  * Né tránh tắc nghẽn (Congestion Avoidance): Tăng tuyến tính (+1 MSS sau mỗi RTT) để thăm dò băng thông an toàn.\n"
            "  * Phát lại nhanh & Phục hồi nhanh (Fast Retransmit & Fast Recovery): Khi nhận liên tiếp 3 Duplicate ACKs, phát lại ngay gói bị mất mà không cần chờ hết hạn bộ đếm Timeout, giảm ssthresh = cwnd / 2.\n\n"
            "3. Quy tắc phát dữ liệu thực tế:\n"
            "Kích thước cửa sổ truyền thực tế của bên gửi luôn là giá trị nhỏ hơn giữa hai tham số: Effective Window = min(rwnd, cwnd)."
        ),
        "bullet_types": [
            "- Flow Control: Bảo vệ bộ đệm bên nhận; dùng tham số rwnd qua TCP header.",
            "- Congestion Control: Bảo vệ mạng trung gian; dùng tham số cwnd nội bộ bên gửi.",
            "- Cửa sổ truyền thực tế: Effective Window = min(rwnd, cwnd).",
        ],
        "example": "Nếu rwnd = 64KB nhưng mạng nghẽn khiến cwnd chỉ còn 4KB, bên gửi chỉ được phép truyền tối đa 4KB trước khi đợi ACK.",
    },
    {
        "id": "net_qa_05",
        "topic": "HTTP, HTTPS và Sự tiến hóa các phiên bản HTTP",
        "category": "application_protocols",
        "canonical_terms": [
            "http",
            "https",
            "tls/ssl",
            "http/1.1",
            "http/2",
            "http/3",
            "quic",
        ],
        "question": "Giao thức HTTP và HTTPS khác nhau như thế nào? Trình bày sự cải tiến từ HTTP/1.1 sang HTTP/2 và HTTP/3?",
        "intent": "web_protocol_evolution",
        "expected_action": "explain_http_https_and_evolution",
        "grounded_answer": (
            "HTTP và HTTPS là các giao thức truyền tải siêu văn bản ở tầng ứng dụng, là nền tảng của World Wide Web.\n\n"
            "1. HTTP vs HTTPS:\n"
            "- HTTP (cổng mặc định 80): Truyền văn bản thuần (plaintext), không mã hóa, dễ bị nghe lén (Sniffing), giả mạo (Tampering) và tấn công trung gian (Man-in-the-Middle).\n"
            "- HTTPS (cổng mặc định 443): Thực chất là HTTP chạy trên lớp bảo mật TLS/SSL (Transport Layer Security). Cung cấp 3 tính năng bảo mật: Mã hóa dữ liệu (Confidentiality), Toàn vẹn thông điệp (Integrity qua HMAC), và Xác thực máy chủ (Authentication qua Digital Certificate).\n\n"
            "2. Sự tiến hóa qua các phiên bản:\n"
            "- HTTP/1.1 (1997): Dùng kết nối bền vững (Persistent Connection: Connection: keep-alive) để tái sử dụng kết nối TCP. Tuy nhiên mắc lỗi tắc nghẽn đầu dòng (Head-of-Line Blocking - HoL) ở tầng ứng dụng: các yêu cầu phải xử lý tuần tự theo hàng đợi.\n"
            "- HTTP/2 (2015): Dựa trên SPDY, chuyển sang định dạng nhị phân (Binary Framing) thay vì text thuần. Đột phá với cơ chế Ghép kênh (Multiplexing): truyền đồng thời nhiều request/response trên cùng 1 kết nối TCP duy nhất qua các Streams độc lập; hỗ trợ nén Header (HPACK) và Server Push.\n"
            "- HTTP/3 (2022): Giải quyết triệt để vấn đề Head-of-Line Blocking ở tầng giao vận của TCP bằng cách chuyển hoàn toàn sang giao thức truyền tải QUIC chạy trên nền UDP. Hỗ trợ bắt tay 0-RTT/1-RTT nhanh chóng, tích hợp sẵn TLS 1.3 và chuyển vùng mạng mượt mà (Connection Migration)."
        ),
        "bullet_types": [
            "- HTTP (Port 80) Plaintext vs HTTPS (Port 443) Mã hóa TLS.",
            "- HTTP/1.1: Persistent TCP nhưng dính Head-of-Line blocking.",
            "- HTTP/2: Binary framing, Ghép kênh Multiplexing trên 1 TCP, HPACK.",
            "- HTTP/3: Chạy trên QUIC/UDP, giải quyết triệt để HoL blocking, TLS 1.3 mặc định.",
        ],
        "example": "Tải một trang web có 100 ảnh: HTTP/1.1 mở 6 kết nối TCP song song và chờ tuần tự; HTTP/2 truyền cả 100 ảnh cùng lúc trên 1 TCP; HTTP/3 truyền qua QUIC/UDP không bị trễ khi rớt mạng.",
    },
    {
        "id": "net_qa_06",
        "topic": "Hệ thống phân giải tên miền (DNS)",
        "category": "dns_architecture",
        "canonical_terms": [
            "dns",
            "domain name system",
            "phân giải tên miền",
            "cổng 53",
            "recursive query",
            "iterative query",
            "a record",
            "cname record",
            "mx record",
        ],
        "question": "Hệ thống phân giải tên miền (DNS) hoạt động như thế nào? Phân biệt truy vấn đệ quy và truy vấn lặp kèm các loại bản ghi DNS thông dụng?",
        "intent": "dns_mechanism_and_records",
        "expected_action": "explain_dns_architecture",
        "grounded_answer": (
            "DNS (Domain Name System - chạy cổng 53 UDP/TCP) là cơ sở dữ liệu phân tán có cấu trúc cây phân cấp, "
            "ánh xạ giữa tên miền dạng người đọc (như www.vnexpress.net) sang địa chỉ IP dạng máy tính xử lý (như 111.65.250.2).\n\n"
            "1. Phân biệt Truy vấn Đệ quy và Truy vấn Lặp:\n"
            "- Truy vấn Đệ quy (Recursive Query): Client yêu cầu Local DNS Resolver phải trả về câu trả lời cuối cùng (địa chỉ IP hoặc thông báo lỗi không tìm thấy). Resolver sẽ chịu trách nhiệm đi hỏi các máy chủ khác thay cho Client.\n"
            "- Truy vấn Lặp (Iterative Query): Local DNS Resolver lần lượt hỏi các tầng máy chủ DNS cấp trên; máy chủ cấp trên nếu không có câu trả lời trực tiếp sẽ trả về địa chỉ của máy chủ cấp thấp hơn kế tiếp để Resolver tự đi hỏi tiếp.\n\n"
            "2. Quá trình phân giải theo cấu trúc cây phân cấp:\n"
            "Bước 1: Client kiểm tra DNS Cache nội bộ -> gửi Recursive Query tới Local DNS Resolver (như 8.8.8.8).\n"
            "Bước 2: Resolver gửi Iterative Query tới Root Nameserver (.). Root trả về địa chỉ TLD Nameserver (.net).\n"
            "Bước 3: Resolver hỏi TLD Nameserver (.net). TLD trả về địa chỉ Authoritative Nameserver phụ trách miền vnexpress.net.\n"
            "Bước 4: Resolver hỏi Authoritative Nameserver. Máy chủ này trả về bản ghi địa chỉ IP chính xác.\n"
            "Bước 5: Resolver lưu kết quả vào bộ đệm (DNS Cache theo TTL) và trả địa chỉ IP cho Client.\n\n"
            "3. Các loại bản ghi DNS (Resource Records) thông dụng:\n"
            "- Bản ghi A: Ánh xạ tên miền sang địa chỉ IPv4 32-bit.\n"
            "- Bản ghi AAAA: Ánh xạ tên miền sang địa chỉ IPv6 128-bit.\n"
            "- Bản ghi CNAME (Canonical Name): Tạo bí danh (alias) trỏ tên miền này sang một tên miền khác.\n"
            "- Bản ghi MX (Mail Exchange): Chỉ định máy chủ tiếp nhận thư điện tử cho miền kèm độ ưu tiên.\n"
            "- Bản ghi NS (Name Server): Khai báo máy chủ DNS có thẩm quyền (Authoritative) quản lý vùng tên miền.\n"
            "- Bản ghi PTR (Pointer): Dùng cho phân giải ngược (Reverse DNS: từ địa chỉ IP suy ra tên miền).\n"
            "- Bản ghi TXT: Chứa thông tin văn bản tùy biến, thường dùng xác thực SPF, DKIM, DMARC chống giả mạo email."
        ),
        "bullet_types": [
            "- Cổng 53: Thường dùng UDP cho truy vấn nhẹ, TCP khi chuyển vùng (Zone Transfer) hoặc gói > 512B.",
            "- Truy vấn đệ quy: Client nhờ Resolver làm hết.",
            "- Truy vấn lặp: Resolver tự đi hỏi lần lượt Root -> TLD -> Authoritative.",
            "- Bản ghi chính: A (IPv4), AAAA (IPv6), CNAME (alias), MX (mail), NS (nameserver), PTR (ngược).",
        ],
        "example": "Truy vấn 'mail.google.com' trả về CNAME trỏ về 'googlemail.l.google.com', sau đó truy vấn A trả về IPv4 142.250.190.37.",
    },
    {
        "id": "net_qa_07",
        "topic": "Tiến trình DORA của giao thức DHCP",
        "category": "dhcp_protocol",
        "canonical_terms": [
            "dhcp",
            "dynamic host configuration protocol",
            "dora",
            "cấp phát ip động",
            "discover",
            "offer",
            "request",
            "ack",
        ],
        "question": "Giải thích quy trình cấp phát địa chỉ IP động của DHCP thông qua tiến trình DORA (Discover, Offer, Request, Acknowledge)?",
        "intent": "dhcp_workflow",
        "expected_action": "explain_dhcp_dora_process",
        "grounded_answer": (
            "DHCP (Dynamic Host Configuration Protocol) hoạt động ở tầng ứng dụng (dùng UDP cổng 67 cho Server và cổng 68 cho Client), "
            "tự động hóa quá trình cấu hình mạng IP, Subnet Mask, Default Gateway và DNS Server cho thiết bị mạng thông qua 4 bước DORA:\n\n"
            "1. D - DHCP Discover:\n"
            "- Thiết bị mới cắm vào mạng chưa có IP, gửi thông điệp Broadcast (IP nguồn: 0.0.0.0, IP đích: 255.255.255.255) tới toàn mạng LAN.\n"
            "- Chứa địa chỉ MAC vật lý của Client và một số định danh giao dịch (Transaction ID - XID).\n\n"
            "2. O - DHCP Offer:\n"
            "- Các DHCP Server trong mạng nhận được Discover sẽ chọn một địa chỉ IP chưa được cấp trong Pool, gửi lại bản tin Offer (Broadcast hoặc Unicast).\n"
            "- Thông điệp chứa: Địa chỉ IP đề xuất (Your IP), Subnet Mask, Thời gian thuê (Lease Time), IP của Server cấp phát, Default Gateway, DNS Servers.\n\n"
            "3. R - DHCP Request:\n"
            "- Client nhận được Offer (nếu có nhiều Server offer, Client thường chọn Offer đầu tiên đến), gửi bản tin Broadcast DHCP Request.\n"
            "- Client gửi broadcast để thông báo công khai cho Server được chọn biết mình chấp nhận lời mời, đồng thời ngầm báo cho các Server khác thu hồi lại các IP đã đề xuất tạm thời.\n\n"
            "4. A - DHCP Acknowledge (ACK):\n"
            "- DHCP Server được chọn ghi nhận thông tin thuê vào cơ sở dữ liệu và gửi gói tin DHCP ACK về cho Client.\n"
            "- Client nhận được ACK chính thức cấu hình địa chỉ IP lên card mạng và có thể giao tiếp trên mạng.\n\n"
            "Gia hạn thuê IP (Lease Renewal):\n"
            "Khi thời gian sử dụng đạt 50% thời hạn thuê (T1 = 0.5 * Lease Time), Client gửi trực tiếp Unicast Request lên Server để gia hạn."
        ),
        "bullet_types": [
            "- D (Discover): Client broadcast tìm DHCP Server (Src: 0.0.0.0, Dst: 255.255.255.255).",
            "- O (Offer): Server gửi đề xuất IP và thông số cấu hình mạng.",
            "- R (Request): Client broadcast xác nhận chấp thuận IP từ Server đã chọn.",
            "- A (Acknowledge): Server chính thức xác nhận hợp đồng thuê IP thành công.",
        ],
        "example": "Laptop bật Wi-Fi gửi Discover -> Router gửi Offer IP 192.168.1.50 -> Laptop gửi Request chấp nhận -> Router gửi ACK xác nhận thuê trong 24 giờ.",
    },
    {
        "id": "net_qa_08",
        "topic": "Giao thức định tuyến OSPF",
        "category": "routing_protocols",
        "canonical_terms": [
            "ospf",
            "open shortest path first",
            "định tuyến ospf",
            "link-state",
            "thuật toán dijkstra",
            "area 0",
            "lsa",
        ],
        "question": "Giao thức định tuyến OSPF hoạt động theo nguyên lý nào? Giải thích vai trò của thuật toán Dijkstra, khái niệm Link-State Advertisement (LSA) và phân chia vùng Area 0?",
        "intent": "interior_routing_mechanism",
        "expected_action": "explain_ospf_protocol",
        "grounded_answer": (
            "OSPF (Open Shortest Path First) là giao thức định tuyến nội miền (IGP - Interior Gateway Protocol) thuộc nhóm Trạng thái liên kết (Link-State), "
            "chạy trực tiếp trên nền IP với Protocol Number 89, được thiết kế cho các mạng doanh nghiệp quy mô vừa và lớn với tốc độ hội tụ cực nhanh.\n\n"
            "1. Nguyên lý hoạt động:\n"
            "- Mỗi Router OSPF gửi các gói tin Hello định kỳ qua địa chỉ Multicast (224.0.0.5 và 224.0.0.6) để khám phá và duy trì quan hệ láng giềng (Neighbor Adjacency).\n"
            "- Các Router trao đổi thông điệp trạng thái liên kết (Link-State Advertisements - LSA) chứa thông tin về các cổng, trạng thái hoạt động, chi phí (Cost) và mạng kết nối.\n"
            "- Toàn bộ các router trong cùng một vùng (Area) lưu trữ các LSA nhận được để xây dựng Cơ sở dữ liệu trạng thái liên kết (LSDB - Link-State Database) hoàn toàn giống hệt nhau, đại diện cho bản đồ toàn cảnh của mạng.\n\n"
            "2. Thuật toán Dijkstra (SPF - Shortest Path First):\n"
            "- Mỗi Router tự lấy chính mình làm gốc (Root) của cây đồ thị, chạy thuật toán Dijkstra tính toán đường đi ngắn nhất tới tất cả các mạng đích.\n"
            "- Metric của OSPF là Chi phí (Cost), tỷ lệ nghịch với băng thông giao diện:\n"
            "  Cost = Reference Bandwidth / Interface Bandwidth (mặc định Reference Bandwidth = 100 Mbps).\n"
            "- Tuyệt đối chống vòng lặp định tuyến (Loop-free) nhờ thuật toán đồ thị có cấu trúc.\n\n"
            "3. Phân chia cấu trúc phân cấp (Hierarchical Areas) và Vùng xương sống (Area 0):\n"
            "- Mạng OSPF lớn được chia thành nhiều Area để giới hạn phạm vi phát tán LSA, giảm kích thước LSDB và tiết kiệm CPU/RAM cho Router.\n"
            "- Area 0 (Backbone Area): Là vùng trung tâm bắt buộc. Mọi vùng chuẩn khác (Non-backbone Area) đều phải kết nối trực tiếp hoặc logic (qua Virtual Link) với Area 0.\n"
            "- Lưu lượng truyền giữa hai vùng bất kỳ bắt buộc phải đi qua Area 0 thông qua các Router biên vùng (ABR - Area Border Router)."
        ),
        "bullet_types": [
            "- Loại giao thức: Link-State IGP, chạy trên IP số hiệu 89, Multicast 224.0.0.5/6.",
            "- Thuật toán: Dijkstra SPF dựng sơ đồ cây không vòng lặp (Loop-free).",
            "- Metric: Cost = Reference Bandwidth / Interface Bandwidth.",
            "- Phân cấp: Bắt buộc có Area 0 (Backbone Area); trao đổi qua ABR Router.",
        ],
        "example": "Đường truyền Gigabit Ethernet (1000 Mbps) có Cost thấp hơn đường Fast Ethernet (100 Mbps), OSPF sẽ tự động chọn đường Gigabit làm Next-Hop tối ưu.",
    },
    {
        "id": "net_qa_09",
        "topic": "Giao thức định tuyến ngoài BGP",
        "category": "routing_protocols",
        "canonical_terms": [
            "bgp",
            "border gateway protocol",
            "định tuyến bgp",
            "path vector",
            "autonomous system",
            "as",
            "ebgp",
            "ibgp",
        ],
        "question": "BGP (Border Gateway Protocol) là gì, tại sao BGP được gọi là giao thức Path Vector và đóng vai trò sống còn trong định tuyến Internet toàn cầu?",
        "intent": "exterior_routing_mechanism",
        "expected_action": "explain_bgp_protocol",
        "grounded_answer": (
            "BGP (Border Gateway Protocol - phiên bản hiện hành BGP-4) là giao thức định tuyến ngoại miền (EGP) tiêu chuẩn thực tế duy nhất "
            "vận hành toàn bộ mạng Internet toàn cầu, kết nối hàng chục nghìn Hệ tự trị (Autonomous Systems - AS) độc lập.\n\n"
            "1. Đặc điểm cốt lõi & Nguyên lý Path Vector:\n"
            "- BGP chạy trên nền TCP cổng 179 để đảm bảo việc truyền tin định tuyến tin cậy.\n"
            "- Khác với IGP tối ưu tốc độ/băng thông đơn thuần, BGP là giao thức Vector đường dẫn (Path Vector Protocol). Khi quảng bá một dải mạng (Prefix), "
            "BGP đính kèm toàn bộ danh sách các số hiệu AS mà gói tin sẽ đi qua trong thuộc tính AS-Path (ví dụ: AS-Path: 11165 7473 13335).\n\n"
            "2. Chống lặp vòng định tuyến (Loop Prevention):\n"
            "- Khi một Router BGP nhận được bản tin cập nhật định tuyến, nó kiểm tra thuộc tính AS-Path. Nếu phát hiện chính ASN (Autonomous System Number) của mình "
            "đã có trong danh sách AS-Path, router sẽ lập tức loại bỏ (Drop) tuyến đường đó ngay, ngăn chặn triệt để hiện tượng Routing Loop xuyên biên giới.\n\n"
            "3. eBGP so với iBGP:\n"
            "- eBGP (External BGP): Chạy giữa các router biên thuộc hai AS khác nhau (quản trị các chính sách trao đổi lưu lượng giữa các ISP/Telecom).\n"
            "- iBGP (Internal BGP): Chạy giữa các router BGP nằm bên trong cùng một AS để truyền bá thông tin định tuyến học từ bên ngoài vào sâu trong mạng nội bộ.\n\n"
            "4. Định tuyến theo Chính sách (Policy-Based Routing):\n"
            "- BGP cho phép các tổ chức áp dụng các thuộc tính phong phú (Path Attributes) như AS-Path, Local Preference, MED (Multi-Exit Discriminator), Weight để quyết định "
            "điều hướng lưu lượng dựa trên lợi ích thương mại, kinh tế và bảo mật quốc gia thay vì chỉ đo khoảng cách vật lý."
        ),
        "bullet_types": [
            "- Vai trò: Giao thức duy nhất điều khiển định tuyến liên AS trên toàn mạng Internet.",
            "- Nền tảng: Chạy qua TCP cổng 179 tin cậy.",
            "- Path Vector: Theo dõi danh sách AS-Path để chọn đường và chống vòng lặp (Loop Prevention).",
            "- eBGP kết nối các ISP/AS khác nhau; iBGP phân phối tuyến nội bộ trong cùng 1 AS.",
        ],
        "example": "Khi ISP Viettel (AS 7552) muốn chuyển lưu lượng sang Google (AS 15169), BGP sẽ đánh giá tuyến đường có AS-Path ngắn nhất và Local Preference cao nhất.",
    },
    {
        "id": "net_qa_10",
        "topic": "Tường lửa (Firewall) và Phân vùng DMZ",
        "category": "security_and_firewalls",
        "canonical_terms": [
            "tường lửa",
            "firewall",
            "packet filtering",
            "stateful inspection",
            "ngfw",
            "dmz",
            "vùng phi quân sự",
            "acl",
        ],
        "question": "Trình bày kiến trúc và nguyên lý hoạt động của Tường lửa (Firewall): phân biệt Packet Filtering, Stateful Inspection, NGFW và vai trò của vùng DMZ?",
        "intent": "network_security_architecture",
        "expected_action": "explain_firewall_and_dmz",
        "grounded_answer": (
            "Tường lửa (Firewall) là hệ thống bảo mật mạng kiểm soát và lọc lưu lượng mạng vào/ra dựa trên các bộ quy tắc (Security Rules / ACLs) định sẵn.\n\n"
            "1. Các thế hệ kiến trúc tường lửa:\n"
            "- Lọc gói không trạng thái (Stateless Packet Filtering - L3/L4): Kiểm tra từng gói tin độc lập dựa trên thông tin IP nguồn, IP đích, Cổng nguồn, Cổng đích, và Giao thức. Không ghi nhớ ngữ cảnh phiên kết nối; dễ bị qua mặt bởi IP Spoofing.\n"
            "- Kiểm tra trạng thái (Stateful Inspection): Theo dõi toàn bộ vòng đời của kết nối (State Table). Nhận biết được một gói tin thuộc kết nối mới (NEW), kết nối đã thiết lập (ESTABLISHED), hay có liên quan (RELATED). Tự động mở cổng cho gói tin phản hồi hợp lệ và chặn gói tin bất thường.\n"
            "- Tường lửa ứng dụng / Proxy (Application Gateway - L7): Kiểm tra sâu nội dung dữ liệu tầng ứng dụng, hiểu rõ ngữ nghĩa HTTP, FTP, DNS để ngăn chặn mã độc nhúng trong payload.\n"
            "- Tường lửa thế hệ mới (Next-Generation Firewall - NGFW): Tích hợp Stateful Inspection, Deep Packet Inspection (DPI), nhận diện ứng dụng (App-ID), kiểm soát người dùng (User-ID), tích hợp hệ thống ngăn chặn xâm nhập (IPS/IDS), và rà quét chống mã độc theo thời gian thực.\n\n"
            "2. Vùng phi quân sự (DMZ - Demilitarized Zone):\n"
            "- DMZ là vùng mạng phụ được cô lập nằm giữa mạng nội bộ riêng tư (Internal LAN) và mạng công cộng không tin cậy (Internet).\n"
            "- Vai trò: Dùng để đặt các máy chủ cung cấp dịch vụ công khai ra ngoài Internet (như Web Server, Mail Server, DNS Server, FTP Server).\n"
            "- Chính sách an ninh DMZ:\n"
            "  * Internet ĐƯỢC PHÉP truy cập vào các dịch vụ chỉ định trong DMZ (qua cổng 80, 443, 25).\n"
            "  * DMZ BỊ CẤM khởi tạo kết nối vào mạng nội bộ LAN.\n"
            "  * Mạng nội bộ LAN có thể tự do khởi tạo kết nối quản trị vào DMZ.\n"
            "- Lợi ích: Nếu một máy chủ công khai trong DMZ bị tin tặc chiếm quyền kiểm soát, kẻ tấn công vẫn bị tường lửa chặn lại, không thể trực tiếp đột nhập vào cơ sở dữ liệu và máy trạm nội bộ của doanh nghiệp."
        ),
        "bullet_types": [
            "- Packet Filtering: Lọc đơn giản theo IP/Port/Protocol (L3/L4).",
            "- Stateful Inspection: Duy trì State Table theo dõi trạng thái phiên kết nối.",
            "- NGFW: Tích hợp DPI, App-ID, User-ID, IPS và Antivirus.",
            "- Vùng DMZ: Cách ly máy chủ công cộng; chặn chiều tấn công từ DMZ vào mạng nội bộ LAN.",
        ],
        "example": "Đặt Web Server bán hàng trong DMZ. Người dùng Internet chỉ truy cập được cổng 443 của Web Server; Web Server không có quyền tùy tiện truy cập vào vùng LAN kế toán.",
    },
    {
        "id": "net_qa_11",
        "topic": "Kỹ thuật NAT và PAT",
        "category": "nat_pat",
        "canonical_terms": [
            "nat",
            "pat",
            "network address translation",
            "port address translation",
            "nat overload",
            "rfc 1918",
            "ip private",
            "ip public",
        ],
        "question": "Kỹ thuật NAT (Network Address Translation) và PAT (Port Address Translation) là gì? Giải thích cơ chế ánh xạ giữa địa chỉ IP riêng và IP công cộng?",
        "intent": "address_translation_mechanism",
        "expected_action": "explain_nat_and_pat",
        "grounded_answer": (
            "NAT (Network Address Translation - RFC 1631) và PAT (Port Address Translation / NAT Overload) là kỹ thuật dịch đổi địa chỉ IP "
            "trong tiêu đề gói tin khi đi qua thiết bị định tuyến hoặc tường lửa, giải quyết bài toán cạn kiệt địa chỉ IPv4 và ẩn giấu cấu trúc mạng nội bộ.\n\n"
            "1. Dải địa chỉ IP riêng (Private IP Addresses theo RFC 1918):\n"
            "- Lớp A: 10.0.0.0/8 (10.0.0.0 - 10.255.255.255).\n"
            "- Lớp B: 172.16.0.0/12 (172.16.0.0 - 172.31.255.255).\n"
            "- Lớp C: 192.168.0.0/16 (192.168.0.0 - 192.168.255.255).\n"
            "Các dải IP này chỉ có ý nghĩa cục bộ và không được phép định tuyến trên mạng Internet công cộng.\n\n"
            "2. Phân loại NAT:\n"
            "- Static NAT (NAT tĩnh): Ánh xạ cố định 1-1 giữa 1 địa chỉ IP riêng và 1 địa chỉ IP công cộng (thường dùng đưa Web/Mail Server nội bộ ra ngoài).\n"
            "- Dynamic NAT (NAT động): Ánh xạ nhiều-nhiều (m-to-n) giữa một nhóm IP riêng với một bể (Pool) các địa chỉ IP công cộng có sẵn theo nguyên tắc ai đến trước phục vụ trước.\n"
            "- PAT (Port Address Translation / NAT Overload): Ánh xạ nhiều IP riêng (n-to-1) vào duy nhất MỘT địa chỉ IP công cộng thông qua việc hoán đổi số hiệu cổng nguồn (Source Port).\n\n"
            "3. Cơ chế bảng dịch PAT (NAT Translation Table):\n"
            "Khi máy trạm LAN (192.168.1.10:54321) gửi gói tin ra Internet, Router NAT thay thế IP nguồn thành IP Public của router (ví dụ: 203.0.113.5) và cấp một Port ngẫu nhiên (ví dụ: 40001).\n"
            "Router lưu vào bảng NAT: [192.168.1.10:54321 <-> 203.0.113.5:40001]. Khi máy chủ Internet gửi gói tin phản hồi về 203.0.113.5:40001, "
            "router tra bảng và dịch ngược lại thành 192.168.1.10:54321 để chuyển về cho đúng máy trạm trong mạng nội bộ."
        ),
        "bullet_types": [
            "- RFC 1918 Private IP: 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16.",
            "- Static NAT: Ánh xạ 1-1 cố định IP riêng sang IP công cộng.",
            "- Dynamic NAT: Ánh xạ từ pool địa chỉ công cộng khả dụng.",
            "- PAT (NAT Overload): Hàng nghìn máy dùng chung 1 IP công cộng nhờ phân biệt qua số hiệu cổng (Port).",
        ],
        "example": "Trong gia đình, điện thoại, laptop, TV đều có IP 192.168.1.x nhưng khi truy cập Google đều hiển thị chung 1 IP Public của modem nhà mạng.",
    },
    {
        "id": "net_qa_12",
        "topic": "Chia mạng con (Subnetting) và Ký hiệu CIDR",
        "category": "subnetting_and_addressing",
        "canonical_terms": [
            "chia mạng con",
            "subnetting",
            "cidr",
            "subnet mask",
            "network address",
            "broadcast address",
            "mặt nạ mạng con",
        ],
        "question": "Trình bày phương pháp chia mạng con (Subnetting) và ký hiệu CIDR: công thức tính số địa chỉ IP, số host sử dụng được, địa chỉ mạng và địa chỉ broadcast?",
        "intent": "ip_addressing_and_subnetting",
        "expected_action": "explain_subnetting_and_cidr",
        "grounded_answer": (
            "Chia mạng con (Subnetting) là quá trình mượn các bit của phần Host ID để mở rộng phần Network ID, "
            "chia nhỏ một mạng lớn thành nhiều mạng con độc lập nhằm giảm xung đột mạng, kiểm soát lưu lượng broadcast và tiết kiệm không gian địa chỉ.\n\n"
            "1. Ký hiệu CIDR (Classless Inter-Domain Routing):\n"
            "- Ký hiệu /n biểu thị có n bit 1 liên tiếp trong mặt nạ mạng con (Subnet Mask).\n"
            "- Ví dụ: /24 tương ứng mặt nạ 255.255.255.0 (24 bit mạng, 8 bit host).\n\n"
            "2. Các công thức toán học cốt lõi:\n"
            "Gọi h là số bit dành cho phần Host (h = 32 - n với IPv4):\n"
            "- Tổng số địa chỉ IP trong mạng con = 2^h.\n"
            "- Số địa chỉ host sử dụng được gán cho thiết bị = 2^h - 2 (trừ 2 địa chỉ đặc biệt không được gán cho host):\n"
            "  * Địa chỉ mạng (Network Address): Phần host chứa toàn bit 0 (đại diện cho cả mạng con).\n"
            "  * Địa chỉ quảng bá (Broadcast Address): Phần host chứa toàn bit 1 (dùng gửi tin nhắn tới mọi máy trong mạng con).\n"
            "- Bước nhảy mạng (Block Size / Magic Number) = 256 - giá trị octet tương ứng của Subnet Mask.\n\n"
            "3. Ví dụ chia mạng con cụ thể:\n"
            "Cho dải mạng 192.168.1.0/26:\n"
            "- Số bit host: h = 32 - 26 = 6 bits.\n"
            "- Subnet Mask: 255.255.255.192 (11111111.11111111.11111111.11000000).\n"
            "- Tổng số IP mỗi mạng con = 2^6 = 64.\n"
            "- Số host dùng được = 64 - 2 = 62 hosts.\n"
            "- Các mạng con:\n"
            "  * Subnet 1: Mạng: 192.168.1.0; Host dùng được: 192.168.1.1 -> 192.168.1.62; Broadcast: 192.168.1.63.\n"
            "  * Subnet 2: Mạng: 192.168.1.64; Host dùng được: 192.168.1.65 -> 192.168.1.126; Broadcast: 192.168.1.127.\n"
            "  * Subnet 3: Mạng: 192.168.1.128; Host dùng được: 192.168.1.129 -> 192.168.1.190; Broadcast: 192.168.1.191.\n"
            "  * Subnet 4: Mạng: 192.168.1.192; Host dùng được: 192.168.1.193 -> 192.168.1.254; Broadcast: 192.168.1.255."
        ),
        "bullet_types": [
            "- Ký hiệu /n: n bit mạng (Network bits), h = 32 - n bit máy (Host bits).",
            "- Tổng số IP = 2^h; Số host sử dụng được = 2^h - 2.",
            "- Trừ 2: Địa chỉ mạng (Host toàn 0) và Địa chỉ Broadcast (Host toàn 1).",
            "- Bước nhảy Magic Number = 256 - Octet Subnet Mask.",
        ],
        "example": "Mạng 10.0.0.0/30 (h=2) có 4 IP: 10.0.0.0 (Network), 10.0.0.1 và 10.0.0.2 (gán cho 2 router nối cổng point-to-point), 10.0.0.3 (Broadcast).",
    },
]

# Alias for backwards and cross-module compatibility
NETWORKING_FEW_SHOT_QA_EXAMPLES = NETWORKING_FEW_SHOT_EXAMPLES


def get_few_shot_examples_by_topic(topic_or_cluster: str) -> List[Dict[str, Any]]:
    """Retrieves all few-shot QA examples matching a given topic or cluster name."""
    norm_target = normalize_term(topic_or_cluster)
    stripped_target = strip_accents(norm_target)

    results = []
    for ex in NETWORKING_FEW_SHOT_EXAMPLES:
        t_norm = normalize_term(ex.get("topic", ""))
        c_norm = normalize_term(ex.get("category", ""))
        if (
            norm_target in t_norm
            or stripped_target in strip_accents(t_norm)
            or norm_target in c_norm
        ):
            results.append(ex)
    return results


def search_few_shot_examples(keyword: str) -> List[Dict[str, Any]]:
    """Searches few-shot examples across questions, answers, and canonical terms."""
    if not keyword or not keyword.strip():
        return []

    norm_kw = normalize_term(keyword)
    stripped_kw = strip_accents(norm_kw)

    matched = []
    for ex in NETWORKING_FEW_SHOT_EXAMPLES:
        combined_text = " ".join(
            [
                ex.get("topic", ""),
                ex.get("question", ""),
                ex.get("grounded_answer", ""),
                " ".join(ex.get("canonical_terms", [])),
            ]
        )
        norm_comb = normalize_term(combined_text)
        stripped_comb = strip_accents(norm_comb)

        if norm_kw in norm_comb or stripped_kw in stripped_comb:
            matched.append(ex)
    return matched
