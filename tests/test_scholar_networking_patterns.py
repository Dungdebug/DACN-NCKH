"""
Unit Tests for Computer Networking & Protocols Semantic Patterns.
Tests bidirectional synonym mapping, domain query expansion, and few-shot QA datasets
for OSI 7 Layers, TCP/IP, UDP, HTTP/HTTPS, DNS, DHCP, OSPF/BGP Routing, and Firewalls/NAT.
100% Offline, Pure Python stdlib, Zero External Dependencies.
"""

import pytest
from src.self_study.datasets.networking_and_protocols_patterns import (
    strip_accents,
    normalize_term,
    get_networking_synonyms,
    expand_networking_query_terms,
    NetworkingThesaurusRegistry,
    NETWORKING_SYNONYM_CLUSTERS,
    NETWORKING_FEW_SHOT_EXAMPLES,
    NETWORKING_FEW_SHOT_QA_EXAMPLES,
    get_few_shot_examples_by_topic,
    search_few_shot_examples,
)


class TestNetworkingTextNormalization:
    """Tests for Vietnamese text normalization and tone mark removal."""

    def test_strip_accents_vietnamese(self):
        assert strip_accents("Mô hình OSI 7 tầng") == "Mo hinh OSI 7 tang"
        assert strip_accents("định tuyến OSPF và BGP") == "dinh tuyen OSPF va BGP"
        assert strip_accents("ĐỊNH TUYẾN NGOÀI") == "DINH TUYEN NGOAI"
        assert strip_accents("tường lửa và kiểm soát luồng") == "tuong lua va kiem soat luong"

    def test_strip_accents_empty_and_ascii(self):
        assert strip_accents("") == ""
        assert strip_accents("three-way handshake") == "three-way handshake"
        assert strip_accents("192.168.1.1/24!@#") == "192.168.1.1/24!@#"

    def test_normalize_term(self):
        assert normalize_term("   Mô   Hình   OSI   7 TẦNG   ") == "mô hình osi 7 tầng"
        assert normalize_term("\n\tTCP   /   IP\tModel\n") == "tcp / ip model"


class TestBidirectionalNetworkingSynonyms:
    """Verifies bidirectional synonym mapping across computer networking concepts."""

    def test_osi_model_cluster(self):
        terms = ["mô hình osi", "osi 7 tầng", "open systems interconnection", "mô hình tham chiếu osi"]
        for term in terms:
            synonyms = get_networking_synonyms(term)
            assert len(synonyms) > 0, f"Expected synonyms for '{term}'"
            for other in terms:
                assert other in synonyms, f"'{other}' not found in synonyms of '{term}'"

    def test_osi_layers_mapping(self):
        # Physical
        phys_syns = get_networking_synonyms("tầng vật lý")
        assert "physical layer" in phys_syns
        assert "bit stream" in phys_syns

        # Data Link
        dl_syns = get_networking_synonyms("tầng liên kết dữ liệu")
        assert "data link layer" in dl_syns
        assert "frame" in dl_syns
        assert "địa chỉ mac" in dl_syns

        # Network
        net_syns = get_networking_synonyms("tầng mạng")
        assert "network layer" in net_syns
        assert "packet" in net_syns
        assert "router" in net_syns

        # Transport
        trans_syns = get_networking_synonyms("tầng giao vận")
        assert "transport layer" in trans_syns
        assert "segment" in trans_syns
        assert "tcp" in trans_syns

        # Application
        app_syns = get_networking_synonyms("tầng ứng dụng")
        assert "application layer" in app_syns
        assert "http" in app_syns

    def test_tcp_protocol_mapping(self):
        tcp_syns = get_networking_synonyms("giao thức tcp")
        assert "transmission control protocol" in tcp_syns
        assert "hướng kết nối" in tcp_syns
        assert "vận chuyển tin cậy" in tcp_syns

    def test_three_way_handshake_mapping(self):
        hs_syns = get_networking_synonyms("bắt tay 3 bước")
        assert "three-way handshake" in hs_syns
        assert "syn" in hs_syns
        assert "ack" in hs_syns
        assert "syn-ack" in hs_syns

        # Reverse lookup
        syn_ack_syns = get_networking_synonyms("syn-ack")
        assert "bắt tay 3 bước" in syn_ack_syns

    def test_udp_protocol_mapping(self):
        udp_syns = get_networking_synonyms("giao thức udp")
        assert "user datagram protocol" in udp_syns
        assert "phi kết nối" in udp_syns
        assert "datagram" in udp_syns
        assert "best effort" in udp_syns

    def test_http_https_mapping(self):
        http_syns = get_networking_synonyms("giao thức http")
        assert "hypertext transfer protocol" in http_syns
        assert "cổng 80" in http_syns

        https_syns = get_networking_synonyms("https")
        assert "http an toàn" in https_syns
        assert "ssl/tls" in https_syns
        assert "cổng 443" in https_syns

    def test_dns_mapping(self):
        dns_syns = get_networking_synonyms("dns")
        assert "hệ thống phân giải tên miền" in dns_syns
        assert "domain name system" in dns_syns
        assert "cổng 53" in dns_syns

        rec_syns = get_networking_synonyms("bản ghi a")
        assert "dns records" in rec_syns
        assert "a record" in rec_syns
        assert "bản ghi cname" in rec_syns

    def test_dhcp_mapping(self):
        dhcp_syns = get_networking_synonyms("dhcp")
        assert "cấp phát ip động" in dhcp_syns
        assert "tiến trình dora" in dhcp_syns
        assert "dora" in dhcp_syns
        assert "cổng 67" in dhcp_syns

    def test_routing_ospf_and_bgp_mapping(self):
        # OSPF
        ospf_syns = get_networking_synonyms("ospf")
        assert "open shortest path first" in ospf_syns
        assert "link-state" in ospf_syns
        assert "thuật toán dijkstra" in ospf_syns
        assert "area 0" in ospf_syns

        # BGP
        bgp_syns = get_networking_synonyms("bgp")
        assert "border gateway protocol" in bgp_syns
        assert "path vector" in bgp_syns
        assert "autonomous system" in bgp_syns
        assert "cổng 179" in bgp_syns

    def test_firewall_and_nat_mapping(self):
        # Firewall
        fw_syns = get_networking_synonyms("tường lửa")
        assert "firewall" in fw_syns
        assert "stateful inspection" in fw_syns
        assert "ngfw" in fw_syns
        assert "dmz" in fw_syns

        # NAT/PAT
        nat_syns = get_networking_synonyms("nat")
        assert "network address translation" in nat_syns
        assert "pat" in nat_syns
        assert "rfc 1918" in nat_syns

    def test_unaccented_lookup(self):
        syns_osi = get_networking_synonyms("mo hinh osi")
        assert "osi 7 tầng" in syns_osi

        syns_ospf = get_networking_synonyms("dinh tuyen ospf")
        assert "open shortest path first" in syns_ospf

        syns_fw = get_networking_synonyms("tuong lua")
        assert "firewall" in syns_fw

    def test_unknown_term_returns_empty(self):
        assert get_networking_synonyms("quantum teleportation protocol xyz") == []
        assert get_networking_synonyms("") == []


class TestExpandNetworkingQueryTerms:
    """Verifies greedy phrase matching and multi-intent domain query expansion."""

    def test_single_term_expansion(self):
        expanded = expand_networking_query_terms("mô hình osi")
        assert len(expanded) >= 4
        assert "osi 7 tầng" in expanded
        assert "open systems interconnection" in expanded

    def test_multi_topic_joint_query(self):
        query = "So sánh giao thức TCP và UDP trong mô hình OSI"
        expanded = expand_networking_query_terms(query)
        # Should detect tcp, udp, and osi clusters
        assert any("tcp" in t for t in expanded)
        assert any("udp" in t for t in expanded)
        assert any("osi" in t for t in expanded)

    def test_ospf_and_bgp_query(self):
        query = "Phân biệt định tuyến OSPF và BGP trên Internet"
        expanded = expand_networking_query_terms(query)
        assert "open shortest path first" in expanded
        assert "border gateway protocol" in expanded
        assert "path vector" in expanded

    def test_dhcp_dora_query(self):
        query = "Giải thích tiến trình DORA khi cấp phát DHCP"
        expanded = expand_networking_query_terms(query)
        assert "dynamic host configuration protocol" in expanded
        assert "cấp phát ip động" in expanded

    def test_firewall_dmz_unaccented_query(self):
        query = "cau hinh tuong lua va phan vung dmz"
        expanded = expand_networking_query_terms(query)
        assert any("tường lửa" in t or "firewall" in t for t in expanded)
        assert any("dmz" in t for t in expanded)

    def test_no_duplicate_terms_in_expansion(self):
        query = "tcp và giao thức tcp hướng kết nối"
        expanded = expand_networking_query_terms(query)
        normalized_terms = [normalize_term(t) for t in expanded]
        assert len(normalized_terms) == len(set(normalized_terms))

    def test_empty_and_irrelevant_queries(self):
        assert expand_networking_query_terms("") == []
        assert expand_networking_query_terms("nhiệt độ hôm nay tại Hà Nội") == []


class TestNetworkingFewShotQaDataset:
    """Verifies schema validity, depth, and domain coverage of few-shot QA blueprints."""

    def test_dataset_size_and_diversity(self):
        assert len(NETWORKING_FEW_SHOT_EXAMPLES) >= 12
        assert NETWORKING_FEW_SHOT_QA_EXAMPLES is NETWORKING_FEW_SHOT_EXAMPLES

    def test_required_fields_presence(self):
        required_fields = {
            "id",
            "topic",
            "category",
            "canonical_terms",
            "question",
            "intent",
            "expected_action",
            "grounded_answer",
            "bullet_types",
        }
        for item in NETWORKING_FEW_SHOT_EXAMPLES:
            missing = required_fields - set(item.keys())
            assert not missing, f"Item {item.get('id')} missing keys: {missing}"
            assert len(item["question"].strip()) > 15
            assert len(item["grounded_answer"].strip()) > 80
            assert isinstance(item["canonical_terms"], list)
            assert len(item["canonical_terms"]) >= 2
            assert isinstance(item["bullet_types"], list)
            assert len(item["bullet_types"]) >= 2

    def test_coverage_of_slide_key_topics(self):
        topics = " ".join([ex["topic"] for ex in NETWORKING_FEW_SHOT_EXAMPLES])
        answers = " ".join([ex["grounded_answer"] for ex in NETWORKING_FEW_SHOT_EXAMPLES])
        combined = (topics + " " + answers).lower()

        # Check required domain topics
        assert "osi" in combined
        assert "tcp" in combined
        assert "udp" in combined
        assert "three-way handshake" in combined or "bắt tay 3 bước" in combined
        assert "kiểm soát luồng" in combined or "flow control" in combined
        assert "tắc nghẽn" in combined or "congestion control" in combined
        assert "http" in combined and "https" in combined
        assert "dns" in combined
        assert "dhcp" in combined and "dora" in combined
        assert "ospf" in combined and "dijkstra" in combined
        assert "bgp" in combined and "path vector" in combined
        assert "tường lửa" in combined or "firewall" in combined
        assert "dmz" in combined
        assert "nat" in combined and "pat" in combined
        assert "subnetting" in combined or "chia mạng con" in combined

    def test_get_few_shot_examples_by_topic(self):
        osi_examples = get_few_shot_examples_by_topic("OSI")
        assert len(osi_examples) >= 1
        assert "osi" in osi_examples[0]["topic"].lower()

        tcp_examples = get_few_shot_examples_by_topic("TCP")
        assert len(tcp_examples) >= 2

        bgp_examples = get_few_shot_examples_by_topic("BGP")
        assert len(bgp_examples) >= 1

    def test_search_few_shot_examples(self):
        results_dora = search_few_shot_examples("DORA")
        assert len(results_dora) >= 1
        assert "dhcp" in results_dora[0]["topic"].lower()

        results_dmz = search_few_shot_examples("DMZ")
        assert len(results_dmz) >= 1

        results_empty = search_few_shot_examples("không_tồn_tại_trong_dataset_12345")
        assert len(results_empty) == 0


class TestCustomRegistryExtensibility:
    """Verifies that NetworkingThesaurusRegistry can be instantiated and customized."""

    def test_custom_registry_override(self):
        custom_clusters = {
            "sdn_networking": ["sdn", "software-defined networking", "openflow", "mạng định nghĩa bằng phần mềm"],
        }
        reg = NetworkingThesaurusRegistry(clusters=custom_clusters)
        assert reg.contains_term("openflow") is True
        assert reg.contains_term("tcp") is False
        syns = reg.get_synonyms("sdn")
        assert "openflow" in syns
        assert "mạng định nghĩa bằng phần mềm" in syns

    def test_all_clusters_method(self):
        reg = NetworkingThesaurusRegistry()
        clusters = reg.all_clusters()
        assert "osi_model" in clusters
        assert "tcp_protocol" in clusters
        assert "routing_ospf" in clusters
