"""
Domain-Specific Cryptography & Information Security Semantic Patterns.
Provides bidirectional synonym mapping, domain query expansion, and few-shot QA blueprints
for authoritative cryptographic educational and technical documents.
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


# Core cryptographic domain synonym clusters
CRYPTO_SYNONYM_CLUSTERS: Dict[str, List[str]] = {
    "symmetric_encryption": [
        "mã hóa đối xứng",
        "hệ mã khóa đối xứng",
        "hệ mã đối xứng",
        "mật mã đối xứng",
        "mã hóa khóa bí mật",
        "hệ mã khóa bí mật",
        "khóa bí mật",
        "mã cổ điển",
        "hệ mã cổ điển",
        "mã hóa cổ điển",
        "symmetric cipher",
        "symmetric encryption",
        "secret key encryption",
        "shared key cipher",
    ],
    "cryptanalysis": [
        "phá mã",
        "thám mã",
        "tấn công mã",
        "tấn công mật mã",
        "phân tích mật mã",
        "phương pháp phá mã",
        "giải mã trộm",
        "cryptanalysis",
        "crypto attack",
        "cipher attack",
    ],
    "stream_cipher": [
        "mã hóa dòng",
        "hệ mã hóa dòng",
        "hệ mã dòng",
        "mã dòng",
        "stream cipher",
        "stream ciphers",
        "mã dòng đối xứng",
    ],
    "one_time_pad": [
        "độn một lần",
        "hệ mã hóa độn một lần",
        "hệ mã độn một lần",
        "mã độn một lần",
        "độn 1 lần",
        "one-time pad",
        "one time pad",
        "otp",
        "vernam",
        "mã hóa vernam",
        "hệ mã vernam",
    ],
    "substitution": [
        "thay thế",
        "mã hóa thay thế",
        "hệ mã thay thế",
        "phương pháp thay thế",
        "mã thay thế",
        "substitution",
        "substitution cipher",
        "phép thay thế",
    ],
    "transposition": [
        "hoán vị",
        "mã hóa hoán vị",
        "hệ mã hoán vị",
        "phương pháp hoán vị",
        "mã hoán vị",
        "transposition",
        "transposition cipher",
        "permutation",
        "permutation cipher",
        "phép hoán vị",
    ],
    "caesar_shift": [
        "dịch vòng",
        "hệ mã dịch vòng",
        "mã dịch vòng",
        "mã caesar",
        "hệ mã caesar",
        "caesar",
        "caesar cipher",
        "shift cipher",
        "mã dịch chuyển",
    ],
    "monoalphabetic": [
        "đơn bảng",
        "hệ mã đơn bảng",
        "mã đơn bảng",
        "monoalphabetic",
        "monoalphabetic cipher",
        "mã hóa đơn bảng",
    ],
    "affine": [
        "affine",
        "hệ mã affine",
        "mã affine",
        "mã hóa affine",
        "affine cipher",
    ],
    "vigenere": [
        "vigenere",
        "hệ mã vigenere",
        "mã vigenere",
        "mã hóa vigenere",
        "vigenere cipher",
        "đa bảng",
        "hệ mã đa bảng",
        "mã đa bảng",
        "polyalphabetic cipher",
    ],
    "hill": [
        "hill",
        "hệ mã hill",
        "mã hill",
        "mã hóa hill",
        "hill cipher",
        "mã hóa ma trận",
        "mã ma trận",
    ],
    "autokey": [
        "autokey",
        "hệ mã autokey",
        "hệ mã hóa autokey",
        "mã autokey",
        "khóa tự sinh",
        "hệ mã khóa tự sinh",
        "autokey cipher",
    ],
    "playfair": [
        "playfair",
        "hệ mã playfair",
        "hệ mã hóa playfair",
        "mã playfair",
        "ma trận 5x5",
        "hệ mã ma trận 5x5",
        "playfair cipher",
    ],
    "block_cipher": [
        "mã hóa khối",
        "hệ mã khối",
        "mã khối",
        "block cipher",
        "block ciphers",
    ],
    "asymmetric_encryption": [
        "mã hóa bất đối xứng",
        "hệ mã bất đối xứng",
        "mã hóa khóa công khai",
        "hệ mã khóa công khai",
        "khóa công khai",
        "public key",
        "public key encryption",
        "asymmetric cipher",
        "asymmetric encryption",
    ],
    "plaintext_ciphertext": [
        "bản rõ",
        "bản mã",
        "thông điệp gốc",
        "thông điệp mã hóa",
        "plaintext",
        "ciphertext",
    ],
    "security_condition": [
        "an ninh vô điều kiện",
        "an toàn tuyệt đối",
        "an ninh hoàn hảo",
        "an ninh độ phức tạp tính toán",
        "unconditional security",
        "perfect secrecy",
        "computational security",
    ],
}


class CryptoThesaurusRegistry:
    """
    Registry for bidirectional cryptographic terminology mappings.
    Supports exact matching, tone-insensitive matching, and n-gram term discovery.
    """

    def __init__(self, clusters: Optional[Dict[str, List[str]]] = None):
        self.clusters = clusters or CRYPTO_SYNONYM_CLUSTERS
        self._norm_to_cluster: Dict[str, str] = {}
        self._stripped_to_cluster: Dict[str, str] = {}
        self._bidirectional_map: Dict[str, Set[str]] = {}
        self._compiled_keywords: List[Tuple[str, str]] = []  # (raw_term, cluster_name)
        self._build_indexes()

    def _build_indexes(self) -> None:
        """Constructs forward, reverse, and accent-insensitive indexes."""
        all_terms_with_len: List[Tuple[str, str, int]] = []

        for cluster_name, terms in self.clusters.items():
            canonical_set = set(terms)
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
        Extracts cryptographic domain terms from the query and expands them into synonyms.
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

            # Check if term appears as full word or phrase boundary
            pattern_raw = r"(?:\b|_|^)" + re.escape(n_t) + r"(?:\b|_|$)"
            pattern_stripped = r"(?:\b|_|^)" + re.escape(s_t) + r"(?:\b|_|$)"

            if re.search(pattern_raw, norm_q, flags=re.IGNORECASE) or re.search(
                pattern_stripped, stripped_q, flags=re.IGNORECASE
            ):
                if cluster_name not in matched_clusters:
                    matched_clusters.add(cluster_name)
                    matched_original_terms.append(term)

        # Assemble result list: primary matched terms first, then all cluster synonyms
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
_DEFAULT_REGISTRY = CryptoThesaurusRegistry()


def get_crypto_synonyms(term: str) -> List[str]:
    """Public helper to get bidirectional cryptographic synonyms."""
    return _DEFAULT_REGISTRY.get_synonyms(term)


def expand_domain_query_terms(query: str) -> List[str]:
    """
    Expands conversational user queries into authoritative technical cryptographic keywords.
    E.g.: 'mã hóa đối xứng' -> ['mã hóa đối xứng', 'hệ mã khóa đối xứng', 'khóa bí mật', 'mã cổ điển', ...]
    """
    return _DEFAULT_REGISTRY.expand_query(query)


# Few-shot QA examples for Cryptography and Information Security documents
CRYPTO_FEW_SHOT_QA_EXAMPLES: List[Dict[str, Any]] = [
    {
        "id": "crypto_qa_01",
        "topic": "Hệ mã khóa đối xứng",
        "category": "definition_and_taxonomy",
        "canonical_terms": [
            "mã hóa đối xứng",
            "hệ mã khóa đối xứng",
            "khóa bí mật",
            "mã cổ điển",
            "mã hóa thay thế",
            "mã hóa hoán vị",
        ],
        "question": "Mã hóa đối xứng là gì, có bao nhiêu loại phương pháp mã hóa và cho ví dụ?",
        "intent": "concept_and_classification",
        "expected_action": "explain_concept_and_list_all_branches",
        "grounded_answer": (
            "Hệ mã khóa đối xứng (hay mã cổ điển, mã khóa bí mật) là hệ mã hóa mà bên gửi và bên nhận "
            "sử dụng chung một khóa bí mật (hoặc khóa giải mã có thể dễ dàng suy ra từ khóa mã hóa).\n\n"
            "Các phương pháp mã hóa đối xứng kinh điển bao gồm 2 phương pháp chính:\n"
            "* Mã hóa thay thế (Substitution): Mỗi ký tự hoặc nhóm ký tự trong bản rõ được thay thế bằng một ký tự hoặc nhóm ký tự khác.\n"
            "* Mã hóa hoán vị (Transposition): Vị trí các ký tự trong bản rõ bị đảo lộn theo một quy tắc nhất định mà không thay đổi bản thân các ký tự.\n\n"
            "Các hệ mã hóa cụ thể trong tài liệu:\n"
            "- Hệ mã dịch vòng (Caesar): dịch chuyển các ký tự theo khóa k (ví dụ: k = 3, 'A' -> 'D').\n"
            "- Hệ mã đơn bảng (Monoalphabetic): ánh xạ 1-1 giữa bảng chữ cái bản rõ và bản mã.\n"
            "- Hệ mã Affine: sử dụng hàm số học e(x) = (ax + b) mod 26 với gcd(a, 26) = 1.\n"
            "- Hệ mã Vigenère: mã thay thế đa bảng dùng từ khóa lặp lại.\n"
            "- Hệ mã Hill: mã hóa thay thế khối nhiều chữ cái bằng phép nhân ma trận trên Z26.\n"
            "- Hệ mã hóa độn một lần (One-Time Pad): dùng khóa ngẫu nhiên dài bằng bản rõ, chỉ dùng một lần.\n"
            "- Hệ mã hóa dòng (Stream cipher): mã hóa từng bit hoặc byte dữ liệu liên tục.\n"
            "- Hệ mã hóa Autokey: sử dụng bản rõ hoặc bản mã để tự sinh dòng khóa.\n"
            "- Hệ mã hóa Playfair: mã hóa từng cặp ký tự (digraph) bằng ma trận 5x5.\n\n"
            "Ví dụ minh họa:\n"
            "Cho không gian P = C = K = Z26. Trong hệ mã Caesar với khóa k = 3, từ 'HELLO' được mã hóa thành:\n"
            "H(7) -> 10(K), E(4) -> 7(H), L(11) -> 14(O), L(11) -> 14(O), O(14) -> 17(R) => Bản mã là 'KHOOR'."
        ),
        "bullet_types": [
            "* Mã hóa thay thế.",
            "* Mã hóa hoán vị.",
        ],
        "specific_systems": [
            "Hệ mã dịch vòng (Caesar)",
            "Hệ mã đơn bảng (Monoalphabetic)",
            "Hệ mã Affine",
            "Hệ mã Vigenère",
            "Hệ mã Hill",
            "Hệ mã hóa độn một lần (One-Time Pad)",
            "Hệ mã hóa dòng (Stream cipher)",
            "Hệ mã hóa Autokey",
            "Hệ mã hóa Playfair",
            "Hệ mã hóa hoán vị",
        ],
        "example": "Cho P = C = K = Z26, k = 3. Bản rõ 'HELLO' được mã hóa thành 'KHOOR'.",
    },
    {
        "id": "crypto_qa_02",
        "topic": "Phá mã",
        "category": "attack_and_cryptanalysis",
        "canonical_terms": ["phá mã", "thám mã", "cryptanalysis", "tấn công mật mã"],
        "question": "Trình bày mục phá mã (thám mã) và các hình thức tấn công mật mã?",
        "intent": "section_detail_and_mechanism",
        "expected_action": "detail_cryptanalysis_models_and_objectives",
        "grounded_answer": (
            "Phá mã (thám mã hay Cryptanalysis) là quá trình nghiên cứu, phân tích nhằm khôi phục bản rõ "
            "hoặc tìm ra khóa bí mật mà không cần biết trước khóa.\n\n"
            "Các mô hình tấn công phá mã kinh điển bao gồm:\n"
            "1. Tấn công chỉ biết bản mã (Ciphertext-only attack): Người thám mã chỉ có trong tay một tập các bản mã.\n"
            "2. Tấn công biết bản rõ (Known-plaintext attack): Người thám mã sở hữu các cặp (bản rõ, bản mã) tương ứng.\n"
            "3. Tấn công chọn bản rõ (Chosen-plaintext attack): Người thám mã có thể chọn các bản rõ tùy ý và thu được các bản mã tương ứng.\n"
            "4. Tấn công chọn bản mã (Chosen-ciphertext attack): Người thám mã có thể chọn các bản mã và thu được bản rõ giải mã tương ứng.\n\n"
            "Mục tiêu cao nhất của phá mã là tìm ra khóa K; nếu không tìm được khóa thì mục tiêu tiếp theo là giải mã bản rõ P."
        ),
        "bullet_types": [
            "- Tấn công chỉ biết bản mã (Ciphertext-only attack)",
            "- Tấn công biết bản rõ (Known-plaintext attack)",
            "- Tấn công chọn bản rõ (Chosen-plaintext attack)",
            "- Tấn công chọn bản mã (Chosen-ciphertext attack)",
        ],
        "example": "Phân tích tần suất xuất hiện của các chữ cái trong bản mã đối với hệ mã đơn bảng.",
    },
    {
        "id": "crypto_qa_03",
        "topic": "Hệ mã hóa độn một lần",
        "category": "cipher_system",
        "canonical_terms": ["độn một lần", "one-time pad", "otp", "vernam", "an ninh vô điều kiện"],
        "question": "Hệ mã hóa độn một lần (One-Time Pad) là gì và tại sao lại an toàn tuyệt đối?",
        "intent": "system_concept_and_proof",
        "expected_action": "explain_one_time_pad_properties",
        "grounded_answer": (
            "Hệ mã hóa độn một lần (One-Time Pad - OTP do Gilbert Vernam đề xuất năm 1917) là hệ mã hóa "
            "trong đó chuỗi khóa bí mật là một chuỗi ngẫu nhiên hoàn toàn, có độ dài bằng chính độ dài của thông điệp bản rõ "
            "và chỉ được sử dụng duy nhất một lần.\n\n"
            "Nguyên lý vận hành:\n"
            "- Mã hóa: ci = (pi ⊕ ki) (hoặc (pi + ki) mod 26).\n"
            "- Giải mã: pi = (ci ⊕ ki) (hoặc (ci - ki) mod 26).\n\n"
            "Đặc tính an toàn:\n"
            "- Đạt an ninh vô điều kiện (Unconditional Security hay Perfect Secrecy do Shannon chứng minh năm 1949): "
            "Kẻ thù dù có năng lực tính toán vô hạn cũng không thể phá vỡ vì mọi bản rõ đều có xác suất xuất hiện như nhau.\n"
            "- Nhược điểm thực tế: Khóa phải dài bằng bản rõ và việc phân phối khóa ngẫu nhiên một lần gặp nhiều khó khăn trong thực tế."
        ),
        "bullet_types": [
            "- Khóa ngẫu nhiên hoàn toàn có độ dài bằng bản rõ.",
            "- Khóa chỉ sử dụng đúng một lần rồi hủy bỏ.",
            "- Đạt an ninh vô điều kiện (an toàn tuyệt đối về mặt lý thuyết thông tin).",
        ],
        "example": "Bản rõ P = 'HELLO', Khóa K = ngẫu nhiên cùng độ dài. ci = (pi + ki) mod 26.",
    },
    {
        "id": "crypto_qa_04",
        "topic": "Hệ mã hóa dòng",
        "category": "cipher_system",
        "canonical_terms": ["mã hóa dòng", "stream cipher", "khóa dòng", "keystream"],
        "question": "Hệ mã hóa dòng (Stream Cipher) hoạt động như thế nào?",
        "intent": "mechanism_and_formula",
        "expected_action": "explain_stream_cipher_mechanism",
        "grounded_answer": (
            "Hệ mã hóa dòng (Stream Cipher) là kỹ thuật mã hóa dữ liệu đối xứng theo từng ký tự hoặc từng bit một một cách liên tục.\n\n"
            "Cơ chế hoạt động:\n"
            "- Một bộ sinh dòng khóa (Pseudorandom Keystream Generator) nhận một khóa bí mật K có độ dài cố định và sinh ra dòng khóa giả ngẫu nhiên z1, z2, z3, ...\n"
            "- Quá trình mã hóa: ci = pi ⊕ zi (hoặc ci = (pi + zi) mod m).\n"
            "- Quá trình giải mã: pi = ci ⊕ zi (hoặc pi = (ci - zi) mod m).\n\n"
            "Ưu điểm:\n"
            "- Tốc độ xử lý cực nhanh, độ trễ thấp, phù hợp với truyền thông thời gian thực (audio, video, mạng viễn thông).\n"
            "- Không yêu cầu bộ đệm dữ liệu lớn như mã khối (Block Cipher)."
        ),
        "bullet_types": [
            "- Mã hóa từng bit/byte liên tục bằng phép toán XOR hoặc modulo.",
            "- Bộ tạo dòng khóa sinh dòng giả ngẫu nhiên từ khóa gốc.",
            "- Tốc độ cao, bộ nhớ đệm nhỏ.",
        ],
        "example": "Mã hóa bit: p = 1, z = 0 => c = 1 ⊕ 0 = 1; giải mã: p = 1 ⊕ 0 = 1.",
    },
    {
        "id": "crypto_qa_05",
        "topic": "Hệ mã Playfair",
        "category": "cipher_system",
        "canonical_terms": ["playfair", "hệ mã playfair", "ma trận 5x5", "digraph"],
        "question": "Quy tắc mã hóa trong hệ mã Playfair là gì và cho ví dụ?",
        "intent": "mechanism_rules_and_example",
        "expected_action": "explain_playfair_rules",
        "grounded_answer": (
            "Hệ mã Playfair là hệ mã hóa thay thế nhiều chữ cái (digraph cipher) sử dụng ma trận chữ cái kích thước 5x5 "
            "được xây dựng dựa trên một từ khóa bí mật (thường gộp I và J chung một ô).\n\n"
            "Quy tắc mã hóa theo từng cặp ký tự (m1, m2):\n"
            "1. Nếu hai chữ cái trùng nhau trong một cặp: Chèn thêm ký tự đệm (thường là 'X') vào giữa.\n"
            "2. Cùng hàng: Thay bằng chữ cái ngay bên phải (vòng lại đầu hàng nếu ở cuối hàng).\n"
            "3. Cùng cột: Thay bằng chữ cái ngay bên dưới (vòng lại đầu cột nếu ở cuối cột).\n"
            "4. Khác hàng và khác cột (tạo thành góc hình chữ nhật): Thay bằng chữ cái nằm trên cùng hàng của nó nhưng ở cột của chữ cái kia."
        ),
        "bullet_types": [
            "- Ma trận khóa 5x5 dựa trên từ khóa (I và J chung ô).",
            "- Mã hóa từng cặp chữ cái (digraph).",
            "- 3 quy tắc biến đổi: cùng hàng, cùng cột, hoặc góc hình chữ nhật.",
        ],
        "example": "Từ khóa 'MONARCHY', bản rõ cặp 'TH' tạo hình chữ nhật sẽ thay thế bằng hai đỉnh đối diện tương ứng.",
    },
    {
        "id": "crypto_qa_06",
        "topic": "Hệ mã Affine",
        "category": "formula_and_example",
        "canonical_terms": ["affine", "hệ mã affine", "mod 26", "nghịch đảo"],
        "question": "Công thức và điều kiện tồn tại của hệ mã Affine là gì?",
        "intent": "mathematical_formulation",
        "expected_action": "explain_affine_math",
        "grounded_answer": (
            "Hệ mã Affine là hệ mã hóa thay thế đơn bảng kết hợp phép nhân và phép cộng trong số học modulo 26.\n\n"
            "Công thức toán học:\n"
            "- Không gian khóa K = (a, b) ∈ Z26 × Z26.\n"
            "- Hàm mã hóa: e(x) = (a * x + b) mod 26.\n"
            "- Hàm giải mã: d(y) = a^(-1) * (y - b) mod 26, trong đó a^(-1) là phần tử nghịch đảo nhân của a trong modulo 26.\n\n"
            "Điều kiện then chốt:\n"
            "Khóa a phải nguyên tố cùng nhau với 26, tức là gcd(a, 26) = 1. "
            "Có đúng 12 giá trị a thỏa mãn: {1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25}."
        ),
        "bullet_types": [
            "- Hàm mã hóa: e(x) = (a*x + b) mod 26.",
            "- Hàm giải mã: d(y) = a^(-1) * (y - b) mod 26.",
            "- Điều kiện: gcd(a, 26) = 1 (12 giá trị khả dĩ cho a).",
        ],
        "example": "Chọn a = 7, b = 3 (gcd(7, 26) = 1, nghịch đảo 7^(-1) = 15). Chữ 'A'(0) -> e(0) = (7*0 + 3) mod 26 = 3 ('D').",
    },
    {
        "id": "crypto_qa_07",
        "topic": "Mã hóa thay thế so với Mã hóa hoán vị",
        "category": "comparison",
        "canonical_terms": [
            "mã hóa thay thế",
            "mã hóa hoán vị",
            "substitution",
            "transposition",
        ],
        "question": "Phân biệt mã hóa thay thế và mã hóa hoán vị?",
        "intent": "comparative_analysis",
        "expected_action": "compare_substitution_and_transposition",
        "grounded_answer": (
            "Mã hóa thay thế và mã hóa hoán vị là hai kỹ thuật nền tảng của mật mã học cổ điển:\n\n"
            "1. Mã hóa thay thế (Substitution):\n"
            "- Bản chất: Thay đổi danh tính ký tự, giữ nguyên vị trí.\n"
            "- Mỗi ký tự trong bản rõ được thay bằng một ký tự khác theo quy tắc (Caesar, Affine, Playfair...).\n"
            "- Điểm yếu: Làm thay đổi ký tự nhưng phân bố tần suất có thể bị lộ nếu không dùng mã đa bảng.\n\n"
            "2. Mã hóa hoán vị (Transposition):\n"
            "- Bản chất: Giữ nguyên danh tính ký tự, thay đổi vị trí.\n"
            "- Các ký tự ban đầu giữ nguyên số lượng và tần suất xuất hiện, chỉ đảo trật tự (hoán vị theo cột, ma trận, hoán vị khối).\n"
            "- Điểm yếu: Tần suất ký tự không đổi, kẻ phá mã có thể dùng biểu đồ tần suất để nhận diện."
        ),
        "bullet_types": [
            "* Mã hóa thay thế: đổi giá trị ký tự, giữ nguyên vị trí.",
            "* Mã hóa hoán vị: đổi vị trí ký tự, giữ nguyên giá trị.",
        ],
        "example": "Bản rõ 'CAT': Thay thế (+1) -> 'DBU'; Hoán vị đảo ngược -> 'TAC'.",
    },
    {
        "id": "crypto_qa_08",
        "topic": "Hệ mã Autokey",
        "category": "cipher_system",
        "canonical_terms": ["autokey", "hệ mã autokey", "khóa tự sinh"],
        "question": "Hệ mã hóa Autokey giải quyết nhược điểm gì của hệ mã Vigenère?",
        "intent": "system_advancement_and_mechanism",
        "expected_action": "explain_autokey_advantage",
        "grounded_answer": (
            "Hệ mã Autokey (khóa tự sinh) được thiết kế nhằm khắc phục nhược điểm chu kỳ tuần hoàn lặp lại của khóa trong hệ mã Vigenère "
            "(vốn dễ bị thám mã bằng phương pháp Kasiski hoặc chỉ số trùng khớp Index of Coincidence).\n\n"
            "Cơ chế:\n"
            "- Khóa bắt đầu bằng một từ khóa ngắn bí mật K.\n"
            "- Sau khi dùng hết các ký tự của khóa K, dòng khóa tiếp theo được tạo ra bằng cách lấy chính các ký tự của bản rõ (Plaintext Autokey) "
            "hoặc bản mã (Ciphertext Autokey).\n"
            "- Nhờ đó dòng khóa không bị lặp lại tuần hoàn, triệt tiêu tính chu kỳ của Vigenère."
        ),
        "bullet_types": [
            "- Khắc phục tính tuần hoàn chu kỳ của Vigenère.",
            "- Sử dụng chính bản rõ để nối dài dòng khóa.",
            "- Ngăn chặn phương pháp thám mã Kasiski.",
        ],
        "example": "Khóa K = 'KEY', Bản rõ P = 'SECURITY'. Dòng khóa trở thành 'KEYSECUR' thay vì 'KEYKEYKE'.",
    },
    {
        "id": "crypto_qa_09",
        "topic": "Hệ mã dịch vòng (Caesar)",
        "category": "cipher_system",
        "canonical_terms": ["dịch vòng", "caesar", "hệ mã dịch vòng", "mã caesar"],
        "question": "Hệ mã dịch vòng (Caesar) là gì và công thức tính như thế nào?",
        "intent": "system_concept_and_formula",
        "expected_action": "explain_caesar_system",
        "grounded_answer": (
            "Hệ mã dịch vòng (Caesar cipher) là một trong những hệ mã thay thế đơn bảng cổ xưa nhất, "
            "trong đó mỗi chữ cái trong bản rõ được thay thế bằng chữ cái cách nó một khoảng k vị trí trong bảng chữ cái.\n\n"
            "Công thức toán học trên vành Z26:\n"
            "- Không gian khóa K = {0, 1, ..., 25}.\n"
            "- Hàm mã hóa: c = (p + k) mod 26.\n"
            "- Hàm giải mã: p = (c - k) mod 26.\n\n"
            "Đặc điểm an toàn:\n"
            "Không gian khóa cực kỳ nhỏ (|K| = 26, trong đó k = 0 không đổi nên thực chất chỉ có 25 khóa có ý nghĩa), "
            "do đó hoàn toàn có thể bị phá mã dễ dàng bằng tấn công duyệt vét cạn (brute-force attack) hoặc phân tích tần suất."
        ),
        "bullet_types": [
            "- Không gian khóa: K = Z26 (26 khóa khả dĩ).",
            "- Hàm mã hóa: c = (p + k) mod 26.",
            "- Hàm giải mã: p = (c - k) mod 26.",
        ],
        "example": "Với k = 3, chữ 'A'(0) -> (0 + 3) mod 26 = 3 ('D'), 'HELLO' -> 'KHOOR'.",
    },
    {
        "id": "crypto_qa_10",
        "topic": "Hệ mã Vigenère",
        "category": "cipher_system",
        "canonical_terms": ["vigenere", "hệ mã vigenere", "đa bảng", "polyalphabetic"],
        "question": "Hệ mã Vigenère hoạt động như thế nào và tại sao gọi là mã đa bảng?",
        "intent": "system_concept_and_mechanism",
        "expected_action": "explain_vigenere_mechanism",
        "grounded_answer": (
            "Hệ mã Vigenère là hệ mã hóa thay thế đa bảng (polyalphabetic substitution cipher), "
            "sử dụng một từ khóa có độ dài m để dịch chuyển các chữ cái bản rõ theo các bước dịch chuyển khác nhau một cách tuần hoàn.\n\n"
            "Nguyên lý hoạt động:\n"
            "- Cho từ khóa K = (k0, k1, ..., km-1).\n"
            "- Mã hóa: ci = (pi + k_(i mod m)) mod 26.\n"
            "- Giải mã: pi = (ci - k_(i mod m)) mod 26.\n\n"
            "Tại sao gọi là mã đa bảng?\n"
            "Vì cùng một chữ cái trong bản rõ (ví dụ chữ 'E') khi xuất hiện ở các vị trí khác nhau có thể được thay thế bằng "
            "các chữ cái bản mã khác nhau tùy theo chữ cái tương ứng của từ khóa. Điều này làm phẳng phân bố tần suất chữ cái, "
            "khiến phương pháp phân tích tần suất đơn giản bị vô hiệu hóa."
        ),
        "bullet_types": [
            "- Thay thế đa bảng với từ khóa tuần hoàn độ dài m.",
            "- Công thức: ci = (pi + k_(i mod m)) mod 26.",
            "- Làm phẳng biểu đồ tần suất xuất hiện ký tự đơn.",
        ],
        "example": "Bản rõ 'ATTACK', Khóa 'LEMON': 'A' + 'L' -> 'L', 'T' + 'E' -> 'X', ...",
    },
    {
        "id": "crypto_qa_11",
        "topic": "Hệ mã Hill",
        "category": "formula_and_example",
        "canonical_terms": ["hill", "hệ mã hill", "mã ma trận", "ma trận khả nghịch"],
        "question": "Hệ mã Hill là gì và điều kiện để giải mã được là gì?",
        "intent": "mathematical_system_and_condition",
        "expected_action": "explain_hill_cipher",
        "grounded_answer": (
            "Hệ mã Hill (do Lester S. Hill phát minh năm 1929) là hệ mã hóa thay thế nhiều chữ cái (polygraphic cipher) "
            "dựa trên đại số tuyến tính modulo 26.\n\n"
            "Cơ chế:\n"
            "- Bản rõ được chia thành các vector kích thước m: P = (p1, p2, ..., pm).\n"
            "- Khóa bí mật là một ma trận vuông K kích thước m × m với các phần tử thuộc Z26.\n"
            "- Mã hóa: C = P * K mod 26.\n"
            "- Giải mã: P = C * K^(-1) mod 26.\n\n"
            "Điều kiện giải mã:\n"
            "Ma trận khóa K phải khả nghịch trên vành Z26, tức là định thức det(K) phải thỏa mãn:\n"
            "gcd(det(K), 26) = 1 (nghĩa là det(K) không chia hết cho 2 hoặc 13)."
        ),
        "bullet_types": [
            "- Phép nhân vector-ma trận: C = P * K mod 26.",
            "- Điều kiện khả nghịch: gcd(det(K), 26) = 1.",
            "- Giải mã bằng ma trận nghịch đảo: K^(-1) mod 26.",
        ],
        "example": "Ma trận K cỡ 2x2: [[11, 8], [3, 7]], det(K) = 77 - 24 = 53 ≡ 1 mod 26 => Khả nghịch.",
    },
    {
        "id": "crypto_qa_12",
        "topic": "Hệ mã đơn bảng",
        "category": "cipher_system",
        "canonical_terms": ["đơn bảng", "hệ mã đơn bảng", "monoalphabetic", "tần suất chữ cái"],
        "question": "Hệ mã đơn bảng là gì và điểm yếu lớn nhất của nó là gì?",
        "intent": "system_concept_and_vulnerability",
        "expected_action": "explain_monoalphabetic_vulnerability",
        "grounded_answer": (
            "Hệ mã đơn bảng (Monoalphabetic cipher) là hệ mã thay thế trong đó mỗi chữ cái trong bảng chữ cái bản rõ "
            "được ánh xạ cố định 1-1 tới một chữ cái duy nhất trong bảng chữ cái bản mã trong suốt toàn bộ thông điệp.\n\n"
            "Không gian khóa:\n"
            "Số lượng hoán vị của 26 chữ cái là 26! ≈ 4 × 10^26 (một con số rất lớn, chống được duyệt vét cạn thô thiển).\n\n"
            "Điểm yếu chí mạng:\n"
            "Do tính ánh xạ cố định 1-1, phân bố tần suất xuất hiện của các chữ cái trong ngôn ngữ tự nhiên được bảo toàn nguyên vẹn trong bản mã. "
            "Kẻ thám mã chỉ cần thống kê tần suất xuất hiện các chữ cái trong bản mã (ví dụ trong tiếng Anh: E xuất hiện nhiều nhất ~12.7%, tiếp theo là T, A, O, N) "
            "để dễ dàng khôi phục hoàn toàn bản rõ mà không cần thử hết 26! khóa."
        ),
        "bullet_types": [
            "- Ánh xạ 1-1 cố định cho từng ký tự.",
            "- Không gian khóa lớn: 26! ≈ 4 * 10^26.",
            "- Bị phá vỡ hoàn toàn bởi phân tích tần suất ngôn ngữ tự nhiên.",
        ],
        "example": "Nếu chữ 'Q' xuất hiện nhiều nhất trong bản mã tiếng Anh, khả năng cao 'Q' là bản mã của 'E'.",
    },
    {
        "id": "crypto_qa_13",
        "topic": "Mã hóa đối xứng so với Mã hóa bất đối xứng",
        "category": "comparison",
        "canonical_terms": [
            "mã hóa đối xứng",
            "mã hóa bất đối xứng",
            "khóa công khai",
            "khóa bí mật",
        ],
        "question": "So sánh mã hóa đối xứng và mã hóa bất đối xứng (khóa công khai)?",
        "intent": "comparative_overview",
        "expected_action": "compare_symmetric_vs_asymmetric",
        "grounded_answer": (
            "So sánh giữa mã hóa đối xứng và mã hóa bất đối xứng:\n\n"
            "1. Số lượng khóa:\n"
            "- Đối xứng: Sử dụng 1 khóa bí mật chung cho cả mã hóa và giải mã.\n"
            "- Bất đối xứng: Sử dụng 1 cặp khóa gồm Khóa công khai (Public key - dùng để mã hóa) và Khóa riêng tư (Private key - dùng để giải mã).\n\n"
            "2. Tốc độ & Hiệu năng:\n"
            "- Đối xứng: Rất nhanh (phù hợp mã hóa khối lượng dữ liệu lớn như AES, DES, ChaCha20, Stream cipher).\n"
            "- Bất đối xứng: Chậm hơn từ 100 đến 1000 lần do phụ thuộc các phép toán số học lớn (RSA, ECC, ElGamal).\n\n"
            "3. Vấn đề phân phối khóa:\n"
            "- Đối xứng: Khó khăn trong việc trao đổi khóa bí mật ban đầu an toàn qua kênh truyền công cộng.\n"
            "- Bất đối xứng: Giải quyết trọn vẹn bài toán phân phối khóa và hỗ trợ chữ ký số (Digital Signature).\n\n"
            "Trong thực tế, các giao thức hiện đại (như TLS/HTTPS) kết hợp mô hình lai (Hybrid Cryptosystem): "
            "dùng mã hóa bất đối xứng để thỏa thuận khóa phiên bí mật, sau đó dùng mã hóa đối xứng để mã hóa dữ liệu truyền tải."
        ),
        "bullet_types": [
            "- Đối xứng: 1 khóa bí mật chung, tốc độ cao, khó phân phối khóa.",
            "- Bất đối xứng: 2 khóa riêng biệt (công khai / bí mật), tính toán nặng, giải quyết phân phối khóa.",
            "- Hệ thống lai: Kết hợp cả hai để đạt tối ưu tốc độ và an toàn.",
        ],
        "example": "HTTPS/TLS dùng RSA/ECDH để trao đổi khóa AES, sau đó truyền dữ liệu bằng AES-GCM.",
    },
]
