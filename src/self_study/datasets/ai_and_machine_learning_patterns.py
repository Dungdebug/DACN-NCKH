"""
Domain-Specific Artificial Intelligence, Machine Learning, RAG & Nested Learning Patterns.
Provides bidirectional synonym mapping, domain query expansion, and few-shot QA blueprints
for authoritative AI/ML curricula, retrieval-augmented generation (RAG), and continuum memory systems.
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


# Core Artificial Intelligence, Machine Learning & RAG domain synonym clusters
AI_ML_SYNONYM_CLUSTERS: Dict[str, List[str]] = {
    "supervised_learning": [
        "học có giám sát",
        "học giám sát",
        "học có hướng dẫn",
        "học có thầy",
        "supervised learning",
        "supervised",
        "phân lớp và hồi quy",
        "classification and regression",
        "mô hình có giám sát",
        "dữ liệu có gán nhãn",
        "labeled data",
    ],
    "unsupervised_learning": [
        "học không giám sát",
        "học không hướng dẫn",
        "học không thầy",
        "unsupervised learning",
        "unsupervised",
        "gom cụm",
        "clustering",
        "phân cụm",
        "giảm chiều dữ liệu",
        "dimensionality reduction",
        "dữ liệu không nhãn",
        "unlabeled data",
    ],
    "reinforcement_learning": [
        "học tăng cường",
        "học củng cố",
        "reinforcement learning",
        "rl",
        "q-learning",
        "policy gradient",
        "môi trường và tác tử",
        "agent and environment",
        "hàm phần thưởng",
        "reward function",
        "markov decision process",
        "mdp",
        "deep reinforcement learning",
    ],
    "overfitting": [
        "quá khớp",
        "quá khớp mô hình",
        "overfitting",
        "overfit",
        "học vẹt",
        "sai số kiểm thử cao",
        "high variance",
        "phương sai cao",
        "chính quy hóa",
        "regularization",
        "l1 l2 regularization",
        "dropout",
        "early stopping",
    ],
    "underfitting": [
        "dưới khớp",
        "chưa khớp",
        "underfitting",
        "underfit",
        "high bias",
        "độ chệch cao",
        "mô hình quá đơn giản",
        "chưa hội tụ",
    ],
    "gradient_descent": [
        "hạ độ dốc",
        "giảm độ dốc",
        "gradient descent",
        "lan truyền ngược",
        "backpropagation",
        "tốc độ học",
        "learning rate",
        "sgd",
        "stochastic gradient descent",
        "hạ độ dốc ngẫu nhiên",
        "mini-batch gradient descent",
        "adam",
        "adam optimizer",
        "tối ưu hóa độ dốc",
        "cập nhật trọng số",
    ],
    "attention_transformer": [
        "cơ chế chú ý",
        "chú ý",
        "attention",
        "attention mechanism",
        "self-attention",
        "tự chú ý",
        "multi-head attention",
        "chú ý đa đầu",
        "transformer",
        "kiến trúc transformer",
        "scaled dot-product attention",
        "encoder decoder",
        "chú ý đa hướng",
        "self attention",
    ],
    "rag": [
        "rag",
        "retrieval-augmented generation",
        "retrieval augmented generation",
        "tạo sinh tăng cường truy xuất",
        "tăng cường truy xuất",
        "truy xuất tăng cường tạo sinh",
        "cơ sở dữ liệu vector",
        "vector database",
        "vector db",
        "embedding",
        "nhúng văn bản",
        "reranking",
        "xếp hạng lại",
        "truy xuất ngữ nghĩa",
        "semantic retrieval",
    ],
    "bm25": [
        "bm25",
        "okapi bm25",
        "best matching 25",
        "truy xuất từ khóa",
        "lexical retrieval",
        "tf-idf",
        "tần suất từ",
        "độ nghịch đảo tần suất tài liệu",
        "lexical search",
        "tìm kiếm từ vựng",
        "khớp từ khóa chính xác",
        "sparse retrieval",
    ],
    "sa_cms": [
        "sa-cms",
        "sa cms",
        "sparse-aware chunk management system",
        "quản lý đoạn văn nhận biết thưa",
        "sparse chunking",
        "phân mảnh nhận thức thưa",
        "chunking",
        "phân đoạn văn bản",
        "semantic chunking",
        "phân đoạn ngữ nghĩa",
        "chunk management",
        "quản lý chunk",
    ],
    "continuum_memory": [
        "continuum memory",
        "bộ nhớ liên tục",
        "nested learning",
        "học lồng ghép",
        "học lồng nhau",
        "continual learning",
        "lifelong learning",
        "học suốt đời",
        "trí nhớ liên tục",
        "bộ nhớ mở rộng",
        "catastrophic forgetting",
        "quên thảm họa",
        "episodic memory",
        "semantic memory",
        "bộ nhớ ngữ nghĩa",
        "bộ nhớ sự kiện",
    ],
    "deep_learning": [
        "học sâu",
        "deep learning",
        "mạng nơ-ron",
        "mạng neural",
        "neural network",
        "artificial neural network",
        "ann",
        "deep neural network",
        "dnn",
        "cnn",
        "convolutional neural network",
        "rnn",
        "recurrent neural network",
        "lstm",
    ],
    "evaluation_metrics": [
        "độ chính xác",
        "accuracy",
        "precision",
        "recall",
        "f1-score",
        "f1 score",
        "độ đo đánh giá",
        "confusion matrix",
        "ma trận nhầm lẫn",
        "roc-auc",
        "roc auc",
        "hàm mất mát",
        "loss function",
    ],
}


class AIMLThesaurusRegistry:
    """
    Registry for bidirectional Artificial Intelligence, Machine Learning and RAG terminology mappings.
    Supports exact matching, tone-insensitive matching, and n-gram term discovery.
    """

    def __init__(self, clusters: Optional[Dict[str, List[str]]] = None):
        self.clusters = clusters or AI_ML_SYNONYM_CLUSTERS
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
        Extracts AI/ML/RAG domain terms from the query and expands them into synonyms.
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
_DEFAULT_REGISTRY = AIMLThesaurusRegistry()


def get_ai_ml_synonyms(term: str) -> List[str]:
    """Public helper to get bidirectional AI, Machine Learning, and RAG synonyms."""
    return _DEFAULT_REGISTRY.get_synonyms(term)


def expand_ai_ml_query_terms(query: str) -> List[str]:
    """
    Expands conversational user queries into authoritative technical AI/ML keywords.
    E.g.: 'học có giám sát' -> ['học có giám sát', 'supervised learning', 'phân lớp và hồi quy', ...]
    """
    return _DEFAULT_REGISTRY.expand_query(query)


# Few-shot QA examples for Artificial Intelligence, Machine Learning, RAG and Nested Learning curricula
AI_ML_FEW_SHOT_EXAMPLES: List[Dict[str, Any]] = [
    {
        "id": "ai_ml_qa_01",
        "topic": "Học có giám sát (Supervised Learning)",
        "category": "definition_and_taxonomy",
        "canonical_terms": [
            "học có giám sát",
            "supervised learning",
            "phân lớp và hồi quy",
            "dữ liệu có gán nhãn",
        ],
        "question": "Học có giám sát (Supervised Learning) là gì, gồm những bài toán cơ bản nào và cho ví dụ?",
        "intent": "concept_and_classification",
        "expected_action": "explain_concept_and_list_all_branches",
        "grounded_answer": (
            "Học có giám sát (Supervised Learning) là một phân nhánh quan trọng của Học máy (Machine Learning), "
            "trong đó thuật toán được huấn luyện trên một tập dữ liệu đã được gán nhãn sẵn: D = {(x1, y1), (x2, y2), ..., (xN, yN)}. "
            "Mục tiêu là học được một hàm ánh xạ f: X -> Y sao cho với một điểm dữ liệu mới x*, mô hình có thể dự đoán chính xác nhãn y*.\n\n"
            "Các bài toán cơ bản trong học có giám sát bao gồm:\n"
            "1. Phân loại (Classification): Đầu ra y là biến rời rạc đại diện cho các lớp (nhãn nhị phân hoặc đa lớp). "
            "Ví dụ: Phân loại email (Spam / Không Spam), chẩn đoán bệnh ung thư (Lành tính / Ác tính), nhận diện chữ số viết tay MNIST (0-9).\n"
            "2. Hồi quy (Regression): Đầu ra y là một giá trị liên tục trong tập số thực. "
            "Ví dụ: Dự đoán giá nhà dựa trên diện tích và vị trí, dự báo nhiệt độ thời tiết, ước tính doanh thu quý.\n\n"
            "Các thuật toán kinh điển:\n"
            "- Hồi quy tuyến tính (Linear Regression), Hồi quy Logistic (Logistic Regression).\n"
            "- Cây quyết định (Decision Tree), Rừng ngẫu nhiên (Random Forest), Gradient Boosting (XGBoost, LightGBM).\n"
            "- Máy học vector hỗ trợ (Support Vector Machines - SVM).\n"
            "- Mạng nơ-ron nhân tạo và Học sâu (Artificial Neural Networks - ANN, CNN, RNN)."
        ),
        "bullet_types": [
            "- Huấn luyện trên tập dữ liệu đã gán nhãn D = {(xi, yi)}.",
            "- Hai bài toán cốt lõi: Phân loại (biến rời rạc) và Hồi quy (biến liên tục).",
            "- Mục tiêu tối ưu hàm mất mát Loss function L(y, f(x)) trên tập huấn luyện.",
        ],
        "example": "Mô hình hồi quy dự đoán giá nhà: y = w1*diện_tích + w2*số_phòng + b, với w là vector trọng số học được từ dữ liệu quá khứ.",
    },
    {
        "id": "ai_ml_qa_02",
        "topic": "Học không giám sát (Unsupervised Learning)",
        "category": "definition_and_taxonomy",
        "canonical_terms": [
            "học không giám sát",
            "unsupervised learning",
            "gom cụm",
            "clustering",
            "giảm chiều dữ liệu",
        ],
        "question": "Học không giám sát (Unsupervised Learning) là gì, ứng dụng vào những bài toán nào?",
        "intent": "concept_and_application",
        "expected_action": "explain_unsupervised_learning_and_tasks",
        "grounded_answer": (
            "Học không giám sát (Unsupervised Learning) là phương pháp học máy trong đó tập dữ liệu huấn luyện "
            "chỉ chứa các vector đầu vào x mà không có bất kỳ nhãn mục tiêu y nào đi kèm: D = {x1, x2, ..., xN}. "
            "Thuật toán có nhiệm vụ tự động khám phá cấu trúc tiềm ẩn, quy luật phân bố hoặc mối tương quan nội tại giữa các điểm dữ liệu.\n\n"
            "Các bài toán chính:\n"
            "1. Gom cụm (Clustering): Nhóm các điểm dữ liệu có đặc trưng tương đồng vào cùng một cụm. "
            "Ví dụ: Phân khúc khách hàng mục tiêu trong Marketing, gom cụm các bài báo có cùng chủ đề tin tức. "
            "Thuật toán tiêu biểu: K-Means, Gaussian Mixture Models (GMM), DBSCAN, Hierarchical Clustering.\n"
            "2. Giảm chiều dữ liệu (Dimensionality Reduction): Nén không gian đặc trưng từ số chiều cao xuống số chiều thấp hơn "
            "mà vẫn bảo toàn tối đa thông tin phương sai và cấu trúc hình học. "
            "Thuật toán tiêu biểu: Phân tích thành phần chính (PCA), t-SNE, UMAP, Autoencoders.\n"
            "3. Phát hiện bất thường (Anomaly Detection): Nhận diện các điểm dữ liệu khác biệt dị thường so với phân phối chung. "
            "Ví dụ: Phát hiện gian lận thẻ tín dụng, phát hiện lỗi cảm biến IoT công nghiệp."
        ),
        "bullet_types": [
            "- Dữ liệu đầu vào không gán nhãn: D = {xi}.",
            "- Khám phá cấu trúc tiềm ẩn: Phân cụm (Clustering), Giảm chiều (Dimensionality Reduction), Phát hiện bất thường.",
            "- Không có giám sát từ con người trong quá trình sinh nhãn mục tiêu.",
        ],
        "example": "K-Means gom 10.000 khách hàng thành 4 nhóm theo tần suất mua và giá trị đơn hàng trung bình.",
    },
    {
        "id": "ai_ml_qa_03",
        "topic": "Học tăng cường (Reinforcement Learning)",
        "category": "concept_and_classification",
        "canonical_terms": [
            "học tăng cường",
            "reinforcement learning",
            "q-learning",
            "policy gradient",
            "môi trường và tác tử",
        ],
        "question": "Học tăng cường (Reinforcement Learning) là gì và các thành phần cốt lõi của nó gồm những gì?",
        "intent": "system_concept_and_mechanism",
        "expected_action": "explain_rl_and_components",
        "grounded_answer": (
            "Học tăng cường (Reinforcement Learning - RL) là mô hình học máy trong đó một tác tử thông minh (Agent) "
            "tương tác qua lại với một môi trường động (Environment) theo chuỗi thời gian nhằm tối đa hóa tổng phần thưởng tích lũy (Cumulative Reward).\n\n"
            "Mô hình được biểu diễn toán học qua Quá trình Quyết định Markov (Markov Decision Process - MDP) bao gồm 5 thành phần cốt lõi:\n"
            "- Trạng thái (State - S): Biểu diễn tình trạng hiện tại của môi trường tại thời điểm t (st in S).\n"
            "- Hành động (Action - A): Tập hợp các hành vi mà tác tử có thể thực thi (at in A).\n"
            "- Môi trường (Environment): Thế giới bao quanh tác tử, phản hồi lại hành động bằng trạng thái mới st+1 và phần thưởng.\n"
            "- Hàm phần thưởng (Reward - R): Giá trị vô hướng rt = R(st, at) phản ánh mức độ tốt/xấu của hành động tức thời.\n"
            "- Chính sách (Policy - pi): Chiến lược ra quyết định của tác tử, ánh xạ từ trạng thái sang xác suất chọn hành động pi(a|s).\n\n"
            "Hai nhóm phương pháp tiếp cận chủ đạo:\n"
            "1. Dựa trên hàm giá trị (Value-based): Ước lượng hàm chất lượng Q(s, a) (như Q-Learning, DQN).\n"
            "2. Dựa trên chính sách (Policy-based): Tối ưu trực tiếp tham số hóa chính sách pi_theta (như REINFORCE, PPO, TRPO)."
        ),
        "bullet_types": [
            "- Tác tử tương tác với môi trường theo chu trình: Trạng thái -> Hành động -> Phần thưởng.",
            "- Khung toán học: Quá trình quyết định Markov (MDP) gồm (S, A, P, R, gamma).",
            "- Mục tiêu: Tối đa hóa kỳ vọng tổng phần thưởng chiết khấu trong tương lai.",
        ],
        "example": "Tác tử AlphaGo học chơi cờ vây bằng cách tự chơi với chính mình hàng triệu ván, nhận phần thưởng +1 khi thắng và -1 khi thua.",
    },
    {
        "id": "ai_ml_qa_04",
        "topic": "Quá khớp (Overfitting) và Kỹ thuật Chính quy hóa",
        "category": "optimization_and_training",
        "canonical_terms": [
            "quá khớp",
            "overfitting",
            "regularization",
            "chính quy hóa",
            "dropout",
            "early stopping",
        ],
        "question": "Hiện tượng Quá khớp (Overfitting) là gì, nguyên nhân và các giải pháp ngăn chặn?",
        "intent": "problem_diagnosis_and_solutions",
        "expected_action": "explain_overfitting_causes_and_preventions",
        "grounded_answer": (
            "Quá khớp (Overfitting) là hiện tượng mô hình học máy học quá mức chi tiết cả các quy luật ngẫu nhiên và nhiễu (noise) "
            "trong tập dữ liệu huấn luyện (Training Set), dẫn đến mất đi khả năng tổng quát hóa (Generalization) trên dữ liệu mới chưa từng thấy (Test Set/Unseen Data).\n\n"
            "Dấu hiệu nhận biết:\n"
            "- Sai số trên tập huấn luyện (Training Loss) cực kỳ thấp, độ chính xác gần như 100%.\n"
            "- Sai số trên tập kiểm định/kiểm thử (Validation/Test Loss) rất cao hoặc tăng vọt (phương sai cao - High Variance).\n\n"
            "Nguyên nhân chính:\n"
            "- Mô hình quá phức tạp so với số lượng mẫu dữ liệu (quá nhiều tham số, mạng nơ-ron quá sâu).\n"
            "- Dữ liệu huấn luyện quá ít hoặc chứa nhiều nhiễu.\n\n"
            "Các giải pháp kỹ thuật ngăn chặn hiệu quả:\n"
            "1. Chính quy hóa (Regularization): Bổ sung số hạng phạt trọng số vào hàm mục tiêu: L_total = L_data + lambda * Omega(w). "
            "Bao gồm L1 Regularization (Lasso - tạo vector thưa) và L2 Regularization (Ridge - co nhỏ trọng số).\n"
            "2. Dropout: Trong mạng nơ-ron sâu, ngắt ngẫu nhiên một tỷ lệ nơ-ron (ví dụ p = 0.5) trong mỗi lượt lan truyền để ngăn các nơ-ron phụ thuộc lẫn nhau.\n"
            "3. Dừng sớm (Early Stopping): Giám sát Validation Loss qua từng epoch, dừng quá trình huấn luyện khi Validation Loss bắt đầu tăng trở lại.\n"
            "4. Tăng cường dữ liệu (Data Augmentation): Làm giàu tập dữ liệu huấn luyện bằng các phép biến đổi hình học, làm mờ, xoay ảnh, dịch chuyển từ ngữ.\n"
            "5. Đơn giản hóa kiến trúc mô hình hoặc thu thập thêm dữ liệu gán nhãn chất lượng."
        ),
        "bullet_types": [
            "- Bản chất: Học vẹt cả nhiễu, sai số huấn luyện thấp nhưng sai số kiểm định cao.",
            "- Biện pháp toán học: L1/L2 Regularization phạt độ lớn ma trận trọng số.",
            "- Biện pháp huấn luyện mạng sâu: Dropout, Early Stopping, Data Augmentation.",
        ],
        "example": "Mạng phân loại ảnh đạt 99.8% độ chính xác trên tập train nhưng chỉ đạt 62.4% trên tập test do học vẹt hình nền.",
    },
    {
        "id": "ai_ml_qa_05",
        "topic": "Thuật toán Hạ độ dốc (Gradient Descent)",
        "category": "formula_and_mechanism",
        "canonical_terms": [
            "hạ độ dốc",
            "gradient descent",
            "lan truyền ngược",
            "learning rate",
            "sgd",
            "adam",
        ],
        "question": "Thuật toán Hạ độ dốc (Gradient Descent) hoạt động như thế nào và các biến thể của nó là gì?",
        "intent": "mathematical_system_and_variants",
        "expected_action": "explain_gradient_descent_mechanism_and_variants",
        "grounded_answer": (
            "Hạ độ dốc (Gradient Descent) là thuật toán tối ưu hóa lặp bậc nhất phổ biến nhất trong Học máy và Học sâu, "
            "dùng để tìm kiếm bộ trọng số w sao cho giá trị hàm mất mát L(w) đạt giá trị cực tiểu toàn cục hoặc cực tiểu địa phương.\n\n"
            "Nguyên lý hoạt động và công thức cập nhật:\n"
            "Vector gradient grad(L(w)) chỉ ra hướng tăng nhanh nhất của hàm mất mát tại điểm w. "
            "Do đó, để giảm hàm mất mát nhanh nhất, ta cập nhật trọng số theo hướng ngược lại với gradient:\n"
            "w_(t+1) = w_t - eta * nabla L(w_t)\n"
            "Trong đó eta > 0 là tốc độ học (Learning Rate), quyết định độ lớn của mỗi bước nhảy cập nhật.\n\n"
            "Các biến thể chính của Gradient Descent:\n"
            "1. Batch Gradient Descent (BGD): Tính toán gradient trên toàn bộ N mẫu dữ liệu trong tập huấn luyện cho mỗi bước cập nhật. "
            "Đảm bảo hội tụ mượt mà nhưng cực kỳ chậm và tốn tài nguyên khi dữ liệu lớn.\n"
            "2. Stochastic Gradient Descent (SGD): Cập nhật trọng số ngay sau mỗi mẫu dữ liệu duy nhất x_i. "
            "Tốc độ tính toán từng bước rất nhanh, có khả năng thoát khỏi cực tiểu địa phương nhưng quỹ đạo cập nhật dao động mạnh.\n"
            "3. Mini-batch Gradient Descent: Dung hòa BGD và SGD bằng cách chia dữ liệu thành các lô nhỏ (batch size: 32, 64, 128, 256). "
            "Đây là tiêu chuẩn thực tế trong huấn luyện Học sâu.\n"
            "4. Thuật toán tối ưu tương thích nâng cao (Adam, RMSProp): Kết hợp động lượng (Momentum) và chia tỷ lệ thích ứng với bình phương độ dốc trong quá khứ."
        ),
        "bullet_types": [
            "- Công thức lặp cơ sở: w_(t+1) = w_t - eta * nabla L(w_t).",
            "- Tốc độ học eta: Nếu quá lớn sẽ phân kỳ dao động, nếu quá nhỏ sẽ hội tụ cực chậm.",
            "- 3 biến thể: Batch GD (toàn bộ dữ liệu), SGD (từng mẫu đơn lẻ), Mini-batch GD (từng lô mẫu nhỏ).",
        ],
        "example": "Với hàm f(w) = w^2, đạo hàm f'(w) = 2w. Tại w_0 = 5, eta = 0.1: w_1 = 5 - 0.1*(2*5) = 4.0; w_2 = 4 - 0.1*(2*4) = 3.2.",
    },
    {
        "id": "ai_ml_qa_06",
        "topic": "Cơ chế Chú ý (Attention) và Kiến trúc Transformer",
        "category": "architecture_and_pipeline",
        "canonical_terms": [
            "cơ chế chú ý",
            "attention",
            "self-attention",
            "multi-head attention",
            "transformer",
        ],
        "question": "Cơ chế Self-Attention trong Transformer hoạt động như thế nào và công thức tính ra sao?",
        "intent": "mechanism_and_mathematical_foundation",
        "expected_action": "explain_self_attention_and_formula",
        "grounded_answer": (
            "Cơ chế Tự chú ý (Self-Attention) là trái tim đột phá của kiến trúc Transformer (Vaswani et al., 2017), "
            "cho phép mô hình tính toán mối tương quan ngữ nghĩa trực tiếp giữa tất cả các vị trí từ vựng trong chuỗi "
            "mà không bị hạn chế phụ thuộc tuần hoàn từng bước như RNN/LSTM, từ đó xử lý song song hóa hoàn toàn trên GPU.\n\n"
            "Công thức Scaled Dot-Product Attention:\n"
            "Cho chuỗi đầu vào X, mô hình chiếu tuyến tính thành 3 không gian vector: Query (Q), Key (K), và Value (V):\n"
            "Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V\n"
            "Giải thích các thành phần:\n"
            "- Q * K^T: Ma trận tích vô hướng đo lường mức độ tương thích/tương đồng giữa từng Query và từng Key.\n"
            "- sqrt(d_k): Hệ số tỷ lệ (scaling factor) nhằm ngăn tích vô hướng nhận giá trị quá lớn ở số chiều cao, "
            "giúp hàm softmax không bị rơi vào vùng gradient cực kỳ nhỏ (triệt tiêu gradient).\n"
            "- softmax(...): Chuẩn hóa điểm tương đồng thành phân phối xác suất trọng số chú ý (tổng mỗi hàng bằng 1).\n"
            "- Nhân với ma trận V: Tạo ra tổ hợp tuyến tính các vector Value có trọng số chú ý tương ứng.\n\n"
            "Cơ chế Chú ý đa đầu (Multi-Head Attention):\n"
            "Thay vì tính attention một lần, mô hình chiếu Q, K, V thành h không gian con độc lập (h heads), "
            "tính attention riêng rẽ rồi ghép nối (concatenate) lại và chiếu qua ma trận tuyến tính W_O. "
            "Điều này giúp mô hình đồng thời nắm bắt được các khía cạnh cú pháp, ngữ nghĩa và ngữ pháp khác nhau."
        ),
        "bullet_types": [
            "- Công thức chuẩn: Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V.",
            "- Triệt tiêu hiện tượng thắt cổ chai tuần hoàn (bottleneck) của RNN/LSTM.",
            "- Multi-Head Attention chiếu song song sang h không gian con để chụp đa góc nhìn ngữ nghĩa.",
        ],
        "example": "Trong câu 'Ngân hàng từ chối cho công ty vay vì họ thiếu tài sản bảo đảm', cơ chế chú ý giúp đại từ 'họ' liên kết mạnh nhất tới 'công ty' thay vì 'ngân hàng'.",
    },
    {
        "id": "ai_ml_qa_07",
        "topic": "Hệ thống RAG (Retrieval-Augmented Generation)",
        "category": "architecture_and_pipeline",
        "canonical_terms": [
            "rag",
            "retrieval-augmented generation",
            "cơ sở dữ liệu vector",
            "nhúng văn bản",
            "tạo sinh tăng cường truy xuất",
        ],
        "question": "RAG (Retrieval-Augmented Generation) là gì và quy trình hoạt động hoàn chỉnh gồm những bước nào?",
        "intent": "system_concept_and_pipeline",
        "expected_action": "explain_rag_concept_and_lifecycle",
        "grounded_answer": (
            "RAG (Retrieval-Augmented Generation - Tạo sinh tăng cường truy xuất) là một kiến trúc AI kết hợp giữa hệ thống truy xuất thông tin (Information Retrieval) "
            "và mô hình ngôn ngữ lớn (Large Language Model - LLM) nhằm giải quyết triệt để 2 điểm yếu chí mạng của LLM: "
            "hiện tượng ảo giác (hallucination) và tri thức tĩnh bị giới hạn tại thời điểm đóng băng dữ liệu huấn luyện.\n\n"
            "Quy trình hoạt động hoàn chỉnh của hệ thống RAG gồm 2 giai đoạn chính:\n"
            "Giai đoạn 1: Lập chỉ mục dữ liệu ngoại tuyến (Offline Ingestion & Indexing):\n"
            "1. Phân đoạn tài liệu (Chunking): Chia nhỏ tài liệu nguồn thành các đoạn văn ngắn (200 - 800 token) có ngữ cảnh trọn vẹn.\n"
            "2. Nhúng vector (Embedding): Sử dụng mô hình nhúng (Embedding Model) chuyển hóa từng đoạn văn thành vector ngữ nghĩa nhiều chiều.\n"
            "3. Lưu trữ: Nạp các vector và siêu dữ liệu (metadata) vào Cơ sở dữ liệu Vector (Vector Database / Index).\n\n"
            "Giai đoạn 2: Truy vấn và Tạo sinh trực tuyến (Online Retrieval & Generation):\n"
            "1. Mã hóa truy vấn: Câu hỏi của người dùng được chuyển đổi thành vector truy vấn q_emb.\n"
            "2. Truy xuất tài liệu (Retrieval): Tìm kiếm Top-K đoạn văn có độ tương đồng ngữ nghĩa cao nhất (Cosine Similarity) hoặc tìm kiếm lai (Hybrid Search).\n"
            "3. Xếp hạng lại (Re-ranking): Dùng Cross-Encoder lọc và tái sắp xếp Top-K đoạn văn chính xác nhất.\n"
            "4. Ghép ngữ cảnh (Prompt Augmentation): Nối Top-K đoạn văn vào câu lệnh prompt hệ thống cùng câu hỏi ban đầu.\n"
            "5. Tạo sinh câu trả lời (Generation): LLM tổng hợp câu trả lời dựa trên đúng dữ liệu bằng chứng được cung cấp, kèm trích dẫn nguồn xác thực."
        ),
        "bullet_types": [
            "- Hai giai đoạn cốt lõi: Đánh chỉ mục ngoại tuyến (Chunking -> Embedding -> Vector DB) và Phục vụ trực tuyến (Retrieval -> Augment -> Generate).",
            "- Khắc phục ảo giác (hallucination) và tri thức hạn chế của LLM.",
            "- Cho phép trích dẫn nguồn gốc (grounded citations) minh bạch cho từng luận điểm.",
        ],
        "example": "Người dùng hỏi chính sách bảo hành mới nhất 2026; RAG truy xuất PDF nội bộ và LLM trả lời chuẩn xác kèm trích dẫn trang 5 thay vì bịa đặt.",
    },
    {
        "id": "ai_ml_qa_08",
        "topic": "Thuật toán Okapi BM25 trong Truy xuất Từ vựng",
        "category": "formula_and_mechanism",
        "canonical_terms": [
            "bm25",
            "okapi bm25",
            "truy xuất từ khóa",
            "lexical retrieval",
            "tf-idf",
        ],
        "question": "Thuật toán Okapi BM25 là gì, công thức tính điểm và tại sao vẫn đóng vai trò quan trọng trong kỷ nguyên AI?",
        "intent": "concept_formula_and_importance",
        "expected_action": "explain_bm25_formula_and_utility",
        "grounded_answer": (
            "Okapi BM25 (Best Matching 25) là thuật toán truy xuất thông tin xếp hạng từ vựng (lexical ranking function) kinh điển, "
            "được xây dựng dựa trên khung xác suất thống kê nâng cấp từ mô hình TF-IDF truyền thống.\n\n"
            "Công thức tính điểm BM25 cho tài liệu D với truy vấn Q gồm các từ q_i:\n"
            "Score(D, Q) = sum_{i=1}^n IDF(q_i) * [ (f(q_i, D) * (k1 + 1)) / (f(q_i, D) + k1 * (1 - b + b * (|D| / avgdl))) ]\n"
            "Trong đó:\n"
            "- f(q_i, D): Tần suất xuất hiện của từ q_i trong tài liệu D (Term Frequency).\n"
            "- |D| và avgdl: Độ dài của tài liệu D và độ dài trung bình của toàn bộ kho tài liệu.\n"
            "- IDF(q_i) = ln((N - n(q_i) + 0.5) / (n(q_i) + 0.5) + 1): Đo lường độ hiếm và lượng thông tin của từ khóa trong kho N tài liệu.\n"
            "- k1 (thường từ 1.2 đến 2.0): Tham số kiểm soát mức độ bão hòa tần suất từ (ngăn việc lặp từ quá nhiều làm điểm số tăng vô hạn).\n"
            "- b (thường 0.75): Tham số chuẩn hóa phạt tài liệu dài do vô tình chứa nhiều từ.\n\n"
            "Tại sao BM25 vẫn không thể thay thế trong kỷ nguyên AI / Vector Retrieval?\n"
            "Mặc dù Vector Embedding rất giỏi bắt ý nghĩa trừu tượng, nhưng Vector lại rất yếu trong việc tìm kiếm chính xác các mã hiệu kỹ thuật, "
            "tên riêng, số hiệu quy định, ISBN, tên thuốc y tế hoặc công thức hóa học (Out-of-Vocabulary). "
            "BM25 cung cấp khả năng tìm kiếm từ vựng chính xác tuyệt đối (exact keyword matching), tạo nên nửa mảnh ghép hoàn hảo cho Tìm kiếm lai (Hybrid Search)."
        ),
        "bullet_types": [
            "- Khung xác suất thống kê: Kết hợp Tần suất từ (TF) có bão hòa và Nghịch đảo tần suất tài liệu (IDF).",
            "- Hai siêu tham số chuẩn: k1 (bão hòa tần suất từ) và b (chuẩn hóa độ dài văn bản).",
            "- Điểm mạnh: Bắt chính xác mã hiệu kỹ thuật, tên riêng, thuật ngữ hiếm mà mô hình nhúng vector thường bỏ sót.",
        ],
        "example": "Tìm kiếm mã lỗi 'ERR_SSL_PROTOCOL_ERROR': BM25 khớp chính xác 100% tài liệu chứa chuỗi này, trong khi vector nhúng có thể nhầm sang lỗi mạng chung.",
    },
    {
        "id": "ai_ml_qa_09",
        "topic": "Hệ thống SA-CMS (Sparse-Aware Chunk Management System)",
        "category": "architecture_and_pipeline",
        "canonical_terms": [
            "sa-cms",
            "sparse-aware chunk management system",
            "sparse chunking",
            "semantic chunking",
            "phân đoạn ngữ nghĩa",
        ],
        "question": "Hệ thống SA-CMS là gì và cơ chế phân mảnh nhận biết thưa (sparse chunking) giải quyết bài toán gì?",
        "intent": "system_concept_and_mechanism",
        "expected_action": "explain_sa_cms_and_sparse_chunking",
        "grounded_answer": (
            "Hệ thống SA-CMS (Sparse-Aware Chunk Management System - Quản lý phân đoạn nhận biết thưa) "
            "là một kiến trúc phân đoạn và quản lý văn bản tiên tiến được thiết kế riêng cho các hệ thống RAG kết hợp đa tầng (Hybrid Sparse-Dense Retrieval).\n\n"
            "Bài toán mà các kỹ thuật cắt chunk truyền thống gặp phải:\n"
            "- Fixed-size chunking (cắt cứng theo độ dài cố định) làm xé toạc các câu văn, làm đứt đoạn ngữ cảnh logic và bảng biểu.\n"
            "- Khi tài liệu chứa nhiều mã nguồn, công thức toán hoặc danh sách thực thể thưa thớt, các chunk thông thường "
            "hoặc bị phân mảnh quá nhỏ khiến mất tín hiệu ngữ nghĩa (dense signal), hoặc quá lớn làm loãng tần suất từ khóa chính xác (sparse BM25 signal).\n\n"
            "Cơ chế cốt lõi của SA-CMS:\n"
            "1. Phân đoạn thích ứng ranh giới ngữ nghĩa (Semantic Boundary-Aware Partitioning): Phát hiện tự nhiên điểm kết thúc của các đoạn văn, tiêu đề mục, bảng mã Markdown để tạo chunk nguyên khối.\n"
            "2. Nhận biết mật độ từ vựng thưa (Sparse-Density Awareness): Đo đạc phân bố thực thể đặc thù (mã định danh, công thức, ký hiệu chuyên môn) trong từng phân đoạn. "
            "Các vùng có mật độ từ thưa quan trọng được giữ liền mạch hoặc bổ sung thêm từ khóa ngữ cảnh vào phần header của chunk.\n"
            "3. Quản lý siêu dữ liệu đa cấp (Hierarchical Chunk Metadata): Mỗi chunk mang metadata chỉ rõ chunk cha (Parent Document), chunk liền trước và liền sau, "
            "cho phép hệ sinh thái RAG mở rộng ngữ cảnh (Context Window Expansion) linh hoạt khi mô hình cần tổng hợp câu trả lời sâu sắc."
        ),
        "bullet_types": [
            "- Khắc phục nhược điểm xé rách ngữ cảnh của phương pháp phân đoạn cố định kích thước.",
            "- Tối ưu hóa đồng thời cho cả truy xuất từ vựng (BM25) và truy xuất ngữ nghĩa (Dense Vector).",
            "- Lưu trữ siêu dữ liệu phân cấp giúp mở rộng cửa sổ ngữ cảnh khi tái tạo câu trả lời.",
        ],
        "example": "Tài liệu kỹ thuật chứa một hàm mã hóa 40 dòng: SA-CMS bảo toàn toàn bộ khối mã trong 1 chunk duy nhất thay vì cắt ngang giữa chừng ở dòng thứ 20.",
    },
    {
        "id": "ai_ml_qa_10",
        "topic": "Continuum Memory và Nested Learning (Học lồng ghép)",
        "category": "architecture_and_pipeline",
        "canonical_terms": [
            "continuum memory",
            "nested learning",
            "bộ nhớ liên tục",
            "catastrophic forgetting",
            "học suốt đời",
        ],
        "question": "Continuum Memory và Nested Learning là gì, giải quyết hiện tượng quên thảm họa (Catastrophic Forgetting) như thế nào?",
        "intent": "concept_and_mechanism",
        "expected_action": "explain_continuum_memory_and_nested_learning",
        "grounded_answer": (
            "Continuum Memory (Bộ nhớ liên tục) và Nested Learning (Học lồng ghép / Học lồng nhau) là các mô thức kiến trúc tiên phong "
            "trong học máy liên tục (Continual / Lifelong Learning), nhằm trang bị cho tác tử AI khả năng tích lũy tri thức mới một cách bền vững "
            "mà không làm mất đi các tri thức nền tảng đã học trước đó.\n\n"
            "Hiện tượng Quên thảm họa (Catastrophic Forgetting):\n"
            "Khi một mô hình mạng nơ-ron truyền thống được tinh chỉnh (fine-tune) trên một nhiệm vụ hoặc miền kiến thức mới B, "
            "quá trình cập nhật trọng số sẽ ghi đè và phá hủy hoàn toàn trọng số tối ưu của nhiệm vụ A cũ, khiến mô hình bị mất trí nhớ đột ngột về A.\n\n"
            "Cơ chế giải quyết của Continuum Memory & Nested Learning:\n"
            "1. Phân tầng Bộ nhớ đa cấp độ (Multi-Level Memory Architecture):\n"
            "- Bộ nhớ sự kiện ngắn hạn (Episodic Memory): Lưu trữ các dấu vết tương tác và bằng chứng thực nghiệm gần nhất dưới dạng đồ thị tri thức hoặc vector thưa.\n"
            "- Bộ nhớ ngữ nghĩa dài hạn (Semantic Memory): Tổng hợp, đúc kết các quy luật phổ quát đã được kiểm chứng vào ma trận tham số đóng băng.\n"
            "2. Học lồng ghép (Nested Learning): Xây dựng các vòng lặp học tập lồng nhau (Inner Loop và Outer Loop). "
            "Vòng ngoài (Meta-Learner) điều phối việc neo giữ các trọng số cốt lõi (Elastic Weight Consolidation), "
            "trong khi vòng trong nhanh chóng thích ứng với dữ liệu mới theo từng phân vùng riêng biệt.\n"
            "3. Tích hợp động bộ nhớ ngoài (External Memory Index): Truy vấn tri thức thông qua cơ chế RAG bộ nhớ liên tục, "
            "giúp mô hình cập nhật dữ liệu thời gian thực mà không cần huấn luyện lại toàn bộ mô hình nền."
        ),
        "bullet_types": [
            "- Ngăn chặn triệt để hiện tượng Quên thảm họa (Catastrophic Forgetting) khi nạp tri thức mới.",
            "- Kiến trúc 2 tầng: Bộ nhớ sự kiện ngắn hạn (Episodic) và Bộ nhớ ngữ nghĩa dài hạn (Semantic).",
            "- Vòng lặp học tập lồng nhau (Nested Loops) kết hợp kỹ thuật neo giữ trọng số quan trọng.",
        ],
        "example": "Hệ thống AI y tế nạp thêm phác đồ điều trị dịch bệnh mới 2026 thông qua Continuum Memory mà vẫn giữ trọn vẹn 100% độ chính xác chẩn đoán bệnh lý tim mạch cổ điển.",
    },
    {
        "id": "ai_ml_qa_11",
        "topic": "So sánh Supervised, Unsupervised và Reinforcement Learning",
        "category": "comparison",
        "canonical_terms": [
            "học có giám sát",
            "học không giám sát",
            "học tăng cường",
            "supervised learning",
            "unsupervised learning",
            "reinforcement learning",
        ],
        "question": "So sánh điểm khác biệt giữa Học có giám sát, Học không giám sát và Học tăng cường?",
        "intent": "comparative_overview",
        "expected_action": "compare_three_learning_paradigms",
        "grounded_answer": (
            "So sánh ba trụ cột chính của Học máy (Machine Learning):\n\n"
            "1. Dữ liệu đầu vào (Input Data):\n"
            "- Học có giám sát (Supervised): Cần tập dữ liệu có gán nhãn rõ ràng {(x, y)} từ chuyên gia.\n"
            "- Học không giám sát (Unsupervised): Chỉ có dữ liệu thô không nhãn {x}, không có đáp án đúng định trước.\n"
            "- Học tăng cường (Reinforcement): Không có dữ liệu tĩnh sẵn có; dữ liệu sinh ra động thông qua các bước tương tác với môi trường.\n\n"
            "2. Cơ chế phản hồi (Feedback Mechanism):\n"
            "- Học có giám sát: Phản hồi trực tiếp, tức thì qua sai lệch giữa nhãn dự đoán và nhãn thực tế L(y, f(x)).\n"
            "- Học không giám sát: Không có phản hồi đúng/sai từ bên ngoài, chỉ dựa vào tiêu chí hàm mục tiêu nội tại (khoảng cách cụm, mật độ dữ liệu).\n"
            "- Học tăng cường: Phản hồi trễ dưới dạng phần thưởng (Reward) có thể đến sau nhiều chuỗi hành động dài.\n\n"
            "3. Mục tiêu học tập (Objective):\n"
            "- Học có giám sát: Dự đoán nhãn hoặc giá trị chính xác cho các quan sát mới (Mapping f: X -> Y).\n"
            "- Học không giám sát: Khám phá cấu trúc, gom cụm mẫu hoặc nén số chiều tiềm ẩn.\n"
            "- Học tăng cường: Học chính sách hành động tối ưu để tích lũy tối đa phần thưởng dài hạn.\n\n"
            "4. Thách thức lớn nhất:\n"
            "- Học có giám sát: Chi phí thu thập và gán nhãn dữ liệu lớn thủ công cực kỳ đắt đỏ.\n"
            "- Học không giám sát: Đánh giá chất lượng cụm kết quả mang tính chủ quan và khó định lượng tuyệt đối.\n"
            "- Học tăng cường: Không gian trạng thái bùng nổ, bài toán đánh đổi giữa Khám phá (Exploration) và Khai thác (Exploitation)."
        ),
        "bullet_types": [
            "- Học có giám sát: Dữ liệu gán nhãn, phản hồi tức thời, bài toán phân lớp và hồi quy.",
            "- Học không giám sát: Dữ liệu không nhãn, không phản hồi đúng/sai, bài toán gom cụm và nén chiều.",
            "- Học tăng cường: Tác tử - Môi trường, phản hồi phần thưởng trễ, bài toán tối ưu hóa chính sách hành vi.",
        ],
        "example": "Xe tự hành: Nhận diện biển báo (Supervised), Phân nhóm điều kiện đường xá (Unsupervised), Lái xe rẽ trái/phải an toàn trên đường (Reinforcement).",
    },
    {
        "id": "ai_ml_qa_12",
        "topic": "Tìm kiếm Lai (Hybrid Search: BM25 + Vector Dense Retrieval)",
        "category": "architecture_and_pipeline",
        "canonical_terms": [
            "rag",
            "bm25",
            "cơ sở dữ liệu vector",
            "hybrid search",
            "reranking",
        ],
        "question": "Tìm kiếm Lai (Hybrid Search) kết hợp giữa BM25 và Vector Retrieval hoạt động như thế nào và mang lại lợi ích gì?",
        "intent": "concept_mechanism_and_benefit",
        "expected_action": "explain_hybrid_search_fusion",
        "grounded_answer": (
            "Tìm kiếm Lai (Hybrid Search) là kiến trúc truy xuất thông tin tối ưu hiện nay trong các hệ thống RAG cấp độ sản xuất, "
            "kết hợp sức mạnh của Truy xuất Từ vựng thưa (Sparse Lexical Retrieval - Okapi BM25) "
            "và Truy xuất Ngữ nghĩa đặc (Dense Semantic Retrieval - Vector Embeddings).\n\n"
            "Quy trình hoạt động:\n"
            "1. Phân luồng truy vấn song song (Dual-Path Retrieval):\n"
            "- Luồng 1 (BM25): So khớp chính xác các từ khóa, mã định danh kỹ thuật, danh từ riêng trong kho văn bản.\n"
            "- Luồng 2 (Vector Search): Sử dụng mô hình nhúng (Embedding Model) tính khoảng cách Cosine trong không gian vector để nắm bắt ý định ngữ nghĩa sâu sắc.\n"
            "2. Hợp nhất kết quả (Score Fusion):\n"
            "Kết hợp kết quả từ hai luồng bằng thuật toán Xếp hạng Tương hỗ (Reciprocal Rank Fusion - RRF):\n"
            "RRF_Score(d) = sum_{m in {BM25, Vector}} (1 / (k + Rank_m(d)))\n"
            "với k là hằng số làm mượt (thường k = 60). Hoặc sử dụng phép kết hợp trọng số tuyến tính: Score = alpha * BM25_Norm + (1 - alpha) * Dense_Norm.\n"
            "3. Tái xếp hạng chính xác cao (Cross-Encoder Re-ranking):\n"
            "Đưa Top 20-50 kết quả thu được sau bước hợp nhất vào một mô hình Cross-Encoder để chấm điểm cặp (Query, Chunk) ở mức độ chi tiết trước khi gửi cho LLM.\n\n"
            "Lợi ích vượt trội:\n"
            "- Triệt tiêu hiện tượng bỏ sót tài liệu khi người dùng nhập câu hỏi có cả thuật ngữ chuyên ngành chính xác lẫn từ ngữ miêu tả trừu tượng.\n"
            "- Nâng cao Recall và Precision của tầng truy xuất lên trên 90% so với việc chỉ dùng đơn lẻ BM25 hoặc Vector."
        ),
        "bullet_types": [
            "- Kết hợp song song: BM25 (chính xác từ vựng) + Vector Dense (ngữ nghĩa trừu tượng).",
            "- Hợp nhất xếp hạng qua thuật toán Reciprocal Rank Fusion (RRF) hoặc Linear Combination.",
            "- Bước tái xếp hạng (Re-ranking) bằng Cross-Encoder tinh lọc chất lượng ngữ cảnh cuối cùng.",
        ],
        "example": "Truy vấn: 'Lỗi OOM khi chạy Transformer với batch size 128': BM25 bắt đúng 'OOM' và '128', Vector bắt khái niệm tràn bộ nhớ đồ họa.",
    },
]
