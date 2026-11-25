"""
Local Offline Training & Evaluation Script for NeuralDiscourseHead.

Builds a clean, stratified Vietnamese + English academic discourse dataset from:
  1. Disjoint training corpus (src/training/vietnamese_training_corpus.py VN_TRAIN_DOCS)
  2. Multi-domain academic discourse templates + hard negative keyword-noise traps
     (pure headings, filler transitions, tangential worked examples, off-topic/redundant lines).
Trains NeuralDiscourseHead with PyTorch AdamW + joint CrossEntropy + BCEWithLogits,
verifies validation metrics + CPU latency, and saves:
  - local_runtime/checkpoints/discourse_scorer.pt
  - results/discourse_scorer_metrics.json
"""

from __future__ import annotations

import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.brain.neural_discourse_scorer import (
    DEFAULT_CHECKPOINT_PATH,
    DISCOURSE_ROLES,
    FEATURE_DIM,
    ROLE_TO_IDX,
    NeuralDiscourseHead,
    extract_clause_features,
)
from src.training.training_corpus import ALL_TRAIN_DOCS
from src.training.vietnamese_training_corpus import VN_TRAIN_DOCS

METRICS_OUTPUT_PATH = ROOT_DIR / "results" / "discourse_scorer_metrics.json"


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    f1s: List[float] = []
    for c in range(num_classes):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        if tp == 0 and fp == 0 and fn == 0:
            continue
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(sum(f1s) / max(1, len(f1s)))


def _binary_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return float((2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0)


def _binary_auroc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    pos = y_scores[y_true == 1]
    neg = y_scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 1.0
    wins = 0.0
    total = len(pos) * len(neg)
    for p in pos:
        wins += float(np.sum(p > neg)) + 0.5 * float(np.sum(p == neg))
    return wins / total


def _infer_clause_role(sentence: str) -> int:
    """Assign clean ground-truth discourse role label for curated training clauses."""
    from src.brain.neural_discourse_scorer import (
        _CAUSE_CUES,
        _COMPARE_CUES,
        _DEF_CUES,
        _EXAMPLE_CUES,
        _LIMIT_CUES,
        _MECH_CUES,
        _METRIC_UNITS_RE,
        _FORMULA_RE,
        strip_accents,
    )

    s_norm = strip_accents(sentence.lower())
    s_pad = f" {s_norm} "
    # Priority captures clear semantic intent
    if any(k in s_pad for k in _COMPARE_CUES):
        return ROLE_TO_IDX["COMPARISON_CONTRAST"]
    if any(k in s_pad for k in _LIMIT_CUES):
        return ROLE_TO_IDX["LIMITATION_CONDITION"]
    if _METRIC_UNITS_RE.search(s_norm) or _FORMULA_RE.search(s_norm) or any(k in s_pad for k in _EXAMPLE_CUES):
        return ROLE_TO_IDX["METRIC_EXAMPLE"]
    if any(k in s_pad for k in _DEF_CUES):
        return ROLE_TO_IDX["DEFINITION"]
    if any(k in s_pad for k in _CAUSE_CUES):
        return ROLE_TO_IDX["CAUSE_EFFECT"]
    if any(k in s_pad for k in _MECH_CUES):
        return ROLE_TO_IDX["MECHANISM"]
    return ROLE_TO_IDX["MECHANISM"]


def build_training_dataset(seed: int = 42) -> List[Dict[str, Any]]:
    """
    Constructs labeled discourse samples across all 6 roles plus calibrated hard negatives.
    Zero overlap with test documents.
    """
    samples: List[Dict[str, Any]] = []

    # 1. Curated multi-domain Vietnamese + English academic samples covering all 6 roles cleanly
    curated: List[Tuple[str, str, str, int, float, float, float]] = [
        # (query, clause, role_name, is_salient, pos_ratio, textrank, redundancy)
        # DEFINITION (spanning diverse pos_ratio values so position never shortcuts role)
        ("Giao thức HTTP là gì?", "HTTP là giao thức truyền tải siêu văn bản ở tầng ứng dụng theo mô hình Client-Server.", "DEFINITION", 1, 0.05, 0.92, 0.0),
        ("Mật mã Caesar là gì?", "Mã Caesar là phương pháp thay thế đơn bảng dịch chuyển mỗi chữ cái đi một khoảng cố định k.", "DEFINITION", 1, 0.35, 0.90, 0.0),
        ("Wavelet Haar là gì?", "Biến đổi Wavelet Haar là phép biến đổi trực giao phân rã tín hiệu thành thành phần trung bình và chi tiết.", "DEFINITION", 1, 0.45, 0.88, 0.0),
        ("Chất khử là gì?", "Chất khử là chất nhường electron trong phản ứng oxi hóa - khử và có số oxi hóa tăng sau phản ứng.", "DEFINITION", 1, 0.12, 0.95, 0.0),
        ("Chất oxi hóa là gì?", "Chất oxi hóa là chất nhận electron trong phản ứng và làm số oxi hóa của nguyên tố giảm xuống.", "DEFINITION", 1, 0.40, 0.93, 0.0),
        ("SA-CMS là gì?", "Hệ thống SA-CMS là bộ nhớ tham số liên tục đa thang căn chỉnh theo cấu trúc tự nhiên của văn bản.", "DEFINITION", 1, 0.50, 0.95, 0.0),
        ("Kiến trúc vi dịch vụ là gì?", "Kiến trúc vi dịch vụ là mô hình tách ứng dụng nguyên khối thành các dịch vụ độc lập giao tiếp qua API.", "DEFINITION", 1, 0.25, 0.90, 0.0),
        ("An ninh vô điều kiện là gì?", "An ninh vô điều kiện là trạng thái không thể phá mã dù đối phương sở hữu tài nguyên tính toán vô hạn.", "DEFINITION", 1, 0.55, 0.89, 0.0),
        ("One-Time Pad là gì?", "One-Time Pad là hệ mã khóa hoàn hảo có độ dài khóa ngẫu nhiên bằng đúng độ dài thông điệp và chỉ dùng một lần.", "DEFINITION", 1, 0.15, 0.91, 0.0),
        ("Conformal Risk Control là gì?", "Conformal Risk Control là khung hiệu chuẩn thống kê dùng để kiểm soát rủi ro trả lời sai dưới ngưỡng epsilon xác định.", "DEFINITION", 1, 0.42, 0.87, 0.0),
        ("Hồi quy logistic là gì?", "Hồi quy logistic là mô hình phân loại tuyến tính ánh xạ đầu ra qua hàm sigmoid để ước lượng xác suất lớp.", "DEFINITION", 1, 0.30, 0.86, 0.0),
        ("What is Sparse-LSE routing?", "Sparse-LSE routing is a tree-structured logarithmic router for hierarchical memory readout.", "DEFINITION", 1, 0.48, 0.88, 0.0),

        # MECHANISM
        ("Hệ thống SA-CMS hoạt động như thế nào?", "Kiến trúc vận hành thông qua 3 tầng bộ nhớ cập nhật tại ranh giới đoạn văn, tiểu mục và toàn văn.", "MECHANISM", 1, 0.15, 0.90, 0.05),
        ("Mã Playfair hoạt động ra sao?", "Thuật toán sử dụng ma trận ký tự 5x5 và áp dụng quy tắc thay thế cặp ký tự theo cùng hàng, cùng cột hoặc hình chữ nhật.", "MECHANISM", 1, 0.12, 0.88, 0.0),
        ("Quy trình xử lý của WaveCMS-R3 gồm gì?", "Hệ thống phân rã tài liệu bằng Wavelet Haar, định tuyến Sparse-LSE theo cây và hậu kiểm bằng Mondrian CRC.", "MECHANISM", 1, 0.10, 0.94, 0.0),
        ("Mã hóa dòng hoạt động như thế nào?", "Hệ mã hóa dòng sinh ra chuỗi khóa giả ngẫu nhiên rồi thực hiện phép XOR từng bit với bản rõ.", "MECHANISM", 1, 0.15, 0.89, 0.0),
        ("Bộ định tuyến cây hoạt động bằng cách nào?", "Bộ định tuyến duyệt từ nút gốc xuống nút lá dựa trên ngưỡng độ tương đồng log-sum-exp theo từng nhánh.", "MECHANISM", 1, 0.20, 0.85, 0.0),
        ("Làm thế nào để mở rộng khóa Autokey?", "Khóa ban đầu của Autokey được mở rộng liên tiếp bằng chính các ký tự bản rõ đi trước trong thông điệp.", "MECHANISM", 1, 0.18, 0.84, 0.0),
        ("How does Attention caching work?", "Attention caching operates by reusing key and value tensors across decoding steps through KV memory buffers.", "MECHANISM", 1, 0.12, 0.85, 0.0),
        ("Cổng API thực hiện những nhiệm vụ gì?", "Cổng điều phối trung tâm xử lý xác thực mã JWT, giới hạn băng thông truy cập và định tuyến tới dịch vụ đích.", "MECHANISM", 1, 0.15, 0.88, 0.0),
        ("Phân tích tần suất hoạt động thế nào?", "Kẻ thám mã thống kê tần suất xuất hiện các ký tự bản mã rồi đối chiếu với phân bố ngôn ngữ tự nhiên.", "MECHANISM", 1, 0.20, 0.83, 0.0),
        ("Mã Vigenere giải quyết vấn đề ra sao?", "Hệ mã Vigenere sử dụng từ khóa lặp lại theo chu kỳ để dịch chuyển đa bảng và làm phẳng phân bố ký tự.", "MECHANISM", 1, 0.15, 0.87, 0.0),
        ("Xóa tri thức int64 thực hiện thế nào?", "Hệ thống lưu trữ thống kê đủ theo định dạng số nguyên int64 giúp cộng trừ tri thức trực tiếp không sai số tích lũy.", "MECHANISM", 1, 0.18, 0.86, 0.0),
        ("Camera đáy mắt xử lý ra sao?", "Mô hình mạng nơ-ron tích chập đa tỷ lệ thu nhận ảnh đáy mắt góc rộng để phát hiện các vi tổn thương sớm.", "MECHANISM", 1, 0.15, 0.84, 0.0),

        # CAUSE_EFFECT
        ("Tại sao phản ứng phải bảo toàn electron?", "Vì số electron chất khử nhường luôn bằng tổng số electron chất oxi hóa nhận nên tổng đại số phải bảo toàn.", "CAUSE_EFFECT", 1, 0.12, 0.94, 0.0),
        ("Nhờ cơ chế nào bộ nhớ giữ được độ chính xác?", "Nhờ dùng phép cộng trừ int64 chuẩn bit nên thao tác xóa tài liệu không làm trôi các tri thức còn lại.", "CAUSE_EFFECT", 1, 0.15, 0.91, 0.0),
        ("Vì sao chọn bộ ngắt mạch tự động?", "Do bộ ngắt mạch tự động kích hoạt khi lỗi vượt ngưỡng nên giúp ngăn chặn hiệu ứng sụp đổ dây chuyền.", "CAUSE_EFFECT", 1, 0.18, 0.89, 0.0),
        ("Tại sao mỗi dịch vụ dùng database riêng?", "Việc tách cơ sở dữ liệu riêng nhằm ngăn chặn phụ thuộc chéo và cho phép mở rộng từng dịch vụ độc lập.", "CAUSE_EFFECT", 1, 0.15, 0.87, 0.0),
        ("Vì sao HTTPS an toàn hơn HTTP?", "Vì HTTPS bổ sung lớp mã hóa TLS giúp chống nghe lén và bảo vệ toàn vẹn dữ liệu trên đường truyền.", "CAUSE_EFFECT", 1, 0.15, 0.90, 0.0),
        ("Tác dụng của định tuyến Sparse-LSE là gì?", "Cơ chế Sparse-LSE giúp giảm độ phức tạp truy vấn từ tuyến tính xuống hàm logarit và giữ độ khớp cao.", "CAUSE_EFFECT", 1, 0.20, 0.92, 0.0),
        ("Lợi ích của chuẩn hóa dữ liệu trước khi train?", "Chuẩn hóa đầu vào giúp gradient hội tụ ổn định và ngăn chặn hiện tượng một đặc trưng áp đảo toàn mô hình.", "CAUSE_EFFECT", 1, 0.15, 0.86, 0.0),
        ("Nhờ đâu giảm được tỷ lệ hư hỏng nông sản?", "Nhờ kiểm soát chuỗi lạnh liên tục qua cảm biến IoT nên tỷ lệ hư hỏng sau vận chuyển giảm mạnh.", "CAUSE_EFFECT", 1, 0.20, 0.88, 0.0),
        ("Why use LayerNorm in small networks?", "LayerNorm stabilizes hidden activations and prevents gradient explosion during training.", "CAUSE_EFFECT", 1, 0.15, 0.84, 0.0),
        ("Động lực của kiểm soát rủi ro Mondrian là gì?", "Phân nhóm Mondrian giúp đảm bảo ngưỡng sai số phân tầng công bằng trên cả câu hỏi dễ lẫn câu hỏi khó.", "CAUSE_EFFECT", 1, 0.18, 0.88, 0.0),

        # COMPARISON_CONTRAST
        ("So sánh hệ mã thay thế và hoán vị?", "Mã hóa thay thế đổi ký tự này thành ký tự khác, trong khi mã hóa hoán vị chỉ sắp xếp lại vị trí các ký tự gốc.", "COMPARISON_CONTRAST", 1, 0.15, 0.94, 0.0),
        ("Phân biệt chất khử và chất oxi hóa?", "Chất khử luôn nhường electron và tăng số oxi hóa, ngược lại chất oxi hóa nhận electron và giảm số oxi hóa.", "COMPARISON_CONTRAST", 1, 0.12, 0.95, 0.0),
        ("So sánh pin thể rắn và lithium-ion?", "Pin thể rắn đạt mật độ năng lượng cao hơn 68% so với pin lithium-ion truyền thống và ít rủi ro cháy nổ hơn.", "COMPARISON_CONTRAST", 1, 0.20, 0.92, 0.0),
        ("Khác biệt giữa an ninh vô điều kiện và tính toán?", "An ninh vô điều kiện bất khả xâm phạm về lý thuyết, trái lại an ninh tính toán dựa trên giới hạn chi phí giải mã.", "COMPARISON_CONTRAST", 1, 0.18, 0.91, 0.0),
        ("Ưu điểm của mã hóa dòng so với mã khối?", "Mã hóa dòng có ưu điểm tốc độ xử lý nhanh hơn và độ trễ thấp hơn phù hợp cho truyền thông thời gian thực.", "COMPARISON_CONTRAST", 1, 0.22, 0.89, 0.0),
        ("So sánh Sparse-LSE và mean_key tại N=64?", "Sparse-LSE đạt routing hit@1 = 0.70 tại 64 tài liệu, vượt trội hoàn toàn so với phương pháp mean_key bị sụp về 0.00.", "COMPARISON_CONTRAST", 1, 0.20, 0.90, 0.0),
        ("Điểm khác nhau giữa fast mode và deep mode?", "Chế độ fast giữ độ trễ thấp hơn cho tra cứu trực tiếp, trong khi deep mode kích hoạt vòng lặp suy luận đa bước.", "COMPARISON_CONTRAST", 1, 0.22, 0.87, 0.0),
        ("So sánh HTTP và HTTPS?", "HTTPS bổ sung lớp bảo mật TLS/SSL vượt trội hơn so với HTTP thuần vốn truyền tải dữ liệu ở dạng văn bản rõ.", "COMPARISON_CONTRAST", 1, 0.15, 0.90, 0.0),
        ("Compare WaveCMS against standard RAG?", "WaveCMS cuts token overhead significantly compared to standard RAG retrieval windows.", "COMPARISON_CONTRAST", 1, 0.20, 0.86, 0.0),
        ("Phân biệt tiêu đề mục và câu giải thích?", "Tiêu đề mục chỉ nêu tên chủ đề khái quát, ngược lại câu giải thích trình bày đầy đủ cơ chế và ý nghĩa.", "COMPARISON_CONTRAST", 1, 0.15, 0.84, 0.0),
        ("Phân biệt GET và POST trong HTTP?", "Phương thức GET lấy tài nguyên qua URL, trong khi POST gửi dữ liệu xử lý ẩn trong thân thông điệp.", "COMPARISON_CONTRAST", 1, 0.18, 0.88, 0.0),
        ("So sánh cây AVL và danh sách liên kết?", "Cây cân bằng cho phép tìm kiếm theo độ phức tạp logarit nhanh hơn so với duyệt tuần tự trên danh sách liên kết.", "COMPARISON_CONTRAST", 1, 0.20, 0.89, 0.0),
        ("Khác nhau giữa khóa công khai và khóa bí mật?", "Mã hóa đối xứng dùng chung một khóa bí mật, ngược lại mã hóa bất đối xứng tách riêng khóa công khai và khóa riêng.", "COMPARISON_CONTRAST", 1, 0.18, 0.92, 0.0),
        ("Ưu điểm của SSD so với HDD truyền thống?", "Ổ cứng thể rắn SSD cho tốc độ đọc ghi cao hơn nhiều lần và chống sốc tốt hơn so với đĩa từ HDD.", "COMPARISON_CONTRAST", 1, 0.22, 0.87, 0.0),
        ("So sánh học có giám sát và không giám sát?", "Học có giám sát huấn luyện từ dữ liệu gắn nhãn sẵn, trong khi học không giám sát tự tìm cụm cấu trúc ẩn.", "COMPARISON_CONTRAST", 1, 0.15, 0.91, 0.0),
        ("Compare synchronous and asynchronous I/O?", "Asynchronous I/O avoids thread blocking compared to synchronous I/O under high concurrency.", "COMPARISON_CONTRAST", 1, 0.20, 0.86, 0.0),
        ("Khác biệt giữa TCP và UDP?", "TCP đảm bảo truyền tin tin cậy có kiểm soát lỗi, trái lại UDP ưu tiên tốc độ thấp trễ hơn độ tin cậy.", "COMPARISON_CONTRAST", 1, 0.18, 0.90, 0.0),
        ("So sánh BatchNorm và LayerNorm?", "LayerNorm chuẩn hóa độc lập trên từng mẫu theo chiều đặc trưng, khác với BatchNorm phụ thuộc kích thước batch.", "COMPARISON_CONTRAST", 1, 0.20, 0.88, 0.0),

        # Additional DEFINITION & MECHANISM & CAUSE_EFFECT for rich balance
        ("Bảng băm là gì?", "Bảng băm là cấu trúc dữ liệu ánh xạ khóa sang chỉ mục mảng thông qua hàm băm xác định.", "DEFINITION", 1, 0.06, 0.90, 0.0),
        ("Đệ quy là gì?", "Đệ quy là kỹ thuật hàm tự gọi lại chính nó trên bài toán con nhỏ hơn cho đến khi gặp điều kiện dừng.", "DEFINITION", 1, 0.08, 0.91, 0.0),
        ("TLS là gì?", "TLS là giao thức mật mã dùng để bảo mật kênh truyền thông tin trên mạng Internet.", "DEFINITION", 1, 0.06, 0.89, 0.0),
        ("Hàng đợi ưu tiên là gì?", "Hàng đợi ưu tiên là cấu trúc dữ liệu trừu tượng luôn xuất phần tử có độ ưu tiên cao nhất trước tiên.", "DEFINITION", 1, 0.08, 0.88, 0.0),
        ("Thuật toán Dijkstra làm gì?", "Dijkstra duy trì tập đỉnh đã chốt và cập nhật liên tục khoảng cách ngắn nhất qua hàng đợi ưu tiên.", "MECHANISM", 1, 0.14, 0.90, 0.0),
        ("Cơ chế phân trang bộ nhớ ảo?", "Hệ điều hành chia bộ nhớ vật lý thành khung trang và ánh xạ địa chỉ ảo qua bảng trang phần cứng.", "MECHANISM", 1, 0.16, 0.89, 0.0),
        ("Lập lịch Round Robin hoạt động ra sao?", "Bộ lập lịch cấp phát lần lượt cho mỗi tiến trình một lát thời gian cố định theo vòng tròn.", "MECHANISM", 1, 0.14, 0.88, 0.0),
        ("Bắt tay ba bước TCP diễn ra thế nào?", "Client và Server trao đổi tuần tự ba gói SYN, SYN-ACK và ACK để thiết lập kết nối tin cậy.", "MECHANISM", 1, 0.18, 0.87, 0.0),
        ("Vì sao bảng băm đạt truy vấn nhanh?", "Nhờ hàm băm tính trực tiếp vị trí bộ nhớ nên thao tác tra cứu đạt thời gian trung bình gần hằng số.", "CAUSE_EFFECT", 1, 0.15, 0.89, 0.0),
        ("Tại sao cần điều kiện dừng trong đệ quy?", "Vì nếu thiếu nhánh cơ sở hợp lệ thì chuỗi tự gọi sẽ lặp vô hạn dẫn đến tràn ngăn xếp.", "CAUSE_EFFECT", 1, 0.16, 0.91, 0.0),
        ("Tác dụng của bộ nhớ đệm L1 cache là gì?", "Đưa dữ liệu hay dùng lại gần CPU giúp rút ngắn chu kỳ chờ RAM và tăng thông lượng xử lý.", "CAUSE_EFFECT", 1, 0.18, 0.88, 0.0),
        ("Do đâu thuật toán quy hoạch động tránh tính lặp?", "Bằng cách lưu kết quả bài toán con vào bảng ghi nhớ nên chương trình không phải giải lại nhánh trùng.", "CAUSE_EFFECT", 1, 0.16, 0.90, 0.0),

        # METRIC_EXAMPLE
        ("Hiệu năng SA-CMS đạt bao nhiêu %?", "Trong thực nghiệm trên GPU 4GB, mô hình giúp giảm 87,5% chi phí token và chỉ tiêu tốn 743 MB VRAM.", "METRIC_EXAMPLE", 1, 0.30, 0.91, 0.0),
        ("Công thức mã hóa dịch vòng là gì?", "Hàm mã hóa dịch vòng được xác định theo công thức e_k(x) = (x + k) mod 26 trên vành Z26.", "METRIC_EXAMPLE", 1, 0.25, 0.90, 0.0),
        ("Độ trễ chế độ nhanh đạt bao nhiêu?", "Độ trễ trung bình khi tắt bộ lọc PMI đạt 28.19 ms trên phần cứng GTX 1650 Ti 4GB.", "METRIC_EXAMPLE", 1, 0.25, 0.89, 0.0),
        ("Sai số xóa tri thức là bao nhiêu?", "Thao tác unlearning int64 đạt sai số dưới 5.55e-17 trong thời gian thực thi 87.2 ms.", "METRIC_EXAMPLE", 1, 0.25, 0.88, 0.0),
        ("Cho ví dụ về mật mã Caesar?", "Ví dụ kinh điển của Caesar sử dụng khóa dịch chuyển k = 3 để biến đổi chữ cái A thành D.", "METRIC_EXAMPLE", 1, 0.45, 0.84, 0.0),
        ("Công thức mã hóa Affine như thế nào?", "Hàm mã hóa Affine tuân theo công thức e(x) = (a * x + b) mod 26 với điều kiện gcd(a, 26) = 1.", "METRIC_EXAMPLE", 1, 0.25, 0.89, 0.0),
        ("Hệ thống AI tầm soát đạt độ nhạy mấy %?", "Trên 8.400 bệnh nhân kiểm định lâm sàng, mô hình đạt độ nhạy 94.6% và độ đặc hiệu 91.8%.", "METRIC_EXAMPLE", 1, 0.30, 0.92, 0.0),
        ("Cổng API chịu tải bao nhiêu yêu cầu?", "Cổng điều phối API giới hạn lưu lượng ở mức 2.500 yêu cầu/giây với ngưỡng ngắt mạch 5% trong 30 giây.", "METRIC_EXAMPLE", 1, 0.28, 0.88, 0.0),
        ("Ví dụ vận hành với 64 tài liệu?", "Giả sử có 64 tài liệu với 1598 nút lá, bộ định tuyến đạt chỉ số hit@1 = 0.70 và F1 = 0.2133.", "METRIC_EXAMPLE", 1, 0.45, 0.85, 0.0),
        ("Tốc độ xử lý chữ ký ML-DSA-65?", "Thời gian ký và xác thực đạt 0.42 ms mỗi giao dịch với thông lượng tối đa 12.000 giao dịch/giây.", "METRIC_EXAMPLE", 1, 0.28, 0.87, 0.0),

        # LIMITATION_CONDITION
        ("Hạn chế của phương pháp SA-CMS là gì?", "Tuy nhiên, hạn chế của phương pháp là đòi hỏi tài liệu đầu vào phải có cấu trúc phân đoạn rõ ràng.", "LIMITATION_CONDITION", 1, 0.65, 0.86, 0.0),
        ("Điều kiện an toàn của mã Affine là gì?", "Khóa nhân a phải thỏa mãn điều kiện bắt buộc gcd(a, 26) = 1 mới đảm bảo hàm giải mã tồn tại duy nhất.", "LIMITATION_CONDITION", 1, 0.40, 0.88, 0.0),
        ("Nhược điểm của hệ mã Caesar là gì?", "Do chỉ có 25 khóa dịch chuyển khả dụng nên mật mã Caesar dễ bị phá bằng tấn công vét cạn trong tích tắc.", "LIMITATION_CONDITION", 1, 0.50, 0.89, 0.0),
        ("Thách thức khi dùng One-Time Pad là gì?", "Ràng buộc lớn nhất của One-Time Pad là chi phí phân phối và bảo mật khóa dài ngang thông điệp.", "LIMITATION_CONDITION", 1, 0.55, 0.87, 0.0),
        ("Rủi ro đối với chuẩn RSA-2048 là gì?", "Hệ mật RSA-2048 đứng trước nguy cơ bị phá vỡ nếu máy tính lượng tử chạy thuật toán Shor ở quy mô lớn.", "LIMITATION_CONDITION", 1, 0.40, 0.86, 0.0),
        ("Điều kiện nào kích hoạt chế độ từ chối?", "Hệ thống chỉ từ chối trả lời khi điểm không phù hợp vượt ngưỡng epsilon 0.10 hoặc thiếu chứng cứ trực tiếp.", "LIMITATION_CONDITION", 1, 0.50, 0.85, 0.0),
        ("Lưu ý khi nén mệnh đề dài?", "Cần lưu ý giữ nguyên các con số định lượng và điều kiện ràng buộc để tránh làm sai lệch ý gốc của tài liệu.", "LIMITATION_CONDITION", 1, 0.60, 0.83, 0.0),
        ("Nhược điểm của phương pháp Kasiski là gì?", "Kỹ thuật Kasiski gặp khó khăn khi văn bản đầu vào quá ngắn hoặc độ dài khóa tiến gần độ dài thông điệp.", "LIMITATION_CONDITION", 1, 0.55, 0.84, 0.0),
        ("What is the main limitation?", "However, the approach requires clear section boundaries and sufficient calibration samples.", "LIMITATION_CONDITION", 1, 0.70, 0.82, 0.0),
        ("Đánh đổi của chỉ mục đồ thị là gì?", "Sự đánh đổi ở đây là tốn thêm thời gian khởi tạo chỉ mục ban đầu để đổi lấy tốc độ truy vấn thời gian thực.", "LIMITATION_CONDITION", 1, 0.60, 0.85, 0.0),

        # IN-PASSAGE SUPPORTING CLAUSES (is_salient = 1 even when topic noun or TextRank central words don't repeat)
        ("Hệ thống WaveCMS hoạt động như thế nào và có hạn chế gì?", "Cơ chế định tuyến Sparse-LSE giúp giảm độ phức tạp truy vấn từ tuyến tính xuống O(log N).", "MECHANISM", 1, 0.25, 0.28, 0.0),
        ("Hệ thống WaveCMS hoạt động như thế nào và có hạn chế gì?", "Độ trễ trung bình khi tắt PMI đạt 28.19 ms trên GTX 1650 Ti.", "METRIC_EXAMPLE", 1, 0.50, 0.15, 0.0),
        ("Hệ thống SA-CMS hoạt động như thế nào?", "Trong thực nghiệm trên GPU RTX 3050 4GB, mô hình giúp giảm 87,5% chi phí token so với RAG truyền thống.", "METRIC_EXAMPLE", 1, 0.40, 0.22, 0.0),
        ("Hệ thống SA-CMS hoạt động như thế nào?", "Tuy nhiên, hạn chế của phương pháp là đòi hỏi tài liệu đầu vào phải có cấu trúc phân đoạn rõ ràng.", "LIMITATION_CONDITION", 1, 0.70, 0.35, 0.0),
        ("Mã hóa đối xứng hoạt động ra sao?", "Khóa bí mật cần được chia sẻ trước qua một kênh truyền an toàn giữa hai bên.", "LIMITATION_CONDITION", 1, 0.55, 0.20, 0.0),
        ("Giao thức HTTP là gì?", "Phiên bản HTTPS bổ sung lớp mã hóa TLS/SSL để bảo vệ tính toàn vẹn dữ liệu trên đường truyền.", "CAUSE_EFFECT", 1, 0.60, 0.32, 0.0),
        ("Hệ mã Vigenere hoạt động thế nào?", "Kỹ thuật Kasiski giúp tìm độ dài từ khóa bằng cách phân tích khoảng cách các cụm ký tự lặp lại.", "MECHANISM", 1, 0.55, 0.24, 0.0),
        ("Pin thể rắn hoạt động như thế nào?", "Mật độ năng lượng của thế hệ mới đạt 420 Wh/kg, cao hơn 68% so với công nghệ truyền thống.", "COMPARISON_CONTRAST", 1, 0.35, 0.80, 0.0),
        ("Quy trình tầm soát bệnh võng mạc bằng AI?", "Trên tập kiểm định 8.400 bệnh nhân, mô hình đạt độ nhạy 94.6% và độ đặc hiệu 91.8%.", "METRIC_EXAMPLE", 1, 0.45, 0.18, 0.0),
        ("Cổng API doanh nghiệp xử lý lỗi thế nào?", "Bộ ngắt mạch kích hoạt khi lỗi vượt ngưỡng 5% trong 30 giây nhằm ngăn sụp đổ toàn hệ thống.", "CAUSE_EFFECT", 1, 0.40, 0.65, 0.0),

        # HARD NEGATIVES (is_salient = 0): Keyword-matched headings, filler, fragments, tangential examples, redundant repeats
        ("Hệ thống SA-CMS là gì?", "# PHẦN 2: KIẾN TRÚC HỆ THỐNG SA-CMS", "DEFINITION", 0, 0.0, 0.25, 0.0),
        ("Mã hóa đối xứng là gì?", "=== Page 1 ===", "DEFINITION", 0, 0.0, 0.10, 0.0),
        ("Hệ mã dịch vòng là gì?", "1. HỆ MÃ DỊCH VÒNG:", "DEFINITION", 0, 0.0, 0.20, 0.0),
        ("Giao thức HTTP hoạt động thế nào?", "Bảng dưới đây trình bày tóm tắt nội dung về giao thức HTTP.", "MECHANISM", 0, 0.4, 0.15, 0.0),
        ("Hạn chế của hệ thống là gì?", "Như đã trình bày ở trên, phần này nói về hạn chế.", "LIMITATION_CONDITION", 0, 0.5, 0.20, 0.0),
        ("Wavelet Haar là gì?", "Xem thêm mục phụ lục về Wavelet Haar ở trang sau.", "DEFINITION", 0, 0.9, 0.12, 0.0),
        ("Chất khử là gì?", "Ví dụ cho p = 5, ta có cân bằng phương trình hóa học của chất khử theo từng bước tính.", "METRIC_EXAMPLE", 0, 0.8, 0.35, 0.0),
        ("Mã Affine là gì?", "Giả sử cho a = 5, b = 8, ta có e(3) = (5*3 + 8) mod 26 = 23.", "METRIC_EXAMPLE", 0, 0.85, 0.32, 0.0),
        ("Phân biệt chất khử và chất oxi hóa?", "Chất khử là chất nhường electron trong phản ứng oxi hóa khử.", "DEFINITION", 0, 0.30, 0.80, 0.82),
        ("SA-CMS hoạt động thế nào?", "SA-CMS là kiến trúc bộ nhớ tham số liên tục cập nhật theo cấu trúc văn bản.", "MECHANISM", 0, 0.25, 0.85, 0.78),
        ("Tóm tắt phần 2 dễ hiểu?", "Hình bên dưới minh họa sơ đồ phần 2:", "MECHANISM", 0, 0.50, 0.10, 0.0),
        ("Nguyên lý của mã Vigenere?", "Tóm tắt.", "MECHANISM", 0, 0.95, 0.05, 0.0),
        ("Hạn chế của RSA-2048?", "Nội dung chi tiet xem tai bang 3.", "LIMITATION_CONDITION", 0, 0.90, 0.10, 0.0),
        ("An ninh vô điều kiện là gì?", "Tài liệu tham khảo chương 4 về an ninh.", "DEFINITION", 0, 0.95, 0.10, 0.0),
        ("Kiến trúc vi dịch vụ là gì?", "- Tổng quan phần kiến trúc vi dịch vụ.", "DEFINITION", 0, 0.10, 0.15, 0.0),
        ("Chất khử là gì?", "Phần dưới đây liệt kê các thuật ngữ về chất khử.", "DEFINITION", 0, 0.30, 0.12, 0.0),
        ("Thuật toán Shor ảnh hưởng gì?", "Độ trễ trung bình 28.19 ms được nhắc lại tương tự đoạn đầu.", "METRIC_EXAMPLE", 0, 0.65, 0.30, 0.80),
        ("Mạng nơ-ron tích chập làm gì?", "Hình 3 minh họa sơ đồ luồng xử lý mạng nơ-ron.", "MECHANISM", 0, 0.70, 0.18, 0.0),
    ]

    for q, c, role_name, sal, pos, tr, red in curated:
        samples.append(
            {
                "query": q,
                "clause": c,
                "role_idx": ROLE_TO_IDX[role_name],
                "is_salient": sal,
                "pos_ratio": pos,
                "textrank": tr,
                "redundancy": red,
            }
        )

    # 2. Extract real grounded examples + negative pairs from VN_TRAIN_DOCS (30 training docs)
    rng = random.Random(seed)
    all_doc_sentences: List[Tuple[str, str]] = []
    for doc in VN_TRAIN_DOCS:
        ctx = doc.get("context", "")
        raw_lines = [ln.strip() for ln in ctx.splitlines() if ln.strip()]
        # Positive headings as hard negative candidates
        for ln in raw_lines:
            if ln.startswith("#"):
                for q, _ in (doc.get("qa_pairs") or [])[:2]:
                    samples.append(
                        {
                            "query": q,
                            "clause": ln,
                            "role_idx": ROLE_TO_IDX["DEFINITION"],
                            "is_salient": 0,
                            "pos_ratio": 0.0,
                            "textrank": 0.20,
                            "redundancy": 0.0,
                        }
                    )
            else:
                sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", ln) if len(s.strip()) >= 20]
                for s in sents:
                    all_doc_sentences.append((doc["doc_id"], s))

        # Match each Q&A pair to its best matching sentence in doc context (positive) + random other doc sentence (negative)
        doc_sents = [s for did, s in all_doc_sentences if did == doc["doc_id"]]
        for q, a in doc.get("qa_pairs") or []:
            q_words = set(re.findall(r"\w+", q.lower()))
            a_words = set(re.findall(r"\w+", a.lower()))
            target_words = q_words | a_words
            if doc_sents:
                best_s = max(
                    doc_sents,
                    key=lambda s: len(set(re.findall(r"\w+", s.lower())) & target_words),
                )
                r_idx = _infer_clause_role(best_s)
                samples.append(
                    {
                        "query": q,
                        "clause": best_s,
                        "role_idx": r_idx,
                        "is_salient": 1,
                        "pos_ratio": round(rng.uniform(0.05, 0.45), 2),
                        "textrank": round(rng.uniform(0.65, 0.95), 2),
                        "redundancy": 0.0,
                    }
                )
                # Also create redundant duplicate negative for same query+sentence
                if rng.random() < 0.35:
                    samples.append(
                        {
                            "query": q,
                            "clause": best_s,
                            "role_idx": r_idx,
                            "is_salient": 0,
                            "pos_ratio": 0.55,
                            "textrank": 0.60,
                            "redundancy": round(rng.uniform(0.70, 0.92), 2),
                        }
                    )

    # 3. Extract clean bilingual training pairs from disjoint ALL_TRAIN_DOCS[:18] (English academic papers)
    for doc in ALL_TRAIN_DOCS[:18]:
        ctx = doc.get("context", "")
        doc_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", ctx) if len(s.strip()) >= 20]
        if not doc_sents:
            continue
        for q, a in (doc.get("qa_pairs") or [])[:6]:
            target_words = set(re.findall(r"\w+", f"{q} {a}".lower()))
            best_s = max(doc_sents, key=lambda s: len(set(re.findall(r"\w+", s.lower())) & target_words))
            r_idx = _infer_clause_role(best_s)
            samples.append(
                {
                    "query": q,
                    "clause": best_s,
                    "role_idx": r_idx,
                    "is_salient": 1,
                    "pos_ratio": round(rng.uniform(0.05, 0.55), 2),
                    "textrank": round(rng.uniform(0.65, 0.95), 2),
                    "redundancy": 0.0,
                }
            )
            all_doc_sentences.append((doc["doc_id"], best_s))

    # Replace cross-doc content negatives with true keyword-noise hard negatives:
    # headings, meta-fillers, fragments without verbs, redundant repeats, and off-topic no-signal filler.
    hard_neg_templates: List[Tuple[str, str, int, float]] = [
        ("# CHƯƠNG {}: TỔNG QUAN VỀ {}", "DEFINITION", 0.0, 0.20),
        ("## {}.1. Kiến trúc hệ thống {}", "MECHANISM", 0.05, 0.22),
        ("{}. THÔNG SỐ KỸ THUẬT:", "METRIC_EXAMPLE", 0.0, 0.15),
        ("Bảng dưới đây trình bày tóm tắt nội dung về {}.", "MECHANISM", 0.45, 0.15),
        ("Hình bên dưới minh họa sơ đồ hoạt động của {}.", "MECHANISM", 0.50, 0.12),
        ("Như đã trình bày ở trên, phần này đề cập tới {}.", "DEFINITION", 0.35, 0.18),
        ("Xem thêm mục phụ lục và tài liệu tham khảo về {}.", "DEFINITION", 0.90, 0.10),
        ("Sau đây là danh sách các ý về {}.", "MECHANISM", 0.20, 0.15),
        ("- Khái niệm {}.", "DEFINITION", 0.15, 0.10),
        ("Ví dụ giả sử cho {} với p = 7, q = 11 ta có bước tính minh họa.", "METRIC_EXAMPLE", 0.75, 0.30),
    ]
    pool_docs = list(VN_TRAIN_DOCS) + list(ALL_TRAIN_DOCS[:18])
    for i in range(85):
        d_src = pool_docs[i % len(pool_docs)]
        if not d_src.get("qa_pairs"):
            continue
        q = rng.choice(d_src["qa_pairs"])[0]
        topic_toks = [w for w in re.findall(r"[A-Za-zÀ-ỹ0-9_-]+", q) if len(w) >= 3][-2:]
        topic = " ".join(topic_toks) if topic_toks else "hệ thống"
        tpl, r_name, pos_r, tr_r = hard_neg_templates[i % len(hard_neg_templates)]
        clause_txt = tpl.format((i % 5) + 1, topic) if tpl.count("{}") == 2 else tpl.format(topic)
        samples.append(
            {
                "query": q,
                "clause": clause_txt,
                "role_idx": ROLE_TO_IDX[r_name],
                "is_salient": 0,
                "pos_ratio": pos_r,
                "textrank": tr_r,
                "redundancy": 0.0,
            }
        )

    return samples


def train_and_evaluate(seed: int = 42, epochs: int = 200) -> Dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    samples = build_training_dataset(seed=seed)

    # Stratified split 80% / 20% across (role_idx, is_salient) buckets
    buckets: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for sp in samples:
        buckets.setdefault((sp["role_idx"], sp["is_salient"]), []).append(sp)

    train_items: List[Dict[str, Any]] = []
    val_items: List[Dict[str, Any]] = []
    rng = random.Random(seed)
    for key in sorted(buckets.keys()):
        group = list(buckets[key])
        rng.shuffle(group)
        n_val = max(1, int(round(len(group) * 0.2))) if len(group) >= 4 else 0
        val_items.extend(group[:n_val])
        train_items.extend(group[n_val:])

    def _featurize(items: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xs = np.stack(
            [
                extract_clause_features(
                    clause=it["clause"],
                    query=it["query"],
                    passage_pos_ratio=it["pos_ratio"],
                    textrank_score=it["textrank"],
                    redundancy_with_selected=it["redundancy"],
                )
                for it in items
            ],
            axis=0,
        )
        y_r = np.array([it["role_idx"] for it in items], dtype=np.int64)
        y_s = np.array([float(it["is_salient"]) for it in items], dtype=np.float32)
        return torch.from_numpy(xs), torch.from_numpy(y_r), torch.from_numpy(y_s)

    x_tr, yr_tr, ys_tr = _featurize(train_items)
    x_va, yr_va, ys_va = _featurize(val_items)

    model = NeuralDiscourseHead(in_dim=FEATURE_DIM, hidden_dim=48, num_roles=len(DISCOURSE_ROLES))
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-3, weight_decay=1e-3)
    role_loss_fn = nn.CrossEntropyLoss()
    sal_loss_fn = nn.BCEWithLogitsLoss()
    ys_tr_smooth = ys_tr * 0.90 + 0.05  # Soft targets [0.05, 0.95] prevent probability saturation
    tr_role_mask = ys_tr > 0.5
    va_role_mask = ys_va > 0.5

    best_score = -1.0
    best_state = None
    best_metrics: Dict[str, Any] = {}

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        r_logits, s_logits = model(x_tr)
        # Train role head exclusively on substantive clauses so headings/fillers never distort role weights
        role_loss = role_loss_fn(r_logits[tr_role_mask], yr_tr[tr_role_mask])
        loss = role_loss + 1.15 * sal_loss_fn(s_logits, ys_tr_smooth)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                vr_logits, vs_logits = model(x_va)
                vr_pred = torch.argmax(vr_logits[va_role_mask], dim=-1).cpu().numpy()
                vs_prob = torch.sigmoid(vs_logits).cpu().numpy()
                vs_pred = (vs_prob >= 0.5).astype(np.int64)

                yr_np = yr_va[va_role_mask].cpu().numpy()
                ys_np = ys_va.cpu().numpy().astype(np.int64)

                role_acc = float(np.mean(vr_pred == yr_np))
                role_macro_f1 = _macro_f1(yr_np, vr_pred, len(DISCOURSE_ROLES))
                sal_f1 = _binary_f1(ys_np, vs_pred)
                sal_auc = _binary_auroc(ys_np, vs_prob)
                combo = 0.65 * role_macro_f1 + 0.35 * sal_f1
                if combo >= best_score:
                    best_score = combo
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    best_metrics = {
                        "epoch": epoch,
                        "val_role_accuracy": round(role_acc, 4),
                        "val_role_macro_f1": round(role_macro_f1, 4),
                        "val_salience_f1": round(sal_f1, 4),
                        "val_salience_auc": round(sal_auc, 4),
                    }

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()

    # Benchmark CPU latency on 32-clause batch
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(50):
            model(x_va[: min(32, len(x_va))])
    avg_ms = ((time.perf_counter() - t0) / 50.0) * 1000.0

    summary = {
        **best_metrics,
        "avg_cpu_inference_ms": round(avg_ms, 3),
        "num_params": int(model.count_parameters()),
        "train_size": len(train_items),
        "val_size": len(val_items),
        "roles": list(DISCOURSE_ROLES),
        "checkpoint_path": str(DEFAULT_CHECKPOINT_PATH),
    }

    DEFAULT_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "hidden_dim": 48,
            "feature_dim": FEATURE_DIM,
            "roles": list(DISCOURSE_ROLES),
            "metrics": summary,
        },
        str(DEFAULT_CHECKPOINT_PATH),
    )

    METRICS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUTPUT_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    res = train_and_evaluate()
    print(json.dumps(res, ensure_ascii=False, indent=2))
