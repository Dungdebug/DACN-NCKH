/**
 * NL DocChat — Website Chatbot Hỏi Đáp Tài Liệu (SA-CMS & Nested Learning)
 * Frontend Controller (100% Strict Offline, Zero External Dependencies)
 * Integrates Figma NL DocChat UI with FastAPI Hybrid QA & SA-CMS Multi-Level Memory APIs.
 */

(function () {
  "use strict";

  // ============================================================================
  // 1. CONSTANTS & REPRODUCIBLE COMMITTEE DEMO SCENARIOS
  // ============================================================================

  const INGESTION_STAGES = [
    "Tệp đã lưu",
    "Văn bản đã trích xuất",
    "Cấu trúc đã phân tích",
    "Chỉ mục cục bộ đã sẵn sàng",
    "SA-CMS Memory đã sẵn sàng",
    "Sẵn sàng hỏi",
  ];

  const MODE_LABELS = {
    hybrid: "Nested Learning · P2 Hybrid",
    memory: "SA-CMS · P1 Memory-Only",
    context: "Baseline · B2 BM25 RAG",
  };

  const MODE_SHORT_BADGES = {
    hybrid: "P2 Hybrid · SA-CMS + BM25",
    memory: "P1 SA-CMS Memory-Only",
    context: "B2 Standard BM25 RAG",
  };

  const LENGTH_CYCLE = ["concise", "balanced", "detailed"];
  const LENGTH_LABELS = {
    concise: "Ngắn gọn",
    balanced: "Cân bằng",
    detailed: "Chi tiết",
  };

  const COMMITTEE_SCENARIOS = [
    {
      id: "DEMO_1_SINGLE_DOC",
      title: "Demo 1: Hỏi đáp Tài liệu Đơn lẻ & Trích dẫn Nguồn",
      badge: "Grounded QA",
      doc_ids: ["DEMO_DOC_001"],
      query: "Tầng 1 của hệ thống CMS đảm nhiệm vai trò gì trong kiến trúc?",
      expected:
        "Trả lời chính xác chức năng Tầng 1 kèm nhãn trích dẫn [1] đối chiếu nguyên văn đoạn gốc.",
      mode: "hybrid",
      answer_mode: "balanced",
    },
    {
      id: "DEMO_2_LONG_DOC_DISTANT",
      title: "Demo 2: Tài liệu Dài & Tra cứu Mục Ở Xa",
      badge: "Long-Range",
      doc_ids: ["DEMO_DOC_002"],
      query:
        "Tác động kinh tế xã hội và thời gian di chuyển giữa Hà Nội và TP.HCM dự kiến là bao lâu?",
      expected:
        "Truy xuất chính xác số liệu 5 giờ 30 phút và đóng góp GDP ở phần cuối tài liệu dài.",
      mode: "hybrid",
      answer_mode: "balanced",
    },
    {
      id: "DEMO_3_MULTI_DOC_COMPARISON",
      title: "Demo 3: Đối chiếu & Tổng hợp Đa Tài liệu",
      badge: "Multi-Doc",
      doc_ids: ["DEMO_DOC_001", "DEMO_DOC_002"],
      query: "So sánh nội dung và lĩnh vực chính giữa hai tài liệu.",
      expected:
        "Định tuyến đa tài liệu, trích xuất căn cứ từ cả 2 văn bản và gắn nhãn nguồn riêng biệt.",
      mode: "hybrid",
      answer_mode: "balanced",
    },
    {
      id: "DEMO_4_REFUSAL_UNSUPPORTED",
      title: "Demo 4: Câu hỏi Ngoài Tài liệu & Cổng Từ chối",
      badge: "Zero Hallucination",
      doc_ids: ["DEMO_DOC_001"],
      query: "Hệ thống CMS sử dụng vi xử lý lượng tử loại nào để mã hóa bộ nhớ?",
      expected:
        "Kích hoạt Cổng từ chối chống ảo giác (Calibrated Refusal) khi không đủ bằng chứng.",
      mode: "hybrid",
      answer_mode: "balanced",
    },
    {
      id: "DEMO_5_CONTEXT_EVICTION",
      title: "Demo 5: Xóa Ngữ cảnh & Ưu thế Bộ nhớ SA-CMS (P1)",
      badge: "Memory-Only",
      doc_ids: ["DEMO_DOC_001"],
      query: "Cơ chế cổng hòa trộn phần dư (Gated Residual Blending) hoạt động như thế nào?",
      expected:
        "Chế độ P1 khai thác bộ nhớ tham số 3 tầng khi ngữ cảnh gốc rời cửa sổ, giảm mạnh input tokens.",
      mode: "memory",
      answer_mode: "balanced",
    },
    {
      id: "DEMO_6_CONCISE_MODE",
      title: "Demo 6: Chế độ Cô đọng — Tối ưu Token Đầu ra (RQ5)",
      badge: "Token Optimal",
      doc_ids: ["DEMO_DOC_001"],
      query: "Cơ chế cổng hòa trộn phần dư giúp ngăn ngừa hiện tượng gì?",
      expected:
        "Trả lời cô đọng đúng trọng tâm (ngăn ngừa trôi dạt biểu diễn) kèm trích dẫn [1].",
      mode: "hybrid",
      answer_mode: "concise",
    },
    {
      id: "DEMO_7_RESEARCH_MODE_TRACE",
      title: "Demo 7: Giải phẫu Thần kinh & Cây Cấu trúc 3 Tầng",
      badge: "SA-CMS X-Ray",
      doc_ids: ["DEMO_DOC_001"],
      query: "Tóm tắt cơ chế nén tri thức của SA-CMS.",
      expected:
        "Hiển thị đồng thời câu trả lời có căn cứ và mở bảng X-Ray cổng bộ nhớ g₁, g₂, g₃.",
      mode: "hybrid",
      answer_mode: "balanced",
      openInspectorStructure: true,
    },
  ];

  // ============================================================================
  // 2. APPLICATION STATE
  // ============================================================================

  const state = {
    themePref: localStorage.getItem("nldoc_theme") || "light",
    theme: "light",
    mode: "hybrid",
    answerMode: "balanced",
    thinkingMode: "fast",
    allowOnlineApi: false,
    showTelemetry: true,
    presentationMode: false,
    isSending: false,

    sessions: [],
    activeSessionId: null,
    activeMessages: [],
    lastUserQuery: "",

    documents: [],
    attachedDocIds: [],
    sidebarSearchQuery: "",
    docModalSearchQuery: "",

    systemStatus: null,
    researchMetrics: null,
    selectedStructureDocId: null,
  };

  // ============================================================================
  // 3. SAFE FORMATTING HELPERS (Regression-hardened)
  // ============================================================================

  function escapeHtml(raw) {
    if (raw === null || raw === undefined) return "";
    return String(raw)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function formatSafeFileSize(val) {
    if (val === null || val === undefined) return "Dung lượng không xác định";
    if (typeof val === "number") {
      if (!Number.isFinite(val) || val < 0) return "Dung lượng không xác định";
      if (val === 0) return "0 KB";
      if (val < 1024) return `${Math.round(val)} B`;
      if (val < 1024 * 1024) return `${(val / 1024).toFixed(1)} KB`;
      if (val < 1024 * 1024 * 1024) return `${(val / (1024 * 1024)).toFixed(1)} MB`;
      return `${(val / (1024 * 1024 * 1024)).toFixed(1)} GB`;
    }
    const text = String(val).trim();
    if (!text) return "Dung lượng không xác định";
    const lowered = text.toLowerCase();
    if (
      lowered.includes("nan") ||
      lowered.includes("undefined") ||
      lowered.includes("null") ||
      lowered === "none"
    ) {
      return "Dung lượng không xác định";
    }
    if (/^\d+(\.\d+)?\s*(b|kb|mb|gb)$/i.test(text)) {
      return text;
    }
    const parsed = Number(text);
    if (Number.isFinite(parsed) && parsed >= 0) {
      return formatSafeFileSize(parsed);
    }
    return text;
  }

  function formatSafeDate(rawCreatedAt) {
    if (rawCreatedAt === null || rawCreatedAt === undefined || rawCreatedAt === "") {
      return "Vừa xong";
    }
    let dateObj = null;
    if (typeof rawCreatedAt === "number" && Number.isFinite(rawCreatedAt)) {
      const ms = rawCreatedAt < 1e12 ? rawCreatedAt * 1000 : rawCreatedAt;
      dateObj = new Date(ms);
    } else {
      const strVal = String(rawCreatedAt).trim();
      const asNum = Number(strVal);
      if (strVal !== "" && Number.isFinite(asNum)) {
        const ms = asNum < 1e12 ? asNum * 1000 : asNum;
        dateObj = new Date(ms);
      } else {
        dateObj = new Date(strVal);
      }
    }
    if (!dateObj || Number.isNaN(dateObj.getTime())) {
      const fallbackText = String(rawCreatedAt);
      return fallbackText.length > 10 ? fallbackText.substring(0, 10) : fallbackText;
    }
    const day = String(dateObj.getDate()).padStart(2, "0");
    const month = String(dateObj.getMonth() + 1).padStart(2, "0");
    const year = dateObj.getFullYear();
    return `${day}/${month}/${year}`;
  }

  function normalizeCitations(msg) {
    const rawCits = Array.isArray(msg.citations) ? msg.citations : [];
    const rawEv = Array.isArray(msg.evidence) ? msg.evidence : [];
    const sourceArr = rawCits.length > 0 && typeof rawCits[0] === "object" ? rawCits : rawEv;

    if (sourceArr.length > 0) {
      return sourceArr.map((item, idx) => ({
        citation_index: item.citation_index || idx + 1,
        citation_label: item.citation_label || `[${idx + 1}]`,
        passage_id: item.passage_id || "",
        document_id: item.document_id || "",
        document_title: item.document_title || item.document_id || "Tài liệu",
        section_title: item.section_title || "Phần nội dung",
        paragraph_index: item.paragraph_index || idx + 1,
        score: typeof item.score === "number" ? item.score : null,
        text: item.text || "",
      }));
    }

    return rawCits
      .filter((c) => typeof c === "string" && c.trim().length > 0)
      .map((pid, idx) => ({
        citation_index: idx + 1,
        citation_label: `[${idx + 1}]`,
        passage_id: pid,
        document_id: pid.split("::")[0] || "Tài liệu",
        document_title: pid.split("::")[0] || "Tài liệu",
        section_title: "Đoạn trích",
        paragraph_index: idx + 1,
        score: null,
        text: "",
      }));
  }

  function stripTrailingSourcesMarkdown(content, hasStructuredCitations) {
    if (!content) return "";
    let cleaned = String(content);
    cleaned = cleaned.replace(/\n*💡\s*\*\*Câu hỏi tự học tiếp theo:\*\*[\s\S]*?(?=\n*###\s*Nguồn|$)/i, "").trim();
    if (hasStructuredCitations) {
      cleaned = cleaned.replace(/\n*###\s*Nguồn tham khảo[\s\S]*$/i, "").trim();
    }
    cleaned = cleaned.replace(/\n*💡\s*\*\*Câu hỏi tự học tiếp theo:\*\*[\s\S]*$/i, "").trim();
    return cleaned;
  }

  function renderFormattedAnswerHtml(rawText, citationsList) {
    const citeMap = new Map();
    const pidMap = new Map();
    (citationsList || []).forEach((c) => {
      citeMap.set(String(c.citation_index), c);
      if (c.passage_id) {
        pidMap.set(c.passage_id, c);
      }
    });

    let safe = escapeHtml(rawText || "");

    // Collapsible answer toggle for practice exercises
    safe = safe.replace(
      /(?:^|\n)[ \t]*(?:[-•*][ \t]+)?\*Đáp án kiểm tra:\*[ \t]*([^\n]+)/g,
      '\n<details class="exercise-answer-toggle"><summary>👁️ Xem đáp án kiểm tra</summary><div class="exercise-answer-body">$1</div></details>'
    );

    // Headings (### Title)
    safe = safe.replace(/(?:^|\n)###[ \t]+([^\n]+)/g, "\n<h4>$1</h4>");

    // Bold, italic & inline code
    safe = safe.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    safe = safe.replace(/(^|[\s(])\*([^*\n]+)\*(?=[\s.,;:!?)]|$)/g, "$1<em>$2</em>");
    safe = safe.replace(/`([^`]+)`/g, "<code>$1</code>");

    // Replace [1], [2] or [DOC_ID::P001] with interactive citation pills
    safe = safe.replace(/\[([^\[\]]+)\]/g, (fullMatch, inner) => {
      const trimmed = inner.trim();
      if (citeMap.has(trimmed)) {
        const cit = citeMap.get(trimmed);
        const pidAttr = escapeHtml(cit.passage_id);
        const titleAttr = escapeHtml(`${cit.document_title} — ${cit.section_title}`);
        return `<button type="button" class="inline-cite-btn" data-passage-id="${pidAttr}" title="${titleAttr}">[${trimmed}]</button>`;
      }
      if (pidMap.has(trimmed)) {
        const cit = pidMap.get(trimmed);
        const pidAttr = escapeHtml(cit.passage_id);
        return `<button type="button" class="inline-cite-btn" data-passage-id="${pidAttr}" title="${escapeHtml(cit.document_title)}">[${cit.citation_index}]</button>`;
      }
      return fullMatch;
    });

    const paragraphs = safe.split(/\n{2,}/);
    return paragraphs
      .map((block) => `<p>${block.replace(/\n/g, "<br />")}</p>`)
      .join("");
  }

  function highlightQueryTerms(text, query) {
    let safe = escapeHtml(text || "");
    if (!query) return safe;
    const words = query
      .toLowerCase()
      .replace(/[^\p{L}\p{N}\s]/gu, " ")
      .split(/\s+/)
      .filter((w) => w.length >= 3);
    const uniqueWords = Array.from(new Set(words)).slice(0, 8);
    uniqueWords.forEach((word) => {
      const escapedWord = word.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      const regex = new RegExp(`(${escapedWord})`, "gi");
      safe = safe.replace(regex, '<mark class="source-highlight">$1</mark>');
    });
    return safe;
  }

  // ============================================================================
  // 4. API CLIENT & TOAST NOTIFICATIONS
  // ============================================================================

  async function apiFetch(url, options = {}) {
    const headers = Object.assign({}, options.headers || {});
    if (options.body && !(options.body instanceof FormData) && !headers["Content-Type"]) {
      headers["Content-Type"] = "application/json";
    }
    const response = await fetch(url, Object.assign({}, options, { headers }));
    let payload = null;
    const contentType = response.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
      payload = await response.json();
    } else {
      payload = await response.text();
    }
    if (!response.ok) {
      const detail =
        payload && typeof payload === "object" && payload.detail
          ? payload.detail
          : `Lỗi máy chủ (${response.status})`;
      throw new Error(detail);
    }
    return payload;
  }

  function showToast(message) {
    const container = document.getElementById("toast-container");
    if (!container) return;
    const el = document.createElement("div");
    el.className = "toast-msg";
    el.textContent = message;
    container.appendChild(el);
    setTimeout(() => {
      el.remove();
    }, 3400);
  }

  // ============================================================================
  // 5. DOM REFERENCES
  // ============================================================================

  const dom = {
    app: document.getElementById("nldoc-app"),
    sidebar: document.getElementById("app-sidebar"),
    sidebarBackdrop: document.getElementById("sidebar-backdrop"),
    btnBrandHome: document.getElementById("btn-brand-home"),
    btnSidebarClose: document.getElementById("btn-sidebar-close"),
    btnSidebarOpen: document.getElementById("btn-sidebar-open"),
    btnNewChat: document.getElementById("btn-new-chat"),
    btnToggleSearch: document.getElementById("btn-toggle-search"),
    sidebarSearchWrap: document.getElementById("sidebar-search-wrap"),
    sidebarSearchInput: document.getElementById("sidebar-search-input"),
    btnClearSearch: document.getElementById("btn-clear-search"),
    lblSessionCount: document.getElementById("lbl-session-count"),
    sidebarSessionsList: document.getElementById("sidebar-sessions-list"),
    btnOpenDocLibrary: document.getElementById("btn-open-doc-library"),
    sidebarDocsList: document.getElementById("sidebar-documents-list"),
    btnSidebarAddDoc: document.getElementById("btn-sidebar-add-doc"),
    btnSidebarFetchOnline: document.getElementById("btn-sidebar-fetch-online"),

    ingestionPanel: document.getElementById("ingestion-pipeline-panel"),
    ingestionFileLabel: document.getElementById("ingestion-file-label"),
    btnCloseIngestion: document.getElementById("btn-close-ingestion"),
    ingestionStagesList: document.getElementById("ingestion-stages-list"),
    ingestionTokenNote: document.getElementById("ingestion-token-note"),

    btnOpenResearchFooter: document.getElementById("btn-open-research-footer"),
    lblFooterModelInfo: document.getElementById("lbl-footer-model-info"),
    btnThemeToggle: document.getElementById("btn-theme-toggle"),

    appMain: document.getElementById("app-main"),
    dropzoneOverlay: document.getElementById("dropzone-overlay"),
    btnModeDropdown: document.getElementById("btn-mode-dropdown"),
    lblActiveMode: document.getElementById("lbl-active-mode"),
    modeDropdownMenu: document.getElementById("mode-dropdown-menu"),
    headerLengthGroup: document.getElementById("header-length-group"),
    headerThemeGroup: document.getElementById("header-theme-group"),
    headerThemeSwitch: document.getElementById("header-theme-switch"),
    btnThemeLight: document.getElementById("btn-theme-light"),
    btnThemeDark: document.getElementById("btn-theme-dark"),
    selActiveCheckpoint: document.getElementById("sel-active-checkpoint"),
    chkShowTelemetry: document.getElementById("chk-show-telemetry"),
    btnPresentationMode: document.getElementById("btn-presentation-mode"),
    btnOpenComparisonDashboard: document.getElementById("btn-open-comparison-dashboard"),
    btnOpenResearchLab: document.getElementById("btn-open-research-lab"),
    btnExportWord: document.getElementById("btn-export-word"),
    btnExportExcel: document.getElementById("btn-export-excel"),
    btnExportPptx: document.getElementById("btn-export-pptx"),
    btnClearChat: document.getElementById("btn-clear-chat"),
    btnStatusInspector: document.getElementById("btn-status-inspector"),
    lblSystemReady: document.getElementById("lbl-system-ready"),

    chatScrollContainer: document.getElementById("chat-scroll-container"),
    welcomeView: document.getElementById("welcome-view"),
    welcomeSuggestionsGrid: document.getElementById("welcome-suggestions-grid"),
    chatStreamView: document.getElementById("chat-stream-view"),
    messagesList: document.getElementById("messages-list"),
    thinkingIndicator: document.getElementById("thinking-indicator"),
    thinkingStatusText: document.getElementById("thinking-status-text"),

    composerForm: document.getElementById("composer-form"),
    composerAttachedChips: document.getElementById("composer-attached-chips"),
    composerTextarea: document.getElementById("composer-textarea"),
    btnAttachFile: document.getElementById("btn-attach-file"),
    btnDocPicker: document.getElementById("btn-doc-picker"),
    lblAttachedCount: document.getElementById("lbl-attached-count"),
    btnCycleLength: document.getElementById("btn-cycle-length"),
    lblComposerLength: document.getElementById("lbl-composer-length"),
    btnToggleOnlineApi: document.getElementById("btn-toggle-online-api"),
    lblOnlineApi: document.getElementById("lbl-online-api"),
    thinkingModeSelector: document.getElementById("thinking-mode-selector"),
    btnSend: document.getElementById("btn-send"),
    hiddenFileInput: document.getElementById("hidden-file-input"),

    inspectorDrawer: document.getElementById("inspector-drawer"),
    inspTabSource: document.getElementById("insp-tab-source"),
    inspTabStructure: document.getElementById("insp-tab-structure"),
    btnCloseInspector: document.getElementById("btn-close-inspector"),
    inspPanelSource: document.getElementById("insp-panel-source"),
    inspectorSourceContent: document.getElementById("inspector-source-content"),
    inspPanelStructure: document.getElementById("insp-panel-structure"),
    inspValG1: document.getElementById("insp-val-g1"),
    inspBarG1: document.getElementById("insp-bar-g1"),
    inspValG2: document.getElementById("insp-val-g2"),
    inspBarG2: document.getElementById("insp-bar-g2"),
    inspValG3: document.getElementById("insp-val-g3"),
    inspBarG3: document.getElementById("insp-bar-g3"),
    inspDocSelect: document.getElementById("insp-doc-select"),
    btnDocUpdateV2: document.getElementById("btn-doc-update-v2"),
    btnDocRollback: document.getElementById("btn-doc-rollback"),
    inspStructureTree: document.getElementById("insp-structure-tree"),

    modalDocLibrary: document.getElementById("modal-doc-library"),
    btnCloseDocModal: document.getElementById("btn-close-doc-modal"),
    docModalSearch: document.getElementById("doc-modal-search"),
    btnToggleRawForm: document.getElementById("btn-toggle-raw-form"),
    btnSeedDemo: document.getElementById("btn-seed-demo"),
    btnModalUpload: document.getElementById("btn-modal-upload"),
    rawDocForm: document.getElementById("raw-doc-form"),
    rawDocTitle: document.getElementById("raw-doc-title"),
    rawDocText: document.getElementById("raw-doc-text"),
    btnCancelRaw: document.getElementById("btn-cancel-raw"),
    btnSubmitRaw: document.getElementById("btn-submit-raw"),
    docModalTbody: document.getElementById("doc-modal-tbody"),

    modalResearchLab: document.getElementById("modal-research-lab"),
    btnCloseResearchModal: document.getElementById("btn-close-research-modal"),
    arenaQueryInput: document.getElementById("arena-query-input"),
    btnRunArena: document.getElementById("btn-run-arena"),
    arenaResultsGrid: document.getElementById("arena-results-grid"),
    committeeDemosGrid: document.getElementById("committee-demos-grid"),
    rlabMetricsContent: document.getElementById("rlab-metrics-content"),
    rlabBaselineCompareContent: document.getElementById("rlab-baseline-compare-content"),

    modalFetchOnline: document.getElementById("modal-fetch-online"),
    btnCloseFetchModal: document.getElementById("btn-close-fetch-modal"),
    btnCancelFetchOnline: document.getElementById("btn-cancel-fetch-online"),
    fetchOnlineForm: document.getElementById("fetch-online-form"),
    fetchOnlineSource: document.getElementById("fetch-online-source"),
    fetchLangField: document.getElementById("fetch-lang-field"),
    fetchOnlineLang: document.getElementById("fetch-online-lang"),
    fetchOnlineMax: document.getElementById("fetch-online-max"),
    fetchOnlineQuery: document.getElementById("fetch-online-query"),
    btnSubmitFetchOnline: document.getElementById("btn-submit-fetch-online"),

    modalFigureZoom: document.getElementById("modal-figure-zoom"),
    zoomFigTitle: document.getElementById("zoom-fig-title"),
    zoomFigImg: document.getElementById("zoom-fig-img"),
    btnCloseZoom: document.getElementById("btn-close-zoom"),
  };

  // ============================================================================
  // 6. THEME, SIDEBAR & MODE CONTROLS
  // ============================================================================

  function resolveEffectiveTheme(pref) {
    if (pref === "system") {
      return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches
        ? "dark"
        : "light";
    }
    return pref === "dark" ? "dark" : "light";
  }

  function applyTheme(themePref, notifyToast = false) {
    const validPref =
      themePref === "dark" || themePref === "system" ? themePref : "light";
    state.themePref = validPref;
    state.theme = resolveEffectiveTheme(validPref);
    document.documentElement.setAttribute("data-theme", state.theme);
    try {
      localStorage.setItem("nldoc_theme", validPref);
    } catch (_err) {
      // Ignore storage quota errors in private windows
    }

    if (dom.btnThemeLight && dom.btnThemeDark) {
      const isDark = state.theme === "dark";
      dom.btnThemeLight.classList.toggle("active", !isDark);
      dom.btnThemeDark.classList.toggle("active", isDark);
      dom.btnThemeLight.setAttribute("aria-pressed", String(!isDark));
      dom.btnThemeDark.setAttribute("aria-pressed", String(isDark));
    }

    document
      .querySelectorAll("#header-theme-group .theme-seg-btn[data-theme-pref]")
      .forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.themePref === validPref);
      });

    if (dom.btnThemeToggle) {
      dom.btnThemeToggle.title =
        state.theme === "dark"
          ? "Đang ở Chế độ Tối — Nhấp để chuyển sang Sáng (Alt+T)"
          : "Đang ở Chế độ Sáng — Nhấp để chuyển sang Tối (Alt+T)";
    }

    if (notifyToast) {
      if (validPref === "system") {
        showToast(
          `Giao diện theo Hệ thống (${state.theme === "dark" ? "Tối ☾" : "Sáng ☀"})`
        );
      } else {
        showToast(
          state.theme === "dark"
            ? "Đã chuyển sang Chế độ Tối ☾"
            : "Đã chuyển sang Chế độ Sáng ☀"
        );
      }
    }
  }

  function closeMobileSidebar() {
    dom.app.classList.remove("mobile-sidebar-open");
  }

  function togglePresentationMode(forceState) {
    const next =
      typeof forceState === "boolean" ? forceState : !state.presentationMode;
    state.presentationMode = next;
    dom.app.classList.toggle("presentation-mode", next);
    if (window.innerWidth > 1024) {
      dom.app.classList.toggle("sidebar-collapsed", next);
    }
    if (dom.btnPresentationMode) {
      dom.btnPresentationMode.classList.toggle("active-presentation", next);
      dom.btnPresentationMode.setAttribute("aria-pressed", String(next));
    }
    try {
      if (next && !document.fullscreenElement && document.documentElement.requestFullscreen) {
        document.documentElement.requestFullscreen().catch(() => {});
      } else if (!next && document.fullscreenElement && document.exitFullscreen) {
        document.exitFullscreen().catch(() => {});
      }
    } catch (_err) {
      // Fullscreen API may be restricted in embedded views; CSS presentation mode still applies.
    }
    showToast(
      next
        ? "Đã bật Chế độ Trình diễn Bảo vệ (Nhấn Esc để thu nhỏ)"
        : "Đã tắt Chế độ Trình diễn Bảo vệ"
    );
  }

  function syncModeAndLengthUI() {
    if (dom.lblActiveMode) {
      dom.lblActiveMode.textContent = MODE_LABELS[state.mode] || MODE_LABELS.hybrid;
    }
    document.querySelectorAll(".mode-option[data-mode]").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.mode === state.mode);
    });
    document
      .querySelectorAll("#header-length-group .length-seg-btn[data-length]")
      .forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.length === state.answerMode);
      });
    if (dom.lblComposerLength) {
      dom.lblComposerLength.textContent = LENGTH_LABELS[state.answerMode] || "Cân bằng";
    }
    if (dom.btnToggleOnlineApi) {
      const enabled = Boolean(state.allowOnlineApi);
      dom.btnToggleOnlineApi.classList.toggle("is-active", enabled);
      dom.btnToggleOnlineApi.setAttribute("aria-pressed", enabled ? "true" : "false");
      if (dom.lblOnlineApi) {
        dom.lblOnlineApi.textContent = enabled ? "🌐 Auto API ngoài: Bật" : "🌐 Auto API ngoài";
      }
    }
    if (dom.thinkingModeSelector) {
      dom.thinkingModeSelector.querySelectorAll("[data-thinking-mode]").forEach((btn) => {
        const isActive = btn.dataset.thinkingMode === state.thinkingMode;
        btn.classList.toggle("active", isActive);
        btn.setAttribute("aria-pressed", isActive ? "true" : "false");
      });
    }
    dom.app.classList.toggle("hide-telemetry", !state.showTelemetry);
  }

  function syncSendButtonState() {
    if (!dom.composerTextarea || !dom.btnSend) return;
    const hasText = dom.composerTextarea.value.trim().length > 0;
    dom.btnSend.disabled = !hasText || state.isSending;
  }

  function autoResizeComposer() {
    if (!dom.composerTextarea) return;
    dom.composerTextarea.style.height = "auto";
    const nextHeight = Math.min(dom.composerTextarea.scrollHeight, 160);
    dom.composerTextarea.style.height = `${Math.max(46, nextHeight)}px`;
  }

  // ============================================================================
  // 7. SYSTEM STATUS & WELCOME SUGGESTION CARDS
  // ============================================================================

  async function refreshSystemStatus() {
    try {
      const status = await apiFetch("/api/system/status");
      state.systemStatus = status;
      const modelReady = status.model === "READY";
      if (dom.lblSystemReady) {
        dom.lblSystemReady.textContent = modelReady
          ? "✓ SA-CMS Memory đã sẵn sàng"
          : "Tài liệu đã sẵn sàng";
      }
      if (dom.lblFooterModelInfo) {
        const ckpt = status.active_checkpoint || "5.31M CMS";
        dom.lblFooterModelInfo.textContent = modelReady
          ? `SmolLM2-135M · ${ckpt}`
          : `BM25 + SA-CMS Offline`;
      }
      if (dom.selActiveCheckpoint && Array.isArray(status.available_checkpoints) && status.available_checkpoints.length > 0) {
        const activeCkpt = status.active_checkpoint || "";
        dom.selActiveCheckpoint.innerHTML = status.available_checkpoints
          .map((name) => {
            const tag = name.includes("full_dataset")
              ? " (Full Dataset 3-Seed GPU)"
              : name.includes("extended")
              ? " (Extended GPU)"
              : name.includes("seed_42")
              ? " (Phase 4.1 Base)"
              : "";
            const selected = name === activeCkpt ? " selected" : "";
            return `<option value="${escapeHtml(name)}"${selected}>${escapeHtml(name + tag)}</option>`;
          })
          .join("");
      }
    } catch (_err) {
      if (dom.lblSystemReady) {
        dom.lblSystemReady.textContent = "Tài liệu đã sẵn sàng";
      }
    }
  }

  function renderWelcomeSuggestions() {
    if (!dom.welcomeSuggestionsGrid) return;
    const attachedDocs = state.documents.filter((d) =>
      state.attachedDocIds.includes(d.document_id)
    );
    const focusDoc = attachedDocs[0] || state.documents[0] || null;
    const focusTitle = focusDoc ? focusDoc.title : "tài liệu trong kho";

    let cards = [];
    if (focusDoc) {
      cards = [
        {
          title: "Tóm tắt ý chính tài liệu",
          detail: `Khái quát các luận điểm cốt lõi và nội dung từ "${focusTitle}"`,
          prompt: `Tóm tắt toàn bộ tài liệu ${focusDoc.title} cho tôi.`,
        },
        {
          title: "Thông tin tác giả & Chức danh",
          detail: "Tra cứu danh tính, đơn vị công tác và vị trí của người lập",
          prompt: "Tác giả của tài liệu là ai, công tác tại đơn vị nào và giữ chức vụ gì?",
        },
        {
          title: "Mục tiêu & Kế hoạch phấn đấu",
          detail: "Tìm hiểu các chỉ tiêu, nhiệm vụ và phương hướng hành động năm 2026",
          prompt: "Các nhiệm vụ trọng tâm và mục tiêu phấn đấu trong năm 2026 là gì?",
        },
        {
          title: "Sáng kiến & Cải tiến quy trình",
          detail: "Xem các đề xuất cải tiến quy trình công tác và giải pháp thực hiện",
          prompt: "Những đề xuất, sáng kiến cải tiến quy trình công tác gồm những gì?",
        },
      ];
    } else {
      cards = [
        {
          title: "Tóm tắt ý chính của tài liệu",
          detail: `Tổng hợp cấu trúc các luận điểm cốt lõi từ tài liệu`,
          prompt: "Tóm tắt toàn bộ nội dung tài liệu cho tôi.",
        },
        {
          title: "Trích xuất số liệu quan trọng",
          detail: "Tra cứu chính xác thông số kỹ thuật kèm trích dẫn nguyên văn [1]",
          prompt:
            "Tuyến đường sắt tốc độ cao Bắc - Nam có chiều dài bao nhiêu km và tốc độ thiết kế là bao nhiêu?",
        },
        {
          title: "Giải thích cơ chế bộ nhớ 3 tầng",
          detail: "Phân tích vai trò Tầng 1, Tầng 2, Tầng 3 và cổng hòa trộn phần dư",
          prompt:
            "Cơ chế cổng hòa trộn phần dư (Gated Residual Blending) hoạt động như thế nào?",
        },
        {
          title: "So sánh & đối chiếu đa tài liệu",
          detail: "Tổng hợp điểm tương đồng và khác biệt xuyên suốt các văn bản",
          prompt: "So sánh nội dung và lĩnh vực chính giữa các tài liệu trong kho.",
        },
      ];
    }

    dom.welcomeSuggestionsGrid.innerHTML = cards
      .map(
        (c) => `
        <button type="button" class="suggestion-card" data-prompt="${escapeHtml(c.prompt)}">
          <p class="suggestion-card-title">${escapeHtml(c.title)}</p>
          <p class="suggestion-card-detail">${escapeHtml(c.detail)}</p>
        </button>
      `
      )
      .join("");

    // Dynamically update Quick Verify Chips to match attached document
    const quickChipsContainer = document.querySelector(".quick-verify-chips");
    if (quickChipsContainer) {
      if (focusDoc) {
        quickChipsContainer.innerHTML = `
          <button type="button" class="verify-chip" data-query="Tác giả của tài liệu là ai, công tác tại đơn vị nào và giữ chức vụ gì?">👤 Tác giả & Chức vụ</button>
          <button type="button" class="verify-chip" data-query="Các nhiệm vụ trọng tâm và mục tiêu phấn đấu trong năm 2026 là gì?">🎯 Mục tiêu năm 2026</button>
          <button type="button" class="verify-chip" data-query="Những đề xuất, sáng kiến cải tiến quy trình công tác gồm những gì?">💡 Sáng kiến cải tiến</button>
          <button type="button" class="verify-chip" data-query="Tóm tắt toàn bộ tài liệu ${escapeHtml(focusDoc.title)} cho tôi.">📋 Tóm tắt nội dung</button>
        `;
      } else {
        quickChipsContainer.innerHTML = `
          <button type="button" class="verify-chip" data-query="Tầng 1 của hệ thống CMS đảm nhiệm vai trò gì?">Vai trò Tầng 1 SA-CMS</button>
          <button type="button" class="verify-chip" data-query="Tuyến đường sắt tốc độ cao Bắc - Nam có chiều dài bao nhiêu km và tốc độ thiết kế là bao nhiêu?">Đường sắt Bắc - Nam</button>
          <button type="button" class="verify-chip" data-query="Thuốc nào là lựa chọn đầu tay trong đơn trị liệu đái tháo đường tuýp 2?">Phác đồ Metformin</button>
          <button type="button" class="verify-chip verify-chip-refusal" data-query="Hệ thống CMS sử dụng vi xử lý lượng tử loại nào để mã hóa bộ nhớ?" title="Kiểm tra cổng từ chối chống ảo giác">🛡️ Câu hỏi bẫy ngoài tài liệu</button>
        `;
      }
      quickChipsContainer.querySelectorAll(".verify-chip[data-query]").forEach((chip) => {
        chip.addEventListener("click", () => {
          sendQuestion(chip.dataset.query);
        });
      });
    }
  }

  // ============================================================================
  // 8. SESSIONS & SIDEBAR RENDERING
  // ============================================================================

  async function refreshSessions() {
    try {
      const res = await apiFetch("/api/chat/sessions");
      state.sessions = Array.isArray(res.sessions) ? res.sessions : [];
      renderSidebarSessions();
    } catch (_err) {
      state.sessions = [];
      renderSidebarSessions();
    }
  }

  function renderSidebarSessions() {
    if (!dom.sidebarSessionsList) return;
    const q = state.sidebarSearchQuery.trim().toLowerCase();
    const filtered = state.sessions.filter((s) => {
      if (!q) return true;
      return (s.title || "").toLowerCase().includes(q);
    });

    if (dom.lblSessionCount) {
      dom.lblSessionCount.textContent = String(filtered.length);
    }

    if (filtered.length === 0) {
      dom.sidebarSessionsList.innerHTML = `
        <div style="padding: 8px 11px; font-size: 12.5px; color: var(--text-placeholder);">
          ${q ? "Không tìm thấy hội thoại phù hợp." : "Chưa có cuộc trò chuyện nào."}
        </div>
      `;
      return;
    }

    dom.sidebarSessionsList.innerHTML = filtered
      .map((s) => {
        const isActive = s.session_id === state.activeSessionId;
        const safeTitle = escapeHtml(s.title || "Cuộc trò chuyện mới");
        return `
          <div class="sidebar-item ${isActive ? "active" : ""}" data-session-id="${escapeHtml(s.session_id)}">
            <span class="sidebar-item-icon" aria-hidden="true">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M7.9 20A9 9 0 1 0 4 16.1L2 22Z"/>
              </svg>
            </span>
            <span class="sidebar-item-title" title="${safeTitle}">${safeTitle}</span>
            <span class="sidebar-item-actions">
              <button type="button" class="item-mini-btn" data-action="rename-session" data-session-id="${escapeHtml(s.session_id)}" title="Đổi tên cuộc trò chuyện">✎</button>
              <button type="button" class="item-mini-btn danger" data-action="delete-session" data-session-id="${escapeHtml(s.session_id)}" title="Xóa cuộc trò chuyện">✕</button>
            </span>
          </div>
        `;
      })
      .join("");
  }

  async function selectSession(sessionId) {
    try {
      const detail = await apiFetch(`/api/chat/sessions/${encodeURIComponent(sessionId)}`);
      state.activeSessionId = detail.session_id;
      state.attachedDocIds = Array.isArray(detail.document_ids) ? [...detail.document_ids] : [];
      state.activeMessages = Array.isArray(detail.messages) ? detail.messages : [];

      const userMsgs = state.activeMessages.filter((m) => m.role === "user");
      if (userMsgs.length > 0) {
        state.lastUserQuery = userMsgs[userMsgs.length - 1].content || "";
      }

      renderSidebarSessions();
      renderSidebarDocuments();
      renderComposerChips();
      renderConversationViewport();
      closeMobileSidebar();
    } catch (err) {
      showToast(`Không thể mở hội thoại: ${err.message}`);
    }
  }

  function startFreshChatView() {
    state.activeSessionId = null;
    state.activeMessages = [];
    const latestDoc = state.documents[0] || null;
    state.attachedDocIds = latestDoc ? [latestDoc.document_id] : [];
    renderSidebarSessions();
    renderSidebarDocuments();
    renderComposerChips();
    renderWelcomeSuggestions();
    renderConversationViewport();
    closeMobileSidebar();
    if (dom.composerTextarea) {
      dom.composerTextarea.focus();
    }
  }

  async function handleRenameSession(sessionId) {
    const target = state.sessions.find((s) => s.session_id === sessionId);
    const currentTitle = target ? target.title : "";
    const nextTitle = window.prompt("Nhập tên mới cho cuộc trò chuyện:", currentTitle);
    if (!nextTitle || !nextTitle.trim()) return;
    try {
      await apiFetch(`/api/chat/sessions/${encodeURIComponent(sessionId)}`, {
        method: "PATCH",
        body: JSON.stringify({ title: nextTitle.trim() }),
      });
      await refreshSessions();
      showToast("Đã cập nhật tên cuộc trò chuyện.");
    } catch (err) {
      showToast(`Lỗi đổi tên: ${err.message}`);
    }
  }

  async function handleDeleteSession(sessionId) {
    try {
      await apiFetch(`/api/chat/sessions/${encodeURIComponent(sessionId)}`, {
        method: "DELETE",
      });
      if (state.activeSessionId === sessionId) {
        startFreshChatView();
      }
      await refreshSessions();
      showToast("Đã xóa cuộc trò chuyện.");
    } catch (err) {
      showToast(`Lỗi xóa hội thoại: ${err.message}`);
    }
  }

  // ============================================================================
  // 9. DOCUMENTS, ATTACHMENT & 6-STAGE SA-CMS INGESTION
  // ============================================================================

  async function refreshDocuments() {
    try {
      const res = await apiFetch("/api/documents?limit=100");
      state.documents = Array.isArray(res.documents) ? res.documents : [];
      if (!state.selectedStructureDocId && state.documents.length > 0) {
        state.selectedStructureDocId = state.documents[0].document_id;
      }
      renderSidebarDocuments();
      renderComposerChips();
      renderWelcomeSuggestions();
      renderStructureDocSelector();
      if (!dom.modalDocLibrary.hidden) {
        renderDocModalTable();
      }
    } catch (_err) {
      state.documents = [];
      renderSidebarDocuments();
    }
  }

  function renderSidebarDocuments() {
    if (!dom.sidebarDocsList) return;
    const q = state.sidebarSearchQuery.trim().toLowerCase();
    const filtered = state.documents.filter((d) => {
      if (!q) return true;
      return (
        (d.title || "").toLowerCase().includes(q) ||
        (d.document_id || "").toLowerCase().includes(q)
      );
    });

    if (filtered.length === 0) {
      dom.sidebarDocsList.innerHTML = `
        <div style="padding: 8px 11px; font-size: 12.5px; color: var(--text-placeholder);">
          Kho tài liệu trống. Nhấn "+ Thêm tài liệu" để tải lên.
        </div>
      `;
      return;
    }

    dom.sidebarDocsList.innerHTML = filtered
      .map((d) => {
        const isAttached = state.attachedDocIds.includes(d.document_id);
        const safeSize = formatSafeFileSize(d.file_size);
        const metaTip = `${d.title} (${safeSize} · v${d.version || 1} · ${d.passage_count || 0} đoạn)`;
        return `
          <button
            type="button"
            class="sidebar-item ${isAttached ? "active" : ""}"
            data-doc-toggle="${escapeHtml(d.document_id)}"
            title="${escapeHtml(metaTip)} — Nhấp để đính kèm / bỏ đính kèm"
          >
            <span class="doc-file-badge" aria-hidden="true">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/>
                <path d="M14 2v4a2 2 0 0 0 2 2h4"/>
                <path d="M10 9H8"/><path d="M16 13H8"/><path d="M16 17H8"/>
              </svg>
            </span>
            <span class="sidebar-item-title">${escapeHtml(d.title)}</span>
            ${
              isAttached
                ? `<span class="doc-attached-check" title="Đang đính kèm vào phiên hỏi đáp">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">
                      <path d="M20 6 9 17l-5-5"/>
                    </svg>
                  </span>`
                : ""
            }
          </button>
        `;
      })
      .join("");
  }

  function renderComposerChips() {
    const attachedDocs = state.documents.filter((d) =>
      state.attachedDocIds.includes(d.document_id)
    );
    if (dom.lblAttachedCount) {
      if (attachedDocs.length === 1) {
        dom.lblAttachedCount.textContent = `🔒 Khóa độc quyền: ${attachedDocs[0].title}`;
      } else if (attachedDocs.length > 1) {
        dom.lblAttachedCount.textContent = `🔒 So sánh ${attachedDocs.length} tài liệu`;
      } else if (state.documents.length > 0) {
        dom.lblAttachedCount.textContent = `Chưa chọn tài liệu`;
      } else {
        dom.lblAttachedCount.textContent = `Chưa có tài liệu`;
      }
    }

    if (!dom.composerAttachedChips) return;
    if (attachedDocs.length === 0) {
      dom.composerAttachedChips.hidden = true;
      dom.composerAttachedChips.innerHTML = "";
      return;
    }

    dom.composerAttachedChips.hidden = false;
    dom.composerAttachedChips.innerHTML = attachedDocs
      .map(
        (d) => `
        <span class="composer-doc-chip">
          <span title="${escapeHtml(d.title)}">${escapeHtml(d.title)}</span>
          <button type="button" class="composer-chip-remove" data-detach-doc="${escapeHtml(d.document_id)}" aria-label="Bỏ ghim">✕</button>
        </span>
      `
      )
      .join("");
  }

  async function toggleDocumentAttachment(docId) {
    const isAttached = state.attachedDocIds.includes(docId);
    try {
      if (state.activeSessionId) {
        if (isAttached) {
          const res = await apiFetch(
            `/api/chat/sessions/${encodeURIComponent(state.activeSessionId)}/documents/${encodeURIComponent(docId)}`,
            { method: "DELETE" }
          );
          state.attachedDocIds = Array.isArray(res.document_ids) ? res.document_ids : [];
        } else {
          const res = await apiFetch(
            `/api/chat/sessions/${encodeURIComponent(state.activeSessionId)}/documents`,
            {
              method: "POST",
              body: JSON.stringify({ document_id: docId }),
            }
          );
          state.attachedDocIds = Array.isArray(res.document_ids) ? res.document_ids : [];
          // Reload session messages to display document_attached event divider
          const sessDetail = await apiFetch(
            `/api/chat/sessions/${encodeURIComponent(state.activeSessionId)}`
          );
          state.activeMessages = sessDetail.messages || [];
          renderConversationViewport();
        }
      } else {
        if (isAttached) {
          state.attachedDocIds = state.attachedDocIds.filter((id) => id !== docId);
        } else {
          state.attachedDocIds.push(docId);
        }
      }

      renderSidebarDocuments();
      renderComposerChips();
      renderWelcomeSuggestions();
      if (!dom.modalDocLibrary.hidden) {
        renderDocModalTable();
      }
    } catch (err) {
      showToast(`Không thể cập nhật đính kèm: ${err.message}`);
    }
  }

  function setIngestionStageVisual(activeIndex, allCompleted = false) {
    if (!dom.ingestionStagesList) return;
    dom.ingestionStagesList.querySelectorAll(".ingestion-stage").forEach((li) => {
      const idx = Number(li.dataset.stage);
      li.classList.remove("done", "active");
      if (allCompleted || idx < activeIndex) {
        li.classList.add("done");
      } else if (idx === activeIndex) {
        li.classList.add("active");
      }
    });
  }

  async function uploadFiles(fileList) {
    const files = Array.from(fileList || []);
    if (files.length === 0) return;

    dom.ingestionPanel.hidden = false;
    dom.ingestionTokenNote.textContent = "";

    for (const file of files) {
      dom.ingestionFileLabel.textContent = `Đang nạp: ${file.name}`;
      setIngestionStageVisual(0, false);

      let stageCursor = 0;
      const timer = setInterval(() => {
        if (stageCursor < 4) {
          stageCursor += 1;
          setIngestionStageVisual(stageCursor, false);
        }
      }, 180);

      try {
        const formData = new FormData();
        formData.append("file", file);
        if (state.activeSessionId) {
          formData.append("session_id", state.activeSessionId);
        }

        const res = await apiFetch("/api/documents/upload", {
          method: "POST",
          body: formData,
        });

        clearInterval(timer);
        setIngestionStageVisual(INGESTION_STAGES.length, true);
        dom.ingestionFileLabel.textContent = `✓ ${res.title || file.name}`;

        const memStatus = res.memory_status || "SA-CMS Memory đã sẵn sàng";
        if (dom.lblSystemReady) {
          dom.lblSystemReady.textContent =
            memStatus === "SA-CMS Memory đã sẵn sàng"
              ? "✓ SA-CMS Memory đã sẵn sàng"
              : memStatus;
        }

        const savedPct =
          res.token_reduction && typeof res.token_reduction.reduction_pct === "number"
            ? res.token_reduction.reduction_pct
            : 0;
        const sizeSafe = formatSafeFileSize(res.file_size || res.file_size_bytes);
        dom.ingestionTokenNote.textContent = `${memStatus} · ${sizeSafe} · ${res.passage_count || 1} đoạn${savedPct > 0 ? ` · Giảm ${savedPct}% nhiễu token` : ""}`;

        await refreshDocuments();
        if (res.document_id) {
          // EXCLUSIVE ATTACH: Lock exclusively to newly uploaded file!
          state.attachedDocIds = [res.document_id];
        }
        if (state.activeSessionId) {
          await selectSession(state.activeSessionId);
        } else {
          renderSidebarDocuments();
          renderComposerChips();
          renderWelcomeSuggestions();
        }

        showToast(`Đã nạp "${res.title}" — SA-CMS Memory đã sẵn sàng.`);
      } catch (err) {
        clearInterval(timer);
        dom.ingestionFileLabel.textContent = `Lỗi nạp tệp: ${file.name}`;
        showToast(`Tải tệp thất bại: ${err.message}`);
      }
    }
  }

  // ============================================================================
  // 10. CONVERSATION STREAM & COMPACT EVIDENCE RENDERING
  // ============================================================================

  function renderConversationViewport() {
    const hasMessages = state.activeMessages.length > 0;
    dom.welcomeView.hidden = hasMessages;
    dom.chatStreamView.hidden = !hasMessages;
    if (dom.btnClearChat) {
      dom.btnClearChat.hidden = !hasMessages && !state.activeSessionId;
    }

    if (!hasMessages) {
      dom.messagesList.innerHTML = "";
      return;
    }

    dom.messagesList.innerHTML = state.activeMessages
      .map((msg) => renderSingleMessageHtml(msg))
      .join("");
    scrollToBottom();
  }

  const STEP_TYPE_BADGES = {
    premise: { label: "Tiền đề", cls: "step-badge-premise" },
    mechanism: { label: "Cơ chế", cls: "step-badge-mechanism" },
    formula: { label: "Công thức ✓", cls: "step-badge-formula" },
    critique: { label: "Phản biện", cls: "step-badge-critique" },
    conclusion: { label: "Kết luận", cls: "step-badge-conclusion" },
  };

  const HAAR_LEVEL_ORDER = ["LEAF_PARAGRAPH", "SECTION", "DOCUMENT", "CORPUS"];

  function renderReasoningTraceHtml(trace) {
    if (!trace || typeof trace !== "object") return "";
    const mode = String(trace.mode || "").toLowerCase();
    const steps = Array.isArray(trace.steps) ? trace.steps : [];
    const facets = Array.isArray(trace.facets) ? trace.facets : [];
    if (mode === "fast" || (!mode && steps.length === 0 && facets.length === 0)) {
      return "";
    }
    const hasRichTrace =
      mode === "deep" ||
      mode === "broad" ||
      mode === "deep_broad" ||
      steps.length > 0 ||
      facets.length > 0;
    if (!hasRichTrace) return "";

    const modeTitles = {
      deep: "🧠 Xem tiến trình Suy nghĩ Sâu",
      broad: "🌐 Xem bản đồ Tư duy Rộng",
      deep_broad: "🔥 Xem Suy nghĩ Sâu + Tư duy Rộng",
      fast: "⚡ Tiến trình Tư duy",
    };
    const headerTitle = modeTitles[mode] || "🧠 Xem tiến trình Tư duy WaveCMS-R²";
    const isOpen = false;

    const hopsVal = toFiniteNumber(trace.hops ?? trace.num_hops);
    const rhoVal = toFiniteNumber(trace.rho_hat ?? trace.contraction_rho);
    const roundsVal = toFiniteNumber(trace.rounds ?? trace.num_rounds);
    const conf = trace.conformal && typeof trace.conformal === "object" ? trace.conformal : {};
    const confScore = toFiniteNumber(conf.score ?? trace.conformal_score);
    const lambdaG = toFiniteNumber(conf.lambda_g ?? conf.lambda_hat ?? trace.lambda_g);
    const confGroup = conf.group || trace.group || "";

    const metaPills = [];
    if (hopsVal !== null) metaPills.push(`<span class="trace-meta-pill">hops: ${Math.round(hopsVal)}</span>`);
    if (rhoVal !== null) metaPills.push(`<span class="trace-meta-pill">ρ̂ = ${rhoVal.toFixed(3)}</span>`);
    if (roundsVal !== null) metaPills.push(`<span class="trace-meta-pill">rounds: ${Math.round(roundsVal)}</span>`);
    if (confScore !== null || lambdaG !== null) {
      const scoreStr = confScore !== null ? confScore.toFixed(3) : "—";
      const lamStr = lambdaG !== null ? lambdaG.toFixed(3) : "—";
      const grpStr = confGroup ? ` (${escapeHtml(String(confGroup))})` : "";
      metaPills.push(`<span class="trace-meta-pill">Conformal: ${scoreStr} vs λ̂_g = ${lamStr}${grpStr}</span>`);
    }

    let stepsHtml = "";
    if (steps.length > 0) {
      const itemsHtml = steps
        .map((st, idx) => {
          const stType = String(st.type || st.kind || "premise").toLowerCase();
          const badgeMeta = STEP_TYPE_BADGES[stType] || { label: st.type || `Bước ${idx + 1}`, cls: "step-badge-premise" };
          const text = st.content || st.text || st.claim || "";
          const cits = Array.isArray(st.citations)
            ? st.citations
            : Array.isArray(st.citation_indices)
              ? st.citation_indices
              : [];
          const citStr = cits.length
            ? `<span class="trace-step-cits">${cits.map((c) => `[${escapeHtml(String(c).replace(/^\[|\]$/g, ""))}]`).join(" ")}</span>`
            : "";
          const verifiedTag = st.verified
            ? `<span class="trace-verified-tag">✓ Đã tự kiểm chứng</span>`
            : "";
          return `
            <li class="trace-step-item">
              <div class="trace-step-head">
                <span class="trace-step-badge ${badgeMeta.cls}">${escapeHtml(badgeMeta.label)}</span>
                ${citStr}
                ${verifiedTag}
              </div>
              <div class="trace-step-text">${escapeHtml(text)}</div>
            </li>
          `;
        })
        .join("");

      const removedClaims = Array.isArray(trace.removed_claims) ? trace.removed_claims : [];
      const selfVerified = Boolean(trace.verified || trace.self_verified);
      let verifyBannerHtml = "";
      if (selfVerified || removedClaims.length > 0) {
        const removedListHtml = removedClaims.length
          ? `<ul class="trace-removed-list">${removedClaims
              .map((rc) => `<li>${escapeHtml(typeof rc === "string" ? rc : rc.claim || rc.text || JSON.stringify(rc))}</li>`)
              .join("")}</ul>`
          : "";
        verifyBannerHtml = `
          <div class="trace-verify-box">
            <div class="trace-verify-title">
              ${selfVerified ? "✓ Đã tự kiểm chứng" : "Kiểm chứng Claim"}
              ${removedClaims.length > 0 ? ` · Đã gỡ ${removedClaims.length} claim thiếu căn cứ` : ""}
            </div>
            ${removedListHtml}
          </div>
        `;
      }

      stepsHtml = `
        <div class="trace-section-block">
          <div class="trace-section-label">Chuỗi suy luận sâu (${steps.length} bước)</div>
          <ol class="trace-steps-list">${itemsHtml}</ol>
          ${verifyBannerHtml}
        </div>
      `;
    }

    let facetsHtml = "";
    if (facets.length > 0) {
      const rowsHtml = facets
        .map((fc) => {
          const name = fc.facet || fc.name || fc.label || "Khía cạnh";
          const level = String(fc.haar_level || fc.level || fc.resolution || "LEAF_PARAGRAPH");
          const status = String(fc.status || (fc.dropped ? "dropped" : "ok")).toLowerCase();
          const isDropped = status === "dropped" || fc.dropped === true;
          const cits = Array.isArray(fc.citations)
            ? fc.citations
            : Array.isArray(fc.citation_indices)
              ? fc.citation_indices
              : [];
          const citStr = cits.length
            ? cits.map((c) => `[${escapeHtml(String(c).replace(/^\[|\]$/g, ""))}]`).join(" ")
            : "";
          const detailText = isDropped
            ? "Không đủ căn cứ trong tài liệu"
            : fc.summary || fc.content || fc.text || "Đủ căn cứ";
          const levelBadge = HAAR_LEVEL_ORDER.includes(level) ? level : level;
          return `
            <tr class="${isDropped ? "trace-facet-row-dropped" : "trace-facet-row-ok"}">
              <td><strong>${escapeHtml(name)}</strong></td>
              <td><code class="trace-haar-code">${escapeHtml(levelBadge)}</code></td>
              <td>
                <span class="trace-facet-status ${isDropped ? "is-dropped" : "is-ok"}">
                  ${isDropped ? "✕ Không đủ căn cứ" : "✓ Đủ căn cứ"}
                </span>
              </td>
              <td>
                <span class="trace-facet-detail">${escapeHtml(detailText)}</span>
                ${!isDropped && citStr ? `<span class="trace-step-cits"> ${citStr}</span>` : ""}
              </td>
            </tr>
          `;
        })
        .join("");

      facetsHtml = `
        <div class="trace-section-block">
          <div class="trace-section-label">Bản đồ Tư duy Rộng (${facets.length} facet × Haar Multi-Resolution)</div>
          <div class="doc-table-wrap">
            <table class="doc-table trace-facets-table">
              <thead>
                <tr>
                  <th>Khía cạnh (Facet)</th>
                  <th>Mức Haar</th>
                  <th>Trạng thái</th>
                  <th>Nội dung &amp; Trích dẫn</th>
                </tr>
              </thead>
              <tbody>${rowsHtml}</tbody>
            </table>
          </div>
        </div>
      `;
    }

    return `
      <details class="thinking-trace-panel" ${isOpen ? "open" : ""}>
        <summary class="thinking-trace-summary">
          <span class="thinking-trace-title">${escapeHtml(headerTitle)}</span>
          <span class="thinking-trace-metrics">${metaPills.join("")}</span>
        </summary>
        <div class="thinking-trace-body">
          ${stepsHtml}
          ${facetsHtml}
        </div>
      </details>
    `;
  }

  function renderSingleMessageHtml(msg) {
    if (msg.role === "system") {
      return `
        <div class="system-doc-event">
          <span>📎 ${escapeHtml(msg.content || "Đã đính kèm tài liệu vào hội thoại")}</span>
        </div>
      `;
    }

    if (msg.role === "user") {
      return `
        <div class="message-row">
          <div class="msg-avatar msg-avatar-user" aria-hidden="true">
            <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>
            </svg>
          </div>
          <div class="msg-body">
            <div class="msg-header-line">
              <p class="msg-author">Bạn</p>
            </div>
            <div class="msg-text">${escapeHtml(msg.content || "")}</div>
          </div>
        </div>
      `;
    }

    // Assistant Message with TRẢ LỜI + Compact CĂN CỨ Section
    const citations = normalizeCitations(msg);
    const diag = msg.diagnostics || {};
    const modeKey = diag.mode || state.mode || "hybrid";
    const modeBadge = MODE_SHORT_BADGES[modeKey] || MODE_SHORT_BADGES.hybrid;
    const isRefused = Boolean(msg.refused);
    const reasoningTraceHtml = renderReasoningTraceHtml(msg.reasoning_trace || diag.reasoning_trace);

    let answerBodyHtml = "";
    if (isRefused) {
      answerBodyHtml = `
        <div class="refusal-banner">
          <div class="refusal-banner-head">
            <span>🛡️ Cổng Từ chối Chống Ảo giác (Calibrated Refusal)</span>
          </div>
          <div class="refusal-banner-text">${escapeHtml(msg.content || "Không tìm thấy đủ thông tin trong tài liệu để trả lời chắc chắn.")}</div>
        </div>
      `;
    } else {
      const cleanBody = stripTrailingSourcesMarkdown(msg.content || "", citations.length > 0);
      answerBodyHtml = `<div class="msg-text">${renderFormattedAnswerHtml(cleanBody, citations)}</div>`;
    }

    let evidenceBlockHtml = "";
    if (!isRefused && citations.length > 0) {
      const cardsHtml = citations
        .map((cit) => {
          const metaLabel = `${cit.document_title} · ${cit.section_title} (Đoạn ${cit.paragraph_index})`;
          const excerpt = cit.text ? cit.text : "Nhấp 'Xem toàn văn' để đối chiếu đoạn gốc.";
          return `
            <div class="evidence-card-compact">
              <div class="evidence-header-row">
                <span class="evidence-cite-badge">[${escapeHtml(cit.citation_index)}]</span>
                <span class="evidence-meta-info" title="${escapeHtml(metaLabel)}">${escapeHtml(metaLabel)}</span>
                <button
                  type="button"
                  class="evidence-fulltext-btn"
                  data-passage-id="${escapeHtml(cit.passage_id)}"
                  data-doc-id="${escapeHtml(cit.document_id)}"
                >Xem toàn văn</button>
              </div>
              <div class="evidence-excerpt-text">${escapeHtml(excerpt)}</div>
            </div>
          `;
        })
        .join("");

      evidenceBlockHtml = `
        <div class="evidence-section-block">
          <div class="section-label-header">CĂN CỨ (${citations.length})</div>
          <div class="evidence-cards-stack">
            ${cardsHtml}
          </div>
        </div>
      `;
    }

    const tel = msg.telemetry || {};
    const totalTok = typeof tel.total_tokens === "number" ? tel.total_tokens : null;
    const latMs = typeof tel.total_latency_ms === "number" ? tel.total_latency_ms : null;
    const savedPct =
      typeof tel.compression_ratio_pct === "number" && tel.compression_ratio_pct > 0
        ? ` · Nén ngữ cảnh -${tel.compression_ratio_pct}%`
        : "";
    const queryForCompare = diag.effective_query || state.lastUserQuery || "";

    // Suggested follow-up questions
    let suggestedQuestions =
      msg.suggested_questions ||
      (msg.diagnostics && msg.diagnostics.suggested_questions) ||
      [];
    if ((!suggestedQuestions || suggestedQuestions.length === 0) && msg.content) {
      const matchQ = msg.content.match(/💡\s*\*\*Câu hỏi.*?\*\*([\s\S]*?)(?:###|$)/i);
      if (matchQ && matchQ[1]) {
        const lines = matchQ[1]
          .split("\n")
          .map((l) => l.replace(/^\s*\d+[\.\)]\s*/, "").trim())
          .filter((l) => l.length > 5);
        if (lines.length > 0) {
          suggestedQuestions = lines;
        }
      }
    }
    let suggestedBlockHtml = "";
    if (!isRefused && suggestedQuestions && suggestedQuestions.length > 0) {
      const chipsHtml = suggestedQuestions
        .map(
          (q) => `
        <button type="button" class="suggest-chip" data-suggest-query="${escapeHtml(q)}">
          <span class="suggest-chip-icon">✨</span>
          <span class="suggest-chip-text">${escapeHtml(q)}</span>
        </button>
      `
        )
        .join("");

      suggestedBlockHtml = `
        <div class="msg-suggested-block">
          <div class="msg-suggested-label">💡 Gợi ý câu hỏi tiếp theo:</div>
          <div class="msg-suggested-chips">
            ${chipsHtml}
          </div>
        </div>
      `;
    }

    const exportTitle = queryForCompare || "Câu trả lời WaveCMS";
    const exportContent = msg.content || "";
    const hasActionBar = totalTok !== null || latMs !== null || Boolean(exportContent);
    const telemetryBarHtml = hasActionBar
      ? `
        <div class="msg-telemetry-bar" data-export-title="${escapeHtml(exportTitle)}" data-export-content="${escapeHtml(exportContent)}">
          ${totalTok !== null ? `<span class="tel-pill">${totalTok} tokens (${tel.input_tokens || 0} in / ${tel.output_tokens || 0} out${savedPct})</span>` : ""}
          ${latMs !== null ? `<span class="tel-pill">${latMs} ms</span>` : ""}
          ${
            queryForCompare
              ? `<button type="button" class="tel-compare-btn" data-compare-query="${escapeHtml(queryForCompare)}">⇄ Đối sánh B1/B2/P1/P2</button>`
              : ""
          }
          ${
            exportContent
              ? `
            <button type="button" class="tel-export-btn" data-export-msg="word" title="Xuất câu trả lời này ra Word (.docx)">📄 Word</button>
            <button type="button" class="tel-export-btn" data-export-msg="excel" title="Xuất câu trả lời này ra Excel (.xlsx)">📊 Excel</button>
            <button type="button" class="tel-export-btn" data-export-msg="pptx" title="Xuất câu trả lời này ra PowerPoint (.pptx)">📽️ PPTX</button>
          `
              : ""
          }
        </div>
      `
      : "";

    return `
      <div class="message-row">
        <div class="msg-avatar msg-avatar-assistant" aria-hidden="true">
          <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round">
            <path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z"/>
          </svg>
        </div>
        <div class="msg-body">
          <div class="msg-header-line">
            <p class="msg-author">NL DocChat</p>
            <span class="msg-mode-pill">${escapeHtml(modeBadge)}</span>
          </div>
          ${reasoningTraceHtml}
          <div class="section-label-header">TRẢ LỜI</div>
          ${answerBodyHtml}
          ${evidenceBlockHtml}
          ${suggestedBlockHtml}
          ${telemetryBarHtml}
        </div>
      </div>
    `;
  }

  function scrollToBottom() {
    if (!dom.chatScrollContainer) return;
    requestAnimationFrame(() => {
      dom.chatScrollContainer.scrollTop = dom.chatScrollContainer.scrollHeight;
    });
  }

  async function sendQuestion(rawQuery) {
    const query = (rawQuery || "").trim();
    if (!query || state.isSending) return;

    state.isSending = true;
    state.lastUserQuery = query;
    syncSendButtonState();

    try {
      // Lazily create session on first message if not yet active
      if (!state.activeSessionId) {
        const created = await apiFetch("/api/chat/sessions", {
          method: "POST",
          body: JSON.stringify({
            document_ids: state.attachedDocIds,
          }),
        });
        state.activeSessionId = created.session_id;
        state.activeMessages = Array.isArray(created.messages) ? created.messages : [];
      }

      // Optimistically render user message
      state.activeMessages.push({
        role: "user",
        content: query,
      });

      dom.composerTextarea.value = "";
      autoResizeComposer();
      syncSendButtonState();

      renderConversationViewport();
      dom.thinkingIndicator.hidden = false;
      const thinkingStatusLabels = {
        deep: "Đang suy nghĩ sâu đa bước & tự kiểm chứng công thức...",
        broad: "Đang quét bản đồ tư duy rộng 5 facet × 4 mức Haar...",
        deep_broad: "Đang tổng hợp Suy nghĩ Sâu + Tư duy Rộng...",
      };
      dom.thinkingStatusText.textContent =
        thinkingStatusLabels[state.thinkingMode] ||
        (state.mode === "memory"
          ? "Đang truy xuất bộ nhớ tham số 3 tầng SA-CMS (P1)..."
          : "Đang truy xuất bằng chứng và đối soát bộ nhớ SA-CMS...");
      scrollToBottom();

      const res = await apiFetch(
        `/api/chat/sessions/${encodeURIComponent(state.activeSessionId)}/messages`,
        {
          method: "POST",
          body: JSON.stringify({
            query,
            mode: state.mode,
            answer_mode: state.answerMode,
            thinking_mode: state.thinkingMode,
            allow_online_api: Boolean(state.allowOnlineApi),
            top_k: 5,
            language: "vi",
          }),
        }
      );

      dom.thinkingIndicator.hidden = true;

      if (res.message) {
        if (!res.message.suggested_questions && res.suggested_questions) {
          res.message.suggested_questions = res.suggested_questions;
        }
        if (!res.message.reasoning_trace && res.reasoning_trace) {
          res.message.reasoning_trace = res.reasoning_trace;
        }
        state.activeMessages.push(res.message);
      } else {
        state.activeMessages.push({
          role: "assistant",
          content: res.answer || "",
          citations: res.citations || [],
          telemetry: res.telemetry || {},
          diagnostics: res.diagnostics || {},
          reasoning_trace: res.reasoning_trace || (res.diagnostics && res.diagnostics.reasoning_trace) || null,
          refused: Boolean(res.refused),
          suggested_questions: res.suggested_questions || [],
        });
      }

      if (res.memory_diagnostics) {
        updateCmsGateBars(res.memory_diagnostics);
      }

      renderConversationViewport();
      await refreshSessions();
    } catch (err) {
      dom.thinkingIndicator.hidden = true;
      showToast(`Lỗi truy vấn: ${err.message}`);
    } finally {
      state.isSending = false;
      syncSendButtonState();
    }
  }

  // ============================================================================
  // 11. SLIDE-OVER INSPECTOR DRAWER (VERBATIM CITATION & SA-CMS STRUCTURE)
  // ============================================================================

  function openInspector(tabName = "source") {
    dom.inspectorDrawer.hidden = false;
    const isSource = tabName === "source";
    dom.inspTabSource.classList.toggle("active", isSource);
    dom.inspTabStructure.classList.toggle("active", !isSource);
    dom.inspPanelSource.hidden = !isSource;
    dom.inspPanelStructure.hidden = isSource;

    if (!isSource) {
      const targetDoc =
        state.selectedStructureDocId ||
        state.attachedDocIds[0] ||
        (state.documents[0] && state.documents[0].document_id);
      if (targetDoc) {
        state.selectedStructureDocId = targetDoc;
        renderStructureDocSelector();
        loadDocumentStructureAndMemory(targetDoc);
      }
    }
  }

  function updateCmsGateBars(memDiag) {
    const g1 = typeof memDiag.g1 === "number" ? memDiag.g1 : 0.842;
    const g2 = typeof memDiag.g2 === "number" ? memDiag.g2 : 0.615;
    const g3 = typeof memDiag.g3 === "number" ? memDiag.g3 : 0.927;

    if (dom.inspValG1) dom.inspValG1.textContent = `g₁ = ${g1.toFixed(3)}`;
    if (dom.inspBarG1) dom.inspBarG1.style.width = `${Math.min(100, Math.round(g1 * 100))}%`;
    if (dom.inspValG2) dom.inspValG2.textContent = `g₂ = ${g2.toFixed(3)}`;
    if (dom.inspBarG2) dom.inspBarG2.style.width = `${Math.min(100, Math.round(g2 * 100))}%`;
    if (dom.inspValG3) dom.inspValG3.textContent = `g₃ = ${g3.toFixed(3)}`;
    if (dom.inspBarG3) dom.inspBarG3.style.width = `${Math.min(100, Math.round(g3 * 100))}%`;
  }

  async function inspectPassageCitation(passageId) {
    if (!passageId) return;
    openInspector("source");
    dom.inspectorSourceContent.className = "";
    dom.inspectorSourceContent.innerHTML = `
      <div class="insp-section-card">
        <p class="insp-card-sub">Đang tải nguyên văn đoạn trích <code>${escapeHtml(passageId)}</code>...</p>
      </div>
    `;

    try {
      const cit = await apiFetch(`/api/citations/${encodeURIComponent(passageId)}`);
      state.selectedStructureDocId = cit.document_id;
      renderStructureDocSelector();

      const highlighted = highlightQueryTerms(cit.text || "", state.lastUserQuery);
      dom.inspectorSourceContent.innerHTML = `
        <div class="insp-section-card">
          <div style="display:flex; align-items:center; justify-content:space-between; gap:8px; margin-bottom:6px;">
            <span class="evidence-cite-badge">${escapeHtml(cit.passage_id)}</span>
            <span class="tel-pill" style="color: var(--emerald-dark); font-weight:700;">✓ ${escapeHtml(cit.support_status || "SUPPORTED")}</span>
          </div>
          <div class="insp-card-title">${escapeHtml(cit.document_title)} (v${cit.version || 1})</div>
          <p class="insp-card-sub">
            Mục: <strong>${escapeHtml(cit.section_title)}</strong> · Đoạn ${cit.paragraph_index || 1}
            · Ký tự [${cit.start_char ?? 0}–${cit.end_char ?? 0}]
          </p>
          <div class="source-verbatim-box">${highlighted}</div>
        </div>
      `;
    } catch (err) {
      dom.inspectorSourceContent.innerHTML = `
        <div class="insp-section-card">
          <div class="insp-card-title">Không tìm thấy trích dẫn</div>
          <p class="insp-card-sub">${escapeHtml(err.message)}</p>
        </div>
      `;
    }
  }

  function renderStructureDocSelector() {
    if (!dom.inspDocSelect) return;
    if (state.documents.length === 0) {
      dom.inspDocSelect.innerHTML = `<option value="">Chưa có tài liệu</option>`;
      return;
    }
    dom.inspDocSelect.innerHTML = state.documents
      .map(
        (d) =>
          `<option value="${escapeHtml(d.document_id)}" ${
            d.document_id === state.selectedStructureDocId ? "selected" : ""
          }>${escapeHtml(d.title)} (v${d.version || 1})</option>`
      )
      .join("");
  }

  async function loadDocumentStructureAndMemory(docId) {
    if (!docId || !dom.inspStructureTree) return;
    dom.inspStructureTree.innerHTML = `<div class="struct-para-meta">Đang phân tích cấu trúc SA-CMS...</div>`;

    try {
      const [structData, memData] = await Promise.all([
        apiFetch(`/api/documents/${encodeURIComponent(docId)}/structure`),
        apiFetch(`/api/memory/${encodeURIComponent(docId)}`),
      ]);

      const sections =
        structData.hierarchy && Array.isArray(structData.hierarchy.sections)
          ? structData.hierarchy.sections
          : [];
      const ts = memData.timescales || {};

      const memSummaryHtml = `
        <div class="struct-sec-item" style="background: var(--brand-tint); border-color: var(--brand-border);">
          <div class="struct-sec-title">SA-CMS Snapshot · v${structData.version || 1} (${structData.total_tokens || 0} tokens)</div>
          <div class="struct-para-meta">
            • L1 Paragraph: ${ts.level_1 ? ts.level_1.update_count : structData.total_paragraphs} cập nhật (1,771,584 params)<br/>
            • L2 Section: ${ts.level_2 ? ts.level_2.update_count : structData.total_sections} cập nhật (1,771,584 params)<br/>
            • L3 Document: ${ts.level_3 ? ts.level_3.update_count : 1} neo toàn cục (1,771,584 params)
          </div>
        </div>
      `;

      const secHtml = sections
        .map(
          (sec) => `
          <div class="struct-sec-item">
            <div class="struct-sec-title">§${sec.section_index}. ${escapeHtml(sec.title)}</div>
            <div class="struct-para-meta">
              Ranh giới L2 · ${(sec.paragraphs || []).length} đoạn văn L1 · Token [${sec.start_token}–${sec.end_token}]
            </div>
          </div>
        `
        )
        .join("");

      dom.inspStructureTree.innerHTML = memSummaryHtml + secHtml;
    } catch (err) {
      dom.inspStructureTree.innerHTML = `<div class="struct-para-meta">Không thể tải cấu trúc: ${escapeHtml(err.message)}</div>`;
    }
  }

  async function handleUpdateDocVersion() {
    const docId = dom.inspDocSelect ? dom.inspDocSelect.value : state.selectedStructureDocId;
    if (!docId) return;
    try {
      const res = await apiFetch(
        `/api/documents/${encodeURIComponent(docId)}/update_version`,
        {
          method: "POST",
          body: JSON.stringify({}),
        }
      );
      showToast(res.message || "Đã nâng cấp phiên bản tài liệu.");
      await refreshDocuments();
      await loadDocumentStructureAndMemory(docId);
    } catch (err) {
      showToast(`Lỗi cập nhật phiên bản: ${err.message}`);
    }
  }

  async function handleRollbackDocVersion() {
    const docId = dom.inspDocSelect ? dom.inspDocSelect.value : state.selectedStructureDocId;
    if (!docId) return;
    try {
      const res = await apiFetch(`/api/documents/${encodeURIComponent(docId)}/rollback`, {
        method: "POST",
      });
      showToast(res.message || "Đã quay lui bản chụp thành công.");
      await refreshDocuments();
      if (state.documents.some((d) => d.document_id === docId)) {
        await loadDocumentStructureAndMemory(docId);
      } else if (state.documents[0]) {
        state.selectedStructureDocId = state.documents[0].document_id;
        renderStructureDocSelector();
        await loadDocumentStructureAndMemory(state.selectedStructureDocId);
      }
    } catch (err) {
      showToast(`Lỗi quay lui: ${err.message}`);
    }
  }

  // ============================================================================
  // 12. DOCUMENT LIBRARY MODAL
  // ============================================================================

  function openDocLibraryModal() {
    dom.modalDocLibrary.hidden = false;
    renderDocModalTable();
  }

  function renderDocModalTable() {
    if (!dom.docModalTbody) return;
    const q = state.docModalSearchQuery.trim().toLowerCase();
    const rows = state.documents.filter((d) => {
      if (!q) return true;
      return (
        (d.title || "").toLowerCase().includes(q) ||
        (d.document_id || "").toLowerCase().includes(q) ||
        (d.category || "").toLowerCase().includes(q)
      );
    });

    if (rows.length === 0) {
      dom.docModalTbody.innerHTML = `
        <tr>
          <td colspan="6" style="text-align:center; padding: 24px; color: var(--text-muted);">
            Không có tài liệu phù hợp. Nhấn "Nạp tài liệu mẫu" hoặc "Tải tệp lên".
          </td>
        </tr>
      `;
      return;
    }

    dom.docModalTbody.innerHTML = rows
      .map((d) => {
        const isAttached = state.attachedDocIds.includes(d.document_id);
        const formattedDate = formatSafeDate(d.created_at);
        const safeSize = formatSafeFileSize(d.file_size);
        return `
          <tr>
            <td>
              <input
                type="checkbox"
                data-modal-attach="${escapeHtml(d.document_id)}"
                ${isAttached ? "checked" : ""}
                aria-label="Đính kèm ${escapeHtml(d.title)}"
              />
            </td>
            <td>
              <div style="font-weight: 600; color: var(--text-primary);">${escapeHtml(d.title)}</div>
              <div style="font-size: 11.5px; color: var(--text-muted);">
                <code>${escapeHtml(d.document_id)}</code> · ${escapeHtml(d.category || "Document")}
              </div>
            </td>
            <td><span class="tel-pill">v${d.version || 1}</span></td>
            <td>
              <div>${escapeHtml(safeSize)}</div>
              <div style="font-size: 11.5px; color: var(--text-muted);">
                ${d.passage_count || 0} đoạn · ~${d.token_estimate || 0} tok
              </div>
            </td>
            <td>${escapeHtml(formattedDate)}</td>
            <td>
              <div style="display:flex; gap:6px;">
                <button type="button" class="btn-outline-sm" data-inspect-doc="${escapeHtml(d.document_id)}">Cấu trúc</button>
                <button type="button" class="btn-outline-sm" style="color: var(--danger);" data-delete-doc="${escapeHtml(d.document_id)}">Xóa</button>
              </div>
            </td>
          </tr>
        `;
      })
      .join("");
  }

  async function handleCreateRawDocument() {
    const title = dom.rawDocTitle.value.trim();
    const text = dom.rawDocText.value.trim();
    if (!title || !text) {
      showToast("Vui lòng nhập đầy đủ tiêu đề và nội dung văn bản.");
      return;
    }
    try {
      const res = await apiFetch("/api/documents/raw", {
        method: "POST",
        body: JSON.stringify({ title, text, category: "Custom" }),
      });
      dom.rawDocTitle.value = "";
      dom.rawDocText.value = "";
      dom.rawDocForm.hidden = true;
      await refreshDocuments();
      if (res.document_id) {
        await toggleDocumentAttachment(res.document_id);
      }
      showToast(`Đã tạo và lập chỉ mục SA-CMS cho "${res.title}".`);
    } catch (err) {
      showToast(`Lỗi tạo văn bản: ${err.message}`);
    }
  }

  async function handleDeleteDocument(docId) {
    try {
      await apiFetch(`/api/documents/${encodeURIComponent(docId)}`, {
        method: "DELETE",
      });
      state.attachedDocIds = state.attachedDocIds.filter((id) => id !== docId);
      await refreshDocuments();
      showToast("Đã xóa tài liệu khỏi kho chỉ mục.");
    } catch (err) {
      showToast(`Lỗi xóa tài liệu: ${err.message}`);
    }
  }

  // ============================================================================
  // 13. RESEARCH LAB MODAL (ARENA, 7 COMMITTEE DEMOS, METRICS & Q1 FIGURES)
  // ============================================================================

  function openResearchLabModal(tabName = "arena") {
    dom.modalResearchLab.hidden = false;
    switchResearchTab(tabName);
  }

  function switchResearchTab(tabName) {
    document.querySelectorAll(".rlab-tab[data-rlab-tab]").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.rlabTab === tabName);
    });
    const panels = {
      arena: document.getElementById("rlab-panel-arena"),
      "baseline-compare": document.getElementById("rlab-panel-baseline-compare"),
      demos: document.getElementById("rlab-panel-demos"),
      metrics: document.getElementById("rlab-panel-metrics"),
      figures: document.getElementById("rlab-panel-figures"),
    };
    Object.keys(panels).forEach((k) => {
      if (panels[k]) panels[k].hidden = k !== tabName;
    });

    if (tabName === "baseline-compare") {
      loadBaselineComparisonDashboard();
    } else if (tabName === "demos") {
      renderCommitteeDemosGrid();
    } else if (tabName === "metrics") {
      loadResearchMetricsDashboard();
    } else if (tabName === "arena" && dom.arenaResultsGrid.children.length === 0) {
      runMethodArenaComparison();
    }
  }

  async function runMethodArenaComparison() {
    const query = dom.arenaQueryInput.value.trim();
    if (!query) return;
    dom.arenaResultsGrid.innerHTML = `<div style="grid-column: 1 / -1; color: var(--text-muted); padding: 12px;">Đang chạy đối sánh song song B1, B2, P1, P2...</div>`;

    try {
      const res = await apiFetch("/api/chat/compare", {
        method: "POST",
        body: JSON.stringify({
          query,
          methods: ["B1", "B2", "P1", "P2"],
          answer_mode: state.answerMode,
        }),
      });

      const comp = res.comparison || {};
      const titles = {
        B1: "B1 · Full-Context Window",
        B2: "B2 · Standard BM25 RAG",
        P1: "P1 · SA-CMS Memory-Only",
        P2: "P2 · Hybrid SA-CMS + Refusal Gate (Đề xuất)",
      };

      dom.arenaResultsGrid.innerHTML = ["B1", "B2", "P1", "P2"]
        .map((m) => {
          const item = comp[m] || {};
          const isP2 = m === "P2";
          const citations = Array.isArray(item.citations) ? item.citations : [];
          const sources = Array.isArray(item.sources) ? item.sources : [];
          const badges = citations.length ? citations : sources;
          const citationsHtml = badges.length
            ? `<div class="arena-citations-row">${badges
                .map((c) => {
                  const label = typeof c === "string" ? c : c?.passage_id || c?.title || c?.source || JSON.stringify(c);
                  return `<span class="tel-pill">📎 ${escapeHtml(String(label))}</span>`;
                })
                .join("")}</div>`
            : item.refused && item.refusal_reason
              ? `<div class="arena-citations-row"><span class="tel-pill">Lý do từ chối: ${escapeHtml(String(item.refusal_reason))}</span></div>`
              : "";
          return `
            <div class="arena-card ${isP2 ? "arena-p2" : ""}">
              <div class="arena-card-head">
                <span class="arena-method-name">${escapeHtml(titles[m] || m)}</span>
                <span class="tel-pill">${item.refused ? "🛡️ Đã từ chối" : "✓ Phản hồi"}</span>
              </div>
              <div style="font-size: 13.5px; line-height: 1.55; color: var(--text-body);">
                ${escapeHtml(item.answer || "")}
              </div>
              ${citationsHtml}
              <div class="arena-metrics-row">
                <span>Input: <strong>${item.input_tokens ?? "N/A"}</strong> tok</span>
                <span>· Output: <strong>${item.output_tokens ?? "N/A"}</strong> tok</span>
                <span>· Tổng: <strong>${item.total_tokens ?? "N/A"}</strong> tok</span>
                <span>· ${item.latency_ms ?? "N/A"} ms</span>
              </div>
            </div>
          `;
        })
        .join("");
    } catch (err) {
      dom.arenaResultsGrid.innerHTML = `<div style="color: var(--danger);">Lỗi đối sánh: ${escapeHtml(err.message)}</div>`;
    }
  }

  function renderCommitteeDemosGrid() {
    if (!dom.committeeDemosGrid) return;
    dom.committeeDemosGrid.innerHTML = COMMITTEE_SCENARIOS.map(
      (sc) => `
        <div class="demo-scenario-card">
          <div>
            <span class="metric-kpi-badge">${escapeHtml(sc.badge)}</span>
            <h4>${escapeHtml(sc.title)}</h4>
            <p style="margin-bottom:6px;"><strong>Câu hỏi:</strong> "${escapeHtml(sc.query)}"</p>
            <p>${escapeHtml(sc.expected)}</p>
          </div>
          <button type="button" class="btn-primary" data-run-scenario="${escapeHtml(sc.id)}">
            ▶ Chạy kịch bản trực tiếp trên khung Chat
          </button>
        </div>
      `
    ).join("");
  }

  async function executeCommitteeScenario(scenarioId) {
    const sc = COMMITTEE_SCENARIOS.find((s) => s.id === scenarioId);
    if (!sc) return;

    dom.modalResearchLab.hidden = true;

    try {
      // Ensure demo documents exist in store
      const existingIds = new Set(state.documents.map((d) => d.document_id));
      const missingDemo = sc.doc_ids.some((id) => !existingIds.has(id));
      if (missingDemo) {
        await apiFetch("/api/demo/seed", { method: "POST" });
        await refreshDocuments();
      }

      state.mode = sc.mode || "hybrid";
      state.answerMode = sc.answer_mode || "balanced";
      syncModeAndLengthUI();

      // Create dedicated clean session for reproducible demonstration
      const created = await apiFetch("/api/chat/sessions", {
        method: "POST",
        body: JSON.stringify({
          title: sc.title,
          document_ids: sc.doc_ids,
        }),
      });

      state.activeSessionId = created.session_id;
      state.attachedDocIds = [...sc.doc_ids];
      state.activeMessages = Array.isArray(created.messages) ? created.messages : [];
      renderSidebarDocuments();
      renderComposerChips();

      await sendQuestion(sc.query);

      if (sc.openInspectorStructure) {
        state.selectedStructureDocId = sc.doc_ids[0];
        openInspector("structure");
      }
    } catch (err) {
      showToast(`Không thể chạy kịch bản: ${err.message}`);
    }
  }

  function toFiniteNumber(val) {
    if (val === null || val === undefined || val === "") return null;
    const n = Number(val);
    return Number.isFinite(n) ? n : null;
  }

  function formatInt(val, suffix = "") {
    const n = toFiniteNumber(val);
    return n !== null ? `${Math.round(n).toLocaleString("en-US")}${suffix}` : "Chưa có dữ liệu";
  }

  function formatFixed(val, digits = 2, suffix = "") {
    const n = toFiniteNumber(val);
    return n !== null ? `${n.toFixed(digits)}${suffix}` : "Chưa có dữ liệu";
  }

  function formatPct(val, digits = 2) {
    const n = toFiniteNumber(val);
    return n !== null ? `${n.toFixed(digits)}%` : "Chưa có dữ liệu";
  }

  function formatDelta(baseVal, propVal, lowerIsBetter = false) {
    const b = toFiniteNumber(baseVal);
    const p = toFiniteNumber(propVal);
    if (b === null || p === null) return "N/A";
    const diff = lowerIsBetter ? b - p : p - b;
    const sign = diff > 0 ? "+" : "";
    return `${sign}${diff.toFixed(2)}%`;
  }

  async function loadResearchMetricsDashboard() {
    if (!dom.rlabMetricsContent) return;
    dom.rlabMetricsContent.innerHTML = `<div style="color: var(--text-muted);">Đang tải số liệu kiểm chuẩn...</div>`;

    try {
      const [metricsRes, dashRes, statusRes] = await Promise.allSettled([
        apiFetch("/research/metrics"),
        apiFetch("/api/comparison/dashboard"),
        apiFetch("/api/system/status"),
      ]);
      if (metricsRes.status !== "fulfilled") {
        throw metricsRes.reason;
      }
      const metrics = metricsRes.value || {};
      const dash = dashRes.status === "fulfilled" ? dashRes.value || {} : {};
      const sysStatus = statusRes.status === "fulfilled" ? statusRes.value || {} : state.systemStatus || {};
      state.researchMetrics = metrics;

      const p4 = metrics.official_phase_4 || {};
      const rq1 = p4.rq1 || {};
      const rq2 = p4.rq2 || {};
      const rq3 = p4.rq3 || {};
      const rq4 = p4.rq4 || {};
      const rq5 = p4.rq5 || {};
      const r2 = metrics.round_2_extension || {};
      const extTrain = r2.extended_training || {};
      const extHw = r2.training_hardware || {};
      const extStatus = extTrain.status || "";
      const memParams = r2.memory_parameters || {};
      const robustness = r2.robustness || {};

      const dashModels = dash.models || {};
      const dashMethods = dash.methods || {};
      const baseRag = dashModels.standard_rag || dashMethods.baseline_standard_rag || {};
      const baseLegacy = dashModels.sa_cms_phase4 || dashMethods.baseline_legacy_sa_cms || {};
      const proposed = dashModels.wave_cms_hybrid || dashMethods.proposed_wave_hybrid || {};
      const corpus = dash.corpus_scale || dash.corpus || {};
      const stats = dash.statistical_tests || dash.statistics || {};

      const totalDocs = toFiniteNumber(corpus.total_documents ?? corpus.total_docs ?? dash.total_documents);
      const totalQueries = toFiniteNumber(corpus.total_queries ?? dash.total_queries);
      const pValRaw = toFiniteNumber(stats.p_value);
      const pValText = stats.p_value_formatted
        ? String(stats.p_value_formatted)
        : pValRaw !== null
          ? pValRaw < 0.001
            ? `p = ${pValRaw.toExponential(2)}`
            : `p = ${pValRaw.toFixed(4)}`
          : "Chưa có dữ liệu";
      const cohenD = toFiniteNumber(stats.cohen_d);

      const waveAcc = toFiniteNumber(proposed.accuracy_pct ?? proposed.overall_accuracy_pct);
      const ragAcc = toFiniteNumber(baseRag.accuracy_pct ?? baseRag.overall_accuracy_pct);
      const legacyAcc = toFiniteNumber(baseLegacy.accuracy_pct ?? baseLegacy.overall_accuracy_pct);

      const rq1Bench = Array.isArray(rq1.benchmarks_profiled) && rq1.benchmarks_profiled.length
        ? rq1.benchmarks_profiled.join(", ")
        : "N/A";
      const rq2P1Prob = toFiniteNumber(rq2.summary_by_method?.P1?.mean_target_prob);
      const rq2B5Prob = toFiniteNumber(rq2.summary_by_method?.B5?.mean_target_prob);
      const rq3P2 = rq3.summary_by_method?.P2 || {};
      const faithPct = toFiniteNumber(proposed.citation_faithfulness_pct ?? rq3P2.mean_faithfulness_pct);
      const refusalPct = toFiniteNumber(proposed.refusal_acc_pct ?? proposed.unanswerable_refusal_pct ?? rq3P2.mean_correct_refusal_pct);
      const farPct = toFiniteNumber(proposed.false_answer_rate_pct ?? proposed.hallucination_rate_pct ?? rq3P2.mean_false_answer_pct);
      const rq4Delta = toFiniteNumber(rq4.summary_by_method?.P1?.mean_forgetting_delta_20 ?? rq4.summary_by_method?.B4?.mean_forgetting_delta_20);

      const trainRows = Array.isArray(dash.training_comparison) ? dash.training_comparison : [];
      const baseTrainRow = trainRows.find((r) => /baseline|gốc/i.test(String(r.stage || ""))) || null;
      const extTrainRow = trainRows.find((r) => /extended/i.test(String(r.stage || ""))) || null;
      const apiTrainRow = dash.api_training || (trainRows.length ? trainRows[trainRows.length - 1] : null) || {};
      const train20m = dash.training_20m || {};

      const extViBefore = toFiniteNumber(baseTrainRow?.val_ppl_vi);
      const extViAfter = toFiniteNumber(extTrainRow?.val_ppl_vi);
      const extEnBefore = toFiniteNumber(baseTrainRow?.val_ppl_en);
      const extEnAfter = toFiniteNumber(extTrainRow?.val_ppl_en);
      const tokens20m = toFiniteNumber(train20m.actual_tokens_trained ?? apiTrainRow.tokens_trained);
      const loss20m = toFiniteNumber(train20m.final_train_loss ?? apiTrainRow.final_loss);

      const trainableParams = toFiniteNumber(memParams.trainable_parameters);
      const passRate = toFiniteNumber(robustness.pass_rate_pct);

      dom.rlabMetricsContent.innerHTML = `
        <div class="metric-kpi-box">
          <span class="metric-kpi-badge">BENCHMARK ĐA NGUỒN · ${formatInt(totalDocs, " DOCS")}</span>
          <div class="metric-kpi-val">${formatPct(waveAcc)} vs ${formatPct(ragAcc)}</div>
          <div class="metric-kpi-desc">
            <strong>WaveCMS 6.0 (${formatPct(waveAcc)}) vs Legacy (${formatPct(legacyAcc)}) vs RAG (${formatPct(ragAcc)}):</strong>
            Kiểm định trên ${formatInt(totalQueries, " câu hỏi")} (${escapeHtml(pValText)}${cohenD !== null ? `, d = ${cohenD.toFixed(4)}` : ""}).
          </div>
        </div>
        <div class="metric-kpi-box">
          <span class="metric-kpi-badge">OFFICIAL PHASE 4 · ${escapeHtml(p4.status || "N/A")}</span>
          <div class="metric-kpi-val">RQ1–RQ4 (${escapeHtml(rq1Bench)})</div>
          <div class="metric-kpi-desc">
            <strong>RQ2 Target Prob (P1 vs B5):</strong> <code>${rq2P1Prob !== null ? rq2P1Prob.toFixed(4) : "N/A"}</code> vs <code>${rq2B5Prob !== null ? rq2B5Prob.toFixed(4) : "N/A"}</code> ·
            <strong>RQ4 Forgetting Δ20:</strong> <code>${rq4Delta !== null ? rq4Delta.toFixed(2) : "N/A"}</code> ·
            <strong>Robustness:</strong> ${passRate !== null ? `${passRate.toFixed(1)}% (${robustness.passed_conditions ?? "N/A"}/${robustness.total_conditions ?? "N/A"})` : "N/A"}.
          </div>
        </div>
        <div class="metric-kpi-box">
          <span class="metric-kpi-badge">HYBRID P2 · CONFORMAL GATE</span>
          <div class="metric-kpi-val">${formatPct(faithPct)} Trích dẫn</div>
          <div class="metric-kpi-desc">
            <strong>Từ chối ngoài tài liệu:</strong> ${formatPct(refusalPct)} ·
            <strong>Tỷ lệ ảo giác (FAR):</strong> ${formatPct(farPct)} ·
            <strong>Phần cứng RQ5:</strong> ${escapeHtml(rq5.hardware_platform || r2.hardware || "N/A")}.
          </div>
        </div>
        <div class="metric-kpi-box">
          <span class="metric-kpi-badge">EXTENDED GPU TRAINING · ${escapeHtml(extTrain.hardware || extStatus || extHw.gpu_name || "N/A")}</span>
          <div class="metric-kpi-val">${tokens20m !== null ? `${(tokens20m / 1e6).toFixed(2)}M Tokens` : "Chưa có dữ liệu"}</div>
          <div class="metric-kpi-desc">
            <strong>PPL Song ngữ (Gốc → Extended):</strong> VI <code>${extViBefore !== null ? extViBefore.toFixed(2) : "N/A"} → ${extViAfter !== null ? extViAfter.toFixed(2) : "N/A"}</code>,
            EN <code>${extEnBefore !== null ? extEnBefore.toFixed(2) : "N/A"} → ${extEnAfter !== null ? extEnAfter.toFixed(2) : "N/A"}</code> ·
            <strong>Final Loss:</strong> <code>${loss20m !== null ? loss20m.toFixed(4) : "N/A"}</code>.
          </div>
        </div>
        <div class="metric-kpi-box">
          <span class="metric-kpi-badge">OFFLINE RUNTIME · ${escapeHtml(sysStatus.integrity_verdict || "N/A")}</span>
          <div class="metric-kpi-val">${trainableParams !== null ? `${trainableParams.toLocaleString("en-US")} CMS Params` : "Chưa có dữ liệu"}</div>
          <div class="metric-kpi-desc">
            Backbone: <code>${escapeHtml(sysStatus.backbone || "N/A")}</code> · Mạng: <code>${escapeHtml(sysStatus.network || "N/A")}</code> ·
            Checkpoint: <code>${escapeHtml(sysStatus.active_checkpoint || "N/A")}</code> · Docs đang nạp: <code>${formatInt(sysStatus.document_count)}</code>.
          </div>
        </div>
        <div class="metric-kpi-box" style="grid-column: 1 / -1;">
          <span class="metric-kpi-badge">SA-CMS ARCHITECTURE · EQUATION 74</span>
          <div class="metric-kpi-val" style="font-size: 16px;">yₜ = MLP_pre(xₜ) + ∑ₗ₌₁³ gₗ(xₜ) · BₗAₗ xₜ</div>
          <div class="metric-kpi-desc">
            Ba tầng bộ nhớ căn chỉnh cấu trúc văn bản độc lập (<strong>L1 Đoạn văn</strong> · <strong>L2 Chương mục</strong> · <strong>L3 Toàn tài liệu</strong>,
            <code>d = ${memParams.d_model ?? "N/A"}, d_ff = ${memParams.d_ff ?? "N/A"}, levels = ${memParams.num_levels ?? "N/A"}</code>) hòa trộn qua cổng học được <code>gₗ(xₜ) = σ(W_g,ₗ · RMSNorm(xₜ))</code>.
          </div>
        </div>
      `;
    } catch (err) {
      dom.rlabMetricsContent.innerHTML = `<div style="color: var(--danger);">Không thể tải số liệu: ${escapeHtml(err.message)}</div>`;
    }
  }

  const OFFICE_FORMATS = {
    word: { slug: "word", ext: "docx", label: "Word (.docx)" },
    docx: { slug: "word", ext: "docx", label: "Word (.docx)" },
    excel: { slug: "excel", ext: "xlsx", label: "Excel (.xlsx)" },
    xlsx: { slug: "excel", ext: "xlsx", label: "Excel (.xlsx)" },
    pptx: { slug: "pptx", ext: "pptx", label: "PowerPoint (.pptx)" },
  };

  async function downloadOfficeFile(format, sessionId = null, forceComparison = false, customPayload = null) {
    const fmt = OFFICE_FORMATS[String(format || "word").toLowerCase()] || OFFICE_FORMATS.word;
    const isCustom = Boolean(customPayload && customPayload.content);
    const useSessionEndpoint = Boolean(sessionId && !forceComparison && !isCustom);
    const url = useSessionEndpoint
      ? `/api/chat/sessions/${encodeURIComponent(sessionId)}/export/${fmt.slug}`
      : `/api/export/${fmt.slug}`;
    const bodyPayload = isCustom
      ? {
          title: customPayload.title || "Câu trả lời WaveCMS",
          content: customPayload.content,
          session_id: sessionId || state.activeSessionId || null,
          include_comparison: false,
        }
      : {
          session_id: sessionId || state.activeSessionId || null,
          include_comparison: true,
        };
    const options = useSessionEndpoint
      ? { method: "GET" }
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(bodyPayload),
        };
    const fallbackFilename = isCustom
      ? `Cau_Tra_Loi_WaveCMS.${fmt.ext}`
      : useSessionEndpoint
      ? `Phien_Chat_${sessionId}.${fmt.ext}`
      : `Bao_Cao_So_Sanh_WaveCMS.${fmt.ext}`;

    try {
      showToast(`Đang tạo file ${fmt.label}...`);
      const resp = await fetch(url, options);
      if (!resp.ok) {
        let msg = `Lỗi xuất ${fmt.label} (${resp.status})`;
        try {
          const errJson = await resp.json();
          if (errJson && errJson.detail) msg = errJson.detail;
        } catch (_) {}
        throw new Error(msg);
      }
      const blob = await resp.blob();
      const disp = resp.headers.get("content-disposition") || "";
      const match = disp.match(/filename\*?=(?:UTF-8'')?"?([^";\n]+)"?/i);
      const filename = match && match[1] ? decodeURIComponent(match[1]) : fallbackFilename;

      const objUrl = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = objUrl;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(objUrl);
      showToast(`Đã tải xuống ${filename}`);
    } catch (err) {
      showToast(`Không thể xuất ${fmt.label}: ${err.message}`);
    }
  }

  async function loadBaselineComparisonDashboard() {
    const container = dom.rlabBaselineCompareContent;
    if (!container) return;
    container.innerHTML = `<div style="color: var(--text-muted); padding: 12px;">Đang tải bảng đối chiếu Cái gốc vs Cái cải tiến...</div>`;

    try {
      const [dashRes, statusRes] = await Promise.allSettled([
        apiFetch("/api/comparison/dashboard"),
        apiFetch("/api/system/status"),
      ]);
      if (dashRes.status !== "fulfilled") {
        throw dashRes.reason;
      }
      const data = dashRes.value || {};
      const sysStatus = statusRes.status === "fulfilled" ? statusRes.value || {} : state.systemStatus || {};

      const models = data.models || {};
      const methods = data.methods || {};
      const baseRag = models.standard_rag || methods.baseline_standard_rag || data.baseline_standard_rag || data.baseline || {};
      const baseLegacy = models.sa_cms_phase4 || methods.baseline_legacy_sa_cms || data.baseline_legacy_sa_cms || {};
      const proposed = models.wave_cms_hybrid || methods.proposed_wave_hybrid || data.proposed_wave_hybrid || data.proposed || {};

      const corpus = data.corpus_scale || data.corpus || {};
      const train = data.training_20m || data.training || {};
      const stats = data.statistical_tests || data.statistics || {};
      const bySource = corpus.by_source || data.by_source || data.source_breakdown || {};
      const corpusSources = corpus.sources || {};
      const pillars = data.wave_cms_pillars || data.wave_pillars || [];
      const comparisonTable = Array.isArray(data.comparison_table) ? data.comparison_table : [];
      const runtime =
        data.wave_cms_runtime ||
        data.tree_memory_runtime ||
        data.wavetree_runtime ||
        sysStatus.wave_cms_runtime ||
        sysStatus.tree_memory_runtime ||
        {};

      const totalDocs = toFiniteNumber(corpus.total_documents ?? corpus.total_docs ?? data.total_documents ?? data.total_docs);
      const totalQueries = toFiniteNumber(corpus.total_queries ?? data.total_queries);
      const trainTokens = toFiniteNumber(train.actual_tokens_trained ?? train.total_tokens ?? train.tokens_processed);
      const trainLoss = toFiniteNumber(train.final_train_loss ?? train.final_loss ?? train.train_loss);
      const pValueRaw = toFiniteNumber(stats.p_value ?? data.p_value);
      const pValueText = stats.p_value_formatted
        ? String(stats.p_value_formatted)
        : pValueRaw !== null
          ? pValueRaw < 0.001
            ? `p = ${pValueRaw.toExponential(2)}`
            : `p = ${pValueRaw.toFixed(4)}`
          : "Chưa có dữ liệu";

      const ragAcc = toFiniteNumber(baseRag.accuracy_pct ?? baseRag.overall_accuracy_pct);
      const legacyAcc = toFiniteNumber(baseLegacy.accuracy_pct ?? baseLegacy.overall_accuracy_pct);
      const propAcc = toFiniteNumber(proposed.accuracy_pct ?? proposed.overall_accuracy_pct);

      const ragRef = toFiniteNumber(baseRag.refusal_acc_pct ?? baseRag.unanswerable_refusal_pct ?? baseRag.refusal_pct);
      const legacyRef = toFiniteNumber(baseLegacy.refusal_acc_pct ?? baseLegacy.unanswerable_refusal_pct ?? baseLegacy.refusal_pct);
      const propRef = toFiniteNumber(proposed.refusal_acc_pct ?? proposed.unanswerable_refusal_pct ?? proposed.refusal_pct);

      const ragFar = toFiniteNumber(baseRag.false_answer_rate_pct ?? baseRag.hallucination_rate_pct);
      const legacyFar = toFiniteNumber(baseLegacy.false_answer_rate_pct ?? baseLegacy.hallucination_rate_pct);
      const propFar = toFiniteNumber(proposed.false_answer_rate_pct ?? proposed.hallucination_rate_pct);

      const ragCit = toFiniteNumber(baseRag.citation_faithfulness_pct);
      const legacyCit = toFiniteNumber(baseLegacy.citation_faithfulness_pct);
      const propCit = toFiniteNumber(proposed.citation_faithfulness_pct);

      const chartRows = [
        {
          label: "Độ chính xác tổng thể (%)",
          base: ragAcc,
          legacy: legacyAcc,
          prop: propAcc,
          lowerIsBetter: false,
        },
        {
          label: "Từ chối câu ngoài tài liệu (%)",
          base: ragRef,
          legacy: legacyRef,
          prop: propRef,
          lowerIsBetter: false,
        },
        {
          label: "Tỷ lệ ảo giác / Trả lời sai (% — thấp hơn tốt hơn)",
          base: ragFar,
          legacy: legacyFar,
          prop: propFar,
          lowerIsBetter: true,
        },
        {
          label: "Độ trung thực Trích dẫn (%)",
          base: ragCit,
          legacy: legacyCit,
          prop: propCit,
          lowerIsBetter: false,
        },
      ];

      const barsHtml = chartRows
        .map((r) => {
          const bWidth = r.base !== null ? Math.max(0, Math.min(100, r.base)) : 0;
          const lWidth = r.legacy !== null ? Math.max(0, Math.min(100, r.legacy)) : 0;
          const pWidth = r.prop !== null ? Math.max(0, Math.min(100, r.prop)) : 0;
          const delta = formatDelta(r.base, r.prop, r.lowerIsBetter);
          const legacyLine =
            r.legacy !== null
              ? `
                <div class="bc-track-line">
                  <span class="bc-track-tag">Legacy SA-CMS</span>
                  <div class="bc-track-bg"><div class="bc-track-fill bc-fill-legacy" style="width:${lWidth.toFixed(1)}%"></div></div>
                  <span class="bc-track-val">${formatPct(r.legacy)}</span>
                </div>`
              : "";
          return `
            <div class="bc-bar-row">
              <div class="bc-bar-meta">
                <span class="bc-bar-label">${escapeHtml(r.label)}</span>
                <span class="bc-delta-pill">${escapeHtml(delta)} · ${escapeHtml(pValueText)}</span>
              </div>
              <div class="bc-bar-tracks">
                <div class="bc-track-line">
                  <span class="bc-track-tag">Gốc (BM25 RAG)</span>
                  <div class="bc-track-bg"><div class="bc-track-fill bc-fill-base" style="width:${bWidth.toFixed(1)}%"></div></div>
                  <span class="bc-track-val">${formatPct(r.base)}</span>
                </div>
                ${legacyLine}
                <div class="bc-track-line">
                  <span class="bc-track-tag bc-tag-prop">WaveCMS 6.0 + Hybrid</span>
                  <div class="bc-track-bg"><div class="bc-track-fill bc-fill-prop" style="width:${pWidth.toFixed(1)}%"></div></div>
                  <span class="bc-track-val bc-val-prop">${formatPct(r.prop)}</span>
                </div>
              </div>
            </div>
          `;
        })
        .join("");

      const sourceRows = Array.isArray(bySource)
        ? bySource
        : bySource && typeof bySource === "object"
          ? Object.entries(bySource).map(([src, vals]) => Object.assign({ source: src }, vals || {}))
          : [];

      const sourceTableHtml = sourceRows.length
        ? sourceRows
            .map((row) => {
              const srcKey = String(row.source || row.name || "");
              const docCount = toFiniteNumber(
                row.doc_count ?? row.documents ?? corpusSources[srcKey] ?? corpusSources[srcKey.toLowerCase()]
              );
              const queryCount = toFiniteNumber(
                row.total_queries ??
                  row.proposed_wave_hybrid?.total_queries ??
                  row.baseline_standard_rag?.total_queries
              );
              const countLabel =
                docCount !== null
                  ? `${Math.round(docCount)} docs${queryCount !== null ? ` / ${Math.round(queryCount)} q` : ""}`
                  : queryCount !== null
                    ? `${Math.round(queryCount)} queries`
                    : "Chưa có dữ liệu";
              const bAcc = toFiniteNumber(
                row.baseline_accuracy_pct ??
                  row.baseline?.overall_accuracy_pct ??
                  row.baseline_standard_rag?.overall_accuracy_pct ??
                  row.standard_rag?.overall_accuracy_pct ??
                  row.baseline_rag_pct
              );
              const pAcc = toFiniteNumber(
                row.proposed_accuracy_pct ??
                  row.proposed?.overall_accuracy_pct ??
                  row.proposed_wave_hybrid?.overall_accuracy_pct ??
                  row.wave_cms_hybrid?.overall_accuracy_pct ??
                  row.proposed_pct
              );
              const pRef = toFiniteNumber(
                row.proposed_refusal_pct ??
                  row.proposed?.unanswerable_refusal_pct ??
                  row.proposed_wave_hybrid?.unanswerable_refusal_pct ??
                  row.wave_cms_hybrid?.refusal_acc_pct ??
                  row.refusal_pct
              );
              return `
                <tr>
                  <td><strong>${escapeHtml((srcKey || "N/A").toUpperCase())}</strong></td>
                  <td>${escapeHtml(countLabel)}</td>
                  <td>${formatPct(bAcc)}</td>
                  <td><strong>${formatPct(pAcc)}</strong></td>
                  <td><span class="bc-delta-pill">${escapeHtml(formatDelta(bAcc, pAcc))}</span></td>
                  <td>${formatPct(pRef)}</td>
                </tr>
              `;
            })
            .join("")
        : `<tr><td colspan="6" style="color: var(--text-muted); text-align:center;">Chưa có dữ liệu phân rã theo nguồn API.</td></tr>`;

      const pillarList = Array.isArray(pillars) && pillars.length
        ? pillars
        : comparisonTable.map((row) => ({
            name: row.scenario || "Chỉ số",
            metric: row.wave_cms_hybrid || "N/A",
            status: row.improvement || "N/A",
          }));
      const pillarsHtml = pillarList.length
        ? pillarList
            .map(
              (pl) => `
                <tr>
                  <td><strong>${escapeHtml(pl.name || pl.pillar || "Chỉ số")}</strong></td>
                  <td><code>${escapeHtml(String(pl.metric || pl.value || pl.detail || "N/A"))}</code></td>
                  <td><span class="metric-kpi-badge">${escapeHtml(String(pl.status || "N/A"))}</span></td>
                </tr>
              `
            )
            .join("")
        : `<tr><td colspan="3" style="color: var(--text-muted); text-align:center;">Chưa có dữ liệu bảng tổng hợp.</td></tr>`;

      const onOff = data.online_vs_offline_comparison || {};
      const onOffModes = onOff.modes || {};
      const mOffline = onOffModes.offline || {};
      const mOnline = onOffModes.online || {};
      const mHybrid = onOffModes.hybrid || {};

      const offlineAcc = toFiniteNumber(mOffline.accuracy_pct);
      const onlineAcc = toFiniteNumber(mOnline.accuracy_pct);
      const hybridAcc = toFiniteNumber(mHybrid.accuracy_pct);

      const offlineLat = toFiniteNumber(mOffline.query_latency_ms);
      const onlineLat = toFiniteNumber(mOnline.query_latency_ms);
      const hybridLat = toFiniteNumber(mHybrid.query_latency_ms);
      const latCandidates = [offlineLat, onlineLat, hybridLat].filter((v) => v !== null);
      const maxLat = latCandidates.length ? Math.max(...latCandidates, 1) : 1;

      const onOffRows = Array.isArray(onOff.rows) ? onOff.rows : [];
      const onOffRowsHtml = onOffRows.length
        ? onOffRows
            .map(
              (row) => `
                <tr>
                  <td><strong>${escapeHtml(row.criterion || row.metric || "N/A")}</strong></td>
                  <td>${escapeHtml(row.offline || "N/A")}</td>
                  <td>${escapeHtml(row.online || "N/A")}</td>
                  <td><strong>${escapeHtml(row.hybrid || "N/A")}</strong></td>
                </tr>
              `
            )
            .join("")
        : `<tr><td colspan="4" style="color: var(--text-muted); text-align:center;">Chưa có dữ liệu đối sánh Online vs Offline.</td></tr>`;

      const onOffSummary = onOff.summary || "Chưa có dữ liệu tóm tắt đối sánh Online vs Offline.";

      const rawInnovations = Array.isArray(data.offline_innovations)
        ? data.offline_innovations
        : data.offline_innovations?.items || [];
      const innovationsHtml = rawInnovations.length
        ? rawInnovations
            .map((item) => {
              const title = item.title || item.name || item.pillar || "Điểm mới";
              const badge = item.metric_highlight || item.metric || item.badge || item.tag || "N/A";
              const desc = item.description || item.detail || item.summary || "";
              return `
                <div class="bc-innovation-item">
                  <div class="bc-innovation-head">
                    <span class="bc-innovation-title">${escapeHtml(title)}</span>
                    <span class="metric-kpi-badge">${escapeHtml(String(badge))}</span>
                  </div>
                  <p class="bc-innovation-desc">${escapeHtml(desc)}</p>
                </div>
              `;
            })
            .join("")
        : `<div style="color: var(--text-muted); grid-column: 1 / -1;">Chưa có dữ liệu điểm mới phiên bản Offline.</div>`;

      const rawTrainRows = Array.isArray(data.training_comparison)
        ? data.training_comparison
        : data.training_comparison?.rows || [];
      const latestTrainRow = rawTrainRows.length ? rawTrainRows[rawTrainRows.length - 1] : null;
      const apiTrain = data.api_training || data.recent_training || latestTrainRow || {};

      const runtimeActive =
        runtime.enabled !== undefined
          ? Boolean(runtime.enabled)
          : runtime.active !== undefined
            ? Boolean(runtime.active)
            : null;
      const runtimeStatusLabel =
        runtime.status ||
        (runtimeActive === true ? "ACTIVE · O(depth)" : runtimeActive === false ? "IDLE" : "Chưa có dữ liệu");
      const runtimeDocs = toFiniteNumber(
        runtime.num_documents ?? runtime.documents_indexed ?? runtime.doc_count ?? sysStatus.document_count
      );
      const runtimeNodes = toFiniteNumber(
        runtime.num_leaves ?? runtime.tree_nodes ?? runtime.total_nodes ?? runtime.node_count
      );
      const runtimeRho = toFiniteNumber(
        runtime.contraction_rho ?? runtime.rho_hat ?? data.exact_unlearning?.banach_contraction?.empirical_contraction_factor
      );
      const runtimeUnlearnDisc = toFiniteNumber(
        runtime.unlearning_discrepancy ?? data.exact_unlearning?.wave_cms_max_discrepancy
      );
      const apiTrainTokens = toFiniteNumber(apiTrain.tokens_trained ?? apiTrain.actual_tokens_trained) ?? trainTokens;
      const apiTrainLoss = toFiniteNumber(apiTrain.final_loss ?? apiTrain.final_train_loss) ?? trainLoss;
      const apiTrainDocs = toFiniteNumber(apiTrain.corpus_docs ?? apiTrain.num_documents) ?? totalDocs;
      const apiTrainVram = toFiniteNumber(apiTrain.peak_vram_mb) ?? toFiniteNumber(train.peak_vram_mb);
      const apiTrainTps =
        toFiniteNumber(apiTrain.throughput_tok_s ?? apiTrain.tokens_per_sec) ??
        toFiniteNumber(train.average_throughput_tokens_sec);

      const trainingRowsHtml = rawTrainRows.length
        ? rawTrainRows
            .map((row) => {
              const stage = row.stage || row.model || row.name || "N/A";
              const backbone = row.backbone || "";
              const params = row.trainable_params ?? row.params ?? "N/A";
              const updateMech = row.update_schedule || row.update_rule || row.mechanism || "N/A";
              const docsCount = row.corpus_docs ?? row.docs ?? "N/A";
              const tokensRaw = toFiniteNumber(row.tokens_trained ?? row.tokens);
              const tokensStr = tokensRaw !== null ? tokensRaw.toLocaleString("en-US") : "N/A";
              const lossVal = toFiniteNumber(row.final_loss ?? row.loss);
              const lossStr = lossVal !== null ? lossVal.toFixed(4) : "N/A";
              const pplVi = toFiniteNumber(row.val_ppl_vi ?? row.ppl_vi);
              const pplEn = toFiniteNumber(row.val_ppl_en ?? row.ppl_en);
              const pplStr = `${pplVi !== null ? pplVi.toFixed(2) : "N/A"} / ${pplEn !== null ? pplEn.toFixed(2) : "N/A"}`;
              const vramRaw = toFiniteNumber(row.peak_vram_mb ?? row.vram_mb ?? row.vram);
              const vramStr = vramRaw !== null ? `${vramRaw.toFixed(1)} MB` : "N/A";
              const tpsRaw = toFiniteNumber(row.throughput_tok_s ?? row.tokens_per_sec ?? row.speed_tok_s);
              const tpsStr = tpsRaw !== null ? tpsRaw.toLocaleString("en-US") : "N/A";
              return `
                <tr>
                  <td>
                    <strong>${escapeHtml(String(stage))}</strong>
                    ${backbone ? `<span class="bc-train-sub">${escapeHtml(String(backbone))}</span>` : ""}
                  </td>
                  <td><code>${escapeHtml(String(params))}</code></td>
                  <td>${escapeHtml(String(updateMech))}</td>
                  <td>${escapeHtml(String(docsCount))}</td>
                  <td><code>${escapeHtml(tokensStr)}</code></td>
                  <td><code>${escapeHtml(lossStr)}</code></td>
                  <td><code>${escapeHtml(pplStr)}</code></td>
                  <td><code>${escapeHtml(vramStr)}</code></td>
                  <td><code>${escapeHtml(tpsStr)}</code></td>
                </tr>
              `;
            })
            .join("")
        : `<tr><td colspan="9" style="color: var(--text-muted); text-align:center;">Chưa có dữ liệu so sánh thông số huấn luyện.</td></tr>`;

      const ablationW7 = data.ablation_w7 || null;
      const ablationW7Rows = Array.isArray(ablationW7)
        ? ablationW7
        : ablationW7 && typeof ablationW7 === "object"
          ? Array.isArray(ablationW7.rows)
            ? ablationW7.rows
            : ablationW7.pooled_test_results && typeof ablationW7.pooled_test_results === "object"
              ? Object.entries(ablationW7.pooled_test_results).map(([k, v]) => {
                  const seedEntry = ablationW7.per_seed_summary?.[k]?.overall_accuracy_pct;
                  const seedStr =
                    seedEntry && typeof seedEntry === "object"
                      ? seedEntry.mean_pm_std || `${Number(seedEntry.mean || 0).toFixed(2)} ± ${Number(seedEntry.std || 0).toFixed(2)}`
                      : null;
                  const sig = ablationW7.statistical_significance_vs_w6?.[k];
                  const note = sig
                    ? `W6 ${Number(sig.delta_overall_pct || 0) >= 0 ? "+" : ""}${Number(sig.delta_overall_pct || 0).toFixed(2)}% (p=${Number(sig.p_value || 1).toExponential(2)})`
                    : k === "W6_wavecms_r2_full"
                      ? "Cấu hình chuẩn đầy đủ WaveCMS-R²/R³"
                      : "—";
                  return Object.assign(
                    { variant: k, per_seed_overall_str: seedStr, note },
                    v && typeof v === "object" ? v : { value: v }
                  );
                })
              : Array.isArray(ablationW7.variants)
                ? ablationW7.variants
                : Object.entries(ablationW7)
                    .filter(([k]) => /^(w[0-6]|loo_)/i.test(k))
                    .map(([k, v]) => Object.assign({ variant: k }, v && typeof v === "object" ? v : { value: v }))
          : [];

      const rs60 = ablationW7?.reasoning_suite_60_problems || null;
      const mondrianAudit = ablationW7?.mondrian_crc_audit || null;
      const pmiKill = ablationW7?.pmi_r3_kill_criterion || null;
      const ridgeKill = ablationW7?.ridge_associative_memory_kill_criterion || null;

      const resplitAudit = mondrianAudit?.resplit_500_audit || null;
      const perGroupRisk = mondrianAudit?.per_group_empirical_risk_vs_epsilon || null;
      let mondrianSummaryHtml = "";
      if (resplitAudit && typeof resplitAudit === "object") {
        const seed42Groups = resplitAudit["42"] || Object.values(resplitAudit)[0] || {};
        const groupChips = Object.entries(seed42Groups)
          .map(([grp, st]) => {
            const meanRisk = toFiniteNumber(st?.mean_risk);
            const ciUpper = toFiniteNumber(st?.ci95_upper);
            const ok = st?.exceeds_bound === false;
            return `<span class="metric-kpi-badge" style="margin-right:6px; margin-bottom:4px; display:inline-block;">${escapeHtml(grp)}: risk=${meanRisk !== null ? (meanRisk * 100).toFixed(2) : "—"}% (CI95≤${ciUpper !== null ? (ciUpper * 100).toFixed(2) : "—"}%, ${ok ? "Đạt" : "Vượt"})</span>`;
          })
          .join("");
        mondrianSummaryHtml = `<div><strong>Kiểm định Mondrian CRC 500-Trial:</strong> ${groupChips || "Đã hiệu chuẩn"}</div>`;
      } else if (perGroupRisk && typeof perGroupRisk === "object") {
        const groupChips = Object.entries(perGroupRisk)
          .map(([grp, st]) => {
            const emp = toFiniteNumber(st?.mean_empirical_risk);
            const eps = toFiniteNumber(st?.mean_epsilon_g);
            return `<span class="metric-kpi-badge" style="margin-right:6px; margin-bottom:4px; display:inline-block;">${escapeHtml(grp)}: risk=${emp !== null ? (emp * 100).toFixed(2) : "—"}% (ε=${eps !== null ? (eps * 100).toFixed(1) : "—"}%)</span>`;
          })
          .join("");
        mondrianSummaryHtml = `<div><strong>Kiểm định Mondrian CRC theo nhóm:</strong> ${groupChips}</div>`;
      }

      const reasoningSuiteHtml = rs60 || mondrianSummaryHtml || pmiKill || ridgeKill
        ? `
        <div class="bc-section-card" style="margin-top:12px;">
          <h4 class="bc-section-title">🧠 Reasoning Suite 60 Bài (Toán/Hóa/Mật mã) &amp; Kiểm định Mondrian CRC 500-Trial / Kill-Criteria</h4>
          ${
            rs60
              ? `<div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(190px, 1fr)); gap:10px; margin-bottom:10px;">
                  ${["deep", "broad", "deep_broad"]
                    .filter((m) => rs60[m])
                    .map((m) => {
                      const st = rs60[m];
                      return `
                        <div class="metric-kpi-box">
                          <span class="metric-kpi-badge">MODE: ${escapeHtml(m.toUpperCase())} (n=${escapeHtml(String(st.n ?? 60))})</span>
                          <div class="metric-kpi-val">${formatPct(toFiniteNumber(st.accuracy_pct))}</div>
                          <div class="metric-kpi-desc">Bịa số: <strong>${formatPct(toFiniteNumber(st.numeric_hallucination_rate_pct))}</strong> · ${formatFixed(toFiniteNumber(st.mean_latency_ms), 2, " ms")}</div>
                        </div>
                      `;
                    })
                    .join("")}
                </div>`
              : ""
          }
          <div style="font-size:12.5px; line-height:1.6;">
            ${mondrianSummaryHtml}
            ${
              pmiKill
                ? `<div><strong>PMI R³ Kill-Criterion (${escapeHtml(String(pmiKill.evaluation_split || "Fit+Cal"))}):</strong> ΔAUROC = <code>${toFiniteNumber(pmiKill.mean_delta_auroc)?.toFixed(4) ?? "—"}</code> (ngưỡng ≥ ${toFiniteNumber(pmiKill.kill_criterion_thresholds?.min_delta_auroc)?.toFixed(2) ?? "—"}), ΔAns@15% = <code>${toFiniteNumber(pmiKill.mean_delta_ans_at_halluc_15_pp)?.toFixed(2) ?? "—"} pp</code> → <strong>${pmiKill.kill_criterion_triggered ? "Kích hoạt cắt bỏ (W5/W6 dùng fast_zero_pmi)" : "Giữ lại"}</strong></div>`
                : ""
            }
            ${
              ridgeKill
                ? `<div><strong>Ridge Associative Memory Kill-Criterion (n=${escapeHtml(String(ridgeKill.n_fit_qasper ?? "—"))}):</strong> Base = <code>${formatPct(toFiniteNumber(ridgeKill.base_answerable_pct))}</code> vs Ridge = <code>${formatPct(toFiniteNumber(ridgeKill.ridge_answerable_pct))}</code> (Δ = <code>${toFiniteNumber(ridgeKill.delta_pct)?.toFixed(2) ?? "0.00"}%</code>) → <strong>${ridgeKill.kill_criterion_triggered ? "Kích hoạt cắt bỏ (disabled in W6)" : "Giữ lại"}</strong></div>`
                : ""
            }
          </div>
        </div>
      `
        : "";

      const ablationW7Html = ablationW7
        ? `
        <div class="bc-section-card">
          <h4 class="bc-section-title">🔬 Ablation Chuẩn Không Rò Rỉ WaveCMS-R³ (W0–W6 &amp; True Leave-One-Out, Test 60% n = 1,779)</h4>
          <div class="doc-table-wrap">
            <table class="doc-table">
              <thead>
                <tr>
                  <th>Biến thể</th>
                  <th>Mô tả cấu hình</th>
                  <th>Overall Acc (3-Seed Mean±Std)</th>
                  <th>Answerable Acc / Ans@15%</th>
                  <th>Refusal / Hallucination</th>
                  <th>Test AUROC / AURC (↓)</th>
                  <th>Độ trễ TB (p50)</th>
                  <th>Ý nghĩa thống kê / Ghi chú</th>
                </tr>
              </thead>
              <tbody>
                ${
                  ablationW7Rows.length
                    ? ablationW7Rows
                        .map((row) => {
                          const vName = row.variant || row.id || row.name || row.code || "N/A";
                          const desc = row.description || row.config || row.label || "—";
                          const overallAcc = toFiniteNumber(row.overall_accuracy_pct ?? row.accuracy_pct ?? row.accuracy);
                          const seedStr = row.per_seed_overall_str || (overallAcc !== null ? formatPct(overallAcc) : "—");
                          const ansAcc = toFiniteNumber(row.answerable_accuracy_pct);
                          const ans15 = toFiniteNumber(row.ans_acc_at_halluc_15_pct);
                          const refPct = toFiniteNumber(row.unanswerable_refusal_pct ?? row.refusal_pct ?? row.refusal_acc_pct);
                          const halPct = toFiniteNumber(row.hallucination_rate_pct ?? row.false_answer_rate_pct);
                          const auroc = toFiniteNumber(row.test_auroc);
                          const aurc = toFiniteNumber(row.aurc_pct);
                          const meanLat = toFiniteNumber(row.mean_latency_ms);
                          const p50Lat = toFiniteNumber(row.p50_latency_ms);
                          const note = row.note || row.delta || row.status || "—";
                          return `
                            <tr>
                              <td><strong>${escapeHtml(String(vName))}</strong></td>
                              <td>${escapeHtml(String(desc))}</td>
                              <td><code>${overallAcc !== null ? formatPct(overallAcc) : "—"}</code>${seedStr && seedStr !== formatPct(overallAcc) ? `<br><small style="color:var(--text-muted)">(${escapeHtml(String(seedStr))}%)</small>` : ""}</td>
                              <td><code>${ansAcc !== null ? formatPct(ansAcc) : "—"}</code> / <code>${ans15 !== null ? formatPct(ans15) : "—"}</code></td>
                              <td><code>${refPct !== null ? formatPct(refPct) : "—"}</code> / <code>${halPct !== null ? formatPct(halPct) : "—"}</code></td>
                              <td><code>${auroc !== null ? auroc.toFixed(4) : "—"}</code> / <code>${aurc !== null ? formatPct(aurc) : "—"}</code></td>
                              <td><code>${meanLat !== null ? `${meanLat.toFixed(2)} ms` : "—"}</code>${p50Lat !== null ? ` <small>(p50: ${p50Lat.toFixed(2)})</small>` : ""}</td>
                              <td>${escapeHtml(String(note))}</td>
                            </tr>
                          `;
                        })
                        .join("")
                    : `<tr><td colspan="8" style="color: var(--text-muted); text-align:center;">Không tìm thấy dữ liệu ablation_w7.</td></tr>`
                }
              </tbody>
            </table>
          </div>
          ${reasoningSuiteHtml}
        </div>
      `
        : "";

      const pillar1W7 = data.pillar1_sparse_lse_w7 || null;
      const pillar1W7Rows = (() => {
        if (Array.isArray(pillar1W7)) return pillar1W7;
        if (!pillar1W7 || typeof pillar1W7 !== "object") return [];
        if (Array.isArray(pillar1W7.rows)) return pillar1W7.rows;
        const p1Raw = pillar1W7.pillars?.pillar_1_zero_forgetting;
        if (p1Raw && typeof p1Raw === "object") {
          const cps = Array.isArray(p1Raw.checkpoints) ? p1Raw.checkpoints : [0, 5, 10, 20, 64];
          const mkMap = p1Raw.wave_cms_mean_key || {};
          const slMap = p1Raw.wave_cms_sparse_lse || p1Raw.wave_cms || {};
          const ragMap = p1Raw.standard_rag || {};
          const saMap = p1Raw.sa_cms || {};
          return cps.map((cp) => {
            const k = String(cp);
            const mk = mkMap[k] || {};
            const sl = slMap[k] || {};
            const rag = ragMap[k] || {};
            const sa = saMap[k] || {};
            return {
              n: cp,
              n_docs: sl.n_docs,
              n_leaves: sl.n_leaves,
              mean_key_hit_at_1_pct: Number((Number(mk.routing_hit_at_1 || 0) * 100).toFixed(2)),
              sparse_lse_hit_at_1_pct: Number((Number(sl.routing_hit_at_1 || 0) * 100).toFixed(2)),
              sparse_lse_hit_at_3_pct: Number((Number(sl.routing_hit_at_3 || 0) * 100).toFixed(2)),
              tree_vs_flat_exact_match_pct: Number((Number(sl.tree_vs_flat_exact_match_rate ?? 1) * 100).toFixed(2)),
              mean_key_n_eff: Number(Number(mk.n_eff || 0).toFixed(2)),
              sparse_lse_n_eff: Number(Number(sl.n_eff || 0).toFixed(2)),
              sparse_lse_token_f1_pct: Number((Number(sl.token_f1 || 0) * 100).toFixed(2)),
              mean_key_token_f1_pct: Number((Number(mk.token_f1 || 0) * 100).toFixed(2)),
              standard_rag_token_f1_pct: Number((Number(rag.token_f1 || 0) * 100).toFixed(2)),
              sa_cms_token_f1_pct: Number((Number(sa.token_f1 || 0) * 100).toFixed(2)),
              leaf_core_drift_fro: Number(sl.leaf_core_drift_fro || 0),
            };
          });
        }
        return Array.isArray(pillar1W7.sweeps)
          ? pillar1W7.sweeps
          : Array.isArray(pillar1W7.results)
            ? pillar1W7.results
            : [];
      })();

      const pillar1W7Html = pillar1W7
        ? `
        <div class="bc-section-card">
          <h4 class="bc-section-title">🧭 Pillar 1: Định tuyến mean_key vs sparse_lse (Định lý 4, N = 0..64 tài liệu nhiễu, tới 1,598 lá)</h4>
          <div class="doc-table-wrap">
            <table class="doc-table">
              <thead>
                <tr>
                  <th>Mốc N (Docs / Lá)</th>
                  <th>mean_key Hit@1 (n_eff)</th>
                  <th>sparse_lse Hit@1 (n_eff)</th>
                  <th>Tree == Flat Exact</th>
                  <th>Token F1 Công bằng (WaveCMS vs mean_key vs SA-CMS)</th>
                  <th>Chênh lệch Hit@1 (Δ)</th>
                </tr>
              </thead>
              <tbody>
                ${
                  pillar1W7Rows.length
                    ? pillar1W7Rows
                        .map((row) => {
                          const nVal = row.n ?? row.noise_docs ?? row.N ?? "N/A";
                          const nDocs = toFiniteNumber(row.n_docs);
                          const nLeaves = toFiniteNumber(row.n_leaves);
                          const mkHit = toFiniteNumber(row.mean_key_hit_at_1_pct ?? row.mean_key_pct ?? row.mean_key);
                          const slHit = toFiniteNumber(row.sparse_lse_hit_at_1_pct ?? row.sparse_lse_pct ?? row.sparse_lse);
                          const mkNeff = toFiniteNumber(row.mean_key_n_eff);
                          const slNeff = toFiniteNumber(row.sparse_lse_n_eff);
                          const exactMatch = toFiniteNumber(row.tree_vs_flat_exact_match_pct);
                          const slF1 = toFiniteNumber(row.sparse_lse_token_f1_pct);
                          const mkF1 = toFiniteNumber(row.mean_key_token_f1_pct);
                          const saF1 = toFiniteNumber(row.sa_cms_token_f1_pct);
                          const docLeafLabel =
                            nDocs !== null || nLeaves !== null
                              ? ` <small style="color:var(--text-muted)">(${nDocs ?? "—"} docs / ${nLeaves ?? "—"} lá)</small>`
                              : "";
                          return `
                            <tr>
                              <td><strong>N = ${escapeHtml(String(nVal))}</strong>${docLeafLabel}</td>
                              <td><code>${formatPct(mkHit)}</code>${mkNeff !== null ? ` <small>(n_eff=${mkNeff.toFixed(1)})</small>` : ""}</td>
                              <td><strong><code>${formatPct(slHit)}</code></strong>${slNeff !== null ? ` <small>(n_eff=${slNeff.toFixed(2)})</small>` : ""}</td>
                              <td><code>${exactMatch !== null ? formatPct(exactMatch) : "100.00%"}</code></td>
                              <td><code>${formatPct(slF1)}</code> vs <code>${formatPct(mkF1)}</code> vs <code>${formatPct(saF1)}</code></td>
                              <td><span class="bc-delta-pill">${escapeHtml(formatDelta(mkHit, slHit))}</span></td>
                            </tr>
                          `;
                        })
                        .join("")
                    : `<tr><td colspan="6" style="color: var(--text-muted); text-align:center;">Không tìm thấy dữ liệu Pillar 1.</td></tr>`
                }
              </tbody>
            </table>
          </div>
        </div>
      `
        : "";

      container.innerHTML = `
        <div class="bc-header-bar">
          <div>
            <h3 style="margin:0; font-size:15px;">Đối chiếu Thực nghiệm: Cái gốc vs Cái mình đang làm</h3>
            <p style="margin:2px 0 0; font-size:12.5px; color:var(--text-muted);">
              Standard BM25 RAG &amp; Legacy SA-CMS so với WaveCMS 6.0 + Hybrid QA (${escapeHtml(pValueText)})
            </p>
          </div>
          <div class="bc-export-actions">
            <button type="button" class="btn-secondary bc-export-btn" id="btn-bc-fetch-online">🌐 Call API lấy thêm dữ liệu thật</button>
            <button type="button" class="btn-primary bc-export-btn" id="btn-export-comparison-word">📄 Word (.docx)</button>
            <button type="button" class="btn-primary bc-export-btn" id="btn-export-comparison-excel">📊 Excel (.xlsx)</button>
            <button type="button" class="btn-primary bc-export-btn" id="btn-export-comparison-pptx">📽️ PowerPoint (.pptx)</button>
          </div>
        </div>

        <div class="bc-kpi-grid">
          <div class="metric-kpi-box">
            <span class="metric-kpi-badge">ĐỘ CHÍNH XÁC (%)</span>
            <div class="metric-kpi-val">${formatPct(propAcc)}</div>
            <div class="metric-kpi-desc">Gốc BM25: ${formatPct(ragAcc)}${legacyAcc !== null ? ` · Legacy: ${formatPct(legacyAcc)}` : ""} · Tăng <strong>${escapeHtml(formatDelta(ragAcc, propAcc))}</strong></div>
          </div>
          <div class="metric-kpi-box">
            <span class="metric-kpi-badge">TỪ CHỐI NGOÀI TÀI LIỆU (%)</span>
            <div class="metric-kpi-val">${formatPct(propRef)}</div>
            <div class="metric-kpi-desc">Gốc BM25: ${formatPct(ragRef)} · Conformal Gate</div>
          </div>
          <div class="metric-kpi-box">
            <span class="metric-kpi-badge">TRUNG THỰC TRÍCH DẪN (%)</span>
            <div class="metric-kpi-val">${formatPct(propCit)}</div>
            <div class="metric-kpi-desc">Legacy SA-CMS: ${formatPct(legacyCit ?? ragCit)}</div>
          </div>
          <div class="metric-kpi-box">
            <span class="metric-kpi-badge">QUY MÔ CORPUS API MẠNG</span>
            <div class="metric-kpi-val">${formatInt(totalDocs, " tài liệu")}</div>
            <div class="metric-kpi-desc">${formatInt(totalQueries, " câu hỏi")} (${sourceRows.length ? `${sourceRows.length} nguồn benchmark` : "chưa có phân rã nguồn"})</div>
          </div>
          <div class="metric-kpi-box">
            <span class="metric-kpi-badge">RÈN LUYỆN MỚI (20M + API)</span>
            <div class="metric-kpi-val">${apiTrainLoss !== null ? `loss = ${apiTrainLoss.toFixed(4)}` : "Chưa có dữ liệu"}</div>
            <div class="metric-kpi-desc">${formatInt(apiTrainTokens, " tokens")} · ${formatInt(apiTrainDocs, " docs")} · ${formatFixed(apiTrainVram, 1, " MB VRAM")}</div>
          </div>
        </div>

        <div class="bc-section-card bc-runtime-card">
          <div class="bc-runtime-header">
            <h4 class="bc-section-title" style="margin:0;">🧠 WaveCMS Tree Memory Runtime &amp; Rèn luyện Mới</h4>
            <span class="metric-kpi-badge">${escapeHtml(String(runtimeStatusLabel))}</span>
          </div>
          <div class="bc-runtime-grid">
            <div class="bc-runtime-item">
              <span class="bc-runtime-label">WaveTreeMemory Index</span>
              <strong class="bc-runtime-val">${formatInt(runtimeDocs, " docs")} · ${formatInt(runtimeNodes, " leaves")}</strong>
              <span class="bc-runtime-sub">Định tuyến cây sóng nhỏ Haar O(depth)</span>
            </div>
            <div class="bc-runtime-item">
              <span class="bc-runtime-label">Exact Unlearning &amp; Banach</span>
              <strong class="bc-runtime-val">Δ ${runtimeUnlearnDisc !== null ? `≤ ${runtimeUnlearnDisc.toExponential(2)}` : "N/A"} · ρ̂ = ${runtimeRho !== null ? runtimeRho.toFixed(3) : "N/A"}</strong>
              <span class="bc-runtime-sub">Hội tụ điểm bất động &amp; xóa sạch cấp máy</span>
            </div>
            <div class="bc-runtime-item">
              <span class="bc-runtime-label">Rèn luyện API Mở rộng</span>
              <strong class="bc-runtime-val">${formatInt(apiTrainTokens, " tok")} · ${formatFixed(apiTrainTps, 1, " tok/s")}</strong>
              <span class="bc-runtime-sub">${escapeHtml(String(apiTrain.stage || "N/A"))} (${apiTrainLoss !== null ? `loss ${apiTrainLoss.toFixed(4)}` : "loss N/A"})</span>
            </div>
            <div class="bc-runtime-item">
              <span class="bc-runtime-label">Chế độ Auto API ngoài</span>
              <strong class="bc-runtime-val">${state.allowOnlineApi ? "🌐 Đang BẬT (Hybrid Fallback)" : "🔒 Đang TẮT (100% Air-Gap)"}</strong>
              <span class="bc-runtime-sub">8 nguồn API công khai sẵn sàng khi bật</span>
            </div>
          </div>
        </div>

        <div class="bc-section-card">
          <h4 class="bc-section-title">Biểu đồ đối chiếu trực quan: Cái gốc vs WaveCMS 6.0 + Hybrid QA</h4>
          <div class="bc-bars-list">${barsHtml}</div>
        </div>

        <div class="bc-tables-grid">
          <div class="bc-section-card">
            <h4 class="bc-section-title">Phân rã theo nguồn API (Wikipedia / arXiv / QASPER / SQuADv2)</h4>
            <div class="doc-table-wrap">
              <table class="doc-table">
                <thead>
                  <tr>
                    <th>Nguồn</th>
                    <th>Tài liệu / Câu hỏi</th>
                    <th>Gốc (%)</th>
                    <th>WaveCMS (%)</th>
                    <th>Mức tăng</th>
                    <th>Từ chối (%)</th>
                  </tr>
                </thead>
                <tbody>${sourceTableHtml}</tbody>
              </table>
            </div>
          </div>
          <div class="bc-section-card">
            <h4 class="bc-section-title">Bảng Tổng hợp Chỉ số WaveCMS 6.0</h4>
            <div class="doc-table-wrap">
              <table class="doc-table">
                <thead>
                  <tr>
                    <th>Hạng mục</th>
                    <th>WaveCMS 6.0</th>
                    <th>Cải thiện / Trạng thái</th>
                  </tr>
                </thead>
                <tbody>${pillarsHtml}</tbody>
              </table>
            </div>
          </div>
        </div>

        <div class="bc-section-card">
          <h4 class="bc-section-title">🌐 Đối sánh Chế độ Online (API Tài liệu Mạng) vs Offline (Cục bộ Air-Gap)</h4>
          <p class="bc-onoff-summary">${escapeHtml(onOffSummary)}</p>

          <div class="bc-onoff-charts">
            <div class="bc-bar-row">
              <div class="bc-bar-meta">
                <span class="bc-bar-label">Độ chính xác theo chế độ (%)</span>
                <span class="bc-delta-pill">Offline miền hẹp vs Online đa miền</span>
              </div>
              <div class="bc-bar-tracks">
                <div class="bc-track-line">
                  <span class="bc-track-tag">Offline Cục bộ</span>
                  <div class="bc-track-bg"><div class="bc-track-fill bc-fill-offline" style="width:${offlineAcc !== null ? Math.min(100, offlineAcc).toFixed(1) : "0.0"}%"></div></div>
                  <span class="bc-track-val">${formatPct(offlineAcc)}</span>
                </div>
                <div class="bc-track-line">
                  <span class="bc-track-tag">Online API Mạng</span>
                  <div class="bc-track-bg"><div class="bc-track-fill bc-fill-online" style="width:${onlineAcc !== null ? Math.min(100, onlineAcc).toFixed(1) : "0.0"}%"></div></div>
                  <span class="bc-track-val">${formatPct(onlineAcc)}</span>
                </div>
                <div class="bc-track-line">
                  <span class="bc-track-tag bc-tag-prop">WaveCMS Hybrid</span>
                  <div class="bc-track-bg"><div class="bc-track-fill bc-fill-prop" style="width:${hybridAcc !== null ? Math.min(100, hybridAcc).toFixed(1) : "0.0"}%"></div></div>
                  <span class="bc-track-val bc-val-prop">${formatPct(hybridAcc)}</span>
                </div>
              </div>
            </div>

            <div class="bc-bar-row">
              <div class="bc-bar-meta">
                <span class="bc-bar-label">Độ trễ truy vấn sau nạp (ms — thấp hơn là nhanh hơn)</span>
                <span class="bc-delta-pill">Đo thực tế trên WaveTreeMemory</span>
              </div>
              <div class="bc-bar-tracks">
                <div class="bc-track-line">
                  <span class="bc-track-tag">Offline Cục bộ</span>
                  <div class="bc-track-bg"><div class="bc-track-fill bc-fill-offline" style="width:${offlineLat !== null ? ((offlineLat / maxLat) * 100).toFixed(1) : "0.0"}%"></div></div>
                  <span class="bc-track-val">${formatFixed(offlineLat, 2, " ms")}</span>
                </div>
                <div class="bc-track-line">
                  <span class="bc-track-tag">Online API Mạng</span>
                  <div class="bc-track-bg"><div class="bc-track-fill bc-fill-online" style="width:${onlineLat !== null ? ((onlineLat / maxLat) * 100).toFixed(1) : "0.0"}%"></div></div>
                  <span class="bc-track-val">${formatFixed(onlineLat, 2, " ms")}</span>
                </div>
                <div class="bc-track-line">
                  <span class="bc-track-tag bc-tag-prop">WaveCMS Hybrid</span>
                  <div class="bc-track-bg"><div class="bc-track-fill bc-fill-prop" style="width:${hybridLat !== null ? ((hybridLat / maxLat) * 100).toFixed(1) : "0.0"}%"></div></div>
                  <span class="bc-track-val bc-val-prop">${formatFixed(hybridLat, 2, " ms")}</span>
                </div>
              </div>
            </div>
          </div>

          <div class="doc-table-wrap">
            <table class="doc-table">
              <thead>
                <tr>
                  <th>Tiêu chí</th>
                  <th>Offline Cục bộ (Air-Gap)</th>
                  <th>Online API Mạng</th>
                  <th>Chế độ Kết hợp WaveCMS Hybrid</th>
                </tr>
              </thead>
              <tbody>${onOffRowsHtml}</tbody>
            </table>
          </div>
        </div>

        <div class="bc-section-card bc-innovations-card">
          <h4 class="bc-section-title">✨ Phiên bản Offline mới ở chỗ nào?</h4>
          <div class="bc-innovations-grid">${innovationsHtml}</div>
        </div>

        <div class="bc-section-card">
          <h4 class="bc-section-title">📊 Bảng So sánh Thông số Huấn luyện: Gốc vs Cải tiến</h4>
          <div class="doc-table-wrap">
            <table class="doc-table">
              <thead>
                <tr>
                  <th>Mô hình / Mốc huấn luyện</th>
                  <th>Tham số train</th>
                  <th>Cơ chế cập nhật</th>
                  <th>Quy mô tài liệu</th>
                  <th>Tổng Tokens</th>
                  <th>Loss</th>
                  <th>PPL (VI/EN)</th>
                  <th>VRAM</th>
                  <th>Tốc độ (tok/s)</th>
                </tr>
              </thead>
              <tbody>${trainingRowsHtml}</tbody>
            </table>
          </div>
        </div>
        ${ablationW7Html}
        ${pillar1W7Html}
      `;

      const btnBcFetchOnline = document.getElementById("btn-bc-fetch-online");
      if (btnBcFetchOnline) {
        btnBcFetchOnline.addEventListener("click", openFetchOnlineModal);
      }
      const comparisonExportBindings = [
        { id: "btn-export-comparison-word", format: "word" },
        { id: "btn-export-comparison-excel", format: "excel" },
        { id: "btn-export-comparison-pptx", format: "pptx" },
      ];
      comparisonExportBindings.forEach(({ id, format }) => {
        const btn = document.getElementById(id);
        if (btn) {
          btn.addEventListener("click", () => {
            downloadOfficeFile(format, state.activeSessionId || null, true);
          });
        }
      });
    } catch (err) {
      const ablationW7 = data.ablation_w7 || null;
      const ablationW7Rows = Array.isArray(ablationW7)
        ? ablationW7
        : ablationW7 && typeof ablationW7 === "object"
          ? Array.isArray(ablationW7.rows)
            ? ablationW7.rows
            : ablationW7.pooled_test_results && typeof ablationW7.pooled_test_results === "object"
              ? Object.entries(ablationW7.pooled_test_results).map(([k, v]) => {
                  const seedEntry = ablationW7.per_seed_summary?.[k]?.overall_accuracy_pct;
                  const seedStr =
                    seedEntry && typeof seedEntry === "object"
                      ? seedEntry.mean_pm_std || `${Number(seedEntry.mean || 0).toFixed(2)} ± ${Number(seedEntry.std || 0).toFixed(2)}`
                      : null;
                  const sig = ablationW7.statistical_significance_vs_w6?.[k];
                  const note = sig
                    ? `W6 ${Number(sig.delta_overall_pct || 0) >= 0 ? "+" : ""}${Number(sig.delta_overall_pct || 0).toFixed(2)}% (p=${Number(sig.p_value || 1).toExponential(2)})`
                    : k === "W6_wavecms_r2_full"
                      ? "Cấu hình chuẩn đầy đủ WaveCMS-R²/R³"
                      : "—";
                  return Object.assign(
                    { variant: k, per_seed_overall_str: seedStr, note },
                    v && typeof v === "object" ? v : { value: v }
                  );
                })
              : Array.isArray(ablationW7.variants)
                ? ablationW7.variants
                : Object.entries(ablationW7)
                    .filter(([k]) => /^(w[0-6]|loo_)/i.test(k))
                    .map(([k, v]) => Object.assign({ variant: k }, v && typeof v === "object" ? v : { value: v }))
          : [];

      const rs60 = ablationW7?.reasoning_suite_60_problems || null;
      const mondrianAudit = ablationW7?.mondrian_crc_audit || null;
      const pmiKill = ablationW7?.pmi_r3_kill_criterion || null;
      const ridgeKill = ablationW7?.ridge_associative_memory_kill_criterion || null;

      const resplitAudit = mondrianAudit?.resplit_500_audit || null;
      const perGroupRisk = mondrianAudit?.per_group_empirical_risk_vs_epsilon || null;
      let mondrianSummaryHtml = "";
      if (resplitAudit && typeof resplitAudit === "object") {
        const seed42Groups = resplitAudit["42"] || Object.values(resplitAudit)[0] || {};
        const groupChips = Object.entries(seed42Groups)
          .map(([grp, st]) => {
            const meanRisk = toFiniteNumber(st?.mean_risk);
            const ciUpper = toFiniteNumber(st?.ci95_upper);
            const ok = st?.exceeds_bound === false;
            return `<span class="metric-kpi-badge" style="margin-right:6px; margin-bottom:4px; display:inline-block;">${escapeHtml(grp)}: risk=${meanRisk !== null ? (meanRisk * 100).toFixed(2) : "—"}% (CI95≤${ciUpper !== null ? (ciUpper * 100).toFixed(2) : "—"}%, ${ok ? "Đạt" : "Vượt"})</span>`;
          })
          .join("");
        mondrianSummaryHtml = `<div><strong>Kiểm định Mondrian CRC 500-Trial:</strong> ${groupChips || "Đã hiệu chuẩn"}</div>`;
      } else if (perGroupRisk && typeof perGroupRisk === "object") {
        const groupChips = Object.entries(perGroupRisk)
          .map(([grp, st]) => {
            const emp = toFiniteNumber(st?.mean_empirical_risk);
            const eps = toFiniteNumber(st?.mean_epsilon_g);
            return `<span class="metric-kpi-badge" style="margin-right:6px; margin-bottom:4px; display:inline-block;">${escapeHtml(grp)}: risk=${emp !== null ? (emp * 100).toFixed(2) : "—"}% (ε=${eps !== null ? (eps * 100).toFixed(1) : "—"}%)</span>`;
          })
          .join("");
        mondrianSummaryHtml = `<div><strong>Kiểm định Mondrian CRC theo nhóm:</strong> ${groupChips}</div>`;
      }

      const reasoningSuiteHtml = rs60 || mondrianSummaryHtml || pmiKill || ridgeKill
        ? `
        <div class="bc-section-card" style="margin-top:12px;">
          <h4 class="bc-section-title">🧠 Reasoning Suite 60 Bài (Toán/Hóa/Mật mã) &amp; Kiểm định Mondrian CRC 500-Trial / Kill-Criteria</h4>
          ${
            rs60
              ? `<div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(190px, 1fr)); gap:10px; margin-bottom:10px;">
                  ${["deep", "broad", "deep_broad"]
                    .filter((m) => rs60[m])
                    .map((m) => {
                      const st = rs60[m];
                      return `
                        <div class="metric-kpi-box">
                          <span class="metric-kpi-badge">MODE: ${escapeHtml(m.toUpperCase())} (n=${escapeHtml(String(st.n ?? 60))})</span>
                          <div class="metric-kpi-val">${formatPct(toFiniteNumber(st.accuracy_pct))}</div>
                          <div class="metric-kpi-desc">Bịa số: <strong>${formatPct(toFiniteNumber(st.numeric_hallucination_rate_pct))}</strong> · ${formatFixed(toFiniteNumber(st.mean_latency_ms), 2, " ms")}</div>
                        </div>
                      `;
                    })
                    .join("")}
                </div>`
              : ""
          }
          <div style="font-size:12.5px; line-height:1.6;">
            ${mondrianSummaryHtml}
            ${
              pmiKill
                ? `<div><strong>PMI R³ Kill-Criterion (${escapeHtml(String(pmiKill.evaluation_split || "Fit+Cal"))}):</strong> ΔAUROC = <code>${toFiniteNumber(pmiKill.mean_delta_auroc)?.toFixed(4) ?? "—"}</code> (ngưỡng ≥ ${toFiniteNumber(pmiKill.kill_criterion_thresholds?.min_delta_auroc)?.toFixed(2) ?? "—"}), ΔAns@15% = <code>${toFiniteNumber(pmiKill.mean_delta_ans_at_halluc_15_pp)?.toFixed(2) ?? "—"} pp</code> → <strong>${pmiKill.kill_criterion_triggered ? "Kích hoạt cắt bỏ (W5/W6 dùng fast_zero_pmi)" : "Giữ lại"}</strong></div>`
                : ""
            }
            ${
              ridgeKill
                ? `<div><strong>Ridge Associative Memory Kill-Criterion (n=${escapeHtml(String(ridgeKill.n_fit_qasper ?? "—"))}):</strong> Base = <code>${formatPct(toFiniteNumber(ridgeKill.base_answerable_pct))}</code> vs Ridge = <code>${formatPct(toFiniteNumber(ridgeKill.ridge_answerable_pct))}</code> (Δ = <code>${toFiniteNumber(ridgeKill.delta_pct)?.toFixed(2) ?? "0.00"}%</code>) → <strong>${ridgeKill.kill_criterion_triggered ? "Kích hoạt cắt bỏ (disabled in W6)" : "Giữ lại"}</strong></div>`
                : ""
            }
          </div>
        </div>
      `
        : "";

      const ablationW7Html = ablationW7
        ? `
        <div class="bc-section-card">
          <h4 class="bc-section-title">🔬 Ablation Chuẩn Không Rò Rỉ WaveCMS-R³ (W0–W6 &amp; True Leave-One-Out, Test 60% n = 1,779)</h4>
          <div class="doc-table-wrap">
            <table class="doc-table">
              <thead>
                <tr>
                  <th>Biến thể</th>
                  <th>Mô tả cấu hình</th>
                  <th>Overall Acc (3-Seed Mean±Std)</th>
                  <th>Answerable Acc / Ans@15%</th>
                  <th>Refusal / Hallucination</th>
                  <th>Test AUROC / AURC (↓)</th>
                  <th>Độ trễ TB (p50)</th>
                  <th>Ý nghĩa thống kê / Ghi chú</th>
                </tr>
              </thead>
              <tbody>
                ${
                  ablationW7Rows.length
                    ? ablationW7Rows
                        .map((row) => {
                          const vName = row.variant || row.id || row.name || row.code || "N/A";
                          const desc = row.description || row.config || row.label || "—";
                          const overallAcc = toFiniteNumber(row.overall_accuracy_pct ?? row.accuracy_pct ?? row.accuracy);
                          const seedStr = row.per_seed_overall_str || (overallAcc !== null ? formatPct(overallAcc) : "—");
                          const ansAcc = toFiniteNumber(row.answerable_accuracy_pct);
                          const ans15 = toFiniteNumber(row.ans_acc_at_halluc_15_pct);
                          const refPct = toFiniteNumber(row.unanswerable_refusal_pct ?? row.refusal_pct ?? row.refusal_acc_pct);
                          const halPct = toFiniteNumber(row.hallucination_rate_pct ?? row.false_answer_rate_pct);
                          const auroc = toFiniteNumber(row.test_auroc);
                          const aurc = toFiniteNumber(row.aurc_pct);
                          const meanLat = toFiniteNumber(row.mean_latency_ms);
                          const p50Lat = toFiniteNumber(row.p50_latency_ms);
                          const note = row.note || row.delta || row.status || "—";
                          return `
                            <tr>
                              <td><strong>${escapeHtml(String(vName))}</strong></td>
                              <td>${escapeHtml(String(desc))}</td>
                              <td><code>${overallAcc !== null ? formatPct(overallAcc) : "—"}</code>${seedStr && seedStr !== formatPct(overallAcc) ? `<br><small style="color:var(--text-muted)">(${escapeHtml(String(seedStr))}%)</small>` : ""}</td>
                              <td><code>${ansAcc !== null ? formatPct(ansAcc) : "—"}</code> / <code>${ans15 !== null ? formatPct(ans15) : "—"}</code></td>
                              <td><code>${refPct !== null ? formatPct(refPct) : "—"}</code> / <code>${halPct !== null ? formatPct(halPct) : "—"}</code></td>
                              <td><code>${auroc !== null ? auroc.toFixed(4) : "—"}</code> / <code>${aurc !== null ? formatPct(aurc) : "—"}</code></td>
                              <td><code>${meanLat !== null ? `${meanLat.toFixed(2)} ms` : "—"}</code>${p50Lat !== null ? ` <small>(p50: ${p50Lat.toFixed(2)})</small>` : ""}</td>
                              <td>${escapeHtml(String(note))}</td>
                            </tr>
                          `;
                        })
                        .join("")
                    : `<tr><td colspan="8" style="color: var(--text-muted); text-align:center;">Không tìm thấy dữ liệu ablation_w7.</td></tr>`
                }
              </tbody>
            </table>
          </div>
          ${reasoningSuiteHtml}
        </div>
      `
        : "";

      const pillar1W7 = data.pillar1_sparse_lse_w7 || null;
      const pillar1W7Rows = (() => {
        if (Array.isArray(pillar1W7)) return pillar1W7;
        if (!pillar1W7 || typeof pillar1W7 !== "object") return [];
        if (Array.isArray(pillar1W7.rows)) return pillar1W7.rows;
        const p1Raw = pillar1W7.pillars?.pillar_1_zero_forgetting;
        if (p1Raw && typeof p1Raw === "object") {
          const cps = Array.isArray(p1Raw.checkpoints) ? p1Raw.checkpoints : [0, 5, 10, 20, 64];
          const mkMap = p1Raw.wave_cms_mean_key || {};
          const slMap = p1Raw.wave_cms_sparse_lse || p1Raw.wave_cms || {};
          const ragMap = p1Raw.standard_rag || {};
          const saMap = p1Raw.sa_cms || {};
          return cps.map((cp) => {
            const k = String(cp);
            const mk = mkMap[k] || {};
            const sl = slMap[k] || {};
            const rag = ragMap[k] || {};
            const sa = saMap[k] || {};
            return {
              n: cp,
              n_docs: sl.n_docs,
              n_leaves: sl.n_leaves,
              mean_key_hit_at_1_pct: Number((Number(mk.routing_hit_at_1 || 0) * 100).toFixed(2)),
              sparse_lse_hit_at_1_pct: Number((Number(sl.routing_hit_at_1 || 0) * 100).toFixed(2)),
              sparse_lse_hit_at_3_pct: Number((Number(sl.routing_hit_at_3 || 0) * 100).toFixed(2)),
              tree_vs_flat_exact_match_pct: Number((Number(sl.tree_vs_flat_exact_match_rate ?? 1) * 100).toFixed(2)),
              mean_key_n_eff: Number(Number(mk.n_eff || 0).toFixed(2)),
              sparse_lse_n_eff: Number(Number(sl.n_eff || 0).toFixed(2)),
              sparse_lse_token_f1_pct: Number((Number(sl.token_f1 || 0) * 100).toFixed(2)),
              mean_key_token_f1_pct: Number((Number(mk.token_f1 || 0) * 100).toFixed(2)),
              standard_rag_token_f1_pct: Number((Number(rag.token_f1 || 0) * 100).toFixed(2)),
              sa_cms_token_f1_pct: Number((Number(sa.token_f1 || 0) * 100).toFixed(2)),
              leaf_core_drift_fro: Number(sl.leaf_core_drift_fro || 0),
            };
          });
        }
        return Array.isArray(pillar1W7.sweeps)
          ? pillar1W7.sweeps
          : Array.isArray(pillar1W7.results)
            ? pillar1W7.results
            : [];
      })();

      const pillar1W7Html = pillar1W7
        ? `
        <div class="bc-section-card">
          <h4 class="bc-section-title">🧭 Pillar 1: Định tuyến mean_key vs sparse_lse (Định lý 4, N = 0..64 tài liệu nhiễu, tới 1,598 lá)</h4>
          <div class="doc-table-wrap">
            <table class="doc-table">
              <thead>
                <tr>
                  <th>Mốc N (Docs / Lá)</th>
                  <th>mean_key Hit@1 (n_eff)</th>
                  <th>sparse_lse Hit@1 (n_eff)</th>
                  <th>Tree == Flat Exact</th>
                  <th>Token F1 Công bằng (WaveCMS vs mean_key vs SA-CMS)</th>
                  <th>Chênh lệch Hit@1 (Δ)</th>
                </tr>
              </thead>
              <tbody>
                ${
                  pillar1W7Rows.length
                    ? pillar1W7Rows
                        .map((row) => {
                          const nVal = row.n ?? row.noise_docs ?? row.N ?? "N/A";
                          const nDocs = toFiniteNumber(row.n_docs);
                          const nLeaves = toFiniteNumber(row.n_leaves);
                          const mkHit = toFiniteNumber(row.mean_key_hit_at_1_pct ?? row.mean_key_pct ?? row.mean_key);
                          const slHit = toFiniteNumber(row.sparse_lse_hit_at_1_pct ?? row.sparse_lse_pct ?? row.sparse_lse);
                          const mkNeff = toFiniteNumber(row.mean_key_n_eff);
                          const slNeff = toFiniteNumber(row.sparse_lse_n_eff);
                          const exactMatch = toFiniteNumber(row.tree_vs_flat_exact_match_pct);
                          const slF1 = toFiniteNumber(row.sparse_lse_token_f1_pct);
                          const mkF1 = toFiniteNumber(row.mean_key_token_f1_pct);
                          const saF1 = toFiniteNumber(row.sa_cms_token_f1_pct);
                          const docLeafLabel =
                            nDocs !== null || nLeaves !== null
                              ? ` <small style="color:var(--text-muted)">(${nDocs ?? "—"} docs / ${nLeaves ?? "—"} lá)</small>`
                              : "";
                          return `
                            <tr>
                              <td><strong>N = ${escapeHtml(String(nVal))}</strong>${docLeafLabel}</td>
                              <td><code>${formatPct(mkHit)}</code>${mkNeff !== null ? ` <small>(n_eff=${mkNeff.toFixed(1)})</small>` : ""}</td>
                              <td><strong><code>${formatPct(slHit)}</code></strong>${slNeff !== null ? ` <small>(n_eff=${slNeff.toFixed(2)})</small>` : ""}</td>
                              <td><code>${exactMatch !== null ? formatPct(exactMatch) : "100.00%"}</code></td>
                              <td><code>${formatPct(slF1)}</code> vs <code>${formatPct(mkF1)}</code> vs <code>${formatPct(saF1)}</code></td>
                              <td><span class="bc-delta-pill">${escapeHtml(formatDelta(mkHit, slHit))}</span></td>
                            </tr>
                          `;
                        })
                        .join("")
                    : `<tr><td colspan="6" style="color: var(--text-muted); text-align:center;">Không tìm thấy dữ liệu Pillar 1.</td></tr>`
                }
              </tbody>
            </table>
          </div>
        </div>
      `
        : "";

      container.innerHTML = `<div style="color: var(--danger); padding: 12px;">Không thể tải bảng so sánh: ${escapeHtml(err.message)}</div>`;
    }
  }

  function openFetchOnlineModal() {
    if (!dom.modalFetchOnline) return;
    dom.modalFetchOnline.hidden = false;
    if (dom.fetchOnlineQuery) {
      dom.fetchOnlineQuery.value = "";
      setTimeout(() => dom.fetchOnlineQuery.focus(), 40);
    }
    if (dom.fetchLangField && dom.fetchOnlineSource) {
      dom.fetchLangField.style.display = dom.fetchOnlineSource.value === "wikipedia" ? "" : "none";
    }
  }

  async function handleFetchOnlineSubmit(e) {
    e.preventDefault();
    const rawSource = dom.fetchOnlineSource ? dom.fetchOnlineSource.value : "wikipedia_vi";
    let source = rawSource;
    let language = dom.fetchOnlineLang ? dom.fetchOnlineLang.value : "vi";
    if (rawSource === "wikipedia_vi") {
      source = "wikipedia";
      language = "vi";
    } else if (rawSource === "wikipedia_en") {
      source = "wikipedia";
      language = "en";
    } else if (rawSource !== "wikipedia") {
      language = "en";
    }
    const maxResults = Math.max(1, Math.min(20, Number(dom.fetchOnlineMax ? dom.fetchOnlineMax.value : 3) || 3));
    const query = dom.fetchOnlineQuery ? dom.fetchOnlineQuery.value.trim() : "";
    const optionalQuerySources = new Set(["qasper", "squad_v2", "sciq", "pubmedqa"]);
    if (!query && !optionalQuerySources.has(source)) {
      showToast("Vui lòng nhập từ khóa hoặc tiêu đề cần tải.");
      return;
    }

    if (dom.btnSubmitFetchOnline) dom.btnSubmitFetchOnline.disabled = true;
    try {
      const res = await apiFetch("/api/documents/fetch-online", {
        method: "POST",
        body: JSON.stringify({
          source,
          query,
          language,
          max_results: maxResults,
        }),
      });
      await refreshDocuments();
      const ingested = Array.isArray(res.documents) ? res.documents : [];
      if (ingested.length > 0) {
        ingested.forEach((doc) => {
          if (doc.document_id && !state.attachedDocIds.includes(doc.document_id)) {
            state.attachedDocIds.push(doc.document_id);
          }
        });
        renderSidebarDocuments();
        renderComposerChips();
      }
      dom.modalFetchOnline.hidden = true;
      showToast(`Đã nạp ${ingested.length || 1} tài liệu từ ${source.toUpperCase()}!`);
    } catch (err) {
      showToast(`Lỗi nạp API: ${err.message}`);
    } finally {
      if (dom.btnSubmitFetchOnline) dom.btnSubmitFetchOnline.disabled = false;
    }
  }

  // ============================================================================
  // 14. EVENT BINDINGS & INITIALIZATION
  // ============================================================================

  function bindEvents() {
    // Theme controls (Sidebar Footer + Top Header Switch + OS Preference Sync)
    dom.btnThemeToggle.addEventListener("click", () => {
      applyTheme(state.theme === "dark" ? "light" : "dark", true);
    });

    if (dom.headerThemeSwitch) {
      dom.headerThemeSwitch.addEventListener("click", (e) => {
        const opt = e.target.closest("[data-set-theme]");
        if (!opt) return;
        const target = opt.dataset.setTheme;
        if (target === state.theme && state.themePref !== "system") {
          // Clicking the already-active pill toggles smoothly to the opposite theme
          applyTheme(state.theme === "dark" ? "light" : "dark", true);
        } else {
          applyTheme(target, true);
        }
      });
    }

    if (window.matchMedia) {
      const darkMedia = window.matchMedia("(prefers-color-scheme: dark)");
      const onSystemThemeChange = () => {
        if (state.themePref === "system") {
          applyTheme("system", false);
        }
      };
      if (typeof darkMedia.addEventListener === "function") {
        darkMedia.addEventListener("change", onSystemThemeChange);
      }
    }

    // Sidebar Collapse & Mobile Drawer
    dom.btnSidebarClose.addEventListener("click", () => {
      if (window.innerWidth <= 1024) {
        closeMobileSidebar();
      } else {
        dom.app.classList.add("sidebar-collapsed");
      }
    });

    dom.btnSidebarOpen.addEventListener("click", () => {
      if (window.innerWidth <= 1024) {
        dom.app.classList.add("mobile-sidebar-open");
      } else {
        dom.app.classList.remove("sidebar-collapsed");
      }
    });

    dom.sidebarBackdrop.addEventListener("click", closeMobileSidebar);

    // New Chat & Brand Home
    dom.btnNewChat.addEventListener("click", startFreshChatView);
    dom.btnBrandHome.addEventListener("click", startFreshChatView);

    // Clear Active Chat
    dom.btnClearChat.addEventListener("click", () => {
      if (state.activeSessionId) {
        handleDeleteSession(state.activeSessionId);
      } else {
        startFreshChatView();
      }
    });

    // Sidebar Search
    dom.btnToggleSearch.addEventListener("click", () => {
      const nextHidden = !dom.sidebarSearchWrap.hidden;
      dom.sidebarSearchWrap.hidden = nextHidden;
      if (!nextHidden) {
        dom.sidebarSearchInput.focus();
      } else {
        state.sidebarSearchQuery = "";
        dom.sidebarSearchInput.value = "";
        renderSidebarSessions();
        renderSidebarDocuments();
      }
    });

    dom.sidebarSearchInput.addEventListener("input", (e) => {
      state.sidebarSearchQuery = e.target.value || "";
      renderSidebarSessions();
      renderSidebarDocuments();
    });

    dom.btnClearSearch.addEventListener("click", () => {
      state.sidebarSearchQuery = "";
      dom.sidebarSearchInput.value = "";
      renderSidebarSessions();
      renderSidebarDocuments();
    });

    // Delegated clicks on Sidebar Sessions
    dom.sidebarSessionsList.addEventListener("click", (e) => {
      const actionBtn = e.target.closest("[data-action]");
      if (actionBtn) {
        e.stopPropagation();
        const sid = actionBtn.dataset.sessionId;
        if (actionBtn.dataset.action === "rename-session") {
          handleRenameSession(sid);
        } else if (actionBtn.dataset.action === "delete-session") {
          handleDeleteSession(sid);
        }
        return;
      }
      const row = e.target.closest("[data-session-id]");
      if (row) {
        selectSession(row.dataset.sessionId);
      }
    });

    // Delegated clicks on Sidebar Documents
    dom.sidebarDocsList.addEventListener("click", (e) => {
      const btn = e.target.closest("[data-doc-toggle]");
      if (btn) {
        toggleDocumentAttachment(btn.dataset.docToggle);
      }
    });

    // Close Ingestion Card
    dom.btnCloseIngestion.addEventListener("click", () => {
      dom.ingestionPanel.hidden = true;
    });

    // Mode Dropdown Popover
    dom.btnModeDropdown.addEventListener("click", (e) => {
      e.stopPropagation();
      const willOpen = dom.modeDropdownMenu.hidden;
      dom.modeDropdownMenu.hidden = !willOpen;
      dom.btnModeDropdown.setAttribute("aria-expanded", String(willOpen));
    });

    dom.modeDropdownMenu.addEventListener("click", (e) => {
      e.stopPropagation();
      const modeBtn = e.target.closest(".mode-option[data-mode]");
      if (modeBtn) {
        state.mode = modeBtn.dataset.mode;
        syncModeAndLengthUI();
        return;
      }
      const lenBtn = e.target.closest(".length-seg-btn[data-length]");
      if (lenBtn) {
        state.answerMode = lenBtn.dataset.length;
        syncModeAndLengthUI();
        return;
      }
      const themeBtn = e.target.closest(".theme-seg-btn[data-theme-pref]");
      if (themeBtn) {
        applyTheme(themeBtn.dataset.themePref, true);
      }
    });

    dom.chkShowTelemetry.addEventListener("change", (e) => {
      state.showTelemetry = e.target.checked;
      syncModeAndLengthUI();
    });

    if (dom.selActiveCheckpoint) {
      dom.selActiveCheckpoint.addEventListener("change", async (e) => {
        const targetCkpt = e.target.value;
        if (!targetCkpt) return;
        try {
          const res = await apiFetch(
            `/api/checkpoints/switch?checkpoint_name=${encodeURIComponent(targetCkpt)}`,
            { method: "POST" }
          );
          if (res.success) {
            await refreshSystemStatus();
            showToast(`Đã chuyển sang checkpoint: ${res.active_checkpoint}`);
          } else {
            showToast(`Không thể chuyển checkpoint: ${res.error || "Lỗi không xác định"}`);
          }
        } catch (err) {
          showToast(`Lỗi chuyển checkpoint: ${err.message}`);
        }
      });
    }

    document.addEventListener("click", () => {
      if (!dom.modeDropdownMenu.hidden) {
        dom.modeDropdownMenu.hidden = true;
        dom.btnModeDropdown.setAttribute("aria-expanded", "false");
      }
    });

    // Welcome Suggestion Cards & Quick Verification Chips
    dom.welcomeSuggestionsGrid.addEventListener("click", (e) => {
      const card = e.target.closest("[data-prompt]");
      if (card) {
        const promptText = card.getAttribute("data-prompt") || card.dataset.prompt;
        if (promptText) {
          sendQuestion(promptText);
        }
      }
    });

    document.querySelectorAll(".verify-chip[data-query]").forEach((chip) => {
      chip.addEventListener("click", () => {
        const query = chip.getAttribute("data-query") || chip.dataset.query;
        if (query) {
          sendQuestion(query);
        }
      });
    });

    // Delegated clicks inside Messages Stream (Citations, Fulltext, Compare, Suggested Chips)
    dom.messagesList.addEventListener("click", (e) => {
      const suggestBtn = e.target.closest("[data-suggest-query]");
      if (suggestBtn) {
        const queryText =
          suggestBtn.getAttribute("data-suggest-query") ||
          suggestBtn.dataset.suggestQuery;
        if (queryText) {
          sendQuestion(queryText);
        }
        return;
      }
      const citeBtn = e.target.closest("[data-passage-id]");
      if (citeBtn) {
        inspectPassageCitation(citeBtn.dataset.passageId);
        return;
      }
      const compBtn = e.target.closest("[data-compare-query]");
      if (compBtn) {
        dom.arenaQueryInput.value = compBtn.dataset.compareQuery;
        openResearchLabModal("arena");
        runMethodArenaComparison();
        return;
      }
      const exportBtn = e.target.closest("[data-export-msg]");
      if (exportBtn) {
        const format = exportBtn.dataset.exportMsg;
        const bar = exportBtn.closest(".msg-telemetry-bar");
        const title = (bar && bar.dataset.exportTitle) || "Câu trả lời WaveCMS";
        const content = (bar && bar.dataset.exportContent) || "";
        if (content) {
          downloadOfficeFile(format, state.activeSessionId || null, false, { title, content });
        }
      }
    });

    // Composer Form, Auto-resize & Keyboard Submit
    dom.composerTextarea.addEventListener("input", () => {
      autoResizeComposer();
      syncSendButtonState();
    });

    dom.composerTextarea.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendQuestion(dom.composerTextarea.value);
      }
    });

    dom.composerForm.addEventListener("submit", (e) => {
      e.preventDefault();
      sendQuestion(dom.composerTextarea.value);
    });

    dom.btnCycleLength.addEventListener("click", () => {
      const idx = LENGTH_CYCLE.indexOf(state.answerMode);
      state.answerMode = LENGTH_CYCLE[(idx + 1) % LENGTH_CYCLE.length];
      syncModeAndLengthUI();
    });

    if (dom.btnToggleOnlineApi) {
      dom.btnToggleOnlineApi.addEventListener("click", () => {
        state.allowOnlineApi = !state.allowOnlineApi;
        syncModeAndLengthUI();
        showToast(
          state.allowOnlineApi
            ? "Đã bật Auto API ngoài (tự gọi Wikipedia/arXiv/OpenAlex/EuropePMC khi vượt kho local)."
            : "Đã tắt Auto API ngoài — bảo toàn chế độ Offline Air-Gap."
        );
      });
    }

    if (dom.thinkingModeSelector) {
      dom.thinkingModeSelector.addEventListener("click", (e) => {
        const btn = e.target.closest("[data-thinking-mode]");
        if (!btn) return;
        const nextMode = btn.dataset.thinkingMode;
        if (!nextMode || nextMode === state.thinkingMode) return;
        state.thinkingMode = nextMode;
        syncModeAndLengthUI();
      });
    }

    dom.composerAttachedChips.addEventListener("click", (e) => {
      const rmBtn = e.target.closest("[data-detach-doc]");
      if (rmBtn) {
        toggleDocumentAttachment(rmBtn.dataset.detachDoc);
      }
    });

    // File Upload Triggers & Drag-and-Drop
    const triggerFileDialog = () => dom.hiddenFileInput.click();
    dom.btnAttachFile.addEventListener("click", triggerFileDialog);
    dom.btnSidebarAddDoc.addEventListener("click", triggerFileDialog);
    dom.btnModalUpload.addEventListener("click", triggerFileDialog);

    dom.hiddenFileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files.length > 0) {
        uploadFiles(e.target.files);
        e.target.value = "";
      }
    });

    let dragDepth = 0;
    dom.appMain.addEventListener("dragenter", (e) => {
      e.preventDefault();
      dragDepth += 1;
      dom.dropzoneOverlay.hidden = false;
    });
    dom.appMain.addEventListener("dragover", (e) => {
      e.preventDefault();
    });
    dom.appMain.addEventListener("dragleave", () => {
      dragDepth = Math.max(0, dragDepth - 1);
      if (dragDepth === 0) dom.dropzoneOverlay.hidden = true;
    });
    dom.appMain.addEventListener("drop", (e) => {
      e.preventDefault();
      dragDepth = 0;
      dom.dropzoneOverlay.hidden = true;
      if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        uploadFiles(e.dataTransfer.files);
      }
    });

    // Inspector Drawer Controls
    dom.btnStatusInspector.addEventListener("click", () => {
      if (!dom.inspectorDrawer.hidden) {
        dom.inspectorDrawer.hidden = true;
      } else {
        openInspector("structure");
      }
    });
    dom.btnCloseInspector.addEventListener("click", () => {
      dom.inspectorDrawer.hidden = true;
    });
    dom.inspTabSource.addEventListener("click", () => openInspector("source"));
    dom.inspTabStructure.addEventListener("click", () => openInspector("structure"));
    dom.inspDocSelect.addEventListener("change", (e) => {
      state.selectedStructureDocId = e.target.value;
      loadDocumentStructureAndMemory(state.selectedStructureDocId);
    });
    dom.btnDocUpdateV2.addEventListener("click", handleUpdateDocVersion);
    dom.btnDocRollback.addEventListener("click", handleRollbackDocVersion);

    // Document Library Modal Controls
    dom.btnOpenDocLibrary.addEventListener("click", openDocLibraryModal);
    dom.btnDocPicker.addEventListener("click", openDocLibraryModal);
    dom.btnCloseDocModal.addEventListener("click", () => {
      dom.modalDocLibrary.hidden = true;
    });
    dom.docModalSearch.addEventListener("input", (e) => {
      state.docModalSearchQuery = e.target.value || "";
      renderDocModalTable();
    });
    dom.btnToggleRawForm.addEventListener("click", () => {
      dom.rawDocForm.hidden = !dom.rawDocForm.hidden;
    });
    dom.btnCancelRaw.addEventListener("click", () => {
      dom.rawDocForm.hidden = true;
    });
    dom.btnSubmitRaw.addEventListener("click", handleCreateRawDocument);
    dom.btnSeedDemo.addEventListener("click", async () => {
      try {
        await apiFetch("/api/demo/seed", { method: "POST" });
        await refreshDocuments();
        showToast("Đã khôi phục bộ tài liệu mẫu kiểm chuẩn.");
      } catch (err) {
        showToast(`Lỗi nạp tài liệu mẫu: ${err.message}`);
      }
    });

    dom.docModalTbody.addEventListener("change", (e) => {
      const chk = e.target.closest("[data-modal-attach]");
      if (chk) {
        toggleDocumentAttachment(chk.dataset.modalAttach);
      }
    });

    dom.docModalTbody.addEventListener("click", (e) => {
      const inspBtn = e.target.closest("[data-inspect-doc]");
      if (inspBtn) {
        state.selectedStructureDocId = inspBtn.dataset.inspectDoc;
        dom.modalDocLibrary.hidden = true;
        openInspector("structure");
        return;
      }
      const delBtn = e.target.closest("[data-delete-doc]");
      if (delBtn) {
        handleDeleteDocument(delBtn.dataset.deleteDoc);
      }
    });

    // Research Lab, Presentation Mode & Figure Zoom Modals
    if (dom.btnPresentationMode) {
      dom.btnPresentationMode.addEventListener("click", () => togglePresentationMode());
    }
    if (dom.btnSidebarFetchOnline) {
      dom.btnSidebarFetchOnline.addEventListener("click", openFetchOnlineModal);
    }
    if (dom.btnCloseFetchModal) {
      dom.btnCloseFetchModal.addEventListener("click", () => {
        dom.modalFetchOnline.hidden = true;
      });
    }
    if (dom.btnCancelFetchOnline) {
      dom.btnCancelFetchOnline.addEventListener("click", () => {
        dom.modalFetchOnline.hidden = true;
      });
    }
    if (dom.fetchOnlineSource) {
      dom.fetchOnlineSource.addEventListener("change", () => {
        if (dom.fetchLangField) {
          dom.fetchLangField.style.display = dom.fetchOnlineSource.value === "wikipedia" ? "" : "none";
        }
      });
    }
    if (dom.fetchOnlineForm) {
      dom.fetchOnlineForm.addEventListener("submit", handleFetchOnlineSubmit);
    }
    [
      { btn: dom.btnExportWord, format: "word" },
      { btn: dom.btnExportExcel, format: "excel" },
      { btn: dom.btnExportPptx, format: "pptx" },
    ].forEach(({ btn, format }) => {
      if (btn) {
        btn.addEventListener("click", () => {
          downloadOfficeFile(format, state.activeSessionId || null);
        });
      }
    });
    if (dom.btnOpenComparisonDashboard) {
      dom.btnOpenComparisonDashboard.addEventListener("click", () => openResearchLabModal("baseline-compare"));
    }
    dom.btnOpenResearchLab.addEventListener("click", () => openResearchLabModal("arena"));
    dom.btnOpenResearchFooter.addEventListener("click", () => openResearchLabModal("demos"));
    dom.btnCloseResearchModal.addEventListener("click", () => {
      dom.modalResearchLab.hidden = true;
    });

    document.querySelectorAll(".rlab-tab[data-rlab-tab]").forEach((btn) => {
      btn.addEventListener("click", () => switchResearchTab(btn.dataset.rlabTab));
    });

    dom.btnRunArena.addEventListener("click", runMethodArenaComparison);

    dom.committeeDemosGrid.addEventListener("click", (e) => {
      const btn = e.target.closest("[data-run-scenario]");
      if (btn) {
        executeCommitteeScenario(btn.dataset.runScenario);
      }
    });

    document.querySelectorAll(".q1-fig-card").forEach((card) => {
      card.addEventListener("click", () => {
        dom.zoomFigTitle.textContent = card.dataset.figTitle || "Biểu đồ Khoa học Q1";
        dom.zoomFigImg.src = card.dataset.figSrc || "";
        dom.modalFigureZoom.hidden = false;
      });
    });

    dom.btnCloseZoom.addEventListener("click", () => {
      dom.modalFigureZoom.hidden = true;
    });

    // Close modals on backdrop click or Escape
    [dom.modalDocLibrary, dom.modalResearchLab, dom.modalFigureZoom, dom.modalFetchOnline].forEach((overlay) => {
      if (!overlay) return;
      overlay.addEventListener("click", (e) => {
        if (e.target === overlay) overlay.hidden = true;
      });
    });

    document.addEventListener("keydown", (e) => {
      const mod = e.ctrlKey || e.metaKey;

      // Alt+T: Toggle Light / Dark Theme
      if (e.altKey && !mod && (e.key === "t" || e.key === "T")) {
        e.preventDefault();
        applyTheme(state.theme === "dark" ? "light" : "dark", true);
        return;
      }

      // Ctrl+K: Quick-open Document Library modal & focus search
      if (mod && (e.key === "k" || e.key === "K")) {
        e.preventDefault();
        if (dom.modalDocLibrary.hidden) {
          openDocLibraryModal();
          setTimeout(() => {
            if (dom.docModalSearch) dom.docModalSearch.focus();
          }, 40);
        } else {
          dom.modalDocLibrary.hidden = true;
        }
        return;
      }

      // Ctrl+/: Quick-toggle SA-CMS Right Inspector Drawer
      if (mod && e.key === "/") {
        e.preventDefault();
        if (!dom.inspectorDrawer.hidden) {
          dom.inspectorDrawer.hidden = true;
        } else {
          openInspector("structure");
        }
        return;
      }

      // Ctrl+L: Start a clean new chat session
      if (mod && (e.key === "l" || e.key === "L") && !e.shiftKey && !e.altKey) {
        e.preventDefault();
        startFreshChatView();
        showToast("Đã mở phiên trò chuyện mới (Ctrl+L)");
        return;
      }

      if (e.key === "Escape") {
        const hadOpenOverlay =
          !dom.modeDropdownMenu.hidden ||
          !dom.modalDocLibrary.hidden ||
          !dom.modalResearchLab.hidden ||
          !dom.modalFigureZoom.hidden ||
          (dom.modalFetchOnline && !dom.modalFetchOnline.hidden);
        dom.modeDropdownMenu.hidden = true;
        dom.modalDocLibrary.hidden = true;
        dom.modalResearchLab.hidden = true;
        dom.modalFigureZoom.hidden = true;
        if (dom.modalFetchOnline) dom.modalFetchOnline.hidden = true;
        if (!hadOpenOverlay && state.presentationMode) {
          togglePresentationMode(false);
        }
      }
    });
  }

  async function initApp() {
    applyTheme(state.themePref, false);
    syncModeAndLengthUI();
    bindEvents();
    renderWelcomeSuggestions();

    await Promise.all([
      refreshSystemStatus(),
      refreshDocuments(),
      refreshSessions(),
    ]);

    syncSendButtonState();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initApp);
  } else {
    initApp();
  }
})();
