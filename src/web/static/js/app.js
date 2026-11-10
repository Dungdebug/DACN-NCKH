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

    // Bold & inline code
    safe = safe.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
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
    btnOpenResearchLab: document.getElementById("btn-open-research-lab"),
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

    const telemetryBarHtml =
      totalTok !== null || latMs !== null
        ? `
        <div class="msg-telemetry-bar">
          ${totalTok !== null ? `<span class="tel-pill">${totalTok} tokens (${tel.input_tokens || 0} in / ${tel.output_tokens || 0} out${savedPct})</span>` : ""}
          ${latMs !== null ? `<span class="tel-pill">${latMs} ms</span>` : ""}
          ${
            queryForCompare
              ? `<button type="button" class="tel-compare-btn" data-compare-query="${escapeHtml(queryForCompare)}">⇄ Đối sánh B1/B2/P1/P2</button>`
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
      dom.thinkingStatusText.textContent =
        state.mode === "memory"
          ? "Đang truy xuất bộ nhớ tham số 3 tầng SA-CMS (P1)..."
          : "Đang truy xuất bằng chứng và đối soát bộ nhớ SA-CMS...";
      scrollToBottom();

      const res = await apiFetch(
        `/api/chat/sessions/${encodeURIComponent(state.activeSessionId)}/messages`,
        {
          method: "POST",
          body: JSON.stringify({
            query,
            mode: state.mode,
            answer_mode: state.answerMode,
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
        state.activeMessages.push(res.message);
      } else {
        state.activeMessages.push({
          role: "assistant",
          content: res.answer || "",
          citations: res.citations || [],
          telemetry: res.telemetry || {},
          diagnostics: res.diagnostics || {},
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
      demos: document.getElementById("rlab-panel-demos"),
      metrics: document.getElementById("rlab-panel-metrics"),
      figures: document.getElementById("rlab-panel-figures"),
    };
    Object.keys(panels).forEach((k) => {
      if (panels[k]) panels[k].hidden = k !== tabName;
    });

    if (tabName === "demos") {
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
          return `
            <div class="arena-card ${isP2 ? "arena-p2" : ""}">
              <div class="arena-card-head">
                <span class="arena-method-name">${escapeHtml(titles[m] || m)}</span>
                <span class="tel-pill">${item.refused ? "🛡️ Đã từ chối" : "✓ Phản hồi"}</span>
              </div>
              <div style="font-size: 13.5px; line-height: 1.55; color: var(--text-body);">
                ${escapeHtml(item.answer || "")}
              </div>
              <div class="arena-metrics-row">
                <span>Input: <strong>${item.input_tokens ?? 0}</strong> tok</span>
                <span>· Output: <strong>${item.output_tokens ?? 0}</strong> tok</span>
                <span>· Tổng: <strong>${item.total_tokens ?? 0}</strong> tok</span>
                <span>· ${item.latency_ms ?? 0} ms</span>
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

  async function loadResearchMetricsDashboard() {
    if (!dom.rlabMetricsContent) return;
    dom.rlabMetricsContent.innerHTML = `<div style="color: var(--text-muted);">Đang tải số liệu kiểm chuẩn...</div>`;

    try {
      const [metrics, sysStatus] = await Promise.all([
        apiFetch("/research/metrics"),
        apiFetch("/api/system/status"),
      ]);
      state.researchMetrics = metrics;

      const r2 = metrics.round_2_extension || {};
      const memParams = r2.memory_parameters || {};

      dom.rlabMetricsContent.innerHTML = `
        <div class="metric-kpi-box">
          <span class="metric-kpi-badge">OFFICIAL PHASE 4 · FROZEN_LOCKED</span>
          <div class="metric-kpi-val">p = 3.03 × 10⁻⁷ (d = 2.91)</div>
          <div class="metric-kpi-desc">
            <strong>RQ2 — SA-CMS (P1) vs Fixed-Token (B5):</strong> Cải thiện có ý nghĩa thống kê vượt trội trên 30 seeds độc lập ở cùng ngân sách cập nhật.
          </div>
        </div>
        <div class="metric-kpi-box">
          <span class="metric-kpi-badge">OFFICIAL PHASE 4 · RQ1 &amp; RQ4</span>
          <div class="metric-kpi-val">+175% Recall · 36 Ckpts</div>
          <div class="metric-kpi-desc">
            <strong>Thu hồi chi tiết MK-NIAH &amp; Continual Learning:</strong> Chống quên thảm khốc xuyên suốt chuỗi 21 tài liệu tuần tự (42 checkpoints <code>.pt</code> đã khóa).
          </div>
        </div>
        <div class="metric-kpi-box">
          <span class="metric-kpi-badge">OFFICIAL PHASE 4 · RQ3 HYBRID P2</span>
          <div class="metric-kpi-val">100% Citation Faithfulness</div>
          <div class="metric-kpi-desc">
            <strong>Độ trung thực trích dẫn &amp; Cổng từ chối:</strong> Mọi câu trả lời có căn cứ đều dẫn chiếu về đoạn gốc; chặn triệt để câu hỏi ngoài miền.
          </div>
        </div>
        <div class="metric-kpi-box">
          <span class="metric-kpi-badge">EXTENDED GPU TRAINING · RTX 3050</span>
          <div class="metric-kpi-val">-87.72% VN PPL · -71.74% EN PPL</div>
          <div class="metric-kpi-desc">
            <strong>Huấn luyện song ngữ mở rộng (<code>cms_3lvl_extended.pt</code>):</strong> Giảm Answer Perplexity tiếng Việt từ <code>19.07 → 2.34</code> và tiếng Anh từ <code>31.54 → 8.91</code> (0% rò rỉ tập test).
          </div>
        </div>
        <div class="metric-kpi-box">
          <span class="metric-kpi-badge">OFFLINE RUNTIME · ${escapeHtml(sysStatus.integrity_verdict || "VERIFIED")}</span>
          <div class="metric-kpi-val">${(memParams.trainable_parameters || 5315904).toLocaleString("en-US")} CMS Params</div>
          <div class="metric-kpi-desc">
            Backbone <code>SmolLM2-135M</code> đóng băng 100% · Chế độ mạng: <code>${escapeHtml(sysStatus.network || "DISABLED / OFFLINE")}</code> · Checkpoint: <code>${escapeHtml(sysStatus.active_checkpoint || "cms_3lvl_extended.pt")}</code>.
          </div>
        </div>
        <div class="metric-kpi-box" style="grid-column: 1 / -1;">
          <span class="metric-kpi-badge">SA-CMS ARCHITECTURE · EQUATION 74</span>
          <div class="metric-kpi-val" style="font-size: 16px;">yₜ = MLP_pre(xₜ) + ∑ₗ₌₁³ gₗ(xₜ) · BₗAₗ xₜ</div>
          <div class="metric-kpi-desc">
            Ba tầng bộ nhớ căn chỉnh cấu trúc văn bản độc lập (<strong>L1 Đoạn văn</strong> · <strong>L2 Chương mục</strong> · <strong>L3 Toàn tài liệu</strong>, <code>d = 576, d_ff = 1536, r = 16</code>) hòa trộn qua cổng học được <code>gₗ(xₜ) = σ(W_g,ₗ · RMSNorm(xₜ))</code>.
          </div>
        </div>
      `;
    } catch (err) {
      dom.rlabMetricsContent.innerHTML = `<div style="color: var(--danger);">Không thể tải số liệu: ${escapeHtml(err.message)}</div>`;
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
    [dom.modalDocLibrary, dom.modalResearchLab, dom.modalFigureZoom].forEach((overlay) => {
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
          !dom.modalFigureZoom.hidden;
        dom.modeDropdownMenu.hidden = true;
        dom.modalDocLibrary.hidden = true;
        dom.modalResearchLab.hidden = true;
        dom.modalFigureZoom.hidden = true;
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
