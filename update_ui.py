import os

def update_file(path, replacements):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for old, new in replacements:
        if old not in content:
            print(f"Failed to find target chunk in {path}:\n{old[:50]}...")
        content = content.replace(old, new)
        
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

# Update CSS
old_css = """.composer-attachments {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 0.6rem;
  background-color: var(--bg-input);
  border: 1px solid var(--border-subtle);
  border-bottom: none;
  border-radius: var(--radius-lg) var(--radius-lg) 0 0;
  flex-wrap: wrap;
  min-height: 44px;
  transition: all var(--transition-fast);
}"""

new_css = """.composer-attachments {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0 0.2rem 0.5rem 0.2rem;
  background-color: transparent;
  border: none;
  flex-wrap: wrap;
  transition: all var(--transition-fast);
}"""

old_css2 = """/* Composer có attachment → input-form bo tròn chỉ phía dưới */
.composer-attachments + .input-form {
  border-radius: 0 0 var(--radius-lg) var(--radius-lg);
  border-top: none;
}"""

new_css2 = "/* Removed .composer-attachments + .input-form style */"

update_file('src/web/static/css/app.css', [(old_css, new_css), (old_css2, new_css2)])

# Update app.js (Remove attachedTagsList)
old_js1 = """function renderAttachedTags(docs) {
  el.attachedTagsList.innerHTML = "";
  if (!docs || docs.length === 0) {
    el.inputDocIndicator.innerHTML = '<span class="doc-badge-pill">● Toàn bộ tài liệu</span>';
    return;
  }

  el.inputDocIndicator.innerHTML = `<span class="doc-badge-pill">● ${docs.length} tài liệu đang chọn</span>`;

  docs.forEach((d) => {
    const tag = document.createElement("div");
    tag.className = "doc-tag";
    const displayName = getDocumentDisplayName(d);
    const sourceName = getDocumentSourceName(d);
    tag.innerHTML = `
      <span>📄</span>
      <span class="doc-tag-name" title="${escapeHtml(sourceName)}">${escapeHtml(displayName)}</span>
      <button class="doc-tag-remove" title="Bỏ tài liệu khỏi cuộc trò chuyện">✕</button>
    `;

    tag.querySelector(".doc-tag-remove").addEventListener("click", () => {
      detachDocumentFromSession(d.document_id);
    });

    el.attachedTagsList.appendChild(tag);
  });
}"""

new_js1 = """function renderAttachedTags(docs) {
  if (el.inputDocIndicator) {
    if (!docs || docs.length === 0) {
      el.inputDocIndicator.innerHTML = '<span class="doc-badge-pill">● Toàn bộ tài liệu</span>';
    } else {
      el.inputDocIndicator.innerHTML = `<span class="doc-badge-pill">● ${docs.length} tài liệu đang chọn</span>`;
    }
  }
}"""

# Update addAttachedDocument
old_js2 = """function addAttachedDocument(doc) {
  const exists = state.attachedDocuments.some((d) => d.document_id === doc.document_id);
  if (!exists) {
    state.attachedDocuments.push(doc);
    renderAttachedTags(state.attachedDocuments);
  }
}"""

new_js2 = """function addAttachedDocument(doc) {
  const exists = state.attachedDocuments.some((d) => d.document_id === doc.document_id);
  if (!exists) {
    state.attachedDocuments.push(doc);
    renderAttachedTags(state.attachedDocuments);
    renderComposerAttachmentsFromState();
  }
}"""

# Update chat history
old_js3 = """function renderMessagesList(messages) {
  el.messagesList.innerHTML = "";
  messages.forEach((m) => {
    if (isDocumentHistoryEvent(m)) {
      renderDocumentHistoryDivider(m);
    } else if (m.role === "user") {
      renderUserMessage(m.content);
    } else if (m.role === "assistant") {
      renderAssistantMessage(m, m.diagnostics);
    }
  });
  scrollToBottom();
}"""

new_js3 = """function renderMessagesList(messages) {
  el.messagesList.innerHTML = "";
  let hasSplit = messages.some(isDocumentHistoryEvent);
  let beforeSplitRendered = false;
  
  messages.forEach((m) => {
    if (hasSplit && !beforeSplitRendered && !isDocumentHistoryEvent(m)) {
      renderHistoryGroupHeader("Lịch sử phiên trước", m.timestamp);
      beforeSplitRendered = true;
    }
    if (isDocumentHistoryEvent(m)) {
      renderHistoryGroupHeader("Lịch sử phiên hiện tại", m.timestamp);
    } else if (m.role === "user") {
      renderUserMessage(m.content);
    } else if (m.role === "assistant") {
      renderAssistantMessage(m, m.diagnostics);
    }
  });
  scrollToBottom();
}"""

old_js4 = """function renderDocumentHistoryDivider(msg) {
  if (!el.messagesList) return;

  const meta = msg.metadata || {};
  const title = meta.title || msg.title || buildDocumentsSummary(state.attachedDocuments);
  const timeText = formatEventTime(msg.timestamp);
  const divider = document.createElement("div");
  divider.className = "message-history-divider";
  divider.innerHTML = `
    <div class="history-divider-line"></div>
    <div class="history-divider-chip">
      <span class="history-divider-label">Lịch sử sau khi thêm tài liệu</span>
      <span class="history-divider-title">${escapeHtml(title)}</span>
      ${timeText ? `<span class="history-divider-time">${escapeHtml(timeText)}</span>` : ""}
    </div>
    <div class="history-divider-line"></div>
  `;
  el.messagesList.appendChild(divider);
}"""

new_js4 = """function renderDocumentHistoryDivider(msg) {
  renderHistoryGroupHeader("Lịch sử phiên hiện tại", msg.timestamp);
}

function renderHistoryGroupHeader(title, timestamp) {
  if (!el.messagesList) return;
  const timeText = timestamp ? formatEventTime(timestamp) : formatEventTime(Date.now());
  const divider = document.createElement("div");
  divider.className = "message-history-divider";
  divider.innerHTML = `
    <div class="history-divider-line"></div>
    <div class="history-divider-chip">
      <span class="history-divider-label">${escapeHtml(title)}</span>
      ${timeText ? `<span class="history-divider-time">${escapeHtml(timeText)}</span>` : ""}
    </div>
    <div class="history-divider-line"></div>
  `;
  el.messagesList.appendChild(divider);
}"""

old_js5 = """    // Cập nhật composer chip - phân biệt rõ SA-CMS Memory đã sẵn sàng với File đã tải
    updateComposerChipStatus(chipId, "SA-CMS Memory đã sẵn sàng", "ready");

    // Add to attached documents
    const docInfo = {
      document_id: data.document_id,
      title: data.title,
      file_name: data.file_name,
      file_size: formatBytes(data.file_size),
      _chipId: chipId,
    };
    addAttachedDocument(docInfo);

    // Keep the "ready" state visible in composer for a moment before clearing
    setTimeout(() => {
      // Remove chip from composer (move to attached docs)
      const chip = document.getElementById(chipId);
      if (chip) chip.remove();
      addAttachedDocument(data.document);
      showWelcomeScreen(); // This triggers renderDocumentHistoryDivider
    }, 1500);"""

new_js5 = """    // Cập nhật composer chip - phân biệt rõ SA-CMS Memory đã sẵn sàng với File đã tải
    updateComposerChipStatus(chipId, "SA-CMS Memory đã sẵn sàng", "ready");

    // Add to attached documents
    const docInfo = {
      document_id: data.document_id,
      title: data.title,
      file_name: data.file_name,
      file_size: formatBytes(data.file_size),
      _chipId: chipId,
    };
    addAttachedDocument(docInfo);

    // Keep the "ready" state visible in composer for a moment before clearing
    setTimeout(() => {
      updateComposerChipStatus(chipId, "Đã đính kèm", "ready");
      showWelcomeScreen(); // This triggers renderDocumentHistoryDivider
    }, 1500);"""

update_file('src/web/static/js/app.js', [
  (old_js1, new_js1),
  (old_js2, new_js2),
  (old_js3, new_js3),
  (old_js4, new_js4),
  (old_js5, new_js5)
])

print("Finished updates.")
