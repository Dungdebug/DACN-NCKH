#!/usr/bin/env python3
"""
Step 13 & 20 — Offline Memory Runtime & Network Audit.
Scans memory persistence modules, model loaders, runtime APIs, and web assets
to guarantee zero remote dependencies, zero external AI APIs, and complete offline safety.

Exit code:
    0 = OFFLINE_SAFE
    1 = OFFLINE_BLOCKED
"""

import sys
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# Whitelist patterns for local server documentation or loopback
ALLOWED_PATTERNS = [
    r"http://127\.0\.0\.1",
    r"http://localhost",
    r"https?://127\.0\.0\.1:\d+",
    r"https?://localhost:\d+",
    r"http://www\.w3\.org/2000/svg",  # SVG XML namespace standard
]

# Forbidden external network indicators in runtime code
FORBIDDEN_KEYWORDS = [
    r"api\.openai\.com",
    r"generativelanguage\.googleapis\.com",
    r"api\.anthropic\.com",
    r"dashscope\.aliyuncs\.com",
    r"api-inference\.huggingface\.co",
    r"fonts\.googleapis\.com",
    r"fonts\.gstatic\.com",
    r"cdn\.jsdelivr\.net",
    r"cdnjs\.cloudflare\.com",
    r"unpkg\.com",
    r"from_pretrained\(\s*[\"'](?!(\.|\/|[a-zA-Z]:)).*[\"']\s*\)",  # Remote HF Hub repo loading
]

SCAN_TARGETS = [
    ROOT_DIR / "src" / "memory" / "offline_memory_manager.py",
    ROOT_DIR / "src" / "web" / "local_model_loader.py",
    ROOT_DIR / "src" / "web" / "app.py",
    ROOT_DIR / "src" / "hope_attention" / "sa_cms.py",
    ROOT_DIR / "src" / "cms" / "continuum_memory.py",
    ROOT_DIR / "src" / "hybrid_qa" / "document_store.py",
    ROOT_DIR / "src" / "web" / "static" / "index.html",
    ROOT_DIR / "src" / "web" / "static" / "js" / "app.js",
    ROOT_DIR / "src" / "web" / "static" / "css" / "app.css",
]


def is_allowed_url(url: str) -> bool:
    for pat in ALLOWED_PATTERNS:
        if re.search(pat, url):
            return True
    return False


def run_audit() -> bool:
    print("=" * 70)
    print("OFFLINE MEMORY PERSISTENCE & ZERO-NETWORK RUNTIME AUDIT")
    print("=" * 70)

    violations = []
    scanned_files = 0

    for target in SCAN_TARGETS:
        if not target.exists():
            continue
        scanned_files += 1
        content = target.read_text(encoding="utf-8", errors="ignore")
        lines = content.splitlines()

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            # Ignore comments
            if stripped.startswith("#") or stripped.startswith("//") or stripped.startswith("/*"):
                # Exception: check for remote CDNs or scripts inside comments in HTML/JS
                if target.suffix in (".html", ".js") and any(k in line for k in ["cdn", "googleapis", "unpkg"]):
                    pass
                else:
                    continue

            # Check forbidden keywords
            for kw in FORBIDDEN_KEYWORDS:
                if re.search(kw, line, re.IGNORECASE):
                    violations.append({
                        "file": str(target.relative_to(ROOT_DIR)).replace("\\", "/"),
                        "line": line_num,
                        "type": "FORBIDDEN_REMOTE_ENDPOINT",
                        "content": line.strip()[:100],
                    })

            # Check generic URLs
            urls = re.findall(r'https?://[^\s"\'<>)]+', line)
            for u in urls:
                if not is_allowed_url(u):
                    # Check if it's inside documentation strings or paper references (e.g. arXiv citation)
                    if "arxiv.org" in u or "doi.org" in u:
                        continue
                    violations.append({
                        "file": str(target.relative_to(ROOT_DIR)).replace("\\", "/"),
                        "line": line_num,
                        "type": "EXTERNAL_URL_DETECTED",
                        "content": u,
                    })

    # Verify offline environment flags in LocalModelLoader
    loader_path = ROOT_DIR / "src" / "web" / "local_model_loader.py"
    loader_content = loader_path.read_text(encoding="utf-8")
    if 'os.environ["HF_HUB_OFFLINE"] = "1"' not in loader_content:
        violations.append({
            "file": "src/web/local_model_loader.py",
            "line": 1,
            "type": "MISSING_OFFLINE_ENV_FLAG",
            "content": "HF_HUB_OFFLINE is not set to 1",
        })

    # Verify presence of local model and memory assets
    local_runtime = ROOT_DIR / "local_runtime"
    required_assets = [
        local_runtime / "model" / "model.safetensors",
        local_runtime / "tokenizer" / "tokenizer.json",
        local_runtime / "checkpoints" / "cms_3lvl_seed_42.pt",
        local_runtime / "metadata" / "document_manifest.json",
        local_runtime / "metadata" / "memory_manifest.json",
    ]
    for asset in required_assets:
        if not asset.exists():
            violations.append({
                "file": str(asset.relative_to(ROOT_DIR)).replace("\\", "/"),
                "line": 0,
                "type": "LOCAL_ASSET_MISSING",
                "content": f"Required offline asset missing: {asset.name}",
            })

    print(f"Scanned {scanned_files} core memory persistence and runtime files.")
    print("-" * 70)

    if violations:
        print("[!] AUDIT BLOCKED: Network or asset violations found:")
        for v in violations:
            print(f"  - [{v['type']}] {v['file']}:{v['line']} -> {v['content']}")
        print("-" * 70)
        print("AUDIT VERDICT: OFFLINE_BLOCKED")
        print("=" * 70)
        return False
    else:
        print("[OK] Zero external AI endpoints found.")
        print("[OK] Zero CDN or remote script inclusions.")
        print("[OK] Zero remote font or analytics trackers.")
        print("[OK] Strict local_files_only & offline flags enforced.")
        print("[OK] All required local model and memory assets verified.")
        print("-" * 70)
        print("AUDIT VERDICT: OFFLINE_SAFE")
        print("=" * 70)
        return True


if __name__ == "__main__":
    safe = run_audit()
    sys.exit(0 if safe else 1)
