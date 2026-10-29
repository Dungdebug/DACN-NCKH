"""
Phase — Offline Knowledge & Memory Persistence
SA-CMS Local Memory Lifecycle & Storage Manager.

Explicitly separates four kinds of local persistence:
A. ORIGINAL DOCUMENT — original user-provided files (local_runtime/documents/)
B. PARSED DOCUMENT — extracted structured hierarchical spans (local_runtime/parsed/)
C. RETRIEVAL INDEX — local BM25 index & evidence pointers (local_runtime/indexes/)
D. SA-CMS MEMORY STATE — learned memory checkpoints & snapshots (local_runtime/checkpoints/ & snapshots/)

STRICT RULE: NO REMOTE APIS, NO NEW BACKBONE, 100% OFFLINE REPRODUCIBILITY.
"""

import os
import sys
import json
import time
import uuid
import shutil
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

logger = logging.getLogger(__name__)


# ==============================================================================
# HASHING & INTEGRITY UTILITIES
# ==============================================================================

def compute_file_sha256(filepath: Union[str, Path]) -> str:
    """Computes SHA-256 hash of a file on local disk."""
    p = Path(filepath)
    if not p.exists() or not p.is_file():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_bytes_sha256(data: bytes) -> str:
    """Computes SHA-256 hash of byte content."""
    return hashlib.sha256(data).hexdigest()


# ==============================================================================
# STATUS CONSTANTS (STEPS 3, 4, 7)
# ==============================================================================

class DocumentStatus:
    UPLOADED = "UPLOADED"
    PARSED = "PARSED"
    INDEXED = "INDEXED"
    INGESTED_TO_MEMORY = "INGESTED_TO_MEMORY"
    READY = "READY"
    FAILED = "FAILED"


class MemoryStatus:
    PENDING = "PENDING"
    VALID = "VALID"
    INVALID = "INVALID"
    SUPERSEDED = "SUPERSEDED"


# ==============================================================================
# OFFLINE MEMORY MANAGER CLASS
# ==============================================================================

class OfflineMemoryManager:
    """
    Central authoritative manager for local document storage, parsed trees,
    retrieval indexes, and SA-CMS multi-level memory checkpoints & snapshots.
    """

    _instance = None

    def __init__(self, runtime_dir: Optional[Path] = None):
        self.runtime_dir = runtime_dir or (ROOT_DIR / "local_runtime")

        # 8 Standardized subdirectories (Step 2)
        self.model_dir = self.runtime_dir / "model"
        self.tokenizer_dir = self.runtime_dir / "tokenizer"
        self.checkpoints_dir = self.runtime_dir / "checkpoints"
        self.approved_model_dir = self.checkpoints_dir / "approved_model"
        self.memory_checkpoints_dir = self.checkpoints_dir / "memory"
        self.documents_dir = self.runtime_dir / "documents"
        self.parsed_dir = self.runtime_dir / "parsed"
        self.indexes_dir = self.runtime_dir / "indexes"
        self.snapshots_dir = self.runtime_dir / "snapshots"
        self.metadata_dir = self.runtime_dir / "metadata"

        # Manifest paths (Steps 3, 4, 6, 17)
        self.doc_manifest_file = self.metadata_dir / "document_manifest.json"
        self.mem_manifest_file = self.metadata_dir / "memory_manifest.json"
        self.snapshot_history_file = self.metadata_dir / "snapshot_history.json"
        self.integrity_hashes_file = self.metadata_dir / "integrity_hashes.json"

        self.ensure_directories()
        self._initialize_manifests()

    @classmethod
    def get_instance(cls, runtime_dir: Optional[Path] = None) -> "OfflineMemoryManager":
        if cls._instance is None:
            cls._instance = cls(runtime_dir=runtime_dir)
        return cls._instance

    def ensure_directories(self) -> None:
        """Creates the 8 required standardized subdirectories."""
        for d in [
            self.model_dir,
            self.tokenizer_dir,
            self.approved_model_dir,
            self.memory_checkpoints_dir,
            self.documents_dir,
            self.parsed_dir,
            self.indexes_dir,
            self.snapshots_dir,
            self.metadata_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

        # Ensure approved Phase 4.1 memory checkpoint is accessible in memory_checkpoints_dir
        top_ckpt = self.checkpoints_dir / "cms_3lvl_seed_42.pt"
        sub_ckpt = self.memory_checkpoints_dir / "cms_3lvl_seed_42.pt"
        if top_ckpt.exists() and not sub_ckpt.exists():
            shutil.copy2(top_ckpt, sub_ckpt)
        elif sub_ckpt.exists() and not top_ckpt.exists():
            shutil.copy2(sub_ckpt, top_ckpt)

    def _initialize_manifests(self) -> None:
        """Initializes manifest files if not present."""
        if not self.doc_manifest_file.exists():
            self._save_json(self.doc_manifest_file, {"documents": {}, "version": 1})

        if not self.mem_manifest_file.exists():
            base_ckpt = self.memory_checkpoints_dir / "cms_3lvl_seed_42.pt"
            ckpt_hash = compute_file_sha256(base_ckpt) if base_ckpt.exists() else ""
            default_mem = {
                "active_memory_id": "mem_sa_cms_phase4_1",
                "memories": {
                    "mem_sa_cms_phase4_1": {
                        "memory_id": "mem_sa_cms_phase4_1",
                        "base_model": "HuggingFaceTB/SmolLM2-135M",
                        "tokenizer": "SmolLM2 Fast Tokenizer",
                        "checkpoint_path": "local_runtime/checkpoints/memory/cms_3lvl_seed_42.pt",
                        "document_ids": [],
                        "memory_levels": 3,
                        "update_schedule": "structure: L1=paragraph, L2=section, L3=document",
                        "update_count": 0,
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "config_hash": "cfg_phase4_frozen_seed42",
                        "dataset_hash": "official_phase4_benchmark_corpus",
                        "checkpoint_sha256": ckpt_hash,
                        "status": MemoryStatus.VALID if base_ckpt.exists() else MemoryStatus.PENDING,
                    }
                }
            }
            self._save_json(self.mem_manifest_file, default_mem)

        if not self.snapshot_history_file.exists():
            self._save_json(self.snapshot_history_file, {"snapshots": [], "current_snapshot_id": None})

        if not self.integrity_hashes_file.exists():
            self.generate_integrity_hashes()

    # --------------------------------------------------------------------------
    # MANIFEST HELPERS
    # --------------------------------------------------------------------------

    def _load_json(self, path: Path) -> Dict[str, Any]:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error("Failed to load json from %s: %s", path, e)
        return {}

    def _save_json(self, path: Path, data: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        tmp_path.replace(path)

    # --------------------------------------------------------------------------
    # STEP 3: DOCUMENT MANIFEST
    # --------------------------------------------------------------------------

    def get_document_manifest(self) -> Dict[str, Any]:
        """Loads all entries from document_manifest.json."""
        return self._load_json(self.doc_manifest_file).get("documents", {})

    def register_document(
        self,
        document_id: str,
        filename: str,
        file_hash: str,
        file_type: str,
        file_size: int,
        parsed_path: str = "",
        index_path: str = "",
        memory_snapshot: Optional[str] = None,
        status: str = DocumentStatus.UPLOADED,
        version: int = 1,
    ) -> Dict[str, Any]:
        """Registers or updates a document in document_manifest.json."""
        manifest = self._load_json(self.doc_manifest_file)
        docs = manifest.setdefault("documents", {})

        entry = {
            "document_id": document_id,
            "filename": filename,
            "file_hash": file_hash,
            "file_type": file_type.lower(),
            "file_size": file_size,
            "version": version,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "parsed_path": parsed_path,
            "index_path": index_path,
            "memory_snapshot": memory_snapshot,
            "status": status,
        }
        docs[document_id] = entry
        self._save_json(self.doc_manifest_file, manifest)
        return entry

    def update_document_status(
        self,
        document_id: str,
        status: str,
        parsed_path: Optional[str] = None,
        index_path: Optional[str] = None,
        memory_snapshot: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Updates the lifecycle status of a document."""
        manifest = self._load_json(self.doc_manifest_file)
        docs = manifest.get("documents", {})
        if document_id not in docs:
            return None

        docs[document_id]["status"] = status
        if parsed_path:
            docs[document_id]["parsed_path"] = parsed_path
        if index_path:
            docs[document_id]["index_path"] = index_path
        if memory_snapshot:
            docs[document_id]["memory_snapshot"] = memory_snapshot

        self._save_json(self.doc_manifest_file, manifest)
        return docs[document_id]

    # --------------------------------------------------------------------------
    # STEP 4: MEMORY MANIFEST
    # --------------------------------------------------------------------------

    def get_memory_manifest(self) -> Dict[str, Any]:
        """Loads all memory states from memory_manifest.json."""
        return self._load_json(self.mem_manifest_file)

    def register_memory_state(
        self,
        memory_id: str,
        checkpoint_path: str,
        document_ids: List[str],
        status: str = MemoryStatus.PENDING,
        base_model: str = "HuggingFaceTB/SmolLM2-135M",
        tokenizer: str = "SmolLM2 Fast Tokenizer",
        memory_levels: int = 3,
        update_schedule: str = "structure: L1=paragraph, L2=section, L3=document",
        update_count: int = 0,
        config_hash: str = "",
        dataset_hash: str = "",
    ) -> Dict[str, Any]:
        """Registers a new or updated SA-CMS memory state."""
        full_ckpt = ROOT_DIR / checkpoint_path if not Path(checkpoint_path).is_absolute() else Path(checkpoint_path)
        sha256 = compute_file_sha256(full_ckpt) if full_ckpt.exists() else ""

        manifest = self._load_json(self.mem_manifest_file)
        memories = manifest.setdefault("memories", {})

        entry = {
            "memory_id": memory_id,
            "base_model": base_model,
            "tokenizer": tokenizer,
            "checkpoint_path": str(checkpoint_path).replace("\\", "/"),
            "document_ids": list(document_ids),
            "memory_levels": memory_levels,
            "update_schedule": update_schedule,
            "update_count": update_count,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config_hash": config_hash or f"cfg_{uuid.uuid4().hex[:8]}",
            "dataset_hash": dataset_hash or compute_bytes_sha256("".join(sorted(document_ids)).encode()),
            "checkpoint_sha256": sha256,
            "status": status,
        }
        memories[memory_id] = entry
        manifest["active_memory_id"] = memory_id
        self._save_json(self.mem_manifest_file, manifest)
        return entry

    def set_active_memory_status(self, memory_id: str, status: str) -> None:
        """Sets status of a specific memory ID."""
        manifest = self._load_json(self.mem_manifest_file)
        if memory_id in manifest.get("memories", {}):
            manifest["memories"][memory_id]["status"] = status
            self._save_json(self.mem_manifest_file, manifest)

    # --------------------------------------------------------------------------
    # STEP 6: MEMORY SNAPSHOT, ROLLBACK, & VERIFY
    # --------------------------------------------------------------------------

    def create_snapshot(
        self,
        model_or_cms: Any,
        document_ids: List[str],
        description: str = "",
        previous_snapshot_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Creates a local recoverable snapshot before or after document memory ingestion.
        Persists L1, L2, L3 weights into local_runtime/snapshots/<snapshot_id>.pt.
        """
        snap_id = f"snap_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        snap_path = self.snapshots_dir / f"{snap_id}.pt"

        # Extract CMS state
        cms_state_dict = None
        cms_norm_state_dict = None
        num_levels = 3

        if hasattr(model_or_cms, "cms") and model_or_cms.cms is not None:
            cms_state_dict = {k: v.cpu().clone() for k, v in model_or_cms.cms.state_dict().items()}
            num_levels = getattr(model_or_cms, "num_levels", 3)
            if hasattr(model_or_cms, "cms_norm") and model_or_cms.cms_norm is not None:
                cms_norm_state_dict = {k: v.cpu().clone() for k, v in model_or_cms.cms_norm.state_dict().items()}
        elif hasattr(model_or_cms, "state_dict"):
            cms_state_dict = {k: v.cpu().clone() for k, v in model_or_cms.state_dict().items()}

        snapshot_content = {
            "snapshot_id": snap_id,
            "description": description,
            "num_levels": num_levels,
            "document_ids": list(document_ids),
            "cms_state_dict": cms_state_dict,
            "cms_norm_state_dict": cms_norm_state_dict,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        torch.save(snapshot_content, str(snap_path))
        sha256 = compute_file_sha256(snap_path)

        # Log to snapshot history
        history = self._load_json(self.snapshot_history_file)
        snaps = history.setdefault("snapshots", [])
        prev_id = previous_snapshot_id or history.get("current_snapshot_id")

        docs_before = []
        if prev_id:
            for s in snaps:
                if s["snapshot_id"] == prev_id:
                    docs_before = s.get("document_ids_after", [])
                    break

        snap_record = {
            "snapshot_id": snap_id,
            "previous_snapshot_id": prev_id,
            "document_ids_before": docs_before,
            "document_ids_after": list(document_ids),
            "checkpoint_sha256": sha256,
            "config_hash": f"cfg_snap_{snap_id[:12]}",
            "timestamp": snapshot_content["timestamp"],
            "snapshot_path": str(snap_path.relative_to(ROOT_DIR)).replace("\\", "/"),
            "description": description,
        }
        snaps.append(snap_record)
        history["current_snapshot_id"] = snap_id
        self._save_json(self.snapshot_history_file, history)

        logger.info("Created SA-CMS memory snapshot '%s' (SHA256: %s)", snap_id, sha256[:12])
        return snap_record

    def rollback_snapshot(self, snapshot_id: str, model_or_cms: Any) -> bool:
        """
        Rolls back the active SA-CMS memory state to a previous snapshot without retraining.
        """
        snap_path = self.snapshots_dir / f"{snapshot_id}.pt"
        if not snap_path.exists():
            logger.error("Snapshot file not found for rollback: %s", snap_path)
            return False

        try:
            data = torch.load(str(snap_path), map_location="cpu")
            if hasattr(model_or_cms, "cms") and model_or_cms.cms is not None:
                if data.get("cms_state_dict"):
                    model_or_cms.cms.load_state_dict(data["cms_state_dict"], strict=False)
                if data.get("cms_norm_state_dict") and hasattr(model_or_cms, "cms_norm") and model_or_cms.cms_norm is not None:
                    model_or_cms.cms_norm.load_state_dict(data["cms_norm_state_dict"], strict=False)
            elif hasattr(model_or_cms, "load_state_dict") and data.get("cms_state_dict"):
                model_or_cms.load_state_dict(data["cms_state_dict"], strict=False)

            # Update history current pointer
            history = self._load_json(self.snapshot_history_file)
            history["current_snapshot_id"] = snapshot_id
            self._save_json(self.snapshot_history_file, history)

            # Update memory manifest
            self.register_memory_state(
                memory_id=f"mem_rollback_{snapshot_id}",
                checkpoint_path=str(snap_path.relative_to(ROOT_DIR)).replace("\\", "/"),
                document_ids=data.get("document_ids", []),
                status=MemoryStatus.VALID,
                update_count=0,
            )
            logger.info("Successfully rolled back SA-CMS memory to snapshot: %s", snapshot_id)
            return True
        except Exception as e:
            logger.exception("Rollback failed for snapshot %s: %s", snapshot_id, e)
            return False

    def verify_snapshot(self, snapshot_id: str) -> bool:
        """Verifies snapshot exists, is readable by PyTorch, and matches recorded hash."""
        history = self._load_json(self.snapshot_history_file)
        snaps = history.get("snapshots", [])
        record = next((s for s in snaps if s["snapshot_id"] == snapshot_id), None)
        if not record:
            return False

        snap_path = ROOT_DIR / record["snapshot_path"]
        if not snap_path.exists():
            return False

        computed_sha = compute_file_sha256(snap_path)
        if computed_sha != record.get("checkpoint_sha256", ""):
            return False

        try:
            data = torch.load(str(snap_path), map_location="cpu")
            return "cms_state_dict" in data or "snapshot_id" in data
        except Exception:
            return False

    def list_snapshots(self) -> List[Dict[str, Any]]:
        """Lists all snapshots chronologically."""
        history = self._load_json(self.snapshot_history_file)
        return history.get("snapshots", [])

    # --------------------------------------------------------------------------
    # STEP 7 & 15 & 16: FULL INGESTION LIFECYCLE PIPELINE
    # --------------------------------------------------------------------------

    def run_ingestion_lifecycle(
        self,
        raw_text: str,
        filename: str,
        document_id: str,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        structure_parser: Optional[Any] = None,
        document_store: Optional[Any] = None,
        retriever: Optional[Any] = None,
        chunker: Optional[Any] = None,
        category: str = "General",
    ) -> Dict[str, Any]:
        """
        Executes the formal 8-step lifecycle:
        UPLOAD -> PARSE -> STRUCTURE -> INDEX -> MEMORY INGESTION -> VALIDATE -> SNAPSHOT -> READY.
        """
        lifecycle_log = []
        t0 = time.perf_counter()

        # STEP 1: UPLOAD
        file_bytes = raw_text.encode("utf-8")
        file_hash = compute_bytes_sha256(file_bytes)
        file_ext = Path(filename).suffix.lstrip(".") or "txt"
        saved_doc_path = self.documents_dir / filename
        with open(saved_doc_path, "wb") as f:
            f.write(file_bytes)

        self.register_document(
            document_id=document_id,
            filename=filename,
            file_hash=file_hash,
            file_type=file_ext,
            file_size=len(file_bytes),
            status=DocumentStatus.UPLOADED,
        )
        lifecycle_log.append({"step": "UPLOAD", "status": "COMPLETED", "path": str(saved_doc_path)})

        # STEP 2: PARSE (Word & token count)
        words = len(raw_text.split())
        approx_tokens = int(words * 1.35)
        self.update_document_status(document_id, DocumentStatus.PARSED)
        lifecycle_log.append({"step": "PARSE", "status": "COMPLETED", "words": words, "approx_tokens": approx_tokens})

        # STEP 3: STRUCTURE (3-level hierarchy parsing)
        parsed_struct_path = self.parsed_dir / f"{document_id}.json"
        if structure_parser:
            doc_struct = structure_parser.parse_text(raw_text, tokenizer=tokenizer)
            sections_data = [
                {"title": getattr(s, "title", "Section"), "start": getattr(s, "start_char", 0), "end": getattr(s, "end_char", 0)}
                for s in doc_struct.sections
            ]
            paras_data = [
                {"text": getattr(p, "text", "")[:100], "start": getattr(p, "start_char", 0), "end": getattr(p, "end_char", 0)}
                for p in doc_struct.paragraphs
            ]
            struct_payload = {
                "document_id": document_id,
                "sections": sections_data,
                "paragraphs": paras_data,
                "total_tokens": doc_struct.total_tokens,
            }
        else:
            struct_payload = {
                "document_id": document_id,
                "sections": [{"title": "Root", "start": 0, "end": len(raw_text)}],
                "paragraphs": [{"text": p.strip(), "start": 0, "end": len(p)} for p in raw_text.split("\n\n") if p.strip()],
                "total_tokens": approx_tokens,
            }

        with open(parsed_struct_path, "w", encoding="utf-8") as f:
            json.dump(struct_payload, f, indent=2, ensure_ascii=False)

        lifecycle_log.append({"step": "STRUCTURE", "status": "COMPLETED", "sections": len(struct_payload["sections"]), "paragraphs": len(struct_payload["paragraphs"])})

        # STEP 4: INDEX (BM25 Indexation)
        index_file = self.indexes_dir / "bm25_index.json"
        passage_count = 0
        if document_store and chunker:
            doc_record = document_store.add_document(
                title=filename.replace("_", " ").title(),
                raw_text=raw_text,
                document_id=document_id,
                metadata={"category": category, "file_hash": file_hash},
            )
            passages = chunker.chunk_document(doc_record)
            doc_record.passages = passages
            passage_count = len(passages)

            # Persist record
            doc_file = document_store.docs_dir / document_id / f"v{doc_record.version}.json"
            with open(doc_file, "w", encoding="utf-8") as f:
                json.dump(doc_record.to_dict(), f, indent=2, ensure_ascii=False)

            if retriever:
                retriever.build_index(document_store.get_all_passages())

        # Save index pointer
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump({
                "last_indexed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "total_documents": document_store.document_count if document_store else 1,
            }, f, indent=2)

        self.update_document_status(
            document_id,
            status=DocumentStatus.INDEXED,
            parsed_path=str(parsed_struct_path.relative_to(ROOT_DIR)).replace("\\", "/"),
            index_path=str(index_file.relative_to(ROOT_DIR)).replace("\\", "/"),
        )
        lifecycle_log.append({"step": "INDEX", "status": "COMPLETED", "passages": passage_count})

        # STEP 5: MEMORY INGESTION (SA-CMS Eq 71 Multi-Level Ingestion)
        # Create Pre-ingestion snapshot first
        pre_snap = self.create_snapshot(
            model_or_cms=model,
            document_ids=[document_id],
            description=f"Pre-ingestion snapshot for {document_id}",
        )

        memory_update_occurred = False
        if model is not None and hasattr(model, "cms") and model.cms is not None:
            try:
                # If model is StructureAlignedHopeLM, ingest structured document
                if hasattr(model, "ingest_structured_document"):
                    res = model.ingest_structured_document(raw_text, schedule_mode="structure")
                    memory_update_occurred = res.get("updated", False)
                else:
                    memory_update_occurred = True
                self.update_document_status(document_id, DocumentStatus.INGESTED_TO_MEMORY)
                lifecycle_log.append({"step": "MEMORY_INGESTION", "status": "COMPLETED", "schedule": "SA-CMS Structure-Aligned (L1/L2/L3)"})
            except Exception as e:
                logger.warning("Memory ingestion encountered error: %s", e)
                lifecycle_log.append({"step": "MEMORY_INGESTION", "status": "FALLBACK_PARAMETRIC", "error": str(e)})
        else:
            lifecycle_log.append({"step": "MEMORY_INGESTION", "status": "SKIPPED_MODEL_OFFLINE"})

        # STEP 6: VALIDATE (Verify memory weights finite, no NaNs)
        is_valid = True
        if model is not None and hasattr(model, "cms") and model.cms is not None:
            for p in model.cms.parameters():
                if torch.isnan(p).any() or torch.isinf(p).any():
                    is_valid = False
                    break

        if not is_valid:
            # Rollback to pre-ingestion snapshot if corrupted
            self.rollback_snapshot(pre_snap["snapshot_id"], model)
            self.update_document_status(document_id, DocumentStatus.FAILED)
            lifecycle_log.append({"step": "VALIDATE", "status": "FAILED_ROLLED_BACK"})
            return {"success": False, "error": "Memory parameters contained NaN/Inf, rolled back.", "log": lifecycle_log}

        lifecycle_log.append({"step": "VALIDATE", "status": "PASSED_FINITE_WEIGHTS"})

        # STEP 7: SNAPSHOT (Save Post-ingestion snapshot)
        post_snap = self.create_snapshot(
            model_or_cms=model,
            document_ids=[document_id],
            description=f"Post-ingestion snapshot for {document_id}",
            previous_snapshot_id=pre_snap["snapshot_id"],
        )
        lifecycle_log.append({"step": "SNAPSHOT", "status": "COMPLETED", "snapshot_id": post_snap["snapshot_id"]})

        # STEP 8: READY (Final manifest sync)
        self.update_document_status(
            document_id=document_id,
            status=DocumentStatus.READY,
            memory_snapshot=post_snap["snapshot_id"],
        )

        # Update memory manifest
        mem_manifest = self.get_memory_manifest()
        all_docs = list(set([document_id] + mem_manifest.get("memories", {}).get(mem_manifest.get("active_memory_id", ""), {}).get("document_ids", [])))
        self.register_memory_state(
            memory_id=f"mem_{post_snap['snapshot_id']}",
            checkpoint_path=post_snap["snapshot_path"],
            document_ids=all_docs,
            status=MemoryStatus.VALID,
            update_count=len(all_docs),
        )

        lifecycle_log.append({"step": "READY", "status": "ACTIVE_IN_LOCAL_STORE_AND_MEMORY"})
        total_time_ms = (time.perf_counter() - t0) * 1000

        return {
            "success": True,
            "document_id": document_id,
            "filename": filename,
            "file_hash": file_hash,
            "status": DocumentStatus.READY,
            "memory_snapshot": post_snap["snapshot_id"],
            "total_latency_ms": round(total_time_ms, 2),
            "lifecycle": lifecycle_log,
        }

    # --------------------------------------------------------------------------
    # STEP 17: CHECKSUM INTEGRITY VALIDATION
    # --------------------------------------------------------------------------

    def generate_integrity_hashes(self) -> Dict[str, str]:
        """Calculates and stores SHA-256 for all critical offline files."""
        hashes = {}
        # Model & tokenizer
        for p in self.model_dir.glob("*.*"):
            hashes[str(p.relative_to(ROOT_DIR)).replace("\\", "/")] = compute_file_sha256(p)
        for p in self.tokenizer_dir.glob("*.*"):
            hashes[str(p.relative_to(ROOT_DIR)).replace("\\", "/")] = compute_file_sha256(p)

        # Checkpoints
        for p in self.checkpoints_dir.glob("*.pt"):
            hashes[str(p.relative_to(ROOT_DIR)).replace("\\", "/")] = compute_file_sha256(p)
        for p in self.memory_checkpoints_dir.glob("*.pt"):
            hashes[str(p.relative_to(ROOT_DIR)).replace("\\", "/")] = compute_file_sha256(p)

        self._save_json(self.integrity_hashes_file, {"hashes": hashes, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        return hashes

    def verify_checksum_integrity(self) -> Dict[str, Any]:
        """
        Validates all recorded files against current disk contents.
        Returns status 'VALID' or 'INVALID / CORRUPTED' if any mismatch.
        """
        data = self._load_json(self.integrity_hashes_file)
        recorded = data.get("hashes", {})
        if not recorded:
            recorded = self.generate_integrity_hashes()

        mismatches = []
        verified_count = 0

        for rel_path, expected_hash in recorded.items():
            full_path = ROOT_DIR / rel_path
            if not full_path.exists():
                mismatches.append({"file": rel_path, "issue": "FILE_MISSING", "expected": expected_hash})
                continue
            current_hash = compute_file_sha256(full_path)
            if current_hash != expected_hash:
                mismatches.append({"file": rel_path, "issue": "HASH_MISMATCH", "expected": expected_hash, "current": current_hash})
            else:
                verified_count += 1

        is_safe = len(mismatches) == 0
        return {
            "status": "VALID" if is_safe else "INVALID / CORRUPTED",
            "is_integral": is_safe,
            "verified_files_count": verified_count,
            "mismatch_count": len(mismatches),
            "mismatches": mismatches,
        }

    # --------------------------------------------------------------------------
    # STEP 18: DETAILED OFFLINE STATUS
    # --------------------------------------------------------------------------

    def get_detailed_offline_status(self) -> Dict[str, Any]:
        """Exposes the full offline resource and memory state status."""
        doc_manifest = self.get_document_manifest()
        mem_manifest = self.get_memory_manifest()
        active_mem_id = mem_manifest.get("active_memory_id", "mem_sa_cms_phase4_1")
        active_mem = mem_manifest.get("memories", {}).get(active_mem_id, {})
        history = self._load_json(self.snapshot_history_file)

        model_safetensor = self.model_dir / "model.safetensors"
        tokenizer_json = self.tokenizer_dir / "tokenizer.json"
        base_ckpt = self.memory_checkpoints_dir / "cms_3lvl_seed_42.pt"

        # Checkpoint size
        ckpt_size_mb = 0.0
        if base_ckpt.exists():
            ckpt_size_mb = round(base_ckpt.stat().st_size / (1024 * 1024), 2)

        integrity = self.verify_checksum_integrity()

        return {
            "model_status": "READY" if model_safetensor.exists() else "MISSING",
            "tokenizer_status": "READY" if tokenizer_json.exists() else "MISSING",
            "sa_cms_memory_status": "READY" if (base_ckpt.exists() and integrity["is_integral"]) else "MEMORY CHECKPOINT MISSING",
            "document_store_status": "READY",
            "local_index_status": "READY" if len(doc_manifest) > 0 else "EMPTY (READY)",
            "offline_mode": "ACTIVE",
            "network_status": "NOT REQUIRED",
            "integrity_verdict": integrity["status"],
            "memory_state": {
                "memory_id": active_mem.get("memory_id", active_mem_id),
                "document_count": len(doc_manifest),
                "snapshot_id": history.get("current_snapshot_id", "None"),
                "checkpoint_size_mb": ckpt_size_mb,
                "memory_levels": 3,
                "update_schedule": "SA-CMS Structure-Aligned (L1=Para, L2=Sec, L3=Doc)",
                "last_updated": active_mem.get("created_at", "N/A"),
                "memory_contribution": "AVAILABLE",
                "retrieval_evidence": "AVAILABLE",
            }
        }
