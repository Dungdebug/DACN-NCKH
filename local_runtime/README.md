# Local Runtime Directory for SA-CMS Offline Chatbot

This directory contains 100% self-contained local assets for the SA-CMS offline research prototype:

- `model/`: Frozen weights (`model.safetensors`, `config.json`, `generation_config.json`)
- `tokenizer/`: Vocabulary and tokenizer specs (`tokenizer.json`, `vocab.json`, `merges.txt`, etc.)
- `checkpoints/`: Frozen SA-CMS Continual Memory checkpoints (`cms_3lvl_seed_42.pt`)
- `datasets/`: Offline evaluation items and reference texts
- `documents/`: Ingested local user documents (PDF, DOCX, TXT, MD)
- `indexes/`: Local BM25 inverted indices and passage stores
- `config/`: `runtime_config.yaml` specifying local paths

NO NETWORK CALLS ARE MADE AT RUNTIME.
