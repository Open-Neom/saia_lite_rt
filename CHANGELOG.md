## 1.2.0 — 2026-10-17

* Enhance `LiteRtServiceWeb` (`lite_rt_service_web.dart`) with progressive model download streaming, IndexedDB/CacheStorage persistence, WebGPU/WASM runtime diagnostics, and expanded `LiteRtModel` metadata.

## 1.1.0

* Added `EmbeddingGemma 2 (740M Multimodal)` (`embeddinggemma-2-740m`) to `LiteRtModelCatalog`.
* Native edge multimodal embedding support (text, code, image, audio) with 8K context and Matryoshka Representation Learning (MRL).

## 1.0.0

* Initial release of `saia_lite_rt`.
* Abstract interfaces for LiteRT (TensorFlow Lite) and LiteRT-LM (LLM) client runtimes.
* Web implementation with hardware acceleration support (WebGPU and WASM fallback).
* Built-in model catalog for edge models including Gemma 3n, MiniLM, YOLOv5, MobileNet, and YAMNet.
