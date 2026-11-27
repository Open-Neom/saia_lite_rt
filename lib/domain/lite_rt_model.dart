import 'dart:typed_data';

import 'package:flutter/foundation.dart' show kIsWeb;

/// The modality or type of the LiteRT model.
enum LiteRtModelType {
  chat,
  embeddings,
  vision,
  audio,
  tts,
}

/// Metadata information for a LiteRt model available for edge execution.
class LiteRtModelInfo {
  final String id;
  final String name;
  final String description;
  final LiteRtModelType type;
  final String downloadUrl;
  final int sizeBytes;

  /// Whether downloading this model requires a HuggingFace token (`hf_...`)
  /// and accepting the upstream model license.
  final bool requiresHfToken;

  /// URL where the user must accept the model license (for gated HF repos).
  final String? licenseUrl;

  /// URL where the user can create or copy their HuggingFace read token.
  final String? tokenSetupUrl;

  /// Human-readable step-by-step requirements if the model requires extra setup.
  final List<String> requirementSteps;

  /// True when this model is managed internally by the orchestrator (e.g.
  /// vision, audio perception, or grammar tool extraction) and should not be
  /// shown as a standalone user-selectable brain card in the UI.
  final bool isInternalOnly;

  const LiteRtModelInfo({
    required this.id,
    required this.name,
    required this.description,
    required this.type,
    required this.downloadUrl,
    required this.sizeBytes,
    this.requiresHfToken = false,
    this.licenseUrl,
    this.tokenSetupUrl,
    this.requirementSteps = const [],
    this.isInternalOnly = false,
  });

  /// Human-readable file size (e.g. "1.5 GB", "30 MB").
  String get displaySize {
    if (!sizeBytes.isFinite || sizeBytes <= 0) {
      return '0 MB';
    }
    if (sizeBytes >= 1024 * 1024 * 1024) {
      return '${(sizeBytes / (1024 * 1024 * 1024)).toStringAsFixed(1)} GB';
    }
    return '${(sizeBytes / (1024 * 1024)).toStringAsFixed(0)} MB';
  }

  /// Whether [urlOrId] uses a browser-supported runtime format (WebLLM `-MLC`,
  /// MediaPipe `.task`/`.bin`/`.litertlm`, `.tflite`, or `.onnx`), excluding
  /// native-only artifacts such as `.gguf` and `.cact`.
  static bool isWebCompatibleArtifact(String urlOrId) {
    final trimmed = urlOrId.trim();
    if (trimmed.isEmpty) return false;
    final lower = trimmed.toLowerCase().split('?').first.split('#').first;
    if (lower.endsWith('.gguf') || lower.endsWith('.cact')) {
      return false;
    }
    if (!lower.contains('://')) {
      return lower.contains('-mlc') ||
          lower.endsWith('.task') ||
          lower.endsWith('.bin') ||
          lower.endsWith('.litertlm') ||
          lower.endsWith('.tflite') ||
          lower.endsWith('.onnx');
    }
    return lower.startsWith('blob:') ||
        lower.endsWith('.task') ||
        lower.endsWith('.bin') ||
        lower.endsWith('.litertlm') ||
        lower.endsWith('.tflite') ||
        lower.endsWith('.onnx');
  }

  /// True when this model can run in a WebGPU / WASM browser runtime.
  bool get isWebCompatible => isWebCompatibleArtifact(downloadUrl);
}

/// A built-in catalog of verified WebGPU / WASM models for browser edge execution.
class LiteRtModelCatalog {
  LiteRtModelCatalog._();

  static const String kHfTokensUrl = 'https://huggingface.co/settings/tokens';

  /// Full catalog including native-only embedded entries (`.gguf`, `.cact`).
  static const List<LiteRtModelInfo> catalogEntries = [
    // ── 1. Conversación y Razonamiento en Navegador (WebGPU — 100% Apache 2.0, 0 llaves) ──
    LiteRtModelInfo(
      id: 'smollm2-360m',
      name: 'Itzli Chispa · 360M',
      description:
          'Nivel ultraligero de respuesta inmediata (360M parámetros · q4f16 · ~376 MB VRAM). '
          'Ideal para equipos con memoria gráfica compacta.',
      type: LiteRtModelType.chat,
      downloadUrl: 'SmolLM2-360M-Instruct-q4f16_1-MLC',
      sizeBytes: 199229440, // ~190 MB
    ),
    LiteRtModelInfo(
      id: 'qwen25-05b',
      name: 'Itzli Fugaz · 0.5B',
      description:
          'Dominio ágil y multilingüe en español (500M parámetros · q4f16). '
          'Descarga ligera y conversación fluida 100% en tu navegador sin enviar datos.',
      type: LiteRtModelType.chat,
      downloadUrl: 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC',
      sizeBytes: 278921216, // ~266 MB verificados en Chrome WebGPU
    ),
    LiteRtModelInfo(
      id: 'qwen3-06b',
      name: 'Itzli Fugaz · 0.6B',
      description:
          'Razonamiento híbrido ligero y baja latencia (600M parámetros · q4f16 · ~1.4 GB VRAM). '
          'Piensa paso a paso y resuelve problemas concretos sin salir del navegador.',
      type: LiteRtModelType.chat,
      downloadUrl: 'Qwen3-0.6B-q4f16_1-MLC',
      sizeBytes: 367001600, // ~350 MB
    ),
    LiteRtModelInfo(
      id: 'qwen25-15b',
      name: 'Itzli Centella · 1.5B',
      description:
          'Dominio conversacional ágil y redacción en español (1,500M parámetros · '
          'ventana 4K). Requiere ~1.6 GB de VRAM.',
      type: LiteRtModelType.chat,
      downloadUrl: 'Qwen2.5-1.5B-Instruct-q4f16_1-MLC',
      sizeBytes: 922746880, // ~880 MB
    ),
    LiteRtModelInfo(
      id: 'qwen3-17b',
      name: 'Itzli Centella · 1.7B',
      description:
          'Dominio intermedio con pensamiento paso a paso y síntesis multilingüe en español '
          '(1,700M parámetros · q4f16 · ~2.0 GB VRAM).',
      type: LiteRtModelType.chat,
      downloadUrl: 'Qwen3-1.7B-q4f16_1-MLC',
      sizeBytes: 1027604480, // ~980 MB
    ),
    LiteRtModelInfo(
      id: 'qwen25-3b',
      name: 'Itzli Centella · 3B',
      description:
          'Dominio completo de conversación natural, síntesis e instrucciones estructuradas '
          '(3,000M parámetros · ventana 4K · ~2.5 GB VRAM).',
      type: LiteRtModelType.chat,
      downloadUrl: 'Qwen2.5-3B-Instruct-q4f16_1-MLC',
      sizeBytes: 1782579200, // ~1.7 GB
    ),
    LiteRtModelInfo(
      id: 'gemma4-e2b-web',
      name: 'Itzli Centella · E2B',
      description:
          'Dominio equilibrado de nueva generación con razonamiento integrado, visión multimodal y contexto amplio '
          '(E2B efectivos · ventana 128K · ~1.8 GB VRAM). Corre 100% en tu navegador sin enviar datos.',
      type: LiteRtModelType.chat,
      downloadUrl:
          'https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/cccc664ea93091c52eeb7867ad4e5466357e14f3/gemma-4-E2B-it-gpu.litertlm',
      sizeBytes: 2340870772, // ~2.34 GB verificados en HF (Apache 2.0 · rev cccc664ea930)
    ),
    LiteRtModelInfo(
      id: 'qwen3-4b',
      name: 'Itzli Semilla · 4B',
      description:
          'Capacidad analítica compacta, código y razonamiento acelerado por GPU '
          'en tu navegador (4,000M parámetros · q4f16 · ~3.4 GB VRAM).',
      type: LiteRtModelType.chat,
      downloadUrl: 'Qwen3-4B-q4f16_1-MLC',
      sizeBytes: 2306867200, // ~2.2 GB
    ),
    LiteRtModelInfo(
      id: 'gemma4-e4b-web',
      name: 'Itzli Semilla · E4B',
      description:
          'Máxima capacidad analítica y estratégica en navegador con razonamiento profundo '
          '(E4B efectivos · ventana 128K · ~3.3 GB VRAM). Ideal para equipos con GPU moderna.',
      type: LiteRtModelType.chat,
      downloadUrl:
          'https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/resolve/main/gemma-4-E4B-it-gpu.litertlm',
      sizeBytes: 3310938740, // ~3.31 GB verificados en HF (Apache 2.0)
    ),

    // ── 2. Memoria Semántica y Embeddings (Familia Itzli Raíz — 100% libres) ──
    LiteRtModelInfo(
      id: 'universal-sentence-encoder',
      name: 'Itzli Raíz · Ligero (6M · 100d)',
      description:
          'Vectores semánticos ultrarrápidos (6M parámetros · 100 dimensiones L2) '
          'para memoria episódica y búsqueda por significado en el navegador.',
      type: LiteRtModelType.embeddings,
      downloadUrl:
          'https://storage.googleapis.com/mediapipe-models/text_embedder/universal_sentence_encoder/float32/1/universal_sentence_encoder.tflite',
      sizeBytes: 6120274, // 6 MB verificados
    ),
    LiteRtModelInfo(
      id: 'bert-embedder',
      name: 'Itzli Raíz · Denso (110M · 512d)',
      description:
          'Representación semántica densa de mayor precisión (110M parámetros · 512 dimensiones '
          'con tokenizador integrado).',
      type: LiteRtModelType.embeddings,
      downloadUrl:
          'https://storage.googleapis.com/mediapipe-models/text_embedder/bert_embedder/float32/1/bert_embedder.tflite',
      sizeBytes: 26142232, // 25 MB verificados
    ),
    LiteRtModelInfo(
      id: 'qwen3-embedding-06b',
      name: 'Itzli Raíz · Profundo (600M · 1024d)',
      description:
          'Dominio profundo de recuperación semántica multilingüe en más de 100 idiomas '
          '(600M parámetros · hasta 1,024 dimensiones MRL · Q8_0).',
      type: LiteRtModelType.embeddings,
      downloadUrl:
          'https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-Q8_0.gguf',
      sizeBytes: 639000000, // ~610 MiB
    ),

    // ── 3. Motores Internos de Percepción Multimodal (No expuestos al usuario) ──
    LiteRtModelInfo(
      id: 'mobilenet-v3',
      name: 'Itzli Percepción · Escenas (5.3M)',
      description:
          'Motor interno de clasificación visual en tiempo real (5.3M parámetros · 1,000 categorías).',
      type: LiteRtModelType.vision,
      downloadUrl:
          'https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/1/efficientnet_lite0.tflite',
      sizeBytes: 18582189, // 18 MB verificados
      isInternalOnly: true,
    ),
    LiteRtModelInfo(
      id: 'ssd-mobilenet-v2',
      name: 'Itzli Percepción · Objetos (4.5M)',
      description:
          'Motor interno de detección espacial de objetos y cajas delimitadoras (4.5M parámetros · float16 · 80 clases).',
      type: LiteRtModelType.vision,
      downloadUrl:
          'https://storage.googleapis.com/mediapipe-models/object_detector/ssd_mobilenet_v2/float16/1/ssd_mobilenet_v2.tflite',
      sizeBytes: 5918749, // 5.6 MB verificados
      isInternalOnly: true,
    ),
    LiteRtModelInfo(
      id: 'yamnet',
      name: 'Itzli Percepción · Audio (3.7M)',
      description:
          'Motor interno de detección de 521 eventos sonoros (3.7M parámetros a 16 kHz).',
      type: LiteRtModelType.audio,
      downloadUrl:
          'https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite',
      sizeBytes: 4126810, // 4 MB verificados
      isInternalOnly: true,
    ),

    // ── 4. Síntesis de Voz y Micro-Herramientas Internas ────────────────
    LiteRtModelInfo(
      id: 'kitten-tts-nano-int8',
      name: 'Itzli Tlahtolli · Nano (15M INT8)',
      description:
          'Síntesis de voz neuronal ultra-ligera (15M parámetros · <25 MB) que corre en tiempo real sobre CPU/WASM sin GPU.',
      type: LiteRtModelType.tts,
      downloadUrl:
          'https://huggingface.co/KittenML/kitten-tts-nano-0.8-int8/resolve/main/kitten_tts_nano_v0_8.onnx',
      sizeBytes: 25000000, // ~25 MB
      isInternalOnly: true,
    ),
    LiteRtModelInfo(
      id: 'kitten-tts-micro',
      name: 'Itzli Tlahtolli · Micro (40M)',
      description:
          'Síntesis de voz expresiva local (40M parámetros · 41 MB) para anuncios comunitarios y lectura fluida sin conexión.',
      type: LiteRtModelType.tts,
      downloadUrl:
          'https://huggingface.co/KittenML/kitten-tts-micro-0.8/resolve/main/kitten_tts_micro_v0_8.onnx',
      sizeBytes: 41000000, // ~41 MB
      isInternalOnly: true,
    ),
    LiteRtModelInfo(
      id: 'kitten-tts-2-emb4',
      name: 'Itzli Tlahtolli · Estudio Multilingüe (1.7B 1-bit)',
      description:
          'Dominio vocal expresivo de gran escala (1,700M parámetros 1-bit · 20 idiomas incl. Español, 47 voces y matices emocionales).',
      type: LiteRtModelType.tts,
      downloadUrl:
          'https://huggingface.co/KittenML/kitten-tts-2-emb4/resolve/main/kitten_tts_2_emb4.gguf',
      sizeBytes: 530579456, // ~506 MiB (1-bit weights + 4-bit embeddings)
      isInternalOnly: true,
    ),
    LiteRtModelInfo(
      id: 'audio8-tts-01b-int8',
      name: 'Itzli Tlahtolli · Clonación Ligera (100M INT8)',
      description:
          'Síntesis multilingüe en Español con clonación de voz Zero-Shot (100M parámetros INT8 · 44.1 kHz).',
      type: LiteRtModelType.tts,
      downloadUrl:
          'https://huggingface.co/Audio8/Audio8-TTS-0.1B-ONNX-INT8/resolve/main/model_int8.onnx',
      sizeBytes: 95000000, // ~95 MB
      isInternalOnly: true,
    ),
    LiteRtModelInfo(
      id: 'needle-3-mobile',
      name: 'Itzli Aguja · Herramientas (80M · 2-bit)',
      description:
          'Micro-motor interno de 18 MB (80M parámetros · 2.125 bits) para extracción JSON restringida por gramática y herramientas.',
      type: LiteRtModelType.chat,
      downloadUrl:
          'https://huggingface.co/cactus-compute/needle-3-mobile/resolve/main/needle-3.cact',
      sizeBytes: 18874368, // ~18 MB (rango 8-29 MB en 2.125 bits/peso)
      isInternalOnly: true,
    ),
    LiteRtModelInfo(
      id: 'whistle-stt-int4',
      name: 'Itzli Percepción · Dictado (39M INT4)',
      description:
          'Reconocimiento interno de voz a texto offline en Español e Inglés (39M parámetros INT4 · 16.9 MB).',
      type: LiteRtModelType.audio,
      downloadUrl:
          'https://huggingface.co/cactus-compute/whistle-stt/resolve/main/whistle.cact',
      sizeBytes: 17720934, // ~16.9 MB
      isInternalOnly: true,
    ),
  ];

  /// Models verified to execute on Web (WebGPU / WASM), excluding native-only
  /// `.gguf` and `.cact` artifacts.
  static List<LiteRtModelInfo> get webCompatible =>
      catalogEntries.where((m) => m.isWebCompatible).toList(growable: false);

  /// Platform-filtered model list: returns [webCompatible] on Web and
  /// [catalogEntries] on native runtimes.
  static List<LiteRtModelInfo> get all =>
      kIsWeb ? webCompatible : catalogEntries;

  /// User-facing models for the LOCAL panel (chat brains and semantic memory
  /// `Itzli Raíz`), excluding internal perception/audio/tool models that are
  /// managed automatically under the hood.
  static List<LiteRtModelInfo> get userVisible =>
      all.where((m) => !m.isInternalOnly).toList(growable: false);

  static List<LiteRtModelInfo> byType(LiteRtModelType type) =>
      all.where((m) => m.type == type).toList(growable: false);

  /// Filtra los modelos ideales para dispositivos móviles de bajos recursos
  /// (por defecto <= 50 MB por modelo), permitiendo correr STT + Intent/Embeddings + TTS
  /// simultáneamente en menos de 65 MB totales de almacenamiento/RAM.
  static List<LiteRtModelInfo> forLowResourceMobile({
    int maxModelSizeBytes = 50 * 1024 * 1024,
  }) =>
      (!maxModelSizeBytes.isFinite || maxModelSizeBytes <= 0)
          ? const []
          : all
              .where((m) => m.sizeBytes <= maxModelSizeBytes)
              .toList(growable: false);

  static const Map<String, String> _legacyIdAliases = {
    'gemma3-270m-web': 'qwen3-06b',
    'gemma3-1b-web': 'qwen3-17b',
    'llama32-1b': 'qwen25-3b',
    'embeddinggemma-300m': 'bert-embedder',
  };

  static LiteRtModelInfo? findById(String id) {
    final canonicalId = _legacyIdAliases[id] ?? id;
    for (final m in all) {
      if (m.id == canonicalId || m.id == id) return m;
    }
    for (final m in catalogEntries) {
      if (m.id == canonicalId || m.id == id) return m;
    }
    return null;
  }

  /// Selects the best zero-token (`requiresHfToken == false`) WebGPU chat model
  /// that fits [availableStorageBytes] and [deviceMemoryGb] for background
  /// internal orchestration in the browser.
  static LiteRtModelInfo? selectRecommendedWebAuxiliaryModel({
    int? availableStorageBytes,
    double? deviceMemoryGb,
  }) {
    const minFugazStorageBytes = 600 * 1024 * 1024; // ~600 MB headroom for 266 MB model
    const minChispaStorageBytes = 240 * 1024 * 1024; // ~240 MB headroom for 190 MB model

    final storageAllowsFugaz = availableStorageBytes == null ||
        availableStorageBytes >= minFugazStorageBytes;
    final memoryAllowsFugaz =
        deviceMemoryGb == null || deviceMemoryGb >= 4.0;
    if (storageAllowsFugaz && memoryAllowsFugaz) {
      return findById('qwen25-05b') ?? findById('smollm2-360m');
    }

    final storageAllowsChispa = availableStorageBytes == null ||
        availableStorageBytes >= minChispaStorageBytes;
    final memoryAllowsChispa =
        deviceMemoryGb == null || deviceMemoryGb >= 2.0;
    if (storageAllowsChispa && memoryAllowsChispa) {
      return findById('smollm2-360m');
    }
    return null;
  }

  /// Selects the strongest local-first primary WebGPU chat model (preferring
  /// Gemma 4 E4B / E2B when browser RAM & OPFS storage allow) to minimize
  /// cloud model usage in Itzli Web.
  static LiteRtModelInfo? selectRecommendedWebPrimaryModel({
    int? availableStorageBytes,
    double? deviceMemoryGb,
  }) {
    const minE4bStorage = 5200 * 1024 * 1024; // ~5.2 GB headroom for 3.3 GB bundle
    const minE2bStorage = 3600 * 1024 * 1024; // ~3.6 GB headroom for 2.34 GB bundle
    const min17bStorage = 1800 * 1024 * 1024; // ~1.8 GB headroom for 980 MB bundle

    final canE4b = (availableStorageBytes == null ||
            availableStorageBytes >= minE4bStorage) &&
        (deviceMemoryGb != null && deviceMemoryGb >= 12.0);
    if (canE4b) {
      return findById('gemma4-e4b-web') ?? findById('gemma4-e2b-web');
    }

    final canE2b = (availableStorageBytes == null ||
            availableStorageBytes >= minE2bStorage) &&
        (deviceMemoryGb == null || deviceMemoryGb >= 8.0);
    if (canE2b) {
      return findById('gemma4-e2b-web');
    }

    final can17b = (availableStorageBytes == null ||
            availableStorageBytes >= min17bStorage) &&
        (deviceMemoryGb == null || deviceMemoryGb >= 4.0);
    if (can17b) {
      return findById('qwen3-17b') ?? findById('qwen25-05b');
    }

    return selectRecommendedWebAuxiliaryModel(
      availableStorageBytes: availableStorageBytes,
      deviceMemoryGb: deviceMemoryGb,
    );
  }
}

/// Snapshot of browser WebGPU, storage quota, and RAM readiness for automatic
/// or 1-click internal orchestration without leaving the main chat view.
class LiteRtBrowserCapabilities {
  final bool hasWebGpu;
  final int? availableStorageBytes;
  final double? deviceMemoryGb;
  final LiteRtModelInfo? recommendedModel;

  const LiteRtBrowserCapabilities({
    required this.hasWebGpu,
    this.availableStorageBytes,
    this.deviceMemoryGb,
    this.recommendedModel,
  });

  /// True when the browser exposes WebGPU and has enough storage/RAM headroom
  /// to run [recommendedModel] for internal orchestration.
  bool get canAddForInternalOrchestration =>
      hasWebGpu && recommendedModel != null;

  /// Human-readable free storage label (e.g. "4.2 GB libres").
  String? get availableStorageLabel {
    final bytes = availableStorageBytes;
    if (bytes == null || !bytes.isFinite || bytes <= 0) return null;
    final gb = bytes / (1024 * 1024 * 1024);
    if (gb >= 1.0) return '${gb.toStringAsFixed(1)} GB libres';
    final mb = bytes / (1024 * 1024);
    return '${mb.toStringAsFixed(0)} MB libres';
  }

  /// Concise status summary for UI badges/tooltips.
  String get summaryLabel {
    if (!hasWebGpu) {
      return 'WebGPU no disponible en este navegador';
    }
    final rec = recommendedModel;
    final parts = <String>['WebGPU activo'];
    final storage = availableStorageLabel;
    if (storage != null) parts.add(storage);
    if (deviceMemoryGb != null && deviceMemoryGb! > 0) {
      parts.add('${deviceMemoryGb!.toStringAsFixed(0)} GB RAM');
    }
    if (rec != null) {
      parts.add('Sugerido: ${rec.name} (${rec.displaySize})');
    } else {
      parts.add('Espacio insuficiente para modelo local');
    }
    return parts.join(' · ');
  }
}

/// Interface representing a loaded and compiled general LiteRT (.tflite) model.
abstract class LiteRtModel {
  /// Runs inference on the model with the given input tensor.
  /// Returns the output tensor as a Float32List.
  Future<Float32List> run(Float32List input, List<int> shape);

  /// Releases the resources allocated for this model.
  Future<void> dispose();
}

/// Embebedor de texto local, sobre MediaPipe Text Embedder.
///
/// Existe aparte de [LiteRtModel] porque ese expone `run(Float32List,
/// shape)` — un único tensor float32 — y un modelo de embeddings tipo BERT
/// necesita tres tensores ENTEROS (`input_ids`, `attention_mask`,
/// `token_type_ids`) más un tokenizador WordPiece. MediaPipe Text Embedder
/// hace la tokenización y el pooling por dentro, así que la interfaz puede
/// ser texto→vector y ya.
abstract class LiteRtTextEmbedder {
  /// Ancho del vector que produce este modelo.
  int get dimensions;

  /// Vectoriza [text]. Síncrono: la API de MediaPipe lo es.
  Float32List embed(String text);

  /// Libera el modelo nativo.
  void dispose();
}

/// Interface representing a loaded and compiled LiteRT-LM language model.
abstract class LiteRtLmModel {
  /// Delegado con el que se compiló realmente: `'GPU'` o `'CPU'`.
  ///
  /// No es cosmético. El delegado de GPU puede rechazar un modelo cuyas
  /// operaciones no soporta, y en ese caso se cae a CPU — que funciona pero
  /// es mucho más lento. La UI tiene que poder decir cuál está corriendo en
  /// vez de afirmar "GPU" siempre.
  String get backend;

  /// Generates a streaming response for the given prompt.
  Stream<String> generateStream(
    String prompt, {
    int maxTokens = 512,
    double temp = 0.4,
    double? temperature,
    double topP = 0.9,
    int topK = 40,
    int? seed,
  });

  /// Releases the resources allocated for this model.
  Future<void> dispose();
}

/// Optional multimodal capability implemented by WebGPU `@litert-lm/core`
/// models (e.g. Gemma 4 E2B / E4B `.litertlm` bundles) supporting local
/// image understanding and persistent KV-cache multi-turn sessions.
abstract interface class LiteRtMultimodalLmModel {
  /// True when the loaded bundle supports inline image inputs in WebGPU.
  bool get supportsImageInput;

  /// Generates a streaming reply conditioned on [prompt] and [images]
  /// (each entry carries raw encoded image bytes + MIME type such as `image/jpeg`).
  Stream<String> generateMultimodalStream(
    String prompt, {
    required List<({Uint8List bytes, String mimeType})> images,
    int maxTokens = 1024,
    double temp = 0.4,
    double topP = 0.9,
    int topK = 40,
    int? seed,
  });

  /// Resets the persistent KV-cache conversation session for this model.
  Future<void> resetConversation();
}
