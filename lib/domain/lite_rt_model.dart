import 'dart:typed_data';

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
  });

  /// Human-readable file size (e.g. "1.5 GB", "30 MB").
  String get displaySize {
    if (sizeBytes >= 1024 * 1024 * 1024) {
      return '${(sizeBytes / (1024 * 1024 * 1024)).toStringAsFixed(1)} GB';
    }
    return '${(sizeBytes / (1024 * 1024)).toStringAsFixed(0)} MB';
  }
}

/// A built-in catalog of verified WebGPU / WASM models for browser edge execution.
class LiteRtModelCatalog {
  LiteRtModelCatalog._();

  static const String kHfTokensUrl = 'https://huggingface.co/settings/tokens';

  static const List<LiteRtModelInfo> all = [
    // ── 1. Chat (WebGPU directo — 100% libres, sin token) ──────────────
    LiteRtModelInfo(
      id: 'qwen25-05b',
      name: 'Qwen2.5 0.5B Instruct (WebGPU)',
      description:
          'Modelo multilingüe en español compilado para WebGPU (MLC q4f16). '
          'Descarga ligera y respuesta fluida en el navegador sin enviar datos.',
      type: LiteRtModelType.chat,
      downloadUrl: 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC',
      sizeBytes: 278921216, // ~266 MB verificados en Chrome WebGPU
    ),
    LiteRtModelInfo(
      id: 'smollm2-360m',
      name: 'SmolLM2 360M Instruct (WebGPU)',
      description:
          'Modelo ultraligero para equipos con poca memoria de video (376 MB VRAM). '
          'Ideal para pruebas rápidas en navegador.',
      type: LiteRtModelType.chat,
      downloadUrl: 'SmolLM2-360M-Instruct-q4f16_1-MLC',
      sizeBytes: 199229440, // ~190 MB
    ),
    LiteRtModelInfo(
      id: 'llama32-1b',
      name: 'Llama 3.2 1B Instruct (WebGPU)',
      description:
          'Modelo compacto de Meta (1B parámetros, ventana 4K) acelerado por WebGPU.',
      type: LiteRtModelType.chat,
      downloadUrl: 'Llama-3.2-1B-Instruct-q4f16_1-MLC',
      sizeBytes: 660602880, // ~630 MB
    ),
    LiteRtModelInfo(
      id: 'qwen25-15b',
      name: 'Qwen2.5 1.5B Instruct (WebGPU)',
      description:
          'Mayor capacidad de razonamiento y redacción en español (1.5B parámetros, '
          'ventana 4K). Requiere ~1.6 GB de VRAM.',
      type: LiteRtModelType.chat,
      downloadUrl: 'Qwen2.5-1.5B-Instruct-q4f16_1-MLC',
      sizeBytes: 922746880, // ~880 MB
    ),

    // ── 2. Chat Gated (Requieren aceptar licencia + Token HF) ──────────
    LiteRtModelInfo(
      id: 'gemma3-270m-web',
      name: 'Gemma 3 270M IT (MediaPipe Web)',
      description:
          'Bundle oficial -web.task de Google para MediaPipe WebGPU. '
          'Requiere aceptar la licencia de Gemma en HuggingFace y configurar un token de lectura (hf_...).',
      type: LiteRtModelType.chat,
      downloadUrl:
          'https://huggingface.co/litert-community/gemma-3-270m-it/resolve/main/gemma3-270m-it-q8-web.task',
      sizeBytes: 293601280, // ~280 MB
      requiresHfToken: true,
      licenseUrl: 'https://huggingface.co/litert-community/gemma-3-270m-it',
      tokenSetupUrl: kHfTokensUrl,
      requirementSteps: [
        'Abre el repositorio en HuggingFace y acepta la licencia de uso de Google Gemma.',
        'Genera un Access Token de tipo Read (empieza con hf_...) en tu cuenta de HuggingFace.',
        'Pega tu token en el botón "Configurar Token HF" para desbloquear la descarga.',
      ],
    ),
    LiteRtModelInfo(
      id: 'gemma3-1b-web',
      name: 'Gemma 3 1B IT (MediaPipe Web)',
      description:
          'Modelo Gemma 3 de 1B cuantizado a int4 específicamente para WebGPU (-web.task). '
          'Requiere aceptar licencia en HuggingFace y token hf_...',
      type: LiteRtModelType.chat,
      downloadUrl:
          'https://huggingface.co/litert-community/Gemma3-1B-IT/resolve/main/gemma3-1b-it-int4-web.task',
      sizeBytes: 576716800, // ~550 MB
      requiresHfToken: true,
      licenseUrl: 'https://huggingface.co/litert-community/Gemma3-1B-IT',
      tokenSetupUrl: kHfTokensUrl,
      requirementSteps: [
        'Abre el repositorio litert-community/Gemma3-1B-IT y acepta la licencia de Google.',
        'Crea un token de lectura (Read) en HuggingFace Settings -> Access Tokens.',
        'Guarda tu token hf_... aquí para autenticar la descarga desde el navegador.',
      ],
    ),

    // ── 3. Embeddings Semánticos (WASM — MediaPipe Text Embedder) ──────
    LiteRtModelInfo(
      id: 'universal-sentence-encoder',
      name: 'Universal Sentence Encoder (Multilingüe)',
      description:
          'Embeddings semánticos ultrarrápidos (100 dimensiones, L2-normalizados) '
          'para memoria episódica y búsqueda por significado en el navegador.',
      type: LiteRtModelType.embeddings,
      downloadUrl:
          'https://storage.googleapis.com/mediapipe-models/text_embedder/universal_sentence_encoder/float32/1/universal_sentence_encoder.tflite',
      sizeBytes: 6120274, // 6 MB verificados
    ),
    LiteRtModelInfo(
      id: 'bert-embedder',
      name: 'BERT Text Embedder (512 dims)',
      description:
          'Embeddings densos de mayor precisión basados en BERT (512 dimensiones, '
          'con tokenizador WordPiece integrado en el bundle .tflite).',
      type: LiteRtModelType.embeddings,
      downloadUrl:
          'https://storage.googleapis.com/mediapipe-models/text_embedder/bert_embedder/float32/1/bert_embedder.tflite',
      sizeBytes: 26142232, // 25 MB verificados
    ),
    LiteRtModelInfo(
      id: 'embeddinggemma-300m',
      name: 'EmbeddingGemma 300M (LiteRT)',
      description:
          'Modelo de embeddings de Google (seq256, precisión mixta). '
          'Requiere aceptar licencia en HuggingFace y token hf_...',
      type: LiteRtModelType.embeddings,
      downloadUrl:
          'https://huggingface.co/litert-community/embeddinggemma-300m/resolve/main/embeddinggemma-300M_seq256_mixed-precision.tflite',
      sizeBytes: 199229440, // ~190 MB
      requiresHfToken: true,
      licenseUrl: 'https://huggingface.co/litert-community/embeddinggemma-300m',
      tokenSetupUrl: kHfTokensUrl,
      requirementSteps: [
        'Acepta la licencia del repositorio litert-community/embeddinggemma-300m en HuggingFace.',
        'Genera un token de lectura (hf_...) en HuggingFace Settings -> Access Tokens.',
        'Configura tu token hf_... para descargar y verificar el modelo.',
      ],
    ),

    // ── 4. Visión por Computadora (WASM/WebGPU — MediaPipe Vision) ─────
    LiteRtModelInfo(
      id: 'mobilenet-v3',
      name: 'EfficientNet-Lite0 (Clasificador Visual)',
      description:
          'Clasificación de imágenes en tiempo real (1,000 categorías ImageNet) '
          'ejecutándose 100% en el navegador vía MediaPipe Vision.',
      type: LiteRtModelType.vision,
      downloadUrl:
          'https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/1/efficientnet_lite0.tflite',
      sizeBytes: 18582189, // 18 MB verificados
    ),
    LiteRtModelInfo(
      id: 'ssd-mobilenet-v2',
      name: 'SSD MobileNet V2 (Detector de Objetos)',
      description:
          'Detección de múltiples objetos y cajas delimitadoras (80 clases COCO) '
          'optimizado en float16 para cámaras y lienzos web.',
      type: LiteRtModelType.vision,
      downloadUrl:
          'https://storage.googleapis.com/mediapipe-models/object_detector/ssd_mobilenet_v2/float16/1/ssd_mobilenet_v2.tflite',
      sizeBytes: 5918749, // 5.6 MB verificados
    ),

    // ── 5. Audio (WASM — MediaPipe Audio Classifier) ───────────────────
    LiteRtModelInfo(
      id: 'yamnet',
      name: 'YAMNet (Clasificador Acústico 521 Eventos)',
      description:
          'Reconocimiento de 521 eventos acústicos (voz, música, alarmas, '
          'naturaleza) a 16 kHz directamente en el navegador.',
      type: LiteRtModelType.audio,
      downloadUrl:
          'https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite',
      sizeBytes: 4126810, // 4 MB verificados
    ),
    // ── Modelos Ultra-Ligeros Embebidos (codebase_ia: KittenTTS, Audio8, Needle 3, Whistle) ──
    LiteRtModelInfo(
      id: 'kitten-tts-nano-int8',
      name: 'KittenTTS Nano 0.8 (15M INT8)',
      description: 'Síntesis de voz neural ultra-ligera (15M parámetros, <25 MB ONNX/LiteRT) que corre en tiempo real en CPU sin GPU.',
      type: LiteRtModelType.tts,
      downloadUrl: 'https://huggingface.co/KittenML/kitten-tts-nano-0.8-int8/resolve/main/kitten_tts_nano_v0_8.onnx',
      sizeBytes: 25000000, // ~25 MB
    ),
    LiteRtModelInfo(
      id: 'kitten-tts-micro',
      name: 'KittenTTS Micro 0.8 (40M)',
      description: 'Síntesis de voz expresiva local (40M parámetros, 41 MB) para anuncios comunitarios y alertas audibles off-grid.',
      type: LiteRtModelType.tts,
      downloadUrl: 'https://huggingface.co/KittenML/kitten-tts-micro-0.8/resolve/main/kitten_tts_micro_v0_8.onnx',
      sizeBytes: 41000000, // ~41 MB
    ),
    LiteRtModelInfo(
      id: 'kitten-tts-2-emb4',
      name: 'KittenTTS 2 (1.7B 1-bit emb4 Multilingüe)',
      description: 'TTS multilingüe (20 idiomas incl. Español, 47 voces) con etiquetas de emoción [excited]/[whisper] y clonación de voz en CPU.',
      type: LiteRtModelType.tts,
      downloadUrl: 'https://huggingface.co/KittenML/kitten-tts-2-emb4/resolve/main/kitten_tts_2_emb4.gguf',
      sizeBytes: 530579456, // ~506 MiB (1-bit weights + 4-bit embeddings)
    ),
    LiteRtModelInfo(
      id: 'audio8-tts-01b-int8',
      name: 'Audio8-TTS 0.1B (Mamba2 + Attention INT8)',
      description: 'TTS multilingüe con soporte nativo de Español y clonación zero-shot basado en arquitectura híbrida Falcon-H1 Mamba2.',
      type: LiteRtModelType.tts,
      downloadUrl: 'https://huggingface.co/Audio8/Audio8-TTS-0.1B-ONNX-INT8/resolve/main/model_int8.onnx',
      sizeBytes: 95000000, // ~95 MB
    ),
    LiteRtModelInfo(
      id: 'needle-3-mobile',
      name: 'Needle 3 (2.125-bit .cact Tool Calling + Embeddings)',
      description: 'Motor unificado de 18 MB para extracción JSON restringida por gramática, clasificación de emergencias, tool-calling y embeddings.',
      type: LiteRtModelType.chat,
      downloadUrl: 'https://huggingface.co/cactus-compute/needle-3-mobile/resolve/main/needle-3.cact',
      sizeBytes: 18874368, // ~18 MB (rango 8-29 MB en 2.125 bits/peso)
    ),
    LiteRtModelInfo(
      id: 'whistle-stt-int4',
      name: 'Whistle STT (16.9 MB Speech-to-Text)',
      description: 'Reconocimiento de voz a texto offline en Español e Inglés (8.6x más pequeño que Whisper base) para comandos de voz comunitarios.',
      type: LiteRtModelType.audio,
      downloadUrl: 'https://huggingface.co/cactus-compute/whistle-stt/resolve/main/whistle.cact',
      sizeBytes: 17720934, // ~16.9 MB
    ),
  ];

  static List<LiteRtModelInfo> byType(LiteRtModelType type) =>
      all.where((m) => m.type == type).toList(growable: false);

  /// Filtra los modelos ideales para dispositivos móviles de bajos recursos
  /// (por defecto <= 50 MB por modelo), permitiendo correr STT + Intent/Embeddings + TTS
  /// simultáneamente en menos de 65 MB totales de almacenamiento/RAM.
  static List<LiteRtModelInfo> forLowResourceMobile({
    int maxModelSizeBytes = 50 * 1024 * 1024,
  }) =>
      all.where((m) => m.sizeBytes <= maxModelSizeBytes).toList(growable: false);

  static LiteRtModelInfo? findById(String id) {
    for (final m in all) {
      if (m.id == id) return m;
    }
    return null;
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
    double topP = 0.9,
    int topK = 40,
    int? seed,
  });

  /// Releases the resources allocated for this model.
  Future<void> dispose();
}
