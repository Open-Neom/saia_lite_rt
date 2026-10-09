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

  const LiteRtModelInfo({
    required this.id,
    required this.name,
    required this.description,
    required this.type,
    required this.downloadUrl,
    required this.sizeBytes,
  });

  /// Human-readable file size (e.g. "1.5 GB", "30 MB").
  String get displaySize {
    if (sizeBytes >= 1024 * 1024 * 1024) {
      return '${(sizeBytes / (1024 * 1024 * 1024)).toStringAsFixed(1)} GB';
    }
    return '${(sizeBytes / (1024 * 1024)).toStringAsFixed(0)} MB';
  }
}

/// A built-in catalog of recommended LiteRT and LiteRT-LM models for edge use.
class LiteRtModelCatalog {
  LiteRtModelCatalog._();

  static const List<LiteRtModelInfo> all = [
    // ── Chat (LiteRT-LM) ───────────────────────────────────────────────
    //
    // Gemma NO está en esta lista a propósito. Sus bundles `.task` en
    // HuggingFace (google/gemma-3n-*, litert-community/Gemma3-*) están
    // *gated*: responden 401 sin un token de HF y sin haber aceptado la
    // licencia de Google. Un fetch desde el navegador no puede hacer eso,
    // así que ofrecerlo aquí sería prometer una descarga que siempre falla.
    //
    // Estas dos URLs se verificaron con HEAD: 200, tamaño real, y cabecera
    // CORS presente para origen local.
    LiteRtModelInfo(
      id: 'qwen25-05b',
      name: 'Qwen2.5 0.5B Instruct',
      description: 'Modelo de lenguaje ligero para chat y razonamiento local en el navegador. Descarga rápida.',
      type: LiteRtModelType.chat,
      downloadUrl:
          'https://huggingface.co/litert-community/Qwen2.5-0.5B-Instruct/resolve/main/Qwen2.5-0.5B-Instruct_multi-prefill-seq_q8_ekv1280.task',
      sizeBytes: 546000000, // 521 MB medidos
    ),
    LiteRtModelInfo(
      id: 'qwen25-15b',
      name: 'Qwen2.5 1.5B Instruct',
      description: 'Modelo de lenguaje de mayor capacidad para chat local. Requiere más memoria de GPU.',
      type: LiteRtModelType.chat,
      downloadUrl:
          'https://huggingface.co/litert-community/Qwen2.5-1.5B-Instruct/resolve/main/Qwen2.5-1.5B-Instruct_multi-prefill-seq_q8_ekv1280.task',
      sizeBytes: 1598000000, // 1524 MB medidos
    ),
    LiteRtModelInfo(
      id: 'embeddinggemma-2-740m',
      name: 'EmbeddingGemma 2 (740M Multimodal)',
      description: 'Embeddings multimodales nativos (texto, código, imágenes y audio) en el Edge con 8K contexto y MRL dinámico.',
      type: LiteRtModelType.embeddings,
      downloadUrl:
          'https://huggingface.co/litert-community/embeddinggemma-2-740m-litert-lm/resolve/main/embeddinggemma_2_740m.litertlm',
      sizeBytes: 567000000, // ~567 MB multimodal completo (~191 MB text-only)
    ),
    LiteRtModelInfo(
      id: 'minilm-l6',
      name: 'MiniLM-L6-v2 Embeddings',
      description: 'Generación local de embeddings de texto de alto rendimiento para búsqueda semántica y base de datos vectorial privada.',
      type: LiteRtModelType.embeddings,
      downloadUrl: 'https://huggingface.co/collections/litert-community/minilm-l6-tflite',
      sizeBytes: 30000000, // ~30 MB
    ),
    LiteRtModelInfo(
      id: 'mobilenet-v3',
      name: 'MobileNet V3',
      description: 'Clasificación de imágenes ultra-rápida y ligera para visión por computadora en el cliente.',
      type: LiteRtModelType.vision,
      downloadUrl: 'https://storage.googleapis.com/tensorflow/models/lite/model_tooling/v3/mobilenet_v3_large_100_224_2022_07_18.tflite',
      sizeBytes: 16000000, // ~16 MB
    ),
    LiteRtModelInfo(
      id: 'yolov5-nano',
      name: 'YOLOv5 Nano',
      description: 'Detección y segmentación de objetos en tiempo real optimizado para webcams y cámaras.',
      type: LiteRtModelType.vision,
      downloadUrl: 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n-fp16.tflite',
      sizeBytes: 5000000, // ~5 MB
    ),
    LiteRtModelInfo(
      id: 'yamnet',
      name: 'YAMNet',
      description: 'Clasificador de eventos acústicos en tiempo real capaz de predecir 521 eventos de sonido cotidianos.',
      type: LiteRtModelType.audio,
      downloadUrl: 'https://storage.googleapis.com/tfhub-lite-models/google/lite-model/yamnet/tflite/1.tflite',
      sizeBytes: 15000000, // ~15 MB
    ),
    LiteRtModelInfo(
      id: 'es_MX-claude-high',
      name: 'Claude (LiteRT TTS)',
      description: 'Modelo neural de síntesis de voz (Text-to-Speech) en español local para lectura y respuestas habladas en tiempo real.',
      type: LiteRtModelType.tts,
      downloadUrl: 'https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_MX/claude/high/es_MX-claude-high.tflite',
      sizeBytes: 28000000, // ~28 MB
    ),
    LiteRtModelInfo(
      id: 'matcha-tts',
      name: 'Matcha-TTS',
      description: 'Modelo local FFT-free acelerado por GPU diseñado para inferencia de síntesis de voz en el navegador.',
      type: LiteRtModelType.tts,
      downloadUrl: 'https://huggingface.co/collections/litert-community/matcha-tts-tflite',
      sizeBytes: 25000000, // ~25 MB
    ),
    LiteRtModelInfo(
      id: 'kokoro-82m',
      name: 'Kokoro-82M',
      description: 'Sintetizador de voz local multi-etapa (StyleTTS2 + ISTFTNet) de alta calidad optimizado para ejecuciones offline.',
      type: LiteRtModelType.tts,
      downloadUrl: 'https://huggingface.co/collections/litert-community/kokoro-82m-tflite',
      sizeBytes: 82000000, // ~82 MB
    ),
  ];
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
