import 'lite_rt_model.dart';

/// Abstract interface for LiteRT and LiteRT-LM web-accelerated runtime services.
abstract class SaiaLiteRtService {
  /// Loads the LiteRT core WebAssembly runtime binaries.
  ///
  /// [wasmPath] is the path (URL) to the directory containing WASM files.
  Future<void> loadLiteRt({String? wasmPath});

  /// Loads and compiles a general `.tflite` model.
  Future<LiteRtModel> loadModel(String modelPath, {String accelerator = 'webgpu'});

  /// Whether this platform really exposes a usable WebGPU adapter.
  ///
  /// Comprueba `navigator.gpu` y pide un adaptador de verdad, en vez de
  /// deducirlo del texto de un error. Sin esto es fácil culpar a WebGPU de
  /// fallos que no tienen nada que ver — descargas 401, CORS, memoria — y
  /// mandar al usuario a "activar WebGPU" cuando ya está activo.
  Future<bool> isWebGpuAvailable();

  /// Loads the LiteRT-LM runtime for LLMs.
  ///
  /// [moduleUrl] apunta al paquete de MediaPipe GenAI si no está precargado.
  Future<void> loadLiteRtLm({String? moduleUrl});

  /// Loads and compiles a language model (WebLLM MLC ID or MediaPipe `.task` URL).
  Future<LiteRtLmModel> loadLmModel(
    String modelPath, {
    String accelerator = 'gpu',
    void Function(String status, double progress)? onProgress,
  });

  /// Explicitly unloads a previously loaded language model from GPU/WASM memory.
  Future<void> unloadLmModel(String modelPath);

  /// Returns true if [modelPath] is currently loaded and ready in memory.
  bool isLmModelLoaded(String modelPath);

  /// Carga un embebedor de texto desde un `.tflite` con metadatos de
  /// tokenizador (los de MediaPipe Text Embedder).
  ///
  /// [l2Normalize] deja el vector con norma 1, que es lo que hace que un
  /// producto interno sea similitud coseno — el modo en que compara
  /// saia_turbovec.
  Future<LiteRtTextEmbedder> loadTextEmbedder(
    String modelPath, {
    bool l2Normalize = true,
  });

  /// Carga y verifica en vivo un modelo `.tflite` de visión (ImageClassifier
  /// u ObjectDetector de MediaPipe Vision) y devuelve el diagnóstico real.
  Future<String> verifyVisionModel(String modelPath);

  /// Carga y verifica en vivo un modelo `.tflite` de audio (AudioClassifier
  /// de MediaPipe Audio, p. ej. YAMNet) con un tono de prueba a 16 kHz.
  Future<String> verifyAudioModel(String modelPath);

  /// Token opcional de HuggingFace (`hf_...`) para descargar modelos *gated*
  /// (como Gemma 3 `-web.task` o EmbeddingGemma).
  String? get huggingFaceToken;
  void setHuggingFaceToken(String? token);
}
