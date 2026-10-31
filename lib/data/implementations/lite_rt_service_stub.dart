import '../../domain/lite_rt_model.dart';
import '../../domain/lite_rt_service.dart';

/// Concrete factory to get the platform-specific implementation.
///
/// On native platforms (Android, iOS, macOS), apps can register a native
/// LiteRT-LM provider (e.g. backed by `flutter_gemma_litertlm`) via
/// [registerInstance].
class SaiaLiteRt {
  static SaiaLiteRtService? _customInstance;

  static SaiaLiteRtService get instance =>
      _customInstance ?? _StubLiteRtService();

  static void registerInstance(SaiaLiteRtService service) {
    _customInstance = service;
  }

  static bool get hasCustomInstance => _customInstance != null;
}

class _StubLiteRtService implements SaiaLiteRtService {
  @override
  Future<void> loadLiteRt({String? wasmPath}) async {
    throw UnsupportedError('LiteRT is only supported on the Web platform.');
  }

  @override
  Future<LiteRtModel> loadModel(String modelPath, {String accelerator = 'webgpu'}) async {
    throw UnsupportedError('LiteRT is only supported on the Web platform.');
  }

  /// Fuera de web no hay WebGPU de navegador. `false`, sin lanzar: es una
  /// consulta, y el llamador debe poder preguntarla en cualquier plataforma.
  @override
  Future<bool> isWebGpuAvailable() async => false;

  @override
  Future<LiteRtTextEmbedder> loadTextEmbedder(
    String modelPath, {
    bool l2Normalize = true,
  }) async {
    // En nativo la vía de embeddings es GGUF (fllama), no esta.
    throw UnsupportedError('LiteRT text embedder is Web-only.');
  }

  @override
  Future<void> loadLiteRtLm({String? moduleUrl}) async {
    throw UnsupportedError('LiteRT-LM is only supported on the Web platform.');
  }

  @override
  Future<LiteRtLmModel> loadLmModel(
    String modelPath, {
    String accelerator = 'gpu',
    void Function(String status, double progress)? onProgress,
  }) async {
    throw UnsupportedError('LiteRT-LM is only supported on the Web platform.');
  }

  @override
  Future<void> unloadLmModel(String modelPath) async {}

  @override
  bool isLmModelLoaded(String modelPath) => false;

  @override
  Future<String> verifyVisionModel(String modelPath) async {
    throw UnsupportedError('LiteRT Vision is only supported on the Web platform.');
  }

  @override
  Future<String> verifyAudioModel(String modelPath) async {
    throw UnsupportedError('LiteRT Audio is only supported on the Web platform.');
  }

  @override
  String? get huggingFaceToken => null;

  @override
  void setHuggingFaceToken(String? token) {}
}
