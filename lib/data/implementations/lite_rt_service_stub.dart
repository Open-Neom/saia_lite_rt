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
    if (wasmPath != null) {
      LiteRtUrlPolicy.validateTrustedModuleUrl(wasmPath);
    }
    throw UnsupportedError('LiteRT is only supported on the Web platform.');
  }

  @override
  Future<LiteRtModel> loadModel(String modelPath, {String accelerator = 'webgpu'}) async {
    LiteRtUrlPolicy.validateModelAssetUrl(modelPath);
    throw UnsupportedError('LiteRT is only supported on the Web platform.');
  }

  /// Fuera de web no hay WebGPU de navegador. `false`, sin lanzar: es una
  /// consulta, y el llamador debe poder preguntarla en cualquier plataforma.
  @override
  Future<bool> isWebGpuAvailable() async => false;

  @override
  Future<LiteRtBrowserCapabilities> probeBrowserCapabilities() async =>
      const LiteRtBrowserCapabilities(hasWebGpu: false);

  @override
  Future<LiteRtTextEmbedder> loadTextEmbedder(
    String modelPath, {
    bool l2Normalize = true,
  }) async {
    LiteRtUrlPolicy.validateModelAssetUrl(modelPath);
    // En nativo la vía de embeddings es GGUF (fllama), no esta.
    throw UnsupportedError('LiteRT text embedder is Web-only.');
  }

  @override
  Future<void> loadLiteRtLm({String? moduleUrl}) async {
    if (moduleUrl != null) {
      LiteRtUrlPolicy.validateTrustedModuleUrl(moduleUrl);
    }
    throw UnsupportedError('LiteRT-LM is only supported on the Web platform.');
  }

  @override
  Future<LiteRtLmModel> loadLmModel(
    String modelPath, {
    String accelerator = 'gpu',
    void Function(String status, double progress)? onProgress,
  }) async {
    final trimmed = modelPath.trim();
    final lower = trimmed.toLowerCase();
    final isBundle = lower.endsWith('.task') ||
        lower.endsWith('.bin') ||
        lower.startsWith('http://') ||
        lower.startsWith('https://') ||
        lower.startsWith('blob:');
    if (isBundle || _controlOrSchemeLike(modelPath)) {
      LiteRtUrlPolicy.validateModelAssetUrl(modelPath);
    } else {
      LiteRtUrlPolicy.validateWebLlmModelId(modelPath);
    }
    throw UnsupportedError('LiteRT-LM is only supported on the Web platform.');
  }

  static bool _controlOrSchemeLike(String s) => s.contains('://') || s.startsWith('//');

  @override
  Future<void> unloadLmModel(String modelPath) async {}

  @override
  bool isLmModelLoaded(String modelPath) => false;

  @override
  Future<String> verifyVisionModel(String modelPath) async {
    LiteRtUrlPolicy.validateModelAssetUrl(modelPath);
    throw UnsupportedError('LiteRT Vision is only supported on the Web platform.');
  }

  @override
  Future<String> verifyAudioModel(String modelPath) async {
    LiteRtUrlPolicy.validateModelAssetUrl(modelPath);
    throw UnsupportedError('LiteRT Audio is only supported on the Web platform.');
  }

  String? _hfToken;

  @override
  String? get huggingFaceToken => _hfToken;

  @override
  void setHuggingFaceToken(String? token) {
    _hfToken = LiteRtUrlPolicy.sanitizeHfToken(token);
  }
}
