import '../../domain/lite_rt_model.dart';
import '../../domain/lite_rt_service.dart';

/// Native NPU / GPU runtime delegate (`LiteRT-LM 0.19` / `litertlm_dart` style)
/// that mobile or desktop hosts can register on [SaiaLiteRt].
typedef NativeLiteRtLmLoader = Future<LiteRtLmModel> Function(
  String modelPath, {
  required String accelerator,
  void Function(String status, double progress)? onProgress,
});

/// Concrete factory to get the platform-specific implementation.
///
/// On native platforms (Android Qualcomm QNN / MediaTek NeuroPilot, iOS CoreML
/// ANE, macOS), apps can register a native LiteRT-LM provider via
/// [registerInstance] or [registerNativeLmLoader].
class SaiaLiteRt {
  static SaiaLiteRtService? _customInstance;
  static NativeLiteRtLmLoader? _nativeLmLoader;
  static final _StubLiteRtService _nativeDefault = _StubLiteRtService();

  static SaiaLiteRtService get instance => _customInstance ?? _nativeDefault;

  static void registerInstance(SaiaLiteRtService service) {
    _customInstance = service;
  }

  static void registerNativeLmLoader(NativeLiteRtLmLoader? loader) {
    _nativeLmLoader = loader;
  }

  static bool get hasCustomInstance =>
      _customInstance != null || _nativeLmLoader != null;

  /// Recommends the optimal `LiteRT-LM 0.19` hardware delegate for the SoC.
  static String recommendMobileAccelerator({
    required String socVendorOrModel,
    bool isIos = false,
  }) {
    if (isIos) return 'coreml';
    final lower = socVendorOrModel.toLowerCase();
    if (lower.contains('snapdragon') ||
        lower.contains('qcom') ||
        lower.contains('qualcomm') ||
        lower.contains('sm8')) {
      return 'qnn';
    }
    if (lower.contains('dimensity') ||
        lower.contains('mediatek') ||
        lower.contains('mt68') ||
        lower.contains('mt69')) {
      return 'neuropilot';
    }
    return 'gpu';
  }
}

class _StubLiteRtService implements SaiaLiteRtService {
  final Map<String, LiteRtLmModel> _loadedNativeModels = {};
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
    final cleanAccel = LiteRtUrlPolicy.validateAccelerator(accelerator);
    final trimmed = modelPath.trim();
    final lower = trimmed.toLowerCase();
    final isRemoteOrBlob = lower.startsWith('http://') ||
        lower.startsWith('https://') ||
        lower.startsWith('blob:');
    final isLocalNativeBundle = !isRemoteOrBlob &&
        !_controlOrSchemeLike(modelPath) &&
        (lower.endsWith('.litertlm') ||
            lower.endsWith('.task') ||
            lower.endsWith('.bin') ||
            lower.endsWith('.tflite'));
    if (isRemoteOrBlob || _controlOrSchemeLike(modelPath)) {
      LiteRtUrlPolicy.validateModelAssetUrl(modelPath);
    } else if (isLocalNativeBundle) {
      if (trimmed.contains('..') ||
          RegExp(r'[\x00-\x1F\x7F]').hasMatch(trimmed)) {
        throw ArgumentError.value(modelPath, 'modelPath', 'Invalid local LiteRT-LM path');
      }
    } else {
      LiteRtUrlPolicy.validateWebLlmModelId(modelPath);
    }

    final loader = SaiaLiteRt._nativeLmLoader;
    if (loader != null) {
      final existing = _loadedNativeModels[trimmed];
      if (existing != null) return existing;
      final loaded = await loader(
        trimmed,
        accelerator: cleanAccel,
        onProgress: onProgress,
      );
      _loadedNativeModels[trimmed] = loaded;
      return loaded;
    }
    throw UnsupportedError(
      'LiteRT-LM native delegate not registered; register via SaiaLiteRt.registerNativeLmLoader or run on Web.',
    );
  }

  static bool _controlOrSchemeLike(String s) => s.contains('://') || s.startsWith('//');

  @override
  Future<void> unloadLmModel(String modelPath) async {
    final model = _loadedNativeModels.remove(modelPath.trim());
    if (model != null) {
      await model.dispose();
    }
  }

  @override
  bool isLmModelLoaded(String modelPath) =>
      _loadedNativeModels.containsKey(modelPath.trim());

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
