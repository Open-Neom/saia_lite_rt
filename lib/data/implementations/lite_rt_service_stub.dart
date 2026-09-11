import '../../domain/lite_rt_model.dart';
import '../../domain/lite_rt_service.dart';

/// Concrete factory to get the platform-specific implementation.
class SaiaLiteRt {
  static SaiaLiteRtService get instance => _StubLiteRtService();
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
  Future<LiteRtLmModel> loadLmModel(String modelPath, {String accelerator = 'gpu'}) async {
    throw UnsupportedError('LiteRT-LM is only supported on the Web platform.');
  }
}
