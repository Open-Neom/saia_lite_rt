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

  @override
  Future<void> loadLiteRtLm({String? moduleUrl}) async {
    throw UnsupportedError('LiteRT-LM is only supported on the Web platform.');
  }

  @override
  Future<LiteRtLmModel> loadLmModel(String modelPath, {String accelerator = 'gpu'}) async {
    throw UnsupportedError('LiteRT-LM is only supported on the Web platform.');
  }
}
