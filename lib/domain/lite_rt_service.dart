import 'lite_rt_model.dart';

/// Abstract interface for LiteRT and LiteRT-LM web-accelerated runtime services.
abstract class SaiaLiteRtService {
  /// Loads the LiteRT core WebAssembly runtime binaries.
  ///
  /// [wasmPath] is the path (URL) to the directory containing WASM files.
  Future<void> loadLiteRt({String? wasmPath});

  /// Loads and compiles a general `.tflite` model.
  Future<LiteRtModel> loadModel(String modelPath, {String accelerator = 'webgpu'});

  /// Loads the LiteRT-LM runtime for LLMs.
  ///
  /// [moduleUrl] is the URL to load the `@litert-lm/core` package if not preloaded.
  Future<void> loadLiteRtLm({String? moduleUrl});

  /// Loads and compiles a `.litertlm` language model.
  Future<LiteRtLmModel> loadLmModel(String modelPath, {String accelerator = 'gpu'});
}
