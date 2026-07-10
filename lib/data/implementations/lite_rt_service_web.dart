@JS()
library;

import 'dart:async';
import 'dart:convert';
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:typed_data';
import 'package:web/web.dart' as web;

import '../../domain/lite_rt_model.dart';
import '../../domain/lite_rt_service.dart';

/// Concrete factory to get the platform-specific implementation.
class SaiaLiteRt {
  static SaiaLiteRtService get instance => WebLiteRtService();
}

// ==========================================
// General LiteRT (@litertjs/core) JS Bindings
// ==========================================

@JS('loadLiteRt')
external JSPromise<JSAny?> _jsLoadLiteRt(JSString wasmPath);

@JS('loadAndCompile')
external JSPromise<_JsLiteRtModel> _jsLoadAndCompile(
  JSString modelPath,
  _JsLiteRtOptions options,
);

@JS('Tensor')
extension type _JsLiteRtTensor._(JSObject _) implements JSObject {
  external factory _JsLiteRtTensor(JSFloat32Array data, JSArray<JSNumber> shape);
  external JSPromise<_JsLiteRtTensor> moveTo(JSString target);
  external JSFloat32Array toTypedArray();
}

@JS()
@anonymous
extension type _JsLiteRtOptions._(JSObject _) implements JSObject {
  external factory _JsLiteRtOptions({JSString accelerator});
}

@JS()
extension type _JsLiteRtModel._(JSObject _) implements JSObject {
  external JSPromise<JSArray<_JsLiteRtTensor>> run(_JsLiteRtTensor input);
}

// ==========================================
// Generative LLM (@litert-lm/core) JS Bindings
// ==========================================

@JS('LiteRtLmEngine')
extension type _LiteRtLmEngineConstructor._(JSObject _) implements JSObject {
  external JSPromise<_LiteRtLmWebEngine> create(JSObject settings);
}

@JS()
extension type _LiteRtLmWebEngine._(JSObject _) implements JSObject {
  external JSPromise<_LiteRtLmWebConversation> createConversation([
    JSObject? config,
  ]);

  @JS('delete')
  external JSPromise<JSAny?>? deleteEngine();
}

@JS()
extension type _LiteRtLmWebConversation._(JSObject _) implements JSObject {
  external _LiteRtLmReadableStream sendMessageStreaming(String message);
  external void cancel();

  @JS('delete')
  external JSPromise<JSAny?>? deleteConversation();
}

@JS()
extension type _LiteRtLmReadableStream._(JSObject _) implements JSObject {
  external _LiteRtLmReadableStreamReader getReader();
}

@JS()
extension type _LiteRtLmReadableStreamReader._(JSObject _) implements JSObject {
  external JSPromise<_LiteRtLmStreamReadResult> read();
  external JSPromise<JSAny?>? cancel();
  external void releaseLock();
}

@JS()
extension type _LiteRtLmStreamReadResult._(JSObject _) implements JSObject {
  external bool get done;
  external JSAny? get value;
}

// ==========================================
// Service Implementation
// ==========================================

class WebLiteRtService implements SaiaLiteRtService {
  @override
  Future<void> loadLiteRt({String? wasmPath}) async {
    if (!web.window.has('loadLiteRt')) {
      await _loadModuleScriptGeneral('https://cdn.jsdelivr.net/npm/@litertjs/core/+esm');
    }
    final wasmUrl = wasmPath ?? 'https://cdn.jsdelivr.net/npm/@litertjs/core/wasm/';
    await _jsLoadLiteRt(wasmUrl.toJS).toDart;
  }

  @override
  Future<LiteRtModel> loadModel(String modelPath, {String accelerator = 'webgpu'}) async {
    final options = _JsLiteRtOptions(accelerator: accelerator.toJS);
    final jsModel = await _jsLoadAndCompile(modelPath.toJS, options).toDart;
    return LiteRtModelWeb(jsModel);
  }

  @override
  Future<void> loadLiteRtLm({String? moduleUrl}) async {
    if (web.window.has('LiteRtLmEngine')) {
      return;
    }

    final loaderUrl = moduleUrl ?? 'https://cdn.jsdelivr.net/npm/@litert-lm/core/+esm';
    await _loadModuleScript(loaderUrl);
  }

  Future<void> _loadModuleScript(String moduleUrl) async {
    final completer = Completer<void>();
    const callbackName = '__saia_litert_lm_init';
    
    web.window.setProperty(
      callbackName.toJS,
      (JSAny? err) {
        if (err != null) {
          if (!completer.isCompleted) {
            completer.completeError(
              StateError('LiteRT-LM web module load failed: $err'),
            );
          }
          return;
        }
        if (!completer.isCompleted) {
          completer.complete();
        }
      }.toJS,
    );

    final script = web.HTMLScriptElement();
    script.type = 'module';
    script.text = '''
      import(${jsonEncode(moduleUrl)}).then(mod => {
        if (mod?.Engine) {
          window.LiteRtLmEngine = mod.Engine;
        }
        if (mod?.Backend) {
          window.LiteRtLmBackendEnum = mod.Backend;
        }
        if (window.$callbackName) {
          window.$callbackName(null);
        }
      }).catch(e => {
        if (window.$callbackName) {
          window.$callbackName(e);
        }
      });
    ''';

    script.addEventListener(
      'error',
      ((web.Event _) {
        if (!completer.isCompleted) {
          completer.completeError(
            StateError('Failed to load LiteRT-LM web module script'),
          );
        }
      }).toJS,
    );

    web.document.head?.append(script);
    try {
      await completer.future.timeout(const Duration(seconds: 15));
    } finally {
      web.window.delete(callbackName.toJS);
    }
  }

  Future<void> _loadModuleScriptGeneral(String moduleUrl) async {
    final completer = Completer<void>();
    const callbackName = '__saia_litert_core_init';
    
    web.window.setProperty(
      callbackName.toJS,
      (JSAny? err) {
        if (err != null) {
          if (!completer.isCompleted) {
            completer.completeError(
              StateError('LiteRT core web module load failed: $err'),
            );
          }
          return;
        }
        if (!completer.isCompleted) {
          completer.complete();
        }
      }.toJS,
    );

    final script = web.HTMLScriptElement();
    script.type = 'module';
    script.text = '''
      import(${jsonEncode(moduleUrl)}).then(mod => {
        if (mod?.loadLiteRt) {
          window.loadLiteRt = mod.loadLiteRt;
        }
        if (mod?.loadAndCompile) {
          window.loadAndCompile = mod.loadAndCompile;
        }
        if (mod?.Tensor) {
          window.LiteRtTensor = mod.Tensor;
        }
        if (window.$callbackName) {
          window.$callbackName(null);
        }
      }).catch(e => {
        if (window.$callbackName) {
          window.$callbackName(e);
        }
      });
    ''';

    script.addEventListener(
      'error',
      ((web.Event _) {
        if (!completer.isCompleted) {
          completer.completeError(
            StateError('Failed to load LiteRT core web module script'),
          );
        }
      }).toJS,
    );

    web.document.head?.append(script);
    try {
      await completer.future.timeout(const Duration(seconds: 15));
    } finally {
      web.window.delete(callbackName.toJS);
    }
  }

  @override
  Future<LiteRtLmModel> loadLmModel(String modelPath, {String accelerator = 'gpu'}) async {
    final raw = web.window.getProperty('LiteRtLmEngine'.toJS);
    if (raw == null || !raw.isA<JSObject>()) {
      throw StateError('LiteRT-LM engine constructor is not available on window. Make sure to call loadLiteRtLm first.');
    }

    final constructor = _LiteRtLmEngineConstructor._(raw as JSObject);
    
    final settings = JSObject();
    settings.setProperty('model'.toJS, modelPath.toJS);
    
    final backendVal = accelerator == 'cpu' ? 3 : 2; // 3 = CPU, 2 = GPU_ARTISAN
    settings.setProperty('backend'.toJS, backendVal.toJS);

    final executorSettings = JSObject();
    executorSettings.setProperty('maxNumTokens'.toJS, 2048.toJS);
    settings.setProperty('mainExecutorSettings'.toJS, executorSettings);

    final jsEngine = await constructor.create(settings).toDart;
    return LiteRtLmModelWeb(jsEngine);
  }
}

// ==========================================
// Model Wrappers
// ==========================================

class LiteRtModelWeb implements LiteRtModel {
  final _JsLiteRtModel _jsModel;

  LiteRtModelWeb(this._jsModel);

  @override
  Future<Float32List> run(Float32List input, List<int> shape) async {
    final jsData = input.toJS;
    final jsShape = JSArray<JSNumber>();
    for (final s in shape) {
      jsShape.add(s.toJS);
    }

    final inputTensor = _JsLiteRtTensor(jsData, jsShape);
    final results = await _jsModel.run(inputTensor).toDart;

    final outputTensor = results.toDart[0];
    final wasmTensor = await outputTensor.moveTo('wasm'.toJS).toDart;
    final array = wasmTensor.toTypedArray();

    return array.toDart;
  }

  @override
  Future<void> dispose() async {}
}

class LiteRtLmModelWeb implements LiteRtLmModel {
  final _LiteRtLmWebEngine _jsEngine;
  _LiteRtLmWebConversation? _activeConversation;

  LiteRtLmModelWeb(this._jsEngine);

  @override
  Stream<String> generateStream(
    String prompt, {
    int maxTokens = 512,
    double temp = 0.4,
    double topP = 0.9,
    int topK = 40,
    int? seed,
  }) {
    final controller = StreamController<String>();

    _runGeneration(
      controller,
      prompt,
      maxTokens: maxTokens,
      temp: temp,
      topP: topP,
      topK: topK,
      seed: seed,
    );

    return controller.stream;
  }

  Future<void> _runGeneration(
    StreamController<String> controller,
    String prompt, {
    required int maxTokens,
    required double temp,
    required double topP,
    required int topK,
    required int? seed,
  }) async {
    try {
      final config = JSObject();
      final sessionConfig = JSObject();
      sessionConfig.setProperty('maxOutputTokens'.toJS, maxTokens.toJS);

      final samplerParams = JSObject();
      samplerParams.setProperty('k'.toJS, topK.toJS);
      samplerParams.setProperty('p'.toJS, topP.toJS);
      samplerParams.setProperty('temperature'.toJS, temp.toJS);
      if (seed != null) {
        samplerParams.setProperty('seed'.toJS, seed.toJS);
      }
      sessionConfig.setProperty('samplerParams'.toJS, samplerParams);
      config.setProperty('sessionConfig'.toJS, sessionConfig);

      final conversation = await _jsEngine.createConversation(config).toDart;
      _activeConversation = conversation;

      final reader = conversation.sendMessageStreaming(prompt).getReader();

      while (true) {
        final result = await reader.read().toDart;
        if (result.done) {
          break;
        }

        final chunk = _extractTextChunk(result.value);
        if (chunk.isNotEmpty) {
          controller.add(chunk);
        }
      }

      reader.releaseLock();
      controller.close();
    } catch (e) {
      controller.addError(e);
      controller.close();
    } finally {
      final conversation = _activeConversation;
      _activeConversation = null;
      if (conversation != null) {
        final deleteResult = conversation.deleteConversation();
        if (deleteResult != null) {
          await deleteResult.toDart;
        }
      }
    }
  }

  String _extractTextChunk(JSAny? value) {
    if (value == null) return '';
    if (value.isA<JSString>()) {
      return (value as JSString).toDart;
    }
    if (!value.isA<JSObject>()) return '';

    final obj = value as JSObject;
    final directText = obj.getProperty('text'.toJS);
    if (directText != null && directText.isA<JSString>()) {
      return (directText as JSString).toDart;
    }

    final content = obj.getProperty('content'.toJS);
    if (content == null || !content.isA<JSArray>()) return '';

    final buffer = StringBuffer();
    final items = content as JSArray;
    for (var i = 0; i < items.length; i += 1) {
      final item = items.getProperty(i.toJS);
      if (item == null || !item.isA<JSObject>()) continue;
      final text = (item as JSObject).getProperty('text'.toJS);
      if (text != null && text.isA<JSString>()) {
        buffer.write((text as JSString).toDart);
      }
    }
    return buffer.toString();
  }

  @override
  Future<void> dispose() async {
    final conversation = _activeConversation;
    _activeConversation = null;
    if (conversation != null) {
      final result = conversation.deleteConversation();
      if (result != null) {
        await result.toDart;
      }
    }
    final deleteResult = _jsEngine.deleteEngine();
    if (deleteResult != null) {
      await deleteResult.toDart;
    }
  }
}
