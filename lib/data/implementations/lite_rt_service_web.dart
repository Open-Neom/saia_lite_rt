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
  static SaiaLiteRtService? _customInstance;
  static final WebLiteRtService _webDefault = WebLiteRtService();

  static SaiaLiteRtService get instance => _customInstance ?? _webDefault;

  static void registerInstance(SaiaLiteRtService service) {
    _customInstance = service;
  }

  static bool get hasCustomInstance => _customInstance != null;
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
// Generative LLM (@mediapipe/tasks-genai) JS Bindings
// ==========================================
//
// NO se usa `@litert-lm/core`. Ese paquete existe en npm pero su tarball
// contiene solo `package.json`: promete `main: ./dist/index.js` y
// `files: [dist/**, wasm/**]`, y no publica ninguno de los dos. Su endpoint
// `+esm` devuelve 404, así que cualquier `import()` contra él falla con
// "Failed to fetch dynamically imported module".
//
// MediaPipe LLM Inference es la vía publicada de Google para ejecutar
// bundles `.task` en el navegador, y consume exactamente los mismos
// archivos (los de litert-community en HuggingFace). Firmas verificadas
// contra `genai.d.ts` de @mediapipe/tasks-genai@0.10.29.

/// Versión fijada de `@mediapipe/tasks-text`, que trae el Text Embedder.
///
/// Fijada por la misma razón que la de genai: sin versión, el módulo ESM y
/// el runtime WASM se resuelven por separado contra `latest` y pueden
/// quedar desincronizados a mitad de una publicación.
const String kMediaPipeTextVersion = '1.0.1';

const String _mpTextBase =
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-text@$kMediaPipeTextVersion';

@JS()
extension type _MpTextFilesetResolver._(JSObject _) implements JSObject {
  /// `static forTextTasks(basePath?): Promise<WasmFileset>`
  external JSPromise<JSObject> forTextTasks([JSString? basePath]);
}

@JS()
extension type _MpTextEmbedderConstructor._(JSObject _) implements JSObject {
  external JSPromise<_MpTextEmbedder> createFromOptions(
    JSObject wasmFileset,
    JSObject options,
  );
}

@JS()
extension type _MpTextEmbedder._(JSObject _) implements JSObject {
  /// `embed(text): TextEmbedderResult` — síncrono, no devuelve Promise.
  external JSObject embed(JSString text);

  external void close();
}

/// Versión fijada de `@mediapipe/tasks-genai`.
///
/// **Fijada a propósito, y a una RC a propósito.**
///
/// El tag `latest` apunta a `0.10.29`, y esa versión falla al compilar los
/// bundles `.task` de litert-community en WebGPU:
///
///   llm_litert_webgpu_executor.cc:407
///   interpreter->ModifyGraphWithDelegate(gpu_delegate.get()) == kTfLiteOk
///
/// El delegado rechaza el grafo por operaciones que no soporta. LiteRT
/// upstream lleva meses añadiéndolas al backend WebGPU (Slice,
/// ResizeBilinear, DepthwiseConv2D…), y esos avances salen en la línea
/// `1.0.1-rc`, que se publica a diario. Verificado que la RC mantiene la
/// misma API: mismos exports y `createFromOptions` / `delegate` /
/// `maxTokens` sin cambios.
///
/// Sin fijar versión había un riesgo peor que usar una RC: el módulo ESM y
/// el runtime WASM se resolvían por separado contra `latest`, y podían
/// quedar desincronizados entre sí a mitad de una publicación.
const String kMediaPipeGenAiVersion = '1.0.1-rc.20260902';

const String _mpBase =
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-genai@$kMediaPipeGenAiVersion';

@JS()
extension type _MpFilesetResolver._(JSObject _) implements JSObject {
  /// `static forGenAiTasks(basePath?, useModule?): Promise<WasmFileset>`
  external JSPromise<JSObject> forGenAiTasks([JSString? basePath]);
}

@JS()
extension type _MpLlmInferenceConstructor._(JSObject _) implements JSObject {
  /// `static createFromOptions(wasmFileset, options): Promise<LlmInference>`
  external JSPromise<_MpLlmInference> createFromOptions(
    JSObject wasmFileset,
    JSObject options,
  );
}

@JS()
extension type _MpLlmInference._(JSObject _) implements JSObject {
  /// `generateResponse(query, progressListener?): Promise<string>`
  ///
  /// El listener recibe `(partialResult, done)` — es el streaming real; el
  /// Promise resuelve con el texto completo al terminar.
  external JSPromise<JSString> generateResponse(
    JSString query,
    JSFunction progressListener,
  );

  external void close();
}

// ==========================================
// Service Implementation
// ==========================================

class WebLiteRtService implements SaiaLiteRtService {
  @override
  Future<void> loadLiteRt({String? wasmPath}) async {
    if (web.window.has('__saia_litert_wasm_loaded')) return;
    if (web.window.has('__saia_litert_wasm_loading')) return;

    web.window.setProperty('__saia_litert_wasm_loading'.toJS, true.toJS);
    try {
      if (!web.window.has('loadLiteRt')) {
        await _loadModuleScriptGeneral('https://cdn.jsdelivr.net/npm/@litertjs/core/+esm');
      }
      final wasmUrl = wasmPath ?? 'https://cdn.jsdelivr.net/npm/@litertjs/core/wasm/';
      await _jsLoadLiteRt(wasmUrl.toJS).toDart;
      web.window.setProperty('__saia_litert_wasm_loaded'.toJS, true.toJS);
    } catch (e) {
      final errStr = e.toString().toLowerCase();
      if (errStr.contains('already loading') || errStr.contains('already loaded')) {
        web.window.setProperty('__saia_litert_wasm_loaded'.toJS, true.toJS);
        return;
      }
      rethrow;
    } finally {
      web.window.delete('__saia_litert_wasm_loading'.toJS);
    }
  }

  @override
  Future<LiteRtModel> loadModel(String modelPath, {String accelerator = 'webgpu'}) async {
    final options = _JsLiteRtOptions(accelerator: accelerator.toJS);
    final jsModel = await _jsLoadAndCompile(modelPath.toJS, options).toDart;
    return LiteRtModelWeb(jsModel);
  }

  @override
  Future<bool> isWebGpuAvailable() async {
    try {
      final nav = web.window.navigator as JSObject;
      if (!nav.has('gpu')) return false;
      final gpu = nav.getProperty('gpu'.toJS);
      if (gpu == null || !gpu.isA<JSObject>()) return false;
      // Tener `navigator.gpu` no basta: puede existir y no devolver
      // adaptador (GPU bloqueada, driver en lista negra, VM sin
      // aceleración). Lo que importa es si hay adaptador.
      final adapter = await (gpu as JSObject)
          .callMethod<JSPromise<JSAny?>>('requestAdapter'.toJS)
          .toDart;
      return adapter != null;
    } catch (_) {
      return false;
    }
  }

  @override
  Future<void> loadLiteRtLm({String? moduleUrl}) async {
    if (web.window.has('__saia_mp_LlmInference')) return;

    final loaderUrl =
        moduleUrl ?? '$_mpBase/+esm';
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
        if (!mod?.LlmInference || !mod?.FilesetResolver) {
          throw new Error(
            'El modulo cargado no expone LlmInference/FilesetResolver: ' +
            Object.keys(mod || {}).join(',')
          );
        }
        window.__saia_mp_LlmInference = mod.LlmInference;
        window.__saia_mp_FilesetResolver = mod.FilesetResolver;
        if (window.$callbackName) {
          window.$callbackName(null);
        }
      }).catch(e => {
        if (window.$callbackName) {
          window.$callbackName(String(e && e.message ? e.message : e));
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

  /// Importa `@mediapipe/tasks-text` y expone sus clases en `window`.
  Future<void> _loadTextModuleScript(String moduleUrl) async {
    final completer = Completer<void>();
    const callbackName = '__saia_mp_text_init';

    web.window.setProperty(
      callbackName.toJS,
      (JSAny? err) {
        if (completer.isCompleted) return;
        if (err != null) {
          completer.completeError(
            StateError('MediaPipe Text module load failed: $err'),
          );
          return;
        }
        completer.complete();
      }.toJS,
    );

    final script = web.HTMLScriptElement();
    script.type = 'module';
    script.text = '''
      import(${jsonEncode(moduleUrl)}).then(mod => {
        if (!mod?.TextEmbedder || !mod?.FilesetResolver) {
          throw new Error(
            'El modulo no expone TextEmbedder/FilesetResolver: ' +
            Object.keys(mod || {}).join(',')
          );
        }
        window.__saia_mp_TextEmbedder = mod.TextEmbedder;
        window.__saia_mp_TextFilesetResolver = mod.FilesetResolver;
        if (window.$callbackName) window.$callbackName(null);
      }).catch(e => {
        if (window.$callbackName) {
          window.$callbackName(String(e && e.message ? e.message : e));
        }
      });
    ''';

    web.document.head?.append(script);
    try {
      await completer.future.timeout(const Duration(seconds: 20));
    } finally {
      web.window.delete(callbackName.toJS);
    }
  }

  /// Deduce el tope de tokens del nombre del bundle.
  ///
  /// Los `.task` de litert-community codifican su caché KV como `ekv<N>`
  /// (p. ej. `…_q8_ekv1280.task`). Si no aparece, se usa un valor
  /// conservador: pasarse hace fallar la creación del engine, quedarse
  /// corto solo acorta la conversación.
  static int _maxTokensFor(String modelPath) {
    final match = RegExp(r'ekv(\d+)').firstMatch(modelPath);
    final declared = match == null ? null : int.tryParse(match.group(1)!);
    return declared ?? 1024;
  }

  @override
  Future<LiteRtLmModel> loadLmModel(String modelPath,
      {String accelerator = 'gpu'}) async {
    final rawLlm = web.window.getProperty('__saia_mp_LlmInference'.toJS);
    final rawFileset = web.window.getProperty('__saia_mp_FilesetResolver'.toJS);
    if (rawLlm == null ||
        !rawLlm.isA<JSObject>() ||
        rawFileset == null ||
        !rawFileset.isA<JSObject>()) {
      throw StateError(
        'MediaPipe GenAI no está cargado. Llama a loadLiteRtLm primero.',
      );
    }

    // El fileset resuelve el runtime WASM (~26 MB, cacheado por el
    // navegador tras la primera vez).
    final resolver = _MpFilesetResolver._(rawFileset as JSObject);
    final fileset = await resolver
        .forGenAiTasks(
          '$_mpBase/wasm'.toJS,
        )
        .toDart;

    final baseOptions = JSObject();
    baseOptions.setProperty('modelAssetPath'.toJS, modelPath.toJS);

    final options = JSObject();
    options.setProperty('baseOptions'.toJS, baseOptions);
    // MediaPipe fija el sampling al CREAR el modelo, no por generación
    // (ver LlmInferenceOptions en genai.d.ts). Estos son los topes; los
    // valores por llamada de generateStream no pueden cambiarlos sin
    // reconstruir el modelo, y reconstruirlo implica recompilar en GPU.
    //
    // `maxTokens` NO puede exceder la caché KV con la que se compiló el
    // bundle, o la creación del engine falla con:
    //   "Max number of tokens is larger than the maximum cache size
    //    supported: 2048 > 1280"
    // Los bundles de litert-community lo llevan en el nombre (`…ekv1280`),
    // así que se lee de ahí en vez de asumirlo.
    options.setProperty('maxTokens'.toJS, _maxTokensFor(modelPath).toJS);
    options.setProperty('topK'.toJS, 40.toJS);
    options.setProperty('temperature'.toJS, 0.4.toJS);

    final constructor = _MpLlmInferenceConstructor._(rawLlm as JSObject);

    // ⚠ NO HAY CAMINO POR CPU PARA LLM EN WEB.
    //
    // Aquí había un reintento con `delegate: 'CPU'` documentado como «CPU sí
    // lo ejecuta — más lento, pero funciona». Se midió y es falso: pasando
    // `delegate: 'CPU'` en `baseOptions`, MediaPipe devuelve el error
    // IDÉNTICO de WebGPU —`llm_litert_webgpu_executor.cc:407`— así que la
    // opción se ignora y la inferencia de LLM en navegador solo tiene ruta
    // de GPU. Comprobado con `@mediapipe/tasks-genai@1.0.1-rc.20260904` en
    // un arnés de 30 líneas, fuera de Itzli.
    //
    // El reintento no era inofensivo: descargaba el bundle otra vez, tardaba
    // el doble y terminaba mostrando el mismo error, dando a entender que se
    // había intentado algo distinto.
    baseOptions.setProperty('delegate'.toJS, 'GPU'.toJS);
    try {
      final llm = await constructor.createFromOptions(fileset, options).toDart;
      return LiteRtLmModelWeb(llm, backend: 'GPU');
    } catch (e) {
      if (!_isDelegateRejection(e)) rethrow;
      // Rechazo del delegado: el grafo del bundle usa operaciones que el
      // backend WebGPU no implementa. Se traduce a algo accionable en vez
      // de propagar 300 caracteres de traza de C++.
      throw LiteRtBundleIncompatible(modelPath, e.toString());
    }
  }

  @override
  Future<LiteRtTextEmbedder> loadTextEmbedder(
    String modelPath, {
    bool l2Normalize = true,
  }) async {
    if (!web.window.has('__saia_mp_TextEmbedder')) {
      await _loadTextModuleScript('$_mpTextBase/+esm');
    }

    final rawEmbedder = web.window.getProperty('__saia_mp_TextEmbedder'.toJS);
    final rawFileset =
        web.window.getProperty('__saia_mp_TextFilesetResolver'.toJS);
    if (rawEmbedder == null ||
        !rawEmbedder.isA<JSObject>() ||
        rawFileset == null ||
        !rawFileset.isA<JSObject>()) {
      throw StateError('MediaPipe Text no expuso TextEmbedder.');
    }

    final resolver = _MpTextFilesetResolver._(rawFileset as JSObject);
    final fileset =
        await resolver.forTextTasks('$_mpTextBase/wasm'.toJS).toDart;

    final baseOptions = JSObject();
    baseOptions.setProperty('modelAssetPath'.toJS, modelPath.toJS);

    final options = JSObject();
    options.setProperty('baseOptions'.toJS, baseOptions);
    // Norma 1: convierte el producto interno en similitud coseno, que es
    // como compara saia_turbovec.
    options.setProperty('l2Normalize'.toJS, l2Normalize.toJS);
    // Sin cuantizar: se quiere el vector float, no bytes escalados.
    options.setProperty('quantize'.toJS, false.toJS);

    final constructor = _MpTextEmbedderConstructor._(rawEmbedder as JSObject);
    final embedder = await constructor.createFromOptions(fileset, options).toDart;
    return LiteRtTextEmbedderWeb(embedder);
  }

  /// Distingue "el delegado de GPU rechazó este modelo" de cualquier otro
  /// fallo, para no reintentar a ciegas.
  static bool _isDelegateRejection(Object e) {
    final s = e.toString();
    return s.contains('ModifyGraphWithDelegate') ||
        s.contains('llm_litert_webgpu_executor') ||
        s.contains('RET_CHECK');
  }
}

/// El bundle no es compatible con el backend WebGPU de este runtime.
///
/// Pasa cuando el `.task` se convirtió con una versión de `ai-edge-torch`
/// que emite operaciones que el delegado no implementa. El síntoma en la
/// consola es concreto:
///
///   RESHAPE: Tensor dimensions must be less than 5
///     …CausalSelfAttention_atten_func
///   ModifyGraphWithDelegate(gpu_delegate.get()) == kTfLiteOk (1 vs. 0)
///
/// No se arregla desde Itzli: hace falta un bundle convertido con una
/// versión compatible. Y no hay ruta por CPU a la que caer.
class LiteRtBundleIncompatible implements Exception {
  final String modelPath;
  final String detalle;

  LiteRtBundleIncompatible(this.modelPath, this.detalle);

  @override
  String toString() =>
      'El backend WebGPU rechazó este modelo: su grafo usa operaciones que '
      'esta versión de MediaPipe no implementa. No es falta de memoria ni de '
      'GPU, y no hay modo CPU para LLM en navegador. Hace falta un bundle '
      '.task compatible.\n\nModelo: ${modelPath.split("/").last}';
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

class LiteRtTextEmbedderWeb implements LiteRtTextEmbedder {
  final _MpTextEmbedder _embedder;
  int _dimensions = 0;

  LiteRtTextEmbedderWeb(this._embedder);

  /// Se conoce tras el primer [embed]: MediaPipe no publica el ancho antes
  /// de vectorizar algo. 0 significa "aún no se sabe", no "modelo roto".
  @override
  int get dimensions => _dimensions;

  @override
  Float32List embed(String text) {
    final result = _embedder.embed(text.toJS);

    // TextEmbedderResult.embeddings[0].floatEmbedding
    final embeddings = result.getProperty<JSArray<JSObject>?>('embeddings'.toJS);
    if (embeddings == null || embeddings.length == 0) {
      return Float32List(0);
    }
    final first = embeddings.toDart[0];

    // MediaPipe devuelve `floatEmbedding` como array JS PLANO (`number[]`),
    // no como `Float32Array`. Leerlo como `JSFloat32Array` lanzaba
    //
    //   type 'List<dynamic>' is not a subtype of type 'NativeFloat32List?'
    //
    // y el proveedor entero caía al `catch`: los embeddings semánticos en web
    // NUNCA cargaban y la memoria se quedaba en el vectorizador léxico, que
    // halla duplicados pero no sinónimos. Sin error visible — el fallback
    // seguía respondiendo.
    //
    // Se lee como array genérico y se convierte elemento a elemento, que
    // funciona con las dos formas por si una versión devuelve tipado.
    final crudo = first.getProperty<JSAny?>('floatEmbedding'.toJS);
    if (crudo == null) {
      // Ocurre si el embebedor se configuró con `quantize: true`, que aquí
      // no se hace. Devolver vacío en vez de inventar un vector.
      return Float32List(0);
    }

    final Float32List out;
    if (crudo.isA<JSFloat32Array>()) {
      out = (crudo as JSFloat32Array).toDart;
    } else if (crudo.isA<JSArray>()) {
      final lista = (crudo as JSArray<JSNumber>).toDart;
      out = Float32List(lista.length);
      for (var i = 0; i < lista.length; i++) {
        out[i] = lista[i].toDartDouble;
      }
    } else {
      return Float32List(0);
    }

    _dimensions = out.length;
    return out;
  }

  @override
  void dispose() {
    try {
      _embedder.close();
    } catch (_) {
      // close() es síncrono y puede lanzar si el runtime ya se fue.
    }
  }
}

class LiteRtLmModelWeb implements LiteRtLmModel {
  final _MpLlmInference _llm;
  bool _busy = false;

  @override
  final String backend;

  LiteRtLmModelWeb(this._llm, {required this.backend});

  /// Genera en streaming.
  ///
  /// **Los parámetros de sampling se ignoran aquí a propósito.** MediaPipe
  /// los fija al construir el modelo (`LlmInferenceOptions`), no por
  /// generación: cambiarlos exigiría recrear el `LlmInference` y recompilar
  /// en GPU. Se conservan en la firma para no romper la interfaz común con
  /// la implementación nativa, que sí los admite por llamada. `topP` además
  /// no existe en la API de MediaPipe.
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
    _runGeneration(controller, prompt);
    return controller.stream;
  }

  Future<void> _runGeneration(
    StreamController<String> controller,
    String prompt,
  ) async {
    // MediaPipe rechaza generaciones concurrentes sobre la misma instancia
    // ("Only one call to generateResponse() can run at a time").
    if (_busy) {
      controller.addError(
        StateError('Ya hay una generación en curso en este modelo.'),
      );
      await controller.close();
      return;
    }
    _busy = true;

    var emitted = 0;
    try {
      final listener = ((JSString partial, JSBoolean done) {
        final text = partial.toDart;
        if (text.isNotEmpty && !controller.isClosed) {
          controller.add(text);
          emitted += text.length;
        }
      }).toJS;

      final full = await _llm.generateResponse(prompt.toJS, listener).toDart;

      // Si el listener nunca entregó nada pero el Promise sí trae texto,
      // emitirlo entero: mejor una respuesta de golpe que ninguna.
      if (emitted == 0) {
        final text = full.toDart;
        if (text.isNotEmpty && !controller.isClosed) controller.add(text);
      }
    } catch (e) {
      if (!controller.isClosed) controller.addError(e);
    } finally {
      _busy = false;
      if (!controller.isClosed) await controller.close();
    }
  }

  @override
  Future<void> dispose() async {
    try {
      _llm.close();
    } catch (_) {
      // close() es síncrono y puede lanzar si el runtime ya se fue; que no
      // impida liberar el resto.
    }
  }
}
