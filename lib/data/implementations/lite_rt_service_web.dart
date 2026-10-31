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
// MediaPipe Tasks (Text, Vision, Audio, GenAI)
// ==========================================

/// Versión verificada en npm/jsdelivr para `@mediapipe/tasks-text`,
/// `@mediapipe/tasks-vision` y `@mediapipe/tasks-audio`.
const String kMediaPipeTasksVersion = '0.10.21';

const String _mpTextBase =
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-text@$kMediaPipeTasksVersion';
const String _mpVisionBase =
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@$kMediaPipeTasksVersion';
const String _mpAudioBase =
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-audio@$kMediaPipeTasksVersion';

@JS()
extension type _MpTextFilesetResolver._(JSObject _) implements JSObject {
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
  external JSObject embed(JSString text);
  external void close();
}

/// Versión de `@mediapipe/tasks-genai` para modelos `.task` 4D (`-web.task`).
const String kMediaPipeGenAiVersion = '1.0.1-rc.20260902';

const String _mpBase =
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-genai@$kMediaPipeGenAiVersion';

@JS()
extension type _MpFilesetResolver._(JSObject _) implements JSObject {
  external JSPromise<JSObject> forGenAiTasks([JSString? basePath]);
}

@JS()
extension type _MpLlmInferenceConstructor._(JSObject _) implements JSObject {
  external JSPromise<_MpLlmInference> createFromOptions(
    JSObject wasmFileset,
    JSObject options,
  );
}

@JS()
extension type _MpLlmInference._(JSObject _) implements JSObject {
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
  static const String _hfTokenStorageKey = 'saia_hf_token';
  final Map<String, LiteRtLmModel> _loadedLmCache = {};

  @override
  String? get huggingFaceToken {
    try {
      final raw = web.window.localStorage.getItem(_hfTokenStorageKey);
      if (raw == null || raw.trim().isEmpty) return null;
      return raw.trim();
    } catch (_) {
      return null;
    }
  }

  @override
  void setHuggingFaceToken(String? token) {
    try {
      final clean = token?.trim() ?? '';
      if (clean.isEmpty) {
        web.window.localStorage.removeItem(_hfTokenStorageKey);
      } else {
        web.window.localStorage.setItem(_hfTokenStorageKey, clean);
      }
    } catch (_) {}
  }

  @override
  Future<void> loadLiteRt({String? wasmPath}) async {
    if (web.window.has('__saia_litert_wasm_loaded')) return;
    if (web.window.has('__saia_litert_wasm_loading')) return;

    web.window.setProperty('__saia_litert_wasm_loading'.toJS, true.toJS);
    try {
      if (!web.window.has('loadLiteRt')) {
        await _loadModuleScriptGeneral(
          'https://cdn.jsdelivr.net/npm/@litertjs/core/+esm',
        );
      }
      final wasmUrl =
          wasmPath ?? 'https://cdn.jsdelivr.net/npm/@litertjs/core/wasm/';
      await _jsLoadLiteRt(wasmUrl.toJS).toDart;
      web.window.setProperty('__saia_litert_wasm_loaded'.toJS, true.toJS);
    } catch (e) {
      final errStr = e.toString().toLowerCase();
      if (errStr.contains('already loading') ||
          errStr.contains('already loaded')) {
        web.window.setProperty('__saia_litert_wasm_loaded'.toJS, true.toJS);
        return;
      }
      rethrow;
    } finally {
      web.window.delete('__saia_litert_wasm_loading'.toJS);
    }
  }

  @override
  Future<LiteRtModel> loadModel(
    String modelPath, {
    String accelerator = 'webgpu',
  }) async {
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
    if (moduleUrl != null) {
      await _loadModuleScript(moduleUrl);
      return;
    }
    // Por defecto aseguramos el helper JS para WebLLM y MediaPipe.
    _ensureWebAiHelper();
  }

  /// Inyecta un helper JS en `window.__saiaWebAi` que gestiona:
  /// - Carga de `@mlc-ai/web-llm` (modelos abiertos Qwen 2.5, SmolLM2, Llama 3.2)
  /// - Descarga autenticada con `Authorization: Bearer hf_...` para modelos
  ///   gated de Hugging Face (Gemma 3 `-web.task`, EmbeddingGemma)
  /// - Verificación real de modelos de Visión (`@mediapipe/tasks-vision`)
  ///   y Audio (`@mediapipe/tasks-audio`)
  void _ensureWebAiHelper() {
    if (web.window.has('__saiaWebAi')) return;
    final script = web.HTMLScriptElement();
    script.text = r'''
window.__saiaWebAi = {
  _webllmMod: null,
  _engines: {},
  _loadedModels: {},
  _loadingPromises: {},
  _progressListeners: {},

  async ensureWebLlm() {
    if (!this._webllmMod) {
      this._webllmMod = await import('https://esm.run/@mlc-ai/web-llm');
    }
    return this._webllmMod;
  },

  isWebLlmLoaded(modelId) {
    return Boolean(this._engines[modelId] && this._loadedModels[modelId] === true);
  },

  async unloadWebLlmEngine(modelId) {
    const pending = this._loadingPromises[modelId];
    delete this._loadingPromises[modelId];
    delete this._progressListeners[modelId];
    this._loadedModels[modelId] = false;
    const engine = this._engines[modelId];
    delete this._engines[modelId];
    if (pending) {
      try {
        const loaded = await pending;
        if (loaded && typeof loaded.unload === 'function') {
          await loaded.unload();
        }
      } catch (_) {}
    }
    if (engine && typeof engine.unload === 'function') {
      try {
        await engine.unload();
      } catch (_) {}
    }
  },

  async createWebLlmEngine(modelId, onProgress) {
    if (this._engines[modelId] && this._loadedModels[modelId] === true) {
      return this._engines[modelId];
    }
    if (!this._progressListeners[modelId]) {
      this._progressListeners[modelId] = [];
    }
    if (onProgress) {
      this._progressListeners[modelId].push(onProgress);
    }
    if (this._loadingPromises[modelId]) {
      return await this._loadingPromises[modelId];
    }
    const loadTask = (async () => {
      const webllm = await this.ensureWebLlm();
      const notify = (report) => {
        const listeners = this._progressListeners[modelId] || [];
        const text = report && report.text ? String(report.text) : 'Cargando...';
        const prog = report && typeof report.progress === 'number' ? report.progress : 0;
        for (const fn of listeners) {
          try { fn(text, prog); } catch (_) {}
        }
      };
      let engine = this._engines[modelId];
      if (engine && typeof engine.reload === 'function') {
        if (typeof engine.setInitProgressCallback === 'function') {
          engine.setInitProgressCallback(notify);
        }
        await engine.reload(modelId);
      } else {
        engine = await webllm.CreateMLCEngine(modelId, {
          initProgressCallback: notify,
        });
      }
      this._engines[modelId] = engine;
      this._loadedModels[modelId] = true;
      return engine;
    })();
    this._loadingPromises[modelId] = loadTask;
    try {
      return await loadTask;
    } finally {
      delete this._loadingPromises[modelId];
      delete this._progressListeners[modelId];
    }
  },

  async streamWebLlm(engine, modelId, prompt, maxTokens, temp, topP, onChunk) {
    const messages = [];
    const regex = /<\|im_start\|>(\w+)\n([\s\S]*?)(?:<\|im_end\|>|$)/g;
    let match;
    while ((match = regex.exec(prompt)) !== null) {
      const role = match[1].trim();
      const content = match[2].trim();
      if ((role === 'system' || role === 'user' || role === 'assistant') && content.length > 0) {
        messages.push({ role, content });
      }
    }
    if (messages.length === 0) {
      messages.push({ role: 'user', content: prompt.trim() });
    }

    let activeEngine = engine;
    if (!activeEngine || this._loadedModels[modelId] !== true) {
      activeEngine = await this.createWebLlmEngine(modelId, null);
    }

    const createCompletion = async (eng) => eng.chat.completions.create({
      messages,
      max_tokens: maxTokens || 512,
      temperature: typeof temp === 'number' ? temp : 0.4,
      top_p: typeof topP === 'number' ? topP : 0.9,
      stream: true,
    });

    let chunks;
    try {
      chunks = await createCompletion(activeEngine);
    } catch (err) {
      const msg = String(err || '');
      if (msg.includes('ModelNotLoadedError') || msg.includes('Model not loaded')) {
        this._loadedModels[modelId] = false;
        activeEngine = await this.createWebLlmEngine(modelId, null);
        chunks = await createCompletion(activeEngine);
      } else {
        throw err;
      }
    }

    let full = '';
    for await (const chunk of chunks) {
      const delta = chunk.choices?.[0]?.delta?.content || '';
      if (delta) {
        full += delta;
        if (onChunk) onChunk(delta);
      }
    }
    return full;
  },

  async fetchWithHfToken(url, token, onProgress) {
    const headers = {};
    if (token && token.trim().length > 0) {
      headers['Authorization'] = 'Bearer ' + token.trim();
    }
    const res = await fetch(url, { headers });
    if (res.status === 401 || res.status === 403) {
      throw new Error(
        'HTTP ' + res.status + ': Acceso restringido en Hugging Face. ' +
        'Sigue los pasos de la tarjeta: (1) Acepta la licencia en Hugging Face y ' +
        '(2) Configura tu Token HF (hf_...) arriba.'
      );
    }
    if (!res.ok) {
      throw new Error('HTTP ' + res.status + ' descargando ' + url);
    }
    const total = Number(res.headers.get('content-length') || 0);
    if (!res.body || !res.body.getReader) {
      const blob = await res.blob();
      return URL.createObjectURL(blob);
    }
    const reader = res.body.getReader();
    const parts = [];
    let loaded = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      parts.push(value);
      loaded += value.byteLength;
      if (onProgress && total > 0) {
        onProgress('Descargando (' + Math.round((loaded / 1048576) * 10) / 10 + ' MB)...', loaded / total);
      }
    }
    const blob = new Blob(parts, { type: 'application/octet-stream' });
    return URL.createObjectURL(blob);
  },

  async verifyVision(modelUrl, mpVisionBase) {
    const t0 = performance.now();
    const mod = await import(mpVisionBase + '/+esm');
    const fileset = await mod.FilesetResolver.forVisionTasks(mpVisionBase + '/wasm');
    const canvas = document.createElement('canvas');
    canvas.width = 224;
    canvas.height = 224;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#87CEEB';
    ctx.fillRect(0, 0, 224, 224);
    ctx.fillStyle = '#228B22';
    ctx.fillRect(0, 160, 224, 64);
    ctx.fillStyle = '#FF4500';
    ctx.beginPath();
    ctx.arc(112, 112, 48, 0, Math.PI * 2);
    ctx.fill();

    const isDetector = modelUrl.toLowerCase().includes('ssd') || modelUrl.toLowerCase().includes('detect');
    if (isDetector) {
      const detector = await mod.ObjectDetector.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: modelUrl },
        scoreThreshold: 0.1,
      });
      const res = detector.detect(canvas);
      detector.close();
      const dt = Math.round(performance.now() - t0);
      const count = res?.detections?.length || 0;
      const top = res?.detections?.[0]?.categories?.[0];
      const topDesc = top
        ? top.categoryName + ' (' + (top.score * 100).toFixed(1) + '%)'
        : 'Escena sintética procesada (0 objetos sobre umbral)';
      return 'Itzli Mirada · Detección activa (' + dt + ' ms) · Detecciones: ' + count + ' · ' + topDesc;
    } else {
      const classifier = await mod.ImageClassifier.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: modelUrl },
        maxResults: 3,
      });
      const res = classifier.classify(canvas);
      classifier.close();
      const dt = Math.round(performance.now() - t0);
      const cats = res?.classifications?.[0]?.categories || [];
      const summary = cats
        .slice(0, 3)
        .map(c => (c.categoryName || c.displayName || 'clase #' + c.index) + ' (' + (c.score * 100).toFixed(1) + '%)')
        .join(', ');
      return 'Itzli Mirada · Clasificación activa (' + dt + ' ms) · Top: ' + (summary || 'OK');
    }
  },

  async verifyAudio(modelUrl, mpAudioBase) {
    const t0 = performance.now();
    const mod = await import(mpAudioBase + '/+esm');
    const fileset = await mod.FilesetResolver.forAudioTasks(mpAudioBase + '/wasm');
    const classifier = await mod.AudioClassifier.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: modelUrl },
      maxResults: 3,
    });
    const sampleRate = 16000;
    const samples = new Float32Array(sampleRate);
    for (let i = 0; i < sampleRate; i++) {
      samples[i] = 0.5 * Math.sin(2 * Math.PI * 440 * (i / sampleRate));
    }
    const res = classifier.classify(samples, sampleRate);
    classifier.close();
    const dt = Math.round(performance.now() - t0);
    const cats = res?.[0]?.classifications?.[0]?.categories || [];
    const summary = cats
      .slice(0, 3)
      .map(c => (c.categoryName || c.displayName || 'clase #' + c.index) + ' (' + (c.score * 100).toFixed(1) + '%)')
      .join(', ');
    return 'Itzli Oído activo (' + dt + ' ms) · Señal 440 Hz → ' + (summary || 'OK');
  }
};
''';
    (web.document.head ?? web.document.body)?.append(script);
  }

  JSObject get _webAiHelper {
    _ensureWebAiHelper();
    return web.window.getProperty<JSObject>('__saiaWebAi'.toJS);
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
      await completer.future.timeout(const Duration(seconds: 20));
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

  static int _maxTokensFor(String modelPath) {
    final match = RegExp(r'ekv(\d+)').firstMatch(modelPath);
    final declared = match == null ? null : int.tryParse(match.group(1)!);
    return declared ?? 1024;
  }

  /// Normaliza rutas antiguas `.task` 5D (Qwen2.5 de Android) a sus
  /// equivalentes funcionales en WebLLM (`-MLC`).
  static String _normalizeLmModelPath(String modelPath) {
    if (modelPath.contains('Qwen2.5-0.5B-Instruct') &&
        modelPath.endsWith('.task')) {
      return 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC';
    }
    if (modelPath.contains('Qwen2.5-1.5B-Instruct') &&
        modelPath.endsWith('.task')) {
      return 'Qwen2.5-1.5B-Instruct-q4f16_1-MLC';
    }
    return modelPath;
  }

  static bool _isTaskBundleUrl(String modelPath) {
    final lower = modelPath.toLowerCase();
    return lower.endsWith('.task') ||
        lower.endsWith('.bin') ||
        lower.startsWith('http://') ||
        lower.startsWith('https://') ||
        lower.startsWith('blob:');
  }

  Future<String> _resolveAuthenticatedUrlIfNeeded(
    String url, {
    void Function(String status, double progress)? onProgress,
  }) async {
    if (!url.contains('huggingface.co')) return url;
    final token = huggingFaceToken;
    // Si es un repo de litert-community gated (Gemma) o si hay token HF,
    // descargamos con fetch + Authorization header y creamos un Blob URL.
    final isGated = url.toLowerCase().contains('gemma');
    if (!isGated && (token == null || token.isEmpty)) {
      return url;
    }
    if (isGated && (token == null || token.isEmpty)) {
      throw StateError(
        'Este modelo de Hugging Face requiere licencia aceptada y un Token HF (hf_...). '
        'Sigue los pasos indicados en la tarjeta del modelo para configurarlo.',
      );
    }
    final progressCb = ((JSString status, JSNumber prog) {
      onProgress?.call(status.toDart, prog.toDartDouble);
    }).toJS;
    final blobUrl = await (_webAiHelper.callMethod(
      'fetchWithHfToken'.toJS,
      url.toJS,
      (token ?? '').toJS,
      progressCb,
    ) as JSPromise<JSString>)
        .toDart;
    return blobUrl.toDart;
  }

  @override
  bool isLmModelLoaded(String modelPath) {
    final normalized = _normalizeLmModelPath(modelPath);
    if (!_loadedLmCache.containsKey(normalized)) return false;
    if (!_isTaskBundleUrl(normalized)) {
      try {
        final ok = _webAiHelper.callMethod(
          'isWebLlmLoaded'.toJS,
          normalized.toJS,
        );
        return ok != null && ok.isA<JSBoolean>() && (ok as JSBoolean).toDart;
      } catch (_) {
        return false;
      }
    }
    return true;
  }

  @override
  Future<void> unloadLmModel(String modelPath) async {
    final normalized = _normalizeLmModelPath(modelPath);
    final cached = _loadedLmCache.remove(normalized);
    if (!_isTaskBundleUrl(normalized)) {
      try {
        await (_webAiHelper.callMethod(
          'unloadWebLlmEngine'.toJS,
          normalized.toJS,
        ) as JSPromise<JSAny?>)
            .toDart;
      } catch (_) {}
      return;
    }
    if (cached is LiteRtLmModelWeb) {
      await cached.forceUnload();
    }
  }

  @override
  Future<LiteRtLmModel> loadLmModel(
    String modelPath, {
    String accelerator = 'gpu',
    void Function(String status, double progress)? onProgress,
  }) async {
    final normalized = _normalizeLmModelPath(modelPath);
    if (isLmModelLoaded(normalized)) {
      final existing = _loadedLmCache[normalized];
      if (existing != null) return existing;
    }

    // 1. Vía WebLLM (modelos abiertos pre-compilados para WebGPU: Qwen2.5, SmolLM2, Llama 3.2)
    if (!_isTaskBundleUrl(normalized)) {
      final progressCb = ((JSString status, JSNumber prog) {
        onProgress?.call(status.toDart, prog.toDartDouble);
      }).toJS;
      final engine = await (_webAiHelper.callMethod(
        'createWebLlmEngine'.toJS,
        normalized.toJS,
        progressCb,
      ) as JSPromise<JSObject>)
          .toDart;
      final model = WebLlmModelWeb(
        _webAiHelper,
        engine,
        modelId: normalized,
        backend: 'WebGPU Local',
      );
      _loadedLmCache[normalized] = model;
      return model;
    }

    // 2. Vía motor WebGPU de bundles `.task` 4D
    if (!web.window.has('__saia_mp_LlmInference')) {
      await _loadModuleScript('$_mpBase/+esm');
    }

    final rawLlm = web.window.getProperty('__saia_mp_LlmInference'.toJS);
    final rawFileset = web.window.getProperty('__saia_mp_FilesetResolver'.toJS);
    if (rawLlm == null ||
        !rawLlm.isA<JSObject>() ||
        rawFileset == null ||
        !rawFileset.isA<JSObject>()) {
      throw StateError(
        'El motor generativo WebGPU de Itzli no está inicializado.',
      );
    }

    final resolvedAssetPath = await _resolveAuthenticatedUrlIfNeeded(
      normalized,
      onProgress: onProgress,
    );

    final resolver = _MpFilesetResolver._(rawFileset as JSObject);
    final fileset = await resolver.forGenAiTasks('$_mpBase/wasm'.toJS).toDart;

    final baseOptions = JSObject();
    baseOptions.setProperty('modelAssetPath'.toJS, resolvedAssetPath.toJS);

    final options = JSObject();
    options.setProperty('baseOptions'.toJS, baseOptions);
    options.setProperty('maxTokens'.toJS, _maxTokensFor(normalized).toJS);
    options.setProperty('topK'.toJS, 40.toJS);
    options.setProperty('temperature'.toJS, 0.4.toJS);

    final constructor = _MpLlmInferenceConstructor._(rawLlm as JSObject);
    baseOptions.setProperty('delegate'.toJS, 'GPU'.toJS);
    try {
      final llm = await constructor.createFromOptions(fileset, options).toDart;
      final model = LiteRtLmModelWeb(llm, backend: 'WebGPU Local');
      _loadedLmCache[normalized] = model;
      return model;
    } catch (e) {
      if (!_isDelegateRejection(e)) rethrow;
      throw LiteRtBundleIncompatible(normalized, e.toString());
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

    final resolvedAssetPath = await _resolveAuthenticatedUrlIfNeeded(modelPath);

    final resolver = _MpTextFilesetResolver._(rawFileset as JSObject);
    final fileset =
        await resolver.forTextTasks('$_mpTextBase/wasm'.toJS).toDart;

    final baseOptions = JSObject();
    baseOptions.setProperty('modelAssetPath'.toJS, resolvedAssetPath.toJS);

    final options = JSObject();
    options.setProperty('baseOptions'.toJS, baseOptions);
    options.setProperty('l2Normalize'.toJS, l2Normalize.toJS);
    options.setProperty('quantize'.toJS, false.toJS);

    final constructor = _MpTextEmbedderConstructor._(rawEmbedder as JSObject);
    final embedder =
        await constructor.createFromOptions(fileset, options).toDart;
    return LiteRtTextEmbedderWeb(embedder);
  }

  @override
  Future<String> verifyVisionModel(String modelPath) async {
    final res = await (_webAiHelper.callMethod(
      'verifyVision'.toJS,
      modelPath.toJS,
      _mpVisionBase.toJS,
    ) as JSPromise<JSString>)
        .toDart;
    return res.toDart;
  }

  @override
  Future<String> verifyAudioModel(String modelPath) async {
    final res = await (_webAiHelper.callMethod(
      'verifyAudio'.toJS,
      modelPath.toJS,
      _mpAudioBase.toJS,
    ) as JSPromise<JSString>)
        .toDart;
    return res.toDart;
  }

  static bool _isDelegateRejection(Object e) {
    final s = e.toString();
    return s.contains('ModifyGraphWithDelegate') ||
        s.contains('llm_litert_webgpu_executor') ||
        s.contains('RET_CHECK');
  }
}

class LiteRtBundleIncompatible implements Exception {
  final String modelPath;
  final String detalle;

  LiteRtBundleIncompatible(this.modelPath, this.detalle);

  @override
  String toString() =>
      'El acelerador WebGPU rechazó este paquete porque utiliza operaciones 5D '
      'no soportadas en este navegador. Selecciona uno de los niveles curados de '
      'Itzli compatibles con WebGPU (Itzli Chispa · 360M, Itzli Fugaz · 0.5B, '
      'Itzli Pulso · 1B o Itzli Centella · 1.5B).';
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

  @override
  int get dimensions => _dimensions;

  @override
  Float32List embed(String text) {
    final result = _embedder.embed(text.toJS);

    final embeddings =
        result.getProperty<JSArray<JSObject>?>('embeddings'.toJS);
    if (embeddings == null || embeddings.length == 0) {
      return Float32List(0);
    }
    final first = embeddings.toDart[0];
    final crudo = first.getProperty<JSAny?>('floatEmbedding'.toJS);
    if (crudo == null) {
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
    } catch (_) {}
  }
}

/// Wrapper sobre un motor `@mlc-ai/web-llm` compilado en WebGPU.
class WebLlmModelWeb implements LiteRtLmModel {
  final JSObject _helper;
  final JSObject _engine;
  final String modelId;
  bool _busy = false;

  @override
  final String backend;

  WebLlmModelWeb(
    this._helper,
    this._engine, {
    required this.modelId,
    required this.backend,
  });

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
    _runGeneration(controller, prompt, maxTokens, temp, topP);
    return controller.stream;
  }

  Future<void> _runGeneration(
    StreamController<String> controller,
    String prompt,
    int maxTokens,
    double temp,
    double topP,
  ) async {
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
      final onChunk = ((JSString delta) {
        final text = delta.toDart;
        if (text.isNotEmpty && !controller.isClosed) {
          controller.add(text);
          emitted += text.length;
        }
      }).toJS;

      final full = await (_helper.callMethodVarArgs(
        'streamWebLlm'.toJS,
        [
          _engine,
          modelId.toJS,
          prompt.toJS,
          maxTokens.toJS,
          temp.toJS,
          topP.toJS,
          onChunk,
        ],
      ) as JSPromise<JSString>)
          .toDart;

      if (emitted == 0) {
        final text = full.toDart;
        if (text.isNotEmpty && !controller.isClosed) {
          controller.add(text);
        }
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
    // El ciclo de vida en VRAM de los motores WebGPU compartidos es persistente
    // y solo se descarga explícitamente mediante SaiaLiteRt.instance.unloadLmModel().
  }
}

class LiteRtLmModelWeb implements LiteRtLmModel {
  final _MpLlmInference _llm;
  bool _busy = false;

  @override
  final String backend;

  LiteRtLmModelWeb(this._llm, {required this.backend});

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

  Future<void> forceUnload() async {
    try {
      _llm.close();
    } catch (_) {}
  }

  @override
  Future<void> dispose() async {
    // Mantenido en caché hasta llamada explícita a unloadLmModel().
  }
}
