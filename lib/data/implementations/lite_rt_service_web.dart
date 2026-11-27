@JS()
library;

import 'dart:async';
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:typed_data';
import 'package:web/web.dart' as web;

import '../../domain/lite_rt_model.dart';
import '../../domain/lite_rt_service.dart';

typedef NativeLiteRtLmLoader = Future<LiteRtLmModel> Function(
  String modelPath, {
  required String accelerator,
  void Function(String status, double progress)? onProgress,
});

/// Concrete factory to get the platform-specific implementation.
class SaiaLiteRt {
  static SaiaLiteRtService? _customInstance;
  static final WebLiteRtService _webDefault = WebLiteRtService();

  static SaiaLiteRtService get instance => _customInstance ?? _webDefault;

  static void registerInstance(SaiaLiteRtService service) {
    _customInstance = service;
  }

  static void registerNativeLmLoader(NativeLiteRtLmLoader? _) {}

  static bool get hasCustomInstance => _customInstance != null;

  static String recommendMobileAccelerator({
    required String socVendorOrModel,
    bool isIos = false,
  }) =>
      'webgpu';
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

@JS('LiteRtTensor')
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

/// Versión de `@litert-lm/core` (`LiteRT-LM.js`) para bundles `.litertlm`
/// modernos con WebGPU (p. ej. Gemma 4 E2B / E4B `-gpu.litertlm`).
const String kLiteRtLmCoreVersion = '0.19.0';

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
  static const String _trustedTypesPolicyName = 'saia-litert-loader';
  static JSObject? _trustedScriptPolicy;

  final Map<String, LiteRtLmModel> _loadedLmCache = {};
  Future<void>? _liteRtLoadFuture;

  /// Installs a minimal W3C Trusted Types tinyfill (`createPolicy: (name, rules) => rules`)
  /// on browsers that do not yet expose `window.trustedTypes`, so our validation
  /// rules execute identically across all browsers.
  static void _ensureTrustedTypesTinyfill() {
    if (web.window.has('trustedTypes')) {
      final existing = web.window.getProperty('trustedTypes'.toJS);
      if (existing != null && existing.isA<JSObject>()) return;
    }
    final tinyfill = JSObject();
    tinyfill.setProperty(
      'createPolicy'.toJS,
      ((JSString _, JSObject rules) => rules).toJS,
    );
    web.window.setProperty('trustedTypes'.toJS, tinyfill);
  }

  /// Assigns inline module/helper code to [script] via an allowlist-enforcing
  /// W3C Trusted Types policy (`saia-litert-loader`), with a tinyfill fallback
  /// for browsers without native `window.trustedTypes`.
  ///
  /// Never acts as an unvalidated pass-through: unauthorized script payloads
  /// throw [ArgumentError] before reaching the DOM sink.
  static void _assignTrustedScriptText(
    web.HTMLScriptElement script,
    String code,
  ) {
    if (!LiteRtUrlPolicy.isAuthorizedInlineScript(
      code,
      helperScript: _webAiHelperScript,
    )) {
      throw ArgumentError(
        'Rejected unauthorized inline script in saia-litert-loader policy.',
      );
    }
    try {
      _ensureTrustedTypesTinyfill();
      var policy = _trustedScriptPolicy;
      if (policy == null && web.window.has('trustedTypes')) {
        final ttAny = web.window.getProperty('trustedTypes'.toJS);
        if (ttAny != null && ttAny.isA<JSObject>()) {
          final tt = ttAny as JSObject;
          if (tt.has('createPolicy')) {
            final rules = JSObject();
            rules.setProperty(
              'createScript'.toJS,
              ((JSString input) {
                final dartCode = input.toDart;
                if (!LiteRtUrlPolicy.isAuthorizedInlineScript(
                  dartCode,
                  helperScript: _webAiHelperScript,
                )) {
                  throw ArgumentError(
                    'Rejected unauthorized inline script in saia-litert-loader policy.',
                  );
                }
                return dartCode.toJS;
              }).toJS,
            );
            final created = tt.callMethod(
              'createPolicy'.toJS,
              _trustedTypesPolicyName.toJS,
              rules,
            );
            if (created != null && created.isA<JSObject>()) {
              policy = created as JSObject;
              _trustedScriptPolicy = policy;
            }
          }
        }
      }
      if (policy != null) {
        final trustedScript = policy.callMethod(
          'createScript'.toJS,
          code.toJS,
        );
        (script as JSObject).setProperty('text'.toJS, trustedScript);
        return;
      }
    } catch (_) {
      // Fallback when CSP restricts policy names; `code` was already verified
      // against `LiteRtUrlPolicy.isAuthorizedInlineScript` above.
    }
    script.text = code;
  }

  @override
  String? get huggingFaceToken {
    try {
      var raw = web.window.sessionStorage.getItem(_hfTokenStorageKey);
      // One-time migration away from persistent localStorage to sessionStorage.
      final legacy = web.window.localStorage.getItem(_hfTokenStorageKey);
      if (legacy != null) {
        web.window.localStorage.removeItem(_hfTokenStorageKey);
        if (raw == null || raw.isEmpty) {
          final migrated = LiteRtUrlPolicy.sanitizeHfToken(legacy);
          if (migrated != null) {
            web.window.sessionStorage.setItem(_hfTokenStorageKey, migrated);
            raw = migrated;
          }
        }
      }
      return LiteRtUrlPolicy.sanitizeHfToken(raw);
    } catch (_) {
      return null;
    }
  }

  @override
  void setHuggingFaceToken(String? token) {
    final clean = LiteRtUrlPolicy.sanitizeHfToken(token);
    try {
      web.window.localStorage.removeItem(_hfTokenStorageKey);
      if (clean == null) {
        web.window.sessionStorage.removeItem(_hfTokenStorageKey);
      } else {
        web.window.sessionStorage.setItem(_hfTokenStorageKey, clean);
      }
    } catch (_) {}
  }

  @override
  Future<bool> isModelCached(String modelUrl) async {
    final safeUrl = LiteRtUrlPolicy.validateModelAssetUrl(modelUrl);
    try {
      final result = await (_webAiHelper.callMethod(
        'isModelCached'.toJS,
        safeUrl.toJS,
      ) as JSPromise<JSBoolean>)
          .toDart;
      return result.toDart;
    } catch (_) {
      return false;
    }
  }

  @override
  Future<bool> deleteCachedModel(String modelUrl) async {
    final safeUrl = LiteRtUrlPolicy.validateModelAssetUrl(modelUrl);
    try {
      final result = await (_webAiHelper.callMethod(
        'deleteCachedModel'.toJS,
        safeUrl.toJS,
      ) as JSPromise<JSBoolean>)
          .toDart;
      return result.toDart;
    } catch (_) {
      return false;
    }
  }

  @override
  Future<bool> clearModelCache() async {
    try {
      final result = await (_webAiHelper.callMethod(
        'clearModelCache'.toJS,
      ) as JSPromise<JSBoolean>)
          .toDart;
      return result.toDart;
    } catch (_) {
      return false;
    }
  }

  @override
  Future<void> loadLiteRt({String? wasmPath}) async {
    final validatedWasmUrl = LiteRtUrlPolicy.validateTrustedModuleUrl(
      wasmPath ?? 'https://cdn.jsdelivr.net/npm/@litertjs/core/wasm/',
    );
    if (web.window.has('__saia_litert_wasm_loaded')) return;

    final inFlight = _liteRtLoadFuture;
    if (inFlight != null) {
      await inFlight;
      return;
    }

    final future = _loadLiteRtInternal(validatedWasmUrl);
    _liteRtLoadFuture = future;
    try {
      await future;
    } finally {
      if (identical(_liteRtLoadFuture, future)) {
        _liteRtLoadFuture = null;
      }
    }
  }

  Future<void> _loadLiteRtInternal(String wasmUrl) async {
    web.window.setProperty('__saia_litert_wasm_loading'.toJS, true.toJS);
    try {
      if (!web.window.has('loadLiteRt')) {
        await _loadModuleScriptGeneral(
          'https://cdn.jsdelivr.net/npm/@litertjs/core/+esm',
        );
      }
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
    final safeModelPath = LiteRtUrlPolicy.validateModelAssetUrl(modelPath);
    if (!LiteRtModelInfo.isWebCompatibleArtifact(safeModelPath)) {
      throw UnsupportedError(
        'Model artifact is not compatible with Web LiteRT runtime: $safeModelPath',
      );
    }
    final safeAccelerator = LiteRtUrlPolicy.validateAccelerator(accelerator);
    final options = _JsLiteRtOptions(accelerator: safeAccelerator.toJS);
    final jsModel = await _jsLoadAndCompile(safeModelPath.toJS, options).toDart;
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
  Future<LiteRtBrowserCapabilities> probeBrowserCapabilities() async {
    final hasGpu = await isWebGpuAvailable();
    int? availableStorageBytes;
    double? deviceMemoryGb;
    try {
      final nav = web.window.navigator as JSObject;
      if (nav.has('deviceMemory')) {
        final rawMem = nav.getProperty('deviceMemory'.toJS);
        if (rawMem != null && rawMem.isA<JSNumber>()) {
          final parsed = (rawMem as JSNumber).toDartDouble;
          if (parsed.isFinite && parsed > 0) {
            deviceMemoryGb = parsed;
          }
        }
      }
      if (nav.has('storage')) {
        final storage = nav.getProperty('storage'.toJS);
        if (storage != null &&
            storage.isA<JSObject>() &&
            (storage as JSObject).has('estimate')) {
          final estAny = await storage
              .callMethod<JSPromise<JSAny?>>('estimate'.toJS)
              .toDart;
          if (estAny != null && estAny.isA<JSObject>()) {
            final est = estAny as JSObject;
            final quotaAny = est.getProperty('quota'.toJS);
            final usageAny = est.getProperty('usage'.toJS);
            final quota = (quotaAny != null && quotaAny.isA<JSNumber>())
                ? (quotaAny as JSNumber).toDartDouble
                : null;
            final usage = (usageAny != null && usageAny.isA<JSNumber>())
                ? (usageAny as JSNumber).toDartDouble
                : 0.0;
            if (quota != null &&
                quota.isFinite &&
                quota > 0 &&
                usage.isFinite &&
                usage >= 0) {
              final remaining = (quota - usage).round();
              if (remaining >= 0) {
                availableStorageBytes = remaining;
              }
            }
          }
        }
      }
    } catch (_) {}

    final recommended = hasGpu
        ? LiteRtModelCatalog.selectRecommendedWebAuxiliaryModel(
            availableStorageBytes: availableStorageBytes,
            deviceMemoryGb: deviceMemoryGb,
          )
        : null;
    return LiteRtBrowserCapabilities(
      hasWebGpu: hasGpu,
      availableStorageBytes: availableStorageBytes,
      deviceMemoryGb: deviceMemoryGb,
      recommendedModel: recommended,
    );
  }

  @override
  Future<void> loadLiteRtLm({String? moduleUrl}) async {
    if (moduleUrl != null) {
      final safeUrl = LiteRtUrlPolicy.validateTrustedModuleUrl(moduleUrl);
      await _loadModuleScript(safeUrl);
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
  /// Public entry point for security verification of the `saia-litert-loader`
  /// Trusted Types policy and tinyfill fallback on Web.
  static void assignTrustedScriptText(
    web.HTMLScriptElement script,
    String code,
  ) => _assignTrustedScriptText(script, code);

  static const String _webAiHelperScript = r'''
window.__saiaWebAi = {
  _webllmMod: null,
  _litertLmMod: null,
  _engines: Object.create(null),
  _litertLmEngines: Object.create(null),
  _litertLmConvs: Object.create(null),
  _litertLmLoading: Object.create(null),
  _loadedModels: Object.create(null),
  _loadingPromises: Object.create(null),
  _progressListeners: Object.create(null),
  _modelCacheName: 'saia-litert-models-v1',
  _opfsDirName: 'saia-litert-opfs-v1',

  _opfsKeyForUrl(url) {
    const str = String(url || '');
    let hash = 2166136261;
    for (let i = 0; i < str.length; i++) {
      hash ^= str.charCodeAt(i);
      hash = Math.imul(hash, 16777619) >>> 0;
    }
    const cleanBase = str
      .split('?')[0]
      .split('#')[0]
      .split('/')
      .pop()
      .replace(/[^A-Za-z0-9._-]/g, '_')
      .slice(-48) || 'model.bin';
    return hash.toString(16).padStart(8, '0') + '_' + cleanBase;
  },

  async _getOpfsDir(create) {
    if (typeof navigator === 'undefined' ||
        !navigator.storage ||
        typeof navigator.storage.getDirectory !== 'function') {
      return null;
    }
    try {
      const root = await navigator.storage.getDirectory();
      return await root.getDirectoryHandle(this._opfsDirName, { create: Boolean(create) });
    } catch (_) {
      return null;
    }
  },

  async ensureLiteRtLmCore() {
    if (!this._litertLmMod) {
      this._litertLmMod = await import('https://cdn.jsdelivr.net/npm/@litert-lm/core@0.19.0/+esm');
    }
    return this._litertLmMod;
  },

  isLiteRtLmCoreLoaded(modelUrl) {
    return Boolean(this._litertLmEngines[modelUrl] && this._loadedModels[modelUrl] === true);
  },

  async resetLiteRtLmCoreConversation(modelUrl) {
    const cached = this._litertLmConvs[modelUrl];
    delete this._litertLmConvs[modelUrl];
    if (cached && cached.conv && typeof cached.conv.delete === 'function') {
      try {
        await cached.conv.delete();
      } catch (_) {}
    }
  },

  async unloadLiteRtLmCore(modelUrl) {
    await this.resetLiteRtLmCoreConversation(modelUrl);
    const pending = this._litertLmLoading[modelUrl];
    delete this._litertLmLoading[modelUrl];
    delete this._progressListeners[modelUrl];
    this._loadedModels[modelUrl] = false;
    const engine = this._litertLmEngines[modelUrl];
    delete this._litertLmEngines[modelUrl];
    if (pending) {
      try {
        const loaded = await pending;
        if (loaded && typeof loaded.delete === 'function') {
          await loaded.delete();
        }
      } catch (_) {}
    }
    if (engine && typeof engine.delete === 'function') {
      try {
        await engine.delete();
      } catch (_) {}
    }
  },

  async createLiteRtLmCoreEngine(modelUrl, token, onProgress) {
    if (this._litertLmEngines[modelUrl] && this._loadedModels[modelUrl] === true) {
      if (onProgress) onProgress('Listo', 1.0);
      return this._litertLmEngines[modelUrl];
    }
    if (onProgress) {
      if (!this._progressListeners[modelUrl]) {
        this._progressListeners[modelUrl] = [];
      }
      this._progressListeners[modelUrl].push(onProgress);
    }
    if (this._litertLmLoading[modelUrl]) {
      return await this._litertLmLoading[modelUrl];
    }
    const notify = (status, progress) => {
      const list = this._progressListeners[modelUrl] || [];
      for (const cb of list) {
        try {
          cb(String(status || 'Cargando...'), Number(progress || 0));
        } catch (_) {}
      }
    };
    const loadTask = (async () => {
      notify('Descargando modelo...', 0.01);
      const blob = await this.fetchModelBlob(modelUrl, token, (status, prog) => {
        notify(status, Math.min(0.92, Number(prog || 0) * 0.92));
      });
      notify('Inicializando acelerador WebGPU...', 0.95);
      const mod = await this.ensureLiteRtLmCore();
      const backend = mod.Backend && mod.Backend.GPU !== undefined ? mod.Backend.GPU : undefined;
      const engine = await mod.Engine.create({
        model: blob,
        ...(backend !== undefined ? { backend } : {}),
      });
      this._litertLmEngines[modelUrl] = engine;
      this._loadedModels[modelUrl] = true;
      notify('Listo', 1.0);
      return engine;
    })();
    this._litertLmLoading[modelUrl] = loadTask;
    try {
      return await loadTask;
    } finally {
      delete this._litertLmLoading[modelUrl];
      delete this._progressListeners[modelUrl];
    }
  },

  async streamLiteRtLmCore(engine, modelUrl, prompt, maxTokens, temp, topP, topK, seed, onChunk, images) {
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
    if (!activeEngine || this._loadedModels[modelUrl] !== true) {
      activeEngine = await this.createLiteRtLmCoreEngine(modelUrl, '', null);
    }

    const lastMsg = messages[messages.length - 1];
    const prefaceMessages = messages.slice(0, -1);
    const prefaceSig = JSON.stringify(prefaceMessages);
    const samplerSig = [
      typeof maxTokens === 'number' ? Math.round(maxTokens) : 512,
      typeof temp === 'number' ? temp : 0.4,
      typeof topP === 'number' ? topP : 0.9,
      typeof topK === 'number' ? Math.round(topK) : 40,
      typeof seed === 'number' && Number.isFinite(seed) ? Math.trunc(seed) : '',
    ].join('|');

    const samplerParams = {
      temperature: typeof temp === 'number' ? temp : 0.4,
      p: typeof topP === 'number' ? topP : 0.9,
      k: typeof topK === 'number' && topK > 0 ? Math.round(topK) : 40,
    };
    if (typeof seed === 'number' && Number.isFinite(seed)) {
      samplerParams.seed = Math.trunc(seed);
    }

    // Reuse hot WebGPU KV-cache conversation when this turn appends to the
    // exact prior turn prefix (dramatically faster multi-turn Gemma 4 Web!).
    let convEntry = this._litertLmConvs[modelUrl];
    const canReuseConv =
      convEntry &&
      convEntry.conv &&
      convEntry.samplerSig === samplerSig &&
      convEntry.expectedNextPrefaceSig === prefaceSig;

    let conv;
    if (canReuseConv) {
      conv = convEntry.conv;
    } else {
      await this.resetLiteRtLmCoreConversation(modelUrl);
      const convConfig = {
        sessionConfig: {
          samplerParams,
          maxOutputTokens: typeof maxTokens === 'number' && maxTokens > 0 ? Math.round(maxTokens) : 512,
        },
        filterChannelContentFromKvCache: true,
      };
      if (prefaceMessages.length > 0) {
        convConfig.preface = { messages: prefaceMessages };
      }
      conv = await activeEngine.createConversation(convConfig);
    }

    let inputPayload = lastMsg.content;
    if (Array.isArray(images) && images.length > 0) {
      const parts = [];
      for (const item of images) {
        if (!item || !item.bytes) continue;
        const mime = typeof item.mimeType === 'string' && item.mimeType.length > 0
          ? item.mimeType
          : 'image/jpeg';
        const blob = new Blob([item.bytes], { type: mime });
        parts.push({ type: 'image', data: blob });
      }
      parts.push({ type: 'text', text: lastMsg.content });
      inputPayload = parts;
    }

    let full = '';
    let succeeded = false;
    try {
      const stream = conv.sendMessageStreaming(inputPayload);
      const reader = stream.getReader();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (!value) continue;
        if (value.channel && String(value.channel).toLowerCase() === 'thought') {
          continue;
        }
        let delta = '';
        if (typeof value.content === 'string') {
          delta = value.content;
        } else if (Array.isArray(value.content)) {
          for (const part of value.content) {
            if (part && part.type === 'text' && typeof part.text === 'string') {
              delta += part.text;
            }
          }
        }
        if (delta) {
          full += delta;
          if (onChunk) onChunk(delta);
        }
      }
      succeeded = true;
    } finally {
      if (succeeded) {
        const nextHistory = [
          ...messages,
          { role: 'assistant', content: full.trim() },
        ];
        this._litertLmConvs[modelUrl] = {
          conv,
          samplerSig,
          expectedNextPrefaceSig: JSON.stringify(nextHistory),
        };
      } else {
        delete this._litertLmConvs[modelUrl];
        try {
          await conv.delete();
        } catch (_) {}
      }
    }
    return full;
  },

  async isModelCached(url) {
    try {
      const opfsDir = await this._getOpfsDir(false);
      if (opfsDir) {
        try {
          const handle = await opfsDir.getFileHandle(this._opfsKeyForUrl(url));
          const file = await handle.getFile();
          if (file && file.size > 0) return true;
        } catch (_) {}
      }
    } catch (_) {}
    if (typeof caches === 'undefined') return false;
    try {
      const cache = await caches.open(this._modelCacheName);
      const match = await cache.match(url);
      return Boolean(match);
    } catch (_) {
      return false;
    }
  },

  async deleteCachedModel(url) {
    let removed = false;
    try {
      const opfsDir = await this._getOpfsDir(false);
      if (opfsDir) {
        try {
          await opfsDir.removeEntry(this._opfsKeyForUrl(url));
          removed = true;
        } catch (_) {}
      }
    } catch (_) {}
    if (typeof caches !== 'undefined') {
      try {
        const cache = await caches.open(this._modelCacheName);
        if (await cache.delete(url)) removed = true;
      } catch (_) {}
    }
    return removed;
  },

  async clearModelCache() {
    let cleared = false;
    try {
      if (typeof navigator !== 'undefined' &&
          navigator.storage &&
          typeof navigator.storage.getDirectory === 'function') {
        const root = await navigator.storage.getDirectory();
        try {
          await root.removeEntry(this._opfsDirName, { recursive: true });
          cleared = true;
        } catch (_) {}
      }
    } catch (_) {}
    if (typeof caches !== 'undefined') {
      try {
        if (await caches.delete(this._modelCacheName)) cleared = true;
      } catch (_) {}
    }
    return cleared;
  },

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

  async streamWebLlm(engine, modelId, prompt, maxTokens, temp, topP, topK, seed, onChunk) {
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

    const reqOpts = {
      messages,
      max_tokens: maxTokens || 512,
      temperature: typeof temp === 'number' ? temp : 0.4,
      top_p: typeof topP === 'number' ? topP : 0.9,
      stream: true,
    };
    if (typeof topK === 'number' && topK > 0) {
      reqOpts.top_k = Math.round(topK);
    }
    if (typeof seed === 'number' && Number.isFinite(seed)) {
      reqOpts.seed = Math.trunc(seed);
    }

    const createCompletion = async (eng) => eng.chat.completions.create(reqOpts);

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

  async runMpLlm(llm, prompt, maxTokens, temp, topP, topK, seed, onPartial) {
    const opts = {
      maxTokens: typeof maxTokens === 'number' && maxTokens > 0 ? Math.round(maxTokens) : 512,
      temperature: typeof temp === 'number' ? temp : 0.4,
      topP: typeof topP === 'number' ? topP : 0.9,
      topK: typeof topK === 'number' && topK > 0 ? Math.round(topK) : 40,
    };
    if (typeof seed === 'number' && Number.isFinite(seed)) {
      opts.randomSeed = Math.trunc(seed);
    }
    if (llm && typeof llm.setOptions === 'function') {
      try {
        await llm.setOptions(opts);
      } catch (_) {}
    }
    return await llm.generateResponse(prompt, onPartial);
  },

  async fetchModelBlob(url, token, onProgress) {
    const maxBytes = 4 * 1024 * 1024 * 1024;
    const parsed = new URL(url);
    const host = (parsed.hostname || '').toLowerCase().replace(/\.+$/, '');
    let isHfHost = false;
    if (parsed.origin !== 'null' && parsed.protocol === 'https:' && !parsed.username && !parsed.password) {
      if ('Origin' in globalThis) {
        try {
          const candidateOrigin = Origin.from(parsed);
          isHfHost = !candidateOrigin.opaque &&
            (candidateOrigin.isSameSite(Origin.from('https://huggingface.co')) ||
             candidateOrigin.isSameSite(Origin.from('https://hf.co')));
        } catch (_) {
          isHfHost = false;
        }
      } else {
        isHfHost = host === 'huggingface.co' || host.endsWith('.huggingface.co') ||
          host === 'hf.co' || host.endsWith('.hf.co');
      }
    }
    if (!isHfHost) {
      throw new Error('Untrusted Hugging Face host for authenticated fetch');
    }

    // 1. Check disk-backed OPFS first (zero-copy File/Blob ideal for 2.3–3.3 GB
    //    Gemma 4 .litertlm bundles without Cache API / RAM Blob spikes).
    try {
      const opfsDir = await this._getOpfsDir(false);
      if (opfsDir) {
        try {
          const handle = await opfsDir.getFileHandle(this._opfsKeyForUrl(url));
          const file = await handle.getFile();
          if (file && file.size > 0) {
            if (onProgress) onProgress('Cargando desde OPFS local...', 1.0);
            return file;
          }
        } catch (_) {}
      }
    } catch (_) {}

    // 1b. Check legacy Cache API next (backwards compatible with existing caches).
    const hasCacheApi = typeof caches !== 'undefined';
    if (hasCacheApi) {
      try {
        const cache = await caches.open(this._modelCacheName);
        const hit = await cache.match(url);
        if (hit) {
          if (onProgress) onProgress('Cargando desde caché local...', 1.0);
          return await hit.blob();
        }
      } catch (_) {}
    }

    const headers = Object.create(null);
    if (token && token.trim().length > 0) {
      headers['Authorization'] = 'Bearer ' + token.trim();
    }
    const res = await fetch(url, {
      headers,
      mode: 'cors',
      credentials: 'omit',
      referrerPolicy: 'no-referrer',
    });
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
    const rawLen = res.headers.get('content-length');
    const total = rawLen === null || rawLen === '' ? 0 : Number(rawLen);
    if (!Number.isFinite(total) || total < 0 || total > maxBytes) {
      throw new Error('Model artifact exceeds maximum allowed size (4 GiB)');
    }

    // 2. Prefer OPFS streaming write for large multi-GB models (O(1) RAM,
    //    persisted on disk, returns disk-backed File without >2 GiB Blob copy).
    if (res.body && res.body.getReader) {
      try {
        if (navigator.storage && typeof navigator.storage.persist === 'function') {
          try { await navigator.storage.persist(); } catch (_) {}
        }
        const opfsDir = await this._getOpfsDir(true);
        if (opfsDir) {
          const finalKey = this._opfsKeyForUrl(url);
          const partKey = finalKey + '.part';
          const partHandle = await opfsDir.getFileHandle(partKey, { create: true });
          if (typeof partHandle.createWritable === 'function') {
            const writable = await partHandle.createWritable();
            const reader = res.body.getReader();
            let loaded = 0;
            try {
              while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                loaded += value.byteLength;
                if (loaded > maxBytes) {
                  try { await reader.cancel(); } catch (_) {}
                  throw new Error('Model artifact stream exceeded maximum allowed size (4 GiB)');
                }
                await writable.write(value);
                if (onProgress && total > 0) {
                  onProgress('Descargando (' + Math.round((loaded / 1048576) * 10) / 10 + ' MB)...', loaded / total);
                }
              }
              await writable.close();
              if (typeof partHandle.move === 'function') {
                await partHandle.move(finalKey);
                const finalHandle = await opfsDir.getFileHandle(finalKey);
                return await finalHandle.getFile();
              }
              // Fallback if FileSystemHandle.move is absent: write directly to finalKey
              const finalHandle = await opfsDir.getFileHandle(finalKey, { create: true });
              const finalWritable = await finalHandle.createWritable();
              const partFile = await partHandle.getFile();
              await partFile.stream().pipeTo(finalWritable);
              try { await opfsDir.removeEntry(partKey); } catch (_) {}
              return await finalHandle.getFile();
            } catch (opfsWriteErr) {
              try { await writable.abort(); } catch (_) {}
              try { await opfsDir.removeEntry(partKey); } catch (_) {}
              const msg = String(opfsWriteErr || '');
              if (msg.includes('maximum allowed size')) throw opfsWriteErr;
              // Re-throw if stream already consumed so caller sees accurate error
              if (loaded > 0) throw opfsWriteErr;
            }
          }
        }
      } catch (opfsErr) {
        const msg = String(opfsErr || '');
        if (msg.includes('maximum allowed size') || res.bodyUsed) {
          throw opfsErr;
        }
      }
    }

    // 2b. Fallback: Stream network response into browser Cache API with O(1) RAM
    //     (prevents V8 WasmGC >1 GiB array traps when building with --wasm).
    if (hasCacheApi && res.body && res.body.getReader && typeof ReadableStream !== 'undefined') {
      try {
        if (navigator.storage && typeof navigator.storage.persist === 'function') {
          try { await navigator.storage.persist(); } catch (_) {}
        }
        const sourceReader = res.body.getReader();
        let loaded = 0;
        const progressStream = new ReadableStream({
          async pull(controller) {
            const { done, value } = await sourceReader.read();
            if (done) {
              controller.close();
              return;
            }
            loaded += value.byteLength;
            if (loaded > maxBytes) {
              try { await sourceReader.cancel(); } catch (_) {}
              controller.error(new Error('Model artifact stream exceeded maximum allowed size (4 GiB)'));
              return;
            }
            controller.enqueue(value);
            if (onProgress && total > 0) {
              onProgress('Descargando (' + Math.round((loaded / 1048576) * 10) / 10 + ' MB)...', loaded / total);
            }
          },
          cancel(reason) {
            return sourceReader.cancel(reason);
          },
        });

        const cache = await caches.open(this._modelCacheName);
        const cacheResp = new Response(progressStream, {
          status: 200,
          headers: { 'Content-Type': 'application/octet-stream' },
        });
        await cache.put(url, cacheResp);
        const stored = await cache.match(url);
        if (stored) {
          return await stored.blob();
        }
      } catch (cacheErr) {
        // If quota or storage error happened after stream started, propagate
        // size-cap violations and fall back only for Cache API unavailability.
        const msg = String(cacheErr || '');
        if (msg.includes('maximum allowed size')) {
          throw cacheErr;
        }
        throw cacheErr;
      }
    }

    if (!res.body || !res.body.getReader) {
      const blob = await res.blob();
      if (blob.size > maxBytes) {
        throw new Error('Model artifact exceeds maximum allowed size (4 GiB)');
      }
      return blob;
    }
    const reader = res.body.getReader();
    const parts = [];
    let loaded = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      loaded += value.byteLength;
      if (loaded > maxBytes) {
        try { await reader.cancel(); } catch (_) {}
        throw new Error('Model artifact stream exceeded maximum allowed size (4 GiB)');
      }
      parts.push(value);
      if (onProgress && total > 0) {
        onProgress('Descargando (' + Math.round((loaded / 1048576) * 10) / 10 + ' MB)...', loaded / total);
      }
    }
    return new Blob(parts, { type: 'application/octet-stream' });
  },

  async fetchWithHfToken(url, token, onProgress) {
    const blob = await this.fetchModelBlob(url, token, onProgress);
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
      return 'Percepción visual · Detección activa (' + dt + ' ms) · Detecciones: ' + count + ' · ' + topDesc;
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
      return 'Percepción visual · Clasificación activa (' + dt + ' ms) · Top: ' + (summary || 'OK');
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
    return 'Percepción acústica activa (' + dt + ' ms) · Señal 440 Hz → ' + (summary || 'OK');
  }
};
''';

  void _ensureWebAiHelper() {
    if (web.window.has('__saiaWebAi')) return;
    final script = web.HTMLScriptElement()
      ..crossOrigin = 'anonymous'
      ..referrerPolicy = 'no-referrer';
    _assignTrustedScriptText(script, _webAiHelperScript);
    (web.document.head ?? web.document.body)?.append(script);
  }

  JSObject get _webAiHelper {
    _ensureWebAiHelper();
    return web.window.getProperty<JSObject>('__saiaWebAi'.toJS);
  }

  Future<void> _loadModuleScript(String moduleUrl) async {
    final safeModuleUrl = LiteRtUrlPolicy.validateTrustedModuleUrl(moduleUrl);
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

    final script = web.HTMLScriptElement()
      ..type = 'module'
      ..crossOrigin = 'anonymous'
      ..referrerPolicy = 'no-referrer';
    _assignTrustedScriptText(
      script,
      LiteRtUrlPolicy.buildLmModuleLoaderScript(safeModuleUrl),
    );

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
      script.remove();
      web.window.delete(callbackName.toJS);
    }
  }

  Future<void> _loadModuleScriptGeneral(String moduleUrl) async {
    final safeModuleUrl = LiteRtUrlPolicy.validateTrustedModuleUrl(moduleUrl);
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

    final script = web.HTMLScriptElement()
      ..type = 'module'
      ..crossOrigin = 'anonymous'
      ..referrerPolicy = 'no-referrer';
    _assignTrustedScriptText(
      script,
      LiteRtUrlPolicy.buildCoreModuleLoaderScript(safeModuleUrl),
    );

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
      script.remove();
      web.window.delete(callbackName.toJS);
    }
  }

  /// Importa `@mediapipe/tasks-text` y expone sus clases en `window`.
  Future<void> _loadTextModuleScript(String moduleUrl) async {
    final safeModuleUrl = LiteRtUrlPolicy.validateTrustedModuleUrl(moduleUrl);
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

    final script = web.HTMLScriptElement()
      ..type = 'module'
      ..crossOrigin = 'anonymous'
      ..referrerPolicy = 'no-referrer';
    _assignTrustedScriptText(
      script,
      LiteRtUrlPolicy.buildTextModuleLoaderScript(safeModuleUrl),
    );

    script.addEventListener(
      'error',
      ((web.Event _) {
        if (!completer.isCompleted) {
          completer.completeError(
            StateError('Failed to load MediaPipe Text web module script'),
          );
        }
      }).toJS,
    );

    web.document.head?.append(script);
    try {
      await completer.future.timeout(const Duration(seconds: 20));
    } finally {
      script.remove();
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
        lower.endsWith('.litertlm') ||
        lower.startsWith('http://') ||
        lower.startsWith('https://') ||
        lower.startsWith('blob:');
  }

  static bool _isLiteRtLmBundleUrl(String modelPath) {
    final clean = modelPath.toLowerCase().split('?').first.split('#').first;
    return clean.endsWith('.litertlm');
  }

  Future<String> _resolveAuthenticatedUrlIfNeeded(
    String url, {
    void Function(String status, double progress)? onProgress,
  }) async {
    if (!LiteRtUrlPolicy.isTrustedHuggingFaceUrl(url)) return url;
    final token = huggingFaceToken;
    // Stream all Hugging Face bundles (gated and public) directly into the
    // browser Cache API so >1 GiB models stay off the WasmGC heap and persist
    // across reloads.
    final lowerUrl = url.toLowerCase();
    // Gemma 4 is published under Apache 2.0 (ungated on Hugging Face); only
    // legacy Gemma 2/3 / EmbeddingGemma repositories require an HF token.
    final isApacheGemma4 =
        lowerUrl.contains('gemma-4') || lowerUrl.contains('gemma4');
    final isGated = lowerUrl.contains('gemma') && !isApacheGemma4;
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
    final normalized = _normalizeLmModelPath(modelPath.trim());
    if (!_loadedLmCache.containsKey(normalized)) return false;
    if (_isLiteRtLmBundleUrl(normalized)) {
      try {
        final safeUrl = LiteRtUrlPolicy.validateModelAssetUrl(normalized);
        final ok = _webAiHelper.callMethod(
          'isLiteRtLmCoreLoaded'.toJS,
          safeUrl.toJS,
        );
        return ok != null && ok.isA<JSBoolean>() && (ok as JSBoolean).toDart;
      } catch (_) {
        return false;
      }
    }
    if (!_isTaskBundleUrl(normalized)) {
      try {
        final safeId = LiteRtUrlPolicy.validateWebLlmModelId(normalized);
        final ok = _webAiHelper.callMethod(
          'isWebLlmLoaded'.toJS,
          safeId.toJS,
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
    final normalized = _normalizeLmModelPath(modelPath.trim());
    final cached = _loadedLmCache.remove(normalized);
    if (_isLiteRtLmBundleUrl(normalized)) {
      try {
        final safeUrl = LiteRtUrlPolicy.validateModelAssetUrl(normalized);
        await (_webAiHelper.callMethod(
          'unloadLiteRtLmCore'.toJS,
          safeUrl.toJS,
        ) as JSPromise<JSAny?>)
            .toDart;
      } catch (_) {}
      return;
    }
    if (!_isTaskBundleUrl(normalized)) {
      try {
        final safeId = LiteRtUrlPolicy.validateWebLlmModelId(normalized);
        await (_webAiHelper.callMethod(
          'unloadWebLlmEngine'.toJS,
          safeId.toJS,
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
    final rawNormalized = _normalizeLmModelPath(modelPath.trim());
    final normalized = _isTaskBundleUrl(rawNormalized)
        ? LiteRtUrlPolicy.validateModelAssetUrl(rawNormalized)
        : LiteRtUrlPolicy.validateWebLlmModelId(rawNormalized);
    if (!LiteRtModelInfo.isWebCompatibleArtifact(normalized)) {
      throw UnsupportedError(
        'Language model artifact is not compatible with Web runtime: $normalized',
      );
    }
    if (isLmModelLoaded(normalized)) {
      final existing = _loadedLmCache[normalized];
      if (existing != null) return existing;
    }

    // 1. Vía WebLLM (modelos abiertos pre-compilados para WebGPU: Qwen3, Qwen2.5, SmolLM2)
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

    // 1b. Vía @litert-lm/core (LiteRT-LM.js) para bundles `.litertlm` (p. ej. Gemma 4 E2B/E4B)
    if (_isLiteRtLmBundleUrl(normalized)) {
      final lowerUrl = normalized.toLowerCase();
      final isApacheGemma4 =
          lowerUrl.contains('gemma-4') || lowerUrl.contains('gemma4');
      final token = huggingFaceToken;
      if (lowerUrl.contains('gemma') &&
          !isApacheGemma4 &&
          (token == null || token.isEmpty)) {
        throw StateError(
          'Este modelo requiere licencia aceptada y un Token HF (hf_...).',
        );
      }
      final progressCb = ((JSString status, JSNumber prog) {
        onProgress?.call(status.toDart, prog.toDartDouble);
      }).toJS;
      final engine = await (_webAiHelper.callMethod(
        'createLiteRtLmCoreEngine'.toJS,
        normalized.toJS,
        (token ?? '').toJS,
        progressCb,
      ) as JSPromise<JSObject>)
          .toDart;
      final model = LiteRtLmCoreModelWeb(
        _webAiHelper,
        engine,
        modelUrl: normalized,
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
    final createdTempBlobUrl =
        resolvedAssetPath != normalized && resolvedAssetPath.startsWith('blob:');

    try {
      final resolver = _MpFilesetResolver._(rawFileset as JSObject);
      final fileset =
          await resolver.forGenAiTasks('$_mpBase/wasm'.toJS).toDart;

      final baseOptions = JSObject();
      baseOptions.setProperty('modelAssetPath'.toJS, resolvedAssetPath.toJS);

      final options = JSObject();
      options.setProperty('baseOptions'.toJS, baseOptions);
      options.setProperty('maxTokens'.toJS, _maxTokensFor(normalized).toJS);
      options.setProperty('topK'.toJS, 40.toJS);
      options.setProperty('topP'.toJS, 0.9.toJS);
      options.setProperty('temperature'.toJS, 0.4.toJS);

      final constructor = _MpLlmInferenceConstructor._(rawLlm as JSObject);
      baseOptions.setProperty('delegate'.toJS, 'GPU'.toJS);
      try {
        final llm =
            await constructor.createFromOptions(fileset, options).toDart;
        final model = LiteRtLmModelWeb(
          _webAiHelper,
          llm,
          backend: 'WebGPU Local',
        );
        _loadedLmCache[normalized] = model;
        return model;
      } catch (e) {
        if (!_isDelegateRejection(e)) rethrow;
        throw LiteRtBundleIncompatible(normalized, e.toString());
      }
    } finally {
      if (createdTempBlobUrl) {
        try {
          web.URL.revokeObjectURL(resolvedAssetPath);
        } catch (_) {}
      }
    }
  }

  @override
  Future<LiteRtTextEmbedder> loadTextEmbedder(
    String modelPath, {
    bool l2Normalize = true,
  }) async {
    final safeModelPath = LiteRtUrlPolicy.validateModelAssetUrl(modelPath);
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

    final resolvedAssetPath =
        await _resolveAuthenticatedUrlIfNeeded(safeModelPath);
    final createdTempBlobUrl = resolvedAssetPath != safeModelPath &&
        resolvedAssetPath.startsWith('blob:');

    try {
      final resolver = _MpTextFilesetResolver._(rawFileset as JSObject);
      final fileset =
          await resolver.forTextTasks('$_mpTextBase/wasm'.toJS).toDart;

      final baseOptions = JSObject();
      baseOptions.setProperty('modelAssetPath'.toJS, resolvedAssetPath.toJS);

      final options = JSObject();
      options.setProperty('baseOptions'.toJS, baseOptions);
      options.setProperty('l2Normalize'.toJS, l2Normalize.toJS);
      options.setProperty('quantize'.toJS, false.toJS);

      final constructor =
          _MpTextEmbedderConstructor._(rawEmbedder as JSObject);
      final embedder =
          await constructor.createFromOptions(fileset, options).toDart;
      return LiteRtTextEmbedderWeb(embedder);
    } finally {
      if (createdTempBlobUrl) {
        try {
          web.URL.revokeObjectURL(resolvedAssetPath);
        } catch (_) {}
      }
    }
  }

  @override
  Future<String> verifyVisionModel(String modelPath) async {
    final safeModelPath = LiteRtUrlPolicy.validateModelAssetUrl(modelPath);
    final safeBase = LiteRtUrlPolicy.validateTrustedModuleUrl(_mpVisionBase);
    final res = await (_webAiHelper.callMethod(
      'verifyVision'.toJS,
      safeModelPath.toJS,
      safeBase.toJS,
    ) as JSPromise<JSString>)
        .toDart;
    return res.toDart;
  }

  @override
  Future<String> verifyAudioModel(String modelPath) async {
    final safeModelPath = LiteRtUrlPolicy.validateModelAssetUrl(modelPath);
    final safeBase = LiteRtUrlPolicy.validateTrustedModuleUrl(_mpAudioBase);
    final res = await (_webAiHelper.callMethod(
      'verifyAudio'.toJS,
      safeModelPath.toJS,
      safeBase.toJS,
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
      'Itzli compatibles con WebGPU (Itzli Chispa, Itzli Fugaz, Itzli Centella '
      'o Itzli Semilla).';
}

// ==========================================
// Model Wrappers
// ==========================================

class LiteRtModelWeb implements LiteRtModel {
  final _JsLiteRtModel _jsModel;

  LiteRtModelWeb(this._jsModel);

  @override
  Future<Float32List> run(Float32List input, List<int> shape) async {
    if (shape.isEmpty) {
      throw ArgumentError.value(shape, 'shape', 'Tensor shape must not be empty');
    }
    var expectedElements = 1;
    final jsShape = JSArray<JSNumber>();
    for (final s in shape) {
      if (!s.isFinite || s <= 0 || s > 268435456) {
        throw ArgumentError.value(
          shape,
          'shape',
          'Tensor shape dimensions must be positive finite integers',
        );
      }
      expectedElements *= s;
      if (!expectedElements.isFinite || expectedElements > 268435456) {
        throw ArgumentError.value(shape, 'shape', 'Tensor shape product overflow');
      }
      jsShape.add(s.toJS);
    }
    if (expectedElements != input.length) {
      throw ArgumentError(
        'Input length (${input.length}) does not match shape $shape ($expectedElements)',
      );
    }
    final jsData = input.toJS;

    final inputTensor = _JsLiteRtTensor(jsData, jsShape);
    final results = await _jsModel.run(inputTensor).toDart;
    final dartResults = results.toDart;
    if (dartResults.isEmpty) {
      throw StateError('LiteRT model returned 0 output tensors');
    }

    final outputTensor = dartResults[0];
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
    double? temperature,
    double topP = 0.9,
    int topK = 40,
    int? seed,
  }) {
    final controller = StreamController<String>();
    _runGeneration(
      controller,
      prompt,
      maxTokens,
      temperature ?? temp,
      topP,
      topK,
      seed,
    );
    return controller.stream;
  }

  Future<void> _runGeneration(
    StreamController<String> controller,
    String prompt,
    int maxTokens,
    double temp,
    double topP,
    int topK,
    int? seed,
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

      final safeMaxTokens =
          (maxTokens.isFinite && maxTokens >= 1) ? maxTokens.clamp(1, 32768) : 512;
      final safeTemp = temp.isFinite ? temp.clamp(0.0, 2.0) : 0.4;
      final safeTopP = topP.isFinite ? topP.clamp(0.0, 1.0) : 0.9;
      final safeTopK = (topK.isFinite && topK >= 1) ? topK.clamp(1, 1000) : 40;

      final full = await (_helper.callMethodVarArgs(
        'streamWebLlm'.toJS,
        [
          _engine,
          modelId.toJS,
          prompt.toJS,
          safeMaxTokens.toJS,
          safeTemp.toJS,
          safeTopP.toJS,
          safeTopK.toJS,
          seed?.toJS,
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
  final JSObject _helper;
  final _MpLlmInference _llm;
  bool _busy = false;

  @override
  final String backend;

  LiteRtLmModelWeb(this._helper, this._llm, {required this.backend});

  @override
  Stream<String> generateStream(
    String prompt, {
    int maxTokens = 512,
    double temp = 0.4,
    double? temperature,
    double topP = 0.9,
    int topK = 40,
    int? seed,
  }) {
    final controller = StreamController<String>();
    _runGeneration(
      controller,
      prompt,
      maxTokens,
      temperature ?? temp,
      topP,
      topK,
      seed,
    );
    return controller.stream;
  }

  Future<void> _runGeneration(
    StreamController<String> controller,
    String prompt,
    int maxTokens,
    double temp,
    double topP,
    int topK,
    int? seed,
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

      final safeMaxTokens =
          (maxTokens.isFinite && maxTokens >= 1) ? maxTokens.clamp(1, 32768) : 512;
      final safeTemp = temp.isFinite ? temp.clamp(0.0, 2.0) : 0.4;
      final safeTopP = topP.isFinite ? topP.clamp(0.0, 1.0) : 0.9;
      final safeTopK = (topK.isFinite && topK >= 1) ? topK.clamp(1, 1000) : 40;

      final full = await (_helper.callMethodVarArgs(
        'runMpLlm'.toJS,
        [
          _llm,
          prompt.toJS,
          safeMaxTokens.toJS,
          safeTemp.toJS,
          safeTopP.toJS,
          safeTopK.toJS,
          seed?.toJS,
          listener,
        ],
      ) as JSPromise<JSString>)
          .toDart;

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

/// Envoltorio WebGPU de `@litert-lm/core` (`LiteRT-LM.js`) para bundles `.litertlm`
/// modernos (p. ej. Gemma 4 E2B / E4B `-gpu.litertlm`) con KV-cache persistente
/// multi-turno y soporte multimodal de imágenes local.
class LiteRtLmCoreModelWeb implements LiteRtLmModel, LiteRtMultimodalLmModel {
  final JSObject _helper;
  final JSObject _engine;
  final String modelUrl;
  bool _busy = false;

  @override
  final String backend;

  @override
  bool get supportsImageInput =>
      modelUrl.toLowerCase().contains('gemma-4') ||
      modelUrl.toLowerCase().endsWith('.litertlm');

  LiteRtLmCoreModelWeb(
    this._helper,
    this._engine, {
    required this.modelUrl,
    required this.backend,
  });

  @override
  Stream<String> generateStream(
    String prompt, {
    int maxTokens = 512,
    double temp = 0.4,
    double? temperature,
    double topP = 0.9,
    int topK = 40,
    int? seed,
  }) {
    final controller = StreamController<String>();
    _runGeneration(
      controller,
      prompt,
      maxTokens,
      temperature ?? temp,
      topP,
      topK,
      seed,
      const [],
    );
    return controller.stream;
  }

  @override
  Stream<String> generateMultimodalStream(
    String prompt, {
    required List<({Uint8List bytes, String mimeType})> images,
    int maxTokens = 1024,
    double temp = 0.4,
    double topP = 0.9,
    int topK = 40,
    int? seed,
  }) {
    final controller = StreamController<String>();
    _runGeneration(
      controller,
      prompt,
      maxTokens,
      temp,
      topP,
      topK,
      seed,
      images,
    );
    return controller.stream;
  }

  @override
  Future<void> resetConversation() async {
    try {
      await (_helper.callMethod(
        'resetLiteRtLmCoreConversation'.toJS,
        modelUrl.toJS,
      ) as JSPromise<JSAny?>)
          .toDart;
    } catch (_) {}
  }

  Future<void> _runGeneration(
    StreamController<String> controller,
    String prompt,
    int maxTokens,
    double temp,
    double topP,
    int topK,
    int? seed,
    List<({Uint8List bytes, String mimeType})> images,
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

      final safeMaxTokens =
          (maxTokens.isFinite && maxTokens >= 1) ? maxTokens.clamp(1, 32768) : 512;
      final safeTemp = temp.isFinite ? temp.clamp(0.0, 2.0) : 0.4;
      final safeTopP = topP.isFinite ? topP.clamp(0.0, 1.0) : 0.9;
      final safeTopK = (topK.isFinite && topK >= 1) ? topK.clamp(1, 1000) : 40;

      JSArray<JSObject>? jsImages;
      if (images.isNotEmpty) {
        jsImages = JSArray<JSObject>();
        for (final item in images) {
          if (item.bytes.isEmpty) continue;
          final obj = JSObject();
          obj.setProperty('bytes'.toJS, item.bytes.toJS);
          obj.setProperty('mimeType'.toJS, item.mimeType.toJS);
          jsImages.add(obj);
        }
      }

      final full = await (_helper.callMethodVarArgs(
        'streamLiteRtLmCore'.toJS,
        [
          _engine,
          modelUrl.toJS,
          prompt.toJS,
          safeMaxTokens.toJS,
          safeTemp.toJS,
          safeTopP.toJS,
          safeTopK.toJS,
          seed?.toJS,
          onChunk,
          jsImages,
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
    // El ciclo de vida en VRAM del motor WebGPU es persistente hasta
    // llamada explícita a SaiaLiteRt.instance.unloadLmModel().
  }
}
