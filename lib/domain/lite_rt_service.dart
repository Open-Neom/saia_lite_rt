import 'dart:convert';

import 'lite_rt_model.dart';

/// Abstract interface for LiteRT and LiteRT-LM web-accelerated runtime services.
abstract class SaiaLiteRtService {
  /// Loads the LiteRT core WebAssembly runtime binaries.
  ///
  /// [wasmPath] is the path (URL) to the directory containing WASM files.
  Future<void> loadLiteRt({String? wasmPath});

  /// Loads and compiles a general `.tflite` model.
  Future<LiteRtModel> loadModel(String modelPath, {String accelerator = 'webgpu'});

  /// Whether this platform really exposes a usable WebGPU adapter.
  ///
  /// Comprueba `navigator.gpu` y pide un adaptador de verdad, en vez de
  /// deducirlo del texto de un error. Sin esto es fácil culpar a WebGPU de
  /// fallos que no tienen nada que ver — descargas 401, CORS, memoria — y
  /// mandar al usuario a "activar WebGPU" cuando ya está activo.
  Future<bool> isWebGpuAvailable();

  /// Probes WebGPU availability, browser StorageManager quota (`quota - usage`),
  /// and `navigator.deviceMemory` to determine whether a lightweight local model
  /// can be added for internal orchestration without manual configuration.
  Future<LiteRtBrowserCapabilities> probeBrowserCapabilities();

  /// Loads the LiteRT-LM runtime for LLMs.
  ///
  /// [moduleUrl] apunta al paquete de MediaPipe GenAI si no está precargado.
  Future<void> loadLiteRtLm({String? moduleUrl});

  /// Loads and compiles a language model (WebLLM MLC ID or MediaPipe `.task` URL).
  Future<LiteRtLmModel> loadLmModel(
    String modelPath, {
    String accelerator = 'gpu',
    void Function(String status, double progress)? onProgress,
  });

  /// Explicitly unloads a previously loaded language model from GPU/WASM memory.
  Future<void> unloadLmModel(String modelPath);

  /// Returns true if [modelPath] is currently loaded and ready in memory.
  bool isLmModelLoaded(String modelPath);

  /// Carga un embebedor de texto desde un `.tflite` con metadatos de
  /// tokenizador (los de MediaPipe Text Embedder).
  ///
  /// [l2Normalize] deja el vector con norma 1, que es lo que hace que un
  /// producto interno sea similitud coseno — el modo en que compara
  /// saia_turbovec.
  Future<LiteRtTextEmbedder> loadTextEmbedder(
    String modelPath, {
    bool l2Normalize = true,
  });

  /// Carga y verifica en vivo un modelo `.tflite` de visión (ImageClassifier
  /// u ObjectDetector de MediaPipe Vision) y devuelve el diagnóstico real.
  Future<String> verifyVisionModel(String modelPath);

  /// Carga y verifica en vivo un modelo `.tflite` de audio (AudioClassifier
  /// de MediaPipe Audio, p. ej. YAMNet) con un tono de prueba a 16 kHz.
  Future<String> verifyAudioModel(String modelPath);

  /// Token opcional de HuggingFace (`hf_...`) para descargar modelos *gated*
  /// (como Gemma 3 `-web.task` o EmbeddingGemma).
  String? get huggingFaceToken;
  void setHuggingFaceToken(String? token);
}

/// Pure-Dart security policy for LiteRT / LiteRT-LM URLs and credentials.
class LiteRtUrlPolicy {
  LiteRtUrlPolicy._();

  static final RegExp _controlOrBidi = RegExp(
    r'[\x00-\x1F\x7F-\x9F\u0085\u200B-\u200F\u061C\u2028\u2029\u202A-\u202E\u2060-\u2069\uFEFF]',
  );
  static final RegExp _encodedUrlBidiRe = RegExp(
    r'%(?:e2%80%(?:8[b-f]|a[8-e])|e2%81%(?:a[0-9])|d8%9c|ef%bb%bf|c2%85)',
    caseSensitive: false,
  );

  static final RegExp _webLlmModelIdPattern = RegExp(
    r'^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$',
  );

  static const Set<String> _forbiddenProtoTokens = <String>{
    '__proto__',
    'constructor',
    'prototype',
  };

  static const Set<String> allowedAccelerators = <String>{
    'webgpu',
    'wasm',
    'cpu',
    'gpu',
  };

  /// Validates the hardware delegate / accelerator identifier passed to JS.
  static String validateAccelerator(String rawAccelerator) {
    if (_controlOrBidi.hasMatch(rawAccelerator)) {
      throw ArgumentError.value(
        rawAccelerator,
        'accelerator',
        'Invalid LiteRT accelerator',
      );
    }
    final clean = rawAccelerator.trim().toLowerCase();
    if (!allowedAccelerators.contains(clean)) {
      throw ArgumentError.value(
        rawAccelerator,
        'accelerator',
        'Unsupported LiteRT accelerator (expected one of $allowedAccelerators)',
      );
    }
    return clean;
  }

  static const Set<String> trustedModuleHosts = {
    'cdn.jsdelivr.net',
    'esm.run',
    'unpkg.com',
    'esm.sh',
    'storage.googleapis.com',
  };

  static bool _hasSuspiciousRawAuthority(String raw) {
    final schemeIdx = raw.indexOf('://');
    if (schemeIdx < 0) return false;
    final rest = raw.substring(schemeIdx + 3);
    final slashIdx = rest.indexOf('/');
    final queryIdx = rest.indexOf('?');
    final hashIdx = rest.indexOf('#');
    var end = rest.length;
    if (slashIdx >= 0 && slashIdx < end) end = slashIdx;
    if (queryIdx >= 0 && queryIdx < end) end = queryIdx;
    if (hashIdx >= 0 && hashIdx < end) end = hashIdx;
    final rawAuthority = rest.substring(0, end);
    return rawAuthority.isEmpty ||
        rawAuthority.contains('@') ||
        rawAuthority.contains('%') ||
        rawAuthority.contains('\\') ||
        rawAuthority.contains('\u3002') ||
        rawAuthority.contains('\uFF0E') ||
        rawAuthority.contains('\uFF61');
  }

  static String _normalizeHost(String rawHost) {
    var host = rawHost.trim().toLowerCase();
    if (host.startsWith('[') && host.endsWith(']') && host.length > 2) {
      host = host.substring(1, host.length - 1);
    }
    while (host.endsWith('.')) {
      host = host.substring(0, host.length - 1);
    }
    return host;
  }

  static List<int>? _tryParseIpv4Octets(String host) {
    if (host.isEmpty || host.contains(':')) return null;
    final parts = host.split('.');
    if (parts.isEmpty || parts.length > 4) return null;
    final parsed = <int>[];
    for (final part in parts) {
      if (part.isEmpty) return null;
      int? val;
      if (part.startsWith('0x') || part.startsWith('0X')) {
        if (part.length <= 2 || part.length > 10) return null;
        val = int.tryParse(part.substring(2), radix: 16);
      } else if (part.length > 1 && part.startsWith('0')) {
        if (!RegExp(r'^[0-7]+$').hasMatch(part)) return null;
        val = int.tryParse(part, radix: 8);
      } else if (RegExp(r'^\d+$').hasMatch(part)) {
        if (part.length > 10) return null;
        val = int.tryParse(part, radix: 10);
      } else {
        return null;
      }
      if (val == null || val < 0) return null;
      parsed.add(val);
    }
    int ipv4Int;
    if (parsed.length == 1) {
      if (parsed[0] > 0xFFFFFFFF) return null;
      ipv4Int = parsed[0];
    } else if (parsed.length == 2) {
      if (parsed[0] > 0xFF || parsed[1] > 0xFFFFFF) return null;
      ipv4Int = parsed[0] * 0x1000000 + parsed[1];
    } else if (parsed.length == 3) {
      if (parsed[0] > 0xFF || parsed[1] > 0xFF || parsed[2] > 0xFFFF) {
        return null;
      }
      ipv4Int = parsed[0] * 0x1000000 + parsed[1] * 0x10000 + parsed[2];
    } else {
      if (parsed.any((v) => v > 0xFF)) return null;
      ipv4Int = parsed[0] * 0x1000000 +
          parsed[1] * 0x10000 +
          parsed[2] * 0x100 +
          parsed[3];
    }
    return <int>[
      (ipv4Int ~/ 0x1000000) & 0xFF,
      (ipv4Int ~/ 0x10000) & 0xFF,
      (ipv4Int ~/ 0x100) & 0xFF,
      ipv4Int & 0xFF,
    ];
  }

  static bool _hasRawDotSegmentOrDoubleEncoding(String raw) {
    final lower = raw.toLowerCase();
    final pathAndAuth = lower.split('?').first.split('#').first;
    return lower.contains('%5c') ||
        lower.contains('%00') ||
        lower.contains('%25') ||
        lower.contains('%2f') ||
        _encodedUrlBidiRe.hasMatch(lower) ||
        pathAndAuth.contains('%2e%2e') ||
        pathAndAuth.contains('%2e.') ||
        pathAndAuth.contains('.%2e') ||
        RegExp(r'(?:^|/)(\.\.)(?:/|$)').hasMatch(pathAndAuth);
  }

  static bool _isPrivateOrReservedHost(String rawHost) {
    final host = _normalizeHost(rawHost);
    if (host.isEmpty || host.contains('%')) return true;
    if (host == 'localhost' ||
        host.endsWith('.localhost') ||
        host.endsWith('.local') ||
        host.endsWith('.internal') ||
        host.endsWith('.invalid') ||
        host.endsWith('.corp') ||
        host.endsWith('.lan') ||
        host.endsWith('.home.arpa') ||
        host.endsWith('.onion') ||
        host.endsWith('.ts.net') ||
        host == 'metadata.google.internal' ||
        host.endsWith('.metadata.google.internal') ||
        host == 'metadata.goog' ||
        host.endsWith('.metadata.goog') ||
        host == 'metadata.azure.com' ||
        host.endsWith('.metadata.azure.com') ||
        host == 'metadata.azure.internal' ||
        host == '100.100.100.200' ||
        host == 'fd00:ec2::254' ||
        host == 'metadata' ||
        host == 'instance-data' ||
        (!host.contains('.') && !host.contains(':'))) {
      return true;
    }

    if (host == '::1' ||
        host == '0:0:0:0:0:0:0:1' ||
        host == '::' ||
        host == '0:0:0:0:0:0:0:0' ||
        RegExp(r'^0+(?::0+)*$').hasMatch(host) ||
        host.startsWith('fe8') ||
        host.startsWith('fe9') ||
        host.startsWith('fea') ||
        host.startsWith('feb') ||
        host.startsWith('fc') ||
        host.startsWith('fd') ||
        host.startsWith('ff') ||
        host.startsWith('::ffff:') ||
        host.startsWith('0:0:0:0:0:ffff:') ||
        host.startsWith('64:ff9b:') ||
        host.startsWith('0:64:ff9b:') ||
        host.startsWith('0064:ff9b:') ||
        host.startsWith('2001:0:') ||
        host.startsWith('2001:0000:') ||
        host.startsWith('2001:db8:') ||
        host.startsWith('2002:')) {
      return true;
    }

    final octets = _tryParseIpv4Octets(host);
    if (octets != null) {
      if (!RegExp(r'^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$')
              .hasMatch(host) ||
          host.split('.').any((p) => p.length > 1 && p.startsWith('0'))) {
        return true;
      }
      final a = octets[0];
      final b = octets[1];
      final c = octets[2];
      if (a == 0 || a == 10 || a == 127) return true;
      if (a == 169 && b == 254) return true;
      if (a == 172 && b >= 16 && b <= 31) return true;
      if (a == 192 && b == 0 && (c == 0 || c == 2)) return true;
      if (a == 192 && b == 168) return true;
      if (a == 100 && b >= 64 && b <= 127) return true;
      if (a == 198 && (b == 18 || b == 19)) return true;
      if (a == 198 && b == 51 && c == 100) return true;
      if (a == 203 && b == 0 && c == 113) return true;
      if (a >= 224) return true;
    }

    return false;
  }

  /// Returns `true` iff [rawUrl] is an `https://` URL on `huggingface.co` or
  /// `hf.co` (or a subdomain thereof) with no embedded `userInfo`.
  static bool isTrustedHuggingFaceUrl(String rawUrl) {
    if (_controlOrBidi.hasMatch(rawUrl)) return false;
    final trimmed = rawUrl.trim();
    if (trimmed.isEmpty ||
        trimmed.length > 2048 ||
        _controlOrBidi.hasMatch(trimmed) ||
        trimmed.contains('\\') ||
        _hasRawDotSegmentOrDoubleEncoding(trimmed) ||
        _hasSuspiciousRawAuthority(trimmed)) {
      return false;
    }
    final uri = Uri.tryParse(trimmed);
    if (uri == null ||
        !uri.hasScheme ||
        uri.scheme.toLowerCase() != 'https' ||
        uri.userInfo.isNotEmpty ||
        uri.pathSegments.any((seg) => seg == '..')) {
      return false;
    }
    final host = _normalizeHost(uri.host);
    if (host.isEmpty) return false;
    return host == 'huggingface.co' ||
        host.endsWith('.huggingface.co') ||
        host == 'hf.co' ||
        host.endsWith('.hf.co');
  }

  /// Validates a dynamic ES module or WASM directory URL before injecting it
  /// into `<script type="module">` or `loadLiteRt()`.
  ///
  /// Allows same-origin relative paths (without `..` or `//`) and `https://`
  /// URLs on [trustedModuleHosts].
  static String validateTrustedModuleUrl(String rawUrl) {
    if (_controlOrBidi.hasMatch(rawUrl)) {
      throw ArgumentError.value(rawUrl, 'rawUrl', 'Empty or invalid module URL');
    }
    final trimmed = rawUrl.trim();
    if (trimmed.isEmpty ||
        trimmed.length > 2048 ||
        _controlOrBidi.hasMatch(trimmed) ||
        trimmed.contains('\\') ||
        _hasRawDotSegmentOrDoubleEncoding(trimmed)) {
      throw ArgumentError.value(rawUrl, 'rawUrl', 'Empty or invalid module URL');
    }
    if (trimmed.startsWith('//')) {
      throw ArgumentError.value(
        rawUrl,
        'rawUrl',
        'Protocol-relative module URLs are not allowed',
      );
    }

    final uri = Uri.tryParse(trimmed);
    if (uri == null) {
      throw ArgumentError.value(rawUrl, 'rawUrl', 'Malformed module URL');
    }

    if (!uri.hasScheme) {
      if (uri.hasAuthority ||
          uri.pathSegments.any((seg) => seg == '..')) {
        throw ArgumentError.value(
          rawUrl,
          'rawUrl',
          'Relative module URL cannot traverse parent directories or set authority',
        );
      }
      return trimmed;
    }

    if (_hasSuspiciousRawAuthority(trimmed) ||
        uri.scheme.toLowerCase() != 'https' ||
        uri.userInfo.isNotEmpty ||
        uri.pathSegments.any((seg) => seg == '..')) {
      throw ArgumentError.value(
        rawUrl,
        'rawUrl',
        'Module URL must use https:// without userInfo or traversal',
      );
    }

    final host = _normalizeHost(uri.host);
    final isAllowedHost = trustedModuleHosts.any(
      (allowed) => host == allowed || host.endsWith('.$allowed'),
    );
    if (!isAllowedHost) {
      throw ArgumentError.value(
        rawUrl,
        'rawUrl',
        'Untrusted module host: $host',
      );
    }
    return uri.toString();
  }

  /// Builds the canonical ES-module loader script for MediaPipe GenAI (`LlmInference`).
  static String buildLmModuleLoaderScript(String moduleUrl) {
    final safeModuleUrl = validateTrustedModuleUrl(moduleUrl);
    const callbackName = '__saia_litert_lm_init';
    return '''
      try {
        const mod = await import(${jsonEncode(safeModuleUrl)});
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
      } catch (e) {
        if (window.$callbackName) {
          window.$callbackName(String(e && e.message ? e.message : e));
        }
      }
      ''';
  }

  /// Builds the canonical ES-module loader script for `@litertjs/core`.
  static String buildCoreModuleLoaderScript(String moduleUrl) {
    final safeModuleUrl = validateTrustedModuleUrl(moduleUrl);
    const callbackName = '__saia_litert_core_init';
    return '''
      try {
        const mod = await import(${jsonEncode(safeModuleUrl)});
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
      } catch (e) {
        if (window.$callbackName) {
          window.$callbackName(e);
        }
      }
      ''';
  }

  /// Builds the canonical ES-module loader script for `@mediapipe/tasks-text`.
  static String buildTextModuleLoaderScript(String moduleUrl) {
    final safeModuleUrl = validateTrustedModuleUrl(moduleUrl);
    const callbackName = '__saia_mp_text_init';
    return '''
      try {
        const mod = await import(${jsonEncode(safeModuleUrl)});
        if (!mod?.TextEmbedder || !mod?.FilesetResolver) {
          throw new Error(
            'El modulo no expone TextEmbedder/FilesetResolver: ' +
            Object.keys(mod || {}).join(',')
          );
        }
        window.__saia_mp_TextEmbedder = mod.TextEmbedder;
        window.__saia_mp_TextFilesetResolver = mod.FilesetResolver;
        if (window.$callbackName) window.$callbackName(null);
      } catch (e) {
        if (window.$callbackName) {
          window.$callbackName(String(e && e.message ? e.message : e));
        }
      }
      ''';
  }

  static final RegExp _dynamicImportLiteral = RegExp(
    r'const mod = await import\(("(?:\\.|[^"\\])*")\);',
  );

  /// Enforces that a `TrustedScript` candidate is either the exact static
  /// [helperScript] or an unmodified canonical ESM loader importing a URL
  /// validated by [validateTrustedModuleUrl] (never a pass-through policy).
  static bool isAuthorizedInlineScript(String code, {String? helperScript}) {
    if (code.isEmpty) return false;
    if (helperScript != null && code == helperScript) {
      return true;
    }
    final match = _dynamicImportLiteral.firstMatch(code);
    if (match == null) return false;
    try {
      final decoded = jsonDecode(match.group(1)!);
      if (decoded is! String) return false;
      final safeUrl = validateTrustedModuleUrl(decoded);
      return code == buildLmModuleLoaderScript(safeUrl) ||
          code == buildCoreModuleLoaderScript(safeUrl) ||
          code == buildTextModuleLoaderScript(safeUrl);
    } catch (_) {
      return false;
    }
  }

  /// Validates a model asset URL (`.tflite`, `.task`, `.bin`) before loading
  /// it in the browser runtime.
  ///
  /// Allows safe relative asset paths, ephemeral `blob:` URLs, and public
  /// `https://` URLs (blocking `javascript:`, `data:`, `http://`, RFC1918,
  /// loopback, and cloud metadata SSRF targets).
  static String validateModelAssetUrl(String rawUrl) {
    if (_controlOrBidi.hasMatch(rawUrl)) {
      throw ArgumentError.value(rawUrl, 'modelPath', 'Invalid model asset URL');
    }
    final trimmed = rawUrl.trim();
    if (trimmed.isEmpty ||
        trimmed.length > 2048 ||
        _controlOrBidi.hasMatch(trimmed) ||
        trimmed.contains('\\') ||
        _hasRawDotSegmentOrDoubleEncoding(trimmed)) {
      throw ArgumentError.value(rawUrl, 'modelPath', 'Invalid model asset URL');
    }
    if (trimmed.startsWith('//')) {
      throw ArgumentError.value(
        rawUrl,
        'modelPath',
        'Protocol-relative model asset URLs are not allowed',
      );
    }

    if (trimmed.toLowerCase().startsWith('blob:')) {
      final inner = trimmed.substring(5);
      final innerUri = Uri.tryParse(inner);
      if (innerUri == null ||
          _hasSuspiciousRawAuthority(inner) ||
          _hasRawDotSegmentOrDoubleEncoding(inner) ||
          !innerUri.hasScheme ||
          (innerUri.scheme != 'https' && innerUri.scheme != 'http') ||
          innerUri.userInfo.isNotEmpty ||
          innerUri.pathSegments.any((seg) => seg == '..')) {
        throw ArgumentError.value(rawUrl, 'modelPath', 'Malformed blob: URL');
      }
      final innerHost = _normalizeHost(innerUri.host);
      final isLocalDevOrigin = innerHost == 'localhost' ||
          innerHost == '127.0.0.1' ||
          innerHost == '::1';
      if (!isLocalDevOrigin &&
          (innerUri.scheme != 'https' || _isPrivateOrReservedHost(innerHost))) {
        throw ArgumentError.value(
          rawUrl,
          'modelPath',
          'Untrusted or private blob: origin',
        );
      }
      return trimmed;
    }

    final uri = Uri.tryParse(trimmed);
    if (uri == null) {
      throw ArgumentError.value(rawUrl, 'modelPath', 'Malformed model URL');
    }

    if (!uri.hasScheme) {
      if (uri.hasAuthority || uri.pathSegments.any((seg) => seg == '..')) {
        throw ArgumentError.value(
          rawUrl,
          'modelPath',
          'Relative model path cannot traverse parent directories',
        );
      }
      return trimmed;
    }

    if (_hasSuspiciousRawAuthority(trimmed) ||
        uri.scheme.toLowerCase() != 'https' ||
        uri.userInfo.isNotEmpty ||
        uri.pathSegments.any((seg) => seg == '..')) {
      throw ArgumentError.value(
        rawUrl,
        'modelPath',
        'Model asset URL must use https:// without userInfo or traversal',
      );
    }
    if (_isPrivateOrReservedHost(uri.host)) {
      throw ArgumentError.value(
        rawUrl,
        'modelPath',
        'Private, loopback, or metadata model hosts are forbidden',
      );
    }
    return uri.toString();
  }

  /// Validates a WebLLM MLC model identifier (e.g. `Qwen2.5-0.5B-Instruct-q4f16_1-MLC`).
  static String validateWebLlmModelId(String rawModelId) {
    if (_controlOrBidi.hasMatch(rawModelId)) {
      throw ArgumentError.value(
        rawModelId,
        'modelId',
        'Invalid WebLLM model identifier',
      );
    }
    final trimmed = rawModelId.trim();
    if (!_webLlmModelIdPattern.hasMatch(trimmed) ||
        trimmed == '.' ||
        trimmed.contains('..') ||
        _forbiddenProtoTokens.contains(trimmed.toLowerCase())) {
      throw ArgumentError.value(
        rawModelId,
        'modelId',
        'Invalid WebLLM model identifier',
      );
    }
    return trimmed;
  }

  /// Normalizes and validates an optional Hugging Face bearer token.
  static String? sanitizeHfToken(String? rawToken) {
    if (rawToken == null) return null;
    final clean = rawToken.replaceAll(RegExp(r'[\r\n\x00]'), '').trim();
    if (clean.isEmpty) return null;
    if (clean.length > 512 ||
        RegExp(r'\s').hasMatch(clean) ||
        _controlOrBidi.hasMatch(clean)) {
      throw ArgumentError.value(
        '[REDACTED]',
        'token',
        'Invalid Hugging Face token format',
      );
    }
    return clean;
  }
}
