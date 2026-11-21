import 'package:flutter_test/flutter_test.dart';
import 'package:saia_lite_rt/saia_lite_rt.dart';

void main() {
  test('LiteRtModelCatalog.webCompatible excludes native .gguf and .cact artifacts', () {
    final webModels = LiteRtModelCatalog.webCompatible;
    expect(webModels, isNotEmpty);

    for (final model in webModels) {
      final lower = model.downloadUrl.toLowerCase();
      expect(lower.endsWith('.gguf'), isFalse, reason: model.id);
      expect(lower.endsWith('.cact'), isFalse, reason: model.id);
      expect(model.isWebCompatible, isTrue, reason: model.id);
    }

    final excludedIds = LiteRtModelCatalog.catalogEntries
        .where((m) => !m.isWebCompatible)
        .map((m) => m.id)
        .toSet();
    expect(excludedIds, containsAll(['kitten-tts-2-emb4', 'needle-3-mobile', 'whistle-stt-int4']));
  });

  test('LiteRtUrlPolicy core loader exposes window.LiteRtTensor', () {
    final script = LiteRtUrlPolicy.buildCoreModuleLoaderScript(
      'https://cdn.jsdelivr.net/npm/@litertjs/core/+esm',
    );
    expect(script, contains('window.LiteRtTensor = mod.Tensor'));
  });

  test('Cache API contract validates URLs and exposes stable bucket name', () async {
    expect(LiteRtUrlPolicy.defaultWebModelCacheName, 'saia-litert-models-v1');
    final svc = SaiaLiteRt.instance;
    expect(
      await svc.isModelCached(
        'https://huggingface.co/litert-community/gemma-3-1b-it/resolve/main/model.task',
      ),
      isFalse,
    );
    expect(
      await svc.deleteCachedModel(
        'https://huggingface.co/litert-community/gemma-3-1b-it/resolve/main/model.task',
      ),
      isFalse,
    );
    expect(await svc.clearModelCache(), isFalse);
    await expectLater(
      () => svc.isModelCached('http://169.254.169.254/latest/meta-data'),
      throwsArgumentError,
    );
  });

  test('Gemma 4 E2B and E4B .litertlm bundles are web-compatible, ungated, and sovereignly branded', () {
    final e2b = LiteRtModelCatalog.findById('gemma4-e2b-web');
    final e4b = LiteRtModelCatalog.findById('gemma4-e4b-web');
    expect(e2b, isNotNull);
    expect(e4b, isNotNull);
    expect(e2b!.name, 'Itzli Centella · E2B');
    expect(e4b!.name, 'Itzli Semilla · E4B');
    expect(e2b.isWebCompatible, isTrue);
    expect(e4b.isWebCompatible, isTrue);
    expect(e2b.requiresHfToken, isFalse);
    expect(e4b.requiresHfToken, isFalse);
    expect(e2b.downloadUrl.endsWith('.litertlm'), isTrue);
    expect(e4b.downloadUrl.endsWith('.litertlm'), isTrue);
  });
}
