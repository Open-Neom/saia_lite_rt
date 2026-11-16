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
}
