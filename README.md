# saia_lite_rt

![saia_lite_rt Banner](https://raw.githubusercontent.com/Open-Neom/saia_lite_rt/main/saia_lite_rt_banner.jpg)

On-device and web-accelerated runtime wrapper for LiteRT (TensorFlow Lite) and LiteRT-LM models in Flutter.

This package provides a unified, platform-aware interface to load and execute machine learning models directly on the client device (Edge Execution). It supports hardware-accelerated WebGPU/WASM runtime execution on Web, and stubs on native platforms.

## Features

* **Unified API:** Common interface for general `.tflite` model inference and generative `.litertlm` (LLMs) model streaming.
* **Zero-Config Web Acceleration:** Automatically loads and compiles the LiteRT core engine and WebAssembly/WebGPU components from CDNs dynamically. No manual `<script>` tags or WASM file copying is required in your web projects.
* **Built-in Model Catalog:** Includes out-of-the-box configurations and metadata for popular models (Gemma 3n, EmbeddingGemma 2, MiniLM Embeddings, MobileNet V3, YOLOv5, YAMNet).

## Installation

Add `saia_lite_rt` to your `pubspec.yaml` file:

```yaml
dependencies:
  saia_lite_rt: ^1.0.0
```

## Usage

### 1. Language Model Inference (Streaming LLM)

```dart
import 'package:saia_lite_rt/saia_lite_rt.dart';

void main() async {
  final runtime = SaiaLiteRt.instance; // Get platform-specific instance

  // 1. Initialize the LiteRT-LM engine
  await runtime.loadLiteRtLm();

  // 2. Load the model
  final lmModel = await runtime.loadLmModel('path/to/gemma-3n.litertlm');

  // 3. Generate a streaming response
  final prompt = 'What is the origin of obsidian?';
  final stream = lmModel.generateStream(prompt, maxTokens: 120);

  await for (final token in stream) {
    print(token); // Prints tokens in real-time
  }

  // Release resources
  await lmModel.dispose();
}
```

### 2. General Inference (TFLite Classification/Vision)

```dart
import 'package:saia_lite_rt/saia_lite_rt.dart';
import 'dart:typed_data';

void main() async {
  final runtime = SaiaLiteRt.instance;

  // 1. Initialize the general LiteRT engine
  await runtime.loadLiteRt();

  // 2. Load a classification model
  final model = await runtime.loadModel('path/to/mobilenet.tflite');

  // 3. Run inference
  final input = Float32List(224 * 224 * 3); // Normalized image tensor data
  final result = await model.run(input, [1, 224, 224, 3]);

  print('Inference output: $result');

  await model.dispose();
}
```
