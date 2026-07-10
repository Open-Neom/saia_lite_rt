import 'package:flutter/material.dart';
import 'package:saia_lite_rt/saia_lite_rt.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Saia LiteRT Example',
      theme: ThemeData.dark(),
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final _runtime = SaiaLiteRt.instance;
  String _status = 'Not initialized';
  final List<String> _tokens = [];

  Future<void> _runModel() async {
    setState(() {
      _status = 'Initializing LiteRT-LM...';
      _tokens.clear();
    });

    try {
      await _runtime.loadLiteRtLm();
      setState(() {
        _status = 'Loading model...';
      });

      // Simulate loading Gemma 3n (Nano)
      final model = await _runtime.loadLmModel('models/gemma-3n.litertlm');

      setState(() {
        _status = 'Generating response...';
      });

      final stream = model.generateStream('Hello Gemma, how are you?');
      await for (final token in stream) {
        setState(() {
          _tokens.add(token);
        });
      }

      setState(() {
        _status = 'Inference completed';
      });

      await model.dispose();
    } catch (e) {
      setState(() {
        _status = 'Error: $e';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Saia LiteRT')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        key: const ValueKey('body_padding'),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            ElevatedButton(
              onPressed: _runModel,
              child: const Text('Test Local Gemma 3n'),
            ),
            const SizedBox(height: 20),
            Text(
              'Status: $_status',
              style: const TextStyle(fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 20),
            const Text('Response:'),
            Expanded(
              child: Container(
                padding: const EdgeInsets.all(8),
                color: Colors.black26,
                child: SingleChildScrollView(
                  child: Text(_tokens.join('')),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
