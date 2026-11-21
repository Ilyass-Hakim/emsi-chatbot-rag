import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_markdown_plus/flutter_markdown_plus.dart';
import 'package:image_picker/image_picker.dart';

import '../services/chat_api.dart';

/// One message of the conversation.
class ChatMessage {
  const ChatMessage({
    required this.text,
    required this.fromUser,
    this.photo,
    this.sources = const [],
  });

  final String text;
  final bool fromUser;
  final Uint8List? photo;
  final List<ChatSource> sources;
}

class ChatPage extends StatefulWidget {
  const ChatPage({super.key});

  @override
  State<ChatPage> createState() => _ChatPageState();
}

class _ChatPageState extends State<ChatPage> {
  static const _suggestions = [
    "Quelles filières propose l'EMSI ?",
    "Depuis quand existe l'EMSI ?",
    'Montre-moi les laboratoires',
  ];

  final _input = TextEditingController();
  final _scroll = ScrollController();
  final _messages = <ChatMessage>[];
  XFile? _photo;
  Uint8List? _photoBytes;
  bool _waiting = false;

  @override
  void dispose() {
    _input.dispose();
    _scroll.dispose();
    super.dispose();
  }

  Future<void> _pickPhoto() async {
    final photo = await ImagePicker().pickImage(
      source: ImageSource.gallery,
      maxWidth: 1280,
    );
    if (photo == null) return;
    final bytes = await photo.readAsBytes();
    setState(() {
      _photo = photo;
      _photoBytes = bytes;
    });
  }

  void _removePhoto() => setState(() {
        _photo = null;
        _photoBytes = null;
      });

  Future<void> _send([String? text]) async {
    final question = (text ?? _input.text).trim();
    if (question.isEmpty || _waiting) return;
    final photo = _photo;

    setState(() {
      _messages.add(ChatMessage(text: question, fromUser: true, photo: _photoBytes));
      _input.clear();
      _photo = null;
      _photoBytes = null;
      _waiting = true;
    });
    _scrollToBottom();

    ChatMessage reply;
    try {
      final answer = await ChatApi.ask(question, photo: photo);
      reply = ChatMessage(text: answer.answer, fromUser: false, sources: answer.sources);
    } catch (error) {
      debugPrint('Chat request failed: $error');
      reply = const ChatMessage(
        text: "Désolé, je n'arrive pas à joindre le serveur. "
            "Vérifiez que l'API est lancée, puis réessayez.",
        fromUser: false,
      );
    }
    if (!mounted) return;
    setState(() {
      _messages.add(reply);
      _waiting = false;
    });
    _scrollToBottom();
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scroll.hasClients) {
        _scroll.animateTo(
          _scroll.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('EMSI ChatBot'),
        centerTitle: true,
        backgroundColor: Colors.teal,
        foregroundColor: Colors.white,
      ),
      body: Column(
        children: [
          Expanded(child: _messages.isEmpty ? _welcome() : _conversation()),
          if (_photoBytes != null) _photoPreview(),
          _inputBar(),
        ],
      ),
    );
  }

  /// Shown before the first question: what the bot does and a few examples.
  Widget _welcome() {
    return Center(
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Image.asset('assets/images/logo.png', height: 96),
            const SizedBox(height: 16),
            const Text(
              "Posez-moi une question sur l'EMSI",
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            const Text(
              'Je réponds à partir de la brochure officielle et je cite mes sources.',
              textAlign: TextAlign.center,
              style: TextStyle(color: Colors.black54),
            ),
            const SizedBox(height: 24),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              alignment: WrapAlignment.center,
              children: [
                for (final suggestion in _suggestions)
                  ActionChip(label: Text(suggestion), onPressed: () => _send(suggestion)),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _conversation() {
    return ListView.builder(
      controller: _scroll,
      padding: const EdgeInsets.all(12),
      itemCount: _messages.length + (_waiting ? 1 : 0),
      itemBuilder: (context, index) =>
          index == _messages.length ? _typingBubble() : _bubble(_messages[index]),
    );
  }

  Widget _typingBubble() {
    return Align(
      alignment: Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 4),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        decoration: BoxDecoration(
          color: Colors.grey.shade200,
          borderRadius: BorderRadius.circular(16),
        ),
        child: const Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2)),
            SizedBox(width: 12),
            Text('EMSI ChatBot réfléchit...'),
          ],
        ),
      ),
    );
  }

  Widget _bubble(ChatMessage message) {
    final pages = {for (final source in message.sources) source.page}.toList()..sort();
    // Pictures are shown only when they rank among the 3 best sources
    final pictures = message.sources.take(3).where((source) => source.imageUrl != null);

    return Align(
      alignment: message.fromUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 4),
        padding: const EdgeInsets.all(12),
        constraints: const BoxConstraints(maxWidth: 360),
        decoration: BoxDecoration(
          color: message.fromUser ? Colors.teal : Colors.grey.shade200,
          borderRadius: BorderRadius.circular(16),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (message.photo != null)
              Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(8),
                  child: Image.memory(message.photo!, height: 140),
                ),
              ),
            if (message.fromUser)
              Text(message.text, style: const TextStyle(color: Colors.white))
            else
              MarkdownBody(data: message.text),
            for (final picture in pictures)
              Padding(
                padding: const EdgeInsets.only(top: 8),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(8),
                  child: Image.network(ChatApi.imageUrl(picture.imageUrl!), height: 140),
                ),
              ),
            if (pages.isNotEmpty) ...[
              const SizedBox(height: 8),
              const Text('Sources :', style: TextStyle(fontSize: 12, color: Colors.black54)),
              const SizedBox(height: 4),
              Wrap(
                spacing: 4,
                runSpacing: 4,
                children: [
                  for (final page in pages)
                    Chip(
                      label: Text('p. $page'),
                      visualDensity: VisualDensity.compact,
                      materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
                    ),
                ],
              ),
            ],
          ],
        ),
      ),
    );
  }

  /// The photo the user is about to send, with a button to remove it.
  Widget _photoPreview() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(12, 0, 12, 4),
      child: Row(
        children: [
          ClipRRect(
            borderRadius: BorderRadius.circular(8),
            child: Image.memory(_photoBytes!, width: 56, height: 56, fit: BoxFit.cover),
          ),
          const SizedBox(width: 8),
          Expanded(child: Text(_photo?.name ?? '', overflow: TextOverflow.ellipsis)),
          IconButton(
            tooltip: 'Retirer la photo',
            icon: const Icon(Icons.close),
            onPressed: _removePhoto,
          ),
        ],
      ),
    );
  }

  Widget _inputBar() {
    return SafeArea(
      top: false,
      child: Padding(
        padding: const EdgeInsets.fromLTRB(8, 4, 8, 8),
        child: Row(
          children: [
            IconButton(
              tooltip: 'Joindre une photo',
              icon: const Icon(Icons.add_photo_alternate_outlined),
              onPressed: _waiting ? null : _pickPhoto,
            ),
            Expanded(
              child: TextField(
                controller: _input,
                textInputAction: TextInputAction.send,
                onSubmitted: (_) => _send(),
                decoration: InputDecoration(
                  hintText: "Votre question sur l'EMSI...",
                  border: OutlineInputBorder(borderRadius: BorderRadius.circular(24)),
                  contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                ),
              ),
            ),
            const SizedBox(width: 4),
            IconButton.filled(
              tooltip: 'Envoyer',
              icon: const Icon(Icons.send),
              style: IconButton.styleFrom(backgroundColor: Colors.teal),
              onPressed: _waiting ? null : () => _send(),
            ),
          ],
        ),
      ),
    );
  }
}
