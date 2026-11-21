import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';

/// A part of the brochure the chatbot relied on to answer.
class ChatSource {
  const ChatSource({required this.page, required this.type, this.imageUrl});

  factory ChatSource.fromJson(Map<String, dynamic> json) => ChatSource(
        page: json['page'] as int,
        type: json['type'] as String,
        imageUrl: json['image_url'] as String?,
      );

  final int page;
  final String type;
  final String? imageUrl;
}

/// The answer of the chatbot and the sources it is based on.
class ChatReply {
  const ChatReply({required this.answer, required this.sources});

  factory ChatReply.fromJson(Map<String, dynamic> json) => ChatReply(
        answer: json['answer'] as String,
        sources: (json['sources'] as List)
            .map((source) => ChatSource.fromJson(source as Map<String, dynamic>))
            .toList(),
      );

  final String answer;
  final List<ChatSource> sources;
}

/// Client of the EMSI ChatBot back-end API.
class ChatApi {
  /// Address of the API. The Android emulator reaches the computer at
  /// 10.0.2.2; the web and desktop versions use localhost.
  static String get baseUrl =>
      !kIsWeb && defaultTargetPlatform == TargetPlatform.android
          ? 'http://10.0.2.2:8000'
          : 'http://localhost:8000';

  /// Full address of an image served by the API, such as /images/x.jpg.
  static String imageUrl(String path) => '$baseUrl$path';

  /// Sends a question, with an optional photo, and returns the answer.
  static Future<ChatReply> ask(String question, {XFile? photo}) async {
    final request = http.MultipartRequest('POST', Uri.parse('$baseUrl/chat'))
      ..fields['question'] = question;
    if (photo != null) {
      request.files.add(http.MultipartFile.fromBytes(
        'photo',
        await photo.readAsBytes(),
        filename: photo.name,
      ));
    }

    final response = await http.Response.fromStream(
      await request.send().timeout(const Duration(minutes: 2)),
    );
    if (response.statusCode != 200) {
      throw Exception('Server error ${response.statusCode}');
    }
    // utf8.decode keeps the French accents intact
    final json = jsonDecode(utf8.decode(response.bodyBytes));
    return ChatReply.fromJson(json as Map<String, dynamic>);
  }
}
