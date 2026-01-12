import 'dart:async';
import 'dart:convert';
import 'dart:developer' as developer;
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';

class ApiException implements Exception {
  final int? statusCode;
  final String message;
  final String? body;

  ApiException(this.message, {this.statusCode, this.body});

  @override
  String toString() =>
      'ApiException: $message (statusCode: $statusCode)${body != null ? ', body: $body' : ''}';
}

class ApiService {
  static const String baseUrl = 'https://hirdeshds-ai-4-alzheimers.hf.space';
  static const Duration timeout = Duration(seconds: 30);

  /// Sends the given image file to the prediction endpoint and returns a Map
  /// containing prediction and confidence on success.
  /// Throws [ApiException] on error with structured information (statusCode, body).
  static Future<Map<String, dynamic>> predictAlzheimer(File imageFile) async {
    final uri = Uri.parse('$baseUrl/predict');
    final filename = imageFile.path.split(RegExp(r'[\\/]+')).last;

    try {
      final length = await imageFile.length();
      developer.log(
        'prepare request',
        name: 'ApiService',
        error: null,
        stackTrace: null,
        level: 800,
      );
      developer.log(
        'Request info: url=$uri, filename=$filename, filesize=$length',
        name: 'ApiService',
      );

      final request = http.MultipartRequest('POST', uri);

      var stream = http.ByteStream(imageFile.openRead());
      var multipartFile = http.MultipartFile(
        'file',
        stream,
        length,
        filename: filename,
        contentType: MediaType('image', 'jpeg'),
      );

      request.files.add(multipartFile);

      final stopwatch = Stopwatch()..start();

      final streamedResponse = await request.send().timeout(timeout);

      final durationMs = stopwatch.elapsedMilliseconds;
      final responseData = await streamedResponse.stream.bytesToString();

      // Log status and truncated body for diagnostics
      final truncatedBody = responseData.length > 1000
          ? '${responseData.substring(0, 1000)}...[truncated]'
          : responseData;
      developer.log(
        'Response: status=${streamedResponse.statusCode}; time=${durationMs}ms; body=${truncatedBody}',
        name: 'ApiService',
      );

      if (streamedResponse.statusCode == 200) {
        // Parse JSON response
        try {
          final decoded = jsonDecode(responseData);

          if (decoded is Map) {
            // Extract prediction and confidence from the API response
            final prediction = decoded['prediction']?.toString() ?? 'Unknown';
            final confidence = decoded['confidence'] is num
                ? (decoded['confidence'] as num).toDouble()
                : 0.0;

            return {'prediction': prediction, 'confidence': confidence};
          } else {
            // Fallback if response isn't a map
            return {'prediction': responseData, 'confidence': 0.0};
          }
        } catch (e) {
          developer.log(
            'Failed to parse JSON response: $e',
            name: 'ApiService',
            level: 900,
          );
          // Return raw response with no confidence
          return {'prediction': responseData, 'confidence': 0.0};
        }
      } else {
        String serverMessage = responseData;
        try {
          final decoded = jsonDecode(responseData);
          if (decoded is Map &&
              (decoded['error'] != null || decoded['message'] != null)) {
            serverMessage = (decoded['error'] ?? decoded['message']).toString();
          }
        } catch (_) {}

        throw ApiException(
          'Request failed: $serverMessage',
          statusCode: streamedResponse.statusCode,
          body: responseData,
        );
      }
    } on TimeoutException {
      developer.log(
        'Request timed out after ${timeout.inSeconds}s',
        name: 'ApiService',
        level: 900,
      );
      throw ApiException('Request timed out after ${timeout.inSeconds}s');
    } on SocketException catch (e, st) {
      developer.log(
        'Network error: $e',
        name: 'ApiService',
        error: e,
        stackTrace: st,
        level: 1000,
      );
      throw ApiException('Network error: ${e.message}');
    } catch (e, st) {
      developer.log(
        'Unhandled exception: $e',
        name: 'ApiService',
        error: e,
        stackTrace: st,
        level: 1000,
      );
      throw ApiException('Unexpected error: $e');
    }
  }
}
