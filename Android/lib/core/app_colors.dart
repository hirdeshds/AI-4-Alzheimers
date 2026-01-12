import 'package:flutter/material.dart';

class AppColors {
  // Primary Colors - Modern Purple Gradient
  static const primary = Color(0xFF6366F1);
  static const primaryDark = Color(0xFF4F46E5);
  static const primaryLight = Color(0xFF818CF8);

  // Accent Colors
  static const accent = Color(0xFF8B5CF6);
  static const accentLight = Color(0xFFA78BFA);

  // Background Colors
  static const background = Color(0xFFF8FAFC);
  static const backgroundDark = Color(0xFF0F172A);
  static const surface = Colors.white;

  // Text Colors
  static const textDark = Color(0xFF0F172A);
  static const textMedium = Color(0xFF475569);
  static const textLight = Color(0xFF94A3B8);

  // Card Colors
  static const card = Colors.white;
  static const cardShadow = Color(0x1A000000);

  // Status Colors
  static const success = Color(0xFF10B981);
  static const warning = Color(0xFFF59E0B);
  static const error = Color(0xFFEF4444);
  static const info = Color(0xFF3B82F6);

  // Gradient Colors
  static const gradientStart = Color(0xFF6366F1);
  static const gradientEnd = Color(0xFF8B5CF6);

  static LinearGradient primaryGradient = const LinearGradient(
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
    colors: [gradientStart, gradientEnd],
  );

  static LinearGradient cardGradient = LinearGradient(
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
    colors: [Colors.white.withOpacity(0.9), Colors.white.withOpacity(0.6)],
  );
}
