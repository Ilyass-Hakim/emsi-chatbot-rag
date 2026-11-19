import 'package:flutter_test/flutter_test.dart';

import 'package:emsi_chatbot/main.dart';

void main() {
  testWidgets('App starts on the login screen', (WidgetTester tester) async {
    await tester.pumpWidget(const MyApp());

    expect(find.text('Welcome Back!'), findsOneWidget);
    expect(find.text('Login'), findsWidgets);
  });
}
