import unittest

from emotion_attention.data import extract_emotion_field


class ExtractEmotionFieldTest(unittest.TestCase):
    def test_returns_emotion_when_present(self):
        row = {'emotion': 'joy', 'context': 'happy'}
        self.assertEqual(extract_emotion_field(row), 'joy')

    def test_falls_back_to_context(self):
        row = {'context': 'sadness'}
        self.assertEqual(extract_emotion_field(row), 'sadness')

    def test_handles_iterable_values(self):
        row = {'emotion': ['anger', 'sadness']}
        self.assertEqual(extract_emotion_field(row), 'anger')

    def test_handles_missing_values(self):
        row = {}
        self.assertEqual(extract_emotion_field(row), '')


if __name__ == '__main__':
    unittest.main()
