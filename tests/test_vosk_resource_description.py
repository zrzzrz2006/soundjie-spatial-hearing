"""Check report terminology without loading fonts or generating desktop files."""
import ast
import re
import unittest
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / 'scripts/host/build_soundjie_submission_report.py'
TREE = ast.parse(SOURCE.read_text(encoding='utf-8'))
FUNCTION = next(node for node in TREE.body
                if isinstance(node, ast.FunctionDef) and node.name == 'validate_vosk_resource_description')
NAMESPACE = {'re': re}
exec(compile(ast.Module(body=[FUNCTION], type_ignores=[]), str(SOURCE), 'exec'), NAMESPACE)
validate = NAMESPACE['validate_vosk_resource_description']


class VoskResourceDescriptionTests(unittest.TestCase):
    def test_model_size_with_or_without_spaces_requires_no_memory_statement(self):
        for content in (
            'Vosk模型文件约为45MB',
            'Vosk 模型文件约 45 MB，运行内存占用约 362 MB。',
            'Vosk小型中文模型大小为45MB。运行时内存约为362MB。',
            '| Vosk 模型文件 | 约 45 MB |\n| 运行内存占用 | 约 362 MB |',
        ):
            with self.subTest(content=content):
                validate(content)

    def test_reject_swapped_figures(self):
        with self.assertRaises(AssertionError):
            validate('Vosk模型文件约362MB，运行内存占用约45MB。')

    def test_reject_unlabelled_or_missing_model_size(self):
        for content in ('Vosk模型文件未说明大小，运行内存362MB。', '45MB，运行内存362MB。'):
            with self.subTest(content=content), self.assertRaises(AssertionError):
                validate(content)

    def test_reject_conflicting_model_sizes(self):
        with self.assertRaises(AssertionError):
            validate('Vosk模型文件45MB，运行内存362MB。模型大小为362MB。')


if __name__ == '__main__':
    unittest.main()
