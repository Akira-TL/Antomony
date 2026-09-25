"""项目技能清单的机械契约测试，不启动模型或外部服务。"""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.project_skills import HEADER, load_skills


class SkillManifestTests(unittest.TestCase):
    def test_real_manifest_has_required_routers(self) -> None:
        root = Path(__file__).resolve().parents[1]
        names = {item.name for item in load_skills(root / '.agents/skill-view.tsv')}
        self.assertEqual(len(names), 44)
        self.assertTrue({'ask-akira', 'ask-matt', 'akira-research', 'research-tree'} <= names)

    def test_header_is_required(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'skills.tsv'
            path.write_text('incorrect\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                load_skills(path)

    def test_duplicate_name_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'skills.tsv'
            row = 'akira\trepo\tmain\tcommit\tpath\n'
            path.write_text('\t'.join(HEADER) + '\n' + row + row, encoding='utf-8')
            with self.assertRaises(ValueError):
                load_skills(path)

    def test_parent_path_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'skills.tsv'
            path.write_text('\t'.join(HEADER) + '\n../escape\trepo\tmain\tcommit\tpath\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                load_skills(path)


if __name__ == '__main__':
    unittest.main()
