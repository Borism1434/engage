import unittest

# Try to import all main modules/components
from matching.pipeline import *
from matching.utils.normalization import *
from matching.negatives.hard_negatives import *
from matching.modeling.train_eval import *
from matching.loaders.load_attempts import *
from matching.gold.gold_pairs import *
from matching.features.feature_builder import *
from matching.config.column_map import *
from matching.config.match_config import *

class TestImports(unsttest.TestCase):
    def test_imports(self):
        self.assertTrue(True)  # If any import fails, test will error out

if __name__ == "__main__":
    unittest.main()