import unittest


class DataPipelineImportTests(unittest.TestCase):
    def test_data_pipeline_imports_as_package_module(self):
        import src.data_pipeline as data_pipeline

        self.assertTrue(hasattr(data_pipeline, "get_dataloaders"))


if __name__ == "__main__":
    unittest.main()
