from pathlib import Path
import os
import unittest
from zipfile import ZipFile

from preprocessing.DataUpdater import DataUpdater


def recursiveDeleteData():

    dataPath = Path(__file__).parent.parent / 'data'

    # First, iterate over all files and delete them
    for file in dataPath.rglob('*'):
        if file.is_file():
            file.unlink()

    # Then, iterate over all directories and delete them
    for dir in sorted(dataPath.rglob('*'), reverse=True):
        if dir.is_dir():
            dir.rmdir()

    dataPath.rmdir()
    return None


def loadDataFromResources():

    zipDataPath = Path(__file__).parent.parent / 'resources' / 'data01.zip'


    with ZipFile(zipDataPath, 'r') as zipData:
        zipData.extractall(path=Path(__file__).parent.parent)
    
    return None



class Test_DataUpdater(unittest.TestCase):

    def setUp(self):
        super().setUp()

        self.DataUpdater = DataUpdater()

        # clean up
        recursiveDeleteData()
        
        # get example data
        loadDataFromResources()
        return None
    

    def test_getPaths(self):
        paths = self.DataUpdater._getPaths()

        self.assertIsInstance(paths, dict)
        self.assertListEqual(list(paths.keys()), ['path', 'csv', 'excel', 'corpus'])

        for key, path in paths.items():
            self.assertIsInstance(key, str)
            self.assertIsInstance(path, Path)

        return None


    def test_false(self):
        self.assertEqual(True, False)
        return None



if __name__ == '__main__':
    unittest.main()