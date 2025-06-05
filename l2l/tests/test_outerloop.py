
import unittest


class OuterLoopTestCase(unittest.TestCase):

    def setUp(self):
        return

def suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(OuterLoopTestCase)
    return suite


def run():
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())