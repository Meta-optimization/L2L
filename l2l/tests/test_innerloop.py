
import unittest


class InnerLoopTestCase(unittest.TestCase):

    def setUp(self):
        return

def suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(InnerLoopTestCase)
    return suite


def run():
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())