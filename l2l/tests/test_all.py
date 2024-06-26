import os
import unittest
import shutil


from l2l.tests import test_ce_optimizer
from l2l.tests import test_ga_optimizer
from l2l.tests import test_sa_optimizer
from l2l.tests import test_gd_optimizer
from l2l.tests import test_gs_optimizer
from l2l.tests import test_innerloop
from l2l.tests import test_outerloop
from l2l.tests import test_setup
from l2l.tests import test_checkpoint


def test_suite():

    suite = unittest.TestSuite()
    suite.addTest(test_setup.suite())
    suite.addTest(test_outerloop.suite())
    suite.addTest(test_innerloop.suite())
    suite.addTest(test_ce_optimizer.suite())
    suite.addTest(test_sa_optimizer.suite())
    suite.addTest(test_gd_optimizer.suite())
    suite.addTest(test_ga_optimizer.suite())
    suite.addTest(test_gs_optimizer.suite())
    suite.addTest(test_checkpoint.suite())

    return suite


if __name__ == "__main__":

    runner = unittest.TextTestRunner(verbosity=2)
    home_path =  os.environ.get("HOME")
    root_dir_path = os.path.join(home_path, 'results')
    runner.run(test_suite())
    if os.path.exists(root_dir_path):
        print(f'removing {root_dir_path}')
        shutil.rmtree(root_dir_path)
