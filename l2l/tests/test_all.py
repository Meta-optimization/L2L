import unittest

import l2l.tests.test_ga_optimizer as test_ga_optimizer
import l2l.tests.test_sa_optimizer as test_sa_optimizer
import l2l.tests.test_gd_optimizer as test_gd_optimizer
import l2l.tests.test_gs_optimizer as test_gs_optimizer
import l2l.tests.test_pt_optimizer as test_pt_optimizer
import l2l.tests.test_face_optimizer as test_face_optimizer
import l2l.tests.test_es_optimizer as test_es_optimizer
import l2l.tests.test_setup as test_setup


suite  = unittest.TestSuite()
loader = unittest.TestLoader()

suite.addTests(test_setup.suite())
suite.addTests(test_es_optimizer.suite())
suite.addTests(test_sa_optimizer.suite())
suite.addTests(test_gd_optimizer.suite())
suite.addTests(test_ga_optimizer.suite())
suite.addTests(test_gs_optimizer.suite())
suite.addTests(test_face_optimizer.suite())
suite.addTests(test_pt_optimizer.suite())
suite.addTests(test_es_optimizer.suite())

runner = unittest.TextTestRunner(verbosity=3)
result = runner.run(suite)