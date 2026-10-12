import ast
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
ast.parse(open(os.path.join(here, "wide_steppers.py")).read())
import replicate_rsk as R

print("import ok:", R.stepper_data.__name__, R.train_batch.__name__, R.net_out.__name__)
