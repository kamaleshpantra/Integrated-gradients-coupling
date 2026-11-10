import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

print("Running test 1: Special tokens and erasure operators...")
from project.tests.test_special_tokens import test_special_tokens_excluded_from_content, test_erasure_operators_shape_and_tokens
test_special_tokens_excluded_from_content()
test_erasure_operators_shape_and_tokens()
print("[PASS] Special tokens test passed.\n")

print("Running test 2: Determinism under fixed seeds...")
from project.tests.test_determinism import test_determinism_under_fixed_seeds
test_determinism_under_fixed_seeds()
print("[PASS] Determinism test passed.\n")

print("Running test 3: Completeness convergence...")
from project.tests.test_completeness import test_completeness_convergence
test_completeness_convergence()
print("[PASS] Completeness convergence test passed.\n")

print("Running test 4: Real IG vs Random attribution control...")
from project.tests.test_random_control import test_ig_beats_random_control
test_ig_beats_random_control()
print("[PASS] Random control test passed.\n")

print("==========================================")
print("ALL UNIT TESTS PASSED SUCCESSFULLY! (4/4)")
print("==========================================")
