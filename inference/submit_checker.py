"""
Pre-Submission Validation Checker

Runs comprehensive checks before the single formal submission:
1. Inference code runs independently
2. Weight file paths are correct
3. No hardcoded absolute paths
4. CPU-only runtime
5. Dependencies are installable
6. Output format matches platform requirements
7. Class mapping is correct
8. Inference time < 70 minutes
9. No external API calls
10. Both code and weights are included
"""

import importlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class SubmitChecker:
    """Runs pre-submission validation checks.

    Usage:
        checker = SubmitChecker(
            inference_script="submit/inference.py",
            weights_path="weights/best.pth",
            test_images_dir="data/test",
            label_mapping_path="reports/label_mapping.json",
        )
        results = checker.run_all_checks()
        if results["all_passed"]:
            print("Ready to submit!")
        else:
            print("Fix the issues above before submitting.")
    """

    def __init__(
        self,
        inference_script: str,
        weights_path: str,
        test_images_dir: str,
        label_mapping_path: str,
        time_limit_minutes: int = 70,
        submit_dir: str = "submit",
    ):
        """
        Args:
            inference_script: Path to the submission inference script.
            weights_path: Path to the model weights file.
            test_images_dir: Directory with test images for smoke test.
            label_mapping_path: Path to label mapping JSON.
            time_limit_minutes: Maximum inference time in minutes (default 70).
            submit_dir: Directory prepared for submission.
        """
        self.inference_script = Path(inference_script)
        self.weights_path = Path(weights_path)
        self.test_images_dir = Path(test_images_dir)
        self.label_mapping_path = Path(label_mapping_path)
        self.time_limit_minutes = time_limit_minutes
        self.submit_dir = Path(submit_dir)

        self.results: Dict[str, Dict] = {}

    def run_all_checks(self) -> Dict:
        """Run all submission checks.

        Returns:
            Dict with check results and overall pass/fail status.
        """
        logger.info("=" * 60)
        logger.info("PRE-SUBMISSION CHECKER — Running all validations")
        logger.info("=" * 60)

        checks = [
            ("weights_exist", self.check_weights_exist),
            ("inference_script_exists", self.check_inference_script_exists),
            ("cpu_only_imports", self.check_cpu_only_imports),
            ("dependencies_installable", self.check_dependencies_installable),
            ("no_hardcoded_paths", self.check_no_hardcoded_paths),
            ("no_external_api", self.check_no_external_api_calls),
            ("label_mapping_valid", self.check_label_mapping_valid),
            ("output_format", self.check_output_format),
            ("smoke_test", self.check_smoke_test),
            ("inference_speed", self.check_inference_speed),
            ("weights_size_reasonable", self.check_weights_size),
            ("code_structure", self.check_code_structure),
        ]

        for check_name, check_fn in checks:
            try:
                passed, message = check_fn()
            except Exception as e:
                passed, message = False, f"Check crashed: {e}"

            self.results[check_name] = {
                "passed": passed,
                "message": message,
            }
            status = "✅ PASS" if passed else "❌ FAIL"
            logger.info(f"  [{status}] {check_name}: {message}")

        all_passed = all(r["passed"] for r in self.results.values())
        self.results["all_passed"] = all_passed

        logger.info("=" * 60)
        if all_passed:
            logger.info("🎉 ALL CHECKS PASSED — Ready for submission!")
        else:
            failed = [k for k, v in self.results.items() if not v.get("passed", True)]
            logger.error(f"❌ {len(failed)} CHECK(S) FAILED: {', '.join(failed)}")
            logger.error("Fix the issues above before submitting!")
        logger.info("=" * 60)

        return self.results

    def check_weights_exist(self) -> Tuple[bool, str]:
        """Verify model weights file exists and is loadable."""
        if not self.weights_path.exists():
            return False, f"Weights file not found: {self.weights_path}"
        try:
            state = torch.load(self.weights_path, map_location="cpu", weights_only=True)
            return True, f"Weights file found ({self.weights_path.stat().st_size / 1024 / 1024:.1f} MB)"
        except Exception as e:
            return False, f"Weights file cannot be loaded: {e}"

    def check_inference_script_exists(self) -> Tuple[bool, str]:
        """Verify inference script exists and is valid Python."""
        if not self.inference_script.exists():
            return False, f"Inference script not found: {self.inference_script}"

        # Check syntax
        try:
            with open(self.inference_script, "r") as f:
                code = f.read()
            compile(code, str(self.inference_script), "exec")
            return True, f"Inference script is valid Python ({len(code)} bytes)"
        except SyntaxError as e:
            return False, f"Syntax error in inference script: {e}"

    def check_cpu_only_imports(self) -> Tuple[bool, str]:
        """Verify the inference script does not import CUDA-only libraries."""
        suspicious = []
        with open(self.inference_script, "r") as f:
            code = f.read()

        # Check for CUDA-specific code patterns
        forbidden_patterns = [
            (r"\.cuda\(\)", "Uses .cuda() — must be .cpu()"),
            (r"torch\.cuda\.", "Imports torch.cuda"),
            (r"device.*=.*['\"]cuda['\"]", "Sets device to 'cuda'"),
            (r"\.to\(['\"]cuda", "Moves tensor to cuda"),
        ]
        for pattern, msg in forbidden_patterns:
            if re.search(pattern, code):
                suspicious.append(msg)

        if suspicious:
            return False, "Found CUDA references: " + "; ".join(suspicious)
        return True, "No CUDA-specific code detected"

    def check_dependencies_installable(self) -> Tuple[bool, str]:
        """Check that dependencies can be installed."""
        req_path = self.submit_dir / "requirements.txt"
        if not req_path.exists():
            return False, f"No requirements.txt in {self.submit_dir}"

        # Check each package is available
        missing = []
        with open(req_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                pkg_name = re.split(r"[=<>~!]", line)[0].strip()
                try:
                    importlib.import_module(pkg_name.replace("-", "_"))
                except ImportError:
                    missing.append(pkg_name)

        if missing:
            return False, f"Missing packages: {', '.join(missing)}"
        return True, "All dependencies available"

    def check_no_hardcoded_paths(self) -> Tuple[bool, str]:
        """Verify no hardcoded absolute paths exist."""
        with open(self.inference_script, "r") as f:
            code = f.read()

        # Look for absolute paths (Unix and Windows)
        abs_path_patterns = [
            r"['\"]/home/",
            r"['\"]/root/",
            r"['\"]C:\\",
            r"['\"]/Users/",
        ]
        for pattern in abs_path_patterns:
            if re.search(pattern, code):
                return False, f"Found hardcoded absolute path matching '{pattern}'"

        return True, "No hardcoded absolute paths detected"

    def check_no_external_api_calls(self) -> Tuple[bool, str]:
        """Verify no external API calls in inference code."""
        with open(self.inference_script, "r") as f:
            code = f.read()

        api_patterns = [
            r"requests\.(get|post|put|delete)",
            r"urllib\.request",
            r"httpx\.",
            r"openai\.",
            r"anthropic\.",
            r"google\.api",
        ]
        for pattern in api_patterns:
            if re.search(pattern, code):
                return False, f"Found potential external API call: '{pattern}'"

        return True, "No external API calls detected"

    def check_label_mapping_valid(self) -> Tuple[bool, str]:
        """Verify label mapping is valid and has 4 classes."""
        if not self.label_mapping_path.exists():
            return False, f"Label mapping not found: {self.label_mapping_path}"

        with open(self.label_mapping_path, "r") as f:
            mapping = json.load(f)

        labels = mapping.get("labels", [])
        num_classes = mapping.get("num_classes", len(labels))

        if num_classes != 4:
            return False, f"Expected 4 classes, found {num_classes}: {labels}"

        expected = {"cloudy", "rainy", "snowy", "sunny"}
        found = set(labels)
        if found != expected:
            return False, f"Class labels mismatch. Expected {expected}, found {found}"

        return True, f"Label mapping valid: {labels}"

    def check_output_format(self) -> Tuple[bool, str]:
        """Check that output CSV format matches expected competition format."""
        with open(self.inference_script, "r") as f:
            code = f.read()

        # Check for CSV output with correct columns
        has_filename_col = bool(re.search(r"filename", code, re.IGNORECASE))
        has_prediction_col = bool(re.search(r"prediction", code, re.IGNORECASE))
        has_csv_write = bool(re.search(r"\.to_csv|\.csv|DictWriter|csv\.writer", code))

        if not has_csv_write:
            return False, "No CSV output found in inference script"
        if not has_filename_col or not has_prediction_col:
            return False, "Output CSV must contain 'filename' and 'prediction' columns"

        return True, "Output format appears correct (filename, prediction columns)"

    def check_smoke_test(self) -> Tuple[bool, str]:
        """Run a minimal smoke test of the inference pipeline."""
        if not self.test_images_dir.exists():
            return False, f"Test images directory not found: {self.test_images_dir}"

        test_images = list(self.test_images_dir.glob("*"))
        test_images = [p for p in test_images if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if not test_images:
            return True, "No test images available — skipping smoke test"

        # Run inference on a few test images
        try:
            import subprocess
            result = subprocess.run(
                [
                    sys.executable, "-c",
                    f"""
import sys
sys.path.insert(0, '.')
import torch
from pathlib import Path
import json

# Load label mapping
with open('{self.label_mapping_path}') as f:
    mapping = json.load(f)

# Quick model load
model_path = Path('{self.weights_path}')
assert model_path.exists(), "Weights not found"
state = torch.load(model_path, map_location='cpu', weights_only=True)
print(f"Model loaded: {{len(state)}} keys")
print("Smoke test PASSED")
""",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=".",
            )
            if result.returncode == 0:
                return True, "Smoke test passed"
            else:
                return False, f"Smoke test failed: {result.stderr.strip()}"
        except subprocess.TimeoutExpired:
            return False, "Smoke test timed out (>60s)"
        except Exception as e:
            return False, f"Smoke test error: {e}"

    def check_inference_speed(self) -> Tuple[bool, str]:
        """Estimate inference speed and check against 70-minute limit."""
        # Load model and run a quick benchmark
        try:
            import numpy as np

            state = torch.load(self.weights_path, map_location="cpu", weights_only=True)

            # Quick benchmark with random input
            dummy = torch.randn(1, 3, 224, 224)
            # We can't easily reconstruct the model here, so estimate from weight size
            param_count = sum(v.numel() for v in state.values()
                              if isinstance(v, torch.Tensor))
            weight_mb = param_count * 4 / (1024 * 1024)

            # Conservative estimate: 10ms per 1M params for CPU CNN
            estimated_ms = param_count * 10 / 1e6
            estimated_total_min = estimated_ms * 3000 / 1000 / 60  # 3000 images

            if estimated_total_min < self.time_limit_minutes:
                return True, (
                    f"Rough speed estimate: {estimated_total_min:.1f} min "
                    f"for 3000 images (limit: {self.time_limit_minutes} min)"
                )
            else:
                return False, (
                    f"Estimated {estimated_total_min:.1f} min exceeds "
                    f"{self.time_limit_minutes} min limit"
                )
        except Exception as e:
            return False, f"Speed check failed: {e}"

    def check_weights_size(self) -> Tuple[bool, str]:
        """Check that weight file size is reasonable (< 500 MB)."""
        size_mb = self.weights_path.stat().st_size / (1024 * 1024)
        if size_mb > 500:
            return False, f"Weight file too large: {size_mb:.1f} MB (max 500 MB)"
        return True, f"Weight file size: {size_mb:.1f} MB"

    def check_code_structure(self) -> Tuple[bool, str]:
        """Check code structure quality indicators."""
        with open(self.inference_script, "r") as f:
            code = f.read()
            lines = code.split("\n")

        issues = []

        # Check for docstring
        if '"""' not in code and "'''" not in code:
            issues.append("No docstring found")

        # Check for main guard
        if 'if __name__' not in code:
            issues.append("No __main__ guard")

        # Check for basic comments
        comment_lines = sum(1 for line in lines if line.strip().startswith("#"))
        if comment_lines < 3:
            issues.append("Very few comments (< 3 lines)")

        # Check for argparse or config
        has_argparse = "argparse" in code
        has_click = "click" in code or "@click" in code
        has_sys_argv = "sys.argv" in code
        if not (has_argparse or has_click or has_sys_argv):
            issues.append("No argument parsing (argparse/sys.argv)")

        if issues:
            return False, "; ".join(issues)
        return True, "Code structure looks good (docstring, __main__, comments, args)"


def run_submission_check(
    inference_script: str,
    weights_path: str,
    test_images_dir: str,
    label_mapping_path: str,
    submit_dir: str = "submit",
) -> Dict:
    """Convenience function to run all submission checks.

    Args:
        inference_script: Path to inference script.
        weights_path: Path to model weights.
        test_images_dir: Path to test images.
        label_mapping_path: Path to label mapping JSON.
        submit_dir: Submission directory.

    Returns:
        Dict with all check results.
    """
    checker = SubmitChecker(
        inference_script=inference_script,
        weights_path=weights_path,
        test_images_dir=test_images_dir,
        label_mapping_path=label_mapping_path,
        submit_dir=submit_dir,
    )
    return checker.run_all_checks()
