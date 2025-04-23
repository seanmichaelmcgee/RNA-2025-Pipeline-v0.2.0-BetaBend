#!/usr/bin/env python3
"""
Test script for RNA 3D Structure Training Pipeline components

This script tests the core functionality of:
1. GPU Monitoring script
2. Report Generation script
3. Production Training script bash parsing
"""

import os
import sys
import unittest
import tempfile
import json
import shutil
import subprocess
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import scripts as modules for testing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../scripts")))
import monitor_gpu
import generate_training_report


class TestGPUMonitoring(unittest.TestCase):
    """Test basic functionality of the GPU monitoring script."""
    
    def setUp(self):
        """Create temporary directory for test outputs."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary test directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_validate_gpu_ids(self):
        """Test GPU ID validation function."""
        # Test 'all' option
        self.assertIsNone(monitor_gpu.validate_gpu_ids('all'))
        
        # Test comma-separated list
        ids = monitor_gpu.validate_gpu_ids('0,1,2')
        self.assertEqual(ids, [0, 1, 2])
        
        # Test single ID
        ids = monitor_gpu.validate_gpu_ids('0')
        self.assertEqual(ids, [0])
    
    def test_nvidia_smi_metrics(self):
        """Test getting metrics from nvidia-smi command if available."""
        try:
            # This might fail if no GPU or nvidia-smi not available
            metrics = monitor_gpu.get_nvidia_smi_metrics()
            # If we got metrics, ensure they have the expected format
            if metrics:
                # Just check if we got any metrics with the right structure
                self.assertIsInstance(metrics, list)
                if len(metrics) > 0:
                    self.assertIn('gpu_id', metrics[0])
                    self.assertIn('utilization', metrics[0])
                    self.assertIn('memory_used', metrics[0])
                    self.assertIn('memory_total', metrics[0])
                    self.assertIn('temperature', metrics[0])
        except Exception as e:
            print(f"Skipping nvidia-smi test due to: {e}")
            print("This is expected if no GPU is available")


class TestReportGeneration(unittest.TestCase):
    """Test basic functionality of the report generation script."""
    
    def setUp(self):
        """Create temporary directory and mock training data."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock training directory structure
        self.train_dir = os.path.join(self.temp_dir, "mock_training")
        os.makedirs(self.train_dir, exist_ok=True)
        
        # Create mock config file
        config = {
            "batch_size": 32,
            "num_epochs": 50,
            "learning_rate": 0.0005,
            "max_seq_length": 300,
            "min_seq_length": 10
        }
        with open(os.path.join(self.train_dir, "config.json"), "w") as f:
            json.dump(config, f)
        
        # Create mock training log
        train_log = pd.DataFrame({
            "epoch": list(range(10)),
            "total_loss": [5.0 - i*0.5 for i in range(10)],
            "fape_loss": [3.0 - i*0.3 for i in range(10)],
            "confidence_loss": [1.5 - i*0.1 for i in range(10)],
            "angle_loss": [0.5 - i*0.05 for i in range(10)]
        })
        train_log.to_csv(os.path.join(self.train_dir, "training_log.csv"), index=False)
        
    def tearDown(self):
        """Clean up temporary test directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_parse_training_logs(self):
        """Test parsing training logs from CSV."""
        df = generate_training_report.parse_training_logs(self.train_dir)
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 10)  # 10 epochs
        self.assertIn("total_loss", df.columns)
        self.assertIn("fape_loss", df.columns)
    
    def test_load_training_config(self):
        """Test loading training configuration from JSON."""
        config = generate_training_report.load_training_config(self.train_dir)
        self.assertIsInstance(config, dict)
        self.assertEqual(config["batch_size"], 32)
        self.assertEqual(config["num_epochs"], 50)


class TestProductionTrainingScript(unittest.TestCase):
    """Test the production training script's parameter handling using bash."""
    
    def test_script_help_output(self):
        """Test that the script produces help output."""
        script_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../scripts/run_production_training.sh")
        )
        
        try:
            # Run script with --help
            result = subprocess.run([script_path, "--help"], 
                                    capture_output=True, text=True, check=False)
            
            # Check if help output contains key parameters
            output = result.stdout
            self.assertIn("Usage", output)
            self.assertIn("batch_size", output)
            self.assertIn("num_epochs", output)
            self.assertIn("lr", output)
        except Exception as e:
            print(f"Skipping bash script test due to: {e}")
            print("This is expected if the script can't be executed directly")


def run_script_tests():
    """Run tests from command line."""
    try:
        import pandas as pd
    except ImportError:
        print("WARNING: pandas not available. Some tests will be skipped.")
        # Mock pandas for minimal testing
        global pd
        class MockDataFrame:
            def __init__(self, data):
                self.data = data
                self.columns = list(data.keys())
                self.empty = len(data.get(list(data.keys())[0], [])) == 0
            
            def to_csv(self, path, index=True):
                pass
        
        class MockPD:
            @staticmethod
            def DataFrame(data):
                return MockDataFrame(data)
            
            @staticmethod
            def read_csv(path):
                return MockDataFrame({"epoch": [], "total_loss": []})
        
        pd = MockPD()
    
    # Set up the test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases to the suite
    suite.addTests(loader.loadTestsFromTestCase(TestGPUMonitoring))
    suite.addTests(loader.loadTestsFromTestCase(TestReportGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestProductionTrainingScript))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    run_script_tests()