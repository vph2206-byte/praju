import requests
import sys
import json
from datetime import datetime

class DryBeanAPITester:
    def __init__(self, base_url="https://bean-type-detector.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []

    def log_test(self, name, success, details=""):
        """Log test result"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
        
        result = {
            "test": name,
            "success": success,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {name}")
        if details:
            print(f"    Details: {details}")

    def run_test(self, name, method, endpoint, expected_status=200, data=None):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}"
        headers = {'Content-Type': 'application/json'}
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=30)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=30)
            
            success = response.status_code == expected_status
            
            if success:
                try:
                    response_data = response.json()
                    details = f"Status: {response.status_code}, Response keys: {list(response_data.keys()) if isinstance(response_data, dict) else 'Non-dict response'}"
                except:
                    details = f"Status: {response.status_code}, Response: {response.text[:100]}..."
            else:
                details = f"Expected {expected_status}, got {response.status_code}. Response: {response.text[:200]}..."
            
            self.log_test(name, success, details)
            return success, response.json() if success and response.text else {}
            
        except Exception as e:
            self.log_test(name, False, f"Exception: {str(e)}")
            return False, {}

    def test_dataset_info(self):
        """Test GET /api/dataset-info"""
        success, response = self.run_test(
            "Dataset Info API",
            "GET", 
            "dataset-info",
            200
        )
        
        if success:
            required_fields = ['total_samples', 'total_features', 'class_distribution', 'feature_names', 'class_names']
            missing_fields = [field for field in required_fields if field not in response]
            if missing_fields:
                self.log_test("Dataset Info - Field Validation", False, f"Missing fields: {missing_fields}")
                return False
            else:
                self.log_test("Dataset Info - Field Validation", True, f"All required fields present. Samples: {response.get('total_samples')}, Features: {response.get('total_features')}")
                return True
        return False

    def test_statistics(self):
        """Test GET /api/statistics"""
        success, response = self.run_test(
            "Statistics API",
            "GET",
            "statistics", 
            200
        )
        
        if success:
            required_fields = ['feature_statistics', 'class_distribution', 'total_samples']
            missing_fields = [field for field in required_fields if field not in response]
            if missing_fields:
                self.log_test("Statistics - Field Validation", False, f"Missing fields: {missing_fields}")
                return False
            else:
                feature_count = len(response.get('feature_statistics', {}))
                self.log_test("Statistics - Field Validation", True, f"All fields present. Feature stats for {feature_count} features")
                return True
        return False

    def test_sample_data(self):
        """Test GET /api/sample-data"""
        success, response = self.run_test(
            "Sample Data API",
            "GET",
            "sample-data",
            200
        )
        
        if success:
            expected_classes = ['DERMASON', 'SIRA', 'SEKER', 'HOROZ', 'CALI', 'BARBUNYA', 'BOMBAY']
            found_classes = list(response.keys())
            missing_classes = [cls for cls in expected_classes if cls not in found_classes]
            
            if missing_classes:
                self.log_test("Sample Data - Class Validation", False, f"Missing classes: {missing_classes}")
                return False
            else:
                self.log_test("Sample Data - Class Validation", True, f"All 7 bean classes present: {found_classes}")
                return True
        return False

    def test_model_status(self):
        """Test GET /api/model-status"""
        success, response = self.run_test(
            "Model Status API",
            "GET",
            "model-status",
            200
        )
        
        if success:
            required_fields = ['trained', 'message']
            missing_fields = [field for field in required_fields if field not in response]
            if missing_fields:
                self.log_test("Model Status - Field Validation", False, f"Missing fields: {missing_fields}")
                return False
            else:
                trained_status = response.get('trained', False)
                accuracy = response.get('accuracy')
                self.log_test("Model Status - Field Validation", True, f"Trained: {trained_status}, Accuracy: {accuracy}")
                return True, response
        return False, {}

    def test_train_model(self):
        """Test POST /api/train"""
        success, response = self.run_test(
            "Train Model API",
            "POST",
            "train",
            200
        )
        
        if success:
            required_fields = ['success', 'accuracy', 'classification_report', 'message']
            missing_fields = [field for field in required_fields if field not in response]
            if missing_fields:
                self.log_test("Train Model - Field Validation", False, f"Missing fields: {missing_fields}")
                return False
            else:
                accuracy = response.get('accuracy', 0)
                success_flag = response.get('success', False)
                self.log_test("Train Model - Field Validation", True, f"Training success: {success_flag}, Accuracy: {accuracy:.3f}")
                return True
        return False

    def test_predict_bean(self):
        """Test POST /api/predict with sample data"""
        # First get sample data
        sample_success, sample_response = self.run_test(
            "Get Sample for Prediction",
            "GET",
            "sample-data",
            200
        )
        
        if not sample_success:
            return False
        
        # Use first available sample
        first_class = list(sample_response.keys())[0]
        sample_data = sample_response[first_class]
        
        success, response = self.run_test(
            "Predict Bean API",
            "POST",
            "predict",
            200,
            data=sample_data
        )
        
        if success:
            required_fields = ['predicted_class', 'confidence', 'all_probabilities']
            missing_fields = [field for field in required_fields if field not in response]
            if missing_fields:
                self.log_test("Predict Bean - Field Validation", False, f"Missing fields: {missing_fields}")
                return False
            else:
                predicted_class = response.get('predicted_class')
                confidence = response.get('confidence', 0)
                self.log_test("Predict Bean - Field Validation", True, f"Predicted: {predicted_class}, Confidence: {confidence:.3f}")
                return True
        return False

    def run_all_tests(self):
        """Run all API tests"""
        print("🧪 Starting Dry Bean Classification API Tests")
        print(f"🌐 Testing against: {self.base_url}")
        print("=" * 60)
        
        # Test basic endpoints
        self.test_dataset_info()
        self.test_statistics()
        self.test_sample_data()
        
        # Test model status
        model_ready, model_status = self.test_model_status()
        
        # If model not trained, train it
        if model_ready and not model_status.get('trained', False):
            print("\n🔄 Model not trained, attempting to train...")
            self.test_train_model()
        
        # Test prediction
        self.test_predict_bean()
        
        # Print summary
        print("\n" + "=" * 60)
        print(f"📊 Test Summary: {self.tests_passed}/{self.tests_run} tests passed")
        
        if self.tests_passed == self.tests_run:
            print("🎉 All tests passed!")
            return 0
        else:
            print("⚠️  Some tests failed!")
            return 1

def main():
    tester = DryBeanAPITester()
    return tester.run_all_tests()

if __name__ == "__main__":
    sys.exit(main())