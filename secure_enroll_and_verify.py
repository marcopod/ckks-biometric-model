import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
from photometric_stereo import PhotometricStereo
import time
from scipy.stats import wasserstein_distance

class FingerprintVerifier:
    """
    Handles fingerprint enrollment and verification using a Z-score based
    on the user's own biometric variance. This version performs all
    calculations locally.
    """
    def __init__(self):
        # --- Configuration ---
        self.IMG_SIZE = (90, 90)
        self.NUM_ENROLL_SAMPLES = 5
        self.SHARPNESS_THRESHOLD = 100.0
        self.FINGER_PRESENCE_THRESHOLD = 2.0
        self.MATCH_Z_SCORE_THRESHOLD = 1.5
        self.CAPTURE_DURATION_S = 0.5 # seconds

        # --- System Components ---
        # Disable all intermediate OpenCV windows coming from the photometric
        # pipeline – we only want to display the final match visualisation
        # rendered with Matplotlib.
        self.photo_stereo = PhotometricStereo(display=False)
        self.pca_model = None
        self.is_initialized = False

        # --- Enrollment Data ---
        self.enrolled_feature_mean = None

        self._initialize()

    def _initialize(self):
        """Initializes necessary components like the PCA model."""
        print("Initializing components...")
        
        if not self.photo_stereo.connect_sensor():
            print("Failed to connect to DIGIT sensor.")
            return

        try:
            self.pca_model = joblib.load('pca_model.joblib')
            print(f"PCA model loaded from 'pca_model.joblib'.")
        except FileNotFoundError:
            print("Error: PCA model 'pca_model.joblib' not found.")
            return
        except Exception as e:
            print(f"An error occurred while loading the PCA model: {e}")
            return
        
        self.is_initialized = True
        print("Initialization complete.")

    def extract_features(self, image):
        """Extracts a feature vector from a single image using the PCA model."""
        if self.pca_model is None:
            return None
        
        img_resized = cv2.resize(image, self.IMG_SIZE)
        # Ensure image is grayscale
        if len(img_resized.shape) > 2 and img_resized.shape[2] > 1:
             img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img_resized

        img_vector = img_gray.flatten().reshape(1, -1)
        features = self.pca_model.transform(img_vector)[0]
        return features

    def _capture_and_process_fingerprint(self):
        """
        Runs the full capture and processing pipeline to get a single,
        high-quality fingerprint image, then extracts its features.
        Returns a feature vector or None if failed.
        """
        print("Capturing reference (no touch)...")
        if not self.photo_stereo.capture_reference(self.CAPTURE_DURATION_S):
            return None
        
        print("\nNOW, place your finger on the sensor. Hold for 2 seconds.")
        time.sleep(2) # Give user time to place finger

        if not self.photo_stereo.capture_measurement(self.CAPTURE_DURATION_S):
            return None
        
        if not self.photo_stereo.process_captures():
            return None

        # 1. Check for finger presence
        if self.photo_stereo.last_diff_std_dev < self.FINGER_PRESENCE_THRESHOLD:
            print(f"Rejected: No finger detected on the sensor (Std Dev: {self.photo_stereo.last_diff_std_dev:.2f})")
            return None
            
        if not self.photo_stereo.calculate_normals_and_albedo():
            return None
        
        albedo_image = self.photo_stereo.albedo_map_enhanced
        
        # This check is crucial
        if albedo_image is None:
            print("Rejected: Albedo image could not be generated.")
            return None

        # Convert to uint8 for sharpness calculation
        albedo_image_uint8 = albedo_image.astype(np.uint8)
        sharpness = cv2.Laplacian(albedo_image_uint8, cv2.CV_64F).var()

        # 2. Check for image quality
        if sharpness < self.SHARPNESS_THRESHOLD:
            print(f"Rejected: Image too blurry (Sharpness: {sharpness:.2f})")
            return None
            
        return self.extract_features(albedo_image)


    def _calculate_enrollment_stats(self, template_vectors):
        if not template_vectors or len(template_vectors) < 2:
            print("Not enough samples to calculate robust statistics.")
            return False

        templates = np.array(template_vectors)
        self.enrolled_feature_mean = np.mean(templates, axis=0)
        
        # Calculate the EMD for each enrollment sample against the mean
        internal_distances = [
            wasserstein_distance(self.enrolled_feature_mean, vec) for vec in templates
        ]
        
        # Calculate the dynamic threshold based on this distribution
        mean_dist = np.mean(internal_distances)
        std_dist = np.std(internal_distances)
        
        # Set a robust threshold, e.g., mean + 2 standard deviations
        # This means a new print must be at least as consistent as the enrollment prints
        self.match_threshold = mean_dist + 2.0 * std_dist
        
        print("\n--- Enrollment Complete ---")
        print("User's unique feature profile has been calculated.")
        print(f"Internal EMD Stats: Mean={mean_dist:.4f}, Std={std_dist:.4f}")
        print(f"Dynamic Match Threshold (EMD) set to: {self.match_threshold:.4f}")

        return True


    def enroll(self):
        print("\n--- Fingerprint Enrollment ---")
        print(f"We will capture {self.NUM_ENROLL_SAMPLES} high-quality images.")
        
        template_vectors = []
        for i in range(self.NUM_ENROLL_SAMPLES):
            print(f"\nCapture {i+1} of {self.NUM_ENROLL_SAMPLES}:")
            input("Press Enter when ready to capture...")
            vector = self._capture_and_process_fingerprint()
            
            if vector is not None:
                template_vectors.append(vector)
                print("Capture successful.")
            else:
                print("Capture failed or was rejected. Enrollment cannot continue.")
                cv2.destroyAllWindows()
                return False
        
        cv2.destroyAllWindows()
        return self._calculate_enrollment_stats(template_vectors)


    def verify(self):
        if self.enrolled_feature_mean is None:
            print("Error: Enrollment has not been completed. Cannot verify.")
            return False

        print("\n--- Fingerprint Verification ---")
        print("Press Enter to capture fingerprint for verification...")
        input()
        
        probe_vector = self._capture_and_process_fingerprint()

        if probe_vector is None:
            print("Verification capture failed or was rejected. Please try again.")
            cv2.destroyAllWindows()
            return True

        # Use Earth Mover's Distance for a more robust comparison
        distance = wasserstein_distance(self.enrolled_feature_mean, probe_vector)

        print("\n--- Verification Result ---")
        print(f"Earth Mover's Distance: {distance:.4f} (Lower is Better)")
        print(f"Threshold (Score must be < {self.match_threshold:.4f}):")

        if distance < self.match_threshold:
            print("Result: MATCH")
            self._visualize_match(probe_vector)
        else:
            print("Result: NO MATCH")
            cv2.destroyAllWindows()

        return True


    def _visualize_match(self, probe_vector):
        if self.enrolled_feature_mean is None:
            return

        enrolled_mean = np.array(self.enrolled_feature_mean)
        probe = np.array(probe_vector)
        num_features = len(enrolled_mean)
        feature_indices = np.arange(num_features)

        plt.figure(figsize=(15, 6))
        plt.suptitle('Fingerprint Feature Comparison', fontsize=16)

        y_min = np.min([enrolled_mean, probe])
        y_max = np.max([enrolled_mean, probe])

        plt.subplot(1, 2, 1)
        plt.bar(feature_indices, enrolled_mean, color='blue', alpha=0.7)
        plt.title('Enrolled Fingerprint Profile (Mean Features)')
        plt.xlabel('Feature Index')
        plt.ylabel('Feature Value')
        plt.grid(axis='y', linestyle='--')
        plt.ylim(y_min, y_max)

        plt.subplot(1, 2, 2)
        plt.bar(feature_indices, probe, color='green', alpha=0.7)
        plt.title('Probe Fingerprint Features')
        plt.xlabel('Feature Index')
        plt.tick_params(axis='y', labelleft=False)
        plt.grid(axis='y', linestyle='--')
        plt.ylim(y_min, y_max)

        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        plt.show()


    def run(self):
        if not self.is_initialized:
            print("Cannot run verification due to initialization failure.")
            return

        try:
            if not self.enroll():
                print("\nEnrollment failed. Exiting application.")
                return

            while True:
                if not self.verify():
                    break

                another = input("\nVerify another fingerprint? (y/n): ").lower()
                if another != 'y':
                    break
        
        except KeyboardInterrupt:
            print("\nApplication interrupted by user.")
        finally:
            # Final, correct cleanup sequence
            if self.photo_stereo and self.photo_stereo.digit_sensor:
                self.photo_stereo.digit_sensor.disconnect()
            cv2.destroyAllWindows()
            print("DIGIT disconnected")
            print("Application finished.")


if __name__ == "__main__":
    verifier = FingerprintVerifier()
    verifier.run() 