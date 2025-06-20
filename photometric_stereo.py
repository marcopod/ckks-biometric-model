import cv2
import numpy as np
import time
from digit import WindowsDigit
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift

class PhotometricStereo:
    def __init__(self, camera_index: int = 1, display: bool = True):
        """Photometric stereo pipeline for DIGIT sensor.

        Parameters
        ----------
        camera_index : int, optional
            Index of the camera to open. Defaults to ``1`` because the DIGIT
            sensor often enumerates as the second webcam on Windows machines.
        display : bool, optional
            If ``True`` (default) the intermediate visualisations such as
            albedo, normals and height-maps are shown in real-time via
            ``cv2.imshow``.  When the class is used as part of a higher-level
            pipeline (e.g. enrol/verify scripts) these intermediate windows
            are often undesirable.  Setting *display=False* cleanly suppresses
            all OpenCV GUI calls while keeping the numerical processing intact.
        """

        # Configuration
        self.display = display

        # ---- Sensor Wrapper ----
        self.digit_sensor = WindowsDigit(camera_index=camera_index)

        # When windows are not desired we monkey-patch the two OpenCV GUI
        # helpers that are used throughout this module so that they become
        # cheap no-ops.  This centralises the switch and avoids adding dozens
        # of conditional statements around every ``cv2.imshow`` invocation.
        if not self.display:
            cv2.imshow = lambda *args, **kwargs: None  # type: ignore
            cv2.namedWindow = lambda *args, **kwargs: None  # type: ignore

        self.reference_avg = None
        self.is_capturing = False
        self.captured_frames = []
        self.processed_image = None
        self.last_diff_std_dev = 0.0 # To check for finger presence
        
        # self.normals_map is the uint8 BGR (nx,ny,nz) scaled map. 
        # It's updated by 'n' (standard normals) or 'B' (bilaterally filtered normals).
        self.normals_map = None 
        
        self.height_map_l2 = None # uint8 - baseline L2, derived from current self.normals_map
        self.height_map_l2_filtered_normals = None # uint8 - L2 derived from newly, internally bilaterally filtered normals ('E' key)
        
        self.albedo_map = None    # uint8
        self.albedo_map_enhanced = None # uint8, CLAHE enhanced
        self.albedo_map_binary = None # uint8, binary thresholded
        
        # Roughness stats for self.normals_map (current active normals)
        self.Ra_normals = None
        self.Rq_normals = None
        
        # Roughness stats for self.height_map_l2
        self.Ra_height_l2 = None 
        self.Rq_height_l2 = None 
        
        # Roughness stats for self.height_map_l2_filtered_normals
        self.Ra_height_l2_filtered_normals = None 
        self.Rq_height_l2_filtered_normals = None

    def connect_sensor(self):
        """Connect to the DIGIT sensor"""
        return self.digit_sensor.connect()
        
    def capture_reference(self, duration_seconds: float = 1.0):
        """Capture and average frames for the reference (no touch)"""
        print(f"Capturing reference for {duration_seconds} seconds...")
        reference_frames = []
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            frame = self.digit_sensor.get_frame()
            if frame is not None:
                reference_frames.append(frame.astype(np.float32))
                display_frame = frame.copy()
                cv2.putText(display_frame, "Capturing Reference...", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Capture", display_frame)
                cv2.waitKey(1)
        
        if reference_frames:
            self.reference_avg = np.mean(reference_frames, axis=0).astype(np.float32)
            print("Reference capture complete!")
            cv2.imshow("Reference Preview", self.reference_avg.astype(np.uint8))
            return True
        else:
            print("No frames captured for reference.")
            return False
            
    def capture_measurement(self, duration_seconds: float = 1.0):
        """Capture and average frames for measurement (with touch)"""
        print(f"Capturing measurement for {duration_seconds} seconds...")
        self.captured_frames = []
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            frame = self.digit_sensor.get_frame()
            if frame is not None:
                self.captured_frames.append(frame.astype(np.float32))
                display_frame = frame.copy()
                cv2.putText(display_frame, "Capturing Measurement...", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Capture", display_frame)
                cv2.waitKey(1)
        
        if self.captured_frames:
            print("Measurement capture complete!")
            avg_measurement = np.mean(self.captured_frames, axis=0).astype(np.uint8)
            cv2.imshow("Measurement Preview", avg_measurement)
            return True
        else:
            print("No frames captured for measurement.")
            return False
    
    def process_captures(self):
        """Process the captured frames to get normalized difference"""
        if self.reference_avg is None or not self.captured_frames:
            print("Reference or measurement data missing.")
            return False
            
        measurement_avg = np.mean(self.captured_frames, axis=0).astype(np.float32)
        diff = measurement_avg - self.reference_avg
        
        # Calculate standard deviation to check for finger presence
        self.last_diff_std_dev = np.std(diff)
        print(f"Difference image standard deviation: {self.last_diff_std_dev:.4f}")

        global_min, global_max = np.min(diff), np.max(diff)
        if global_max > global_min:
            self.processed_image = (diff - global_min) / (global_max - global_min)
        else:
            self.processed_image = np.zeros_like(diff)
        print(f"Processing complete. Diff image range: [{global_min:.2f}, {global_max:.2f}]")
        cv2.imshow("Processed Image", (self.processed_image * 255).astype(np.uint8))
        return True
        
    def calculate_normals_and_albedo(self, apply_bilateral_filter=False):
        """
        Calculate surface normals and albedo from self.processed_image.
        Updates self.normals_map (to standard or filtered normals) and self.albedo_map.
        Also updates self.Ra_normals and self.Rq_normals based on the new self.normals_map.
        """
        if self.processed_image is None:
            print("No processed image. Please process captures first.")
            return False
        
        processed_img_for_normals = self.processed_image 
        b, g, r = cv2.split(processed_img_for_normals)

        # Vectorized implementation for performance
        h, w = r.shape
        intensity = np.stack((r, g, b), axis=-1) # Shape: (h, w, 3)
        intensity_flat = intensity.reshape(-1, 3) # Shape: (h*w, 3)

        # These light directions are fixed for the DIGIT sensor configuration
        phi = np.array([0, 2*np.pi/3, 4*np.pi/3])
        light_dirs = np.array([
            [np.sin(phi[0]), np.cos(phi[0]), 1],
            [np.sin(phi[1]), np.cos(phi[1]), 1],
            [np.sin(phi[2]), np.cos(phi[2]), 1],
        ])
        # Pre-calculate the pseudo-inverse of the light direction matrix
        L_inv = np.linalg.pinv(light_dirs) # Shape: (3, 3)

        # Solve for g = rho * n for all pixels at once
        # g = (L_inv @ I^T)^T
        g_flat = (L_inv @ intensity_flat.T).T # Shape: (h*w, 3)

        # Calculate albedo (rho) and normals (n) from g
        rho_flat = np.linalg.norm(g_flat, axis=1) # Shape: (h*w,)
        
        # Avoid division by zero for black pixels
        normals_flat = np.zeros_like(g_flat)
        valid_pixels = rho_flat > 0
        normals_flat[valid_pixels] = g_flat[valid_pixels] / rho_flat[valid_pixels, np.newaxis]

        # Reshape back to image dimensions
        _normals_float_calc = normals_flat.reshape(h, w, 3)
        _albedo_raw_calc = rho_flat.reshape(h, w)
        
        current_normals_float_to_use = _normals_float_calc

        # Update self.normals_map (the active normal map for other calculations like 'h')
        temp_normals_map_uint8 = np.zeros_like(current_normals_float_to_use, dtype=np.uint8)
        temp_normals_map_uint8[..., 0] = (current_normals_float_to_use[..., 0] + 1) * 127.5
        temp_normals_map_uint8[..., 1] = (current_normals_float_to_use[..., 1] + 1) * 127.5
        temp_normals_map_uint8[..., 2] = (current_normals_float_to_use[..., 2] + 1) * 127.5
        self.normals_map = temp_normals_map_uint8

        # Albedo is based on the initial calculation from _albedo_raw_calc (before normal filtering)
        min_a, max_a = np.min(_albedo_raw_calc), np.max(_albedo_raw_calc) 
        if max_a > min_a: self.albedo_map = ((_albedo_raw_calc - min_a) / (max_a - min_a) * 255).astype(np.uint8)
        else: self.albedo_map = np.zeros_like(_albedo_raw_calc, dtype=np.uint8)
        
        # Create a high-contrast version using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.albedo_map_enhanced = clahe.apply(self.albedo_map)

        # Create a binary version using adaptive thresholding
        self.albedo_map_binary = cv2.adaptiveThreshold(self.albedo_map, 255,
                                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                       cv2.THRESH_BINARY_INV, 11, 2)

        filter_status = "Standard"
        print(f"Normals ({filter_status}) and Albedo calculation complete.")
        cv2.imshow(f"Normals Map ({filter_status})", self.normals_map)
        if self.albedo_map is not None:
            cv2.imshow("Albedo Map", self.albedo_map)
            cv2.imshow("Albedo Map (Enhanced)", self.albedo_map_enhanced)
            cv2.imshow("Albedo Map (Binary)", self.albedo_map_binary)
        
        # Skipping roughness stats for performance
        # self.calculate_roughness_stats_from_normals()
        return True

    def _get_gradients_from_normals_map(self, normals_map_uint8_to_use):
        """Helper to get gradients (gx, gy) from a provided uint8 normals map."""
        if normals_map_uint8_to_use is None: 
            print("Error: No normals map provided to _get_gradients_from_normals_map.")
            return None, None
        _normals_float = (normals_map_uint8_to_use.astype(np.float32) / 127.5) - 1.0
        
        # Ensure _normals_float has 3 channels
        if _normals_float.ndim != 3 or _normals_float.shape[2] != 3:
            print(f"Error: Normals float data is not 3-channel. Shape: {_normals_float.shape}")
            return None, None
            
        nx, ny, nz = _normals_float[...,0], _normals_float[...,1], _normals_float[...,2]
        
        # Avoid division by zero or very small nz values
        nz_safe = nz.copy()
        small_nz_mask = np.abs(nz_safe) < 1e-6
        nz_safe[small_nz_mask] = np.sign(nz_safe[small_nz_mask] + 1e-10) * 1e-6 # Add signed epsilon
        nz_safe[nz_safe == 0] = 1e-6 # Ensure no zeros remain after epsilon addition
        
        gx, gy = -nx / nz_safe, -ny / nz_safe
        return gx, gy

    def _integrate_gradients_l2_fourier(self, gx, gy, return_raw=False):
        """Integrates gradients gx, gy using Fourier method. Returns uint8 map or raw float map."""
        if gx is None or gy is None :
            print("Error: Gradients not available for L2 integration.")
            return None

        h, w = gx.shape
        wx = ifftshift(np.linspace(-np.pi, np.pi, w, endpoint=False))
        wy = ifftshift(np.linspace(-np.pi, np.pi, h, endpoint=False))
        WX, WY = np.meshgrid(wx, wy)
        
        DZ_DX_fft, DZ_DY_fft = fft2(gx), fft2(gy)
        
        denominator = WX**2 + WY**2
        # Add a small epsilon to the denominator to avoid division by zero at the DC component (0,0)
        # and for numerical stability.
        denominator_safe = denominator + 1e-9 
        
        Z_fft = (1j * WX * DZ_DX_fft + 1j * WY * DZ_DY_fft) / denominator_safe
        Z_fft[0, 0] = 0 # Set DC component to zero (mean height is zero)
        
        height_map_raw = np.real(ifft2(Z_fft))
        
        if return_raw: 
            return height_map_raw
        
        # Normalize to uint8 for display
        min_h, max_h = np.min(height_map_raw), np.max(height_map_raw)
        if max_h > min_h: 
            height_map_uint8 = ((height_map_raw - min_h) / (max_h - min_h) * 255).astype(np.uint8)
            return height_map_uint8
        else: # Handle flat height map case
            return np.zeros_like(height_map_raw, dtype=np.uint8)

    def calculate_height_map_l2_baseline(self, return_raw=False):
        """
        Calculates L2 height map from the current self.normals_map.
        Stores result in self.height_map_l2 and updates its roughness stats.
        """
        print("Calculating L2 Height Map (Baseline from current self.normals_map)...")
        if self.normals_map is None:
            print("Current self.normals_map is not available. Calculate normals first ('n' or 'B').")
            self.height_map_l2, self.Ra_height_l2, self.Rq_height_l2 = None, None, None
            return False if not return_raw else None

        gx, gy = self._get_gradients_from_normals_map(self.normals_map)
        if gx is None: 
            self.height_map_l2 = None
            self.Ra_height_l2, self.Rq_height_l2 = None, None
            return False if not return_raw else None
        
        # Get uint8 or raw float map based on return_raw
        result_map = self._integrate_gradients_l2_fourier(gx, gy, return_raw=return_raw)

        if result_map is None:
            print("L2 Height Map (Baseline) calculation failed during integration.")
            self.height_map_l2, self.Ra_height_l2, self.Rq_height_l2 = None, None, None
            return False if not return_raw else None

        if not return_raw: # Store and display uint8 map
            self.height_map_l2 = result_map
            cv2.imshow("L2 Height Map (Baseline - from current normals)", self.height_map_l2)
            print("L2 Height Map (Baseline) calculation complete.")
            self.calculate_roughness_stats_from_heightmap(height_map_type='l2_baseline')
            return self.height_map_l2 is not None
        else: # Just return the raw float map
            return result_map


    def calculate_height_map_l2_from_filtered_normals(self, return_raw=False):
        """
        Generates normals with bilateral filter internally, then L2 height map.
        Stores result in self.height_map_l2_filtered_normals and updates its roughness stats.
        This method does NOT change self.normals_map.
        """
        print("Starting: L2 Height Map from new Bilaterally Filtered Normals...")
        if self.processed_image is None:
            print("Processed image needed for L2 from filtered normals. Please process captures first.")
            self.height_map_l2_filtered_normals, self.Ra_height_l2_filtered_normals, self.Rq_height_l2_filtered_normals = None, None, None
            return False if not return_raw else None

        # --- Internal Normal Calculation & Filtering ---
        print("  Step 1: Internally calculating and bilaterally filtering normals...")
        processed_img = self.processed_image
        b, g, r = cv2.split(processed_img)
        phi = np.array([0, 2*np.pi/3, 4*np.pi/3])
        light_dirs = np.array([[np.sin(phi[0]),np.cos(phi[0]),1],[np.sin(phi[1]),np.cos(phi[1]),1],[np.sin(phi[2]),np.cos(phi[2]),1]])
        light_dirs = light_dirs / np.linalg.norm(light_dirs, axis=1, keepdims=True)
        intensity = np.stack((r, g, b), axis=2)
        h, w = r.shape
        _temp_normals_float = np.zeros((h, w, 3), dtype=np.float32)
        for y_idx in range(h):
            for x_idx in range(w):
                try:
                    g_s = np.linalg.lstsq(light_dirs, intensity[y_idx,x_idx], rcond=None)[0]
                    rho = np.linalg.norm(g_s)
                    if rho > 0: _temp_normals_float[y_idx,x_idx] = g_s/rho
                    else: _temp_normals_float[y_idx,x_idx] = [0,0,1]
                except np.linalg.LinAlgError: _temp_normals_float[y_idx,x_idx] = [0,0,1]
        
        _temp_normals_float_filtered = np.zeros_like(_temp_normals_float)
        for i in range(3):
            _temp_normals_float_filtered[...,i] = cv2.bilateralFilter(_temp_normals_float[...,i].astype(np.float32), d=5, sigmaColor=0.1, sigmaSpace=5)
        
        norms_m = np.linalg.norm(_temp_normals_float_filtered, axis=2, keepdims=True)
        zero_m = (norms_m < 1e-9)
        norms_m[zero_m] = 1e-9
        _final_filtered_normals_float = _temp_normals_float_filtered / norms_m
        # _final_filtered_normals_float[np.squeeze(zero_m), :] = [0,0,1] # Optional explicit set for zero norms

        temp_filtered_normals_uint8_map = np.zeros_like(_final_filtered_normals_float, dtype=np.uint8)
        temp_filtered_normals_uint8_map[...,0] = (_final_filtered_normals_float[...,0]+1)*127.5
        temp_filtered_normals_uint8_map[...,1] = (_final_filtered_normals_float[...,1]+1)*127.5
        temp_filtered_normals_uint8_map[...,2] = (_final_filtered_normals_float[...,2]+1)*127.5
        cv2.imshow("Normals (Temp Filtered for 'E' key)", temp_filtered_normals_uint8_map)
        print("  Internal filtered normal calculation complete.")
        # --- End Internal Normal Calculation & Filtering ---

        print("  Step 2: Integrating gradients from these bilaterally filtered normals...")
        gx_f, gy_f = self._get_gradients_from_normals_map(temp_filtered_normals_uint8_map)
        
        if gx_f is None:
            print("Failed to get gradients from internally filtered normals.")
            self.height_map_l2_filtered_normals, self.Ra_height_l2_filtered_normals, self.Rq_height_l2_filtered_normals = None, None, None
            return False if not return_raw else None

        # Get uint8 or raw float map based on return_raw
        result_map_filtered = self._integrate_gradients_l2_fourier(gx_f, gy_f, return_raw=return_raw)

        if result_map_filtered is None:
            print("L2 Height Map from Filtered Normals calculation failed during integration.")
            self.height_map_l2_filtered_normals, self.Ra_height_l2_filtered_normals, self.Rq_height_l2_filtered_normals = None, None, None
            return False if not return_raw else None

        if not return_raw: # Store and display uint8 map
            self.height_map_l2_filtered_normals = result_map_filtered
            cv2.imshow("L2 Height Map (from Filtered Normals)", self.height_map_l2_filtered_normals)
            print("L2 Height map from Bilaterally Filtered Normals calculation complete.")
            self.calculate_roughness_stats_from_heightmap(height_map_type='l2_filtered_normals')
            return self.height_map_l2_filtered_normals is not None
        else: # Just return the raw float map
             # If raw is requested, we don't store it in the class attribute directly, just return.
             # The class attribute self.height_map_l2_filtered_normals should be the uint8 version.
             # For consistency, if raw is needed externally, the caller should handle it.
             # Or, we could store the raw if a flag is set, but let's keep it simple: this method stores uint8 if not return_raw.
            return result_map_filtered


    def calculate_roughness_stats_from_normals(self, scale_factor=1.0):
        """Calculates Ra, Rq roughness statistics from the current self.normals_map."""
        if self.normals_map is None: 
            print("Normals map (self.normals_map) not available for roughness stats.")
            self.Ra_normals, self.Rq_normals = None, None
            return False

        _normals_float = (self.normals_map.astype(np.float32) / 127.5) - 1.0
        if _normals_float.ndim == 3 and _normals_float.shape[2] == 3:
            nz = _normals_float[...,2]
        else: 
            print(f"Error: Current self.normals_map data is not 3-channel. Shape: {_normals_float.shape}")
            self.Ra_normals, self.Rq_normals = None, None
            return False
            
        angle_rad = np.arccos(np.clip(nz, -1.0, 1.0))
        roughness_values = angle_rad / (np.pi/2) 
        roughness_values = np.clip(roughness_values * scale_factor, 0, 1)
        
        if roughness_values.size == 0: 
            print("Warning: No roughness values to calculate stats from (current normals).")
            self.Ra_normals, self.Rq_normals = None, None
            return False
            
        self.Ra_normals = np.mean(np.abs(roughness_values - np.mean(roughness_values)))
        self.Rq_normals = np.sqrt(np.mean((roughness_values - np.mean(roughness_values))**2))
        print(f"Roughness Stats from Current Normals (self.normals_map): Ra={self.Ra_normals:.4f}, Rq={self.Rq_normals:.4f}")
        return True

    def calculate_roughness_stats_from_heightmap(self, height_map_type='l2_baseline', scale_factor=1.0):
        """Calculates Ra, Rq stats from L2 Baseline or L2 from Filtered Normals height map."""
        height_map_to_use = None
        map_label = ""
        target_Ra_attr, target_Rq_attr = None, None # String names of attributes to update

        if height_map_type == 'l2_baseline':
            if self.height_map_l2 is not None:
                height_map_to_use = self.height_map_l2
                map_label = "L2 Baseline"
                target_Ra_attr, target_Rq_attr = "Ra_height_l2", "Rq_height_l2"
            else: # Clear specific stats if map is None
                self.Ra_height_l2, self.Rq_height_l2 = None, None
        elif height_map_type == 'l2_filtered_normals':
            if self.height_map_l2_filtered_normals is not None:
                height_map_to_use = self.height_map_l2_filtered_normals
                map_label = "L2 from Filtered Normals"
                target_Ra_attr, target_Rq_attr = "Ra_height_l2_filtered_normals", "Rq_height_l2_filtered_normals"
            else: # Clear specific stats if map is None
                 self.Ra_height_l2_filtered_normals, self.Rq_height_l2_filtered_normals = None, None
        
        if height_map_to_use is None: 
            print(f"Height map '{map_label if map_label else height_map_type}' not available for roughness stats.")
            # Attributes already cleared if map was None
            return False
        
        h_float = height_map_to_use.astype(np.float32) / 255.0 # Normalize to 0-1 range
        gy, gx = np.gradient(h_float) # Gradients represent slope
        grad_mag_values = np.sqrt(gx**2 + gy**2) 
        # grad_mag_values are now unnormalized slope magnitudes. scale_factor could be used here
        # For example, if scale_factor is pixel size in mm, grad_mag_values are dh/dx (unitless),
        # and Ra/Rq would be of these unitless slope variations.
        
        # Apply scale_factor to the metric being evaluated (gradient magnitudes)
        # This interpretation of scale_factor for heightmap roughness might need clarification.
        # If scale_factor is to make it comparable to normal-derived roughness,
        # it should map gradient magnitudes to a 0-1 range similar to angle deviation.
        # For now, let's assume scale_factor applies to the raw gradient magnitudes if used.
        # grad_mag_values = np.clip(grad_mag_values * scale_factor, 0, 1) # Example if scaling to 0-1

        if grad_mag_values.size == 0: 
            print(f"Warning: No gradient magnitude values to calculate stats from (H-{map_label}).")
            current_Ra, current_Rq = None, None
        else:
            current_Ra = np.mean(np.abs(grad_mag_values - np.mean(grad_mag_values)))
            current_Rq = np.sqrt(np.mean((grad_mag_values - np.mean(grad_mag_values))**2))

        # Update the target attributes
        if target_Ra_attr and target_Rq_attr:
            setattr(self, target_Ra_attr, current_Ra)
            setattr(self, target_Rq_attr, current_Rq)
            if current_Ra is not None: 
                print(f"Roughness Stats from H-{map_label} (Grad Mag): Ra={current_Ra:.4f}, Rq={current_Rq:.4f}")
            else: # Failed to calculate Ra/Rq
                print(f"Failed to calculate roughness stats for H-{map_label}.")
        
        return current_Ra is not None

    def save_results(self, prefix="photometric"):
        """Saves all generated maps to files."""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        if self.processed_image is not None:
            cv2.imwrite(f"{prefix}_processed_{timestamp}.png", (self.processed_image*255).astype(np.uint8))
        if self.normals_map is not None:
            cv2.imwrite(f"{prefix}_normals_map_active_{timestamp}.png", self.normals_map)
        if self.albedo_map is not None:
            cv2.imwrite(f"{prefix}_albedo_map_{timestamp}.png", self.albedo_map)
            cv2.imwrite(f"{prefix}_albedo_map_enhanced_{timestamp}.png", self.albedo_map_enhanced)
            cv2.imwrite(f"{prefix}_albedo_map_binary_{timestamp}.png", self.albedo_map_binary)
        if self.height_map_l2 is not None:
            cv2.imwrite(f"{prefix}_height_map_L2_baseline_{timestamp}.png", self.height_map_l2)
        if self.height_map_l2_filtered_normals is not None:
            cv2.imwrite(f"{prefix}_height_map_L2_from_filtered_normals_{timestamp}.png", self.height_map_l2_filtered_normals)

        roughness_stats_file = f"{prefix}_roughness_stats_{timestamp}.txt"
        with open(roughness_stats_file, 'w') as f:
            f.write(f"Roughness Statistics for {prefix} captured at {timestamp}\n")
            f.write("---------------------------------------------------\n")
            f.write("(Stats for 'Current Normals' are based on the active 'self.normals_map')\n")
            if self.Ra_normals is not None:
                f.write(f"  Ra (Current Normals - Angular Deviation): {self.Ra_normals:.6f}\n")
            if self.Rq_normals is not None:
                f.write(f"  Rq (Current Normals - Angular Deviation): {self.Rq_normals:.6f}\n\n")
            
            f.write("(Stats for 'L2 Baseline Height' are based on 'self.height_map_l2')\n")
            if self.Ra_height_l2 is not None:
                f.write(f"  Ra (L2 Baseline Height - Grad Mag): {self.Ra_height_l2:.6f}\n")
            if self.Rq_height_l2 is not None:
                f.write(f"  Rq (L2 Baseline Height - Grad Mag): {self.Rq_height_l2:.6f}\n\n")
            
            f.write("(Stats for 'L2 Height from Filtered Normals' are based on 'self.height_map_l2_filtered_normals')\n")
            if self.Ra_height_l2_filtered_normals is not None:
                f.write(f"  Ra (L2 Height from Filtered Normals - Grad Mag): {self.Ra_height_l2_filtered_normals:.6f}\n")
            if self.Rq_height_l2_filtered_normals is not None:
                f.write(f"  Rq (L2 Height from Filtered Normals - Grad Mag): {self.Rq_height_l2_filtered_normals:.6f}\n")
        
        print(f"All available results saved with timestamp {timestamp}. Roughness stats in '{roughness_stats_file}'")

    def run(self):
        if not self.connect_sensor():
            print("Failed to connect to DIGIT sensor.")
            return

        print("\nPhotometric Stereo - L2 Suite with Optional Bilateral Normal Filtering")
        print("-----------------------------------------------------------------------")
        print("  SPACE - Full BASLINE pipeline (capture, process, standard normals, albedo, L2H baseline & stats, normal stats)")
        print("  r - Capture reference (1s)")
        print("  R - Capture reference (5s)")
        print("  m - Capture measurement (1s)")
        print("  M - Capture measurement (5s)")
        print("  p - Process captures -> difference image")
        print("  ---------------------------------------------------------------------")
        print("  n - Calculate normals & albedo (STANDARD). Updates current normals map.")
        print("  B - Calculate normals & albedo (BILATERAL FILTERED). Updates current normals map.")
        print("  ---------------------------------------------------------------------")
        print("  h - Calculate L2 Height (BASELINE from CURRENT normals map) & its roughness stats.")
        print("  E - Calculate L2 Height (from NEWLY BILATERALLY FILTERED NORMALS) & its roughness stats.")
        print("      (Note: 'E' uses fresh filtered normals; does NOT change current normals map used by 'h')")
        print("  ---------------------------------------------------------------------")
        print("  1 - (Re)Calculate Roughness Stats from CURRENT normals_map")
        print("  2 - (Re)Calculate Roughness Stats from L2 BASELINE Heightmap (if available)")
        print("  3 - (Re)Calculate Roughness Stats from L2 (from Filtered Normals) Heightmap (if available)")
        print("  ---------------------------------------------------------------------")
        print("  S - Save all available maps & stats")
        print("  q - Quit")
        print("-----------------------------------------------------------------------")

        cv2.namedWindow("Capture", cv2.WINDOW_NORMAL)
        # Create other windows that will be used to prevent them from resizing/moving
        cv2.namedWindow("Processed Image", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Normals Map (Standard)", cv2.WINDOW_AUTOSIZE) # Placeholder name
        cv2.namedWindow("Albedo Map", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("L2 Height Map (Baseline - from current normals)", cv2.WINDOW_AUTOSIZE)

        while True:
            frame = self.digit_sensor.get_frame()
            if frame is not None:
                cv2.imshow("Capture", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                print("\n--- Running Full Baseline Pipeline ---")
                if self.capture_reference(0.5) and self.capture_measurement(0.5):
                    if self.process_captures():
                        if self.calculate_normals_and_albedo(apply_bilateral_filter=False):
                            self.calculate_height_map_l2_baseline()
                print("--- Pipeline Complete ---\n")
            elif key == ord('r'):
                self.capture_reference(1.0)
            elif key == ord('R'):
                self.capture_reference(5.0)
            elif key == ord('m'):
                self.capture_measurement(1.0)
            elif key == ord('M'):
                self.capture_measurement(5.0)
            elif key == ord('p'):
                self.process_captures()
            elif key == ord('n'):
                self.calculate_normals_and_albedo(apply_bilateral_filter=False)
            elif key == ord('B'):
                # This option is now disabled for simplification, but kept in menu for consistency
                print("Bilateral filter option is disabled for performance. Running standard calculation.")
                self.calculate_normals_and_albedo(apply_bilateral_filter=False)
            elif key == ord('h'):
                self.calculate_height_map_l2_baseline()
            elif key == ord('E'):
                self.calculate_height_map_l2_from_filtered_normals()
            elif key == ord('1'):
                self.calculate_roughness_stats_from_normals()
            elif key == ord('2'):
                self.calculate_roughness_stats_from_heightmap(height_map_type='l2_baseline')
            elif key == ord('3'):
                self.calculate_roughness_stats_from_heightmap(height_map_type='l2_filtered_normals')
            elif key == ord('S'):
                self.save_results()

        self.digit_sensor.disconnect()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    ps = PhotometricStereo()
    ps.run() 