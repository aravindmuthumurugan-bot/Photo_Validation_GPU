import cv2
import os
import numpy as np
import base64
import tempfile
from typing import Dict, List, Tuple, Optional
from retinaface import RetinaFace
from nudenet import NudeDetector

# InsightFace imports (BACKBONE)
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

# DeepFace imports (AGE & ETHNICITY ONLY)
from deepface import DeepFace

# ==================== GPU CONFIGURATION (CRITICAL) ====================
import os
import sys

# Set environment variables BEFORE importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# TensorFlow GPU configuration
import tensorflow as tf

# Configure GPU memory growth to prevent OOM
gpus = tf.config.experimental.list_physical_devices('GPU')
GPU_AVAILABLE = False
GPU_NAME = "CPU"

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Optionally limit GPU memory usage (uncomment if needed)
        # tf.config.set_logical_device_configuration(
        #     gpus[0],
        #     [tf.config.LogicalDeviceConfiguration(memory_limit=8192)]  # 8GB limit
        # )
        GPU_AVAILABLE = True
        GPU_NAME = tf.config.experimental.get_device_details(gpus[0]).get('device_name', 'GPU')
        print(f"TensorFlow GPU configured: {GPU_NAME}")
        print(f"Number of GPUs available: {len(gpus)}")
    except RuntimeError as e:
        print(f"TensorFlow GPU configuration error: {e}")
        GPU_AVAILABLE = False
else:
    print("No TensorFlow GPU devices found")

# ONNX Runtime GPU configuration (for InsightFace)
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f"ONNX Runtime providers: {providers}")

    if 'CUDAExecutionProvider' in providers:
        GPU_AVAILABLE = True
        # Set ONNX Runtime to use GPU with optimized settings
        ONNX_PROVIDERS = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 8 * 1024 * 1024 * 1024,  # 8GB limit
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }),
            'CPUExecutionProvider'
        ]
        print(f"GPU Available: CUDA (ONNX Runtime)")
    else:
        ONNX_PROVIDERS = ['CPUExecutionProvider']
        print("ONNX Runtime: Running on CPU")
except ImportError:
    ONNX_PROVIDERS = ['CPUExecutionProvider']
    print("ONNX Runtime not available")
except Exception as e:
    ONNX_PROVIDERS = ['CPUExecutionProvider']
    print(f"ONNX Runtime GPU detection error: {e}")


# ==================== STAGE 1 CONFIGURATION ====================

# Stage 1 Thresholds
MIN_RESOLUTION = 360
MIN_FACE_SIZE = 120
BLUR_REJECT = 35
MIN_FACE_COVERAGE_S1 = 0.05
MAX_YAW_ANGLE = 20
BLUR_AFTER_CROP_MIN = 40

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif"
}

# NudeNet detector (loaded once)
nsfw_detector = NudeDetector()


# ==================== STAGE 2 CONFIGURATION ====================

# Gender validation
GENDER_CONFIDENCE_THRESHOLD = 0.70

# Ethnicity thresholds (DeepFace)
INDIAN_PROBABILITY_MIN = 0.30
DISALLOWED_ETHNICITIES = {
    "white": 0.60,
    "black": 0.60,
    "asian": 0.50,
    "middle eastern": 0.60,
    "latino hispanic": 0.60
}

# Age variance thresholds
AGE_VARIANCE_PASS = 8
AGE_VARIANCE_REVIEW = 15

# Face coverage for Stage 2
MIN_FACE_COVERAGE_S2 = 0.05

# Enhancement/filter detection
FILTER_SATURATION_THRESHOLD = 1.5

# Paper-of-photo indicators
PAPER_WHITE_THRESHOLD = 240

# Face similarity thresholds
DUPLICATE_THRESHOLD_STRICT = 0.40
DUPLICATE_THRESHOLD_REVIEW = 0.50
PRIMARY_PERSON_MATCH_THRESHOLD = 0.50


# ==================== INSIGHTFACE INITIALIZATION (BACKBONE) ====================

print("Initializing InsightFace (BACKBONE)...")
print(f"Using providers: {ONNX_PROVIDERS}")

try:
    if GPU_AVAILABLE:
        # Use optimized GPU providers
        app = FaceAnalysis(
            name='buffalo_l',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
    else:
        app = FaceAnalysis(
            name='buffalo_l',
            providers=['CPUExecutionProvider']
        )

    # ctx_id: 0 = first GPU, -1 = CPU
    app.prepare(ctx_id=0 if GPU_AVAILABLE else -1, det_size=(640, 640))
    print(f"InsightFace FaceAnalysis initialized (GPU: {GPU_AVAILABLE})")

except Exception as e:
    print(f"InsightFace initialization error: {e}")
    print("Falling back to CPU mode...")
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=-1, det_size=(640, 640))

# Initialize recognition model for face comparison
print("Loading InsightFace recognition model...")
try:
    if GPU_AVAILABLE:
        recognition_model = get_model(
            'buffalo_l',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
    else:
        recognition_model = get_model('buffalo_l', providers=['CPUExecutionProvider'])
except Exception as e:
    print(f"Recognition model load error: {e}")
    recognition_model = get_model('buffalo_l', providers=['CPUExecutionProvider'])

print(f"InsightFace initialized successfully (GPU: {GPU_AVAILABLE}, Device: {GPU_NAME})")


# ==================== DEEPFACE CONFIGURATION (AGE & ETHNICITY ONLY) ====================

print("DeepFace will be used for:")
print("  - Age verification (PRIMARY photos only)")
print("  - Ethnicity validation (PRIMARY photos only)")
print("  - Gender validation (optional fallback)")


# ==================== STAGE 1 UTILITY FUNCTIONS ====================

def image_to_base64(image_array):
    """Convert numpy image array to base64 string"""
    try:
        _, buffer = cv2.imencode('.jpg', image_array)
        base64_str = base64.b64encode(buffer).decode('utf-8')
        return base64_str
    except Exception as e:
        print(f"Error converting image to base64: {str(e)}")
        return None

def reject(reason, checks, cropped_image=None):
    return {
        "stage": 1,
        "result": "REJECT",
        "reason": reason,
        "checks": checks,
        "cropped_image": cropped_image
    }

def pass_stage(checks, cropped_image=None):
    return {
        "stage": 1,
        "result": "PASS",
        "reason": None,
        "checks": checks,
        "cropped_image": cropped_image
    }

def is_supported_format(image_path):
    ext = os.path.splitext(image_path.lower())[1]
    return ext in SUPPORTED_EXTENSIONS

def is_resolution_ok(img):
    h, w = img.shape[:2]
    return min(h, w) >= MIN_RESOLUTION

def blur_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def is_orientation_ok(landmarks):
    """Eyes must be above nose"""
    le_y = landmarks["left_eye"][1] 
    re_y = landmarks["right_eye"][1]
    nose_y = landmarks["nose"][1]
    
    if le_y > nose_y or re_y > nose_y:
        return False
    return True

def is_face_covered(landmarks):
    """If mouth landmarks missing → likely mask / full cover"""
    return (
        "mouth_left" not in landmarks or
        "mouth_right" not in landmarks
    )

def detect_hand_occlusion_improved(img, face_area, landmarks):
    """Simplified and more reliable hand occlusion detection"""
    try:
        x1, y1, x2, y2 = face_area
        face_width = x2 - x1
        face_height = y2 - y1
        
        left_eye = landmarks["left_eye"]
        right_eye = landmarks["right_eye"]
        nose = landmarks["nose"]
        
        critical_regions = {
            "left_eye": {
                "center": left_eye,
                "radius_x": face_width * 0.08,
                "radius_y": face_height * 0.06
            },
            "right_eye": {
                "center": right_eye,
                "radius_x": face_width * 0.08,
                "radius_y": face_height * 0.06
            },
            "nose": {
                "center": nose,
                "radius_x": face_width * 0.10,
                "radius_y": face_height * 0.10
            }
        }
        
        padding = 30
        x1_pad = max(0, x1 - padding)
        y1_pad = max(0, y1 - padding)
        x2_pad = min(img.shape[1], x2 + padding)
        y2_pad = min(img.shape[0], y2 + padding)
        
        face_roi = img[y1_pad:y2_pad, x1_pad:x2_pad]
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        edges = cv2.Canny(gray_face, 50, 150)
        
        for region_name, region_info in critical_regions.items():
            cx, cy = region_info["center"]
            rx, ry = region_info["radius_x"], region_info["radius_y"]
            
            cx_roi = int(cx - x1_pad)
            cy_roi = int(cy - y1_pad)
            rx_int = int(rx)
            ry_int = int(ry)
            
            region_x1 = max(0, cx_roi - rx_int)
            region_y1 = max(0, cy_roi - ry_int)
            region_x2 = min(face_roi.shape[1], cx_roi + rx_int)
            region_y2 = min(face_roi.shape[0], cy_roi + ry_int)
            
            if region_x2 <= region_x1 or region_y2 <= region_y1:
                continue
            
            feature_region = face_roi[region_y1:region_y2, region_x1:region_x2]
            
            if feature_region.size == 0:
                continue
            
            hsv_region = cv2.cvtColor(feature_region, cv2.COLOR_BGR2HSV)
            
            if "eye" in region_name:
                avg_brightness = np.mean(hsv_region[:, :, 2])
                if avg_brightness < 25:
                    return True, f"Hand covering {region_name} - feature not visible"
            
            feature_edges = edges[region_y1:region_y2, region_x1:region_x2]
            edge_density = np.count_nonzero(feature_edges) / feature_edges.size if feature_edges.size > 0 else 0
            
            if edge_density > 0.40:
                return True, f"Hand/object detected covering {region_name}"
        
        upper_face = face_roi[:int(face_roi.shape[0] * 0.6), :]
        upper_edges = cv2.Canny(upper_face, 50, 150)
        contours, _ = cv2.findContours(upper_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < face_width * face_height * 0.08:
                continue
            
            for region_name, region_info in critical_regions.items():
                if "eye" not in region_name:
                    continue
                
                cx, cy = region_info["center"]
                cx_roi = int(cx - x1_pad)
                cy_roi = int(cy - y1_pad)
                
                if cv2.pointPolygonTest(contour, (cx_roi, cy_roi), False) > 0:
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    if hull_area > 0:
                        solidity = area / hull_area
                        if solidity < 0.7:
                            return True, f"Hand/object detected near {region_name}"
        
        return False, "No facial feature occlusion detected"
    
    except Exception as e:
        print(f"Hand occlusion detection error: {str(e)}")
        return False, f"Occlusion detection error: {str(e)}"


def check_yaw_improved(landmarks, img_shape):
    """Enhanced yaw detection with multiple methods"""
    left_eye = np.array(landmarks["left_eye"])
    right_eye = np.array(landmarks["right_eye"])
    nose = np.array(landmarks["nose"])
    
    dl = np.linalg.norm(nose - left_eye)
    dr = np.linalg.norm(nose - right_eye)
    distance_ratio = abs(dl - dr) / max(dl, dr)
    yaw_angle = distance_ratio * 90
    
    eye_midpoint_x = (left_eye[0] + right_eye[0]) / 2
    nose_x = nose[0]
    horizontal_offset = abs(nose_x - eye_midpoint_x)
    eye_distance = np.linalg.norm(right_eye - left_eye)
    
    offset_ratio = horizontal_offset / eye_distance if eye_distance > 0 else 1.0
    
    MAX_YAW_ANGLE = 15
    MAX_OFFSET_RATIO = 0.12
    
    issues = []
    
    if yaw_angle > MAX_YAW_ANGLE:
        issues.append(f"Side angle detected ({yaw_angle:.1f}°)")
    
    if offset_ratio > MAX_OFFSET_RATIO:
        issues.append(f"Face not centered (nose offset: {offset_ratio:.2f})")
    
    if issues:
        return False, yaw_angle, "; ".join(issues)
    
    return True, yaw_angle, f"Frontal face verified ({yaw_angle:.1f}°)"


def check_face_symmetry(img, face_area, landmarks):
    """Check if face is frontal by verifying both sides are equally visible"""
    try:
        x1, y1, x2, y2 = face_area
        face_roi = img[y1:y2, x1:x2]
        
        if face_roi.size == 0:
            return True, "Could not verify face symmetry"
        
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        h, w = gray_face.shape
        
        left_eye_x = landmarks["left_eye"][0] - x1
        right_eye_x = landmarks["right_eye"][0] - x1
        nose_x = landmarks["nose"][0] - x1
        
        face_center_x = w / 2
        nose_offset_ratio = abs(nose_x - face_center_x) / w
        
        if nose_offset_ratio > 0.15:
            return False, f"Face not frontal - nose offset {nose_offset_ratio*100:.1f}% from center"
        
        left_eye_from_left = left_eye_x
        right_eye_from_right = w - right_eye_x
        
        if left_eye_from_left > 0 and right_eye_from_right > 0:
            distance_ratio = max(left_eye_from_left, right_eye_from_right) / min(left_eye_from_left, right_eye_from_right)
            
            if distance_ratio > 1.4:
                return False, f"Face not frontal - one side significantly more visible than other"
        
        left_half = gray_face[:, :w//2]
        right_half = gray_face[:, w//2:]
        right_half_flipped = cv2.flip(right_half, 1)
        
        min_width = min(left_half.shape[1], right_half_flipped.shape[1])
        left_half = left_half[:, :min_width]
        right_half_flipped = right_half_flipped[:, :min_width]
        
        diff = cv2.absdiff(left_half, right_half_flipped)
        asymmetry = np.mean(diff)
        
        if asymmetry > 65:
            return False, f"Face not frontal - significant asymmetry detected (score: {asymmetry:.1f})"
        
        face_width = x2 - x1
        left_side_width = nose_x - left_eye_x
        right_side_width = right_eye_x - nose_x
        
        if left_side_width > 0 and right_side_width > 0:
            width_ratio = max(left_side_width, right_side_width) / min(left_side_width, right_side_width)
            
            if width_ratio > 1.5:
                return False, f"Face not frontal - uneven face width distribution"
        
        return True, f"Face is frontal (symmetry score: {asymmetry:.1f})"
    
    except Exception as e:
        print(f"Face symmetry check error: {str(e)}")
        return True, f"Face symmetry check error: {str(e)}"

def calculate_face_coverage(face_area, img_shape):
    """Calculate what percentage of the image the face covers"""
    fw = face_area[2] - face_area[0]
    fh = face_area[3] - face_area[1]
    face_area_pixels = fw * fh
    
    img_h, img_w = img_shape[:2]
    img_area_pixels = img_h * img_w
    
    coverage = face_area_pixels / img_area_pixels
    return coverage

def crop_image_for_face_coverage(img, face_area, target_coverage=MIN_FACE_COVERAGE_S1):
    """Crop the image so that the face covers at least target_coverage of the image"""
    img_h, img_w = img.shape[:2]
    
    fx1, fy1, fx2, fy2 = face_area
    fw = fx2 - fx1
    fh = fy2 - fy1
    face_area_pixels = fw * fh
    
    current_coverage = face_area_pixels / (img_h * img_w)
    
    if current_coverage >= target_coverage:
        return img, False
    
    required_img_area = face_area_pixels / target_coverage
    
    face_aspect = fw / fh
    
    crop_w = int(np.sqrt(required_img_area * face_aspect))
    crop_h = int(np.sqrt(required_img_area / face_aspect))
    
    face_center_x = (fx1 + fx2) // 2
    face_center_y = (fy1 + fy2) // 2
    
    crop_x1 = max(0, face_center_x - crop_w // 2)
    crop_y1 = max(0, face_center_y - crop_h // 2)
    crop_x2 = min(img_w, crop_x1 + crop_w)
    crop_y2 = min(img_h, crop_y1 + crop_h)
    
    if crop_x2 - crop_x1 < crop_w:
        crop_x1 = max(0, crop_x2 - crop_w)
    if crop_y2 - crop_y1 < crop_h:
        crop_y1 = max(0, crop_y2 - crop_h)
    
    cropped_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
    
    return cropped_img, True


# ==================== STAGE 1 NSFW / BARE BODY CHECK ====================

def check_nsfw_stage1(image_path):
    """Stage-1 NSFW policy: ANY nudity / bare body → REJECT"""
    disallowed_classes = {
        "FEMALE_GENITALIA_EXPOSED",
        "MALE_GENITALIA_EXPOSED",
        "ANUS_EXPOSED",
        "FEMALE_BREAST_EXPOSED",
        "MALE_BREAST_EXPOSED",
        "FEMALE_BREAST_COVERED",
        "BELLY_EXPOSED",
        "BUTTOCKS_EXPOSED",
        "BUTTOCKS_COVERED",
        "UNDERWEAR",
        "SWIMWEAR"
    }
    
    detections = nsfw_detector.detect(image_path)
    
    for d in detections:
        if d["class"] in disallowed_classes and d["score"] > 0.6:
            return False, f"Disallowed content detected ({d['class']})"
    
    return True, None


# ==================== GROUP PHOTO VALIDATION (INSIGHTFACE) ====================

def find_primary_person_in_group(
    group_photo_path: str,
    reference_photo_path: str
) -> Tuple[bool, Optional[int], Optional[float], str]:
    """Find the primary person in a group photo using InsightFace"""
    try:
        group_img = cv2.imread(group_photo_path)
        ref_img = cv2.imread(reference_photo_path)
        
        group_faces = app.get(group_img)
        
        if not group_faces or len(group_faces) == 0:
            return False, None, None, "No faces detected in group photo"
        
        if len(group_faces) == 1:
            return False, None, None, "Only one face detected - not a group photo"
        
        ref_faces = app.get(ref_img)
        if not ref_faces or len(ref_faces) == 0:
            return False, None, None, "No face detected in reference photo"
        
        ref_embedding = ref_faces[0].embedding
        
        best_match_index = None
        best_match_similarity = -1
        
        for idx, face in enumerate(group_faces):
            similarity = np.dot(ref_embedding, face.embedding) / (
                np.linalg.norm(ref_embedding) * np.linalg.norm(face.embedding)
            )
            
            if similarity > best_match_similarity:
                best_match_similarity = similarity
                best_match_index = idx
        
        if best_match_index is not None and best_match_similarity > PRIMARY_PERSON_MATCH_THRESHOLD:
            return True, best_match_index, best_match_similarity, f"Primary person found at face #{best_match_index + 1}"
        else:
            return False, best_match_index, best_match_similarity, f"Primary person not clearly identifiable in group photo (best match similarity: {best_match_similarity:.3f})"
            
    except Exception as e:
        return False, None, None, f"Error finding primary person: {str(e)}"


# ==================== STAGE 1 MAIN VALIDATOR ====================

def stage1_validate(
    image_path: str,
    photo_type: str = "PRIMARY",
    reference_photo_path: Optional[str] = None
):
    """Stage 1 validation - Basic quality and appropriateness checks"""

    checks = {}
    cropped_image = None

    # FORMAT
    if not is_supported_format(image_path):
        return reject("Unsupported image format", checks)
    checks["format"] = "PASS"

    # IMAGE READ
    img = cv2.imread(image_path)
    if img is None:
        return reject("Invalid or unreadable image", checks)
    checks["image_read"] = "PASS"

    # FACE DETECTION
    faces = RetinaFace.detect_faces(image_path)
    if not faces:
        return reject("No face detected", checks)

    face_count = len(faces)
    checks["face_count"] = f"{face_count} face(s) detected"

    # PRIMARY PHOTO: Must have exactly 1 face
    if photo_type == "PRIMARY":
        if face_count > 1:
            return reject("Group photo not allowed as primary photo. Primary photo must contain only your face.", checks)
        checks["photo_type_validation"] = "PASS - Single face for PRIMARY photo"

    # SECONDARY PHOTO: Simplified validation - only NSFW, face detection, quality, resolution, and matching
    elif photo_type == "SECONDARY":
        # RESOLUTION CHECK for secondary photos
        if not is_resolution_ok(img):
            return reject("Low resolution image", checks)
        checks["resolution"] = "PASS"

        # QUALITY (BLUR) CHECK for secondary photos
        blur = blur_score(img)
        checks["blur_score"] = f"{blur:.2f}"

        if blur < BLUR_REJECT:
            return reject("Image is too blurry", checks)
        checks["quality"] = "PASS"

        # NSFW CHECK for secondary photos
        nsfw_ok, nsfw_reason = check_nsfw_stage1(image_path)
        if not nsfw_ok:
            return reject(nsfw_reason, checks)
        checks["nsfw"] = "PASS"

        # FACE DETECTION for secondary photos
        checks["face_detection"] = f"PASS - {face_count} face(s) detected"

        # Handle group photos vs individual photos
        if face_count > 1:
            # ========== GROUP PHOTO VALIDATION ==========
            # For group photos: Only check NSFW, face detection, quality, resolution, and face matching
            # Skip: age, gender, ethnicity, face size, face coverage, cropping

            checks["photo_type_validation"] = f"PASS - Group photo with {face_count} faces detected"

            if reference_photo_path is None:
                return reject("Group photo detected but no reference photo provided to identify primary person", checks)

            # Detect all faces in the group photo and check if primary person is present
            found, face_idx, similarity, message = find_primary_person_in_group(
                image_path,
                reference_photo_path
            )

            checks["group_photo_validation"] = message

            if not found:
                # REJECT: Primary person not found in the group photo
                return reject(
                    f"The person in the primary photo is not present in the group photo. {message}",
                    checks
                )

            # ACCEPT: Primary person found in the group photo
            checks["face_matching"] = f"PASS - Primary person found in group photo (face #{face_idx + 1}, similarity: {similarity:.3f})"
            checks["face_coverage_check"] = "SKIPPED - Not applicable for group photos"
            checks["cropping_applied"] = "NO - Group photos are not cropped"

            # For group photos, we're done after matching check
            return pass_stage(checks, cropped_image)

        else:
            # ========== INDIVIDUAL PHOTO VALIDATION ==========
            # For individual photos: Check NSFW, face detection, quality, resolution, face matching, and 5% face coverage
            # Skip: age, gender, ethnicity checks (only for PRIMARY photos)

            checks["photo_type_validation"] = "PASS - Single face for SECONDARY photo"

            # Get face information
            face = list(faces.values())[0]
            area = face["facial_area"]

            # FACE SIZE CHECK for individual secondary photos (basic validation)
            fw = area[2] - area[0]
            fh = area[3] - area[1]

            if min(fw, fh) < MIN_FACE_SIZE:
                return reject(f"Face too small or unclear (size: {min(fw, fh):.0f}px, minimum: {MIN_FACE_SIZE}px)", checks)
            checks["face_size"] = "PASS"

            # FACE MATCHING: Check if face matches primary photo
            if reference_photo_path is None:
                return reject("No reference photo provided to verify face matching", checks)

            # Use InsightFace to verify the person matches
            try:
                ref_img = cv2.imread(reference_photo_path)
                ref_faces = app.get(ref_img)
                if not ref_faces or len(ref_faces) == 0:
                    return reject("No face detected in reference photo", checks)

                curr_img = cv2.imread(image_path)
                curr_faces = app.get(curr_img)
                if not curr_faces or len(curr_faces) == 0:
                    return reject("No face detected in secondary photo", checks)

                ref_embedding = ref_faces[0].embedding
                curr_embedding = curr_faces[0].embedding

                similarity = np.dot(ref_embedding, curr_embedding) / (
                    np.linalg.norm(ref_embedding) * np.linalg.norm(curr_embedding)
                )

                if similarity < PRIMARY_PERSON_MATCH_THRESHOLD:
                    return reject(
                        f"Face does not match primary photo (similarity: {similarity:.3f})",
                        checks
                    )

                checks["face_matching"] = f"PASS - Matches primary photo (similarity: {similarity:.3f})"
            except Exception as e:
                return reject(f"Error during face matching: {str(e)}", checks)

            # FACE COVERAGE CHECK: Ensure at least 5% face coverage
            coverage = calculate_face_coverage(area, img.shape)
            checks["face_coverage_original"] = f"{coverage * 100:.2f}%"

            if coverage < MIN_FACE_COVERAGE_S1:
                # Perform crop to achieve 5% coverage for individual secondary photos
                img_for_validation, was_cropped = crop_image_for_face_coverage(img, area)

                if not was_cropped:
                    return reject("Face coverage insufficient (less than 5%) and auto-crop failed", checks)

                checks["cropping_applied"] = "YES"

                # Save and re-detect face in cropped image
                cropped_temp_path = image_path.replace(".", "_cropped.")
                cv2.imwrite(cropped_temp_path, img_for_validation)

                faces_cropped = RetinaFace.detect_faces(cropped_temp_path)
                if not faces_cropped:
                    os.remove(cropped_temp_path)
                    return reject("Face lost after cropping", checks)

                face = list(faces_cropped.values())[0]
                area = face["facial_area"]

                new_coverage = calculate_face_coverage(area, img_for_validation.shape)
                checks["face_coverage_after_crop"] = f"{new_coverage * 100:.2f}%"

                cropped_image = img_for_validation
                os.remove(cropped_temp_path)
            else:
                checks["cropping_applied"] = "NO"
                checks["face_coverage_check"] = f"PASS - Face coverage sufficient ({coverage * 100:.2f}%)"

            # For individual secondary photos, we're done
            return pass_stage(checks, cropped_image)
    
    face = list(faces.values())[0]
    area = face["facial_area"]
    landmarks = face["landmarks"]
    
    # FACE COVERAGE & CROPPING
    coverage = calculate_face_coverage(area, img.shape)
    checks["face_coverage_original"] = f"{coverage * 100:.2f}%"
    
    img_for_validation = img
    was_cropped = False
    
    if photo_type == "PRIMARY" and coverage < MIN_FACE_COVERAGE_S1:
        img_for_validation, was_cropped = crop_image_for_face_coverage(img, area)
        checks["cropping_applied"] = "YES"
        
        cropped_temp_path = image_path.replace(".", "_cropped.")
        cv2.imwrite(cropped_temp_path, img_for_validation)
        
        faces_cropped = RetinaFace.detect_faces(cropped_temp_path)
        if not faces_cropped:
            os.remove(cropped_temp_path)
            return reject("Face lost after cropping", checks)
        
        face = list(faces_cropped.values())[0]
        area = face["facial_area"]
        landmarks = face["landmarks"]
        
        new_coverage = calculate_face_coverage(area, img_for_validation.shape)
        checks["face_coverage_after_crop"] = f"{new_coverage * 100:.2f}%"
        
        if photo_type == "PRIMARY":
            blur_after_crop = blur_score(img_for_validation)
            checks["blur_after_crop"] = f"{blur_after_crop:.2f}"
            
            if blur_after_crop < BLUR_AFTER_CROP_MIN:
                os.remove(cropped_temp_path)
                return reject(f"Image quality too low after cropping (blur score: {blur_after_crop:.2f})", checks)
            
            crop_h, crop_w = img_for_validation.shape[:2]
            if min(crop_h, crop_w) < MIN_RESOLUTION:
                os.remove(cropped_temp_path)
                return reject(f"Cropped image resolution too low ({crop_w}x{crop_h})", checks)
        
        cropped_image = img_for_validation
        validation_image_path = cropped_temp_path
    else:
        checks["cropping_applied"] = "NO"
        validation_image_path = image_path
    
    # FACE SIZE
    fw = area[2] - area[0]
    fh = area[3] - area[1]
    
    if min(fw, fh) < MIN_FACE_SIZE:
        if was_cropped:
            os.remove(cropped_temp_path)
        return reject("Face too small or unclear", checks, cropped_image)
    
    checks["face_size"] = "PASS"
    
    # RESOLUTION
    if not is_resolution_ok(img_for_validation):
        if was_cropped:
            os.remove(cropped_temp_path)
        return reject("Low resolution image", checks, cropped_image)
    checks["resolution"] = "PASS"
    
    # BLUR
    blur = blur_score(img_for_validation)
    checks["blur_score"] = f"{blur:.2f}"
    
    if blur < BLUR_REJECT:
        if was_cropped:
            os.remove(cropped_temp_path)
        return reject("Image is too blurry", checks, cropped_image)
    
    checks["blur"] = "PASS"
    
    # ORIENTATION
    if not is_orientation_ok(landmarks):
        if was_cropped:
            os.remove(cropped_temp_path)
        return reject("Improper image orientation", checks, cropped_image)
    
    checks["orientation"] = "PASS"
    
    # YAW / SIDE FACE CHECK
    yaw_ok, yaw_angle, yaw_message = check_yaw_improved(landmarks, img_for_validation.shape)
    checks["yaw_angle"] = yaw_message
    
    if not yaw_ok:
        if was_cropped:
            os.remove(cropped_temp_path)
        return reject(yaw_message, checks, cropped_image)
    
    checks["face_pose"] = "PASS"
    
    # FACE SYMMETRY CHECK (PRIMARY ONLY)
    if photo_type == "PRIMARY":
        is_frontal, symmetry_message = check_face_symmetry(img_for_validation, area, landmarks)
        checks["face_symmetry"] = symmetry_message
        
        if not is_frontal:
            if was_cropped:
                os.remove(cropped_temp_path)
            return reject(symmetry_message, checks, cropped_image)
        
        checks["face_frontal"] = "PASS - " + symmetry_message
    else:
        checks["face_symmetry"] = "SKIPPED - Not required for family/group photos"
    
    # MASK / FACE COVER
    if is_face_covered(landmarks):
        if was_cropped:
            os.remove(cropped_temp_path)
        return reject("Face is covered or wearing a mask", checks, cropped_image)
    
    checks["face_cover"] = "PASS"
    
    # HAND OCCLUSION DETECTION (PRIMARY ONLY)
    if photo_type == "PRIMARY":
        is_occluded, occlusion_msg = detect_hand_occlusion_improved(img_for_validation, area, landmarks)
        checks["hand_occlusion"] = occlusion_msg
        
        if is_occluded:
            if was_cropped:
                os.remove(cropped_temp_path)
            return reject(occlusion_msg, checks, cropped_image)
        
        checks["hand_occlusion"] = "PASS - " + occlusion_msg
    else:
        checks["hand_occlusion"] = "SKIPPED - Not required for family/group photos"
    
    # NSFW / BARE BODY
    nsfw_ok, nsfw_reason = check_nsfw_stage1(validation_image_path)
    if not nsfw_ok:
        if was_cropped:
            os.remove(cropped_temp_path)
        return reject(nsfw_reason, checks, cropped_image)
    
    checks["nsfw"] = "PASS"
    
    # CLEANUP & FINAL
    if was_cropped:
        os.remove(cropped_temp_path)
    
    return pass_stage(checks, cropped_image)


# ==================== INSIGHTFACE FACE ANALYSIS ====================

def analyze_face_insightface(img_path: str) -> Dict:
    """InsightFace analysis for face detection and embeddings"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return {"error": "Could not read image", "data": None}
        
        faces = app.get(img)
        
        if not faces or len(faces) == 0:
            return {"error": "No face detected", "data": None}
        
        face = faces[0]
        
        face_data = {
            "bbox": face.bbox.tolist(),
            "kps": face.kps.tolist(),
            "det_score": float(face.det_score),
            "embedding": face.embedding,
            "gender": face.gender if hasattr(face, 'gender') else None,
            "age": int(face.age) if hasattr(face, 'age') else None,
        }
        
        return {"error": None, "data": face_data}
        
    except Exception as e:
        return {"error": str(e), "data": None}


# ==================== DEEPFACE: AGE VALIDATION ====================

def validate_age_deepface(img_path: str, profile_age: int) -> Dict:
    """
    Age verification using DeepFace (PRIMARY photos only)
    DeepFace has better age accuracy than InsightFace
    """
    try:
        print(f"[DeepFace] Running age detection for profile age: {profile_age}...")
        
        # DeepFace analyze
        result = DeepFace.analyze(
            img_path=img_path,
            actions=['age'],
            enforce_detection=True,
            detector_backend='retinaface',
            silent=True
        )
        
        # Handle list or dict result
        if isinstance(result, list):
            result = result[0]
        
        detected_age = int(result.get('age', 0))
        
        if detected_age == 0:
            return {
                "status": "REVIEW",
                "reason": "Could not detect age from photo",
                "detected_age": None,
                "profile_age": profile_age
            }
        
        variance = abs(detected_age - profile_age)
        
        # CRITICAL: Check for underage
        if detected_age < 18:
            return {
                "status": "FAIL",
                "reason": f"Underage detected: {detected_age} years",
                "detected_age": detected_age,
                "profile_age": profile_age,
                "variance": variance,
                "action": "SUSPEND"
            }
        
        # Extra scrutiny for young ages
        if detected_age < 23:
            return {
                "status": "REVIEW",
                "reason": f"Young age detected: {detected_age} years. Manual verification recommended.",
                "detected_age": detected_age,
                "profile_age": profile_age,
                "variance": variance
            }
        
        if variance < AGE_VARIANCE_PASS:
            return {
                "status": "PASS",
                "reason": f"Age verified: {detected_age} (profile: {profile_age}, variance: {variance} years)",
                "detected_age": detected_age,
                "profile_age": profile_age,
                "variance": variance
            }
        elif variance <= AGE_VARIANCE_REVIEW:
            return {
                "status": "REVIEW",
                "reason": f"Moderate age variance: profile {profile_age}, detected {detected_age} (variance: {variance} years)",
                "detected_age": detected_age,
                "profile_age": profile_age,
                "variance": variance
            }
        else:
            return {
                "status": "FAIL",
                "reason": f"Large age variance: profile {profile_age}, detected {detected_age} (variance: {variance} years)",
                "detected_age": detected_age,
                "profile_age": profile_age,
                "variance": variance
            }
        
    except Exception as e:
        print(f"[DeepFace] Age detection error: {str(e)}")
        return {
            "status": "REVIEW",
            "reason": f"Age detection failed: {str(e)}",
            "detected_age": None,
            "profile_age": profile_age
        }


# ==================== DEEPFACE: ETHNICITY VALIDATION ====================

def validate_ethnicity_deepface(img_path: str) -> Dict:
    """
    Ethnicity validation using DeepFace (PRIMARY photos only)
    DeepFace has ethnicity detection, InsightFace doesn't
    
    Logic:
    1. DeepFace returns probabilities on 0-100 scale (e.g., 37.96 = 37.96%)
    2. Indian probability should be >= 30% (INDIAN_PROBABILITY_MIN = 0.30)
    3. Disallowed ethnicities should not exceed their thresholds
       - DISALLOWED_ETHNICITIES are in decimal format (0.60 = 60%)
    """
    try:
        print("[DeepFace] Running ethnicity detection...")
        
        # DeepFace analyze
        result = DeepFace.analyze(
            img_path=img_path,
            actions=['race'],
            enforce_detection=True,
            detector_backend='retinaface',
            silent=True
        )
        
        # Handle list or dict result
        if isinstance(result, list):
            result = result[0]
        
        race_scores = result.get('race', {})
        
        if not race_scores:
            return {
                "status": "REVIEW",
                "reason": "Could not detect ethnicity",
                "indian_probability": None,
                "all_scores": None
            }
        
        # Get Indian probability (already on 0-100 scale from DeepFace)
        indian_prob = race_scores.get('indian', 0.0)
        
        print(f"[DeepFace] Ethnicity scores: {race_scores}")
        print(f"[DeepFace] Indian probability: {indian_prob:.2f}%")
        
        # Check disallowed ethnicities
        # DISALLOWED_ETHNICITIES thresholds are in decimal format (0.60 = 60%)
        # DeepFace scores are on 0-100 scale
        # So we need to multiply threshold by 100 for comparison
        for ethnicity, threshold_decimal in DISALLOWED_ETHNICITIES.items():
            # Convert threshold from decimal to percentage (0.60 -> 60.0)
            threshold_percentage = threshold_decimal * 100
            
            # Try to get the probability for this ethnicity
            prob = race_scores.get(ethnicity, 0.0)
            
            # Compare: if actual probability > threshold, reject
            if prob > threshold_percentage:
                return {
                    "status": "FAIL",
                    "reason": f"Ethnicity check failed: High {ethnicity} probability ({prob:.2f}% exceeds threshold {threshold_percentage:.0f}%)",
                    "indian_probability": indian_prob,
                    "all_scores": race_scores
                }
        
        # Check if Indian probability is sufficient
        # INDIAN_PROBABILITY_MIN = 0.30 (30%)
        # Convert to percentage for comparison: 0.30 * 100 = 30.0
        indian_threshold = INDIAN_PROBABILITY_MIN * 100
        
        if indian_prob < indian_threshold:
            return {
                "status": "REVIEW",
                "reason": f"Low Indian ethnicity probability ({indian_prob:.2f}%). Manual review recommended.",
                "indian_probability": indian_prob,
                "all_scores": race_scores
            }
        
        # PASS: Indian probability is sufficient and no disallowed ethnicity exceeds threshold
        return {
            "status": "PASS",
            "reason": f"Ethnicity verified: Indian ({indian_prob:.2f}%)",
            "indian_probability": indian_prob,
            "all_scores": race_scores
        }
        
    except Exception as e:
        print(f"[DeepFace] Ethnicity detection error: {str(e)}")
        return {
            "status": "REVIEW",
            "reason": f"Ethnicity detection failed: {str(e)}",
            "indian_probability": None,
            "all_scores": None
        }


# ==================== DEEPFACE: GENDER VALIDATION (OPTIONAL FALLBACK) ====================

def validate_gender_deepface(img_path: str, profile_gender: str) -> Dict:
    """
    Gender validation using DeepFace (optional fallback if InsightFace gender is unreliable)
    """
    try:
        print("[DeepFace] Running gender detection...")
        
        result = DeepFace.analyze(
            img_path=img_path,
            actions=['gender'],
            enforce_detection=True,
            detector_backend='retinaface',
            silent=True
        )
        
        if isinstance(result, list):
            result = result[0]
        
        gender_scores = result.get('gender', {})
        
        if not gender_scores:
            return {
                "status": "REVIEW",
                "reason": "Could not detect gender",
                "detected": None,
                "expected": profile_gender
            }
        
        # Get dominant gender
        detected_gender = max(gender_scores, key=gender_scores.get)
        confidence = gender_scores[detected_gender] / 100.0
        
        if detected_gender.lower() != profile_gender.lower():
            return {
                "status": "FAIL",
                "reason": f"Gender mismatch: detected {detected_gender}, profile says {profile_gender}",
                "detected": detected_gender,
                "expected": profile_gender,
                "confidence": confidence
            }
        
        return {
            "status": "PASS",
            "reason": f"Gender verified as {detected_gender}",
            "detected": detected_gender,
            "expected": profile_gender,
            "confidence": confidence
        }
        
    except Exception as e:
        print(f"[DeepFace] Gender detection error: {str(e)}")
        return {
            "status": "REVIEW",
            "reason": f"Gender detection failed: {str(e)}",
            "detected": None,
            "expected": profile_gender
        }


# ==================== INSIGHTFACE: GENDER VALIDATION ====================

def validate_gender_insightface(img_path: str, profile_gender: str, face_data: Dict = None) -> Dict:
    """Gender validation using InsightFace (fast but may be less accurate)"""
    try:
        if face_data is None:
            analysis = analyze_face_insightface(img_path)
            if analysis["error"]:
                return {
                    "status": "REVIEW",
                    "reason": f"Face analysis failed: {analysis['error']}",
                    "detected": None,
                    "expected": profile_gender
                }
            face_data = analysis["data"]
        
        gender_value = face_data.get("gender")
        
        if gender_value is None:
            return {
                "status": "REVIEW",
                "reason": "Gender detection not available",
                "detected": None,
                "expected": profile_gender
            }
        
        detected_gender = "Male" if gender_value == 1 else "Female"
        confidence = 0.85
        
        if detected_gender.lower() != profile_gender.lower():
            return {
                "status": "FAIL",
                "reason": f"Gender mismatch: detected {detected_gender}, profile says {profile_gender}",
                "detected": detected_gender,
                "expected": profile_gender,
                "confidence": confidence
            }
        
        return {
            "status": "PASS",
            "reason": f"Gender verified as {detected_gender}",
            "detected": detected_gender,
            "expected": profile_gender,
            "confidence": confidence
        }
        
    except Exception as e:
        return {
            "status": "REVIEW",
            "reason": f"Gender detection failed: {str(e)}",
            "detected": None,
            "expected": profile_gender
        }


# ==================== DATABASE CHECKS REMOVED ====================
# Fraud database, celebrity database, and duplicate detection checks
# have been removed as they require database support not available in this phase


# ==================== INSIGHTFACE: FACE COVERAGE CHECK ====================

def check_face_coverage(img_path: str, face_data: Dict = None) -> Dict:
    """Face coverage check using InsightFace"""
    try:
        if face_data is None:
            analysis = analyze_face_insightface(img_path)
            if analysis["error"]:
                return {
                    "status": "REVIEW",
                    "reason": f"Face coverage check failed: {analysis['error']}"
                }
            face_data = analysis["data"]
        
        bbox = face_data.get("bbox")
        if bbox is None:
            return {
                "status": "REVIEW",
                "reason": "Could not get face bounding box"
            }
        
        face_x, face_y, face_x2, face_y2 = bbox
        face_w = face_x2 - face_x
        face_h = face_y2 - face_y
        
        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]
        
        face_area = face_w * face_h
        img_area = img_w * img_h
        coverage = face_area / img_area if img_area > 0 else 0
        
        if coverage < MIN_FACE_COVERAGE_S2:
            return {
                "status": "FAIL",
                "reason": f"Face too small in frame ({coverage:.2%} coverage)",
                "coverage": coverage
            }
        
        face_center_x = face_x + face_w / 2
        face_center_y = face_y + face_h / 2
        img_center_x = img_w / 2
        img_center_y = img_h / 2
        
        offset_x = abs(face_center_x - img_center_x) / img_w
        offset_y = abs(face_center_y - img_center_y) / img_h
        
        if offset_x > 0.3 or offset_y > 0.3:
            return {
                "status": "REVIEW",
                "reason": f"Face not centered. May indicate improper framing.",
                "coverage": coverage,
                "offset_x": offset_x,
                "offset_y": offset_y
            }
        
        return {
            "status": "PASS",
            "reason": f"Proper face framing ({coverage:.2%} coverage)",
            "coverage": coverage
        }
        
    except Exception as e:
        return {
            "status": "REVIEW",
            "reason": f"Face coverage check failed: {str(e)}"
        }




# ==================== REMAINING CHECKS (OPENCV-BASED) ====================

def detect_digital_enhancement(img_path: str) -> Dict:
    """Enhancement detection using OpenCV"""
    img = cv2.imread(img_path)
    
    checks = {}
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].mean()
    saturation_std = hsv[:, :, 1].std()
    
    if saturation > 150 and saturation_std < 40:
        checks["saturation"] = {
            "status": "FAIL",
            "reason": f"Unnaturally high saturation detected (filter applied)"
        }
    else:
        checks["saturation"] = {
            "status": "PASS",
            "reason": "Natural color saturation"
        }
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_ratio = np.count_nonzero(edges) / edges.size
    
    if edge_ratio > 0.20:
        checks["cartoon"] = {
            "status": "REVIEW",
            "reason": f"Possible cartoon/anime filter (high edge ratio: {edge_ratio:.3f})"
        }
    else:
        checks["cartoon"] = {
            "status": "PASS",
            "reason": "Natural photograph"
        }
    
    return checks


def detect_photo_of_photo(img_path: str) -> Dict:
    """Photo-of-photo detection using OpenCV"""
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    h, w = gray.shape
    border_width = int(min(h, w) * 0.05)
    
    top_border = gray[:border_width, :].mean()
    bottom_border = gray[-border_width:, :].mean()
    left_border = gray[:, :border_width].mean()
    right_border = gray[:, -border_width:].mean()
    
    border_mean = np.mean([top_border, bottom_border, left_border, right_border])
    
    edges = cv2.Canny(gray, 50, 150)
    
    if border_mean > PAPER_WHITE_THRESHOLD:
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > (h * w * 0.3):
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                if len(approx) == 4:
                    return {
                        "status": "FAIL",
                        "reason": "Photo of printed photo detected"
                    }
    
    return {
        "status": "PASS",
        "reason": "Original digital photo"
    }


def detect_ai_generated(img_path: str) -> Dict:
    """Enhanced AI-generated/cartoon/gibberish image detection"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return {
                "status": "REVIEW",
                "reason": "Could not read image for AI detection",
                "confidence": "LOW",
                "details": {}
            }
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        issues = []
        scores = {}
        
        # Texture Variance (RELAXED - was 80, now 50)
        kernel_size = 3
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
        mean = cv2.filter2D(gray, -1, kernel)
        sqr_mean = cv2.filter2D(gray ** 2, -1, kernel)
        variance = sqr_mean - mean ** 2
        avg_variance = variance.mean()
        scores["texture_variance"] = avg_variance

        if avg_variance < 50:
            issues.append(f"Very low texture variance ({avg_variance:.1f})")

        # Color Saturation (RELAXED - was 140/40, now 160/30)
        saturation = hsv[:, :, 1]
        avg_saturation = saturation.mean()
        saturation_std = saturation.std()
        scores["avg_saturation"] = avg_saturation
        scores["saturation_std"] = saturation_std

        if avg_saturation > 160 and saturation_std < 30:
            issues.append(f"Cartoon-like colors detected")

        # Edge Sharpness Consistency (RELAXED - was 0.08, now 0.12)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.count_nonzero(edges) / edges.size
        scores["edge_density"] = edge_density

        h, w = edges.shape
        block_size = h // 4
        edge_densities = []

        for i in range(4):
            for j in range(4):
                block = edges[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                if block.size > 0:
                    block_density = np.count_nonzero(block) / block.size
                    edge_densities.append(block_density)

        if len(edge_densities) > 0:
            edge_consistency = np.std(edge_densities)
            scores["edge_consistency"] = edge_consistency

            if edge_consistency > 0.12:
                issues.append(f"Inconsistent edge sharpness")

        # Color Distribution (RELAXED - was 15, now 10)
        b, g, r = cv2.split(img)

        hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
        hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])

        def count_peaks(hist, threshold=0.02):
            max_val = hist.max()
            peaks = 0
            for i in range(1, len(hist) - 1):
                if hist[i] > threshold * max_val:
                    if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                        peaks += 1
            return peaks

        total_peaks = count_peaks(hist_b) + count_peaks(hist_g) + count_peaks(hist_r)
        scores["color_peaks"] = total_peaks

        if total_peaks < 10:
            issues.append(f"Limited color palette")

        # Smoothness Analysis (RELAXED - was 150, now 40)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = laplacian.var()
        scores["laplacian_variance"] = laplacian_var

        if laplacian_var < 40:
            issues.append(f"Image too smooth")
        
        scores["total_issues"] = len(issues)
        
        if len(issues) >= 3:
            return {
                "status": "FAIL",
                "reason": "Image appears to be AI-generated, cartoon, or heavily filtered",
                "confidence": "HIGH",
                "details": {
                    "issues": issues,
                    "scores": scores
                }
            }
        elif len(issues) == 2:
            return {
                "status": "REVIEW",
                "reason": "Possible AI-generated or heavily filtered image",
                "confidence": "MEDIUM",
                "details": {
                    "issues": issues,
                    "scores": scores
                }
            }
        elif len(issues) == 1:
            return {
                "status": "REVIEW",
                "reason": f"Potential image quality concern: {issues[0]}",
                "confidence": "LOW",
                "details": {
                    "issues": issues,
                    "scores": scores
                }
            }
        else:
            return {
                "status": "PASS",
                "reason": "Image appears to be authentic photograph",
                "confidence": "HIGH",
                "details": {
                    "scores": scores
                }
            }
    
    except Exception as e:
        return {
            "status": "REVIEW",
            "reason": f"AI detection error: {str(e)}",
            "confidence": "LOW",
            "details": {"error": str(e)}
        }


# ==================== STAGE 2 MAIN VALIDATOR (HYBRID) ====================

def stage2_validate_hybrid(
    image_path: str,
    profile_data: Dict,
    photo_type: str = "PRIMARY",
    use_deepface_gender: bool = False
) -> Dict:
    """
    Stage 2 validation with HYBRID approach:
    - InsightFace as backbone (detection, embeddings, face coverage)
    - DeepFace for age and ethnicity (PRIMARY only)
    - Optional: DeepFace for gender if InsightFace is unreliable

    Note: Fraud database, celebrity database, and duplicate detection checks
    have been removed as they require database support not available in this phase.
    """

    results = {
        "stage": 2,
        "photo_type": photo_type,
        "matri_id": profile_data.get("matri_id"),
        "gpu_used": GPU_AVAILABLE,
        "library_usage": {
            "insightface": ["detection", "embeddings", "face_coverage"],
            "deepface": []
        },
        "checks": {},
        "checks_performed": [],
        "checks_skipped": [],
        "final_decision": None,
        "action": None,
        "reason": None,
        "early_exit": False
    }
    
    # ============= EARLY EXIT FOR SECONDARY PHOTOS =============
    # Secondary photos already completed all necessary checks in Stage 1:
    # - NSFW check
    # - Face detection and quality
    # - Face matching with primary photo
    # - Face coverage check (for individual photos only, group photos skip this)
    if photo_type == "SECONDARY":
        print("[Stage 2] SECONDARY photo detected - all checks completed in Stage 1")
        results["final_decision"] = "APPROVE"
        results["action"] = "PUBLISH"
        results["reason"] = "SECONDARY photo validation completed in Stage 1"
        results["early_exit"] = True
        results["checks_skipped"] = ["age", "gender", "ethnicity", "face_coverage",
                                     "enhancement", "photo_of_photo", "ai_generated"]
        return results

    # ============= INSIGHTFACE ANALYSIS (BACKBONE) =============
    face_data = None

    if photo_type == "PRIMARY":
        print("[InsightFace] Running face analysis (BACKBONE)...")
        analysis = analyze_face_insightface(image_path)
        face_data = analysis["data"] if not analysis["error"] else None

        if analysis["error"]:
            results["final_decision"] = "REVIEW"
            results["action"] = "SEND_TO_HUMAN"
            results["reason"] = f"Face detection failed: {analysis['error']}"
            results["early_exit"] = True
            return results
    
    # ============= PRIORITY 1: CRITICAL CHECKS =============
    
    if photo_type == "PRIMARY":
        # 1. AGE CHECK (DEEPFACE - PRIMARY ONLY)
        print("[P1] Checking age with DeepFace...")
        results["checks"]["age"] = validate_age_deepface(image_path, profile_data.get("age", 25))
        results["checks_performed"].append("age")
        results["library_usage"]["deepface"].append("age")
        
        if results["checks"]["age"]["status"] == "FAIL" and results["checks"]["age"].get("action") == "SUSPEND":
            results["final_decision"] = "SUSPEND"
            results["action"] = "SUSPEND_PROFILE"
            results["reason"] = "Underage detected - immediate suspension"
            results["early_exit"] = True
            results["checks_skipped"] = ["gender", "ethnicity", "face_coverage", "enhancement",
                                         "photo_of_photo", "ai_generated"]
            return results
    else:
        results["checks_skipped"].append("age")
        print("[P1] Skipping age check for SECONDARY photo")
    
    # ============= PRIORITY 2: HIGH IMPORTANCE CHECKS =============
    
    if photo_type == "PRIMARY":
        # 3. GENDER CHECK (INSIGHTFACE or DEEPFACE)
        if use_deepface_gender:
            print("[P2] Checking gender with DeepFace...")
            results["checks"]["gender"] = validate_gender_deepface(image_path, profile_data.get("gender", "Unknown"))
            results["library_usage"]["deepface"].append("gender")
        else:
            print("[P2] Checking gender with InsightFace...")
            results["checks"]["gender"] = validate_gender_insightface(image_path, profile_data.get("gender", "Unknown"), face_data)
        
        results["checks_performed"].append("gender")
        
        if results["checks"]["gender"]["status"] == "FAIL":
            results["final_decision"] = "REJECT"
            results["action"] = "SELFIE_VERIFICATION"
            results["reason"] = "Gender mismatch detected"
            results["early_exit"] = True
            results["checks_skipped"].extend(["ethnicity", "face_coverage", "enhancement",
                                         "photo_of_photo", "ai_generated"])
            return results

        # 4. ETHNICITY CHECK (DEEPFACE - PRIMARY ONLY)
        print("[P2] Checking ethnicity with DeepFace...")
        results["checks"]["ethnicity"] = validate_ethnicity_deepface(image_path)
        results["checks_performed"].append("ethnicity")
        results["library_usage"]["deepface"].append("ethnicity")

        if results["checks"]["ethnicity"]["status"] == "FAIL":
            results["final_decision"] = "REJECT"
            results["action"] = "SELFIE_VERIFICATION"
            results["reason"] = "Ethnicity check failed"
            results["early_exit"] = True
            results["checks_skipped"].extend(["face_coverage", "enhancement",
                                         "photo_of_photo", "ai_generated"])
            return results
    else:
        results["checks_skipped"].extend(["gender", "ethnicity"])
        print("[P2] Skipping gender/ethnicity checks for SECONDARY photo")
    
    # ============= PRIORITY 3: STANDARD CHECKS =============
    
    print("[P3] Running standard checks...")
    
    if photo_type == "PRIMARY":
        # 5. Face Coverage (INSIGHTFACE)
        results["checks"]["face_coverage"] = check_face_coverage(image_path, face_data)
        results["checks_performed"].append("face_coverage")
    else:
        results["checks_skipped"].append("face_coverage")
        print("[P3] Skipping face coverage check for SECONDARY photo")

    # 6. Digital Enhancement (OPENCV)
    results["checks"]["enhancement"] = detect_digital_enhancement(image_path)
    results["checks_performed"].append("enhancement")

    # 7. Photo-of-photo (OPENCV)
    results["checks"]["photo_of_photo"] = detect_photo_of_photo(image_path)
    results["checks_performed"].append("photo_of_photo")

    # 8. AI-generated (OPENCV)
    results["checks"]["ai_generated"] = detect_ai_generated(image_path)
    results["checks_performed"].append("ai_generated")
    
    # ============= FINAL DECISION LOGIC =============
    
    fail_checks = []
    review_checks = []
    
    for check_name, check_result in results["checks"].items():
        if isinstance(check_result, dict) and "status" in check_result:
            if check_result["status"] == "FAIL":
                fail_checks.append(check_name)
            elif check_result["status"] == "REVIEW":
                review_checks.append(check_name)
        else:
            for sub_check, sub_result in check_result.items():
                if sub_result["status"] == "FAIL":
                    fail_checks.append(f"{check_name}.{sub_check}")
                elif sub_result["status"] == "REVIEW":
                    review_checks.append(f"{check_name}.{sub_check}")
    
    if fail_checks:
        results["final_decision"] = "REJECT"
        results["action"] = determine_rejection_action(fail_checks, results["checks"])
        results["reason"] = f"Failed checks: {', '.join(fail_checks)}"
    elif review_checks:
        results["final_decision"] = "MANUAL_REVIEW"
        results["action"] = "SEND_TO_HUMAN"
        results["reason"] = f"Requires manual review: {', '.join(review_checks)}"
    else:
        results["final_decision"] = "APPROVE"
        results["action"] = "PUBLISH"
        results["reason"] = "All checks passed"
    
    return results


def determine_rejection_action(fail_checks: List[str], all_checks: Dict) -> str:
    """Determine action based on failed checks"""

    if any(check in fail_checks for check in ["age"]):
        return "SUSPEND_PROFILE"

    if any(check in fail_checks for check in ["gender", "ethnicity"]):
        return "SELFIE_VERIFICATION"

    if "enhancement" in fail_checks or any("enhancement" in check for check in fail_checks):
        return "NUDGE_UPLOAD_ORIGINAL"

    if "photo_of_photo" in fail_checks:
        return "NUDGE_UPLOAD_DIGITAL"

    return "NUDGE_REUPLOAD_PROPER"


def compile_checklist_summary(stage1_result: Dict, stage2_result: Optional[Dict], photo_type: str) -> Dict:
    """Compile a comprehensive 20-point checklist summary"""
    checklist = {
        "total_checks": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "review": 0,
        "checks": []
    }
    
    # Stage 1 checks
    stage1_checks_config = [
        {"id": 1, "name": "Format Validation", "stage": "S1", "check_key": "format"},
        {"id": 2, "name": "Image Readable", "stage": "S1", "check_key": "image_read"},
        {"id": 3, "name": "Face Detection", "stage": "S1", "check_key": "face_count"},
        {"id": 4, "name": "Auto-Cropping", "stage": "S1", "check_key": "cropping_applied"},
        {"id": 5, "name": "Face Size", "stage": "S1", "check_key": "face_size"},
        {"id": 6, "name": "Resolution", "stage": "S1", "check_key": "resolution"},
        {"id": 7, "name": "Blur Detection", "stage": "S1", "check_key": "blur"},
        {"id": 8, "name": "Orientation", "stage": "S1", "check_key": "orientation"},
        {"id": 9, "name": "Face Cover/Mask", "stage": "S1", "check_key": "face_cover"},
        {"id": 10, "name": "NSFW Content", "stage": "S1", "check_key": "nsfw"}
    ]
    
    for check_config in stage1_checks_config:
        check_id = check_config["id"]
        check_name = check_config["name"]
        check_key = check_config["check_key"]
        
        if check_key == "cropping_applied":
            if photo_type == "SECONDARY":
                checklist["checks"].append({
                    "id": check_id,
                    "name": check_name,
                    "stage": "S1",
                    "status": "SKIPPED",
                    "reason": "SECONDARY photos are not auto-cropped",
                    "details": None
                })
                checklist["skipped"] += 1
            else:
                cropping_applied = stage1_result["checks"].get("cropping_applied", "NO")
                if cropping_applied == "YES":
                    checklist["checks"].append({
                        "id": check_id,
                        "name": check_name,
                        "stage": "S1",
                        "status": "APPLIED",
                        "reason": "Face coverage was low, image auto-cropped",
                        "details": {
                            "original_coverage": stage1_result["checks"].get("face_coverage_original"),
                            "after_crop": stage1_result["checks"].get("face_coverage_after_crop")
                        }
                    })
                    checklist["passed"] += 1
                else:
                    checklist["checks"].append({
                        "id": check_id,
                        "name": check_name,
                        "stage": "S1",
                        "status": "NOT_NEEDED",
                        "reason": "Face coverage already sufficient",
                        "details": {
                            "coverage": stage1_result["checks"].get("face_coverage_original")
                        }
                    })
                    checklist["passed"] += 1
        else:
            check_value = stage1_result["checks"].get(check_key, "UNKNOWN")
            
            if check_value == "PASS":
                status = "PASS"
                checklist["passed"] += 1
            elif "face(s) detected" in str(check_value):
                status = "PASS"
                checklist["passed"] += 1
            else:
                status = "INFO"
                checklist["passed"] += 1
            
            checklist["checks"].append({
                "id": check_id,
                "name": check_name,
                "stage": "S1",
                "status": status,
                "reason": None,
                "details": check_value
            })
    
    # Stage 2 checks
    if stage2_result:
        stage2_checks_config = [
            {"id": 11, "name": "Age Verification (DeepFace)", "check_key": "age"},
            {"id": 12, "name": "Gender Validation", "check_key": "gender"},
            {"id": 13, "name": "Ethnicity Validation (DeepFace)", "check_key": "ethnicity"},
            {"id": 14, "name": "Face Coverage (InsightFace)", "check_key": "face_coverage"},
            {"id": 15, "name": "Digital Enhancement", "check_key": "enhancement"},
            {"id": 16, "name": "Photo-of-Photo", "check_key": "photo_of_photo"},
            {"id": 17, "name": "AI-Generated", "check_key": "ai_generated"}
        ]
        
        performed = stage2_result.get("checks_performed", [])
        skipped = stage2_result.get("checks_skipped", [])
        
        for check_config in stage2_checks_config:
            check_id = check_config["id"]
            check_name = check_config["name"]
            check_key = check_config["check_key"]
            
            if check_key in skipped:
                skip_reasons = {
                    "age": "SECONDARY photos skip age check (family members have different ages)",
                    "gender": "SECONDARY photos skip gender check (family has both genders)",
                    "ethnicity": "SECONDARY photos skip ethnicity check (family members may differ)",
                    "face_coverage": "SECONDARY photos skip face coverage (group photos have smaller faces)"
                }
                
                checklist["checks"].append({
                    "id": check_id,
                    "name": check_name,
                    "stage": "S2",
                    "status": "SKIPPED",
                    "reason": skip_reasons.get(check_key, f"Skipped for {photo_type} photos"),
                    "details": None
                })
                checklist["skipped"] += 1
                
            elif check_key in performed:
                check_result = stage2_result["checks"].get(check_key, {})
                
                if isinstance(check_result, dict) and "status" not in check_result:
                    all_statuses = []
                    details = {}
                    
                    for sub_key, sub_result in check_result.items():
                        if isinstance(sub_result, dict) and "status" in sub_result:
                            all_statuses.append(sub_result["status"])
                            details[sub_key] = {
                                "status": sub_result["status"],
                                "reason": sub_result.get("reason")
                            }
                    
                    if "FAIL" in all_statuses:
                        status = "FAIL"
                        checklist["failed"] += 1
                    elif "REVIEW" in all_statuses:
                        status = "REVIEW"
                        checklist["review"] += 1
                    else:
                        status = "PASS"
                        checklist["passed"] += 1
                    
                    checklist["checks"].append({
                        "id": check_id,
                        "name": check_name,
                        "stage": "S2",
                        "status": status,
                        "reason": check_result.get("reason"),
                        "details": details
                    })
                else:
                    status = check_result.get("status", "UNKNOWN")
                    
                    if status == "PASS":
                        checklist["passed"] += 1
                    elif status == "FAIL":
                        checklist["failed"] += 1
                    elif status == "REVIEW":
                        checklist["review"] += 1
                    
                    checklist["checks"].append({
                        "id": check_id,
                        "name": check_name,
                        "stage": "S2",
                        "status": status,
                        "reason": check_result.get("reason"),
                        "details": {
                            k: v for k, v in check_result.items() 
                            if k not in ["status", "reason"]
                        } if check_result else None
                    })
    
    checklist["total_checks"] = len(checklist["checks"])
    
    return checklist


# ==================== COMBINED VALIDATION PIPELINE (HYBRID) ====================

def validate_photo_complete_hybrid(
    image_path: str,
    photo_type: str = "PRIMARY",
    profile_data: Dict = None,
    reference_photo_path: Optional[str] = None,
    run_stage2: bool = True,
    use_deepface_gender: bool = False
) -> Dict:
    """
    Complete photo validation pipeline with HYBRID approach:
    - InsightFace as backbone
    - DeepFace for age and ethnicity (PRIMARY only)

    Note: Fraud database, celebrity database, and duplicate detection checks
    have been removed as they require database support not available in this phase.
    """
    
    print("\n" + "="*70)
    print("STARTING HYBRID PHOTO VALIDATION PIPELINE")
    print("InsightFace (Backbone) + DeepFace (Age/Ethnicity)")
    print("="*70)
    
    results = {
        "image_path": image_path,
        "photo_type": photo_type,
        "stage1": None,
        "stage2": None,
        "final_decision": None,
        "final_action": None,
        "final_reason": None
    }
    
    # ============= STAGE 1 VALIDATION =============
    print(f"\n[STAGE 1] Running basic quality checks for {photo_type} photo...")
    stage1_result = stage1_validate(image_path, photo_type, reference_photo_path)
    results["stage1"] = stage1_result
    
    if stage1_result["result"] == "REJECT":
        print(f"[STAGE 1] ❌ REJECTED: {stage1_result['reason']}")
        results["final_decision"] = "REJECT"
        results["final_action"] = "REJECT_PHOTO"
        results["final_reason"] = f"Stage 1 failure: {stage1_result['reason']}"
        return results
    
    print("[STAGE 1] ✅ PASSED")
    
    # Handle cropped image
    validation_image_path = image_path
    cropped_image_array = None
    
    if stage1_result.get("cropped_image") is not None:
        cropped_image_array = stage1_result["cropped_image"]
        cropped_path = image_path.replace(".", "_cropped_final.")
        cv2.imwrite(cropped_path, cropped_image_array)
        print(f"[STAGE 1] Cropped image saved: {cropped_path}")
        results["cropped_image_path"] = cropped_path
        results["image_was_cropped"] = True
        validation_image_path = cropped_path
    else:
        results["image_was_cropped"] = False
    
    # ============= STAGE 2 VALIDATION (HYBRID) =============
    if run_stage2:
        if profile_data is None:
            print("[STAGE 2] ⚠️  Skipping - No profile data provided")
            results["final_decision"] = "PASS_STAGE1_ONLY"
            results["final_action"] = "MANUAL_REVIEW"
            results["final_reason"] = "Stage 1 passed, Stage 2 skipped (no profile data)"
            return results
        
        print("\n[STAGE 2] Running HYBRID validation...")
        print("[STAGE 2] InsightFace: detection, embeddings, matching")
        if photo_type == "PRIMARY":
            print("[STAGE 2] DeepFace: age, ethnicity")
        print(f"[STAGE 2] Validating image: {validation_image_path}")
        
        stage2_result = stage2_validate_hybrid(
            image_path=validation_image_path,
            profile_data=profile_data,
            photo_type=photo_type,
            use_deepface_gender=use_deepface_gender
        )
        results["stage2"] = stage2_result
        
        results["final_decision"] = stage2_result["final_decision"]
        results["final_action"] = stage2_result["action"]
        results["final_reason"] = stage2_result["reason"]
        
        # Compile checklist
        results["checklist_summary"] = compile_checklist_summary(
            stage1_result, 
            stage2_result, 
            photo_type
        )
        
        # Print library usage summary
        print(f"\n[STAGE 2] Library Usage Summary:")
        print(f"  InsightFace: {', '.join(stage2_result['library_usage']['insightface'])}")
        if stage2_result['library_usage']['deepface']:
            print(f"  DeepFace: {', '.join(stage2_result['library_usage']['deepface'])}")
        
        if stage2_result["final_decision"] == "SUSPEND":
            print(f"[STAGE 2] 🚨 SUSPEND: {stage2_result['reason']}")
        elif stage2_result["final_decision"] == "REJECT":
            print(f"[STAGE 2] ❌ REJECT: {stage2_result['reason']}")
        elif stage2_result["final_decision"] == "MANUAL_REVIEW":
            print(f"[STAGE 2] ⚠️  MANUAL REVIEW: {stage2_result['reason']}")
        else:
            print(f"[STAGE 2] ✅ APPROVED: {stage2_result['reason']}")
            
            if results.get("image_was_cropped") and cropped_image_array is not None:
                print("[STAGE 2] Converting cropped image to base64...")
                cropped_base64 = image_to_base64(cropped_image_array)
                if cropped_base64:
                    results["cropped_image_base64"] = cropped_base64
                    print("[STAGE 2] ✅ Cropped image base64 ready")
    else:
        results["final_decision"] = "PASS_STAGE1_ONLY"
        results["final_action"] = "PUBLISH"
        results["final_reason"] = "Stage 1 passed, Stage 2 not requested"
        
        results["checklist_summary"] = compile_checklist_summary(
            stage1_result, 
            None,
            photo_type
        )
    
    return results


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("EXAMPLE: HYBRID PHOTO VALIDATION")
    print("="*70)
    
    profile_data = {
        "matri_id": "BM123456",
        "gender": "Male",
        "age": 28
    }
    
    result = validate_photo_complete_hybrid(
        image_path="test_image.jpg",
        photo_type="PRIMARY",
        profile_data=profile_data,
        run_stage2=True,
        use_deepface_gender=False  # Use InsightFace for gender (faster)
    )
    
    print(f"\nFinal Decision: {result['final_decision']}")
    print(f"Final Action: {result['final_action']}")
    print(f"Reason: {result['final_reason']}")
    
    if result.get('checklist_summary'):
        checklist = result['checklist_summary']
        print(f"\n{'='*70}")
        print("CHECKLIST SUMMARY")
        print(f"{'='*70}")
        print(f"Total: {checklist['total_checks']}, Passed: {checklist['passed']}, Failed: {checklist['failed']}")
        print(f"Skipped: {checklist['skipped']}, Review: {checklist['review']}")
    
    if result.get('stage2'):
        print(f"\n{'='*70}")
        print("LIBRARY USAGE")
        print(f"{'='*70}")
        print(f"InsightFace: {', '.join(result['stage2']['library_usage']['insightface'])}")
        if result['stage2']['library_usage']['deepface']:
            print(f"DeepFace: {', '.join(result['stage2']['library_usage']['deepface'])}")
