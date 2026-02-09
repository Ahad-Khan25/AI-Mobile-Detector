# Prototype Version(My Version)
# import cv2
# import numpy as np
#
# class VideoProcessor:
#     def __init__(self, width=640, height=480):
#         self.width = width
#         self.height = height
#
#     # VIDEO CAPTURE
#     def capture_video(self, source=0):
#         """source = 0 Opens webcam or video file a filename can also be passed."""
#         cap = cv2.VideoCapture(source) #.VideoCapture is cv2 function that tries to open the device/file.
#         if not cap.isOpened():
#             raise Exception("Video source not available!")
#         return cap #cap here is a cv2.VideoCapture object used later to read frames
#
#     # BASIC PROCESSING
#     def resize_frame(self, img):
#         """Resizes frame to consistent dimensions."""
#         return cv2.resize(img, (self.width, self.height)) #.resize is cv2 function that resizes frame to the height and width specified above
#
#     def convert_to_rgb(self, img):
#         """Convert BGR(BGR is how OpenCV reads images) (OpenCV default) to RGB(ML models expect RGB)."""
#         return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#     def normalize(self, img):
#         """Normalizes pixel values to 0–1."""
#         return img.astype(np.float32) / 255.0 #this function will convert the image array and scale pixel values form[0,255] to [0.0,1.0]
#
#     # ENHANCEMENTS
#     def adjust_brightness_contrast(self, img, brightness=20, contrast=30):
#         """Adjust brightness and contrast."""
#         alpha = 1 + contrast / 100  # contrast gain
#         beta = brightness           # brightness shift
#         return cv2.convertScaleAbs(img, alpha=alpha, beta=beta) #alpha multiplies pixel values(contrast), beta adds to them(brightness). cv2.convertScaleAbs performs dst = saturate(|alpha*src + beta|) and casts to uint8
#
#     def denoise(self, img):
#         """Apply Gaussian smoothing."""
#         return cv2.GaussianBlur(img, (5, 5), 0) #Useful for noisy cameras. Noise here doesn't mean sound but the extra objects like unwanted pixels or distortions
#
#     def sharpen(self, img):
#         """Sharpen edges."""
#         kernel = np.array([[0, -1, 0],
#                            [-1, 5, -1],
#                            [0, -1, 0]])
#         return cv2.filter2D(img, -1, kernel)
#
#     # FULL PIPELINE
#     def preprocess(self, img):
#         """Full preprocessing pipeline."""
#         img = self.resize_frame(img)
#         img = self.convert_to_rgb(img)
#         img = self.normalize(img)
#         return img
#
#
# # ------------------------------
# # TESTING PIPELINE (OPTIONAL)
# # ------------------------------
# if __name__ == "__main__":
#     processor = VideoProcessor()
#
#     cap = processor.capture_video(0)
#
#     while True:
#         ret, frame_original = cap.read() #ret is boolean, frame_original is a NumPy array shape(H,W,3) with dtype uint8 in BGR
#         if not ret:
#             break
#
#         processed = processor.preprocess(frame_original)
#
#         # Convert processed image back for OpenCV viewing
#         display_img = (processed * 255).astype(np.uint8)
#         display_img = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
#
#         cv2.imshow("Original", frame_original)
#         cv2.imshow("Processed", display_img)
#
#         if cv2.waitKey(1) & 0xFF == ord('0'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
# """Documentation
# This code opens a webcam/video file, resizes and normalizes frames, converts color spaces, offers simple image
# enhancements(brightness.contrast, denoise, sharpen), and includes a small test loop when run as __main__.
# It uses two libraries cv2(OpenCV)--image/video I/O and image processing functions.
# numpy numerical arrays and kernels for filtering
# import cv2
# OpenCV here is being used for reading frames, resizing images(cv2.resize), color conversion(cv2.cvtColor),
# value scaling(cv2.convertScaleAbs), blurring(cv2.GaussianBlur), filtering(cv2.filter2D), showing window(cv2.imshow,
# cv2.waitKey), and cleaning up (cv2.destroyAllWindows). OpenCV functions expect/return images as NumPy arrays with dtype
# usually uint8 and channel order BGR.
#
# import numpy as np
# NumPy provides n-dimensional arrays and numerical operations. Used here for creating file kernels(np.array), type conversion
# and arithmetic in normalize, array operations when converting back to uint8 to display.
#
# cv2 functions operate on NumPy arrays, That tight integration is why these two are used together.
# """

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# # Rafay Version
# import cv2
# import numpy as np
#
#
# class VideoProcessor:
#     """
#     Module 9: Video Capture & Frame Preprocessing
#     ------------------------------------------------
#     This module handles:
#         - Reading video frames from webcam or video file
#         - Resizing frames to a fixed resolution
#         - Converting BGR → RGB (required by ML models such as YOLO / MediaPipe)
#         - Normalizing pixel values to [0,1] for neural networks
#         - Simple enhancement utilities (brightness/contrast, denoising, sharpening)
#     """
#
#     def __init__(self, width=640, height=480):
#         """
#         Initialize the processor with a fixed output size.
#
#         width, height: Output dimensions for all processed frames.
#         Using consistent resolution helps performance and ensures
#         that detection models behave predictably.
#         """
#         self.width = width
#         self.height = height
#
#     # ------------------------------------------------------
#     # 1) VIDEO CAPTURE
#     # ------------------------------------------------------
#     def capture_video(self, source=0):
#         """
#         Opens a webcam or video file.
#
#         Parameters:
#             source:
#                 0  default webcam
#                 "video.mp4"  video file path
#
#         Returns:
#             cv2.VideoCapture object
#
#         Notes:
#             VideoCapture tries to access the camera/file.
#             If it fails, cap.isOpened() will be False.
#         """
#         cap = cv2.VideoCapture(source)
#
#         if not cap.isOpened():
#             raise Exception("Error: Video source not available!")
#
#         return cap
#
#     # ------------------------------------------------------
#     # 2) BASIC PREPROCESSING OPERATIONS
#     # ------------------------------------------------------
#     def resize_frame(self, img):
#         """
#         Resizes the frame to (width x height).
#         A fixed size ensures ML models always get the input shape they expect.
#         """
#         return cv2.resize(img, (self.width, self.height))
#
#     def convert_to_rgb(self, img):
#         """
#         Converts BGR to RGB.
#
#         WHY?
#             OpenCV loads images as BGR.
#             Deep learning frameworks (PyTorch, MediaPipe, TensorFlow, YOLO)
#             expect RGB. If not converted, model accuracy drops.
#         """
#         return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#     def normalize(self, img):
#         """
#         Converts pixel values from [0,255] to [0,1].
#
#         WHY NORMALIZE?
#             Neural networks converge and perform better when inputs have
#             small, consistent numeric ranges.
#
#         Returns:
#             Float32 normalized image
#         """
#         return img.astype(np.float32) / 255.0
#
#     # ------------------------------------------------------
#     # 3) OPTIONAL ENHANCEMENT UTILITIES
#     # ------------------------------------------------------
#     def adjust_brightness_contrast(self, img, brightness=20, contrast=30):
#         """
#         Adjusts brightness and contrast using:
#             new_pixel = alpha * pixel + beta
#
#         alpha (contrast): multiplier
#         beta  (brightness): addition
#
#         cv2.convertScaleAbs handles saturation and type conversion.
#         """
#         alpha = 1 + (contrast / 100)
#         beta = brightness
#         return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
#
#     def denoise(self, img):
#         """
#         Applies Gaussian blur to reduce noise.
#
#         WHY?
#             Some webcams produce grainy images.
#             Denoising helps detection models perform better.
#         """
#         return cv2.GaussianBlur(img, (5, 5), 0)
#
#     def sharpen(self, img):
#         """
#         Sharpens the image by enhancing edges.
#         Useful when the image appears too soft.
#         """
#         kernel = np.array([
#             [0, -1, 0],
#             [-1, 5, -1],
#             [0, -1, 0]
#         ])
#         return cv2.filter2D(img, -1, kernel)
#
#     # ------------------------------------------------------
#     # 4) FULL PREPROCESSING PIPELINE
#     # ------------------------------------------------------
#     def preprocess(self, img):
#         """
#         Applies the full preprocessing pipeline in the correct order:
#             1. Resize
#             2. Convert to RGB
#             3. Normalize to [0,1]
#
#         This is what will be sent to all higher-level modules.
#         """
#         img = self.resize_frame(img)
#         img = self.convert_to_rgb(img)
#         img = self.normalize(img)
#         return img
#
#
# # =====================================================================
# # OPTIONAL TESTING SECTION
# # =====================================================================
# if __name__ == "__main__":
#     """
#     This block only runs when executing the file directly.
#     It will NOT run when imported as a module.
#
#     Purpose:
#         Capture video from webcam
#         Show original + preprocessed frames
#         Allows developer to verify Module 9 works correctly
#     """
#
#     processor = VideoProcessor()
#     cap = processor.capture_video(0)  # opens the webcam
#
#     while True:
#         ret, frame_original = cap.read()
#
#         if not ret:
#             print("Frame read error!")
#             break
#
#         processed = processor.preprocess(frame_original)
#
#         # Convert normalized RGB back to uint8 BGR for display
#         display_img = (processed * 255).astype(np.uint8)
#         display_img = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
#
#         cv2.imshow("Original Frame", frame_original)
#         cv2.imshow("Processed Frame", display_img)
#
#         # Press '0' to exit
#         if cv2.waitKey(1) & 0xFF == ord('0'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Rafay Version 2
import cv2
import numpy as np
import time
import threading


class VideoProcessor:
    """
    =============================================================================
                MODULE 9 – VIDEO CAPTURE & FRAME PREPROCESSING
    =============================================================================
    This module is responsible for:
        Capturing frames from webcam or video file
        Resizing them to a consistent dimension (important for ML models)
        Converting BGR → RGB (OpenCV uses BGR; ML models expect RGB)
        Normalizing pixel values to [0,1] for deep learning models
        Optional image enhancement: brightness/contrast, denoising, sharpening
        Thread-based frame capture (for smoother FPS)
        FPS measurement (useful for real-time systems)

    Every other module (student detection, phone detection, hand tracking,
    head-pose estimation, glow detection) depends on the preprocessed frames
    produced here.

    Providing a clean, normalized, and consistently sized frame ensures the
    entire AI pipeline remains stable and predictable.
    =============================================================================
    """

    def __init__(self,
                 width=640,
                 height=480,
                 threaded=True,
                 output_format="rgb_normalized"):

        self.width = width
        self.height = height
        self.threaded = threaded
        self.output_format = output_format

        # These will be used if threaded mode is enabled
        self.cap = None
        self.latest_frame = None
        self.running = False
        self.fps = 0

    # ==========================================================================
    #                          VIDEO CAPTURE METHODS
    # ==========================================================================
    def capture_video(self, source=0):
        self.cap = cv2.VideoCapture(source)

        # Safety check: Did OpenCV successfully open the webcam/file?
        if not self.cap.isOpened():
            raise Exception("Error: Video source not available!")

        # If threading enabled: start background frame grabbing thread
        if self.threaded:
            self.running = True
            threading.Thread(target=self._frame_reader, daemon=True).start()

        return self.cap

    def _frame_reader(self):
        prev_time = time.time()

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            self.latest_frame = frame

            # FPS calculation:
            now = time.time()
            self.fps = 1 / (now - prev_time)
            prev_time = now

    def read_frame(self):
        if self.threaded:
            return True, self.latest_frame
        else:
            return self.cap.read()

    # ==========================================================================
    #                          BASIC PREPROCESSING
    # ==========================================================================
    def resize_frame(self, img):
        return cv2.resize(img, (self.width, self.height))

    def convert_to_rgb(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def convert_to_gray(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def normalize(self, img):
        return img.astype(np.float32) / 255.0

    # ==========================================================================
    #                    OPTIONAL IMAGE ENHANCEMENT UTILITIES
    # ==========================================================================
    def adjust_brightness_contrast(self, img, brightness=20, contrast=30):
        alpha = 1 + (contrast / 100)
        beta = brightness
        return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    def denoise(self, img):
        return cv2.GaussianBlur(img, (5, 5), 0)

    def sharpen(self, img):

        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])
        return cv2.filter2D(img, -1, kernel)

    # ==========================================================================
    #                       FULL PREPROCESSING PIPELINE
    # ==========================================================================
    def preprocess(self, img):

        # 1. Resize first
        img = self.resize_frame(img)

        # 2. Optional enhancements (user can enable these as needed)
        # img = self.adjust_brightness_contrast(img)   # optional
        # img = self.denoise(img)                      # optional
        # img = self.sharpen(img)                      # optional

        # 3. Select desired final format
        if self.output_format == "gray":
            return self.convert_to_gray(img)

        if self.output_format == "rgb":
            return self.convert_to_rgb(img)

        if self.output_format == "rgb_normalized":
            img = self.convert_to_rgb(img)
            return self.normalize(img)

        # Default: return BGR (unchanged except resizing)
        return img

    # ==========================================================================
    #                             STOP CAPTURE
    # ==========================================================================
    def stop(self):
        self.running = False


# =============================================================================
#                      TESTING (ONLY RUNS IF FILE IS MAIN)
# =============================================================================
if __name__ == "__main__":
    """
    This section is for testing Module 9 independently.

    It:
        - Opens the webcam
        - Reads frames
        - Preprocesses them
        - Displays both original & processed frames
        - Shows FPS in threaded mode
    """

    processor = VideoProcessor(threaded=True, output_format="rgb_normalized")
    processor.capture_video(0)

    while True:
        ret, frame = processor.read_frame()

        if not ret or frame is None:
            continue

        processed = processor.preprocess(frame)

        # Convert normalized RGB back to BGR uint8 for OpenCV display
        display = (processed * 255).astype(np.uint8)
        display = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)

        # Show FPS
        cv2.putText(frame,
                    f"FPS: {processor.fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2)

        cv2.imshow("Original Frame", frame)
        cv2.imshow("Processed Frame", display)

        if cv2.waitKey(1) & 0xFF == ord('0'):
            break

    processor.stop()
    cv2.destroyAllWindows()
