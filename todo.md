# OpenCV Interactive Playground — TODO

## Platform Status

### Phase 1: Backend
- [x] Project folder structure
- [x] FastAPI app, SQLite, Pydantic schemas
- [x] Sandboxed code execution (subprocess)
- [x] Kata, execute, and auth API routes
- [x] Kata seeding from Markdown + YAML frontmatter

### Phase 2: Frontend
- [x] SolidJS + Vite scaffold
- [x] API client, sidebar, header, editor, output panel
- [x] Routing, dark/light theme, reactive sidebar active state
- [x] Reset output panel, code, and focus state on kata navigation
- [ ] Wire auth into frontend (login/register UI)
- [ ] Interactive demo sliders in demo-panel
- [ ] End-to-end verification: editor → run → see output

---

## Kata Roadmap (100 Katas)

Legend: `[x]` = implemented, `[ ]` = planned

### Beginner — Foundations (Katas 00–19)

Core building blocks. Every kata after this assumes these are mastered.

- [x] **00 — OpenCV Basics** — What is OpenCV, images as arrays, BGR, coordinate system, uint8
- [x] **01 — Image Loading & Display** — cv2.imread, np.zeros, img.shape, cv2.imshow
- [x] **02 — Color Spaces** — cv2.cvtColor, BGR, Grayscale, HSV, RGB
- [x] **03 — Pixel Access & Manipulation** — numpy indexing, ROI, array slicing, channel access
- [x] **04 — Drawing Lines & Rectangles** — cv2.line, cv2.rectangle, thickness, color
- [x] **05 — Drawing Circles & Ellipses** — cv2.circle, cv2.ellipse, filled vs outlined
- [x] **06 — Drawing Text** — cv2.putText, font faces, scale, baseline calculation
- [x] **07 — Image Resizing** — cv2.resize, fx/fy, INTER_LINEAR vs INTER_AREA vs INTER_CUBIC
- [x] **08 — Image Cropping** — NumPy slicing for cropping, aspect ratio preservation
- [x] **09 — Image Rotation** — cv2.getRotationMatrix2D, cv2.warpAffine, center rotation
- [x] **10 — Image Flipping** — cv2.flip (horizontal, vertical, both)
- [x] **11 — Image Translation** — cv2.warpAffine with translation matrix
- [x] **12 — Image Arithmetic** — cv2.add, cv2.subtract, saturation vs modulo arithmetic
- [x] **13 — Image Blending** — cv2.addWeighted, alpha blending, transparency
- [x] **14 — Bitwise Operations** — cv2.bitwise_and/or/xor/not, masking basics
- [x] **15 — Creating Masks** — Binary masks from thresholds, combining masks
- [x] **16 — Image Padding & Borders** — cv2.copyMakeBorder, BORDER_CONSTANT, BORDER_REFLECT
- [x] **17 — Image Channels: Split & Merge** — cv2.split, cv2.merge, single-channel visualization
- [x] **18 — Understanding Histograms** — cv2.calcHist, plotting pixel intensity distributions
- [x] **19 — Histogram Equalization** — cv2.equalizeHist, CLAHE, improving contrast

### Intermediate — Core Processing (Katas 20–49)

Image filtering, edge detection, morphology, and contour analysis.

- [x] **20 — Smoothing: Averaging & Box Filter** — cv2.blur, cv2.boxFilter, kernel size effects
- [x] **21 — Gaussian Blur** — cv2.GaussianBlur, sigma, noise reduction
- [x] **22 — Median Blur** — cv2.medianBlur, salt-and-pepper noise removal
- [x] **23 — Bilateral Filter** — cv2.bilateralFilter, edge-preserving smoothing
- [x] **24 — Image Sharpening** — Kernel convolution, unsharp masking, cv2.filter2D
- [x] **25 — Custom Kernels & Convolution** — cv2.filter2D, designing custom filters
- [x] **26 — Simple Thresholding** — cv2.threshold, THRESH_BINARY, THRESH_TRUNC, THRESH_TOZERO
- [x] **27 — Adaptive Thresholding** — cv2.adaptiveThreshold, MEAN vs GAUSSIAN, blockSize
- [x] **28 — Otsu's Thresholding** — Automatic threshold selection, bimodal histograms
- [x] **29 — Sobel Edge Detection** — cv2.Sobel, gradient direction, dx/dy
- [x] **30 — Scharr Operator** — cv2.Scharr, better accuracy than Sobel for small kernels
- [x] **31 — Laplacian Edge Detection** — cv2.Laplacian, second-order derivatives
- [x] **32 — Canny Edge Detection** — cv2.Canny, double thresholds, non-maximum suppression
- [x] **33 — Morphology: Erosion** — cv2.erode, structuring elements, iteration count
- [x] **34 — Morphology: Dilation** — cv2.dilate, expanding regions, filling gaps
- [x] **35 — Morphology: Opening & Closing** — cv2.morphologyEx, noise removal strategies
- [x] **36 — Morphology: Gradient, TopHat, BlackHat** — Advanced morphological operations
- [x] **37 — Connected Components** — cv2.connectedComponents, labeling regions
- [x] **38 — Finding Contours** — cv2.findContours, RETR modes, CHAIN_APPROX methods
- [x] **39 — Contour Properties** — cv2.contourArea, cv2.arcLength, bounding boxes
- [x] **40 — Contour Drawing & Filtering** — cv2.drawContours, filtering by area/shape
- [x] **41 — Convex Hull & Defects** — cv2.convexHull, cv2.convexityDefects
- [x] **42 — Contour Approximation** — cv2.approxPolyDP, shape simplification, epsilon
- [x] **43 — Shape Detection** — Classifying contours as triangle, rectangle, circle
- [x] **44 — Moments & Centroids** — cv2.moments, center of mass, hu moments
- [x] **45 — Bounding Shapes** — cv2.boundingRect, cv2.minAreaRect, cv2.minEnclosingCircle
- [x] **46 — Flood Fill** — cv2.floodFill, magic wand-style selection
- [x] **47 — Watershed Algorithm** — cv2.watershed, marker-based segmentation
- [x] **48 — GrabCut Segmentation** — cv2.grabCut, foreground extraction
- [x] **49 — Distance Transform** — cv2.distanceTransform, skeleton extraction

### Intermediate — Color & Frequency Domain (Katas 50–59)

Deeper color analysis and frequency-domain processing.

- [x] **50 — Color-Based Object Detection** — cv2.inRange with HSV, creating color masks
- [x] **51 — HSV Trackbar Color Picker** — Interactive HSV range tuning
- [x] **52 — Back Projection** — cv2.calcBackProject, histogram-based detection
- [x] **53 — Color Quantization** — K-means clustering on pixel colors
- [x] **54 — Fourier Transform** — cv2.dft, magnitude spectrum, frequency visualization
- [x] **55 — Frequency Domain Filtering** — Low-pass and high-pass filters in frequency domain
- [x] **56 — Image Inpainting** — cv2.inpaint, removing objects/text from images
- [x] **57 — Image Denoising** — cv2.fastNlMeansDenoisingColored, non-local means
- [x] **58 — Perspective Transform** — cv2.getPerspectiveTransform, cv2.warpPerspective
- [x] **59 — Affine Transform** — cv2.getAffineTransform, cv2.warpAffine, three-point mapping

### Advanced — Feature Detection & Matching (Katas 60–69)

Keypoints, descriptors, and matching for recognition tasks.

- [x] **60 — Harris Corner Detection** — cv2.cornerHarris, corner response, non-max suppression
- [x] **61 — Shi-Tomasi Corners** — cv2.goodFeaturesToTrack, quality level, min distance
- [x] **62 — FAST Keypoint Detection** — cv2.FastFeatureDetector, real-time corners
- [x] **63 — ORB Descriptors** — cv2.ORB_create, keypoints + descriptors, oriented BRIEF
- [x] **64 — SIFT Descriptors** — cv2.SIFT_create, scale-invariant features
- [x] **65 — Brute-Force Matching** — cv2.BFMatcher, Hamming vs L2 distance, crossCheck
- [x] **66 — FLANN Matching** — cv2.FlannBasedMatcher, KD-tree, faster matching
- [x] **67 — Homography & Warping** — cv2.findHomography, RANSAC, perspective correction
- [x] **68 — Image Stitching Basics** — Feature matching → homography → panorama
- [x] **69 — Template Matching** — cv2.matchTemplate, TM_CCOEFF_NORMED, multi-scale

### Advanced — Video & Real-Time (Katas 70–79)

Processing video streams frame by frame.

- [x] **70 — Reading Video Files** — cv2.VideoCapture from file, frame-by-frame loop
- [x] **71 — Webcam Capture** — cv2.VideoCapture(0), live feed processing
- [x] **72 — Writing Video Files** — cv2.VideoWriter, codec selection, FourCC
- [x] **73 — Frame Differencing** — Background subtraction via frame delta, motion detection
- [x] **74 — MOG2 Background Subtractor** — cv2.createBackgroundSubtractorMOG2
- [x] **75 — Optical Flow (Dense)** — cv2.calcOpticalFlowFarneback, motion field visualization
- [x] **76 — Optical Flow (Sparse)** — cv2.calcOpticalFlowPyrLK, Lucas-Kanade tracking
- [x] **77 — Object Tracking: CSRT** — cv2.TrackerCSRT, ROI selection, real-time tracking
- [x] **78 — Multi-Object Tracking** — cv2.MultiTracker, tracking multiple ROIs
- [x] **79 — Video Stabilization** — Feature matching between frames, affine correction

### Advanced — Detection & Recognition (Katas 80–89)

Classical detection pipelines and pre-trained models.

- [x] **80 — Haar Cascade: Face Detection** — cv2.CascadeClassifier, detectMultiScale
- [x] **81 — Haar Cascade: Eye & Smile Detection** — Nested cascades, ROI-based detection
- [x] **82 — HOG Pedestrian Detection** — cv2.HOGDescriptor, detectMultiScale
- [x] **83 — QR Code & Barcode Detection** — cv2.QRCodeDetector, decode and locate
- [x] **84 — Text Detection with EAST** — cv2.dnn with EAST model, text bounding boxes
- [x] **85 — DNN: Loading Pre-trained Models** — cv2.dnn.readNet, blob creation, forward pass
- [x] **86 — DNN: Image Classification** — MobileNet/ResNet classification with OpenCV DNN
- [x] **87 — DNN: Object Detection (SSD)** — Single-shot detection, bounding boxes + confidence
- [x] **88 — DNN: Object Detection (YOLO)** — YOLO with OpenCV DNN, NMS post-processing
- [x] **89 — DNN: Semantic Segmentation** — FCN/DeepLab with OpenCV DNN, pixel-wise labels

### Expert — Real-World Pipelines (Katas 90–99)

End-to-end projects combining multiple techniques.

- [x] **90 — Document Scanner** — Edge detection → contour → perspective transform → threshold
- [x] **91 — Lane Detection** — ROI masking → Canny → Hough lines → lane overlay
- [x] **92 — Panorama Stitching Pipeline** — Multi-image feature matching → homography → blend
- [x] **93 — Face Blurring Pipeline** — Haar detection → Gaussian blur on detected regions
- [x] **94 — Motion Heatmap** — Accumulate frame differences → colorize → overlay
- [x] **95 — Color-Based Object Tracker** — HSV mask → contours → centroid tracking
- [x] **96 — Image Comparison & Similarity** — SSIM, histogram comparison, feature matching
- [x] **97 — Augmented Reality Overlay** — Marker detection → homography → overlay image
- [x] **98 — Performance Optimization** — UMat (GPU), vectorized NumPy, avoiding Python loops
- [x] **99 — Building a CV Pipeline Framework** — Composable stages, error handling, benchmarking

---

## Live Camera & Video Projects (Katas 100+)

Real-world projects using actual webcam/camera feeds. These katas combine
techniques from earlier lessons into practical, runnable applications.
All starter code uses `cv2.VideoCapture(0)` with a real camera.

Legend: `[x]` = implemented, `[ ]` = planned

### Live Camera — Foundations (Katas 100–104)

Setting up robust camera pipelines and real-time display patterns.

- [ ] **100 — Live Camera Feed with FPS Overlay** — VideoCapture loop, FPS calculation, graceful exit, resolution setting
- [ ] **101 — Live Edge Detection** — Real-time Canny on webcam, trackbar-controlled thresholds
- [ ] **102 — Live Color Picker** — Click on webcam feed to get BGR/HSV values, crosshair overlay
- [ ] **103 — Live Histogram Display** — Real-time histogram alongside camera feed, per-channel visualization
- [ ] **104 — Live Split-Screen Filters** — Show original + multiple filters (gray, blur, edges) in a 2x2 grid

### Live Camera — Detection & Tracking (Katas 105–112)

Real-time detection and object tracking with a physical camera.

- [ ] **105 — Live Face Detection & Counting** — Haar cascade on webcam, face count overlay, bounding boxes
- [ ] **106 — Live Eye & Smile Detection** — Nested cascades on webcam, ROI-based eye/smile detection
- [ ] **107 — Live Color Object Tracking** — HSV masking on webcam, track a colored object, draw trail
- [ ] **108 — Live Motion Detection Alarm** — Frame differencing on webcam, motion zones, visual alert
- [ ] **109 — Live Hand Detection** — Skin-color segmentation, convex hull, finger counting
- [ ] **110 — Live Object Tracking with Selection** — Click-to-select ROI, CSRT tracker on webcam feed
- [ ] **111 — Live People Counter (Line Crossing)** — Background subtraction + centroid tracking, count crosses
- [ ] **112 — Live Multi-Object Color Tracker** — Track multiple colored objects simultaneously, ID assignment

### Live Camera — AR & Visual Effects (Katas 113–117)

Augmented reality overlays and creative visual effects on live video.

- [ ] **113 — Live Face Blur (Privacy Filter)** — Detect + blur faces in real-time, toggle pixelation vs gaussian
- [ ] **114 — Live Virtual Sunglasses** — Detect eyes, overlay sunglasses image with alpha blending
- [ ] **115 — Live Background Removal** — GrabCut/MOG2 background subtraction, replace with solid color or image
- [ ] **116 — Live Cartoon Effect** — Bilateral filter + edge mask for cartoon/comic look on webcam
- [ ] **117 — Live Mirror & Kaleidoscope** — Split, flip, and tile webcam for mirror/kaleidoscope effects

### Live Camera — Measurement & Analysis (Katas 118–122)

Practical measurement and analysis tools using live camera.

- [ ] **118 — Live QR Code Scanner** — Detect, decode, and highlight QR codes in real-time
- [ ] **119 — Live Document Edge Detection** — Real-time contour detection for document capture, perspective guide
- [ ] **120 — Live Color Dominant Analyzer** — K-means on live frames, show dominant colors palette
- [ ] **121 — Live Brightness & Contrast Monitor** — Real-time histogram stats, exposure warning indicators
- [ ] **122 — Live Object Size Estimation** — Reference object calibration, measure real-world dimensions

### Live Camera — Advanced Projects (Katas 123–127)

Multi-technique projects combining detection, tracking, and analysis.

- [ ] **123 — Live Motion Heatmap** — Accumulated frame differences, real-time heatmap overlay
- [ ] **124 — Live Speed Estimation** — Track object centroids between frames, estimate px/s velocity
- [ ] **125 — Live Gesture-Controlled Drawing** — Draw on canvas with colored object or finger as a pen
- [ ] **126 — Live Security Camera System** — Motion detection + face logging + timestamp + recording trigger
- [ ] **127 — Live Video Streaming Dashboard** — Multi-camera grid, per-feed processing, stats overlay
