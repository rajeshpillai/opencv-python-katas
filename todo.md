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

- [ ] **70 — Reading Video Files** — cv2.VideoCapture from file, frame-by-frame loop
- [ ] **71 — Webcam Capture** — cv2.VideoCapture(0), live feed processing
- [ ] **72 — Writing Video Files** — cv2.VideoWriter, codec selection, FourCC
- [ ] **73 — Frame Differencing** — Background subtraction via frame delta, motion detection
- [ ] **74 — MOG2 Background Subtractor** — cv2.createBackgroundSubtractorMOG2
- [ ] **75 — Optical Flow (Dense)** — cv2.calcOpticalFlowFarneback, motion field visualization
- [ ] **76 — Optical Flow (Sparse)** — cv2.calcOpticalFlowPyrLK, Lucas-Kanade tracking
- [ ] **77 — Object Tracking: CSRT** — cv2.TrackerCSRT, ROI selection, real-time tracking
- [ ] **78 — Multi-Object Tracking** — cv2.MultiTracker, tracking multiple ROIs
- [ ] **79 — Video Stabilization** — Feature matching between frames, affine correction

### Advanced — Detection & Recognition (Katas 80–89)

Classical detection pipelines and pre-trained models.

- [ ] **80 — Haar Cascade: Face Detection** — cv2.CascadeClassifier, detectMultiScale
- [ ] **81 — Haar Cascade: Eye & Smile Detection** — Nested cascades, ROI-based detection
- [ ] **82 — HOG Pedestrian Detection** — cv2.HOGDescriptor, detectMultiScale
- [ ] **83 — QR Code & Barcode Detection** — cv2.QRCodeDetector, decode and locate
- [ ] **84 — Text Detection with EAST** — cv2.dnn with EAST model, text bounding boxes
- [ ] **85 — DNN: Loading Pre-trained Models** — cv2.dnn.readNet, blob creation, forward pass
- [ ] **86 — DNN: Image Classification** — MobileNet/ResNet classification with OpenCV DNN
- [ ] **87 — DNN: Object Detection (SSD)** — Single-shot detection, bounding boxes + confidence
- [ ] **88 — DNN: Object Detection (YOLO)** — YOLO with OpenCV DNN, NMS post-processing
- [ ] **89 — DNN: Semantic Segmentation** — FCN/DeepLab with OpenCV DNN, pixel-wise labels

### Expert — Real-World Pipelines (Katas 90–99)

End-to-end projects combining multiple techniques.

- [ ] **90 — Document Scanner** — Edge detection → contour → perspective transform → threshold
- [ ] **91 — Lane Detection** — ROI masking → Canny → Hough lines → lane overlay
- [ ] **92 — Panorama Stitching Pipeline** — Multi-image feature matching → homography → blend
- [ ] **93 — Face Blurring Pipeline** — Haar detection → Gaussian blur on detected regions
- [ ] **94 — Motion Heatmap** — Accumulate frame differences → colorize → overlay
- [ ] **95 — Color-Based Object Tracker** — HSV mask → contours → centroid tracking
- [ ] **96 — Image Comparison & Similarity** — SSIM, histogram comparison, feature matching
- [ ] **97 — Augmented Reality Overlay** — Marker detection → homography → overlay image
- [ ] **98 — Performance Optimization** — UMat (GPU), vectorized NumPy, avoiding Python loops
- [ ] **99 — Building a CV Pipeline Framework** — Composable stages, error handling, benchmarking
