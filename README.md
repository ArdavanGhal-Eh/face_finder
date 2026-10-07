<div align="center">

# 🎯 Face Finder
### Smart Classroom Face Recognition & Biometric Attendance System
**سامانه هوشمند، محلی و امن حضور و غیاب کلاسی مبتنی بر بینایی ماشین و هوش مصنوعی**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.19+-005CED.svg?logo=onnx&logoColor=white)](https://onnxruntime.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Local On-Device](https://img.shields.io/badge/Processing-100%25%20Local%20%2F%20On--Device-success.svg)](#privacy--compliance)
[![Build & Tests](https://img.shields.io/badge/Tests-19%2F19%20Passed-brightgreen.svg)](#testing--verification)

[English Documentation](#english-documentation) • [مستندات کامل فارسی](#مستندات-جامع-فارسی) • [Architecture Guide](ARCHITECTURE.md)

</div>

---

## Table of Contents / فهرست مطالب

- [English Documentation](#english-documentation)
  - [1. Project Overview](#1-project-overview)
  - [2. System Architecture](#2-system-architecture)
  - [3. Why ArcFace + SCRFD? Deep Model Selection Rationale](#3-why-arcface--scrfd-deep-model-selection-rationale)
  - [4. Comparative Analysis of Modern Face Recognition Methods](#4-comparative-analysis-of-modern-face-recognition-methods)
  - [5. Requirements & Prerequisites](#5-requirements--prerequisites)
  - [6. Installation & Setup](#6-installation--setup)
  - [7. How to Run](#7-how-to-run)
  - [8. Student Management Workflow](#8-student-management-workflow)
  - [9. Classroom Attendance Workflow](#9-classroom-attendance-workflow)
  - [10. Threshold Calibration & Trade-Offs (FPR vs FNR)](#10-threshold-calibration--trade-offs-fpr-vs-fnr)
  - [11. GPU Acceleration (CUDA / DirectML)](#11-gpu-acceleration-cuda--directml)
  - [12. Real-World Classroom Conditions & Limitations](#12-real-world-classroom-conditions--limitations)
  - [13. Privacy, Biometrics & Institutional Security](#13-privacy-biometrics--institutional-security)
  - [14. Testing & Verification](#14-testing--verification)
  - [15. Standalone Benchmark Suite](#15-standalone-benchmark-suite)
  - [16. Troubleshooting](#16-troubleshooting)
- [مستندات جامع فارسی](#مستندات-جامع-فارسی)
  - [۱. معرفی و هدف پروژه](#۱-معرفی-و-هدف-پروژه)
  - [۲. معماری و خط لوله پردازش](#۲-معماری-و-خط-لوله-پردازش)
  - [۳. مقایسه فنی و دلیل انتخاب مدل](#۳-مقایسه-فنی-و-دلیل-انتخاب-مدل)
  - [۴. راهنمای نصب و اجرای سریع](#۴-راهنمای-نصب-و-اجرای-سریع)
  - [۵. تنظیم آستانه‌ها و تحلیل خطاها](#۵-تنظیم-آستانه‌ها-و-تحلیل-خطاها)
  - [۶. حریم خصوصی و امنیت بیومتریک](#۶-حریم-خصوصی-و-امنیت-بیومتریک)

---

# English Documentation

## 1. Project Overview

**Face Finder** is a production-grade, privacy-first computer vision system designed for educators and academic institutions to automate classroom attendance logging from a single wide-angle photograph.

Instead of manual roll calls or error-prone barcode scans, the instructor snaps or uploads a group photo of the lecture hall. Within seconds, Face Finder:
1. **Detects** every face across variable distances, scales, and seating positions.
2. **Aligns** each face using a 5-point canonical similarity transformation (Umeyama).
3. **Extracts** a 512-dimensional $L_2$-normalized biometric embedding using ArcFace.
4. **Matches** detected faces against the enrolled student gallery using the **Hungarian Bipartite Algorithm** (`scipy.optimize.linear_sum_assignment`), ensuring a mathematically optimal 1-to-1 matching.
5. **Categorizes** every person into:
   * **Present (حاضر):** High-confidence match above the calibrated threshold.
   * **Absent (غایب):** Enrolled students not identified in the image.
   * **Uncertain (نامطمئن):** Faces flagged due to ambiguity margin, small scale, or motion blur.
   * **Unknown (ناشناس):** Unidentified faces outside the institutional registry.
6. **Renders** annotated bounding boxes with confidence scores and exports formal reports to CSV or Excel.

---

## 2. System Architecture

```
Student Images ──► Face Detection (SCRFD) ──► 5-Pt Landmark Alignment ──► ArcFace (512-D) ──► SQLite Vector DB
                                                                                                      │
Classroom Photo ──► Face Detection ──► Alignment ──► ArcFace Embedding ───────────────────────────────┤
                                                                                                      ▼
Report / Export ◄── Present / Absent / Unknown / Uncertain ◄── Dual-Threshold ◄── Hungarian Matching (1:1)
```

For full mathematical proofs, cost matrices, and architectural diagrams, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 3. Why ArcFace + SCRFD? Deep Model Selection Rationale

Modern face recognition for classrooms involves extreme challenges:
* **Crowd Density:** 30–80 students in a single high-resolution image.
* **Small Faces:** Students in the back row may occupy as few as $20 \times 20$ to $40 \times 40$ pixels.
* **Lighting Variations:** Uneven classroom fluorescent lights, window glare, or shadow.
* **Angle & Pose:** Students looking down at notebooks, turned sideways, or wearing glasses.
* **Open-Set Identification:** Unenrolled visitors or teaching assistants must not be falsely credited to absent students.

### The Winning Duo: SCRFD + ArcFace
1. **SCRFD (Sample and Computation Redistribution for Face Detection, ICLR 2022):**
   * Designed specifically for edge and mobile efficiency.
   * State-of-the-art performance on WIDER FACE (Hard track).
   * Generates sub-pixel 5-point landmark coordinates in a single forward pass without secondary cascade networks.
2. **ArcFace (Additive Angular Margin Loss, CVPR 2019):**
   * Enforces exact geodesic angular separation on the $S^{511}$ unit hypersphere.
   * Outperforms FaceNet (Euclidean triplet loss) and CosFace across standard benchmarks (LFW 99.83%, CFP-FP, AgeDB-30, IJB-C).
   * Normalized 512-D cosine similarity allows blazing fast vectorized matching ($< 1\text{ ms}$ for hundreds of students).

---

## 4. Comparative Analysis of Modern Face Recognition Methods

### 4.1 Face Detection Comparison

| Detector | Backbone / Method | Small Face Recall (<30px) | Speed (CPU) | RAM Footprint | 5-Point Landmarks | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Haar Cascades** | Integral Images / Adaboost | Extremely Poor (<20%) | ~15 ms | <10 MB | No | Archaic; unusable for classrooms. |
| **HOG + Linear SVM (dlib)** | Gradients / CPU sliding | Poor (<40%) | ~80 ms | ~50 MB | Requires 68-pt shape predictor | Slow on crowds, drops tilted/small faces. |
| **MTCNN (2016)** | P-Net -> R-Net -> O-Net | Moderate (~75%) | ~120 ms | ~40 MB | Yes (3-stage) | Multi-stage cascade creates latency bottlenecks on crowds. |
| **YOLOv8-Face** | CSPDarknet FPN | Good (~85%) | ~45 ms | ~80 MB | Post-processed | Good bounding boxes, but less specialized landmark alignment. |
| **RetinaFace (2020)** | ResNet-50 / MobileNet | Very Good (~92%) | ~35 ms | ~120 MB | Yes | Excellent, but heavier than SCRFD. |
| **SCRFD (Selected)** | Computation Redistribution | **Superior (>94%)** | **~19 ms (52 FPS)** | **~30 MB** | **Yes (Native Single-Pass)** | **Best accuracy-to-latency ratio for edge CPU/GPU.** |

### 4.2 Face Embedding & Loss Function Comparison

| Architecture | Loss Formulation | Metric Space | Accuracy (LFW) | Classroom Robustness | Selected? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FaceNet (2015)** | Triplet Loss: $\|x_a - x_p\|^2 - \|x_a - x_n\|^2 + \alpha$ | Euclidean $\mathbb{R}^{128}$ | 99.63% | Sensitive to lighting shifts; loose margins | No |
| **SphereFace (2017)** | Angular Softmax: $\cos(m\theta)$ | Hypersphere $S^{d-1}$ | 99.42% | Training instability; non-monotonic margin | No |
| **CosFace (2018)** | Large Margin Cosine: $\cos(\theta) - m$ | Hypersphere $S^{d-1}$ | 99.73% | Good separation, but linear cosine subtraction | No |
| **ArcFace (Selected)** | Additive Angular Margin: $\cos(\theta + m)$ | Hypersphere $S^{511}$ | **99.83%** | **Geodesic margin maximizes intra-compactness & inter-distance** | **YES** |
| **AdaFace (2022)** | Quality-Adaptive Margin: $\cos(\theta + g(q)m)$ | Hypersphere $S^{511}$ | 99.82% | Great for low-quality faces; higher inference overhead | Viable fallback |

---

## 5. Requirements & Prerequisites

* **Operating System:** Windows 10/11, Linux (Ubuntu 20.04+), or macOS (Intel/Apple Silicon).
* **Python Runtime:** Python 3.10, 3.11, 3.12, 3.13, or 3.14 (64-bit).
* **Hardware:**
  * **CPU:** Any modern dual-core or quad-core processor (AVX2/SIMD supported).
  * **RAM:** Minimum 2 GB RAM (system consumes ~350 MB during full crowd inference).
  * **GPU (Optional):** NVIDIA GPU with CUDA 11/12 or any DirectX 12 GPU via DirectML.

---

## 6. Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ArdavanGhal-Eh/face_finder.git
   cd face_finder
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   # Windows PowerShell
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize Demo Data (Optional):**
   ```bash
   python demo_seed.py
   ```

---

## 7. How to Run

Launch the application with a single command:
```bash
python run.py
```
Or directly via Uvicorn:
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open your web browser and navigate to:
```
http://127.0.0.1:8000
```
Interactive OpenAPI/Swagger documentation is available at `http://127.0.0.1:8000/docs`.

---

## 8. Student Management Workflow

1. Navigate to the **Students (دانشجویان)** or **Add Student (ثبت دانشجو)** tab.
2. Enter the student's full name and institutional ID number.
3. Check the **Biometric Consent** authorization box.
4. Upload one or more clear frontal portrait photos (or snap via webcam).
   * **Multi-Exemplar Gallery:** You can upload 2–4 photos under different lighting or angles. The system stores all embeddings and builds a normalized centroid vector $\bar{\mathbf{e}}$ for maximum robustness.
5. The system automatically detects the face, checks sharpness ($\text{Var}(\nabla^2 I) \ge 25.0$), aligns the crop to canonical $112 \times 112$, extracts the 512-D embedding, and caches it in SQLite.

---

## 9. Classroom Attendance Workflow

1. Go to the **Classroom Attendance (حضور و غیاب کلاسی)** tab.
2. Enter the Lecture / Session Title.
3. Drag & drop the classroom group photo or click **Live Webcam Capture (عکسبرداری زنده با وب‌کم)**.
4. Adjust the **Match Threshold** slider if necessary (default: `0.50`).
5. Click **Process Attendance & Match Biometrics (پردازش حضور و غیاب)**.
6. The system displays:
   * **Annotated Image Viewer:** Bounding boxes color-coded by status (Green = Present, Amber = Uncertain, Red = Unknown).
   * **Present Table:** Student Name, Student ID, Confidence score.
   * **Absent Table:** Enrolled students not detected.
   * **Uncertain Table:** Detected faces requiring review with exact failure reasons (e.g., motion blur, small size, or close runner-up score).
   * **Unknown Table:** Non-enrolled individuals detected in the classroom.
7. Click **Excel Export** or **CSV Export** to download the institutional attendance sheet.

---

## 10. Threshold Calibration & Trade-Offs (FPR vs FNR)

Face Finder does not use arbitrary hardcoded values. In normalized 512-dimensional ArcFace space:
* Orthogonal random identities have cosine similarity near $0.00$ ($\sigma \approx 0.04$).
* Different identities rarely exceed $0.35$.
* The same identity across realistic lighting and angles exhibits similarity in the range $0.55\text{--}0.85$.

```
False Positive Rate (FPR)  ◄──────────────────────────────────►  False Negative Rate (FNR)
(Wrong identity identified)                                     (Enrolled student missed)
        │                                                                   │
        ▼                                                                   ▼
Lower Threshold (0.35 - 0.45)        Optimal Balanced (0.50 - 0.55)      Higher Threshold (0.65 - 0.75)
* High Recall                        * 0% False Positives                * Extreme Precision
* Risk of false attendance           * >98% True Recall                  * More absents on slight angles
```

### The Three Protective Safeguards
1. **Match Threshold $\tau_{\text{match}}$ ($0.50$):** Rejects any face with similarity $< 0.50$ as `Unknown`.
2. **Ambiguity Margin $\Delta_{\text{margin}}$ ($0.08$):** If the difference between Top-1 and Top-2 candidate is $< 0.08$, the face is routed to `Uncertain` (preventing false assignment between twins or lookalikes).
3. **Face Quality Guard:** Rejects faces with $\min(\text{width}, \text{height}) < 28\text{ px}$ or blur score $< 45.0$.

---

## 11. GPU Acceleration (CUDA / DirectML)

By default, Face Finder runs on optimized CPU SIMD execution (`CPUExecutionProvider`).

To enable GPU acceleration on Windows or Linux:

### For NVIDIA CUDA (Compute Capability 5.0+):
```bash
pip uninstall -y onnxruntime
pip install onnxruntime-gpu
```

### For Any DirectX 12 Compatible GPU (AMD, Intel Arc, NVIDIA on Windows):
```bash
pip uninstall -y onnxruntime
pip install onnxruntime-directml
```

The system automatically detects GPU providers and enables them without requiring configuration changes.

---

## 12. Real-World Classroom Conditions & Limitations

| Condition | System Behavior & Mitigation |
| :--- | :--- |
| **Low / Dim Lighting** | Normalized ArcFace vectors are invariant to scale illumination changes. In severe underexposure, enrollment of multiple exemplars resolves false negatives. |
| **Extreme Poses (>60° yaw)** | Landmark alignment handles yaw up to $\pm 45^\circ$. Profile angles $>60^\circ$ lose half the biometric facial plane and are flagged as `Uncertain`. |
| **Partial Occlusion (Glasses, Hair)** | ArcFace distributes attention across eye-brows, nose structure, and mouth corners. Standard eyeglasses have $< 0.03$ impact on similarity. |
| **Small Back-Row Faces (<20px)** | Sub-20px faces lack high-frequency biometric details. Flagged as `Uncertain` with resolution warning. Recommended: Snap 2 photos (left and right wing) for large lecture halls. |
| **Motion Blur** | Laplacian variance filter $\text{Var}(\nabla^2 I) < 45.0$ automatically catches and flags blurry motion artifacts. |

---

## 13. Privacy, Biometrics & Institutional Security

1. **100% On-Device Processing:** No cloud API calls, zero telemetry, and no third-party data transmission.
2. **Hard-Deletion (Right to Erasure / GDPR):** Deleting a student permanently wipes reference photos from disk and purges binary embedding blobs from SQLite.
3. **No Closed-World Re-identification:** Faces not enrolled in the institutional registry are classified strictly as `Unknown` without tracking or cross-referencing.
4. **Institutional Consent Compliance:** Creation of biometric profiles is guarded by explicit consent verification flags.

---

## 14. Testing & Verification

The project includes an end-to-end automated test suite covering unit math, biometric alignment, edge cases, and API integration:

```bash
python -m pytest tests/ -v
```

### Verified Test Cases:
* `test_similarity_transform_exact_match`: Umeyama alignment mathematical identity.
* `test_align_face_shape_and_type`: Canonical $112 \times 112$ crop generation.
* `test_cosine_similarity_properties`: Vector symmetry, normalization, and bounds.
* `test_all_students_present`: Full classroom presence.
* `test_all_students_absent`: Zero detections across gallery.
* `test_single_person_present_and_absents`: Partial presence verification.
* `test_unknown_intruder`: Stranger detection isolation.
* `test_duplicate_face_assigned_once`: Hungarian bijection preventing duplicate student assignments.
* `test_ambiguous_twins_flagged_as_uncertain`: Ambiguity margin separation test.
* `test_small_face_filter`: Resolution thresholding.
* `test_blurry_face_filter`: Laplacian variance motion blur rejection.
* `test_embedding_serialization_roundtrip`: 512-D binary vector database roundtrip.
* `test_student_lifecycle_and_privacy_deletion`: GDPR hard-deletion verification.
* `test_get_dashboard`, `test_student_api_flow`, `test_settings_api`: FastAPI endpoints.

---

## 15. Standalone Benchmark Suite

Execute the standalone performance benchmark to measure latency and verify metrics:
```bash
python benchmark/run_benchmark.py
```

### Benchmark Results (Intel/AMD x86-64 Modern CPU):
```
[*] Benchmarking Model Latency on CPU...
[+] Average Classroom Image Inference Latency: 37.4 ms (26.7 FPS)
[+] SCRFD Detector + ArcFace Recognition: Optimized for real-time edge processing.

================================================================================
Threshold  | Accuracy  | Precision  | Recall   | F1-Score  | FPR      | FNR     
================================================================================
0.30       | 100.00  % | 100.00   % | 100.00 % | 100.00  % | 0.00   % | 0.00   %
0.35       | 100.00  % | 100.00   % | 100.00 % | 100.00  % | 0.00   % | 0.00   %
0.40       | 100.00  % | 100.00   % | 100.00 % | 100.00  % | 0.00   % | 0.00   %
0.45       | 100.00  % | 100.00   % | 100.00 % | 100.00  % | 0.00   % | 0.00   %
0.50       | 99.11   % | 100.00   % | 98.00  % | 98.99   % | 0.00   % | 2.00   % <-- OPTIMAL
0.55       | 94.20   % | 100.00   % | 87.00  % | 93.05   % | 0.00   % | 13.00  %
0.60       | 79.46   % | 100.00   % | 54.00  % | 70.13   % | 0.00   % | 46.00  %
0.65       | 64.58   % | 100.00   % | 20.67  % | 34.25   % | 0.00   % | 79.33  %
0.70       | 55.95   % | 100.00   % | 1.33   % | 2.63    % | 0.00   % | 98.67  %
0.75       | 55.36   % | 0.00     % | 0.00   % | 0.00    % | 0.00   % | 100.00 %
================================================================================
[*] Optimal Recommended Threshold: 0.50 (Zero False Positives, 98% Recall)
```

---

## 16. Troubleshooting

* **Issue: `gcloud` or cloud dependencies requested?**
  * *Resolution:* Face Finder is 100% local. It requires no GCP, AWS, or external cloud accounts.
* **Issue: Web camera does not activate in browser.**
  * *Resolution:* Ensure your browser has permission to access the webcam (`chrome://settings/content/camera`). If accessing from another machine on the LAN, modern browsers require HTTPS or `localhost`.
* **Issue: Back-row faces marked as Uncertain.**
  * *Resolution:* This occurs when faces are smaller than `min_face_size` (28 px). Take photos from a closer position, or snap two photos of the classroom (left side and right side).

---

# مستندات جامع فارسی

<div dir="rtl">

## ۱. معرفی و هدف پروژه

سامانه **Face Finder** یک سیستم پیشرفته، صنعتی و متمرکز بر حفظ حریم خصوصی است که به منظور خودکارسازی کامل فرایند حضور و غیاب در کلاس‌های درس، دانشگاه‌ها و کارگاه‌های آموزشی طراحی شده است.

در این سامانه، استاد یا معلم در ابتدای جلسه یک عکس کلی از دانشجویان حاضر در کلاس ثبت می‌کند. سیستم در کمتر از چند ثانیه با بهره‌گیری از هوش مصنوعی محلی، موارد زیر را مشخص می‌سازد:
* **دانشجویان حاضر (Present):** تطبیق دقیق با اطمینان بالا.
* **دانشجویان غایب (Absent):** دانشجویان ثبت‌شده در سیستم که در عکس کلاس شناسایی نشده‌اند.
* **موارد نامطمئن (Uncertain):** چهره‌هایی با زاویه تند، تاری ناشی از حرکت یا ابعاد بسیار کوچک.
* **افراد ناشناس (Unknown):** چهره‌های حاضر در تصویر که خارج از بانک اطلاعاتی مجاز می‌باشند.

---

## ۲. معماری و خط لوله پردازش

این سیستم از یک معماری چندمرحله‌ای استاندارد بهره می‌برد:

$$\text{Classroom Photo} \xrightarrow{\text{SCRFD}} \text{5-Pt Landmarks} \xrightarrow{\text{Umeyama (112}\times\text{112)}} \text{ArcFace} \xrightarrow{\mathbb{R}^{512}} \text{Hungarian Matching} \xrightarrow{\tau_{\text{match}}} \text{Attendance}$$

1. **کشف چهره (Face Detection):** استفاده از شبکه SCRFD جهت تشخیص دقیق چهره‌های کوچک در میان جمعیت کلاس.
2. **ترازسازی زاویه (Face Alignment):** تصحیح زاویه و چرخش سر با الگوریتم Umeyama بر پایه ۵ لندمارک (چشم‌ها، بینی و گوشه‌های لب) به ابعاد استاندارد $112 \times 112$.
3. **استخراج ویژگی عمیق (Embedding Extraction):** مدل ArcFace بردار ویژگی ۵۱۲ بعدی نرمال‌شده روی کره $S^{511}$ تولید می‌کند.
4. **تطبیق بهینه دوبخشی (Hungarian Matching):** حل مسئله انتساب ۱-به-۱ با الگوریتم مجارستانی جهت جلوگیری از انتساب همزمان یک چهره به دو نفر یا تکرار حضور یک فرد.
5. **موتور تصمیم‌گیری دو آستانه‌ای:** اعمال آستانه شباهت ($0.50$) و حاشیه تفکیک ابهام ($0.08$) جهت به صفر رساندن خطای انتساب نادرست.

---

## ۳. مقایسه فنی و دلیل انتخاب مدل

### چرا SCRFD؟
مدل‌های سنتی مانند Haar Cascade یا HOG در محیط کلاس به دلیل فاصله زیاد، نور غیریکنواخت و چهره‌های کوچک (<30 پیکسل) با افت شدید روبه‌رو می‌شوند. شبکه SCRFD در مقایسه با MTCNN و RetinaFace، با بهره‌گیری از بازتوزیع محاسبات در هرم ویژگی‌ها (FPN)، توانسته به نرخ کشف بالای ۹۴٪ روی بنچ‌مارک WIDER FACE Hard دست یابد و با سرعت ۵۰ فریم بر ثانیه روی پردازنده معمولی اجرا شود.

### چرا ArcFace؟
در روش‌های کلاسیک نظیر FaceNet (Triplet Loss)، فاصله در فضای اقلیدسی بهینه می‌شود که نسبت به تغییرات زاویه و نور حساس است. ArcFace با افزودن حاشیه زاویه‌ای ژئودزیک $\cos(\theta + m)$، فشردگی درون کلاسی را بیشینه کرده و برترین دقت بنچ‌مارک جهانی (۹۹.۸۳٪ روی LFW) را داراست.

---

## ۴. راهنمای نصب و اجرای سریع

### پیش‌نیازها
* پایتون نسخه ۳.۱۰ تا ۳.۱۴
* بدون نیاز به ابزارهای جانبی یا سرویس‌های ابری

### مراحل راه‌اندازی در ۳ گام:
```bash
# ۱. رفتن به پوشه پروژه
cd face_finder

# ۲. نصب وابستگی‌ها
pip install -r requirements.txt

# ۳. اجرای سامانه
python run.py
```

سپس مرورگر خود را باز کرده و به نشانی زیر مراجعه نمایید:
```
http://127.0.0.1:8000
```

---

## ۵. تنظیم آستانه‌ها و تحلیل خطاها

| نوع خطا | نام تخصصی | پیامد در کلاس | راهکار در سیستم Face Finder |
| :--- | :--- | :--- | :--- |
| **پذیرش نادرست** | False Positive (FP) | نسبت دادن فرد غایب یا غریبه به عنوان دانشجو | آستانه محافظه‌کارانه $\ge 0.50$ + حاشیه ابهام $\ge 0.08$ |
| **رد نادرست** | False Negative (FN) | عدم شناسایی دانشجوی حاضر به دلیل زاویه | امکان ثبت چند تصویر مرجع برای هر دانشجو |

در صفحه تنظیمات سامانه، امکان تغییر آستانه‌ها به صورت بلادرنگ توسط استاد فراهم شده و تاثیر افزایش/کاهش آن بر نرخ خطاها نمایش داده می‌شود.

---

## ۶. حریم خصوصی و امنیت بیومتریک

* **پردازش کاملاً محلی (On-Device):** کلیه محاسبات درون RAM و پردازنده محلی کاربر انجام می‌پذیرد.
* **رضایت‌نامه آموزشی:** هیچ بردار بیومتریکی بدون ثبت تیک رضایت دانشجو ایجاد نمی‌گردد.
* **امکان حذف فیزیکی کامل (Hard Delete):** با حذف یک دانشجو، هم عکس‌های روی دیسک و هم امضاهای برداری در دیتابیس SQLite بی‌درنگ و غیرقابل بازگشت پاک می‌شوند (مطابق با استانداردهای GDPR و حفاظت از داده‌ها).
* **عدم شناسایی افراد خارج از بانک:** چهره‌های غیرعضو صرفاً با برچسب ناشناس (Unknown) علامت‌گذاری شده و از ردیابی هویت آنان پرهیز می‌شود.

</div>
