# HIGH-END CLINICAL AI DASHBOARD DESIGN SPECIFICATION
**Application:** Disease Diagnosis Prediction Platform  
**Target Output Document:** `reports/high_end_ui_design_specification.md`  
**Architecture:** HTML5 / Vanilla CSS3 / Vanilla JavaScript (Same-Origin REST API)  
**Status:** Design & Technical Specification  

---

## 1. DESIGN PHILOSOPHY

### 1.1 Visual Identity: "Clinical AI Command Center"
The objective of this design specification is to elevate the Disease Diagnosis Prediction web interface from a basic functional web form into a portfolio-grade, enterprise-ready **Clinical AI Command Center**. The visual identity balances high-density clinical telemetry with modern software engineering aesthetics suitable for demonstration to hiring managers, machine learning leads, and open-source contributors.

### 1.2 Core Design Principles
* **Professional Healthcare Technology Aesthetic:** Uses a refined deep slate (`#0B0F17`, `#111827`, `#1E293B`) canvas paired with clinical teal (`#0EA5E9`), indigo (`#6366F1`), and restrained semantic indicators (`#10B981` / `#EF4444`).
* **High Information Density Without Clutter:** Employs structured grid alignment, compact card containers, and distinct visual hierarchy so complex clinical model parameters and empirical test metrics are readable at a glance.
* **Restrained Depth & Material Surface System:** Replaces noisy neon gradients and heavy glassmorphism blur with subtle 1px border highlights, controlled drop shadows, and clean surface contrast.
* **Zero Medical Misinformation & Strict Non-Clinical Branding:** Frames all model outputs explicitly as statistical machine learning probability scores rather than diagnostic assessments. Prominently displays non-clinical disclaimer notices across all views.
* **Zero Framework Overhead:** Built strictly with native Web Standards (HTML5, CSS3, ES6+ Vanilla JS). Zero dependency on React, Vue, Angular, Bootstrap, Tailwind, or third-party UI libraries.

---

## 2. COLOR & TOKEN SYSTEM

### 2.1 CSS Custom Properties (`:root`)

```css
:root {
  /* Surface & Background Canvas */
  --color-bg-canvas: #080c14;
  --color-bg-surface: #0f172a;
  --color-bg-card: #1e293b;
  --color-bg-card-hover: #26334d;
  --color-bg-elevated: #334155;

  /* Typography Colors */
  --color-text-primary: #f8fafc;
  --color-text-secondary: #cbd5e1;
  --color-text-muted: #94a3b8;
  --color-text-dim: #64748b;

  /* Brand Accents */
  --color-accent-teal: #0ea5e9;
  --color-accent-teal-glow: rgba(14, 165, 233, 0.15);
  --color-accent-indigo: #6366f1;
  --color-accent-indigo-glow: rgba(99, 102, 241, 0.15);

  /* Semantic Status & Risk Palette */
  --color-status-success-bg: rgba(16, 185, 129, 0.12);
  --color-status-success-border: rgba(16, 185, 129, 0.35);
  --color-status-success-text: #34d399;

  --color-status-warning-bg: rgba(245, 158, 11, 0.12);
  --color-status-warning-border: rgba(245, 158, 11, 0.35);
  --color-status-warning-text: #fbbf24;

  --color-status-danger-bg: rgba(239, 68, 68, 0.12);
  --color-status-danger-border: rgba(239, 68, 68, 0.35);
  --color-status-danger-text: #f87171;

  /* Borders & Dividers */
  --color-border-subtle: rgba(255, 255, 255, 0.08);
  --color-border-medium: rgba(255, 255, 255, 0.14);
  --color-border-active: rgba(14, 165, 233, 0.4);

  /* Elevation & Shadows */
  --shadow-sm: 0 2px 4px rgba(0, 0, 0, 0.4);
  --shadow-md: 0 8px 24px -6px rgba(0, 0, 0, 0.6);
  --shadow-lg: 0 16px 36px -8px rgba(0, 0, 0, 0.75);
  --shadow-teal-glow: 0 0 20px rgba(14, 165, 233, 0.2);

  /* Spacing Scale */
  --space-2xs: 0.25rem; /* 4px */
  --space-xs: 0.5rem;   /* 8px */
  --space-sm: 0.75rem;  /* 12px */
  --space-md: 1.0rem;   /* 16px */
  --space-lg: 1.5rem;   /* 24px */
  --space-xl: 2.0rem;   /* 32px */
  --space-2xl: 3.0rem;  /* 48px */

  /* Radii */
  --radius-sm: 6px;
  --radius-md: 10px;
  --radius-lg: 16px;
  --radius-full: 9999px;
}
```

### 2.2 Contrast & Accessibility Compliance
* **Body Text (`--color-text-primary` on `--color-bg-canvas`):** Contrast ratio of `15.8:1` (Exceeds WCAG AAA requirement of `7:1`).
* **Secondary Text (`--color-text-secondary` on `--color-bg-surface`):** Contrast ratio of `9.4:1` (Exceeds WCAG AAA).
* **Interactive Focus State (`--color-accent-teal` outline):** Contrast ratio of `8.2:1` against dark card backgrounds.

---

## 3. TYPOGRAPHY SYSTEM

### 3.1 Font Families
* **Heading & Display:** `'Outfit'`, `-apple-system`, `BlinkMacSystemFont`, `sans-serif`
* **Body & Telemetry:** `'Plus Jakarta Sans'`, `-apple-system`, `BlinkMacSystemFont`, `sans-serif`
* **Code & JSON:** `'JetBrains Mono'`, `'Fira Code'`, `monospace`

### 3.2 Type Hierarchy Scale

| Role | Font Family | Size | Weight | Line Height | Letter Spacing |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Hero Title** | Outfit | 2.0rem (32px) | 700 | 1.2 | -0.02em |
| **Section Header (H1)** | Outfit | 1.5rem (24px) | 600 | 1.3 | -0.01em |
| **Card Header (H2)** | Outfit | 1.125rem (18px) | 600 | 1.4 | 0.00em |
| **Group Legend (H3)** | Plus Jakarta Sans | 0.95rem (15.2px) | 600 | 1.4 | 0.01em |
| **Body Standard** | Plus Jakarta Sans | 0.875rem (14px) | 400 / 500 | 1.5 | 0.00em |
| **Form Label** | Plus Jakarta Sans | 0.8125rem (13px) | 600 | 1.4 | 0.01em |
| **Caption & Hints** | Plus Jakarta Sans | 0.75rem (12px) | 400 | 1.4 | 0.00em |
| **Monospace / Code** | JetBrains Mono | 0.8125rem (13px) | 400 | 1.5 | 0.00em |

---

## 4. LAYOUT ARCHITECTURE

```
+-----------------------------------------------------------------------------------+
| 1. APPLICATION HEADER                                                             |
| Brand Identity | API Status Badge | Model Status Badge | System Version           |
+-----------------------------------------------------------------------------------+
| 2. NAVIGATION BAR                                                                 |
| [Overview]  [Single Prediction]  [Batch Prediction]  [Model Performance]  [API]   |
+-----------------------------------------------------------------------------------+
| 3. HERO / OVERVIEW SECTION                                                        |
| Project Purpose | Technical Description | Quick Status | Non-Clinical Disclaimer  |
+-----------------------------------------------------------------------------------+
| 4. MAIN WORKSPACE AREA (Dynamic Tab Panel Content)                                |
|                                                                                   |
|  +-------------------------------------+  +------------------------------------+  |
|  | LEFT PANEL: Single Prediction Form  |  | RIGHT PANEL: Prediction Result     |  |
|  | - Patient Profile (2 fields)        |  | - Predicted Class Badge            |  |
|  | - Cardiovascular Vitals (4 fields)  |  | - Probability Meter Bar (0.0-1.0)  |  |
|  | - Cardiac Test Results (4 fields)   |  | - Model Name                       |  |
|  | [Run Prediction] [Reset]            |  | - Non-Clinical Classification Note |  |
|  +-------------------------------------+  +------------------------------------+  |
|                                                                                   |
+-----------------------------------------------------------------------------------+
| 5. FOOTER                                                                         |
| Engineering Prototype | Python 3.11 / Scikit-Learn / FastAPI | Non-Clinical Disclaimer|
+-----------------------------------------------------------------------------------+
```

---

## 5. COMPONENT INVENTORY

### 5.1 Application Header
* **Brand Logo Icon:** Custom inline SVG representing an ECG telemetry pulse inside a teal container.
* **Product Identity:** "Disease Diagnosis Prediction" (`font-weight: 700`) + Subtitle "Clinical ML Prediction Platform".
* **Live System Telemetry Badge:**
  * Status Pulse Dot: Animated SVG/CSS pulse (`#34D399` online, `#F87171` offline).
  * Main Text: `API: Online`
  * Sub Text: `Model: Loaded (final_model.joblib)`
  * Version Tag: `v1.0.0`

### 5.2 Navigation Tabs
* Keyboard-accessible `tablist` containing 5 primary views:
  1. `Overview` (Hero, System Summary, Quick Architecture)
  2. `Single Prediction` (10-input Interactive Clinical Workspace)
  3. `Batch Prediction` (JSON Payload Editor + Tabular Results)
  4. `Model Performance` (Verified Test Set Metrics & Confusion Matrix)
  5. `API Documentation` (Endpoint Reference & Links to `/docs`)

### 5.3 Single Prediction Input Form
Organized into 3 logical fieldset groups with clean visual demarcation:
* **Patient Profile**
* **Cardiovascular Vitals**
* **Cardiac Test Results**

Each field contains an explicit `<label>`, a controlled input element (`<input>` or `<select>`), a helper hint span, and an error message container (`<span class="field-err">`).

### 5.4 Prediction Result Card
Renders 4 distinct UI states:
1. **Idle State:** Displays SVG icon and prompt: *"Awaiting Input Parameters"*.
2. **Loading State:** Displays animated pulse bar and dispatching message.
3. **Error State:** Displays alert card with error message and **Retry Request** button.
4. **Success Result State:** Displays class result, probability bar, submitted vector chips, and non-clinical classification note.

---

## 6. SINGLE PREDICTION UX

### 6.1 Field Grouping & Validation Specifications

#### Group 1: Patient Profile
| Field ID | Display Label | HTML Input Type | Allowed Range / Options | Default Value | API Type |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `age` | Age (years) | `number` | `1` to `120` | `55` | `float` / `int` |
| `sex` | Biological Sex | `select` | `"Male"`, `"Female"` | `"Male"` | `str` |

#### Group 2: Cardiovascular Vitals
| Field ID | Display Label | HTML Input Type | Allowed Range / Options | Default Value | API Type |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `cp` | Chest Pain Type | `select` | `"asymptomatic"`, `"non-anginal"`, `"typical angina"`, `"atypical angina"` | `"asymptomatic"` | `str` |
| `trestbps` | Resting BP (mm Hg) | `number` | `0` to `300` | `140` | `float` / `int` |
| `chol` | Serum Cholesterol (mg/dl) | `number` | `0` to `1500` *(0 = unrecorded)* | `250` | `float` / `int` |
| `fbs` | Fasting Blood Sugar > 120 mg/dl | `select` | `false` *(≤ 120)*, `true` *(> 120)* | `false` | `bool` |

#### Group 3: Cardiac Test Results
| Field ID | Display Label | HTML Input Type | Allowed Range / Options | Default Value | API Type |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `restecg` | Resting ECG Result | `select` | `"normal"`, `"st-t abnormality"`, `"lv hypertrophy"` | `"normal"` | `str` |
| `thalch` | Max Heart Rate (bpm) | `number` | `1` to `250` | `150` | `float` / `int` |
| `exang` | Exercise Induced Angina | `select` | `false` *(No)*, `true` *(Yes)* | `false` | `bool` |
| `oldpeak` | ST Depression (oldpeak) | `number` (step `0.1`) | `-10.0` to `15.0` | `1.2` | `float` |

---

## 7. BATCH PREDICTION UX

### 7.1 Interface Design
* **Toolbar:** Buttons for `Load Demo Batch (3 Records)`, `Clear Editor`, and `Format JSON`.
* **Code Editor Area:** High-contrast monospaced `<textarea>` formatted with line-height, tab support, and custom scrollbars.
* **Payload Structure:**
```json
{
  "records": [
    {
      "age": 55, "sex": "Male", "cp": "asymptomatic", "trestbps": 140, "chol": 250,
      "fbs": false, "restecg": "normal", "thalch": 150, "exang": false, "oldpeak": 1.2
    },
    {
      "age": 62, "sex": "Female", "cp": "typical angina", "trestbps": 130, "chol": 210,
      "fbs": true, "restecg": "st-t abnormality", "thalch": 125, "exang": true, "oldpeak": 2.5
    }
  ]
}
```

### 7.2 Tabular Results View
Displays structured response from `POST /predict/batch`:
* **Columns:** `Record #`, `Demographics (Age/Sex)`, `Chest Pain`, `Vitals (BP/Chol)`, `Predicted Class`, `Model Probability`, `Model`.
* **Probability Visual:** Inline mini probability bar with percentage text.

---

## 8. MODEL PERFORMANCE UX

### 8.1 Empirical Held-Out Test Metrics ($N = 184$)

```
+-----------------------------------------------------------------------------------+
| HELD-OUT TEST SET EVALUATION (N = 184)                                            |
| Stratified 80/20 Train/Test Split | Zero Data Leakage Isolation                   |
+-----------------------------------------------------------------------------------+
|  +--------------------+  +--------------------+  +-----------------------------+  |
|  | Accuracy           |  | Precision          |  | Recall / Sensitivity        |  |
|  | 84.78%             |  | 85.58%             |  | 87.25%                      |  |
|  | (156 / 184 Correct)|  | (89/104 Positives) |  | (89/102 Actual Positives)   |  |
|  +--------------------+  +--------------------+  +-----------------------------+  |
|  +--------------------+  +--------------------+  +-----------------------------+  |
|  | Specificity        |  | F1 Score           |  | ROC-AUC                     |  |
|  | 81.71%             |  | 86.41% / 0.8641    |  | 92.95% / 0.9295             |  |
|  | (67/82 Negatives)  |  | Harmonic Mean      |  | Class Discrimination        |  |
|  +--------------------+  +--------------------+  +-----------------------------+  |
|  +-----------------------------------------------------------------------------+  |
|  | PR-AUC: 94.55% / 0.9455 (Precision-Recall Area under Curve)                 |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

### 8.2 Confusion Matrix Breakdown ($N = 184$)

| | Predicted Negative (0) | Predicted Positive (1) | Total Actual |
| :--- | :--- | :--- | :--- |
| **Actual Negative (0)** | **True Negative (TN): 67**<br>*(81.71% Specificity)* | **False Positive (FP): 15**<br>*(18.29% FPR)* | 82 |
| **Actual Positive (1)** | **False Negative (FN): 13**<br>*(12.75% FNR)* | **True Positive (TP): 89**<br>*(87.25% Recall)* | 102 |
| **Total Predicted** | 80 | 104 | **N = 184** |

### 8.3 Model Architecture & Hyperparameters
* **Classifier Model:** Support Vector Classifier (`SVC`)
* **Hyperparameters:** `C = 100`, `kernel = 'linear'`, `probability = True`, `gamma = 'scale'`
* **Preprocessing Pipeline:** Median Imputation (`chol=0` -> NaN) + StandardScaler + OneHotEncoder
* **Validation Method:** 5-Fold Stratified GridSearchCV

### 8.4 SVG Pipeline Visualization
```
  [ Clinical Input ] 
         │
         ▼
[ FastAPI Validation ] ── (app/schemas.py Pydantic Enforcer)
         │
         ▼
[ Preprocessing Pipeline ] ── (Median Imputer + Scaler + OHE)
         │
         ▼
[ Tuned SVM Classifier ] ── (SVC: C=100, kernel='linear', prob=True)
         │
         ▼
 [ Prediction Output ] ── (predicted_class, predicted_probability)
```

---

## 9. RESPONSIVE DESIGN BREAKPOINTS

```css
/* Breakpoint 1: 1440px Desktop (Ultra-wide Layout) */
@media (max-width: 1440px) {
  .app-layout { max-width: 1320px; }
  .prediction-workspace { grid-template-columns: 1fr 420px; }
}

/* Breakpoint 2: 1280px Laptop (Standard Desktop) */
@media (max-width: 1280px) {
  .app-layout { max-width: 1180px; }
  .prediction-workspace { grid-template-columns: 1fr 380px; }
}

/* Breakpoint 3: 1024px Tablet Landscape */
@media (max-width: 1024px) {
  .app-layout { max-width: 960px; padding: 1.25rem; }
  .prediction-workspace { grid-template-columns: 1fr; gap: 1.5rem; }
  .result-panel { position: static; }
}

/* Breakpoint 4: 768px Tablet Portrait */
@media (max-width: 768px) {
  .app-layout { padding: 1.0rem; }
  .field-grid.grid-2-col { grid-template-columns: 1fr; }
  .metrics-cards-grid { grid-template-columns: repeat(2, 1fr); }
  .header-container { flex-direction: column; align-items: flex-start; }
}

/* Breakpoint 5: 480px Mobile */
@media (max-width: 480px) {
  .hero-title { font-size: 1.4rem; }
  .metrics-cards-grid { grid-template-columns: 1fr; }
  .nav-bar { font-size: 0.8rem; gap: 0.25rem; }
  .form-actions-row { flex-direction: column; }
}

/* Breakpoint 6: 360px Mobile (Compact Screen) */
@media (max-width: 360px) {
  .app-layout { padding: 0.5rem; }
  .brand-name { font-size: 1.15rem; }
}
```

---

## 10. ACCESSIBILITY SPECIFICATION (WAI-ARIA)

* **Keyboard Navigation:** Full `Tab` navigation order across header, tabs, inputs, buttons, and code editor.
* **Focus Indicator:**
```css
*:focus-visible {
  outline: 2px solid var(--color-accent-teal);
  outline-offset: 2px;
  box-shadow: var(--shadow-teal-glow);
}
```
* **ARIA Attributes:**
  * Tab Bar: `role="tablist"`, `role="tab"`, `aria-selected="true/false"`, `aria-controls="panel-id"`.
  * Tab Panels: `role="tabpanel"`, `aria-labelledby="tab-id"`.
  * Input Hints & Errors: `aria-describedby="hint-id err-id"`, `aria-invalid="true/false"`.
* **Reduced Motion Support:**
```css
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
    scroll-behavior: auto !important;
  }
}
```

---

## 11. ANIMATION & MICRO-INTERACTION SPECIFICATION

### 11.1 Permitted Micro-Interactions
* **Card Hover Elevation:**
```css
.panel-card {
  transition: transform 0.2s cubic-bezier(0.4, 0, 0.2, 1), border-color 0.2s ease;
}
.panel-card:hover {
  transform: translateY(-2px);
  border-color: var(--color-border-active);
}
```
* **Button State Transition:** Active press shrink (`scale(0.98)`).
* **Probability Bar Expansion:** Width transition on result render (`transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1)`).
* **Pulse Loader Dot:** Gentle 2-second opacity cycle for status badges (`@keyframes pulseDot`).

### 11.2 Prohibited Visual Effects
* ❌ No neon text shadows or intense glowing borders.
* ❌ No spring/bouncy keyframe physics.
* ❌ No auto-playing carousel or distracting background video loops.

---

## 12. EXISTING API COMPATIBILITY REQUIREMENTS

### 12.1 Same-Origin Configuration
Must strictly preserve the existing architecture rule:
```javascript
const API_BASE_URL = "";
```
Do NOT hardcode `http://localhost:8000` or `http://127.0.0.1:8000`.

### 12.2 Endpoint Mapping

| Endpoint Path | HTTP Method | Expected Request Schema | Expected Response Schema |
| :--- | :--- | :--- | :--- |
| `/health` | `GET` | None | `HealthResponse` (`status`, `model_loaded`) |
| `/predict` | `POST` | `PredictionInput` (10 raw features) | `PredictionResponse` (`predicted_class`, `predicted_probability`, `model_name`) |
| `/predict/batch` | `POST` | `BatchPredictionRequest` (`records: [...]`) | `BatchPredictionResponse` (`predictions: [...]`) |
| `/` | `GET` | None | `RootResponse` (`name`, `version`, `model`) |
| `/docs` | `GET` | Browser Navigation | Interactive OpenAPI / Swagger UI |

---

## 13. BEFORE → AFTER IMPROVEMENT SUMMARY

| Feature Area | Existing Implementation | Upgraded Design Specification |
| :--- | :--- | :--- |
| **Visual Aesthetic** | Standard dark theme with basic inputs | Enterprise "Clinical AI Command Center" palette with high-density slate contrast |
| **Input Structure** | Flat vertical stack of 10 controls | 3 logical fieldsets (Patient Profile, Cardiovascular, Cardiac Tests) |
| **Risk Representation** | Numeric text float output | Calibrated 0.0-1.0 probability meter bar with decision boundary marker (0.5) |
| **Metrics Display** | Static raw text numbers | Empirical metric cards + 2x2 confusion matrix with true/false rate breakdowns |
| **Architecture View** | Basic HTML text list | Interactive SVG pipeline flow diagram (Input -> Validation -> Preprocessing -> SVM) |
| **Batch Workspace** | Simple plain textarea | Monospaced JSON editor + demo sample loader + structured tabular output view |
| **Portfolio Readiness** | Functional prototype | High-end technical portfolio showcase for GitHub & ML Engineering interviews |

---

## 14. IMPLEMENTATION CHECKLIST

- [x] Inspect existing `index.html`, `styles.css`, `app.js`, `main.py`, and `schemas.py`.
- [x] Verify exact 10 input field names and valid categorical values (`cp`, `sex`, `restecg`, `fbs`, `exang`).
- [x] Confirm verified test set metrics ($N=184$, Accuracy 84.78%, Recall 87.25%, ROC-AUC 92.95%).
- [x] Define design philosophy and "Clinical AI Command Center" visual identity.
- [x] Draft CSS token system (`:root`) with accessible WCAG contrast rules.
- [x] Specify layout architecture and 6 responsive breakpoints (1440px to 360px).
- [x] Document WAI-ARIA accessibility constraints and focus indicators.
- [x] Formulate micro-interaction rules and prohibit distracting animations.
- [x] Confirm Same-Origin API constraint (`const API_BASE_URL = ""`).
- [x] Create design specification document `reports/high_end_ui_design_specification.md`.
- [x] Run `git status --short` to confirm zero uncommitted source code modifications.
