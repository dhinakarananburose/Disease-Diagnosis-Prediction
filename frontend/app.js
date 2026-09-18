/**
 * Disease Diagnosis Prediction — Client Application Architecture (Phase 27.2)
 * Consumes existing FastAPI backend endpoints (GET /health, POST /predict, POST /predict/batch).
 * Zero local ML probability calculations, zero third-party telemetry, zero credentials/storage.
 */

// 1. API BASE URL CONFIGURATION
const API_BASE_URL = "";

// DOM Element Registry
const elements = {
    // Header Status Badge
    statusPulseDot: document.getElementById("statusPulseDot"),
    statusMainText: document.getElementById("statusMainText"),
    statusSubText: document.getElementById("statusSubText"),
    diagApiStatus: document.getElementById("diagApiStatus"),
    diagModelLoaded: document.getElementById("diagModelLoaded"),

    // Navigation Tabs
    navTabs: document.querySelectorAll(".nav-tab"),
    tabPanels: document.querySelectorAll(".tab-panel"),

    // Single Prediction Form & Controls
    singlePredictionForm: document.getElementById("singlePredictionForm"),
    btnPredict: document.getElementById("btnPredict"),
    predictBtnText: document.getElementById("predictBtnText"),
    predictBtnIcon: document.getElementById("predictBtnIcon"),
    predictBtnSpinner: document.getElementById("predictBtnSpinner"),
    btnReset: document.getElementById("btnReset"),
    btnPresetLow: document.getElementById("btnPresetLow"),
    btnPresetHigh: document.getElementById("btnPresetHigh"),
    resultCardPanel: document.getElementById("resultCardPanel"),

    // Result Card State Containers
    stateIdle: document.getElementById("stateIdle"),
    stateLoading: document.getElementById("stateLoading"),
    stateError: document.getElementById("stateError"),
    stateSuccess: document.getElementById("stateSuccess"),
    btnRetryRequest: document.getElementById("btnRetryRequest"),
    errorHeaderTitle: document.getElementById("errorHeaderTitle"),
    errorBodyMsg: document.getElementById("errorBodyMsg"),

    // Result Output Controls
    resModelNameText: document.getElementById("resModelNameText"),
    classResultCard: document.getElementById("classResultCard"),
    classIconCircle: document.getElementById("classIconCircle"),
    resPredictedClassName: document.getElementById("resPredictedClassName"),
    resPredictedClassCode: document.getElementById("resPredictedClassCode"),
    resProbValueText: document.getElementById("resProbValueText"),
    resProbFillBar: document.getElementById("resProbFillBar"),
    submittedChipsFlex: document.getElementById("submittedChipsFlex"),

    // Batch Prediction Controls
    batchJsonArea: document.getElementById("batchJsonArea"),
    btnExecuteBatch: document.getElementById("btnExecuteBatch"),
    btnLoadBatchSample: document.getElementById("btnLoadBatchSample"),
    btnClearBatchInput: document.getElementById("btnClearBatchInput"),
    batchErrorBlock: document.getElementById("batchErrorBlock"),
    batchErrorHeader: document.getElementById("batchErrorHeader"),
    batchErrorBody: document.getElementById("batchErrorBody"),
    batchResultsBlock: document.getElementById("batchResultsBlock"),
    batchTableBody: document.getElementById("batchTableBody"),
};

// Global App State
let isApiOnline = false;

// Initialization Entry Point
document.addEventListener("DOMContentLoaded", () => {
    initNavigationTabs();
    initFormHandlers();
    initBatchHandlers();
    
    // Initial health check & periodic polling
    checkHealth();
    setInterval(checkHealth, 15000);
});

/* ==========================================================================
   1. SERVICE HEALTH CHECK & STATUS MONITORING
   ========================================================================== */
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`, {
            method: "GET",
            headers: { "Accept": "application/json" }
        });

        if (response.ok) {
            const data = await response.json();
            if (data.status === "healthy" && data.model_loaded) {
                updateHealthStatus(true, "API: Online", "Model Loaded (final_model.joblib)");
            } else {
                updateHealthStatus(false, "API: Warning", "Model Unloaded / Warning");
            }
        } else {
            updateHealthStatus(false, "API: Offline", `HTTP Status ${response.status}`);
        }
    } catch (err) {
        updateHealthStatus(false, "API: Offline", "Backend Unreachable");
    }
}

function updateHealthStatus(online, mainText, subText) {
    isApiOnline = online;

    if (elements.statusPulseDot) {
        elements.statusPulseDot.className = "status-pulse-dot " + (online ? "online" : "offline");
    }
    if (elements.statusMainText) elements.statusMainText.textContent = mainText;
    if (elements.statusSubText) elements.statusSubText.textContent = subText;

    if (elements.diagApiStatus) elements.diagApiStatus.textContent = online ? "Online" : "Offline / Unreachable";
    if (elements.diagModelLoaded) elements.diagModelLoaded.textContent = online ? "Loaded (final_model.joblib)" : "Unavailable";
}

/* ==========================================================================
   2. NAVIGATION TAB CONTROLLER
   ========================================================================== */
function initNavigationTabs() {
    elements.navTabs.forEach((tab, index) => {
        tab.addEventListener("click", () => switchTab(tab));
        
        tab.addEventListener("keydown", (e) => {
            let targetIndex = null;
            if (e.key === "ArrowRight") {
                targetIndex = (index + 1) % elements.navTabs.length;
            } else if (e.key === "ArrowLeft") {
                targetIndex = (index - 1 + elements.navTabs.length) % elements.navTabs.length;
            } else if (e.key === "Home") {
                targetIndex = 0;
            } else if (e.key === "End") {
                targetIndex = elements.navTabs.length - 1;
            }

            if (targetIndex !== null) {
                e.preventDefault();
                elements.navTabs[targetIndex].focus();
                switchTab(elements.navTabs[targetIndex]);
            }
        });
    });
}

function switchTab(selectedTab) {
    const targetPanelId = selectedTab.getAttribute("aria-controls");

    elements.navTabs.forEach(t => {
        t.classList.remove("active");
        t.setAttribute("aria-selected", "false");
    });
    elements.tabPanels.forEach(p => p.classList.add("hidden"));

    selectedTab.classList.add("active");
    selectedTab.setAttribute("aria-selected", "true");
    
    const targetPanel = document.getElementById(targetPanelId);
    if (targetPanel) targetPanel.classList.remove("hidden");
}

/* ==========================================================================
   3. SINGLE PREDICTION FORM CONTROLLER
   ========================================================================== */
function initFormHandlers() {
    if (elements.singlePredictionForm) {
        elements.singlePredictionForm.addEventListener("submit", handleFormSubmission);
    }
    if (elements.btnReset) {
        elements.btnReset.addEventListener("click", resetForm);
    }
    if (elements.btnRetryRequest) {
        elements.btnRetryRequest.addEventListener("click", handleFormSubmission);
    }
    if (elements.btnPresetLow) {
        elements.btnPresetLow.addEventListener("click", () => loadPresetProfile("low"));
    }
    if (elements.btnPresetHigh) {
        elements.btnPresetHigh.addEventListener("click", () => loadPresetProfile("high"));
    }
}

function loadPresetProfile(profileType) {
    clearFieldErrors();

    if (elements.btnPresetLow) elements.btnPresetLow.classList.remove("active");
    if (elements.btnPresetHigh) elements.btnPresetHigh.classList.remove("active");

    const data = (profileType === "low") ? {
        age: 42, sex: "Female", cp: "typical angina", trestbps: 120, chol: 195,
        fbs: "false", restecg: "normal", thalch: 165, exang: "false", oldpeak: 0.0
    } : {
        age: 65, sex: "Male", cp: "asymptomatic", trestbps: 150, chol: 260,
        fbs: "false", restecg: "st-t abnormality", thalch: 125, exang: "true", oldpeak: 2.5
    };

    document.getElementById("age").value = data.age;
    document.getElementById("sex").value = data.sex;
    document.getElementById("cp").value = data.cp;
    document.getElementById("trestbps").value = data.trestbps;
    document.getElementById("chol").value = data.chol;
    document.getElementById("fbs").value = data.fbs;
    document.getElementById("restecg").value = data.restecg;
    document.getElementById("thalch").value = data.thalch;
    document.getElementById("exang").value = data.exang;
    document.getElementById("oldpeak").value = data.oldpeak;

    if (profileType === "low" && elements.btnPresetLow) elements.btnPresetLow.classList.add("active");
    if (profileType === "high" && elements.btnPresetHigh) elements.btnPresetHigh.classList.add("active");
}

function collectFormData() {
    const ageVal = parseFloat(document.getElementById("age").value);
    const sexVal = document.getElementById("sex").value;
    const cpVal = document.getElementById("cp").value;
    const trestbpsVal = parseFloat(document.getElementById("trestbps").value);
    const cholVal = parseFloat(document.getElementById("chol").value);
    const fbsStr = document.getElementById("fbs").value;
    const fbsVal = (fbsStr === "true" || fbsStr === "True" || fbsStr === "1");
    const restecgVal = document.getElementById("restecg").value;
    const thalchVal = parseFloat(document.getElementById("thalch").value);
    const exangStr = document.getElementById("exang").value;
    const exangVal = (exangStr === "true" || exangStr === "True" || exangStr === "1");
    const oldpeakVal = parseFloat(document.getElementById("oldpeak").value);

    return {
        age: ageVal,
        sex: sexVal,
        cp: cpVal,
        trestbps: trestbpsVal,
        chol: cholVal,
        fbs: fbsVal,
        restecg: restecgVal,
        thalch: thalchVal,
        exang: exangVal,
        oldpeak: oldpeakVal
    };
}

function clearFieldErrors() {
    document.querySelectorAll(".field-err").forEach(el => el.textContent = "");
    document.querySelectorAll(".input-control").forEach(el => el.classList.remove("has-error"));
}

function setFieldError(fieldId, errorMsg) {
    const errSpan = document.getElementById(`err-${fieldId}`);
    if (errSpan) errSpan.textContent = errorMsg;
    const controlGroup = document.getElementById(`control-${fieldId}`);
    if (controlGroup) controlGroup.classList.add("has-error");
}

function validateForm(payload) {
    clearFieldErrors();
    let valid = true;

    // age: 1..120
    if (isNaN(payload.age) || payload.age < 1 || payload.age > 120) {
        setFieldError("age", "Age must be between 1 and 120.");
        valid = false;
    }

    // trestbps: 0..300
    if (isNaN(payload.trestbps) || payload.trestbps < 0 || payload.trestbps > 300) {
        setFieldError("trestbps", "Resting BP must be between 0 and 300 mm Hg.");
        valid = false;
    }

    // chol: 0..1500
    if (isNaN(payload.chol) || payload.chol < 0 || payload.chol > 1500) {
        setFieldError("chol", "Cholesterol must be between 0 and 1500 mg/dl.");
        valid = false;
    }

    // thalch: 1..250
    if (isNaN(payload.thalch) || payload.thalch < 1 || payload.thalch > 250) {
        setFieldError("thalch", "Max heart rate must be between 1 and 250 bpm.");
        valid = false;
    }

    // oldpeak: -10.0..15.0
    if (isNaN(payload.oldpeak) || payload.oldpeak < -10.0 || payload.oldpeak > 15.0) {
        setFieldError("oldpeak", "ST depression must be between -10.0 and 15.0.");
        valid = false;
    }

    return valid;
}

function showResultState(stateName) {
    elements.stateIdle.classList.add("hidden");
    elements.stateLoading.classList.add("hidden");
    elements.stateError.classList.add("hidden");
    elements.stateSuccess.classList.add("hidden");

    if (stateName === "idle") elements.stateIdle.classList.remove("hidden");
    if (stateName === "loading") elements.stateLoading.classList.remove("hidden");
    if (stateName === "error") elements.stateError.classList.remove("hidden");
    if (stateName === "success") elements.stateSuccess.classList.remove("hidden");
}

function setPredictButtonLoading(loading) {
    elements.btnPredict.disabled = loading;
    if (loading) {
        elements.predictBtnSpinner.classList.remove("hidden");
        elements.predictBtnIcon.classList.add("hidden");
        elements.predictBtnText.textContent = "Analyzing Vector...";
    } else {
        elements.predictBtnSpinner.classList.add("hidden");
        elements.predictBtnIcon.classList.remove("hidden");
        elements.predictBtnText.textContent = "Run Prediction";
    }
}

async function handleFormSubmission(event) {
    if (event) event.preventDefault();

    const payload = collectFormData();
    if (!validateForm(payload)) {
        return;
    }

    showResultState("loading");
    setPredictButtonLoading(true);

    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            body: JSON.stringify(payload)
        });

        if (response.ok) {
            const resultData = await response.json();
            renderPredictionResult(resultData, payload);
            showResultState("success");

            // Auto-scroll to result panel on mobile viewports (< 1024px)
            if (window.innerWidth < 1024 && elements.resultCardPanel) {
                const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
                elements.resultCardPanel.scrollIntoView({
                    behavior: prefersReducedMotion ? "auto" : "smooth",
                    block: "start"
                });
            }
        } else {
            let errorText = `HTTP Error ${response.status}`;
            try {
                const errJson = await response.json();
                if (errJson.detail) {
                    if (typeof errJson.detail === "string") errorText = errJson.detail;
                    else if (Array.isArray(errJson.detail)) {
                        errorText = errJson.detail.map(d => `${d.loc.join('.')}: ${d.msg}`).join("; ");
                    }
                }
            } catch (e) {}

            elements.errorHeaderTitle.textContent = "Validation or Inference Error";
            elements.errorBodyMsg.textContent = errorText;
            showResultState("error");
        }
    } catch (err) {
        elements.errorHeaderTitle.textContent = "Network / API Unavailable";
        elements.errorBodyMsg.textContent = "Failed to connect to the FastAPI inference service. Please verify server status.";
        showResultState("error");
    } finally {
        setPredictButtonLoading(false);
    }
}

function renderPredictionResult(result, payload) {
    const isDisease = (result.predicted_class === 1);
    const probPct = (result.predicted_probability * 100).toFixed(1);

    // Model Name
    if (elements.resModelNameText) {
        elements.resModelNameText.textContent = result.model_name || "Tuned Support Vector Machine";
    }

    // Class Badge & Styling
    if (elements.classResultCard) {
        elements.classResultCard.className = "class-result-card " + (isDisease ? "positive" : "negative");
    }

    if (elements.classIconCircle) {
        if (isDisease) {
            elements.classIconCircle.innerHTML = `
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">
                    <path d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/>
                </svg>`;
        } else {
            elements.classIconCircle.innerHTML = `
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">
                    <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
                </svg>`;
        }
    }

    if (elements.resPredictedClassName) {
        elements.resPredictedClassName.textContent = isDisease 
            ? "Heart Disease Risk Detected" 
            : "No Heart Disease Predicted";
    }

    if (elements.resPredictedClassCode) {
        elements.resPredictedClassCode.textContent = `Predicted Class Label: ${result.predicted_class}`;
    }

    // Probability Gauge Bar
    if (elements.resProbValueText) {
        elements.resProbValueText.textContent = `${probPct}%`;
    }

    if (elements.resProbFillBar) {
        const clampedProb = Math.min(Math.max(result.predicted_probability, 0), 1);
        elements.resProbFillBar.style.width = `${(clampedProb * 100).toFixed(1)}%`;
    }

    // Vector Chips Rendering
    if (elements.submittedChipsFlex) {
        elements.submittedChipsFlex.innerHTML = `
            <div class="chip-item">Age: <strong>${payload.age}</strong></div>
            <div class="chip-item">Sex: <strong>${payload.sex}</strong></div>
            <div class="chip-item">CP: <strong>${payload.cp}</strong></div>
            <div class="chip-item">BP: <strong>${payload.trestbps} mm Hg</strong></div>
            <div class="chip-item">Chol: <strong>${payload.chol} mg/dl</strong></div>
            <div class="chip-item">FBS: <strong>${payload.fbs ? '>120' : '≤120'}</strong></div>
            <div class="chip-item">ECG: <strong>${payload.restecg}</strong></div>
            <div class="chip-item">MaxHR: <strong>${payload.thalch} bpm</strong></div>
            <div class="chip-item">ExAng: <strong>${payload.exang ? 'Yes' : 'No'}</strong></div>
            <div class="chip-item">Oldpeak: <strong>${payload.oldpeak}</strong></div>
        `;
    }
}

function resetForm() {
    if (elements.singlePredictionForm) {
        elements.singlePredictionForm.reset();
    }
    if (elements.btnPresetLow) elements.btnPresetLow.classList.remove("active");
    if (elements.btnPresetHigh) elements.btnPresetHigh.classList.remove("active");
    clearFieldErrors();
    showResultState("idle");
}

/* ==========================================================================
   4. BATCH PREDICTION CONTROLLER
   ========================================================================== */
function initBatchHandlers() {
    if (elements.btnLoadBatchSample) {
        elements.btnLoadBatchSample.addEventListener("click", loadBatchDemoSample);
    }
    if (elements.btnClearBatchInput) {
        elements.btnClearBatchInput.addEventListener("click", clearBatchInput);
    }
    if (elements.btnExecuteBatch) {
        elements.btnExecuteBatch.addEventListener("click", handleBatchSubmission);
    }
}

function loadBatchDemoSample() {
    const samplePayload = {
        records: [
            { age: 55, sex: "Male", cp: "asymptomatic", trestbps: 140, chol: 250, fbs: false, restecg: "normal", thalch: 150, exang: false, oldpeak: 1.2 },
            { age: 62, sex: "Female", cp: "typical angina", trestbps: 130, chol: 210, fbs: true, restecg: "st-t abnormality", thalch: 125, exang: true, oldpeak: 2.5 },
            { age: 44, sex: "Male", cp: "non-anginal", trestbps: 120, chol: 180, fbs: false, restecg: "normal", thalch: 172, exang: false, oldpeak: 0.0 }
        ]
    };
    if (elements.batchJsonArea) {
        elements.batchJsonArea.value = JSON.stringify(samplePayload, null, 2);
    }
    hideBatchError();
}

function clearBatchInput() {
    if (elements.batchJsonArea) elements.batchJsonArea.value = "";
    if (elements.batchResultsBlock) elements.batchResultsBlock.classList.add("hidden");
    hideBatchError();
}

function showBatchError(headerText, bodyText) {
    if (elements.batchErrorHeader) elements.batchErrorHeader.textContent = headerText;
    if (elements.batchErrorBody) elements.batchErrorBody.textContent = bodyText;
    if (elements.batchErrorBlock) elements.batchErrorBlock.classList.remove("hidden");
}

function hideBatchError() {
    if (elements.batchErrorBlock) elements.batchErrorBlock.classList.add("hidden");
}

async function handleBatchSubmission() {
    hideBatchError();
    const rawText = elements.batchJsonArea ? elements.batchJsonArea.value.trim() : "";

    if (!rawText) {
        showBatchError("Empty Batch Payload", "Please enter or load a valid JSON batch request structure.");
        return;
    }

    let parsedJson = null;
    try {
        parsedJson = JSON.parse(rawText);
    } catch (e) {
        showBatchError("Invalid JSON Syntax", "Syntax error parsing JSON payload. Check for missing quotes or trailing commas.");
        return;
    }

    if (!parsedJson.records || !Array.isArray(parsedJson.records) || parsedJson.records.length === 0) {
        showBatchError("Invalid Batch Schema", "JSON payload must contain a top-level 'records' array with at least 1 record.");
        return;
    }

    elements.btnExecuteBatch.disabled = true;

    try {
        const response = await fetch(`${API_BASE_URL}/predict/batch`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            body: JSON.stringify(parsedJson)
        });

        if (response.ok) {
            const batchResult = await response.json();
            renderBatchResults(batchResult.predictions, parsedJson.records);
            if (elements.batchResultsBlock) elements.batchResultsBlock.classList.remove("hidden");
        } else {
            let errorText = `HTTP Error ${response.status}`;
            try {
                const errJson = await response.json();
                if (errJson.detail) {
                    if (typeof errJson.detail === "string") errorText = errJson.detail;
                    else if (Array.isArray(errJson.detail)) {
                        errorText = errJson.detail.map(d => `${d.loc.join('.')}: ${d.msg}`).join("; ");
                    }
                }
            } catch (e) {}

            showBatchError("Batch Inference Request Error", errorText);
        }
    } catch (err) {
        showBatchError("Network Error", "Unable to connect to FastAPI `/predict/batch` endpoint.");
    } finally {
        elements.btnExecuteBatch.disabled = false;
    }
}

function renderBatchResults(predictions, originalRecords) {
    if (!elements.batchTableBody) return;
    elements.batchTableBody.innerHTML = "";

    predictions.forEach((pred, idx) => {
        const rec = originalRecords[idx] || {};
        const isDisease = (pred.predicted_class === 1);
        const probPct = (pred.predicted_probability * 100).toFixed(1);

        const row = document.createElement("tr");
        row.innerHTML = `
            <td><strong>#${idx + 1}</strong></td>
            <td>${rec.age || '-'} / ${rec.sex || '-'}</td>
            <td>${rec.cp || '-'}</td>
            <td>${rec.trestbps || '-'} / ${rec.chol || '-'}</td>
            <td>
                <span class="badge-class ${isDisease ? 'pos' : 'neg'}">
                    ${isDisease ? 'Disease Present (1)' : 'No Disease (0)'}
                </span>
            </td>
            <td><strong>${probPct}%</strong></td>
            <td><code style="font-size: 0.75rem;">${pred.model_name || 'Tuned SVM'}</code></td>
        `;
        elements.batchTableBody.appendChild(row);
    });
}
