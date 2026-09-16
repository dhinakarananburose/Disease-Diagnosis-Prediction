/**
 * Disease Diagnosis Prediction — Client Application Architecture (Phase 22)
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
    elements.navTabs.forEach(tab => {
        tab.addEventListener("click", () => {
            const targetPanelId = tab.getAttribute("aria-controls");

            elements.navTabs.forEach(t => {
                t.classList.remove("active");
                t.setAttribute("aria-selected", "false");
            });
            elements.tabPanels.forEach(p => p.classList.add("hidden"));

            tab.classList.add("active");
            tab.setAttribute("aria-selected", "true");
            
            const targetPanel = document.getElementById(targetPanelId);
            if (targetPanel) targetPanel.classList.remove("hidden");
        });
    });
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

    // chol: 0..1500 (0 allowed and treated as missing by backend)
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
        setFieldError("oldpeak", "ST depression must be between -10.0 and 15.0 mm.");
        valid = false;
    }

    // cp canonical values
    const allowedCp = ["typical angina", "atypical angina", "non-anginal", "asymptomatic"];
    if (!allowedCp.includes(payload.cp)) {
        setFieldError("cp", "Invalid chest pain type selected.");
        valid = false;
    }

    // restecg canonical values
    const allowedRestEcg = ["normal", "st-t abnormality", "lv hypertrophy"];
    if (!allowedRestEcg.includes(payload.restecg)) {
        setFieldError("restecg", "Invalid resting ECG value selected.");
        valid = false;
    }

    return valid;
}

async function handleFormSubmission(e) {
    if (e && e.preventDefault) e.preventDefault();

    const formData = collectFormData();
    if (!validateForm(formData)) {
        return;
    }

    // Trigger Predict API Request
    await predict(formData);
}

async function predict(formData) {
    setButtonLoading(true);
    showResultState("loading");

    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            body: JSON.stringify(formData)
        });

        if (response.ok) {
            const result = await response.json();
            displayPrediction(result, formData);
            showResultState("success");
        } else {
            let errorDetail = `HTTP Error ${response.status}`;
            try {
                const errJson = await response.json();
                if (errJson.detail) {
                    errorDetail = typeof errJson.detail === "string" ? errJson.detail : JSON.stringify(errJson.detail);
                }
            } catch (_) {}
            displayError("Validation / API Error", errorDetail);
        }
    } catch (err) {
        displayError("API Connection Offline", `Could not connect to FastAPI server at ${API_BASE_URL}. Please ensure the server is active.`);
    } finally {
        setButtonLoading(false);
    }
}

function setButtonLoading(isLoading) {
    if (!elements.btnPredict) return;
    elements.btnPredict.disabled = isLoading;
    
    if (elements.predictBtnText) {
        elements.predictBtnText.textContent = isLoading ? "Analyzing..." : "Run Prediction";
    }
    if (elements.predictBtnIcon) {
        elements.predictBtnIcon.classList.toggle("hidden", isLoading);
    }
    if (elements.predictBtnSpinner) {
        elements.predictBtnSpinner.classList.toggle("hidden", !isLoading);
    }
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

function displayError(titleText, msgText) {
    if (elements.errorHeaderTitle) elements.errorHeaderTitle.textContent = titleText;
    if (elements.errorBodyMsg) elements.errorBodyMsg.textContent = msgText;
    showResultState("error");
}

/* ==========================================================================
   4. RENDER PREDICTION OUTPUT
   ========================================================================== */
function displayPrediction(result, inputPayload) {
    // Model Name
    if (elements.resModelNameText) {
        elements.resModelNameText.textContent = result.model_name || "Tuned Support Vector Machine";
    }

    // Predicted Class (0 or 1) & Probability
    const predClass = Number(result.predicted_class);
    const predProb = Number(result.predicted_probability);

    // Format probability percentage safely
    const probPct = isNaN(predProb) ? 0 : (predProb * 100);
    const probFormattedStr = probPct.toFixed(1) + "%";

    if (elements.resProbValueText) {
        elements.resProbValueText.textContent = probFormattedStr;
    }

    if (elements.resProbFillBar) {
        elements.resProbFillBar.style.width = `${Math.min(100, Math.max(0, probPct))}%`;
        if (predProb >= 0.5) {
            elements.resProbFillBar.classList.add("high-risk");
        } else {
            elements.resProbFillBar.classList.remove("high-risk");
        }
    }

    // Classification Box (Class 0 vs Class 1)
    if (elements.classResultCard) {
        if (predClass === 1) {
            elements.classResultCard.className = "class-result-card disease-present";
            elements.classIconCircle.innerHTML = `
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/>
                </svg>`;
            elements.resPredictedClassName.textContent = "Heart Disease Predicted";
            elements.resPredictedClassCode.textContent = "Predicted Class Label: 1 (Positive Class)";
        } else {
            elements.classResultCard.className = "class-result-card no-disease";
            elements.classIconCircle.innerHTML = `
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
                </svg>`;
            elements.resPredictedClassName.textContent = "No Heart Disease Predicted";
            elements.resPredictedClassCode.textContent = "Predicted Class Label: 0 (Negative Class)";
        }
    }

    // Render Submitted Vector Chips
    if (elements.submittedChipsFlex) {
        elements.submittedChipsFlex.innerHTML = `
            <span class="summary-chip">Age: ${inputPayload.age} y/o</span>
            <span class="summary-chip">Sex: ${inputPayload.sex}</span>
            <span class="summary-chip">CP: ${inputPayload.cp}</span>
            <span class="summary-chip">BP: ${inputPayload.trestbps} mm Hg</span>
            <span class="summary-chip">Chol: ${inputPayload.chol === 0 ? "0 (Unrecorded)" : inputPayload.chol + " mg/dl"}</span>
            <span class="summary-chip">FBS > 120: ${inputPayload.fbs ? "True" : "False"}</span>
            <span class="summary-chip">ECG: ${inputPayload.restecg}</span>
            <span class="summary-chip">Max HR: ${inputPayload.thalch} bpm</span>
            <span class="summary-chip">ExAngina: ${inputPayload.exang ? "True" : "False"}</span>
            <span class="summary-chip">Oldpeak: ${inputPayload.oldpeak}</span>
        `;
    }
}

function resetForm() {
    clearFieldErrors();

    // Reset controls to standard demo baseline values
    document.getElementById("age").value = 55;
    document.getElementById("sex").value = "Male";
    document.getElementById("cp").value = "asymptomatic";
    document.getElementById("trestbps").value = 140;
    document.getElementById("chol").value = 250;
    document.getElementById("fbs").value = "false";
    document.getElementById("restecg").value = "normal";
    document.getElementById("thalch").value = 150;
    document.getElementById("exang").value = "false";
    document.getElementById("oldpeak").value = 1.2;

    showResultState("idle");
    setButtonLoading(false);
}

/* ==========================================================================
   5. BATCH PREDICTION CONTROLLER
   ========================================================================== */
function initBatchHandlers() {
    if (elements.btnExecuteBatch) {
        elements.btnExecuteBatch.addEventListener("click", predictBatch);
    }
    if (elements.btnLoadBatchSample) {
        elements.btnLoadBatchSample.addEventListener("click", loadBatchDemoSample);
    }
    if (elements.btnClearBatchInput) {
        elements.btnClearBatchInput.addEventListener("click", () => {
            elements.batchJsonArea.value = "";
            elements.batchResultsBlock.classList.add("hidden");
        });
    }
}

const DEMO_BATCH_JSON = {
    "records": [
        {
            "age": 55, "sex": "Male", "cp": "asymptomatic", "trestbps": 140, "chol": 250,
            "fbs": false, "restecg": "normal", "thalch": 150, "exang": false, "oldpeak": 1.2
        },
        {
            "age": 41, "sex": "Female", "cp": "atypical angina", "trestbps": 120, "chol": 157,
            "fbs": false, "restecg": "normal", "thalch": 182, "exang": false, "oldpeak": 0.0
        },
        {
            "age": 67, "sex": "Male", "cp": "asymptomatic", "trestbps": 160, "chol": 286,
            "fbs": true, "restecg": "lv hypertrophy", "thalch": 108, "exang": true, "oldpeak": 1.5
        }
    ]
};

function loadBatchDemoSample() {
    if (elements.batchJsonArea) {
        elements.batchJsonArea.value = JSON.stringify(DEMO_BATCH_JSON, null, 2);
    }
}

async function predictBatch() {
    const rawJson = elements.batchJsonArea.value.trim();
    if (!rawJson) {
        alert("Please enter or load a valid batch JSON payload.");
        return;
    }

    let parsedJson;
    try {
        parsedJson = JSON.parse(rawJson);
    } catch (e) {
        alert("Syntax Error: Invalid JSON formatting.");
        return;
    }

    if (!parsedJson.records || !Array.isArray(parsedJson.records) || parsedJson.records.length === 0) {
        alert("Payload must contain a non-empty 'records' array.");
        return;
    }

    elements.btnExecuteBatch.disabled = true;
    const btnSpan = elements.btnExecuteBatch.querySelector("span");
    if (btnSpan) btnSpan.textContent = "Processing Batch...";

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
            const data = await response.json();
            renderBatchTable(data.predictions, parsedJson.records);
            elements.batchResultsBlock.classList.remove("hidden");
        } else {
            const errBody = await response.json().catch(() => ({}));
            alert(`Batch Prediction Error (${response.status}): ` + (errBody.detail || "Validation Error"));
        }
    } catch (err) {
        alert("Failed to connect to API backend for batch inference.");
    } finally {
        elements.btnExecuteBatch.disabled = false;
        if (btnSpan) btnSpan.textContent = "Execute Batch Inference";
    }
}

function renderBatchTable(predictions, records) {
    if (!elements.batchTableBody) return;
    elements.batchTableBody.innerHTML = "";

    predictions.forEach((pred, i) => {
        const rec = records[i] || {};
        const tr = document.createElement("tr");

        const predClass = Number(pred.predicted_class);
        const predProb = Number(pred.predicted_probability);
        const probStr = (isNaN(predProb) ? 0 : (predProb * 100)).toFixed(1) + "%";

        const textClass = predClass === 1 ? "status-rose-text" : "status-green-text";
        const labelText = predClass === 1 ? "1 (Heart Disease)" : "0 (No Disease)";

        tr.innerHTML = `
            <td>#${i + 1}</td>
            <td>${rec.age || "-"} y/o, ${rec.sex || "-"}</td>
            <td>${rec.cp || "-"}</td>
            <td>${rec.trestbps || "-"} / ${rec.chol !== undefined ? rec.chol : "-"}</td>
            <td class="${textClass}" style="font-weight: 600;">${labelText}</td>
            <td style="font-weight: 600;">${probStr}</td>
            <td style="font-size: 0.78rem; color: #94a3b8;">${pred.model_name || "SVM"}</td>
        `;
        elements.batchTableBody.appendChild(tr);
    });
}
