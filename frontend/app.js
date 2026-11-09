const featureNames = Array.from({ length: 28 }, (_, index) => `V${index + 1}`);
const form = document.querySelector("#transaction-form");
const featureGrid = document.querySelector("#feature-grid");
const submitButton = document.querySelector("#submit-button");
const formError = document.querySelector("#form-error");
const emptyResult = document.querySelector("#empty-result");
const resultContent = document.querySelector("#result-content");
const loadingState = document.querySelector("#loading-state");
const sampleStatus = document.querySelector("#sample-status");
let activeModelThreshold = 0.5;
let demoExamplesPromise;

for (const feature of featureNames) {
    const label = document.createElement("label");
    label.className = "feature-field";
    label.htmlFor = feature;

    const name = document.createElement("span");
    name.textContent = feature;

    const input = document.createElement("input");
    input.id = feature;
    input.name = feature;
    input.type = "number";
    input.step = "any";
    input.value = "0";
    input.required = true;
    input.setAttribute("aria-label", `${feature} feature value`);

    label.append(name, input);
    featureGrid.append(label);
}

document.querySelectorAll("[data-sample]").forEach((button) => {
    button.addEventListener("click", async () => {
        sampleStatus.textContent = "Loading sample…";

        try {
            demoExamplesPromise ??= fetch("/ui/demo_examples.json").then((response) => {
                if (!response.ok) throw new Error("Could not load demo examples.");
                return response.json();
            });

            const examples = await demoExamplesPromise;
            const sampleName = button.dataset.sample;
            const sample = examples[sampleName];

            for (const [name, value] of Object.entries(sample)) {
                form.elements[name].value = String(value);
            }

            sampleStatus.textContent = `${sampleName === "fraud" ? "Fraud" : "Normal"} dataset example loaded. Submit it to see the model result.`;
            formError.hidden = true;
        } catch (error) {
            sampleStatus.textContent = "";
            showError(error.message || "Could not load the sample transaction.");
        }
    });
});

function formatPercent(value) {
    return `${(Number(value) * 100).toFixed(2)}%`;
}

function showError(message) {
    formError.textContent = message;
    formError.hidden = false;
}

function renderReasons(reasons) {
    const container = document.querySelector("#reasons-list");
    container.replaceChildren();

    const maxImpact = Math.max(...reasons.map((reason) => Math.abs(reason.impact)), 0.0001);

    for (const reason of reasons) {
        const row = document.createElement("div");
        row.className = `reason-row ${reason.impact < 0 ? "negative" : "positive"}`;

        const feature = document.createElement("span");
        feature.className = "reason-name";
        feature.textContent = reason.feature;

        const track = document.createElement("span");
        track.className = "reason-track";
        const bar = document.createElement("span");
        bar.style.width = `${(Math.abs(reason.impact) / maxImpact) * 100}%`;
        track.append(bar);

        const impact = document.createElement("span");
        impact.className = "reason-impact";
        impact.textContent = Number(reason.impact).toFixed(4);

        row.append(feature, track, impact);
        container.append(row);
    }
}

function renderResult(result) {
    const predictionRow = document.querySelector(".prediction-row");
    const probabilityTrack = document.querySelector(".probability-track");
    const probability = Number(result.probability);
    const isFraud = result.prediction === "Fraud";

    document.querySelector("#prediction-label").textContent = result.prediction;
    document.querySelector("#prediction-icon").textContent = isFraud ? "!" : "✓";
    document.querySelector("#probability-value").textContent = formatPercent(probability);
    document.querySelector("#probability-bar").style.width = `${Math.min(probability * 100, 100)}%`;
    document.querySelector("#threshold-note").textContent =
        `Decision threshold: ${(Number(result.threshold) * 100).toFixed(1)}%`;

    predictionRow.classList.toggle("is-fraud", isFraud);
    probabilityTrack.classList.toggle("is-fraud", isFraud);
    renderReasons(result.top_reasons);

    emptyResult.hidden = true;
    resultContent.hidden = false;
}

async function loadModelInfo() {
    const response = await fetch("/fraud/model-info");
    if (!response.ok) throw new Error("Could not load active model information.");

    const info = await response.json();
    activeModelThreshold = Number(info.threshold);
    document.querySelector("#model-version").textContent = info.version;
    document.querySelector("#model-precision").textContent = formatPercent(info.precision);
    document.querySelector("#model-recall").textContent = formatPercent(info.recall);
    document.querySelector("#model-f1").textContent = formatPercent(info.f1);
    document.querySelector("#model-threshold").textContent = formatPercent(info.threshold);
    submitButton.disabled = false;
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    formError.hidden = true;
    submitButton.disabled = true;
    emptyResult.hidden = true;
    resultContent.hidden = true;
    loadingState.hidden = false;

    const payload = Object.fromEntries(
        [...new FormData(form)].map(([key, value]) => [key, Number(value)])
    );

    try {
        const response = await fetch("/fraud/explain", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        const result = await response.json();
        if (!response.ok) {
            const detail = Array.isArray(result.detail)
                ? result.detail.map((item) => item.msg).join("; ")
                : result.detail;
            throw new Error(detail || "The API could not analyze this transaction.");
        }

        renderResult({ ...result, threshold: activeModelThreshold });
    } catch (error) {
        emptyResult.hidden = false;
        showError(error.message || "Could not connect to the API.");
    } finally {
        loadingState.hidden = true;
        submitButton.disabled = false;
    }
});

loadModelInfo().catch((error) => {
    document.querySelector("#model-version").textContent = "Unavailable";
    showError(error.message);
});
