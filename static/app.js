document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('analyze-form');
    const fileInput = document.getElementById('file-input');
    const fileNameSpan = document.getElementById('file-name');
    
    const statusSection = document.getElementById('status-section');
    const statusText = document.getElementById('status-text');
    const elapsedTime = document.getElementById('elapsed-time');
    
    const resultsSection = document.getElementById('results-section');
    const errorSection = document.getElementById('error-section');
    const errorText = document.getElementById('error-text');

    let pollInterval;

    // Update file name display
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            fileNameSpan.textContent = e.target.files[0].name;
            fileNameSpan.style.color = 'var(--accent-color)';
        } else {
            fileNameSpan.textContent = 'Choose a CSV, TSV, or DOCX file';
            fileNameSpan.style.color = '';
        }
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const file = fileInput.files[0];
        const query = document.getElementById('query').value;

        if (!file || !query) return;

        // Reset UI
        form.closest('.upload-section').classList.add('hidden');
        resultsSection.classList.add('hidden');
        errorSection.classList.add('hidden');
        statusSection.classList.remove('hidden');
        statusText.textContent = 'Uploading...';

        const formData = new FormData();
        formData.append('file', file);
        formData.append('query', query);

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'Failed to start analysis');
            }

            // Start polling
            statusText.textContent = 'Agent Team Processing...';
            pollJob(data.job_id);

        } catch (err) {
            showError(err.message);
        }
    });

    async function pollJob(jobId) {
        pollInterval = setInterval(async () => {
            try {
                const response = await fetch(`/jobs/${jobId}`);
                const data = await response.json();

                if (data.elapsed_seconds) {
                    elapsedTime.textContent = `Elapsed: ${data.elapsed_seconds}s`;
                }

                if (data.status === 'completed') {
                    clearInterval(pollInterval);
                    fetchResults(data.session_id);
                } else if (data.status === 'failed') {
                    clearInterval(pollInterval);
                    showError(data.error || 'Analysis failed during execution');
                }
            } catch (err) {
                clearInterval(pollInterval);
                showError('Lost connection to server');
            }
        }, 2000);
    }

    async function fetchResults(sessionId) {
        statusText.textContent = 'Rendering Results...';
        try {
            const response = await fetch(`/sessions/${sessionId}`);
            const data = await response.json();

            if (!response.ok) {
                throw new Error('Failed to fetch session results');
            }

            renderResults(data);
            
            statusSection.classList.add('hidden');
            resultsSection.classList.remove('hidden');
            
            // Allow new analysis
            form.closest('.upload-section').classList.remove('hidden');
            form.reset();
            fileNameSpan.textContent = 'Choose a CSV, TSV, or DOCX file';
            fileNameSpan.style.color = '';
            
        } catch (err) {
            showError(err.message);
        }
    }

    function renderResults(data) {
        // Dataset Info
        const infoGrid = document.getElementById('dataset-info');
        infoGrid.innerHTML = '';
        if (data.dataset_metadata && data.dataset_metadata.shape) {
            infoGrid.innerHTML += `
                <div class="info-item">
                    <div class="info-label">Rows</div>
                    <div class="info-value">${data.dataset_metadata.shape.rows || '?'}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Columns</div>
                    <div class="info-value">${data.dataset_metadata.shape.columns || '?'}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Inferred Domain</div>
                    <div class="info-value">${data.dataset_metadata.inferred_domain || 'General'}</div>
                </div>
            `;
        }

        // Analysis Plan
        const planList = document.getElementById('plan-list');
        planList.innerHTML = '';
        (data.analysis_plan || []).forEach(step => {
            const li = document.createElement('li');
            li.textContent = step;
            planList.appendChild(li);
        });

        // Steps
        const stepsContainer = document.getElementById('steps-container');
        stepsContainer.innerHTML = '';
        (data.steps || []).forEach((step, index) => {
            const stepCard = document.createElement('div');
            stepCard.className = 'card step-card';
            
            let html = `<h3>Step ${index + 1}: ${step.description}</h3>`;
            
            if (step.code) {
                html += `<div class="code-block">${escapeHtml(step.code)}</div>`;
            }
            
            if (step.output) {
                html += `<div class="output-block">${escapeHtml(step.output)}</div>`;
            }
            
            if (step.visualizations && step.visualizations.length > 0) {
                html += `<div class="vis-container">`;
                step.visualizations.forEach(vis => {
                    const filename = vis.split(/[\\/]/).pop();
                    html += `<img src="/visualizations/${filename}" alt="Visualization">`;
                });
                html += `</div>`;
            }
            
            stepCard.innerHTML = html;
            stepsContainer.appendChild(stepCard);
        });

        // Final Insights
        const insightsDiv = document.getElementById('final-insights');
        insightsDiv.innerHTML = data.final_insights ? escapeHtml(data.final_insights).replace(/\n/g, '<br>') : 'No final insights generated.';
    }

    function showError(message) {
        statusSection.classList.add('hidden');
        errorSection.classList.remove('hidden');
        errorText.textContent = message;
        form.closest('.upload-section').classList.remove('hidden');
    }

    function escapeHtml(unsafe) {
        return (unsafe || '').toString()
             .replace(/&/g, "&amp;")
             .replace(/</g, "&lt;")
             .replace(/>/g, "&gt;")
             .replace(/"/g, "&quot;")
             .replace(/'/g, "&#039;");
    }
});
