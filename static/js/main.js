// static/js/main.js
document.addEventListener('DOMContentLoaded', function() {
    const predictBtn = document.getElementById('predict-btn');
    const findingText = document.getElementById('finding-text');
    
    if (predictBtn && findingText) {
        predictBtn.addEventListener('click', async function() {
            const finding = findingText.value.trim();
            
            if (!finding) {
                alert('Please enter a security finding to analyze.');
                return;
            }
            
            try {
                showLoading(true);
                
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ finding: finding })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    alert(data.error);
                } else {
                    displayResults(data);
                }
            } catch (error) {
                alert('Error connecting to the prediction service.');
                console.error('Prediction error:', error);
            } finally {
                showLoading(false);
            }
        });
    }
});

function showLoading(show) {
    const loading = document.getElementById('loading');
    if (loading) {
        loading.style.display = show ? 'block' : 'none';
    }
}

function displayResults(data) {
    const results = document.getElementById('results');
    if (results) {
        results.style.display = 'block';
        
        // Update predicted role
        const roleElement = document.getElementById('predicted-role');
        if (roleElement) {
            roleElement.textContent = `Predicted Role: ${data.predicted_role}`;
        }
        
        // Update confidence scores
        const confidenceList = document.getElementById('confidence-list');
        if (confidenceList && data.confidence_scores) {
            confidenceList.innerHTML = '';
            
            Object.entries(data.confidence_scores)
                .sort(([,a], [,b]) => b - a)
                .slice(0, 5)
                .forEach(([role, score]) => {
                    const div = document.createElement('div');
                    div.className = 'score-item';
                    div.innerHTML = `
                        <span class="role-name">${role}</span>
                        <span class="score-value">${(score * 100).toFixed(1)}%</span>
                    `;
                    confidenceList.appendChild(div);
                });
        }
        
        // Update keywords
        const keywordsList = document.getElementById('keywords-list');
        if (keywordsList && data.keywords_extracted) {
            keywordsList.innerHTML = '';
            data.keywords_extracted.slice(0, 10).forEach(([keyword, score]) => {
                const span = document.createElement('span');
                span.className = 'keyword-tag';
                span.textContent = keyword;
                keywordsList.appendChild(span);
            });
        }
    }
}