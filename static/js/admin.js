// static/js/admin.js
document.addEventListener('DOMContentLoaded', function() {
    loadStats();
    
    const retrainBtn = document.getElementById('retrain-btn');
    const uploadForm = document.getElementById('upload-form');
    
    if (retrainBtn) {
        retrainBtn.addEventListener('click', async function() {
            if (confirm('Are you sure you want to retrain the model?')) {
                try {
                    retrainBtn.textContent = 'Retraining...';
                    retrainBtn.disabled = true;
                    
                    const response = await fetch('/retrain', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        }
                    });
                    
                    const data = await response.json();
                    alert(data.message || 'Model retrained successfully!');
                    
                } catch (error) {
                    alert('Error retraining model.');
                    console.error(error);
                } finally {
                    retrainBtn.textContent = 'Retrain Model';
                    retrainBtn.disabled = false;
                }
            }
        });
    }
    
    if (uploadForm) {
        uploadForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const fileInput = document.getElementById('file-input');
            const file = fileInput.files[0];
            
            if (!file) {
                alert('Please select a file to upload.');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                alert(data.message || 'File uploaded successfully!');
                
            } catch (error) {
                alert('Error uploading file.');
                console.error(error);
            }
        });
    }
});

async function loadStats() {
    try {
        const response = await fetch('/stats');
        const stats = await response.json();
        
        document.getElementById('total-predictions').textContent = stats.total_predictions || '0';
        document.getElementById('model-accuracy').textContent = (stats.accuracy * 100).toFixed(1) + '%' || 'N/A';
        document.getElementById('training-size').textContent = stats.training_size || '0';
        
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}