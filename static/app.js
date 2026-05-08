document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadStatus = document.getElementById('upload-status');
    const statusText = uploadStatus.querySelector('.status-text');
    const apiStatus = document.getElementById('api-status');
    
    const chatHistory = document.getElementById('chat-history');
    const queryForm = document.getElementById('query-form');
    const queryInput = document.getElementById('query-input');
    const sendButton = document.getElementById('send-button');
    
    // RBAC Elements
    const tenantInput = document.getElementById('tenant-input');
    const accessInput = document.getElementById('access-input');

    // --- System Status Check ---
    async function checkStatus() {
        try {
            const res = await fetch('/status');
            if (res.ok) {
                apiStatus.className = 'stat-value online';
                apiStatus.innerHTML = '● Online';
            } else {
                throw new Error('API down');
            }
        } catch (e) {
            apiStatus.className = 'stat-value error';
            apiStatus.innerHTML = '● Offline';
            apiStatus.style.color = 'var(--error)';
            apiStatus.style.textShadow = '0 0 10px rgba(239, 68, 68, 0.5)';
        }
    }
    
    // Check status immediately and every 30s
    checkStatus();
    setInterval(checkStatus, 30000);

    // --- File Upload Handling ---
    
    // Click to browse
    dropZone.addEventListener('click', () => fileInput.click());

    // Drag and drop events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    });

    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    async function handleFiles(files) {
        if (files.length === 0) return;
        
        // Take the first file (backend currently handles 1 by 1 nicely in this UI)
        const file = files[0];
        if (!file.name.endsWith('.pdf') && !file.name.endsWith('.csv')) {
            alert("Only PDF and CSV files are supported.");
            return;
        }

        uploadFile(file);
    }

    async function uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('tenant_id', tenantInput.value.trim() || 'default');
        formData.append('access_level', accessInput.value);

        // UI Update
        uploadStatus.classList.remove('hidden');
        statusText.innerText = `Uploading & processing ${file.name}...`;

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (response.ok) {
                statusText.innerText = `✅ ${file.name} ready!`;
                setTimeout(() => {
                    uploadStatus.classList.add('hidden');
                }, 3000);
            } else {
                throw new Error(data.detail || 'Upload failed');
            }
        } catch (error) {
            console.error('Error:', error);
            statusText.innerText = `❌ Error: ${error.message}`;
            statusText.style.color = 'var(--error)';
        }
    }

    // --- Chat Handling ---

    queryForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = queryInput.value.trim();
        if (!query) return;

        // 1. Add user message to UI
        addMessage(query, 'user');
        queryInput.value = '';
        
        // Disable input while processing
        queryInput.disabled = true;
        sendButton.disabled = true;

        // 2. Add loading AI message
        const loadingId = addMessage('...', 'ai', true);

        try {
            const tenantId = tenantInput.value.trim() || 'default';
            const accessLevel = accessInput.value;
            const userToken = `${tenantId}_${accessLevel}`;
            
            const response = await fetch('/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-User-Token': userToken
                },
                body: JSON.stringify({ 
                    query: query,
                    hybrid_top_k: 20,
                    rerank_top_k: 5,
                    confidence_threshold: 0.4
                })
            });

            const data = await response.json();
            
            if (response.ok) {
                // If it's an OOD query (Out of Distribution), it returns a standard message
                // but the structure might vary based on the backend implementation.
                // Assuming `results` contains the LLM answer directly if it's a string,
                // or a list of chunks if we modified it. 
                // Let's assume the API returns `data.results` as the final answer string from the LLM.
                
                // Wait, the current backend returns `data.results` as a list of dicts from the retriever!
                // Ah, the user didn't actually implement the full LLM generation endpoint in main.py yet,
                // it just returns the reranked chunks from `retriever.search`.
                // Let's format the retrieved chunks nicely.
                
                updateMessage(loadingId, data.results);
            } else {
                throw new Error(data.detail || 'Query failed');
            }
        } catch (error) {
            updateMessage(loadingId, `Error: ${error.message}`, true);
        } finally {
            queryInput.disabled = false;
            sendButton.disabled = false;
            queryInput.focus();
        }
    });

    function addMessage(text, sender, isLoading = false) {
        const id = 'msg-' + Date.now();
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${sender}-message`;
        msgDiv.id = id;

        const avatar = sender === 'user' ? '👤' : '✨';
        
        msgDiv.innerHTML = `
            <div class="avatar">${avatar}</div>
            <div class="content ${isLoading ? 'loading' : ''}">
                ${escapeHTML(text)}
            </div>
        `;

        chatHistory.appendChild(msgDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
        
        return id;
    }

    function updateMessage(id, data, isError = false) {
        const msgDiv = document.getElementById(id);
        if (!msgDiv) return;
        
        const contentDiv = msgDiv.querySelector('.content');
        contentDiv.classList.remove('loading');
        
        if (isError) {
            contentDiv.innerHTML = `<span style="color: var(--error)">${escapeHTML(data)}</span>`;
            return;
        }

        // Handle the backend response format. 
        // If it's a string (e.g. OOD message), display it.
        if (typeof data === 'string') {
            contentDiv.innerHTML = formatText(data);
            return;
        }

        // If the backend returns our new structured JSON Analytical Engine format
        if (data && data.answer !== undefined) {
            let html = `<p>${formatText(data.answer)}</p>`;
            
            // Render confidence score badge
            if (data.confidence_score !== undefined) {
                const score = (data.confidence_score * 100).toFixed(0);
                let colorClass = 'success';
                if (score < 50) colorClass = 'error';
                else if (score < 80) colorClass = 'warning';
                
                html += `<div style="margin-top: 1rem; padding-top: 0.8rem; border-top: 1px solid rgba(255,255,255,0.1); display: flex; gap: 1rem; align-items: center; flex-wrap: wrap;">`;
                html += `<span style="font-size: 0.75rem; padding: 0.2rem 0.6rem; border-radius: 12px; background: rgba(255,255,255,0.1); border: 1px solid var(--border-color);">Confidence: <strong style="color: var(--${colorClass})">${score}%</strong></span>`;
                
                // Render citations if present
                if (data.citations && Array.isArray(data.citations) && data.citations.length > 0) {
                    html += `<span style="font-size: 0.75rem; color: var(--text-secondary);">Citations: `;
                    data.citations.forEach((cit) => {
                        html += `<span style="background: rgba(59, 130, 246, 0.2); color: var(--accent-primary); padding: 0.1rem 0.4rem; border-radius: 4px; margin-right: 0.4rem; font-family: monospace;">${cit.substring(0,6)}...</span>`;
                    });
                    html += `</span>`;
                }
                
                html += `</div>`;
            }
            
            contentDiv.innerHTML = html;
            
        } else if (Array.isArray(data)) {
            if (data.length === 0) {
                contentDiv.innerHTML = "I couldn't find any relevant information in the documents.";
                return;
            }

            let html = `<p><strong>Retrieved ${data.length} relevant chunks:</strong></p>`;
            data.forEach((item, index) => {
                html += `
                    <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.1)">
                        <p style="font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 0.5rem;">
                            📄 ${item.metadata?.source || 'Unknown'} (Score: ${item.score?.toFixed(3) || 'N/A'})
                        </p>
                        <p>${formatText(item.text || JSON.stringify(item))}</p>
                    </div>
                `;
            });
            contentDiv.innerHTML = html;
        } else {
            contentDiv.innerHTML = "<pre>" + JSON.stringify(data, null, 2) + "</pre>";
        }
        
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    function escapeHTML(str) {
        return str.replace(/[&<>'"]/g, 
            tag => ({
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                "'": '&#39;',
                '"': '&quot;'
            }[tag])
        );
    }
    
    function formatText(text) {
        // Simple formatting to handle newlines
        return escapeHTML(text).replace(/\n/g, '<br>');
    }
});
