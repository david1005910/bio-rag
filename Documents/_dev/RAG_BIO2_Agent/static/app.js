const API_BASE = '/api/v1';
let authToken = localStorage.getItem('authToken');
let currentSessionId = null;

document.addEventListener('DOMContentLoaded', () => {
    checkAuth();
    loadStats();
    showSection('chat');
});

function getHeaders() {
    const headers = { 'Content-Type': 'application/json' };
    if (authToken) {
        headers['Authorization'] = `Bearer ${authToken}`;
    }
    return headers;
}

function checkAuth() {
    const user = localStorage.getItem('user');
    if (authToken && user) {
        document.getElementById('logged-out').classList.add('hidden');
        document.getElementById('logged-in').classList.remove('hidden');
        document.getElementById('user-name').textContent = JSON.parse(user).name || JSON.parse(user).email;
        loadChatSessions();
    } else {
        document.getElementById('logged-out').classList.remove('hidden');
        document.getElementById('logged-in').classList.add('hidden');
    }
}

function showModal(type) {
    document.getElementById('modal-overlay').classList.remove('hidden');
    document.getElementById('login-modal').classList.add('hidden');
    document.getElementById('register-modal').classList.add('hidden');
    document.getElementById(`${type}-modal`).classList.remove('hidden');
}

function hideModal() {
    document.getElementById('modal-overlay').classList.add('hidden');
}

async function handleLogin(e) {
    e.preventDefault();
    const email = document.getElementById('login-email').value;
    const password = document.getElementById('login-password').value;
    
    try {
        const response = await fetch(`${API_BASE}/auth/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || '로그인 실패');
        }
        
        const data = await response.json();
        authToken = data.access_token;
        localStorage.setItem('authToken', authToken);
        localStorage.setItem('user', JSON.stringify(data.user));
        hideModal();
        checkAuth();
    } catch (error) {
        document.getElementById('login-error').textContent = error.message;
        document.getElementById('login-error').classList.remove('hidden');
    }
}

async function handleRegister(e) {
    e.preventDefault();
    const name = document.getElementById('register-name').value;
    const email = document.getElementById('register-email').value;
    const password = document.getElementById('register-password').value;
    
    try {
        const response = await fetch(`${API_BASE}/auth/register`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, email, password })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || '회원가입 실패');
        }
        
        const data = await response.json();
        authToken = data.access_token;
        localStorage.setItem('authToken', authToken);
        localStorage.setItem('user', JSON.stringify(data.user));
        hideModal();
        checkAuth();
    } catch (error) {
        document.getElementById('register-error').textContent = error.message;
        document.getElementById('register-error').classList.remove('hidden');
    }
}

function logout() {
    authToken = null;
    localStorage.removeItem('authToken');
    localStorage.removeItem('user');
    checkAuth();
    newChat();
}

function showSection(section) {
    document.querySelectorAll('main > section').forEach(s => s.classList.add('hidden'));
    document.getElementById(`${section}-section`).classList.remove('hidden');
    
    document.querySelectorAll('.nav-btn').forEach(btn => btn.classList.remove('bg-gray-100'));
    event?.target?.closest('.nav-btn')?.classList.add('bg-gray-100');
}

async function searchPapers() {
    const query = document.getElementById('search-input').value;
    if (!query.trim()) return;
    
    const searchType = document.querySelector('input[name="search-type"]:checked').value;
    const resultsDiv = document.getElementById('search-results');
    resultsDiv.innerHTML = '<div class="text-center py-8"><i class="fas fa-spinner fa-spin text-3xl text-teal-600"></i></div>';
    
    try {
        const endpoint = searchType === 'semantic' ? 'semantic-search' : 'search';
        const response = await fetch(`${API_BASE}/papers/${endpoint}?query=${encodeURIComponent(query)}&limit=10`, {
            headers: getHeaders()
        });
        
        if (!response.ok) throw new Error('검색 실패');
        
        const papers = await response.json();
        
        if (papers.length === 0) {
            resultsDiv.innerHTML = '<p class="text-center text-gray-500 py-8">검색 결과가 없습니다.</p>';
            return;
        }
        
        resultsDiv.innerHTML = papers.map(paper => `
            <div class="bg-white rounded-lg shadow p-4 hover:shadow-lg transition">
                <h3 class="font-semibold text-lg text-gray-800 mb-2">${paper.title}</h3>
                <div class="flex flex-wrap gap-2 text-sm text-gray-500 mb-2">
                    <span><i class="fas fa-book mr-1"></i>${paper.journal || 'Unknown Journal'}</span>
                    <span><i class="fas fa-calendar mr-1"></i>${paper.publication_date || 'N/A'}</span>
                    <span><i class="fas fa-hashtag mr-1"></i>PMID: ${paper.pmid}</span>
                    ${paper.relevance ? `<span class="text-teal-600"><i class="fas fa-chart-line mr-1"></i>${(paper.relevance * 100).toFixed(1)}%</span>` : ''}
                </div>
                <p class="text-gray-600 text-sm line-clamp-3">${paper.abstract || paper.excerpt || 'No abstract available'}</p>
                <div class="mt-3 flex gap-2">
                    <a href="https://pubmed.ncbi.nlm.nih.gov/${paper.pmid}" target="_blank" 
                        class="text-sm text-teal-600 hover:text-teal-700">
                        <i class="fas fa-external-link-alt mr-1"></i>PubMed에서 보기
                    </a>
                    <button onclick="askAboutPaper('${paper.pmid}', \`${paper.title.replace(/`/g, "'")}\`)" 
                        class="text-sm text-teal-600 hover:text-teal-700">
                        <i class="fas fa-comments mr-1"></i>AI에게 질문하기
                    </button>
                </div>
            </div>
        `).join('');
        
    } catch (error) {
        resultsDiv.innerHTML = `<p class="text-center text-red-500 py-8">오류: ${error.message}</p>`;
    }
}

function askAboutPaper(pmid, title) {
    showSection('chat');
    document.getElementById('chat-input').value = `PMID:${pmid} 논문 "${title}"의 주요 내용과 연구 결과를 설명해주세요.`;
    document.getElementById('chat-input').focus();
}

async function sendMessage() {
    if (!authToken) {
        showModal('login');
        return;
    }
    
    const input = document.getElementById('chat-input');
    const question = input.value.trim();
    if (!question) return;
    
    const messagesDiv = document.getElementById('chat-messages');
    
    if (messagesDiv.querySelector('.text-center.text-gray-500')) {
        messagesDiv.innerHTML = '';
    }
    
    messagesDiv.innerHTML += `
        <div class="chat-message flex justify-end">
            <div class="bg-teal-600 text-white rounded-lg px-4 py-2 max-w-[80%]">
                ${question}
            </div>
        </div>
    `;
    
    input.value = '';
    
    messagesDiv.innerHTML += `
        <div id="typing-indicator" class="chat-message flex justify-start">
            <div class="bg-gray-200 rounded-lg px-4 py-3">
                <div class="typing-indicator flex space-x-1">
                    <span class="w-2 h-2 bg-gray-500 rounded-full"></span>
                    <span class="w-2 h-2 bg-gray-500 rounded-full"></span>
                    <span class="w-2 h-2 bg-gray-500 rounded-full"></span>
                </div>
            </div>
        </div>
    `;
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
    
    try {
        const reasoningMode = document.getElementById('reasoning-mode').checked;
        
        const response = await fetch(`${API_BASE}/chat/query`, {
            method: 'POST',
            headers: getHeaders(),
            body: JSON.stringify({ 
                question,
                session_id: currentSessionId,
                reasoning_mode: reasoningMode
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'AI 응답 실패');
        }
        
        const data = await response.json();
        currentSessionId = data.session_id;
        
        document.getElementById('typing-indicator')?.remove();
        
        let reasoningHtml = '';
        if (data.reasoning_steps && data.reasoning_steps.length > 0) {
            reasoningHtml = `
                <div class="mt-3 pt-3 border-t border-gray-200">
                    <details class="text-xs" open>
                        <summary class="cursor-pointer font-semibold text-teal-600 mb-2">
                            <i class="fas fa-brain mr-1"></i>추론 과정 보기 (${data.reasoning_steps.length} 단계)
                        </summary>
                        <div class="space-y-2 mt-2 pl-2 border-l-2 border-teal-200">
                            ${data.reasoning_steps.map(step => `
                                <div class="bg-gray-50 p-2 rounded">
                                    <div class="font-medium text-gray-700">
                                        <i class="fas fa-${step.type === 'decomposition' ? 'puzzle-piece' : step.type === 'sub_answer' ? 'search' : 'check-circle'} text-teal-500 mr-1"></i>
                                        ${step.description}
                                    </div>
                                    ${step.sub_question ? `<div class="text-gray-600 mt-1 italic">질문: ${step.sub_question}</div>` : ''}
                                    ${step.sources_found ? `<div class="text-gray-500">관련 문서: ${step.sources_found}개</div>` : ''}
                                    ${step.type === 'decomposition' && step.content ? `
                                        <div class="mt-2 p-2 bg-white rounded border text-gray-700">
                                            <div class="mb-1"><strong>복잡도:</strong> ${step.content.complexity || 'N/A'}</div>
                                            ${step.content.main_concepts && step.content.main_concepts.length ? `<div class="mb-1"><strong>핵심 개념:</strong> ${step.content.main_concepts.join(', ')}</div>` : ''}
                                            ${step.content.reasoning_approach ? `<div class="mb-1"><strong>접근 방식:</strong> ${step.content.reasoning_approach}</div>` : ''}
                                            ${step.content.sub_questions && step.content.sub_questions.length ? `<div><strong>하위 질문:</strong><ol class="list-decimal ml-4 mt-1">${step.content.sub_questions.map(q => `<li>${q}</li>`).join('')}</ol></div>` : ''}
                                        </div>
                                    ` : ''}
                                    ${step.type === 'sub_answer' && step.content ? `
                                        <details class="mt-2">
                                            <summary class="cursor-pointer text-teal-600">분석 결과 보기</summary>
                                            <div class="mt-1 p-2 bg-white rounded border text-gray-700">${formatMessage(step.content)}</div>
                                        </details>
                                    ` : ''}
                                </div>
                            `).join('')}
                        </div>
                    </details>
                </div>
            `;
        }
        
        let sourcesHtml = '';
        if (data.sources && data.sources.length > 0) {
            sourcesHtml = `
                <div class="mt-3 pt-3 border-t border-gray-200">
                    <p class="text-xs font-semibold text-gray-500 mb-2">출처:</p>
                    <div class="space-y-1">
                        ${data.sources.map(s => `
                            <a href="https://pubmed.ncbi.nlm.nih.gov/${s.pmid}" target="_blank" 
                                class="source-card block text-xs p-2 bg-gray-50 rounded hover:bg-gray-100 transition">
                                <span class="text-teal-600">[PMID: ${s.pmid}]</span> 
                                <span class="text-gray-700">${s.title}</span>
                                ${s.relevance ? `<span class="text-gray-400 ml-1">(${(s.relevance * 100).toFixed(0)}%)</span>` : ''}
                            </a>
                        `).join('')}
                    </div>
                </div>
            `;
        }
        
        messagesDiv.innerHTML += `
            <div class="chat-message flex justify-start">
                <div class="bg-gray-100 rounded-lg px-4 py-3 max-w-[85%]">
                    <div class="prose prose-sm max-w-none text-gray-800">${formatMessage(data.answer)}</div>
                    ${reasoningHtml}
                    ${sourcesHtml}
                    <div class="mt-2 text-xs text-gray-400">
                        신뢰도: ${(data.confidence * 100).toFixed(0)}%
                        ${data.reasoning_steps ? ' | <i class="fas fa-brain"></i> 추론 RAG' : ''}
                    </div>
                </div>
            </div>
        `;
        
        loadChatSessions();
        
    } catch (error) {
        document.getElementById('typing-indicator')?.remove();
        messagesDiv.innerHTML += `
            <div class="chat-message flex justify-start">
                <div class="bg-red-100 text-red-700 rounded-lg px-4 py-2">
                    오류: ${error.message}
                </div>
            </div>
        `;
    }
    
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function formatMessage(text) {
    return text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\[PMID:\s*(\d+)\]/g, '<a href="https://pubmed.ncbi.nlm.nih.gov/$1" target="_blank" class="text-teal-600 hover:underline">[PMID: $1]</a>')
        .replace(/\n/g, '<br>');
}

function newChat() {
    currentSessionId = null;
    document.getElementById('chat-messages').innerHTML = `
        <div class="text-center text-gray-500 py-8">
            <i class="fas fa-robot text-6xl text-teal-200 mb-4"></i>
            <p class="text-lg">바이오 연구 관련 질문을 해보세요!</p>
            <p class="text-sm mt-2">예: "CAR-T 세포치료의 최신 연구 동향은?" 또는 "CRISPR off-target 효과를 줄이는 방법은?"</p>
        </div>
    `;
}

async function loadChatSessions() {
    if (!authToken) return;
    
    try {
        const response = await fetch(`${API_BASE}/chat/sessions`, {
            headers: getHeaders()
        });
        
        if (!response.ok) return;
        
        const sessions = await response.json();
        const container = document.getElementById('chat-sessions');
        
        container.innerHTML = sessions.slice(0, 10).map(s => `
            <button onclick="loadSession('${s.id}')" 
                class="w-full text-left px-3 py-2 rounded text-sm hover:bg-gray-100 truncate ${s.id === currentSessionId ? 'bg-gray-100' : ''}">
                <i class="fas fa-comment-alt text-gray-400 mr-2"></i>${s.title}
            </button>
        `).join('');
        
    } catch (error) {
        console.error('Failed to load sessions:', error);
    }
}

async function loadSession(sessionId) {
    if (!authToken) return;
    
    try {
        const response = await fetch(`${API_BASE}/chat/sessions/${sessionId}`, {
            headers: getHeaders()
        });
        
        if (!response.ok) throw new Error('Failed to load session');
        
        const messages = await response.json();
        currentSessionId = sessionId;
        
        const messagesDiv = document.getElementById('chat-messages');
        messagesDiv.innerHTML = messages.map(m => {
            if (m.role === 'user') {
                return `
                    <div class="chat-message flex justify-end">
                        <div class="bg-teal-600 text-white rounded-lg px-4 py-2 max-w-[80%]">
                            ${m.content}
                        </div>
                    </div>
                `;
            } else {
                let sources = [];
                try {
                    sources = m.sources ? JSON.parse(m.sources) : [];
                } catch (e) {}
                
                let sourcesHtml = '';
                if (sources.length > 0) {
                    sourcesHtml = `
                        <div class="mt-3 pt-3 border-t border-gray-200">
                            <p class="text-xs font-semibold text-gray-500 mb-2">출처:</p>
                            <div class="space-y-1">
                                ${sources.map(s => `
                                    <a href="https://pubmed.ncbi.nlm.nih.gov/${s.pmid}" target="_blank" 
                                        class="block text-xs p-2 bg-gray-50 rounded hover:bg-gray-100">
                                        <span class="text-teal-600">[PMID: ${s.pmid}]</span> 
                                        <span class="text-gray-700">${s.title}</span>
                                    </a>
                                `).join('')}
                            </div>
                        </div>
                    `;
                }
                
                return `
                    <div class="chat-message flex justify-start">
                        <div class="bg-gray-100 rounded-lg px-4 py-3 max-w-[85%]">
                            <div class="prose prose-sm max-w-none text-gray-800">${formatMessage(m.content)}</div>
                            ${sourcesHtml}
                        </div>
                    </div>
                `;
            }
        }).join('');
        
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
        loadChatSessions();
        
    } catch (error) {
        console.error('Failed to load session:', error);
    }
}

async function indexPapers() {
    if (!authToken) {
        showModal('login');
        return;
    }
    
    const query = document.getElementById('index-query').value;
    const limit = document.getElementById('index-limit').value;
    
    if (!query.trim()) return;
    
    const statusDiv = document.getElementById('index-status');
    const indexBtn = document.getElementById('index-btn');
    
    indexBtn.disabled = true;
    indexBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>인덱싱 중...';
    statusDiv.innerHTML = '<p class="text-teal-600"><i class="fas fa-spinner fa-spin mr-2"></i>PubMed에서 논문을 가져오는 중...</p>';
    
    try {
        const response = await fetch(`${API_BASE}/papers/index-from-pubmed?query=${encodeURIComponent(query)}&limit=${limit}`, {
            method: 'POST',
            headers: getHeaders()
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || '인덱싱 실패');
        }
        
        const data = await response.json();
        statusDiv.innerHTML = `
            <div class="p-4 bg-green-50 rounded-lg text-green-700">
                <i class="fas fa-check-circle mr-2"></i>
                성공! ${data.papers_indexed}개의 논문이 인덱싱되었습니다.
            </div>
        `;
        loadStats();
        
    } catch (error) {
        statusDiv.innerHTML = `
            <div class="p-4 bg-red-50 rounded-lg text-red-700">
                <i class="fas fa-exclamation-circle mr-2"></i>
                오류: ${error.message}
            </div>
        `;
    } finally {
        indexBtn.disabled = false;
        indexBtn.innerHTML = '<i class="fas fa-plus-circle mr-2"></i>인덱싱';
    }
}

async function loadStats() {
    try {
        const response = await fetch('/api/v1/stats');
        const data = await response.json();
        document.getElementById('indexed-count').textContent = data.indexed_chunks || 0;
    } catch (error) {
        document.getElementById('indexed-count').textContent = '0';
    }
}

document.getElementById('modal-overlay').addEventListener('click', (e) => {
    if (e.target.id === 'modal-overlay') hideModal();
});
