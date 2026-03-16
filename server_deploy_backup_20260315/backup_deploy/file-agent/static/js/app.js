let currentPath = '';
let files = [];
let selectedFiles = [];
let pendingUploadFiles = [];
let isAuthenticated = false;
let mounts = [];
let currentMount = null;
let isReadonlyPath = false;
let selectedMoveTarget = '';

async function checkAuth() {
    try {
        const res = await fetch('/api/auth/check');
        if (res.ok) {
            isAuthenticated = true;
            const data = await res.json();
            showUserInfo(data.user);
            return true;
        } else {
            isAuthenticated = false;
            window.location.href = '/login';
            return false;
        }
    } catch (e) {
        isAuthenticated = false;
        window.location.href = '/login';
        return false;
    }
}

function showUserInfo(user) {
    const headerRight = document.querySelector('.header-right');
    if (headerRight) {
        const userInfo = document.createElement('span');
        userInfo.style.cssText = 'color:var(--text-dim);font-size:13px;margin-right:10px;';
        userInfo.innerHTML = `👤 已登录`;
        headerRight.insertBefore(userInfo, headerRight.firstChild);
    }
}

async function apiCall(url, options = {}) {
    if (!options.headers) {
        options.headers = {};
    }
    options.credentials = 'include';
    
    const res = await fetch(url, options);
    
    if (res.status === 401) {
        isAuthenticated = false;
        window.location.href = '/login';
        return null;
    }
    
    return res;
}

async function loadFiles(path = '') {
    if (!isAuthenticated) return;
    
    currentPath = path;
    selectedFiles = [];
    updateSelectionBar();
    
    try {
        const res = await apiCall(`/api/files?path=${encodeURIComponent(path)}`);
        if (!res) return;
        
        const data = await res.json();
        files = data.files || [];
        renderBreadcrumb(data.breadcrumb || []);
        renderFiles(files);
    } catch (e) {
        showToast('加载失败', 'error');
    }
}

function renderBreadcrumb(items) {
    const el = document.getElementById('breadcrumb');
    el.innerHTML = items.map((item, i) => {
        const isLast = i === items.length - 1;
        return `<span class="breadcrumb-item ${isLast ? 'current' : ''}" data-path="${escapeAttr(item.path)}" onclick="navigateToPath(this)">${item.name}</span>${!isLast ? '<span>›</span>' : ''}`;
    }).join('');
}

function renderFiles(files) {
    const el = document.getElementById('file-list');
    
    if (files.length === 0) {
        el.innerHTML = '<div style="padding:40px;text-align:center;color:var(--text-dim);">空文件夹</div>';
        return;
    }
    
    el.innerHTML = files.map(f => `
        <div class="file-item ${selectedFiles.includes(f.path) ? 'selected' : ''}" data-path="${escapeAttr(f.path)}">
            <label class="checkbox"><input type="checkbox" ${selectedFiles.includes(f.path) ? 'checked' : ''} onchange="toggleSelect('${escapeAttr(f.path)}')"></label>
            <div class="file-name" onclick="${f.is_dir ? `navigateToPath(this)` : `previewFileByEl(this)`}">
                <span class="file-icon">${getFileIcon(f.ext, f.is_dir)}</span>
                ${escapeHtml(f.name)}
                ${f.mount_name ? `<span class="mount-tag">${escapeHtml(f.mount_name)}</span>` : ''}
            </div>
            <div class="file-size">${f.is_dir ? '-' : formatSize(f.size)}</div>
            <div class="file-date">${formatDate(f.modified_at)}</div>
            <div class="file-actions">
                ${!f.is_dir ? `<button class="btn-sm" onclick="downloadByPath('${escapeAttr(f.path)}')">📥</button>` : ''}
                ${!isReadonlyPath ? `<button class="btn-sm btn-danger" onclick="deleteByPath('${escapeAttr(f.path)}')">🗑️</button>` : ''}
            </div>
        </div>
    `).join('');
}

function escapeAttr(str) {
    return str.replace(/\\/g, '\\\\').replace(/'/g, "\\'").replace(/"/g, '\\"');
}

function navigateToPath(el) {
    const path = el.closest('.file-item').dataset.path.replace(/\\\\/g, '\\');
    navigateTo(path);
}

function previewFileByEl(el) {
    const path = el.closest('.file-item').dataset.path.replace(/\\\\/g, '\\');
    previewFile(path);
}

function downloadByPath(path) {
    downloadFile(path.replace(/\\\\/g, '\\'));
}

function deleteByPath(path) {
    deleteFile(path.replace(/\\\\/g, '\\'));
}

function getFileIcon(ext, isDir) {
    if (isDir) return '📁';
    const icons = {
        '.pdf': '📕', '.doc': '📘', '.docx': '📘', '.xls': '📗', '.xlsx': '📗',
        '.ppt': '📙', '.pptx': '📙', '.txt': '📄', '.md': '📝',
        '.jpg': '🖼️', '.jpeg': '🖼️', '.png': '🖼️', '.gif': '🖼️', '.webp': '🖼️',
        '.mp4': '🎬', '.mov': '🎬', '.avi': '🎬', '.mkv': '🎬',
        '.mp3': '🎵', '.wav': '🎵', '.flac': '🎵',
        '.zip': '📦', '.rar': '📦', '.7z': '📦',
        '.py': '🐍', '.js': '⚡', '.html': '🌐', '.css': '🎨', '.json': '🔧'
    };
    return icons[ext] || '📄';
}

function formatSize(bytes) {
    if (!bytes) return '-';
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    return (bytes / (1024 * 1024 * 1024)).toFixed(1) + ' GB';
}

function formatDate(date) {
    if (!date) return '-';
    const d = new Date(date);
    return d.toLocaleDateString() + ' ' + d.toLocaleTimeString().slice(0, 5);
}

function escapeHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function navigateTo(path) {
    currentMount = null;
    isReadonlyPath = false;
    
    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
    
    if (!path) {
        document.getElementById('nav-root')?.classList.add('active');
    }
    
    const mount = mounts.find(m => path.startsWith(m.path));
    if (mount) {
        isReadonlyPath = mount.readonly;
        currentMount = mount;
    }
    
    updateToolbarForReadonly();
    loadFiles(path);
}

function toggleSelect(path) {
    path = path.replace(/\\\\/g, '\\');
    const idx = selectedFiles.indexOf(path);
    if (idx === -1) {
        selectedFiles.push(path);
    } else {
        selectedFiles.splice(idx, 1);
    }
    renderFiles(files);
    updateSelectionBar();
}

function toggleSelectAll() {
    const allPaths = files.map(f => f.path);
    if (selectedFiles.length === allPaths.length) {
        selectedFiles = [];
    } else {
        selectedFiles = [...allPaths];
    }
    renderFiles(files);
    updateSelectionBar();
}

function updateSelectionBar() {
    const bar = document.getElementById('selection-bar');
    const count = document.getElementById('selected-count');
    count.textContent = selectedFiles.length;
    bar.classList.toggle('visible', selectedFiles.length > 0);
}

function clearSelection() {
    selectedFiles = [];
    renderFiles(files);
    updateSelectionBar();
}

function showModal(id) { document.getElementById(id).classList.add('visible'); }
function closeModal(id) { document.getElementById(id).classList.remove('visible'); }

function showUploadModal() {
    if (!isAuthenticated) {
        showToast('请先登录', 'error');
        return;
    }
    if (isReadonlyPath) {
        showToast('挂载目录为只读，不允许上传', 'error');
        return;
    }
    pendingUploadFiles = [];
    document.getElementById('upload-list').innerHTML = '';
    showModal('upload-modal');
}

document.getElementById('upload-area')?.addEventListener('click', () => {
    document.getElementById('file-input').click();
});

document.getElementById('file-input')?.addEventListener('change', (e) => {
    handleFiles(e.target.files);
});

document.getElementById('upload-area')?.addEventListener('dragover', (e) => {
    e.preventDefault();
    e.currentTarget.classList.add('dragover');
});

document.getElementById('upload-area')?.addEventListener('dragleave', (e) => {
    e.currentTarget.classList.remove('dragover');
});

document.getElementById('upload-area')?.addEventListener('drop', (e) => {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');
    handleFiles(e.dataTransfer.files);
});

function handleFiles(fileList) {
    for (const file of fileList) {
        pendingUploadFiles.push(file);
        const item = document.createElement('div');
        item.className = 'upload-item';
        item.innerHTML = `<span class="name">${escapeHtml(file.name)}</span><span class="size">${formatSize(file.size)}</span>`;
        document.getElementById('upload-list').appendChild(item);
    }
}

async function doUploadFiles() {
    if (!isAuthenticated) {
        showToast('请先登录', 'error');
        return;
    }

    if (pendingUploadFiles.length === 0) {
        showToast('请选择文件', 'error');
        return;
    }

    // 立即提示，确认按钮已生效
    showToast(`开始上传 ${pendingUploadFiles.length} 个文件...`, '');

    const formData = new FormData();
    pendingUploadFiles.forEach(file => formData.append('files', file));
    formData.append('target_path', currentPath);
    
    try {
        const res = await apiCall('/api/files/upload', { method: 'POST', body: formData });
        if (!res) return;
        
        let data;
        try {
            data = await res.json();
        } catch (e) {
            showToast('上传失败：返回数据格式异常', 'error');
            return;
        }
        
        if (data.success_count > 0) {
            showToast(`上传成功 ${data.success_count} 个文件`, 'success');
            closeModal('upload-modal');
            loadFiles(currentPath);
        } else {
            showToast('上传失败', 'error');
        }
    } catch (e) {
        showToast('上传失败: ' + (e.message || ''), 'error');
    }
}

function showNewFolderModal() {
    if (!isAuthenticated) {
        showToast('请先登录', 'error');
        return;
    }
    if (isReadonlyPath) {
        showToast('挂载目录为只读，不允许创建文件夹', 'error');
        return;
    }
    showModal('new-folder-modal');
}

async function createFolder() {
    const name = document.getElementById('folder-name').value.trim();
    if (!name) {
        showToast('请输入文件夹名称', 'error');
        return;
    }
    
    try {
        const formData = new FormData();
        formData.append('name', name);
        formData.append('parent_path', currentPath);
        
        const res = await apiCall('/api/files/folder', { method: 'POST', body: formData });
        if (!res) return;
        
        const data = await res.json();
        
        if (data.success) {
            showToast('创建成功', 'success');
            closeModal('new-folder-modal');
            loadFiles(currentPath);
            document.getElementById('folder-name').value = '';
        } else {
            showToast(data.detail || '创建失败', 'error');
        }
    } catch (e) {
        showToast('创建失败', 'error');
    }
}

async function downloadFile(path) {
    window.location.href = `/api/files/download?paths=${encodeURIComponent(path)}`;
}

async function downloadSelected() {
    if (selectedFiles.length === 0) return;
    window.location.href = `/api/files/download?paths=${encodeURIComponent(selectedFiles.join(','))}`;
}

async function deleteFile(path) {
    if (!confirm('确定要删除吗？文件将移入回收站')) return;

    try {
        const res = await apiCall(`/api/files?paths=${encodeURIComponent(path)}`, { method: 'DELETE' });
        if (!res) return;
        
        const data = await res.json();
        
        if (data.success_count > 0) {
            showToast('已移入回收站', 'success');
            loadFiles(currentPath);
        } else if (data.failed && data.failed.length > 0) {
            showToast(data.failed[0].error || '删除失败', 'error');
        }
    } catch (e) {
        showToast('删除失败', 'error');
    }
}

async function deleteSelected() {
    if (selectedFiles.length === 0) return;
    if (!confirm(`确定要删除选中的 ${selectedFiles.length} 个文件吗？`)) return;

    try {
        const res = await apiCall(`/api/files?paths=${encodeURIComponent(selectedFiles.join(','))}`, { method: 'DELETE' });
        if (!res) return;
        
        const data = await res.json();
        
        if (data.success_count > 0) {
            showToast(`已删除 ${data.success_count} 个文件`, 'success');
            loadFiles(currentPath);
            clearSelection();
        } else if (data.failed && data.failed.length > 0) {
            showToast(data.failed[0].error || '删除失败', 'error');
        }
    } catch (e) {
        showToast('删除失败', 'error');
    }
}

async function previewFile(path) {
    try {
        const res = await apiCall(`/api/preview?path=${encodeURIComponent(path)}`);
        if (!res) return;
        
        const data = await res.json();
        
        const title = document.getElementById('preview-title');
        const body = document.getElementById('preview-body');
        
        title.textContent = data.filename || '预览';
        
        if (data.type === 'image') {
            if (data.data) {
                body.innerHTML = `<img src="${data.data}" alt="${data.filename}" style="max-width:100%;max-height:80vh;">`;
            } else if (data.url) {
                body.innerHTML = `<img src="${data.url}" alt="${data.filename}">`;
            } else {
                body.innerHTML = `<p style="color:var(--text-dim);">无法加载图片</p>`;
            }
        } else if (data.type === 'video') {
            body.innerHTML = `<video src="${data.url}" controls autoplay></video>`;
        } else if (data.type === 'audio') {
            body.innerHTML = `<audio src="${data.url}" controls autoplay></audio>`;
        } else if (data.type === 'text') {
            body.innerHTML = `<pre>${escapeHtml(data.content)}</pre>`;
        } else if (data.type === 'pdf') {
            body.innerHTML = `<embed src="${data.url}" type="application/pdf" width="100%" height="500px">`;
        } else {
            body.innerHTML = `<p style="color:var(--text-dim);">不支持预览此类型文件</p>`;
        }
        
        showModal('preview-modal');
    } catch (e) {
        showToast('预览失败', 'error');
    }
}

async function searchFiles() {
    const keyword = document.getElementById('search-input').value.trim();
    if (!keyword) return;
    
    try {
        const res = await apiCall(`/api/search?keyword=${encodeURIComponent(keyword)}`);
        if (!res) return;
        
        const data = await res.json();
        
        files = data.results || [];
        renderBreadcrumb([{ name: `搜索: ${keyword}`, path: '' }]);
        renderFiles(files);
    } catch (e) {
        showToast('搜索失败', 'error');
    }
}

async function searchByType(type) {
    try {
        const res = await apiCall(`/api/search/type/${type}`);
        if (!res) return;
        
        const data = await res.json();
        
        files = data.results || [];
        const typeNames = { image: '图片', video: '视频', document: '文档', audio: '音频' };
        renderBreadcrumb([{ name: typeNames[type] || type, path: '' }]);
        renderFiles(files);
    } catch (e) {
        showToast('加载失败', 'error');
    }
}

function showStarred() {
    showToast('收藏功能开发中', '');
}

async function showTrash() {
    try {
        const res = await apiCall('/api/trash');
        if (!res) return;
        
        const data = await res.json();
        
        const trashItems = (data.items || []).map(item => ({
            id: item.id,
            name: item.name,
            path: item.trash_path,
            original_path: item.original_path,
            is_dir: item.is_dir,
            size: item.size,
            ext: null,
            modified_at: item.deleted_at
        }));
        
        renderBreadcrumb([{ name: '回收站', path: '' }]);
        renderTrashFiles(trashItems);
    } catch (e) {
        showToast('加载失败', 'error');
    }
}

function renderTrashFiles(items) {
    const el = document.getElementById('file-list');
    
    if (items.length === 0) {
        el.innerHTML = '<div style="padding:40px;text-align:center;color:var(--text-dim);">回收站是空的</div>';
        return;
    }
    
    el.innerHTML = `
        <div style="padding:10px;margin-bottom:10px;border-bottom:1px solid var(--border);">
            <button class="btn btn-danger" onclick="emptyTrash()">🗑️ 清空回收站</button>
        </div>
    ` + items.map(item => `
        <div class="file-item" data-id="${item.id}">
            <div class="file-name">
                <span class="file-icon">${item.is_dir ? '📁' : '📄'}</span>
                ${escapeHtml(item.name)}
                <span style="color:var(--text-dim);font-size:12px;margin-left:8px;">原位置: ${escapeHtml(item.original_path)}</span>
            </div>
            <div class="file-size">${item.size ? formatSize(item.size) : '-'}</div>
            <div class="file-date">${formatDate(item.modified_at)}</div>
            <div class="file-actions">
                <button class="btn-sm" onclick="restoreFile(${item.id})" title="恢复">↩️ 恢复</button>
                <button class="btn-sm btn-danger" onclick="deletePermanently(${item.id})" title="永久删除">🗑️</button>
            </div>
        </div>
    `).join('');
}

async function restoreFile(id) {
    if (!confirm('确定要恢复这个文件吗？')) return;
    
    try {
        const res = await apiCall('/api/trash/restore', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ids: [id] })
        });
        if (!res) return;
        
        const data = await res.json();
        
        if (data.success_count > 0) {
            showToast('恢复成功', 'success');
            showTrash();
        } else {
            showToast(data.failed?.[0]?.error || '恢复失败', 'error');
        }
    } catch (e) {
        showToast('恢复失败', 'error');
    }
}

async function deletePermanently(id) {
    if (!confirm('确定要永久删除吗？此操作不可恢复！')) return;
    
    try {
        const res = await apiCall('/api/trash', {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ids: [id] })
        });
        if (!res) return;
        
        const data = await res.json();
        
        if (data.success_count > 0) {
            showToast('已永久删除', 'success');
            showTrash();
        } else {
            showToast('删除失败', 'error');
        }
    } catch (e) {
        showToast('删除失败', 'error');
    }
}

async function emptyTrash() {
    if (!confirm('确定要清空回收站吗？此操作不可恢复！')) return;
    
    try {
        const res = await apiCall('/api/trash/empty', {
            method: 'DELETE'
        });
        if (!res) return;
        
        const data = await res.json();
        showToast(`已清空 ${data.cleared_count} 个文件`, 'success');
        showTrash();
    } catch (e) {
        showToast('清空失败', 'error');
    }
}

function toggleAgent() {
    document.getElementById('agent-panel').classList.toggle('visible');
}

async function sendAgentMsg(text) {
    if (!isAuthenticated) {
        showToast('请先登录', 'error');
        return;
    }
    
    const input = document.getElementById('agent-input');
    const msg = text || input.value.trim();
    if (!msg) return;
    
    input.value = '';
    
    const messagesEl = document.getElementById('agent-messages');
    messagesEl.innerHTML += `<div class="agent-msg user">${escapeHtml(msg)}</div>`;
    
    try {
        const res = await apiCall('/api/agent/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: msg, context: { current_path: currentPath } })
        });
        
        if (!res) return;
        
        const data = await res.json();
        messagesEl.innerHTML += `<div class="agent-msg">${escapeHtml(data.reply)}</div>`;
        messagesEl.scrollTop = messagesEl.scrollHeight;
        
        if (data.files && data.files.length > 0) {
            files = data.files;
            renderFiles(files);
        }
    } catch (e) {
        messagesEl.innerHTML += `<div class="agent-msg">请求失败，请重试</div>`;
    }
}

function showMoveModal() {
    if (selectedFiles.length === 0) return;
    selectedMoveTarget = '';
    loadFolderTree();
    showModal('move-modal');
}

async function loadFolderTree() {
    const el = document.getElementById('folder-tree');
    el.innerHTML = '<div style="padding:10px;color:var(--text-dim);">加载中...</div>';
    
    try {
        const res = await apiCall('/api/files/folders');
        if (!res) return;
        
        const data = await res.json();
        renderFolderTree(data.folders || []);
    } catch (e) {
        el.innerHTML = '<div style="padding:10px;color:var(--text-dim);">加载失败</div>';
    }
}

function renderFolderTree(folders, indent = 0) {
    const el = document.getElementById('folder-tree');
    
    if (folders.length === 0 && indent === 0) {
        el.innerHTML = '<div style="padding:10px;color:var(--text-dim);">没有可用的文件夹</div>';
        return;
    }
    
    if (indent === 0) {
        el.innerHTML = `
            <div class="folder-item ${selectedMoveTarget === '' ? 'selected' : ''}" 
                 onclick="selectMoveTarget('')" style="padding:8px;cursor:pointer;border-radius:4px;">
                🏠 根目录
            </div>
        `;
    }
    
    folders.forEach(f => {
        const isSelected = selectedMoveTarget === f.path;
        const isDisabled = selectedFiles.includes(f.path);
        
        el.innerHTML += `
            <div class="folder-item ${isSelected ? 'selected' : ''} ${isDisabled ? 'disabled' : ''}" 
                 data-path="${escapeAttr(f.path)}"
                 onclick="${isDisabled ? '' : 'selectMoveTargetByEl(this)'}"
                 style="padding:8px;padding-left:${16 + indent * 16}px;cursor:${isDisabled ? 'not-allowed' : 'pointer'};border-radius:4px;opacity:${isDisabled ? 0.5 : 1};">
                📁 ${escapeHtml(f.name)} ${isDisabled ? '(当前选择)' : ''}
            </div>
        `;
        
        if (f.children && f.children.length > 0) {
            renderFolderTree(f.children, indent + 1);
        }
    });
}

function selectMoveTargetByEl(el) {
    const path = el.dataset.path ? el.dataset.path.replace(/\\\\/g, '\\') : '';
    selectedMoveTarget = path;
    document.querySelectorAll('.folder-item').forEach(item => {
        item.classList.remove('selected');
        item.style.background = '';
    });
    el.classList.add('selected');
    el.style.background = 'var(--primary-light)';
}

async function moveFiles() {
    if (selectedMoveTarget === null || selectedMoveTarget === undefined) {
        showToast('请选择目标文件夹', 'error');
        return;
    }
    
    try {
        const res = await apiCall(`/api/files/move?paths=${encodeURIComponent(selectedFiles.join(','))}&target=${encodeURIComponent(selectedMoveTarget)}`, {
            method: 'POST'
        });
        if (!res) return;
        
        const data = await res.json();
        
        if (data.count > 0) {
            showToast(`已移动 ${data.count} 个文件`, 'success');
            closeModal('move-modal');
            clearSelection();
            loadFiles(currentPath);
        } else {
            showToast('移动失败', 'error');
        }
    } catch (e) {
        showToast('移动失败', 'error');
    }
}

let selectedCopyTarget = '';
let renameTargetPath = '';

function showCopyModal() {
    if (selectedFiles.length === 0) return;
    selectedCopyTarget = '';
    loadCopyFolderTree();
    showModal('copy-modal');
}

async function loadCopyFolderTree() {
    const el = document.getElementById('copy-folder-tree');
    el.innerHTML = '<div style="padding:10px;color:var(--text-dim);">加载中...</div>';
    
    try {
        const res = await apiCall('/api/files/folders');
        if (!res) return;
        
        const data = await res.json();
        renderCopyFolderTree(data.folders || []);
    } catch (e) {
        el.innerHTML = '<div style="padding:10px;color:var(--text-dim);">加载失败</div>';
    }
}

function renderCopyFolderTree(folders, indent = 0) {
    const el = document.getElementById('copy-folder-tree');
    
    if (folders.length === 0 && indent === 0) {
        el.innerHTML = '<div style="padding:10px;color:var(--text-dim);">没有可用的文件夹</div>';
        return;
    }
    
    if (indent === 0) {
        el.innerHTML = `
            <div class="folder-item ${selectedCopyTarget === '' ? 'selected' : ''}" 
                 onclick="selectCopyTarget('')" style="padding:8px;cursor:pointer;border-radius:4px;">
                🏠 根目录
            </div>
        `;
    }
    
    folders.forEach(f => {
        const isSelected = selectedCopyTarget === f.path;
        const isDisabled = selectedFiles.includes(f.path);
        
        el.innerHTML += `
            <div class="folder-item ${isSelected ? 'selected' : ''} ${isDisabled ? 'disabled' : ''}" 
                 data-path="${escapeAttr(f.path)}"
                 onclick="${isDisabled ? '' : 'selectCopyTargetByEl(this)'}"
                 style="padding:8px;padding-left:${16 + indent * 16}px;cursor:${isDisabled ? 'not-allowed' : 'pointer'};border-radius:4px;opacity:${isDisabled ? 0.5 : 1};">
                📁 ${escapeHtml(f.name)} ${isDisabled ? '(当前选择)' : ''}
            </div>
        `;
        
        if (f.children && f.children.length > 0) {
            renderCopyFolderTree(f.children, indent + 1);
        }
    });
}

function selectCopyTargetByEl(el) {
    const path = el.dataset.path ? el.dataset.path.replace(/\\\\/g, '\\') : '';
    selectedCopyTarget = path;
    document.querySelectorAll('#copy-folder-tree .folder-item').forEach(item => {
        item.classList.remove('selected');
        item.style.background = '';
    });
    el.classList.add('selected');
    el.style.background = 'var(--primary-light)';
}

function selectCopyTarget(path) {
    selectedCopyTarget = path;
    document.querySelectorAll('#copy-folder-tree .folder-item').forEach(item => {
        item.classList.remove('selected');
        item.style.background = '';
    });
    const rootItem = document.querySelector('#copy-folder-tree .folder-item');
    if (rootItem) {
        rootItem.classList.add('selected');
        rootItem.style.background = 'var(--primary-light)';
    }
}

async function copyFiles() {
    if (selectedCopyTarget === null || selectedCopyTarget === undefined) {
        showToast('请选择目标文件夹', 'error');
        return;
    }
    
    try {
        const res = await apiCall(`/api/files/copy?paths=${encodeURIComponent(selectedFiles.join(','))}&target=${encodeURIComponent(selectedCopyTarget)}`, {
            method: 'POST'
        });
        if (!res) return;
        
        const data = await res.json();
        
        if (data.count > 0) {
            showToast(`已复制 ${data.count} 个文件`, 'success');
            closeModal('copy-modal');
            clearSelection();
            loadFiles(currentPath);
        } else {
            showToast('复制失败', 'error');
        }
    } catch (e) {
        showToast('复制失败', 'error');
    }
}

function showRenameModal() {
    if (selectedFiles.length !== 1) {
        showToast('请选择一个文件进行重命名', 'error');
        return;
    }
    renameTargetPath = selectedFiles[0];
    const fileName = renameTargetPath.split(/[/\\]/).pop();
    document.getElementById('rename-input').value = fileName;
    showModal('rename-modal');
}

async function doRename() {
    const newName = document.getElementById('rename-input').value.trim();
    if (!newName) {
        showToast('请输入新名称', 'error');
        return;
    }
    
    try {
        const res = await apiCall(`/api/files/rename?path=${encodeURIComponent(renameTargetPath)}&new_name=${encodeURIComponent(newName)}`, {
            method: 'PUT'
        });
        if (!res) return;
        
        const data = await res.json();
        
        if (data.success) {
            showToast('重命名成功', 'success');
            closeModal('rename-modal');
            clearSelection();
            loadFiles(currentPath);
        } else {
            showToast(data.detail || '重命名失败', 'error');
        }
    } catch (e) {
        showToast('重命名失败', 'error');
    }
}

function showToast(msg, type = '') {
    const toast = document.getElementById('toast');
    toast.textContent = msg;
    toast.className = 'toast visible ' + type;
    setTimeout(() => toast.classList.remove('visible'), 3000);
}

function logout() {
    fetch('/api/auth/logout', { method: 'POST', credentials: 'include' }).then(() => {
        window.location.href = '/login';
    });
}

async function loadMounts() {
    try {
        const res = await apiCall('/api/mounts');
        if (!res) return;
        
        const data = await res.json();
        mounts = data.mounts || [];
        renderMounts();
    } catch (e) {
        console.error('加载挂载点失败', e);
    }
}

function renderMounts() {
    const el = document.getElementById('mounts-list');
    if (!el) return;
    
    if (mounts.length === 0) {
        el.innerHTML = '<div style="padding:8px 16px;color:var(--text-dim);font-size:12px;">暂无挂载目录</div>';
        return;
    }
    
    el.innerHTML = mounts.map(m => `
        <div class="nav-item mount-item ${currentMount && currentMount.path === m.path ? 'active' : ''}" 
             data-path="${escapeAttr(m.path)}"
             data-name="${escapeHtml(m.name)}"
             onclick="navigateToMountByEl(this)"
             title="${m.path}${m.readonly ? ' (只读)' : ''}">
            <span class="icon">${m.icon || '📁'}</span> 
            ${escapeHtml(m.name)}
            ${m.readonly ? '<span class="readonly-badge">只读</span>' : ''}
        </div>
    `).join('');
}

function navigateToMountByEl(el) {
    const path = el.dataset.path.replace(/\\\\/g, '\\');
    const name = el.dataset.name;
    navigateToMount(path, name);
    document.querySelectorAll('.nav-item').forEach(item => item.classList.remove('active'));
    el.classList.add('active');
}

async function navigateToMount(path, name) {
    currentMount = { path, name };
    const mount = mounts.find(m => m.path === path);
    isReadonlyPath = mount ? mount.readonly : false;
    
    await loadFiles(path);
    updateToolbarForReadonly();
}

function updateToolbarForReadonly() {
    const uploadBtn = document.querySelector('button[onclick="showUploadModal()"]');
    const newFolderBtn = document.querySelector('button[onclick="showNewFolderModal()"]');
    
    if (uploadBtn) {
        uploadBtn.disabled = isReadonlyPath;
        uploadBtn.style.opacity = isReadonlyPath ? '0.5' : '1';
    }
    if (newFolderBtn) {
        newFolderBtn.disabled = isReadonlyPath;
        newFolderBtn.style.opacity = isReadonlyPath ? '0.5' : '1';
    }
}

document.getElementById('search-btn')?.addEventListener('click', searchFiles);
document.getElementById('search-input')?.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') searchFiles();
});

checkAuth().then(auth => {
    if (auth) {
        loadMounts();
        loadFiles();
    }
});