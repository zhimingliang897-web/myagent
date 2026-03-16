"""
PDF 工具集 Web UI (Flask 版本)
包含：图片转PDF、PDF合并、删页、插入PDF、页面重排、统一尺寸、PDF转图片
"""

import atexit
import os
import shutil
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path

from flask import Flask, render_template_string, request, send_file, jsonify
from PIL import Image
from pypdf import PdfReader, PdfWriter
from werkzeug.utils import secure_filename

# PyMuPDF（用于 PDF 转图片）
try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max

UPLOAD_FOLDER = tempfile.mkdtemp()

# 程序退出时自动清理临时目录
atexit.register(shutil.rmtree, UPLOAD_FOLDER, ignore_errors=True)


HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PDF 工具集</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f5f5f5; min-height: 100vh; }
        .container { max-width: 900px; margin: 0 auto; padding: 20px; }
        h1 { text-align: center; color: #333; margin-bottom: 10px; }
        .subtitle { text-align: center; color: #666; margin-bottom: 30px; }
        .tabs { display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 20px; }
        .tab { padding: 10px 16px; background: #fff; border: 1px solid #ddd; border-radius: 8px; cursor: pointer; transition: all 0.2s; font-size: 14px; }
        .tab:hover { background: #e3f2fd; }
        .tab.active { background: #1976d2; color: #fff; border-color: #1976d2; }
        .panel { display: none; background: #fff; border-radius: 12px; padding: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .panel.active { display: block; }
        .panel h3 { color: #333; margin-bottom: 16px; }
        .form-group { margin-bottom: 16px; }
        .form-group label { display: block; margin-bottom: 6px; color: #555; font-weight: 500; }
        input[type="file"] { width: 100%; padding: 12px; border: 2px dashed #ddd; border-radius: 8px; background: #fafafa; cursor: pointer; }
        input[type="file"]:hover { border-color: #1976d2; }
        input[type="number"], input[type="text"] { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 6px; font-size: 14px; }
        select { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 6px; font-size: 14px; background: #fff; }
        .checkbox-group { display: flex; align-items: center; gap: 8px; }
        .checkbox-group input { width: 18px; height: 18px; }
        .btn { padding: 12px 24px; background: #1976d2; color: #fff; border: none; border-radius: 8px; cursor: pointer; font-size: 16px; transition: background 0.2s; }
        .btn:hover { background: #1565c0; }
        .btn:disabled { background: #ccc; cursor: not-allowed; }
        .result { margin-top: 16px; padding: 12px; border-radius: 8px; display: none; }
        .result.success { display: block; background: #e8f5e9; color: #2e7d32; }
        .result.error { display: block; background: #ffebee; color: #c62828; }
        .download-link { display: inline-block; margin-top: 10px; padding: 10px 20px; background: #4caf50; color: #fff; text-decoration: none; border-radius: 6px; }
        .download-link:hover { background: #43a047; }
        .info-box { background: #e3f2fd; padding: 12px; border-radius: 6px; margin-bottom: 16px; color: #1565c0; }
        .row { display: flex; gap: 16px; }
        .row > * { flex: 1; }
        .loading { display: none; }
        .loading.show { display: inline-block; margin-left: 10px; color: #888; }
        @media (max-width: 600px) {
            .row { flex-direction: column; }
            .tabs { justify-content: center; }
            .tab { font-size: 12px; padding: 8px 10px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>PDF 工具集</h1>
        <p class="subtitle">一站式 PDF 处理工具</p>

        <div class="tabs">
            <div class="tab active" data-tab="tab1">图片转PDF</div>
            <div class="tab" data-tab="tab2">PDF合并</div>
            <div class="tab" data-tab="tab3">删除页面</div>
            <div class="tab" data-tab="tab4">插入PDF</div>
            <div class="tab" data-tab="tab5">页面重排</div>
            <div class="tab" data-tab="tab6">统一尺寸</div>
            <div class="tab" data-tab="tab7">PDF转图片</div>
        </div>

        <!-- Tab 1: 图片转PDF -->
        <div class="panel active" id="tab1">
            <h3>图片转 PDF</h3>
            <p style="color:#666;margin-bottom:16px;">将多张图片合成为一个 PDF 文件</p>
            <form id="form1" enctype="multipart/form-data">
                <div class="form-group">
                    <label>上传图片（可多选）</label>
                    <input type="file" name="images" multiple accept="image/*" required>
                </div>
                <div class="form-group checkbox-group">
                    <input type="checkbox" name="force_landscape" id="landscape1">
                    <label for="landscape1">强制横向（竖图自动旋转）</label>
                </div>
                <button type="submit" class="btn">生成 PDF</button>
                <span class="loading">处理中...</span>
            </form>
            <div class="result" id="result1"></div>
        </div>

        <!-- Tab 2: PDF合并 -->
        <div class="panel" id="tab2">
            <h3>PDF 合并</h3>
            <p style="color:#666;margin-bottom:16px;">将多个 PDF 文件按顺序合并为一个</p>
            <form id="form2" enctype="multipart/form-data">
                <div class="form-group">
                    <label>上传 PDF 文件（可多选，按选择顺序合并）</label>
                    <input type="file" name="pdfs" multiple accept=".pdf" required>
                </div>
                <button type="submit" class="btn">合并 PDF</button>
                <span class="loading">处理中...</span>
            </form>
            <div class="result" id="result2"></div>
        </div>

        <!-- Tab 3: 删除页面 -->
        <div class="panel" id="tab3">
            <h3>删除页面</h3>
            <p style="color:#666;margin-bottom:16px;">删除 PDF 中指定的页面</p>
            <form id="form3" enctype="multipart/form-data">
                <div class="form-group">
                    <label>上传 PDF 文件</label>
                    <input type="file" name="pdf" accept=".pdf" required onchange="getPdfInfo(this, 'info3')">
                </div>
                <div class="info-box" id="info3" style="display:none;"></div>
                <div class="form-group">
                    <label>要删除的页码</label>
                    <input type="text" name="pages" placeholder="支持格式：1,3,5 或 1-5 或 1,3-5,8" required>
                    <small style="color:#888;display:block;margin-top:4px;">支持单页(1,3,5)、区间(1-5)、混合(1,3-5,8)</small>
                </div>
                <button type="submit" class="btn">删除页面</button>
                <span class="loading">处理中...</span>
            </form>
            <div class="result" id="result3"></div>
        </div>

        <!-- Tab 4: 插入PDF -->
        <div class="panel" id="tab4">
            <h3>插入 PDF</h3>
            <p style="color:#666;margin-bottom:16px;">在指定位置插入另一个 PDF</p>
            <form id="form4" enctype="multipart/form-data">
                <div class="form-group">
                    <label>主文档</label>
                    <input type="file" name="main_pdf" accept=".pdf" required onchange="getPdfInfo(this, 'info4')">
                </div>
                <div class="info-box" id="info4" style="display:none;"></div>
                <div class="form-group">
                    <label>要插入的文档</label>
                    <input type="file" name="insert_pdf" accept=".pdf" required>
                </div>
                <div class="form-group">
                    <label>插入位置</label>
                    <select name="position">
                        <option value="start">在文档最前面（第0页之后）</option>
                        <option value="after" selected>在指定页码之后</option>
                        <option value="before">在指定页码之前</option>
                        <option value="end">在文档最后面</option>
                    </select>
                </div>
                <div class="form-group" id="pageIndexGroup4">
                    <label>目标页码</label>
                    <input type="number" name="page_index" value="1" min="1">
                    <small style="color:#888;display:block;margin-top:4px;">选择"最前面"或"最后面"时此项无效</small>
                </div>
                <button type="submit" class="btn">插入</button>
                <span class="loading">处理中...</span>
            </form>
            <div class="result" id="result4"></div>
        </div>

        <!-- Tab 5: 页面重排 -->
        <div class="panel" id="tab5">
            <h3>页面重排</h3>
            <p style="color:#666;margin-bottom:16px;">将竖版页面排到横版页面前面</p>
            <form id="form5" enctype="multipart/form-data">
                <div class="form-group">
                    <label>上传 PDF 文件</label>
                    <input type="file" name="pdf" accept=".pdf" required onchange="getPdfInfo(this, 'info5')">
                </div>
                <div class="info-box" id="info5" style="display:none;"></div>
                <button type="submit" class="btn">重排页面</button>
                <span class="loading">处理中...</span>
            </form>
            <div class="result" id="result5"></div>
        </div>

        <!-- Tab 6: 统一尺寸 -->
        <div class="panel" id="tab6">
            <h3>统一横向尺寸</h3>
            <p style="color:#666;margin-bottom:16px;">将所有横向页面统一为相同尺寸（以第一个横向页为基准）</p>
            <form id="form6" enctype="multipart/form-data">
                <div class="form-group">
                    <label>上传 PDF 文件</label>
                    <input type="file" name="pdf" accept=".pdf" required onchange="getPdfInfo(this, 'info6')">
                </div>
                <div class="info-box" id="info6" style="display:none;"></div>
                <button type="submit" class="btn">统一尺寸</button>
                <span class="loading">处理中...</span>
            </form>
            <div class="result" id="result6"></div>
        </div>

        <!-- Tab 7: PDF转图片 -->
        <div class="panel" id="tab7">
            <h3>PDF 转图片</h3>
            <p style="color:#666;margin-bottom:16px;">将 PDF 页面导出为图片</p>
            <form id="form7" enctype="multipart/form-data">
                <div class="form-group">
                    <label>上传 PDF 文件</label>
                    <input type="file" name="pdf" accept=".pdf" required onchange="getPdfInfo(this, 'info7'); autoSetEndPage(this)">
                </div>
                <div class="info-box" id="info7" style="display:none;"></div>
                <div class="row">
                    <div class="form-group">
                        <label>起始页</label>
                        <input type="number" name="start_page" id="start_page7" value="1" min="1" required>
                    </div>
                    <div class="form-group">
                        <label>结束页</label>
                        <input type="number" name="end_page" id="end_page7" value="1" min="1" required>
                    </div>
                    <div class="form-group">
                        <label>DPI</label>
                        <input type="number" name="dpi" value="150" min="72" max="600">
                    </div>
                </div>
                <div class="row">
                    <div class="form-group">
                        <label>输出格式</label>
                        <select name="format">
                            <option value="jpg" selected>JPG（较小体积）</option>
                            <option value="png">PNG（无损质量）</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>目标大小限制 (KB)</label>
                        <input type="number" name="max_size" value="0" min="0" max="10000">
                        <small style="color:#888;display:block;margin-top:4px;">0 表示不限制，仅对 JPG 有效</small>
                    </div>
                </div>
                <div class="form-group checkbox-group">
                    <input type="checkbox" name="long_image" id="long_image7">
                    <label for="long_image7">合并为长图（将所有页面垂直拼接成一张图）</label>
                </div>
                <button type="submit" class="btn">导出图片</button>
                <span class="loading">处理中...</span>
            </form>
            <div class="result" id="result7"></div>
        </div>
    </div>

    <script>
        // Tab 切换
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', () => {
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
                tab.classList.add('active');
                document.getElementById(tab.dataset.tab).classList.add('active');
            });
        });

        // 获取PDF信息
        async function getPdfInfo(input, infoId) {
            const file = input.files[0];
            if (!file) return;
            const formData = new FormData();
            formData.append('pdf', file);
            try {
                const res = await fetch('/api/pdf_info', { method: 'POST', body: formData });
                const data = await res.json();
                const infoBox = document.getElementById(infoId);
                infoBox.style.display = 'block';
                infoBox.textContent = data.info;
            } catch (e) {
                console.error(e);
            }
        }

        // Tab7 专用：上传 PDF 后自动把"结束页"填为总页数
        async function autoSetEndPage(input) {
            const file = input.files[0];
            if (!file) return;
            const formData = new FormData();
            formData.append('pdf', file);
            try {
                const res = await fetch('/api/pdf_info', { method: 'POST', body: formData });
                const data = await res.json();
                // 从 "总页数：N | ..." 中提取 N
                const match = data.info && data.info.match(/总页数：(\d+)/);
                if (match) {
                    document.getElementById('end_page7').value = match[1];
                }
            } catch (e) {
                console.error(e);
            }
        }

        // 表单提交
        const endpoints = {
            form1: '/api/images_to_pdf',
            form2: '/api/merge_pdfs',
            form3: '/api/remove_pages',
            form4: '/api/insert_pdf',
            form5: '/api/reorder_pages',
            form6: '/api/normalize_landscape',
            form7: '/api/pdf_to_images'
        };

        Object.keys(endpoints).forEach(formId => {
            document.getElementById(formId).addEventListener('submit', async (e) => {
                e.preventDefault();
                const form = e.target;
                const resultDiv = document.getElementById('result' + formId.slice(-1));
                const loading = form.querySelector('.loading');
                const btn = form.querySelector('.btn');

                loading.classList.add('show');
                btn.disabled = true;
                resultDiv.className = 'result';
                resultDiv.innerHTML = '';

                const formData = new FormData(form);

                try {
                    const res = await fetch(endpoints[formId], { method: 'POST', body: formData });
                    const data = await res.json();

                    if (data.success) {
                        resultDiv.className = 'result success';
                        resultDiv.innerHTML = data.message + '<br><a class="download-link" href="' + data.download_url + '" download>下载文件</a>';
                    } else {
                        resultDiv.className = 'result error';
                        resultDiv.textContent = data.message;
                    }
                } catch (err) {
                    resultDiv.className = 'result error';
                    resultDiv.textContent = '请求失败：' + err.message;
                }

                loading.classList.remove('show');
                btn.disabled = false;
            });
        });
    </script>
</body>
</html>
'''


# ============ 工具函数 ============

def safe_temp_path(original_filename, suffix=None):
    """生成安全的临时文件路径，兼容中文文件名"""
    safe_name = secure_filename(original_filename)
    if not safe_name:
        safe_name = "upload" + (suffix or ".pdf")
    return os.path.join(UPLOAD_FOLDER, f"{os.urandom(4).hex()}_{safe_name}")


def format_size(size_bytes):
    """格式化文件大小为可读字符串"""
    if size_bytes > 1024 * 1024:
        return f"{size_bytes / 1024 / 1024:.1f}MB"
    return f"{size_bytes / 1024:.1f}KB"


def close_reader(reader):
    """安全关闭 PdfReader 的文件句柄（Windows 兼容）"""
    try:
        if hasattr(reader, 'stream') and reader.stream:
            reader.stream.close()
    except Exception:
        pass


def get_pdf_info(pdf_path):
    """获取 PDF 信息"""
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    portrait_count = 0
    landscape_count = 0
    for page in reader.pages:
        w = float(page.mediabox.width)
        h = float(page.mediabox.height)
        if h > w:
            portrait_count += 1
        else:
            landscape_count += 1
    close_reader(reader)
    return f"总页数：{total_pages} | 竖版：{portrait_count} | 横版：{landscape_count}"


# ============ 路由 ============

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/pdf_info', methods=['POST'])
def api_pdf_info():
    if 'pdf' not in request.files:
        return jsonify({'info': '未上传文件'})
    pdf_file = request.files['pdf']
    temp_path = safe_temp_path(pdf_file.filename)
    try:
        pdf_file.save(temp_path)
        info = get_pdf_info(temp_path)
    except Exception as e:
        info = f'读取失败: {e}'
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return jsonify({'info': info})


@app.route('/api/images_to_pdf', methods=['POST'])
def api_images_to_pdf():
    if 'images' not in request.files:
        return jsonify({'success': False, 'message': '请上传图片'})

    files = request.files.getlist('images')
    if not files or files[0].filename == '':
        return jsonify({'success': False, 'message': '请上传至少一张图片'})

    force_landscape = 'force_landscape' in request.form

    try:
        images = []
        for f in files:
            img = Image.open(f)
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            if force_landscape and img.height > img.width:
                img = img.rotate(90, expand=True)
            images.append(img)

        output_filename = f"images_to_pdf_{os.urandom(4).hex()}.pdf"
        output_path = os.path.join(UPLOAD_FOLDER, output_filename)
        first, rest = images[0], images[1:]
        first.save(output_path, save_all=True, append_images=rest)

        return jsonify({
            'success': True,
            'message': f'✅ 完成：{len(images)} 张图片合成 PDF',
            'download_url': f'/download/{output_filename}'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'处理失败: {e}'})


@app.route('/api/merge_pdfs', methods=['POST'])
def api_merge_pdfs():
    if 'pdfs' not in request.files:
        return jsonify({'success': False, 'message': '请上传 PDF 文件'})

    files = request.files.getlist('pdfs')
    if len(files) < 2:
        return jsonify({'success': False, 'message': '请上传至少 2 个 PDF 文件'})

    temp_paths = []
    try:
        writer = PdfWriter()
        total_pages = 0

        for pdf_file in files:
            temp_path = safe_temp_path(pdf_file.filename)
            pdf_file.save(temp_path)
            temp_paths.append(temp_path)
            reader = PdfReader(temp_path)
            for page in reader.pages:
                writer.add_page(page)
            total_pages += len(reader.pages)
            close_reader(reader)

        output_filename = f"merged_{os.urandom(4).hex()}.pdf"
        output_path = os.path.join(UPLOAD_FOLDER, output_filename)
        with open(output_path, "wb") as f:
            writer.write(f)

        return jsonify({
            'success': True,
            'message': f'✅ 合并完成：{len(files)} 个文件，共 {total_pages} 页',
            'download_url': f'/download/{output_filename}'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'处理失败: {e}'})
    finally:
        for p in temp_paths:
            if os.path.exists(p):
                os.remove(p)


def parse_page_range(pages_str, total_pages):
    """解析页码字符串，支持 1,3,5 或 1-5 或 1,3-5,8 格式"""
    pages = set()
    parts = pages_str.replace(' ', '').split(',')
    for part in parts:
        if '-' in part:
            try:
                start, end = part.split('-')
                start = int(start)
                end = int(end)
                if start > end:
                    start, end = end, start
                for p in range(start, end + 1):
                    if 1 <= p <= total_pages:
                        pages.add(p)
            except ValueError:
                continue
        else:
            try:
                p = int(part)
                if 1 <= p <= total_pages:
                    pages.add(p)
            except ValueError:
                continue
    return pages


@app.route('/api/remove_pages', methods=['POST'])
def api_remove_pages():
    if 'pdf' not in request.files:
        return jsonify({'success': False, 'message': '请上传 PDF 文件'})

    pages_str = request.form.get('pages', '')
    if not pages_str.strip():
        return jsonify({'success': False, 'message': '请输入要删除的页码'})

    pdf_file = request.files['pdf']
    temp_path = safe_temp_path(pdf_file.filename)

    try:
        pdf_file.save(temp_path)
        reader = PdfReader(temp_path)
        writer = PdfWriter()
        total_pages = len(reader.pages)

        pages_to_remove = parse_page_range(pages_str, total_pages)
        if not pages_to_remove:
            return jsonify({'success': False, 'message': '没有有效的页码'})

        remove_idx = {p - 1 for p in pages_to_remove}
        for i in range(total_pages):
            if i not in remove_idx:
                writer.add_page(reader.pages[i])
        close_reader(reader)

        output_filename = f"removed_{os.urandom(4).hex()}.pdf"
        output_path = os.path.join(UPLOAD_FOLDER, output_filename)
        with open(output_path, "wb") as f:
            writer.write(f)

        return jsonify({
            'success': True,
            'message': f'✅ 原 {total_pages} 页，删除 {len(remove_idx)} 页，剩余 {total_pages - len(remove_idx)} 页',
            'download_url': f'/download/{output_filename}'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'处理失败: {e}'})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route('/api/insert_pdf', methods=['POST'])
def api_insert_pdf():
    if 'main_pdf' not in request.files or 'insert_pdf' not in request.files:
        return jsonify({'success': False, 'message': '请上传主文档和要插入的文档'})

    position = request.form.get('position', 'after')

    try:
        page_index = int(request.form.get('page_index', 1))
    except ValueError:
        page_index = 1

    main_file = request.files['main_pdf']
    insert_file = request.files['insert_pdf']

    main_path = safe_temp_path(main_file.filename)
    insert_path = safe_temp_path(insert_file.filename)

    try:
        main_file.save(main_path)
        insert_file.save(insert_path)

        reader_main = PdfReader(main_path)
        reader_ins = PdfReader(insert_path)
        writer = PdfWriter()

        total_main = len(reader_main.pages)
        insert_count = len(reader_ins.pages)

        if position == 'start':
            for p in reader_ins.pages:
                writer.add_page(p)
            for p in reader_main.pages:
                writer.add_page(p)
            position_text = "文档最前面"

        elif position == 'end':
            for p in reader_main.pages:
                writer.add_page(p)
            for p in reader_ins.pages:
                writer.add_page(p)
            position_text = "文档最后面"

        elif position == 'before':
            if not 1 <= page_index <= total_main:
                return jsonify({'success': False, 'message': f'页码超范围：1 ~ {total_main}'})
            pos = page_index - 1
            for i in range(total_main):
                if i == pos:
                    for p in reader_ins.pages:
                        writer.add_page(p)
                writer.add_page(reader_main.pages[i])
            position_text = f"第 {page_index} 页之前"

        else:  # after
            if not 1 <= page_index <= total_main:
                return jsonify({'success': False, 'message': f'页码超范围：1 ~ {total_main}'})
            pos = page_index - 1
            for i in range(total_main):
                writer.add_page(reader_main.pages[i])
                if i == pos:
                    for p in reader_ins.pages:
                        writer.add_page(p)
            position_text = f"第 {page_index} 页之后"

        close_reader(reader_main)
        close_reader(reader_ins)

        output_filename = f"inserted_{os.urandom(4).hex()}.pdf"
        output_path = os.path.join(UPLOAD_FOLDER, output_filename)
        with open(output_path, "wb") as f:
            writer.write(f)

        return jsonify({
            'success': True,
            'message': f'✅ 在{position_text}插入 {insert_count} 页',
            'download_url': f'/download/{output_filename}'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'处理失败: {e}'})
    finally:
        for p in [main_path, insert_path]:
            if os.path.exists(p):
                os.remove(p)


@app.route('/api/reorder_pages', methods=['POST'])
def api_reorder_pages():
    if 'pdf' not in request.files:
        return jsonify({'success': False, 'message': '请上传 PDF 文件'})

    pdf_file = request.files['pdf']
    temp_path = safe_temp_path(pdf_file.filename)

    try:
        pdf_file.save(temp_path)
        reader = PdfReader(temp_path)
        writer = PdfWriter()

        portrait_pages = []
        landscape_pages = []

        for page in reader.pages:
            w = float(page.mediabox.width)
            h = float(page.mediabox.height)
            if h > w:
                portrait_pages.append(page)
            else:
                landscape_pages.append(page)

        close_reader(reader)

        for p in portrait_pages:
            writer.add_page(p)
        for p in landscape_pages:
            writer.add_page(p)

        output_filename = f"reordered_{os.urandom(4).hex()}.pdf"
        output_path = os.path.join(UPLOAD_FOLDER, output_filename)
        with open(output_path, "wb") as f:
            writer.write(f)

        return jsonify({
            'success': True,
            'message': f'✅ 竖版 {len(portrait_pages)} 页在前，横版 {len(landscape_pages)} 页在后',
            'download_url': f'/download/{output_filename}'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'处理失败: {e}'})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route('/api/normalize_landscape', methods=['POST'])
def api_normalize_landscape():
    if 'pdf' not in request.files:
        return jsonify({'success': False, 'message': '请上传 PDF 文件'})

    pdf_file = request.files['pdf']
    temp_path = safe_temp_path(pdf_file.filename)

    try:
        pdf_file.save(temp_path)
        reader = PdfReader(temp_path)
        writer = PdfWriter()

        target_width = None
        target_height = None

        for page in reader.pages:
            w = float(page.mediabox.width)
            h = float(page.mediabox.height)
            if w > h:
                target_width = w
                target_height = h
                break

        if target_width is None:
            return jsonify({'success': False, 'message': 'PDF 中没有横向页面'})

        landscape_count = 0
        for page in reader.pages:
            w = float(page.mediabox.width)
            h = float(page.mediabox.height)
            if w > h:
                page.scale_to(target_width, target_height)
                landscape_count += 1
            writer.add_page(page)

        close_reader(reader)

        output_filename = f"normalized_{os.urandom(4).hex()}.pdf"
        output_path = os.path.join(UPLOAD_FOLDER, output_filename)
        with open(output_path, "wb") as f:
            writer.write(f)

        return jsonify({
            'success': True,
            'message': f'✅ 统一 {landscape_count} 个横向页面尺寸为 {target_width:.0f}x{target_height:.0f}',
            'download_url': f'/download/{output_filename}'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'处理失败: {e}'})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def compress_image_to_size(img, max_size_kb, output_path):
    """压缩图片到指定大小以下（仅支持 JPG）"""
    max_size_bytes = max_size_kb * 1024

    # 先尝试不同的质量等级
    for quality in range(95, 10, -5):
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        if buffer.tell() <= max_size_bytes:
            img.save(output_path, format='JPEG', quality=quality)
            return os.path.getsize(output_path)

    # 如果还是太大，缩小尺寸
    scale = 0.9
    while scale > 0.1:
        new_size = (int(img.width * scale), int(img.height * scale))
        resized = img.resize(new_size, Image.LANCZOS)

        for quality in range(85, 10, -10):
            buffer = BytesIO()
            resized.save(buffer, format='JPEG', quality=quality)
            if buffer.tell() <= max_size_bytes:
                resized.save(output_path, format='JPEG', quality=quality)
                return os.path.getsize(output_path)

        scale -= 0.1

    # 最后保底
    resized = img.resize((int(img.width * 0.3), int(img.height * 0.3)), Image.LANCZOS)
    resized.save(output_path, format='JPEG', quality=20)
    return os.path.getsize(output_path)


@app.route('/api/pdf_to_images', methods=['POST'])
def api_pdf_to_images():
    if 'pdf' not in request.files:
        return jsonify({'success': False, 'message': '请上传 PDF 文件'})

    if not HAS_FITZ:
        return jsonify({'success': False, 'message': '请安装 PyMuPDF: pip install pymupdf'})

    try:
        start_page = int(request.form.get('start_page', 1))
        end_page = int(request.form.get('end_page', 1))
        dpi = int(request.form.get('dpi', 150))
        max_size = int(request.form.get('max_size', 0))  # KB
    except ValueError:
        return jsonify({'success': False, 'message': '参数必须是数字'})

    output_format = request.form.get('format', 'jpg').lower()
    if output_format not in ['jpg', 'png']:
        output_format = 'jpg'

    long_image = 'long_image' in request.form

    pdf_file = request.files['pdf']
    temp_path = safe_temp_path(pdf_file.filename)
    temp_dir = None

    try:
        pdf_file.save(temp_path)
        doc = fitz.open(temp_path)
        total_pages = len(doc)

        if start_page < 1:
            start_page = 1
        if end_page > total_pages:
            end_page = total_pages
        if start_page > end_page:
            doc.close()
            return jsonify({'success': False, 'message': f'页码范围错误，PDF 共 {total_pages} 页'})

        temp_dir = tempfile.mkdtemp()
        images = []
        zoom = dpi / 72

        for page_num in range(start_page - 1, end_page):
            page = doc[page_num]
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)

        doc.close()

        # 合并为长图
        if long_image and len(images) > 0:
            max_width = max(img.width for img in images)
            total_height = sum(img.height for img in images)

            long_img = Image.new('RGB', (max_width, total_height), (255, 255, 255))
            y_offset = 0
            for img in images:
                x_offset = (max_width - img.width) // 2
                long_img.paste(img, (x_offset, y_offset))
                y_offset += img.height

            ext = 'jpg' if output_format == 'jpg' else 'png'
            output_filename = f"long_image_{os.urandom(4).hex()}.{ext}"
            output_path = os.path.join(UPLOAD_FOLDER, output_filename)

            if output_format == 'png':
                long_img.save(output_path, format='PNG')
            else:
                if max_size > 0:
                    compress_image_to_size(long_img, max_size, output_path)
                else:
                    long_img.save(output_path, format='JPEG', quality=85)

            file_size = os.path.getsize(output_path)
            return jsonify({
                'success': True,
                'message': f'✅ 第 {start_page}-{end_page} 页合并为长图（{max_width}x{total_height}，{format_size(file_size)}）',
                'download_url': f'/download/{output_filename}'
            })

        # 正常导出多张图片
        image_paths = []
        total_size = 0

        for i, img in enumerate(images):
            page_num = start_page + i
            if output_format == 'png':
                out_path = os.path.join(temp_dir, f"page_{page_num}.png")
                img.save(out_path, format='PNG')
            else:
                out_path = os.path.join(temp_dir, f"page_{page_num}.jpg")
                if max_size > 0:
                    compress_image_to_size(img, max_size, out_path)
                else:
                    img.save(out_path, format='JPEG', quality=85)
            total_size += os.path.getsize(out_path)
            image_paths.append(out_path)

        zip_filename = f"pdf_images_{os.urandom(4).hex()}.zip"
        zip_path = os.path.join(UPLOAD_FOLDER, zip_filename)
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for img_path in image_paths:
                zf.write(img_path, os.path.basename(img_path))

        format_str = "JPG" if output_format == 'jpg' else "PNG"
        size_limit_str = f"，每张≤{max_size}KB" if max_size > 0 and output_format == 'jpg' else ""

        return jsonify({
            'success': True,
            'message': f'✅ 导出第 {start_page}-{end_page} 页，共 {len(image_paths)} 张 {format_str} 图片（总计 {format_size(total_size)}{size_limit_str}）',
            'download_url': f'/download/{zip_filename}'
        })

    except Exception as e:
        return jsonify({'success': False, 'message': f'处理失败: {e}'})
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route('/download/<filename>')
def download_file(filename):
    """安全下载：防止路径穿越攻击"""
    safe_dir = Path(UPLOAD_FOLDER).resolve()
    target = (safe_dir / filename).resolve()
    if not str(target).startswith(str(safe_dir)):
        return "禁止访问", 403
    if not target.exists():
        return "文件不存在或已过期", 404
    return send_file(str(target), as_attachment=True, download_name=filename)


if __name__ == '__main__':
    print("=" * 50)
    print("PDF 工具集已启动")
    print("请在浏览器中打开: http://127.0.0.1:5000")
    print("=" * 50)
    app.run(host='0.0.0.0', debug=False, port=int(os.environ.get("PORT", 5002)))
