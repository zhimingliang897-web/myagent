from flask import Blueprint, request, jsonify, send_file
import os
import tempfile
import shutil
import zipfile
import atexit
from io import BytesIO
from pathlib import Path

from PIL import Image
from pypdf import PdfReader, PdfWriter
from werkzeug.utils import secure_filename

pdf_bp = Blueprint('pdf', __name__)

UPLOAD_FOLDER = tempfile.mkdtemp()
atexit.register(shutil.rmtree, UPLOAD_FOLDER, ignore_errors=True)

try:
    import fitz
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

def safe_temp_path(original_filename, suffix=None):
    safe_name = secure_filename(original_filename)
    if not safe_name:
        safe_name = "upload" + (suffix or ".pdf")
    return os.path.join(UPLOAD_FOLDER, f"{os.urandom(4).hex()}_{safe_name}")

def format_size(size_bytes):
    if size_bytes > 1024 * 1024:
        return f"{size_bytes / 1024 / 1024:.1f}MB"
    return f"{size_bytes / 1024:.1f}KB"

def close_reader(reader):
    try:
        if hasattr(reader, 'stream') and reader.stream:
            reader.stream.close()
    except Exception:
        pass

def get_pdf_info(pdf_path):
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

def parse_page_range(pages_str, total_pages):
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

@pdf_bp.route('/info', methods=['POST'])
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

@pdf_bp.route('/images_to_pdf', methods=['POST'])
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
            'message': f'完成：{len(images)} 张图片合成 PDF',
            'download_url': f'/api/pdf/download/{output_filename}'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'处理失败: {e}'})

@pdf_bp.route('/merge', methods=['POST'])
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
            'message': f'合并完成：{len(files)} 个文件，共 {total_pages} 页',
            'download_url': f'/api/pdf/download/{output_filename}'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'处理失败: {e}'})
    finally:
        for p in temp_paths:
            if os.path.exists(p):
                os.remove(p)

@pdf_bp.route('/remove_pages', methods=['POST'])
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
            'message': f'原 {total_pages} 页，删除 {len(remove_idx)} 页，剩余 {total_pages - len(remove_idx)} 页',
            'download_url': f'/api/pdf/download/{output_filename}'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'处理失败: {e}'})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@pdf_bp.route('/download/<filename>')
def download_file(filename):
    safe_dir = Path(UPLOAD_FOLDER).resolve()
    target = (safe_dir / filename).resolve()
    if not str(target).startswith(str(safe_dir)):
        return "禁止访问", 403
    if not target.exists():
        return "文件不存在或已过期", 404
    return send_file(str(target), as_attachment=True, download_name=filename)