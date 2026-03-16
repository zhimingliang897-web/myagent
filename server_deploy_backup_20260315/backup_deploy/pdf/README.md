# PDF 工具集 (12pdffun)

一组实用的 Python PDF 处理脚本，以及一个基于 Flask 的 Web UI，用于日常 PDF 操作。

## 快速启动（Web UI）

```bash
pip install flask pypdf Pillow pymupdf
python app.py
```

浏览器打开 [http://127.0.0.1:5000](http://127.0.0.1:5000) 即可使用。

### Web UI 功能

| 功能 | 说明 |
|------|------|
| 图片转 PDF | 将多张图片合成为一个 PDF，支持强制横向 |
| PDF 合并 | 将多个 PDF 按顺序合并为一个 |
| 删除页面 | 删除指定页码（支持单页、区间、混合格式） |
| 插入 PDF | 在指定位置（最前/最后/页前/页后）插入另一个 PDF |
| 页面重排 | 将竖版页面排到横版页面前面 |
| 统一尺寸 | 将所有横向页面统一为相同尺寸 |
| PDF 转图片 | 将 PDF 指定页范围导出为 JPG/PNG，支持合并长图 |

---

## 命令行脚本依赖安装

```bash
pip install pypdf Pillow
```

## 功能列表

### 1. 图片合成 PDF

| 脚本 | 功能 |
|------|------|
| [fun.py](fun.py) | 将目录下的图片合成为 PDF |
| [横向.py](横向.py) | 将图片合成为 PDF，竖版图片自动旋转为横向 |

**使用示例**：
```python
# fun.py - 基础图片转 PDF
images_to_pdf("./images", "output.pdf")

# 横向.py - 强制横向输出
images_to_pdf("./images", "output_landscape.pdf", force_landscape=True)
```

### 2. PDF 合并

| 脚本 | 功能 |
|------|------|
| [和一.py](和一.py) | 合并多个 PDF 文件 |
| [和一2.py](和一2.py) | 同上（不同文件列表） |
| [和一3.py](和一3.py) | 同上（不同文件列表） |

**使用示例**：
```python
files = ["file1.pdf", "file2.pdf", "file3.pdf"]
merge_pdfs(files, "merged.pdf")
```

### 3. PDF 页面操作

| 脚本 | 功能 |
|------|------|
| [删页.py](删页.py) | 删除指定页码 |
| [插入pdf.py](插入pdf.py) | 在指定位置插入另一个 PDF |
| [竖版提前.py](竖版提前.py) | 重排页面：竖版页在前，横版页在后 |
| [统一横板大小.py](统一横板大小.py) | 统一所有横向页面的尺寸 |

**使用示例**：

```python
# 删页.py - 删除第 3、5 页
remove_pages("input.pdf", "output.pdf", pages_to_remove=[3, 5])

# 插入pdf.py - 在第 5 页后插入另一个 PDF
insert_pdf(
    main_pdf="main.pdf",
    insert_pdf="insert.pdf",
    output_pdf="result.pdf",
    page_index=5,
    after=True  # False 则在该页前插入
)

# 竖版提前.py - 将竖版页面排到横版页面前面
reorder_portrait_before_landscape("input.pdf", "output.pdf")

# 统一横板大小.py - 统一所有横向页面尺寸
normalize_landscape_pages("input.pdf", "output.pdf")
```

### 4. PDF 转图片

| 脚本 | 功能 |
|------|------|
| [to_picture.py](to_picture.py) | 将 PDF 指定页范围导出为图片 |

**使用示例**：
```python
# 导出第 15-29 页为 PNG 图片
pdf_range_to_images(
    pdf_path="input.pdf",
    start_page=15,
    end_page=29,
    out_dir="export_images",
    fmt="png",
    dpi=300
)
```

> **注意**：`to_picture.py` 使用 `pdf2image`，需要安装 [Poppler](https://github.com/oschwartz10612/poppler-windows/releases)，并在代码中配置 `POPPLER_PATH`。  
> Web UI 中的 PDF 转图片功能直接使用 `pymupdf`，**无需 Poppler**。

## 快速使用（命令行脚本）

每个脚本都可以直接修改 `if __name__ == "__main__":` 部分的参数后运行：

```bash
python fun.py
python 和一.py
python 删页.py
# ...
```

## 注意事项

- 页码参数均从 **1** 开始计数（人类习惯）
- 图片转 PDF 支持格式：`.png`, `.jpg`, `.jpeg`, `.bmp`
- 部分脚本输入输出可以是同一文件（会覆盖原文件）
