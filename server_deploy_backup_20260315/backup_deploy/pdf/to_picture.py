from pdf2image import convert_from_path
import os

# === 你自己的 poppler 路径（刚才你发的那个） ===
POPPLER_PATH = r"C:\Users\L\Downloads\Release-25.07.0-0\poppler-25.07.0\Library\bin"


def pdf_range_to_images(pdf_path, start_page, end_page,
                        out_dir="out_images", fmt="png", dpi=300):
    """
    把 PDF 的连续页导出为图片：
    [start_page, end_page]（包含）
    """
    if start_page > end_page:
        raise ValueError("start_page 不能大于 end_page")

    # 让输出目录固定在【当前脚本所在目录】下面
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    images = convert_from_path(
        pdf_path,
        dpi=dpi,
        fmt=fmt,
        first_page=start_page,
        last_page=end_page,
        poppler_path=POPPLER_PATH,
    )

    for offset, img in enumerate(images):
        page_num = start_page + offset  # 真实页码
        out_path = os.path.join(out_dir, f"page_{page_num}.{fmt}")
        img.save(out_path, fmt.upper())
        print(f"✅ 已保存: {out_path}")

    print(f"总共导出 {len(images)} 页（第 {start_page} 页 到 第 {end_page} 页）。")
    print(f"图片都在这个文件夹里：{out_dir}")


if __name__ == "__main__":
    # ===== 只改这里这几个参数就行 =====
    pdf_path   = "output_final.pdf"      # 你的 PDF 文件名（放在和这个脚本同一目录）
    start_page = 15               # 起始页（包含）
    end_page   = 29               # 结束页（包含）
    out_dir    = "export_15_28"   # 输出文件夹名（会自动建在当前目录下）

    pdf_range_to_images(pdf_path, start_page, end_page, out_dir, fmt="png", dpi=300)
