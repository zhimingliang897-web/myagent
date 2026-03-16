from pypdf import PdfReader, PdfWriter

def reorder_portrait_before_landscape(input_pdf, output_pdf):
    reader = PdfReader(input_pdf)
    writer = PdfWriter()

    portrait_pages = []   # 竖版（高 > 宽）
    landscape_pages = []  # 横版（宽 >= 高）

    for idx, page in enumerate(reader.pages):
        w = float(page.mediabox.width)
        h = float(page.mediabox.height)

        # 打个 log，看下每页情况（可注释掉）
        print(f"页 {idx+1}: width={w:.1f}, height={h:.1f}", end="  ")

        if h > w:
            portrait_pages.append(page)
            print("-> 竖版")
        else:
            landscape_pages.append(page)
            print("-> 横版")

    # 先加竖版，再加横版，内部顺序保持原文件顺序
    for p in portrait_pages:
        writer.add_page(p)
    for p in landscape_pages:
        writer.add_page(p)

    with open(output_pdf, "wb") as f:
        writer.write(f)

    print(
        f"✅ 完成：原本 {len(reader.pages)} 页，"
        f"竖版 {len(portrait_pages)} 页在前，横版 {len(landscape_pages)} 页在后 -> {output_pdf}"
    )

if __name__ == "__main__":
    # 改成你的文件名
    reorder_portrait_before_landscape("output_final.pdf", "output_final.pdf")
