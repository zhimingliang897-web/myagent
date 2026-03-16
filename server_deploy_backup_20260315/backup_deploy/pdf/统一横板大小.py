from pypdf import PdfReader, PdfWriter

def normalize_landscape_pages(input_pdf, output_pdf):
    reader = PdfReader(input_pdf)
    writer = PdfWriter()

    pages = reader.pages
    num_pages = len(pages)

    # 先找一个“基准横向页面”
    target_width = None
    target_height = None

    for page in pages:
        w = float(page.mediabox.width)
        h = float(page.mediabox.height)
        if w > h:  # 横向
            target_width = w
            target_height = h
            break

    if target_width is None:
        raise ValueError("这个 PDF 里根本没有横向页面，没啥好统一的。")

    print(f"基准横向尺寸: width={target_width}, height={target_height}")

    for idx, page in enumerate(pages):
        w = float(page.mediabox.width)
        h = float(page.mediabox.height)

        # 只处理横向页（宽 > 高）
        if w > h:
            # 把这一页缩放到跟基准横向页一样大
            # 注意：scale_to 会缩放内容 + 页面尺寸
            page.scale_to(target_width, target_height)
            print(f"页面 {idx+1}: 横向 -> 统一为 {target_width}x{target_height}")
        else:
            print(f"页面 {idx+1}: 竖向，保持不动")

        writer.add_page(page)

    with open(output_pdf, "wb") as f:
        writer.write(f)

    print(f"✅ 完成：{input_pdf} -> {output_pdf}")


if __name__ == "__main__":
    # 自己改成你的源文件名 / 目标文件名
    normalize_landscape_pages("output_final.pdf", "output_normalized.pdf")
