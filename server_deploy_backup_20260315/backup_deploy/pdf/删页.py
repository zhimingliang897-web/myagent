from pypdf import PdfReader, PdfWriter

def remove_pages(input_pdf, output_pdf, pages_to_remove):
    """
    pages_to_remove 用【1 开始计数】的页码，比如 [3, 5] 表示删第 3 页和第 5 页
    """
    reader = PdfReader(input_pdf)
    writer = PdfWriter()

    total_pages = len(reader.pages)
    # 转成 0-based 索引
    remove_idx = {p - 1 for p in pages_to_remove if 1 <= p <= total_pages}

    for i in range(total_pages):
        if i not in remove_idx:
            writer.add_page(reader.pages[i])

    with open(output_pdf, "wb") as f:
        writer.write(f)

    print(f"原来 {total_pages} 页，删除 {len(remove_idx)} 页 -> {output_pdf}")

if __name__ == "__main__":
    # 例如：从 input.pdf 里删掉第 3 页，生成 output.pdf
    remove_pages("output_final.pdf", "output_final.pdf", pages_to_remove=[15,16,17,18,19,20,21,22,23,24,25,26,27,28,29])
