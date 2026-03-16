from pypdf import PdfReader, PdfWriter

def merge_pdfs(pdf_list, output_path="merged.pdf"):
    writer = PdfWriter()

    for pdf in pdf_list:
        reader = PdfReader(pdf)
        for page in reader.pages:
            writer.add_page(page)

    with open(output_path, "wb") as f:
        writer.write(f)

    print(f"✅ 合并完成 -> {output_path}")
if __name__ == "__main__":
    files = ["merged.pdf", "本科生中英文成绩单-梁致铭-学号（2191312144）-2025-03-06.pdf", "雅思单.pdf"]  # 按这个顺序合并
    merge_pdfs(files, "merged1.pdf")
