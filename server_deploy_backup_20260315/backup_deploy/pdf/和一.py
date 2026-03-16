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
    files = ["简历中文.pdf", "output_landscape.pdf", "教育部学籍在线验证报告_梁致铭.pdf", "offer.pdf"]  # 按这个顺序合并
    merge_pdfs(files, "merged.pdf")
