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
    files = ["merged1.pdf", "output1.pdf", "22三等奖学金.pdf", "优秀学生.pdf" ,"output_landscape1.pdf", "output_landscape3.pdf", "output3.pdf"]  # 按这个顺序合并
    merge_pdfs(files, "merged1.pdf")
