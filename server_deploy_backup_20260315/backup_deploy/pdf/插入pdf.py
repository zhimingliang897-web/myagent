from pypdf import PdfReader, PdfWriter

def insert_pdf(
    main_pdf: str,
    insert_pdf: str,
    output_pdf: str,
    page_index: int,        # 人类习惯的页码：从 1 开始
    after: bool = False,    # False = 在该页“前面”插入；True = 在该页“后面”插入
):
    """
    main_pdf:      主文档
    insert_pdf:    要插入的那份 PDF
    output_pdf:    输出文件
    page_index:    目标页码（1-based）
    after:         False => 在 page_index 前插
                   True  => 在 page_index 后插
    """
    reader_main = PdfReader(main_pdf)
    reader_ins = PdfReader(insert_pdf)
    writer = PdfWriter()

    total_main = len(reader_main.pages)

    if not 1 <= page_index <= total_main:
        raise ValueError(f"页码超范围：1 ~ {total_main}，你传的是 {page_index}")

    # 0-based 下标
    pos = page_index - 1

    if not after:
        # 在第 page_index 页“前面”插入
        for i in range(total_main):
            if i == pos:
                # 先插入插入文档的所有页
                for p in reader_ins.pages:
                    writer.add_page(p)
            # 再写主文档当前页
            writer.add_page(reader_main.pages[i])
    else:
        # 在第 page_index 页“后面”插入
        for i in range(total_main):
            # 先写主文档当前页
            writer.add_page(reader_main.pages[i])
            # 如果这是目标页，就在“后面”插入
            if i == pos:
                for p in reader_ins.pages:
                    writer.add_page(p)

    with open(output_pdf, "wb") as f:
        writer.write(f)

    print(
        f"✅ 主文档 {total_main} 页，在第 {page_index} 页"
        f"{'之后' if after else '之前'}插入 {len(reader_ins.pages)} 页 -> {output_pdf}"
    )


if __name__ == "__main__":
    # 例子 1：在第 27 页“前面”插入
    insert_pdf(
        main_pdf="output_final_after.pdf",
        insert_pdf="博士.pdf",
        output_pdf="output_final_after.pdf",
        page_index=3,
        after=True,
    )

    # 例子 2：在第 27 页“后面”插入
    # insert_pdf(
    #     main_pdf="output_final.pdf",
    #     insert_pdf="腾飞杯.pdf",
    #     output_pdf="output_final_after_27.pdf",
    #     page_index=27,
    #     after=True,    # 在第 27 页后
    # )
