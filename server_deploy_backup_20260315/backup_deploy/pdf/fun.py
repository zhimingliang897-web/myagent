from PIL import Image
from pathlib import Path

def images_to_pdf(
    img_dir,
    pdf_path="output.pdf",
    exts=(".png", ".jpg", ".jpeg", ".bmp"),
):
    img_dir = Path(img_dir)

    # 收集所有图片文件（按文件名排序）
    files = [f for f in img_dir.iterdir() if f.suffix.lower() in exts]
    files = sorted(files)

    if not files:
        raise ValueError("目录里没有找到图片文件")

    images = []
    for f in files:
        img = Image.open(f)
        # PDF 不支持 RGBA / P 模式，统一转成 RGB
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        images.append(img)

    # 第一张作为首页，后面的用 append_images 追加为多页
    first, rest = images[0], images[1:]
    first.save(pdf_path, save_all=True, append_images=rest)

    print(f"✅ 完成：共 {len(images)} 张图片 -> {pdf_path}")

if __name__ == "__main__":
    # 举例：把 ./images 目录下所有图片合成 output.pdf
    images_to_pdf("./images", "output3.pdf")
