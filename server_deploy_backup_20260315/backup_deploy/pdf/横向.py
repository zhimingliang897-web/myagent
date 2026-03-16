from PIL import Image
from pathlib import Path

def images_to_pdf(
    img_dir,
    pdf_path="output_landscape.pdf",
    exts=(".png", ".jpg", ".jpeg", ".bmp"),
    force_landscape=True,   # 控制是否横向
):
    img_dir = Path(img_dir)
    files = sorted(
        [f for f in img_dir.iterdir() if f.suffix.lower() in exts]
    )
    if not files:
        raise ValueError("目录里没有找到图片文件")

    images = []
    for f in files:
        img = Image.open(f)

        # 统一成 RGB，避免 RGBA/P 在保存 PDF 时报错
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        # 如果需要横向：高度大于宽度就旋转 90 度
        if force_landscape and img.height > img.width:
            img = img.rotate(90, expand=True)

        images.append(img)

    first, rest = images[0], images[1:]
    first.save(pdf_path, save_all=True, append_images=rest)
    print(f"✅ 完成：{len(images)} 张图片 -> {pdf_path}")

if __name__ == "__main__":
    # 把 ./images 目录里的图横向合成到一个 PDF
    images_to_pdf("./images", "output_landscape4.pdf", force_landscape=True)
