import os
from PIL import Image

Image.MAX_IMAGE_PIXELS = 200000000

def png_segment(input_folder, output_folder, n=3):
    """
    segment a 1024x1024 png image into n x n tiles, with each tile being (1024//n)x(1024//n)
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        
        if filename.endswith((".png", ".jpg", ".jpeg", ".bmp")):
            with Image.open(file_path) as img:
                img_width, img_height = img.size
                # print(f"image_size: {img_width} x {img_height}")

                tile_width = img_width // n
                tile_height = img_height // n
                
                for i in range(n):
                    for j in range(n):
                        left = j * tile_width
                        upper = i * tile_height
                        right = left + tile_width
                        lower = upper + tile_height
                        
                        # 从图片中裁剪小块
                        tile = img.crop((left, upper, right, lower))
                        
                        # 构造输出文件路径
                        tile_filename = f"{os.path.splitext(filename)[0]}_{i+1}_{j+1}.png"
                        tile_path = os.path.join(output_folder, tile_filename)
                        
                        # 保存小块
                        tile.save(tile_path)
        print(f"Finished segmenting {filename}")


def tif_segment(input_folder, output_folder, n=3):
    """
    Segment a tif image into n x n tiles, with each tile being (image_width//n)x(image_height//n),
    and save each tile as a png file.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        
        if filename.endswith(".tif"):
            with Image.open(file_path) as img:
                img_width, img_height = img.size

                # Calculate tile dimensions
                tile_width = img_width // n
                tile_height = img_height // n
                
                # Iterate over each tile position
                for i in range(n):
                    for j in range(n):
                        left = j * tile_width
                        upper = i * tile_height
                        right = left + tile_width
                        lower = upper + tile_height
                        
                        # Crop the tile from the original image
                        tile = img.crop((left, upper, right, lower))
                        
                        # Construct output file path with PNG format
                        tile_filename = f"{os.path.splitext(filename)[0]}_{i+1}_{j+1}.png"
                        tile_path = os.path.join(output_folder, tile_filename)
                        
                        # Save the tile as a PNG file
                        tile.save(tile_path, format="PNG")
                        
                print(f"Finished segmenting {filename}")

if __name__ == "__main__":
    input_folder = "./data/48RVU"
    output_folder = "./data/48RVU_n_times_n"
    # image_width, image_height = 1024, 1024
    n = 32
    tif_segment(input_folder, output_folder, n)