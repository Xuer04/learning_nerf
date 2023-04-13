import os
import imageio.v2 as imageio


def make_video(imgs_path, num_imgs, result_path, is_gt=False):
    file_prefix = 'view'
    file_postfix = '_gt' if is_gt else '_pred'

    file_pattern = file_prefix + f'{{:03d}}' + file_postfix + '.png'

    file_list = []
    images = []
    for i in range(1, num_imgs+1):
        file_name = file_pattern.format(i)
        file_path = os.path.join(imgs_path, file_name)
        if os.path.exists(file_path):
            file_list.append(file_path)
            image = imageio.imread(file_path)
            images.append(image)

    file_list.sort()

    video_name = 'video' + file_postfix + '.mp4'
    video_path = os.path.join(result_path, video_name)

    imageio.mimwrite(video_path, images, fps=30, quality=10)


if __name__ == '__main__':
    imgs_path = './images'
    result_path = './videos'
    make_video(imgs_path, 174, result_path, True)
