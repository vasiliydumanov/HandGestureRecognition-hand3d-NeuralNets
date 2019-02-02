
import os
from PIL import Image
import skvideo
import tensorflow as tf
import numpy as np
from orig_model.nets.ColorHandPose3DNetwork import ColorHandPose3DNetwork
from orig_model.utils.general import detect_keypoints
from tqdm import tqdm
from shutil import rmtree


def video_to_frames(video_path, frame_dir, every_n_frames=5):
    from itertools import islice
    if not skvideo._HAS_FFMPEG:
        skvideo.setFFmpegPath("/home/vasiliy/anaconda3/bin")
    from skvideo import io

    reader = skvideo.io.FFmpegReader(video_path)
    frames_count = reader.getShape()[0] // every_n_frames
    videogen = reader.nextFrame()
    videogen = islice(videogen, 0, None, every_n_frames)
    with tqdm(total=frames_count) as pbar:
        for idx, frame in enumerate(videogen):
            video_name = os.path.basename(video_path).rsplit('.', 1)[0]
            frame_file = os.path.join(frame_dir, f"{video_name}-{idx}.jpg")
            frame_img = Image.fromarray(frame)
            frame_img.save(frame_file, "jpeg")
            pbar.update(1)


def convert_videos_to_frames():
    current_dir = os.path.dirname(__file__)
    source_dir = os.path.join(current_dir, "source")
    root_frames_dir = os.path.join(current_dir, "frames")
    if not os.path.exists(root_frames_dir):
        os.mkdir(root_frames_dir)
    video_dirs = os.listdir(source_dir)
    for video_dir in video_dirs:
        print(f"Saving video frames for '{video_dir}'")
        video_dir_path = os.path.join(source_dir, video_dir)
        videos = os.listdir(video_dir_path)
        frames_dir = os.path.join(root_frames_dir, video_dir)
        if os.path.exists(frames_dir):
            rmtree(frames_dir)
        os.mkdir(frames_dir)
        for idx, video in enumerate(videos):
            print(f"Video {idx + 1} of {len(videos)}")
            video_path = os.path.join(video_dir_path, video)
            video_to_frames(video_path, frames_dir)


def augment_frames():
    current_dir = os.path.dirname(__file__)
    root_frames_dir = os.path.join(current_dir, "frames")
    root_augmented_frames_dir = os.path.join(current_dir, "augmented_frames")
    if not os.path.exists(root_augmented_frames_dir):
        os.mkdir(root_augmented_frames_dir)
    frames_dirs = os.listdir(root_frames_dir)
    for frames_dir in frames_dirs:
        print(f"Augmenting '{frames_dir}'...")
        frames_dir_path = os.path.join(root_frames_dir, frames_dir)
        frames = os.listdir(frames_dir_path)
        augmented_frames_dir = os.path.join(root_augmented_frames_dir, frames_dir)
        if os.path.exists(augmented_frames_dir):
            rmtree(augmented_frames_dir)
        os.mkdir(augmented_frames_dir)
        with tqdm(total=len(frames)) as pbar:
            for frame in frames:
                frame_path = os.path.join(frames_dir_path, frame)
                image_raw = Image.open(frame_path)
                rotated_imgs = []
                np.random.rand()
                for i in range(4):
                    angle = np.random.randint(-180, 180)
                    rotated_img = image_raw.rotate(angle, expand=True)
                    rotated_imgs.append(rotated_img)
                all_imgs = [image_raw] + rotated_imgs
                for idx, img in enumerate(all_imgs):
                    img_name = frame.rsplit('.', 1)[0]
                    img_path = os.path.join(augmented_frames_dir, f"{img_name}_{idx}.jpg")
                    img.save(img_path, "jpeg")
                pbar.update(1)


def resize_image_keeping_aspect_ratio(image):
    target_size = (320, 240)
    if image.size[0] / image.size[1] > target_size[0] / target_size[1]:
        scale = target_size[0] / image.size[0]
    else:
        scale = target_size[1] / image.size[1]
    new_size = (
        int(round(image.size[0] * scale)),
        int(round(image.size[1] * scale)))
    resized_img = image.resize(new_size)
    paste_rect = (
        int(round((target_size[0] - new_size[0]) / 2)),
        int(round((target_size[1] - new_size[1]) / 2))
    )
    backdrop = Image.new('RGB', target_size, color='black')
    backdrop.paste(resized_img, paste_rect)
    return backdrop


def convert_frames_to_keypoints(use_augmented=True):
    current_dir = os.path.dirname(__file__)
    root_frames_dir = os.path.join(current_dir, "augmented_frames" if use_augmented else "frames")
    keypoints_dir = os.path.join(current_dir, "augmented_keypoints" if use_augmented else "keypoints")
    if not os.path.exists(keypoints_dir):
        os.mkdir(keypoints_dir)
    frames_dirs = os.listdir(root_frames_dir)
    with tf.Session() as sess:
        construct_graph(current_dir, sess)
        for frames_dir in frames_dirs:
            print(f"Saving for class '{frames_dir}'")
            frames_dir_path = os.path.join(root_frames_dir, frames_dir)
            frames = os.listdir(frames_dir_path)
            class_keypoints = np.empty([len(frames), 21 * 2])
            with tqdm(total=len(frames)) as pbar:
                for idx, frame in enumerate(frames):
                    frame_path = os.path.join(frames_dir_path, frame)
                    image_raw = Image.open(frame_path)
                    image_raw = resize_image_keeping_aspect_ratio(image_raw)
                    image_raw = np.array(image_raw)
                    image_v = (image_raw.astype(np.float32) / 255.0) - 0.5
                    frame_keypoints = get_keypoints(np.expand_dims(image_v, 0), sess)
                    class_keypoints[idx] = frame_keypoints.flatten()
                    pbar.update(1)
            keypoints_path = os.path.join(keypoints_dir, frames_dir)
            np.save(keypoints_path, class_keypoints)


def construct_graph(current_dir, sess):
    root_dir = os.path.dirname(current_dir)
    weight_files = ['orig_model/weights/handsegnet-rhd.pickle',
                    'orig_model/weights/posenet-rhd-stb-slr-finetuned.pickle']
    weight_files = [os.path.join(root_dir, wf) for wf in weight_files]
    image = tf.placeholder(tf.float32, shape=(1, 240, 320, 3), name="input")
    net = ColorHandPose3DNetwork()
    net.inference2d(image)
    net.init(sess, weight_files=weight_files)


def get_keypoints(image, sess):
    scoremaps = sess.run("keypoints_scoremap:0", feed_dict={"input:0": image})
    keypoints = detect_keypoints(np.squeeze(scoremaps))
    return keypoints


def show_mask_and_keypoints(filepath):
    current_dir = os.path.dirname(__file__)
    image = Image.open(filepath)
    image = resize_image_keeping_aspect_ratio(image)
    image = (np.array(image).astype(np.float32) / 255.0) - 0.5
    image = np.expand_dims(image, 0)
    with tf.Session() as sess:
        construct_graph(current_dir, sess)
        scoremaps, hand_mask = sess.run(["keypoints_scoremap:0", "hand_mask:0"], feed_dict={"input:0": image})
        keypoints = detect_keypoints(np.squeeze(scoremaps))
        print(keypoints)
        hand_mask_img = Image.fromarray((hand_mask.squeeze() * 255), mode='F')
        hand_mask_img.show()


def main():
    convert_videos_to_frames()
    augment_frames()
    convert_frames_to_keypoints(use_augmented=True)


if __name__ == "__main__":
    main()
