import numpy as np
import tensorflow as tf
from orig_model.nets.ColorHandPose3DNetwork import ColorHandPose3DNetwork, single_obj_scoremap, calc_center_bb, \
    crop_image_from_xy
import coremltools
import scipy.misc
from utils import save_model, update_multiarray_to_float32, rename_input, rename_output
import os


# <editor-fold desc="Utils">
def _rpc(path):
    cur_dir = os.path.dirname(__file__)
    return os.path.join(cur_dir, path)


def _rpp(path):
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(parent_dir, path)


def _prepare_test_img():
    image_raw = scipy.misc.imread(_rpc("test_img.png"))
    image_raw = scipy.misc.imresize(image_raw, (240, 320))
    image_v = np.expand_dims(image_raw, 0)
    return image_v
# </editor-fold>


# <editor-fold desc="Convert Tensorflow models to CoreML">
def _convert_hand_seg():
    test_img = _prepare_test_img()

    image_tf = tf.placeholder(tf.float32, shape=(1, 240, 320, 3), name="input")
    normalized_img = image_tf / 255.0 - 0.5
    net = ColorHandPose3DNetwork()
    scoremap_list_large = net.inference_detection(normalized_img, False)
    scoremap = tf.identity(scoremap_list_large[-1], name="scoremap")
    tf.identity(scoremap[..., 0], name="bg_scoremap")
    tf.identity(scoremap[..., 1], name="fg_scoremap")

    saver = tf.train.Saver()

    with tf.Session() as sess:
        net.init(sess, weight_files=[_rpp("orig_model/weights/handsegnet-rhd.pickle")])
        saver.save(sess, _rpc("out/hand-seg-net/HandSegNet.ckpt"))
        tf.train.write_graph(sess.graph, _rpc('out/hand-seg-net/'), 'HandSegNet.pbtxt')
        scoremap_val = sess.run(scoremap, feed_dict={image_tf: test_img})

    np.save(_rpc('out/hand-seg-net/HandSegOutput.npy'), scoremap_val)
    model = save_model(["bg_scoremap", "fg_scoremap"],
                       input_name_shape_dict={"input:0": (1, 240, 320, 3)},
                       model_dir=_rpc("out/hand-seg-net"),
                       model_name="HandSegNet")

    spec = model.get_spec()
    metadata = spec.description.metadata
    metadata.license = "GPL v2"
    metadata.author = "Christian Zimmermann and Thomas Brox"
    metadata.shortDescription = "ConvNet for extracting hand probability scoremaps from RGB image"

    input = spec.description.input[0]
    rename_input(spec, input, "image")
    import coremltools.proto.FeatureTypes_pb2 as ft
    input.type.imageType.colorSpace = ft.ImageFeatureType.RGB
    input.type.imageType.height = 240
    input.type.imageType.width = 320

    for idx, feature in enumerate(spec.description.output):
        if idx == 0:
            rename_output(spec, feature, "bgScoremap")
        else:
            rename_output(spec, feature, "fgScoremap")
        feature.type.multiArrayType.shape.extend([1, 240, 320])
        update_multiarray_to_float32(feature)

    coremltools.utils.save_spec(spec, _rpc("out/hand-seg-net/HandSegNet.mlmodel"))


def _save_hand_seg_as_16_bit():
    spec = coremltools.utils.load_spec(_rpc("out/hand-seg-net/HandSegNet.mlmodel"))
    coremltools.utils.convert_neural_network_spec_weights_to_fp16(spec)
    coremltools.utils.save_spec(spec, _rpc("out/hand-seg-net/HandSegNet16bit.mlmodel"))


def _prepare_posenet_input():
    image = tf.identity(_prepare_test_img())
    scoremap = tf.identity(np.load(_rpc("out/hand-seg-net/HandSegOutput.npy")))
    hand_mask = single_obj_scoremap(scoremap)
    center, _, crop_size_best = calc_center_bb(hand_mask)
    crop_size_best *= 1.25
    scale_crop = tf.minimum(tf.maximum(256 / crop_size_best, 0.25), 5.0)
    image_crop = crop_image_from_xy(image, center, 256, scale=scale_crop)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        init.run()
        image_crop_val = sess.run(image_crop)
    np.save(_rpc("out/pose-net/PoseNetInput.npy"), image_crop_val)


def _convert_posenet():
    image_crop = np.load(_rpc("out/pose-net/PoseNetInput.npy"))

    image = tf.placeholder(tf.float32, shape=(1, 256, 256, 3), name="input")
    image = image / 255.0 - 0.5
    net = ColorHandPose3DNetwork()
    scoremap_list_large = net.inference_pose2d(image)
    scoremap = tf.image.resize_images(scoremap_list_large[-1], (256, 256))
    scoremap = tf.identity(scoremap, name="scoremap")

    saver = tf.train.Saver()

    with tf.Session() as sess:
        net.init(sess, weight_files=[_rpp("orig_model/weights/posenet-rhd-stb-slr-finetuned.pickle")])
        saver.save(sess, _rpc("out/pose-net/PoseNet.ckpt"))
        tf.train.write_graph(sess.graph, _rpc("out/pose-net"), 'PoseNet.pbtxt')
        scoremap_val = sess.run(scoremap, feed_dict={image: image_crop})

    np.save(_rpc("out/pose-net/PoseOutput.npy"), scoremap_val)
    model = save_model(["scoremap"],
                       input_name_shape_dict={"input:0": (1, 256, 256, 3)},
                       model_dir=_rpc("out/pose-net"),
                       model_name="PoseNet")

    spec = model.get_spec()

    metadata = spec.description.metadata
    metadata.license = "GPL v2"
    metadata.author = "Christian Zimmermann and Thomas Brox"
    metadata.shortDescription = "ConvNet for extracting keypoint scoremaps from cropped hand image"

    input = spec.description.input[0]
    rename_input(spec, input, "image")
    import coremltools.proto.FeatureTypes_pb2 as ft
    input.type.imageType.colorSpace = ft.ImageFeatureType.RGB
    input.type.imageType.height = 256
    input.type.imageType.width = 256

    for feature in spec.description.output:
        rename_output(spec, feature, "keypointScoremaps")
        update_multiarray_to_float32(feature)

    coremltools.utils.save_spec(spec, _rpc("out/pose-net/PoseNet.mlmodel"))


def _convert_all(hand_seg_16_bit=False):
    _convert_hand_seg()
    if hand_seg_16_bit:
        _save_hand_seg_as_16_bit()
    _prepare_posenet_input()
    _convert_posenet()
# </editor-fold>


# <editor-fold desc="Comparing Tensorflow and CoreML models outputs">
def _test_model(model_path, input_name, input, output_name, output_file):
    from PIL import Image

    if not isinstance(input, Image):
        if isinstance(input, str):
            input = np.load(input)
        input = input.squeeze()
    input = scipy.misc.toimage(input)
    original_output = np.load(output_file)
    model = coremltools.models.MLModel(model_path)
    coreml_output = model.predict({input_name: input})[output_name]
    if len(coreml_output.shape) == 3:
        coreml_output = np.transpose(coreml_output, [1, 2, 0])
    coreml_output = np.reshape(coreml_output, original_output.shape)
    coreml_output = coreml_output.astype(np.float32)
    outputs_diff = coreml_output - original_output
    print("mean:", np.mean(outputs_diff), "std:", np.std(outputs_diff))


def _test_hand_seg():
    _test_model(
        model_path=_rpc("out/hand-seg-net/HandSegNet.mlmodel"),
        input_name="image",
        input=_prepare_test_img(),
        output_name="scoremap__0",
        output_file=_rpc("out/hand-seg-net/HandSegOutput.npy")
    )


def _test_posenet():
    _test_model(
        model_path=_rpc("out/pose-net/PoseNet.mlmodel"),
        input_name="input__0",
        input=_rpc("out/pose-net/PoseNetInput.npy"),
        output_name="scoremap__0",
        output_file=_rpc("out/pose-net/PoseOutput.npy")
    )


def _test_all():
    print("Testing HandSegNet...")
    _test_hand_seg()
    print("Testing PoseNet...")
    _test_posenet()


# </editor-fold>


# <editor-fold desc="Debugging">
def _show_hand_crop_image():
    image_crop = np.load(_rpc("out/pose-net/PoseNetInput.npy"))
    image_crop = np.squeeze(image_crop)
    scipy.misc.toimage(image_crop).show()


def _save_hand_seg_output_normalized():
    scoremap = np.load(_rpc("out/hand-seg-net/HandSegOutput.npy"))
    scoremap = scoremap[..., 1]
    scoremap = (scoremap - scoremap.min()) / (scoremap.max() - scoremap.min())
    scoremap *= 255.0
    scoremap = np.round(scoremap)
    scoremap = scoremap.astype(np.uint8)
    scoremap.tofile(_rpc("out/hand-seg-net/HandSegOutputNormalized"))


def _save_hand_seg_output_raw():
    scoremap = np.load(_rpc("out/hand-seg-net/HandSegOutput.npy"))
    scoremap = scoremap.astype(np.float32)
    bg_scoremap = scoremap[..., 0]
    fg_scoremap = scoremap[..., 1]
    fg_scoremap.tofile(_rpc("out/hand-seg-net/FgScoremap"))
    bg_scoremap.tofile(_rpc("out/hand-seg-net/BgScoremap"))


def _save_hand_seg_output_as_image():
    scoremap = np.load(_rpc("out/hand-seg-net/HandSegOutput.npy"))
    scoremap = np.squeeze(scoremap)
    scipy.misc.imsave(_rpc("out/hand-seg-net/HandSegOutput.jpg"), scoremap[..., 1])


def _show_hand_mask():
    mask = np.load(_rpc("out/pose-net-prep/HandMask.npy"))
    mask = np.squeeze(mask)
    scipy.misc.toimage(mask).show()
# </editor-fold>


def main():
    _convert_all(hand_seg_16_bit=False)
    _convert_posenet()
    _test_all()


if __name__ == "__main__":
    main()
