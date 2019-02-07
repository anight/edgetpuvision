"""A demo which runs object detection and streams video to the browser."""

# export TEST_DATA=/usr/lib/python3/dist-packages/edgetpu/test_data
#
# Run face detection model:
# python3 -m edgetpuvision.detect_server \
#   --model ${TEST_DATA}/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite
#
# Run coco model:
# python3 -m edgetpuvision.detect_server \
#   --model ${TEST_DATA}/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
#   --labels ${TEST_DATA}/coco_labels.txt

from .detect import add_render_gen_args, render_gen
from .server import run

def main():
    run(add_render_gen_args, render_gen)

if __name__ == '__main__':
    main()
