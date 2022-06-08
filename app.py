#!/usr/bin/env python

from __future__ import annotations

import argparse
import pathlib
import tarfile

import gradio as gr

from model import AppModel

DESCRIPTION = '''# ViTPose

This is an unofficial demo for [https://github.com/ViTAE-Transformer/ViTPose](https://github.com/ViTAE-Transformer/ViTPose).

Related app: [https://huggingface.co/spaces/Gradio-Blocks/ViTPose](https://huggingface.co/spaces/Gradio-Blocks/ViTPose)

'''
FOOTER = '<img id="visitor-badge" alt="visitor badge" src="https://visitor-badge.glitch.me/badge?page_id=hysts.vitpose_video" />'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--theme', type=str)
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    return parser.parse_args()


def set_example_video(example: list) -> dict:
    return gr.Video.update(value=example[0])


def extract_tar() -> None:
    if pathlib.Path('mmdet_configs/configs').exists():
        return
    with tarfile.open('mmdet_configs/configs.tar') as f:
        f.extractall('mmdet_configs')


def main():
    args = parse_args()

    extract_tar()

    model = AppModel(device=args.device)

    with gr.Blocks(theme=args.theme, css='style.css') as demo:
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column():
                input_video = gr.Video(label='Input Video',
                                       format='mp4',
                                       elem_id='input_video')
                with gr.Group():
                    detector_name = gr.Dropdown(
                        list(model.det_model.MODEL_DICT.keys()),
                        value=model.det_model.model_name,
                        label='Detector')
                    pose_model_name = gr.Dropdown(
                        list(model.pose_model.MODEL_DICT.keys()),
                        value=model.pose_model.model_name,
                        label='Pose Model')
                    det_score_threshold = gr.Slider(
                        0,
                        1,
                        step=0.05,
                        value=0.5,
                        label='Box Score Threshold')
                    max_num_frames = gr.Slider(
                        1,
                        300,
                        step=1,
                        value=60,
                        label='Maximum Number of Frames')
                    predict_button = gr.Button(value='Predict')
                    pose_preds = gr.Variable()

                    paths = sorted(pathlib.Path('videos').rglob('*.mp4'))
                    example_videos = gr.Dataset(components=[input_video],
                                                samples=[[path.as_posix()]
                                                         for path in paths])

            with gr.Column():
                with gr.Group():
                    result = gr.Video(label='Result',
                                      format='mp4',
                                      elem_id='result')
                    vis_kpt_score_threshold = gr.Slider(
                        0,
                        1,
                        step=0.05,
                        value=0.3,
                        label='Visualization Score Threshold')
                    vis_dot_radius = gr.Slider(1,
                                               10,
                                               step=1,
                                               value=4,
                                               label='Dot Radius')
                    vis_line_thickness = gr.Slider(1,
                                                   10,
                                                   step=1,
                                                   value=2,
                                                   label='Line Thickness')
                    redraw_button = gr.Button(value='Redraw')

        gr.Markdown(FOOTER)

        detector_name.change(fn=model.det_model.set_model,
                             inputs=detector_name,
                             outputs=None)
        pose_model_name.change(fn=model.pose_model.set_model,
                               inputs=pose_model_name,
                               outputs=None)
        predict_button.click(fn=model.run,
                             inputs=[
                                 input_video,
                                 detector_name,
                                 pose_model_name,
                                 det_score_threshold,
                                 max_num_frames,
                                 vis_kpt_score_threshold,
                                 vis_dot_radius,
                                 vis_line_thickness,
                             ],
                             outputs=[
                                 result,
                                 pose_preds,
                             ])
        redraw_button.click(fn=model.visualize_pose_results,
                            inputs=[
                                input_video,
                                pose_preds,
                                vis_kpt_score_threshold,
                                vis_dot_radius,
                                vis_line_thickness,
                            ],
                            outputs=result)

        example_videos.click(fn=set_example_video,
                             inputs=example_videos,
                             outputs=input_video)

    demo.launch(
        enable_queue=args.enable_queue,
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()
