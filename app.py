#!/usr/bin/env python

from __future__ import annotations

import pathlib
import tarfile

import gradio as gr

from model import AppModel

DESCRIPTION = '''## [ViTPose](https://github.com/ViTAE-Transformer/ViTPose)'''


def extract_tar() -> None:
    if pathlib.Path('mmdet_configs/configs').exists():
        return
    with tarfile.open('mmdet_configs/configs.tar') as f:
        f.extractall('mmdet_configs')


extract_tar()

model = AppModel()

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column():
            input_video = gr.Video(label='Input Video',
                                   format='mp4',
                                   elem_id='input_video')
            detector_name = gr.Dropdown(label='Detector',
                                        choices=list(
                                            model.det_model.MODEL_DICT.keys()),
                                        value=model.det_model.model_name)
            pose_model_name = gr.Dropdown(
                label='Pose Model',
                choices=list(model.pose_model.MODEL_DICT.keys()),
                value=model.pose_model.model_name)
            det_score_threshold = gr.Slider(label='Box Score Threshold',
                                            minimum=0,
                                            maximum=1,
                                            step=0.05,
                                            value=0.5)
            max_num_frames = gr.Slider(label='Maximum Number of Frames',
                                       minimum=1,
                                       maximum=300,
                                       step=1,
                                       value=60)
            predict_button = gr.Button('Predict')
            pose_preds = gr.Variable()

            paths = sorted(pathlib.Path('videos').rglob('*.mp4'))
            gr.Examples(examples=[[path.as_posix()] for path in paths],
                        inputs=input_video)

        with gr.Column():
            result = gr.Video(label='Result', format='mp4', elem_id='result')
            vis_kpt_score_threshold = gr.Slider(
                label='Visualization Score Threshold',
                minimum=0,
                maximum=1,
                step=0.05,
                value=0.3)
            vis_dot_radius = gr.Slider(label='Dot Radius',
                                       minimum=1,
                                       maximum=10,
                                       step=1,
                                       value=4)
            vis_line_thickness = gr.Slider(label='Line Thickness',
                                           minimum=1,
                                           maximum=10,
                                           step=1,
                                           value=2)
            redraw_button = gr.Button('Redraw')

    detector_name.change(fn=model.det_model.set_model, inputs=detector_name)
    pose_model_name.change(fn=model.pose_model.set_model,
                           inputs=pose_model_name)
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

demo.queue(max_size=10).launch(share=True)
