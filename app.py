from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import mediapipe as mp
import os
import tempfile
import pandas as pd
import time
import logging
from flask_cors import CORS
import traceback

# 设置日志级别，屏蔽警告信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 屏蔽 TensorFlow 日志
logging.getLogger('absl').setLevel(logging.ERROR)  # 屏蔽 absl 库的警告

# 配置Flask应用
app = Flask(__name__)
CORS(app)
# 增加最大内容长度限制到100MB
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 从16MB增加到100MB

# 创建结果输出目录（如果不存在）
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
# 确保结果目录存在
if not os.path.exists(results_dir):
    os.makedirs(results_dir)


# 添加根路由处理
@app.route('/')
def index():
    return send_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test.html'))


@app.route('/api/process_video', methods=['POST'])
def process_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': '没有上传视频文件'}), 400

        video_file = request.files['video']

        # 创建临时文件保存上传的视频
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_input_path = temp_input.name
        temp_input.close()

        # 创建临时文件保存处理后的视频
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_output_path = temp_output.name
        temp_output.close()

        # 保存上传的视频
        video_file.save(temp_input_path)

        # 处理视频并计数跳绳次数
        count, output_path = process_skipping_rope(temp_input_path, temp_output_path, video_file.filename)

        # 清理临时文件
        if os.path.exists(temp_input_path):
            os.unlink(temp_input_path)

        return jsonify({
            'success': True,
            'count': count,
            'message': f'共检测到 {count} 次跳绳动作',
            'video_url': f'/api/results/{os.path.basename(output_path)}'
        })
    except Exception as e:
        # 记录详细错误信息
        error_msg = traceback.format_exc()
        print(f"处理视频时出错: {error_msg}")
        return jsonify({'error': f'处理视频时出错: {str(e)}'}), 500


@app.route('/api/batch_process', methods=['POST'])
def batch_process():
    if 'videos' not in request.files:
        return jsonify({'error': '没有上传视频文件'}), 400

    video_files = request.files.getlist('videos')

    if not video_files:
        return jsonify({'error': '没有选择视频文件'}), 400

    results = []

    for video_file in video_files:
        # 创建临时文件保存上传的视频
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_input_path = temp_input.name
        temp_input.close()

        # 创建临时文件保存处理后的视频
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_output_path = temp_output.name
        temp_output.close()

        # 保存上传的视频
        video_file.save(temp_input_path)

        # 处理视频并计数跳绳次数
        count, output_path = process_skipping_rope(temp_input_path, temp_output_path, video_file.filename)

        # 清理临时文件
        os.unlink(temp_input_path)

        results.append({
            'filename': video_file.filename,
            'count': count,
            'video_url': f'/api/results/{os.path.basename(output_path)}'
        })

    # 生成Excel文件
    excel_data = [[result['filename'], result['count']] for result in results]
    df = pd.DataFrame(excel_data, columns=['视频名字', '跳绳计数'])
    excel_path = os.path.join(results_dir, 'results.xlsx')
    df.to_excel(excel_path, index=False)

    return jsonify({
        'success': True,
        'results': results,
        'excel_url': '/api/results/results.xlsx'
    })


@app.route('/api/results/<filename>', methods=['GET'])
def get_result_file(filename):
    file_path = os.path.join(results_dir, filename)
    print(f"请求文件: {file_path}")
    print(f"文件存在: {os.path.exists(file_path)}")

    if not os.path.exists(file_path):
        return jsonify({'error': '文件不存在'}), 404
    try:
        # 修改为不设置as_attachment，这样浏览器会直接播放视频而不是下载
        return send_file(file_path, mimetype='video/mp4')
    except Exception as e:
        print(f"获取文件时出错: {str(e)}")
        return jsonify({'error': f'获取文件时出错: {str(e)}'}), 500


def process_skipping_rope(input_path, temp_output_path, original_filename):
    try:
        # 打开视频文件
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return 0, None

        # 获取视频的帧率、宽度和高度
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 定义视频编码器和创建输出视频文件
        # 修改编码器为H.264，更兼容浏览器
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # 使用H.264编码
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

        # 初始化计数器
        count = 0
        # 初始化Mediapipe姿态检测模型
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # 初始化上一帧的关键点y坐标
        prev_hip_y = None
        # 初始化跳跃状态
        is_jumping = False
        # 初始化上一次检测到运动的时间
        last_move_time = time.time()
        # 定义运动静止的时间阈值（秒）
        still_threshold = 2

        center_x = 0
        center_y = 0
        landmark_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 将BGR图像转换为RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 进行姿态检测
            results = pose.process(rgb_frame)

            center_x = 0
            center_y = 0
            landmark_count = 0

            if results.pose_landmarks:
                # 计算人体中心点
                for landmark in results.pose_landmarks.landmark:
                    center_x += landmark.x
                    center_y += landmark.y
                    landmark_count += 1
                if landmark_count > 0:
                    center_x = int(center_x / landmark_count * width)
                    center_y = int(center_y / landmark_count * height)

                # 获取髋关节（关键点23和24）的y坐标
                hip_landmark_23 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
                hip_landmark_24 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
                hip_y = (hip_landmark_23.y + hip_landmark_24.y) / 2

                if prev_hip_y is not None:
                    # 判断是否有运动
                    if abs(hip_y - prev_hip_y) > 0.01:
                        last_move_time = time.time()

                    # 判断是否开始跳跃
                    if hip_y < prev_hip_y and not is_jumping and (time.time() - last_move_time < still_threshold):
                        is_jumping = True
                    # 判断是否落地
                    elif hip_y > prev_hip_y and is_jumping and (time.time() - last_move_time < still_threshold):
                        count += 1
                        is_jumping = False

                prev_hip_y = hip_y

            # 在帧上显示计数，字体更大且位置稍往下
            cv2.putText(frame, f"Count: {count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

            # 绘制中心点
            if landmark_count > 0:
                # 增大中心点的半径
                cv2.circle(frame, (center_x, center_y), 15, (0, 0, 255), -1)
                # 在中心点位置添加文字：人体中心点
                cv2.putText(frame, 'CenterPoint', (center_x + 20, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255),
                            2)

            # 写入输出视频文件
            out.write(frame)

        # 释放资源
        cap.release()
        out.release()
        pose.close()

        # 将处理后的视频移动到结果目录
        # 确保文件名不包含特殊字符
        safe_filename = ''.join(c for c in original_filename if c.isalnum() or c in '._-')
        output_filename = os.path.splitext(safe_filename)[0] + '_output.mp4'
        output_path = os.path.join(results_dir, output_filename)

        # 如果已存在同名文件，先删除
        if os.path.exists(output_path):
            os.remove(output_path)

        # 复制临时文件到结果目录
        import shutil
        shutil.copy2(temp_output_path, output_path)

        # 删除临时输出文件
        if os.path.exists(temp_output_path):
            os.unlink(temp_output_path)

        return count, output_path
    except Exception as e:
        # 确保资源被释放
        if 'cap' in locals() and cap is not None:
            cap.release()
        if 'out' in locals() and out is not None:
            out.release()
        if 'pose' in locals() and pose is not None:
            pose.close()

        # 清理临时文件
        if os.path.exists(temp_output_path):
            os.unlink(temp_output_path)

        # 重新抛出异常
        raise


if __name__ == '__main__':
    # 移除不支持的timeout参数
    app.run(debug=True, threaded=True)
