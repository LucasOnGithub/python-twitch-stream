#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stream NumPy RGB frames and optional PCM audio to Twitch with FFmpeg."""

import os
import queue
import socket
import subprocess
import threading
import time

import numpy as np
import requests

AUDIORATE = 44100


class TwitchOutputStream(object):
    """Stream RGB video and stereo audio to Twitch.

    Existing positional constructor parameters and send methods are retained.
    Optional video_codec='h264_nvenc' enables NVIDIA hardware encoding with
    an FFmpeg build that includes NVENC. ingest_url permits an RTMP test sink.
    """

    def __init__(self, twitch_stream_key, width=640, height=480,
                 fps=30., ffmpeg_binary="ffmpeg", enable_audio=False,
                 verbose=False, video_codec="libx264", video_bitrate=3000,
                 audio_bitrate=160, ingest_url=None):
        self.twitch_stream_key = twitch_stream_key
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        if self.width <= 0 or self.height <= 0 or self.fps <= 0:
            raise ValueError("Width, height and fps must be positive")
        if self.width % 2 or self.height % 2:
            raise ValueError("H.264 video width and height must be even")
        self.ffmpeg_binary = ffmpeg_binary
        self.verbose = verbose
        self.audio_enabled = enable_audio
        self.video_codec = video_codec
        self.video_bitrate = int(video_bitrate)
        self.audio_bitrate = int(audio_bitrate)
        self.ingest_url = ingest_url
        if self.video_bitrate <= 0 or self.audio_bitrate <= 0:
            raise ValueError("Bitrates must be positive")
        self.ffmpeg_process = None
        self.audio_pipe = None  # Compatibility: audio now uses a TCP socket.
        self._audio_listener = None
        self._audio_connection = None
        self.closed = False
        self.reset()

    def get_closest_ingest(self):
        """Choose a Twitch default ingest, with a stable fallback.

        The ingest API does not promise the geographically closest server.
        Callers may override its choice using ingest_url.
        """
        if self.ingest_url:
            return self.ingest_url
        fallback = "rtmp://live.twitch.tv/app/{stream_key}"
        template = fallback
        try:
            response = requests.get("https://ingest.twitch.tv/ingests",
                                    timeout=8)
            response.raise_for_status()
            ingests = response.json().get("ingests") or []
            candidate = next((i for i in ingests if i.get("default")),
                             ingests[0] if ingests else None)
            if candidate:
                supplied = candidate.get("url_template", "")
                if (supplied.startswith(("rtmp://", "rtmps://"))
                        and "{stream_key}" in supplied):
                    template = supplied
        except (requests.RequestException, ValueError, TypeError, KeyError):
            if self.verbose:
                print("Twitch ingest discovery failed; using default server")
        if self.verbose:
            print("Selected Twitch ingest:", template.split("/app/")[0])
        return template.replace("{stream_key}", self.twitch_stream_key)

    def reset(self):
        """Restart FFmpeg, cleaning up any previous process and audio pipe."""
        self.close()
        self.closed = False
        command = [
            self.ffmpeg_binary,
            "-hide_banner", "-loglevel", "info" if self.verbose else "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-video_size", "%dx%d" % (self.width, self.height),
            "-framerate", str(self.fps), "-i", "pipe:0",
        ]

        if self.audio_enabled:
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                listener.bind(("127.0.0.1", 0))
                listener.listen(1)
                listener.settimeout(10)
            except Exception:
                listener.close()
                raise
            self._audio_listener = listener
            port = listener.getsockname()[1]
            command += [
                "-f", "s16le", "-ar", str(AUDIORATE), "-ac", "2",
                "-i", "tcp://127.0.0.1:%d?tcp_nodelay=1" % port,
            ]
        else:
            command += [
                "-f", "lavfi", "-i",
                "anullsrc=channel_layout=stereo:sample_rate=%d" % AUDIORATE,
            ]

        gop = max(1, int(round(self.fps * 2)))
        command += [
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", self.video_codec, "-pix_fmt", "yuv420p",
            "-r", str(self.fps),
            "-b:v", "%dk" % self.video_bitrate,
            "-minrate", "%dk" % self.video_bitrate,
            "-maxrate", "%dk" % self.video_bitrate,
            "-bufsize", "%dk" % (self.video_bitrate * 2),
            "-g", str(gop),
        ]
        if self.video_codec == "libx264":
            command += ["-preset", "veryfast", "-tune", "zerolatency"]
        elif self.video_codec == "h264_nvenc":
            command += ["-preset", "p4", "-rc", "cbr"]
        command += [
            "-c:a", "aac", "-ar", str(AUDIORATE), "-ac", "2",
            "-b:a", "%dk" % self.audio_bitrate,
            "-f", "flv", self.get_closest_ingest(),
        ]
        try:
            self.ffmpeg_process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=None if self.verbose else subprocess.DEVNULL,
                stderr=None if self.verbose else subprocess.DEVNULL,
            )
        except Exception:
            self.close()
            raise

    def close(self):
        """Stop sending, release sockets, and reap FFmpeg on all platforms."""
        self.closed = True
        for attr in ("_audio_connection", "_audio_listener"):
            conn = getattr(self, attr, None)
            if conn is not None:
                try:
                    conn.close()
                finally:
                    setattr(self, attr, None)
        process = getattr(self, "ffmpeg_process", None)
        if process is not None:
            self.ffmpeg_process = None
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def send_video_frame(self, frame):
        """Write an RGB frame, shape (height, width, 3), floats in [0, 1]."""
        if self.closed or self.ffmpeg_process is None:
            raise OSError("Twitch video stream is closed")
        frame = np.asarray(frame)
        if frame.shape != (self.height, self.width, 3):
            raise ValueError("Expected video frame shape (%d, %d, 3)" %
                             (self.height, self.width))
        pixels = np.clip(frame * 255, 0, 255).astype(np.uint8)
        self.ffmpeg_process.stdin.write(pixels.tobytes())

    def send_audio(self, left_channel, right_channel):
        """Write two equally sized PCM float channels with values [-1, 1]."""
        if not self.audio_enabled:
            raise RuntimeError("Audio is disabled for this stream")
        if self.closed or self.ffmpeg_process is None:
            raise OSError("Twitch audio stream is closed")
        left = np.asarray(left_channel)
        right = np.asarray(right_channel)
        if left.ndim != 1 or right.shape != left.shape:
            raise ValueError("Audio channels must be equal one-dimensional arrays")
        if self._audio_connection is None:
            self._audio_connection, _ = self._audio_listener.accept()
            self._audio_connection.settimeout(10)
        pcm = np.column_stack((left, right))
        pcm = np.clip(pcm * 32767, -32768, 32767).astype("<i2")
        self._audio_connection.sendall(pcm.tobytes())


class TwitchOutputStreamRepeater(TwitchOutputStream):
    """
    This stream makes sure a steady framerate is kept by repeating the
    last frame when needed.

    Note: this will not generate a stable, stutter-less stream!
     It does not keep a buffer and you cannot synchronize using this
     stream. Use TwitchBufferedOutputStream for this.
    """
    def __init__(self, *args, **kwargs):
        super(TwitchOutputStreamRepeater, self).__init__(*args, **kwargs)

        self.lastframe = np.ones((self.height, self.width, 3))
        self._send_last_video_frame()   # Start sending the stream

        if self.audio_enabled:
            # some audible sine waves
            xl = np.linspace(0.0, 10*np.pi, int(AUDIORATE/self.fps) + 1)[:-1]
            xr = np.linspace(0.0, 100*np.pi, int(AUDIORATE/self.fps) + 1)[:-1]
            self.lastaudioframe_left = np.sin(xl)
            self.lastaudioframe_right = np.sin(xr)
            self._send_last_audio()   # Start sending the stream

    def _send_last_video_frame(self):
        if self.closed:
            return
        try:
            super(TwitchOutputStreamRepeater,
                  self).send_video_frame(self.lastframe)
        except OSError:
            # stream has been closed.
            # This function is still called once when that happens.
            pass
        else:
            # send the next frame at the appropriate time
            threading.Timer(1./self.fps,
                            self._send_last_video_frame).start()

    def _send_last_audio(self):
        if self.closed:
            return
        try:
            super(TwitchOutputStreamRepeater,
                  self).send_audio(self.lastaudioframe_left,
                                   self.lastaudioframe_right)
        except OSError:
            # stream has been closed.
            # This function is still called once when that happens.
            pass
        else:
            # send the next frame at the appropriate time
            threading.Timer(1./self.fps,
                            self._send_last_audio).start()

    def send_video_frame(self, frame):
        """Send frame of shape (height, width, 3)
        with values between 0 and 1.

        :param frame: array containing the frame.
        :type frame: numpy array with shape (height, width, 3)
            containing values between 0.0 and 1.0
        """
        self.lastframe = frame

    def send_audio(self, left_channel, right_channel):
        """Add the audio samples to the stream. The left and the right
        channel should have the same shape.

        :param left_channel: array containing the audio signal.
        :type left_channel: numpy array with shape (k, )
            containing values between -1.0 and 1.0. k can be any integer
        :param right_channel: array containing the audio signal.
        :type right_channel: numpy array with shape (k, )
            containing values between -1.0 and 1.0. k can be any integer
        """
        self.lastaudioframe_left = left_channel
        self.lastaudioframe_right = right_channel


class TwitchBufferedOutputStream(TwitchOutputStream):
    """
    This stream makes sure a steady framerate is kept by buffering
    frames. Make sure not to have too many frames in buffer, since it
    will increase the memory load considerably!

    Adding frames is thread safe.
    """
    def __init__(self, *args, **kwargs):
        super(TwitchBufferedOutputStream, self).__init__(*args, **kwargs)
        self.last_frame = np.ones((self.height, self.width, 3))
        self.last_frame_time = None
        self.next_video_send_time = None
        self.frame_counter = 0
        self.q_video = queue.PriorityQueue()

        # don't call the functions directly, as they block on the first
        # call
        self.t = threading.Timer(0.0, self._send_video_frame)
        self.t.daemon = True
        self.t.start()

        if self.audio_enabled:
            # send audio at about the same rate as video
            # this can be changed
            self.last_audio = (np.zeros((int(AUDIORATE/self.fps), )),
                               np.zeros((int(AUDIORATE/self.fps), )))
            self.last_audio_time = None
            self.next_audio_send_time = None
            self.audio_frame_counter = 0
            self.q_audio = queue.PriorityQueue()
            self.t = threading.Timer(0.0, self._send_audio)
            self.t.daemon = True
            self.t.start()

    def _send_video_frame(self):
        if self.closed:
            return
        start_time = time.monotonic()
        try:
            frame = self.q_video.get_nowait()
            # frame[0] is frame count of the frame
            # frame[1] is the frame
            frame = frame[1]
        except IndexError:
            frame = self.last_frame
        except queue.Empty:
            frame = self.last_frame
        else:
            self.last_frame = frame

        try:
            super(TwitchBufferedOutputStream, self
                  ).send_video_frame(frame)
        except OSError:
            # stream has been closed.
            # This function is still called once when that happens.
            # Don't call this function again and everything should be
            # cleaned up just fine.
            return

        # send the next frame at the appropriate time
        if self.next_video_send_time is None:
            self.t = threading.Timer(1./self.fps, self._send_video_frame)
            self.next_video_send_time = start_time + 1./self.fps
        else:
            self.next_video_send_time += 1./self.fps
            next_event_time = self.next_video_send_time - start_time
            if next_event_time > 0:
                self.t = threading.Timer(next_event_time,
                                         self._send_video_frame)
            else:
                # we should already have sent something!
                #
                # not allowed for recursion problems :-(
                # (maximum recursion depth)
                # self.send_me_last_frame_again()
                #
                # other solution:
                self.t = threading.Timer(0.001, self._send_video_frame)

        self.t.daemon = True
        self.t.start()

    def _send_audio(self):
        if self.closed:
            return
        start_time = time.monotonic()
        try:
            _, left_audio, right_audio = self.q_audio.get_nowait()
        except IndexError:
            left_audio, right_audio = self.last_audio
        except queue.Empty:
            left_audio, right_audio = self.last_audio
        else:
            self.last_audio = (left_audio, right_audio)

        try:
            super(TwitchBufferedOutputStream, self
                  ).send_audio(left_audio, right_audio)
        except OSError:
            # stream has been closed.
            # This function is still called once when that happens.
            # Don't call this function again and everything should be
            # cleaned up just fine.
            return

        # send the next frame at the appropriate time
        downstream_time = len(left_audio) / AUDIORATE

        if self.next_audio_send_time is None:
            self.t = threading.Timer(downstream_time,
                                     self._send_audio)
            self.next_audio_send_time = start_time + downstream_time
        else:
            self.next_audio_send_time += downstream_time
            next_event_time = self.next_audio_send_time - start_time
            if next_event_time > 0:
                self.t = threading.Timer(next_event_time,
                                         self._send_audio)
            else:
                # we should already have sent something!
                #
                # not allowed for recursion problems :-(
                # (maximum recursion depth)
                # self.send_me_last_frame_again()
                #
                # other solution:
                self.t = threading.Timer(0.001, self._send_audio)

        self.t.daemon = True
        self.t.start()

    def send_video_frame(self, frame, frame_counter=None):
        """send frame of shape (height, width, 3)
        with values between 0 and 1

        :param frame: array containing the frame.
        :type frame: numpy array with shape (height, width, 3)
            containing values between 0.0 and 1.0
        :param frame_counter: frame position number within stream.
            Provide this when multi-threading to make sure frames don't
            switch position
        :type frame_counter: int
        """
        if frame_counter is None:
            frame_counter = self.frame_counter
            self.frame_counter += 1

        self.q_video.put((frame_counter, frame))

    def send_audio(self,
                   left_channel,
                   right_channel,
                   frame_counter=None):
        """Add the audio samples to the stream. The left and the right
        channel should have the same shape.

        :param left_channel: array containing the audio signal.
        :type left_channel: numpy array with shape (k, )
            containing values between -1.0 and 1.0. l can be any integer
        :param right_channel: array containing the audio signal.
        :type right_channel: numpy array with shape (k, )
            containing values between -1.0 and 1.0. l can be any integer
        :param frame_counter: frame position number within stream.
            Provide this when multi-threading to make sure frames don't
            switch position
        :type frame_counter: int
        """
        if frame_counter is None:
            frame_counter = self.audio_frame_counter
            self.audio_frame_counter += 1

        self.q_audio.put((frame_counter, left_channel, right_channel))

    def get_video_frame_buffer_state(self):
        """Find out how many video frames are left in the buffer.
        The buffer should never run dry, or audio and video will go out
        of sync. Likewise, the more filled the buffer, the higher the
        memory use and the delay between you putting your frame in the
        stream and the frame showing up on Twitch.

        :return integer estimate of the number of video frames left.
        """
        return self.q_video.qsize()

    def get_audio_buffer_state(self):
        """Find out how many audio fragments are left in the buffer.
        The buffer should never run dry, or audio and video will go out
        of sync. Likewise, the more filled the buffer, the higher the
        memory use and the delay between you putting your frame in the
        stream and the frame showing up on Twitch.

        :return integer estimate of the number of audio fragments left.
        """
        return self.q_audio.qsize()

