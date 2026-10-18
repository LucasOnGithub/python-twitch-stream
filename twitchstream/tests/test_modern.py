"""Offline regression tests for modern chat and FFmpeg streaming."""

import io
import socket
import threading
from unittest import mock

import numpy as np
import pytest

from twitchstream.chat import TwitchChatStream
from twitchstream.outputvideo import TwitchOutputStream


class FakeSocket:
    def __init__(self, received=()):
        self.received = list(received)
        self.sent = []
        self.closed = False

    def sendall(self, value):
        self.sent.append(value)

    def recv(self, size):
        if self.received:
            return self.received.pop(0)
        raise BlockingIOError()

    def close(self):
        self.closed = True

    def settimeout(self, value):
        pass

    def setblocking(self, value):
        pass


class FakeProcess:
    def __init__(self):
        self.stdin = io.BytesIO()
        self.waited = False

    def wait(self, timeout=None):
        self.waited = True
        return 0

    def terminate(self):
        pass

    def kill(self):
        pass


def test_secure_chat_login_and_channel_selection():
    fake = FakeSocket()
    context = mock.Mock()
    context.wrap_socket.return_value = fake
    with mock.patch("twitchstream.chat.socket.create_connection") as connect:
        with mock.patch("twitchstream.chat.ssl.create_default_context",
                        return_value=context):
            chat = TwitchChatStream("EXAMPLE", "testtoken", verbose=True)
            chat.connect()
    connect.assert_called_once_with(("irc.chat.twitch.tv", 6697), timeout=10)
    assert fake.sent[0] == b"PASS oauth:testtoken\r\n"
    assert fake.sent[1] == b"NICK example\r\n"
    assert fake.sent[3] == b"JOIN #example\r\n"
    chat.join_channel("OtherChannel")
    chat.send_chat_message("hello")
    chat._push_from_buffer()
    assert fake.sent[-1] == b"PRIVMSG #otherchannel :hello\r\n"
    chat.close()
    assert fake.closed


def test_tagged_chat_and_partial_irc_packets():
    fake = FakeSocket([
        b"PING :tmi.twitch.tv\r\n@badge-info=;display-name=Hello :",
        b"tester!tester@tester.tmi.twitch.tv PRIVMSG #example :hello world\r\n"
    ])
    chat = TwitchChatStream("example", "oauth:abc")
    chat.s = fake
    assert chat.twitch_receive_messages() == [{
        "channel": "#example", "username": "tester",
        "message": "hello world"
    }]
    assert b"PONG :tmi.twitch.tv\r\n" in fake.sent


def test_login_failure_is_reported_without_echoing_token():
    chat = TwitchChatStream("example", "secret")
    chat.s = FakeSocket()
    with pytest.raises(OSError, match="rejected"):
        chat._parse_message(":tmi.twitch.tv NOTICE * :Login authentication failed")
    assert chat.s is None


def make_stream(**kwargs):
    process = FakeProcess()
    factory = mock.Mock(return_value=process)
    patcher = mock.patch("twitchstream.outputvideo.subprocess.Popen", factory)
    patcher.start()
    try:
        stream = TwitchOutputStream("dummy", width=4, height=4, fps=30,
                                    ingest_url="test.flv", **kwargs)
    finally:
        patcher.stop()
    return stream, process, factory.call_args.args[0]


def test_video_uses_aac_and_two_second_keyframes():
    stream, process, args = make_stream()
    try:
        assert args[args.index("-c:v") + 1] == "libx264"
        assert args[args.index("-c:a") + 1] == "aac"
        assert args[args.index("-g") + 1] == "60"
        assert "test.flv" == args[-1]
        stream.send_video_frame(np.ones((4, 4, 3)))
        assert len(process.stdin.getvalue()) == 4 * 4 * 3
        with pytest.raises(ValueError):
            stream.send_video_frame(np.zeros((4, 3, 3)))
        with pytest.raises(RuntimeError):
            stream.send_audio(np.zeros(10), np.zeros(10))
    finally:
        stream.close()
    assert process.waited


def test_nvenc_encoder_option():
    stream, process, args = make_stream(video_codec="h264_nvenc")
    try:
        assert args[args.index("-c:v") + 1] == "h264_nvenc"
        assert args[args.index("-preset") + 1] == "p4"
    finally:
        stream.close()


def test_cross_platform_audio_transport():
    stream, process, args = make_stream(enable_audio=True)
    client = None
    try:
        endpoint = stream._audio_listener.getsockname()
        client = socket.create_connection(endpoint, timeout=5)
        stream.send_audio(np.array([0.0, 1.0]),
                          np.array([0.0, -1.0]))
        payload = client.recv(8)
        assert len(payload) == 8
        assert args[args.index("-f") + 1] == "rawvideo"
        assert any("tcp://127.0.0.1:" in arg for arg in args)
    finally:
        if client is not None:
            client.close()
        stream.close()


def test_twitch_ingest_uses_documented_endpoint():
    response = mock.Mock()
    response.json.return_value = {
        "ingests": [{"default": True,
                     "url_template": "rtmp://server.example/app/{stream_key}"}]
    }
    with mock.patch("twitchstream.outputvideo.requests.get",
                    return_value=response) as request:
        with mock.patch("twitchstream.outputvideo.subprocess.Popen",
                        return_value=FakeProcess()) as spawn:
            with TwitchOutputStream("sample-key", width=4, height=4):
                assert spawn.call_args.args[0][-1] == (
                    "rtmp://server.example/app/sample-key"
                )
    request.assert_called_once_with("https://ingest.twitch.tv/ingests",
                                    timeout=8)


def test_ffmpeg_can_write_a_local_flv(tmp_path):
    """Integration test for FFmpeg command construction, without Twitch."""
    import shutil
    if shutil.which("ffmpeg") is None:
        pytest.skip("FFmpeg is not on PATH")
    output = str(tmp_path / "sample.flv")
    with TwitchOutputStream("unused", width=32, height=32, fps=10,
                            ingest_url=output) as stream:
        for index in range(20):
            image = np.full((32, 32, 3), index / 20.0, dtype=float)
            stream.send_video_frame(image)
    assert (tmp_path / "sample.flv").stat().st_size > 1000
