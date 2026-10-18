#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read and send Twitch IRC chat messages with a TLS connection."""

import errno
import re
import socket
import ssl
import time


class TwitchChatStream(object):
    """Twitch IRC client preserving the existing public chat API.

    Use a user access token with chat:read and chat:edit scopes.
    """

    HOST = "irc.chat.twitch.tv"
    PORT = 6697
    _MESSAGE = re.compile(
        r"^(?:@[^ ]+ )?:([A-Za-z0-9_]+)![^ ]+ PRIVMSG "
        r"#([A-Za-z0-9_]+) :(.*)$"
    )
    _JOIN = re.compile(
        r"^(?:@[^ ]+ )?:[A-Za-z0-9_]+![^ ]+ JOIN #([A-Za-z0-9_]+)$"
    )

    def __init__(self, username, oauth, verbose=False):
        self.username = username.lower()
        self.oauth = oauth if oauth.startswith("oauth:") else "oauth:" + oauth
        self.verbose = verbose
        self.current_channel = self.username
        self.last_sent_time = 0.0
        self.buffer = []
        self.s = None
        self._partial = ""

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if self.s is not None:
            try:
                self.s.close()
            finally:
                self.s = None

    @staticmethod
    def _logged_in_successful(data):
        return not bool(re.search(
            r"Login (?:authentication failed|unsuccessful)|"
            r"Error logging in|Improperly formatted auth",
            data, re.IGNORECASE
        ))

    @staticmethod
    def _check_has_ping(data):
        return data.startswith("PING ")

    @staticmethod
    def _check_has_channel(data):
        match = TwitchChatStream._JOIN.match(data)
        return [match.group(1)] if match else []

    @staticmethod
    def _check_has_message(data):
        return TwitchChatStream._MESSAGE.match(data)

    def connect(self):
        """Connect over TLS, authenticate and join the selected channel."""
        self.close()
        raw = socket.create_connection((self.HOST, self.PORT), timeout=10)
        try:
            conn = ssl.create_default_context().wrap_socket(
                raw, server_hostname=self.HOST
            )
            conn.settimeout(10)
            conn.sendall(("PASS %s\r\n" % self.oauth).encode("utf-8"))
            conn.sendall(("NICK %s\r\n" % self.username).encode("utf-8"))
            conn.sendall(
                b"CAP REQ :twitch.tv/membership twitch.tv/tags twitch.tv/commands\r\n"
            )
            self.s = conn
            self._partial = ""
            self.join_channel(self.current_channel)
            conn.setblocking(False)
        except Exception:
            if self.s is not None:
                self.close()
            else:
                raw.close()
            raise

    def _push_from_buffer(self):
        """Send queued chat without exceeding ordinary Twitch rate limits."""
        if self.buffer and self.s is not None:
            now = time.monotonic()
            if now - self.last_sent_time >= 2.0:
                message = self.buffer[0]
                try:
                    self.s.sendall((message + "\r\n").encode("utf-8"))
                except (BlockingIOError, ssl.SSLWantWriteError):
                    return
                self.buffer.pop(0)
                self.last_sent_time = now
                if self.verbose:
                    print(message)

    def _send(self, message):
        if message:
            self.buffer.append(message.replace("\r", " ").replace("\n", " "))

    def _send_pong(self, payload=":tmi.twitch.tv"):
        # PONG is never rate-limited or queued, or Twitch may disconnect.
        if self.s is not None:
            self.s.sendall(("PONG " + payload + "\r\n").encode("utf-8"))

    def join_channel(self, channel):
        """Join a channel. Messages are now sent to that channel."""
        channel = channel.lstrip("#").lower()
        if not re.fullmatch(r"[a-z0-9_]+", channel):
            raise ValueError("Invalid Twitch channel")
        self.current_channel = channel
        if self.s is not None:
            self.s.sendall(("JOIN #%s\r\n" % channel).encode("utf-8"))

    def send_chat_message(self, message):
        """Queue a message for the selected channel."""
        self._send("PRIVMSG #{0} :{1}".format(self.current_channel, message))

    def _parse_message(self, data):
        if self._check_has_ping(data):
            self._send_pong(data[5:])
            return None

        if not self._logged_in_successful(data):
            self.close()
            raise OSError("Twitch rejected the IRC login/token")

        channels = self._check_has_channel(data)
        if channels:
            self.current_channel = channels[0].lower()

        match = self._check_has_message(data)
        if match:
            return {
                "channel": "#" + match.group(2),
                "username": match.group(1),
                "message": match.group(3),
            }
        return None

    def twitch_receive_messages(self):
        """Nonblocking poll of IRC messages; safe with split TCP packets."""
        if self.s is None:
            raise RuntimeError("Twitch chat is not connected")
        self._push_from_buffer()
        result = []
        while True:
            try:
                chunk = self.s.recv(4096)
            except (BlockingIOError, ssl.SSLWantReadError, ssl.SSLWantWriteError):
                return result
            except OSError as exc:
                if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                    return result
                self.connect()
                return result

            if not chunk:
                self.connect()
                return result

            self._partial += chunk.decode("utf-8", errors="replace")
            if len(self._partial) > 65536:
                self._partial = ""
                raise OSError("Twitch IRC line exceeded maximum length")

            while "\r\n" in self._partial:
                line, self._partial = self._partial.split("\r\n", 1)
                if self.verbose:
                    print(line)
                message = self._parse_message(line)
                if message is not None:
                    result.append(message)
