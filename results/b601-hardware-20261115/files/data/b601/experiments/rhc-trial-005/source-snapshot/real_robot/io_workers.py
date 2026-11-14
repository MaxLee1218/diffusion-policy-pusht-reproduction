"""Bounded background I/O; actuator code only polls or appends memory queues."""

import json
from pathlib import Path
import queue
import re
import threading


class AsyncFileRPC:
    def __init__(self, directory, prepare):
        self.directory, self.prepare = Path(directory), prepare
        self.pending, self.responses = queue.Queue(8), queue.Queue(64)
        self.stop_event = threading.Event()
        self.seen, self.error = set(), None
        try:
            previous=json.loads((self.directory/"model-request.json").read_text(encoding="utf-8"))
            self.seen.add(previous["id"])
        except (OSError,ValueError,KeyError,TypeError):
            pass
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def pop(self):
        try:
            return self.pending.get_nowait()
        except queue.Empty:
            return None

    def respond(self, request, **result):
        try:
            self.responses.put_nowait((request["id"], {"id":request["id"], **result}))
        except queue.Full:
            self.error = "RPC response queue is full"

    def _run(self):
        while not self.stop_event.is_set() or not self.responses.empty():
            try:
                while True:
                    identifier, response = self.responses.get_nowait()
                    (self.directory/(identifier+".response.json")).write_text(
                        json.dumps(response, allow_nan=False), encoding="utf-8")
            except queue.Empty:
                pass
            except BaseException as error:
                self.error = repr(error)
            if not self.stop_event.is_set():
                try:
                    request = json.loads((self.directory/"model-request.json").read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    request = None
                if isinstance(request,dict):
                    identifier = request.get("id", "")
                    if isinstance(identifier,str) and re.fullmatch(r"[0-9a-f]{32}",identifier) and identifier not in self.seen:
                        self.seen.add(identifier)
                        try:
                            prepared = self.prepare(request)
                            if prepared is not None:
                                self.pending.put_nowait(prepared)
                        except Exception as error:
                            self.respond(request,ok=False,error=str(error))
            self.stop_event.wait(.003)

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=5)


class AsyncJSONLines:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.pending = queue.Queue(2000)
        self.stop_event, self.finalized = threading.Event(), threading.Event()
        self.error = None
        self.submitted = self.written = 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, name, row):
        if self.error or self.stop_event.is_set():
            return
        try:
            self.pending.put_nowait((name,row))
            self.submitted += 1
        except queue.Full:
            self.error = "Event log queue is full"

    def _run(self):
        streams = {}
        try:
            while not self.stop_event.is_set() or not self.pending.empty():
                try:
                    name,row=self.pending.get(timeout=.1)
                except queue.Empty:
                    continue
                if name not in streams:
                    streams[name]=(self.directory/(name+".jsonl")).open("a",encoding="utf-8",buffering=1)
                streams[name].write(json.dumps(row,allow_nan=False)+"\n")
                self.written += 1
        except BaseException as error:
            self.error=repr(error)
        finally:
            for stream in streams.values():
                stream.close()
            self.finalized.set()

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=5)
