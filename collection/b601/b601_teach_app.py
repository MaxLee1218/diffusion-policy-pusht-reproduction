"""Native macOS interface for B601-DM pick-and-place teaching."""

import argparse
import fcntl
import json
import os
import queue
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import AppKit as cocoa
import cv2
import objc
from Foundation import NSBundle, NSData, NSObject, NSTimer

from b601_pick_place import (FAST_ARM_STEP, HORIZONTAL_KEYS, MOVES, Kinematics, URDF,
                            control_loop, open_camera, read_camera, run_hardware)


BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "data/b601"
TITLE = "B601 示教"


def output_path():
    return DATA / datetime.now().strftime("pick-place-%Y%m%d-%H%M%S-%f")


def cameras():
    bundle = NSBundle.bundleWithPath_("/System/Library/Frameworks/AVFoundation.framework")
    bundle.load()
    capture_device = objc.lookUpClass("AVCaptureDevice")
    video = capture_device.devicesWithMediaType_("vide")
    muxed = capture_device.devicesWithMediaType_("muxx")
    # Match OpenCV 4.12's AVFoundation index order.
    devices = sorted([*video, *muxed], key=lambda device: str(device.uniqueID()))
    return [(str(device.uniqueID()), str(device.localizedName())) for device in devices]


class TeachApp(NSObject):
    @objc.python_method
    def configure(self, options):
        self.options = options
        self.phase = "idle"
        self.error = ""
        self.latest = None
        self.last_preview = 0
        self.commands = queue.Queue(maxsize=1)
        self.horizontal_lock = threading.Lock()
        self.horizontal_press = None
        self.fine_horizontal_code = None
        self.enable = threading.Event()
        self.stop = threading.Event()
        self.previous_display = None
        self.shown_image = None
        self.saving = False
        self.devices = []

        self.window = cocoa.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            ((0, 0), (890, 572)), cocoa.NSWindowStyleMaskTitled |
            cocoa.NSWindowStyleMaskClosable | cocoa.NSWindowStyleMaskMiniaturizable,
            cocoa.NSBackingStoreBuffered, False,
        )
        self.window.setTitle_(TITLE)
        self.window.setDelegate_(self)
        self.window.setReleasedWhenClosed_(False)
        self.window.center()
        self.label("B601-DM 抓放示教", 24, 524, 520, 32, 24, bold=True)
        self.camera_menu = cocoa.NSPopUpButton.alloc().initWithFrame_pullsDown_(
            ((24, 483), (309, 31)), False
        )
        self.window.contentView().addSubview_(self.camera_menu)
        self.camera_reload = self.button("刷新", "reloadCameras:", 334, 482, 75)
        self.camera_preview = self.button("预览相机", "previewCamera:", 411, 482, 113)
        self.reloadCameras_(None)
        self.preview = cocoa.NSImageView.alloc().initWithFrame_(((24, 121), (500, 357)))
        self.preview.setImageFrameStyle_(cocoa.NSImageFrameGrayBezel)
        self.preview.setImageScaling_(cocoa.NSImageScaleProportionallyUpOrDown)
        self.window.contentView().addSubview_(self.preview)
        self.pose = self.label("末端位置　等待连接", 24, 89, 502, 26, 15)
        self.detail = self.label("", 24, 63, 840, 23, 14)
        self.status = self.label("", 24, 13, 840, 46, 13)
        self.status.setLineBreakMode_(cocoa.NSLineBreakByWordWrapping)

        self.connection = self.button("检查连接", "connect:", 552, 483, 148)
        self.disconnect = self.button("结束示教", "disconnect:", 714, 483, 148)
        self.mode = self.button("切换快档", "command:", 552, 433, 310, "f")
        self.motion_label = self.label("移动末端", 552, 393, 310, 26, 16, bold=True)
        controls = [
            ("X −  ·  S", "s", 552, 355), ("X +  ·  W", "w", 714, 355),
            ("Y −  ·  D", "d", 552, 316), ("Y +  ·  A", "a", 714, 316),
            ("↓ 下降  ·  C", "c", 552, 277), ("↑ 上升  ·  E", "e", 714, 277),
            ("打开夹爪  ·  O", "o", 552, 226), ("闭合夹爪  ·  P", "p", 714, 226),
        ]
        self.motion = [self.button(label, "command:", x, y, 148, key)
                       for label, key, x, y in controls]
        self.record = self.button("开始录制  ·  R", "command:", 552, 165, 148, "r")
        self.save = self.button("保存本轮  ·  ↵", "command:", 714, 165, 148, "\n")
        self.button("打开数据目录", "reveal:", 552, 120, 310)
        self.refresh_(None)
        self.timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            0.10, self, "refresh:", None, True
        )
        self.input_monitor = cocoa.NSEvent.addLocalMonitorForEventsMatchingMask_handler_(
            cocoa.NSEventMaskKeyDown | cocoa.NSEventMaskKeyUp | cocoa.NSEventMaskFlagsChanged |
            cocoa.NSEventMaskLeftMouseDown | cocoa.NSEventMaskLeftMouseUp |
            cocoa.NSEventMaskLeftMouseDragged, self.handle_input
        )
        self.window.makeKeyAndOrderFront_(None)

    @objc.python_method
    def label(self, text, x, y, width, height, size, bold=False):
        label = cocoa.NSTextField.labelWithString_(text)
        label.setFrame_(((x, y), (width, height)))
        font = cocoa.NSFont.boldSystemFontOfSize_ if bold else cocoa.NSFont.systemFontOfSize_
        label.setFont_(font(size))
        self.window.contentView().addSubview_(label)
        return label

    @objc.python_method
    def button(self, text, action, x, y, width, key=None):
        button = cocoa.NSButton.buttonWithTitle_target_action_(text, self, action)
        button.setFrame_(((x, y), (width, 32)))
        if key:
            button.setIdentifier_(key)
        self.window.contentView().addSubview_(button)
        return button

    @objc.python_method
    def ask(self, title, detail, accept):
        alert = cocoa.NSAlert.alloc().init()
        alert.setMessageText_(title)
        alert.setInformativeText_(detail)
        alert.addButtonWithTitle_(accept)
        alert.addButtonWithTitle_("取消")
        alert.buttons()[0].setKeyEquivalent_("")
        return alert.runModal() == cocoa.NSAlertFirstButtonReturn

    def reloadCameras_(self, _sender):
        selected = self.camera_menu.indexOfSelectedItem()
        previous = self.devices[selected][0] if 0 <= selected < len(self.devices) else None
        self.devices = cameras()
        self.camera_menu.removeAllItems()
        if not self.devices:
            self.camera_menu.addItemWithTitle_("没有可用相机")
            return
        for index, (_, name) in enumerate(self.devices):
            self.camera_menu.addItemWithTitle_(f"{name}  [{index}]")
        preferred = next((index for index, (_, name) in enumerate(self.devices)
                          if "realsense" in name.lower() and "405" in name), 0)
        if self.options.camera is not None and 0 <= self.options.camera < len(self.devices):
            preferred = self.options.camera
        selected = next((index for index, (uid, _) in enumerate(self.devices)
                         if uid == previous), preferred)
        self.camera_menu.selectItemAtIndex_(selected)

    @objc.python_method
    def selected_camera(self):
        return self.devices[self.camera_menu.indexOfSelectedItem()]

    @objc.python_method
    def camera_index(self, uid):
        for index, (device_uid, _) in enumerate(cameras()):
            if device_uid == uid:
                return index
        raise RuntimeError("所选相机已断开。请重新接入，再点击「刷新」。")

    def previewCamera_(self, _sender):
        if self.phase != "idle" or not self.devices or self.options.preview:
            return
        uid, name = self.selected_camera()
        self.error = ""
        self.shown_image = None
        self.preview.setImage_(None)
        self.last_preview = 0
        self.phase = "previewing"
        threading.Thread(target=self.preview_camera, args=(uid, name), daemon=True).start()

    @objc.python_method
    def preview_camera(self, uid, name):
        try:
            camera = open_camera(self.camera_index(uid))
            try:
                image, _ = read_camera(camera)
                self.update_preview(image)
                self.error = f"预览成功 · {name} · {image.shape[1]} × {image.shape[0]}"
            finally:
                camera.release()
        except Exception as error:
            self.show_error(error)
        finally:
            self.phase = "idle"

    @objc.python_method
    def show_error(self, error):
        self.error = str(error)
        if self.error.startswith(("Cannot open camera", "Camera frame unavailable")):
            self.error += " 请检查所选相机是否被占用，以及系统设置 → 隐私与安全性 → 相机中的授权。"

    @objc.python_method
    @contextmanager
    def horizontal_input(self):
        with self.horizontal_lock:
            yield self.horizontal_press if self.phase == "active" and not self.saving else None

    @objc.python_method
    def begin_horizontal(self, key, source):
        with self.horizontal_lock:
            self.horizontal_press = (key, source)

    @objc.python_method
    def cancel_horizontal(self, source=None):
        with self.horizontal_lock:
            press = self.horizontal_press
            if press is not None and (source is None or press[1] == source):
                self.horizontal_press = None
                return True
        return False

    @objc.python_method
    def handle_input(self, event):
        kind = event.type()
        if kind == cocoa.NSEventTypeKeyUp:
            code = event.keyCode()
            if self.fine_horizontal_code == code:
                self.fine_horizontal_code = None
            return None if self.cancel_horizontal(("key", code)) else event
        if kind == cocoa.NSEventTypeLeftMouseUp:
            return None if self.cancel_horizontal(("mouse",)) else event

        modifiers = (cocoa.NSEventModifierFlagCommand | cocoa.NSEventModifierFlagControl |
                     cocoa.NSEventModifierFlagOption)
        if event.modifierFlags() & modifiers:
            self.cancel_horizontal()
            self.fine_horizontal_code = None
            return event
        if self.phase != "active" or self.saving or event.window() != self.window or not self.window.isKeyWindow():
            return event
        fast = (self.latest or {}).get("xy_mm") == FAST_ARM_STEP * 1000
        if kind in (cocoa.NSEventTypeLeftMouseDown, cocoa.NSEventTypeLeftMouseDragged):
            return self.handle_horizontal_mouse(event, fast)
        if kind != cocoa.NSEventTypeKeyDown:
            return event
        key = (event.charactersIgnoringModifiers() or "").lower()
        if key == "\r":
            key = "\n"
        if key in HORIZONTAL_KEYS:
            code = event.keyCode()
            if fast:
                if not event.isARepeat():
                    self.begin_horizontal(key, ("key", code))
            elif not event.isARepeat() or self.fine_horizontal_code == code:
                self.fine_horizontal_code = code
                self.enqueue(key)
            return None
        if key in MOVES or key in ("o", "p", "f", "r", "\n"):
            if not event.isARepeat() or key in MOVES:
                self.enqueue(key)
            return None
        return event

    @objc.python_method
    def handle_horizontal_mouse(self, event, fast):
        if not fast:
            return event
        key = next((str(button.identifier()) for button in self.motion[:4] if button.isEnabled() and
                    cocoa.NSPointInRect(button.convertPoint_fromView_(event.locationInWindow(), None),
                                        button.bounds())), None)
        if event.type() == cocoa.NSEventTypeLeftMouseDown:
            if key is None:
                self.cancel_horizontal()
                return event
            self.begin_horizontal(key, ("mouse",))
        else:
            with self.horizontal_input() as press:
                if press is None or press[1] != ("mouse",):
                    return event
            if press[0] != key:
                self.cancel_horizontal(("mouse",))
        return None

    def windowDidResignKey_(self, _notification):
        self.cancel_horizontal()
        self.fine_horizontal_code = None

    def applicationDidResignActive_(self, _notification):
        self.windowDidResignKey_(_notification)

    def connect_(self, _sender):
        if self.options.preview:
            return
        if self.phase == "ready":
            if self.ask("使能 B601-DM？", "请确认工作区已清空，机械臂有稳固支撑，"
                        "并且已核对零位与夹爪方向。", "使能电机"):
                self.phase = "connecting"
                self.enable.set()
            return
        if self.phase != "idle" or not self.devices:
            return
        self.camera_uid, _ = self.selected_camera()
        self.cancel_horizontal()
        self.commands = queue.Queue(maxsize=1)
        self.latest = None
        self.shown_image = None
        self.previous_display = None
        self.preview.setImage_(None)
        self.last_preview = 0
        self.saving = False
        self.enable.clear()
        self.stop.clear()
        self.error = ""
        self.phase = "checking"
        threading.Thread(target=self.run_session, daemon=True).start()

    @objc.python_method
    def run_session(self):
        try:
            with (Path(tempfile.gettempdir()) / f"b601-teach-{os.getuid()}.lock").open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise RuntimeError("另一个 B601 示教窗口正在使用机械臂。") from error
                probe = subprocess.run(
                    ["/usr/bin/pgrep", "-f", "b601_pick_place[.]py record|b601-z-step/record/retry[.]py"],
                    capture_output=True, check=False,
                )
                if probe.returncode == 0:
                    raise RuntimeError("终端示教仍在运行。请先保存并正常退出终端示教。")
                bounds = json.loads(self.options.bounds.read_text(encoding="utf-8"))
                args = SimpleNamespace(port=self.options.port, camera=self.camera_index(self.camera_uid),
                                       bounds=bounds, output=output_path())
                run_hardware(args, Kinematics(URDF), self.confirm, self.run_controls)
        except Exception as error:
            self.show_error(error)
        finally:
            self.cancel_horizontal()
            self.phase = "idle"

    @objc.python_method
    def confirm(self, image):
        if self.stop.is_set():
            return False
        self.update_preview(image)
        self.phase = "ready"
        self.enable.wait()
        return not self.stop.is_set()

    @objc.python_method
    def run_controls(self, robot, camera, model, bounds, limits, output):
        control_loop(self.read_key, self.publish, robot, camera, model, bounds, limits,
                     output, next_output=output_path, horizontal_input=self.horizontal_input)

    @objc.python_method
    def read_key(self):
        if self.stop.is_set():
            return "q"
        try:
            return self.commands.get_nowait()
        except queue.Empty:
            return ""

    @objc.python_method
    def update_preview(self, image):
        now = time.monotonic()
        if now - self.last_preview >= 0.20:
            ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                                    [cv2.IMWRITE_JPEG_QUALITY, 75])
            self.last_preview = now
            if ok:
                self.shown_image = jpeg.tobytes()

    @objc.python_method
    def publish(self, view):
        self.update_preview(view.pop("image"))
        self.latest = view
        if self.phase == "connecting":
            self.phase = "active"

    def command_(self, sender):
        self.enqueue(str(sender.identifier()))

    @objc.python_method
    def enqueue(self, key):
        if self.phase != "active" or self.saving:
            return
        view = self.latest or {}
        if key == "r" and view.get("recording"):
            return
        if key == "\n" and not (view.get("recording") and view.get("samples")):
            return
        self.cancel_horizontal()
        if key not in HORIZONTAL_KEYS:
            self.fine_horizontal_code = None
        try:
            self.commands.put_nowait(key)
        except queue.Full:
            return
        if key == "\n":
            self.saving = True

    def disconnect_(self, _sender):
        self.cancel_horizontal()
        self.fine_horizontal_code = None
        if self.phase == "ready":
            self.phase = "stopping"
            self.stop.set()
            self.enable.set()
        elif self.phase == "active" and not self.saving:
            recording = self.latest and self.latest["recording"]
            detail = "结束示教会关闭电机扭矩。请先确认机械臂有稳固支撑。"
            if recording:
                detail += "\n当前回合还未保存。继续会结束这次录制。"
            if self.ask("结束示教？", detail, "结束示教"):
                self.phase = "stopping"
                self.stop.set()

    def reveal_(self, _sender):
        latest = self.latest or {}
        path = latest.get("saved_path")
        if path:
            cocoa.NSWorkspace.sharedWorkspace().selectFile_inFileViewerRootedAtPath_(
                path, str(DATA))
        else:
            cocoa.NSWorkspace.sharedWorkspace().openFile_(str(DATA))

    def windowShouldClose_(self, _window):
        if self.phase == "idle":
            cocoa.NSApplication.sharedApplication().terminate_(None)
            return False
        self.disconnect_(None)
        return False

    def applicationShouldTerminate_(self, _app):
        if self.phase == "idle":
            return cocoa.NSTerminateNow
        self.disconnect_(None)
        return cocoa.NSTerminateCancel

    def refresh_(self, _timer):
        phase, view = self.phase, self.latest or {}
        image = self.shown_image
        if image is not None and image is not self.previous_display:
            data = NSData.dataWithBytes_length_(image, len(image))
            self.preview.setImage_(cocoa.NSImage.alloc().initWithData_(data))
            self.previous_display = image
        active = phase == "active"
        recording = view.get("recording", False)
        samples = view.get("samples", 0)
        saved = view.get("saved_path")
        if self.saving and (not recording or phase == "idle"):
            self.saving = False
        saving = self.saving
        self.connection.setTitle_("使能电机" if phase == "ready" else "检查连接")
        available = bool(self.devices) and not self.options.preview
        self.connection.setEnabled_(phase in ("idle", "ready") and available)
        self.camera_menu.setEnabled_(phase == "idle" and available)
        self.camera_reload.setEnabled_(phase == "idle" and not self.options.preview)
        self.camera_preview.setEnabled_(phase == "idle" and available)
        self.disconnect.setTitle_("取消连接" if phase == "ready" else "结束示教")
        self.disconnect.setEnabled_(phase in ("ready", "active") and not saving)
        for button in [self.mode, *self.motion]:
            button.setEnabled_(active and not saving)
        self.record.setEnabled_(active and not recording and not saving)
        self.save.setEnabled_(active and recording and samples > 0 and not saving)
        if view:
            x, y, z = view["xyz"] * 1000
            self.pose.setStringValue_(f"末端位置　X {x:+.1f}　Y {y:+.1f}　Z {z:+.1f} mm")
            self.mode.setTitle_(f"切换步长 · F　XY {view['xy_mm']:g} mm / Z {view['z_mm']:g} mm")
        else:
            self.pose.setStringValue_("末端位置　等待连接")
            self.mode.setTitle_("切换步长 · F　XY 2 mm / Z 4 mm")
        fast = view.get("xy_mm") == FAST_ARM_STEP * 1000
        self.motion_label.setStringValue_("水平长按移动 · 松开停止" if fast else "移动末端")
        self.detail.setStringValue_(
            f"{'正在录制' if recording else '当前回合'} · {samples} 帧"
            + (f"　　上次保存 {Path(saved).name}" if saved else "")
        )
        messages = {
            "idle": self.error or ("演示模式" if self.options.preview else "点击「检查连接」开始。"),
            "previewing": "正在读取所选相机…",
            "checking": "正在检查机械臂位置和相机…",
            "ready": "机械臂位置和相机检查通过。点击「使能电机」继续。",
            "connecting": "正在连接机械臂…",
            "stopping": "正在断开机械臂…",
            "active": view.get("message", ""),
        }
        status = messages[phase]
        translations = {
            "Ready. R starts recording after positioning.": "可以操作。准备好后点击「开始录制」。",
            "Recording.": "正在录制。完成抓放后点击「保存本轮」。",
            "Fast steps selected.": "已切换到快档。",
            "Fine steps selected.": "已切换到细调档。",
            "Target sent.": "已发送目标。",
            "Waiting for arm to reach the previous target.": "等待机械臂跟上前一个目标。",
            "Near workspace boundary. Move inward.": "接近工作区边界，请向内移动。",
            "No nearby solution with the current wrist orientation.": "当前夹爪朝向下找不到可行位置。",
            "Solved target is too close to the workspace boundary.": "目标过于接近工作区边界。",
            "Horizontal jog released. Holding position.": "水平移动已松开，保持当前位置。",
        }
        if phase == "active" and status.startswith("Saved "):
            status = f"已保存 {samples} 帧。可以开始下一轮。"
        if phase == "active" and status.endswith(" target sent."):
            key, step, _, axis, *_ = status.split()
            status = f"{key} 已发送 {axis} 轴目标，步长 {step} mm。"
        self.status.setStringValue_("正在保存本轮…" if saving else translations.get(status, status))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bounds", type=Path, required=True)
    parser.add_argument("--port", help="Path to the USB-CAN serial adapter.")
    parser.add_argument("--camera", type=int)
    parser.add_argument("--preview", action="store_true", help="Open without accessing hardware.")
    options = parser.parse_args()
    if not options.preview and not options.port:
        parser.error("--port is required for live recording")

    app = cocoa.NSApplication.sharedApplication()
    app.setActivationPolicy_(cocoa.NSApplicationActivationPolicyRegular)
    delegate = TeachApp.alloc().init()
    app.setDelegate_(delegate)
    delegate.configure(options)
    app.activateIgnoringOtherApps_(True)
    app.run()


if __name__ == "__main__":
    main()
