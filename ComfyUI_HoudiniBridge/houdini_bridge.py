import torch
import gc
import os
import time
from PIL import Image
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class FileWatcher(FileSystemEventHandler):
    def __init__(self, target_path, callback):
        self.target_path = os.path.abspath(target_path)
        self.callback = callback
        self.last_processed = None

    def on_modified(self, event):
        if event.is_directory:
            return
        if os.path.abspath(event.src_path) == self.target_path:
            if self.last_processed != event.src_path:
                self.last_processed = event.src_path
                self.callback(event.src_path)

class HoudiniBridge:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "watch_file": ("STRING", {
                    "default": "/media/gero/Qsync_Ubuntu/Qsync/55_Houdini_Projects_Linux/1_3D/Houdini/1_Scenes/StableHoudini_Linux/Render/Temp/Depth.jpg",
                    "multiline": False
                }),
                "auto_reload": ("BOOLEAN", {
                    "default": True
                }),
                "wait_for_render": ("BOOLEAN", {
                    "default": True
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_houdini_render"
    CATEGORY = "loaders/houdini"

    def __init__(self):
        self.observer = None
        self.latest_image = None
        self.is_waiting = False

    def setup_watcher(self, file_path, callback):
        if self.observer is not None:
            self.observer.stop()
            self.observer.join()
        
        directory = os.path.dirname(file_path)
        self.observer = Observer()
        self.observer.schedule(FileWatcher(file_path, callback), directory, recursive=False)
        self.observer.start()

    def load_image(self, path):
        if path.lower().endswith('.exr'):
            import OpenEXR
            import Imath
            
            exr = OpenEXR.InputFile(path)
            dw = exr.header()['dataWindow']
            size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
            
            FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
            channels = ['R', 'G', 'B']
            pixels = [np.frombuffer(exr.channel(c, FLOAT), dtype=np.float32) for c in channels]
            img = np.stack(pixels).reshape(3, size[1], size[0])
            
            return torch.from_numpy(img).float()
        else:
            try:
                i = Image.open(path)
                i = np.array(i).astype(np.float32) / 255.0
                i = torch.from_numpy(i)[None,]
                if len(i.shape) == 3:
                    i = i.permute(0, 3, 1, 2)
                return i
            except Exception as e:
                print(f"Error loading image {path}: {str(e)}")
                return torch.zeros((1, 3, 64, 64))

    def load_houdini_render(self, watch_file, auto_reload, wait_for_render):
        watch_file = os.path.expandvars(os.path.expanduser(watch_file))
        
        if auto_reload:
            def on_new_render(path):
                if os.path.exists(path):
                    self.latest_image = self.load_image(path)
                    self.is_waiting = False
            
            if self.observer is None:
                self.setup_watcher(watch_file, on_new_render)
        
        if wait_for_render and (self.latest_image is None or self.is_waiting):
            self.is_waiting = True
            while self.is_waiting and not os.path.exists(watch_file):
                time.sleep(0.1)
            
            if os.path.exists(watch_file):
                self.latest_image = self.load_image(watch_file)
                self.is_waiting = False
        
        if self.latest_image is not None:
            return (self.latest_image,)
        
        if os.path.exists(watch_file):
            return (self.load_image(watch_file),)
        
        return (torch.zeros((1, 3, 64, 64)),)

    def __del__(self):
        if self.observer is not None:
            self.observer.stop()
            self.observer.join()
