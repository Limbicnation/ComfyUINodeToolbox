import os
import time
import torch
from PIL import Image
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class HoudiniRenderWatcher(FileSystemEventHandler):
    def __init__(self, callback):
        self.callback = callback
        self.last_processed = None

    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith(('.png', '.jpg', '.exr')):
            if self.last_processed != event.src_path:
                self.last_processed = event.src_path
                self.callback(event.src_path)

class HoudiniBridge:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "watch_directory": ("STRING", {
                    "default": "/tmp/houdini_renders",
                    "multiline": False
                }),
                "file_pattern": ("STRING", {
                    "default": "*.png",
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

    def setup_watcher(self, directory, callback):
        if self.observer is not None:
            self.observer.stop()
        
        self.observer = Observer()
        self.observer.schedule(HoudiniRenderWatcher(callback), directory, recursive=False)
        self.observer.start()

    def load_image(self, path):
        if path.lower().endswith('.exr'):
            import OpenEXR
            import Imath
            
            exr = OpenEXR.InputFile(path)
            dw = exr.header()['dataWindow']
            size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
            
            # Read all channels
            FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
            channels = ['R', 'G', 'B']
            pixels = [np.frombuffer(exr.channel(c, FLOAT), dtype=np.float32) for c in channels]
            img = np.stack(pixels).reshape(3, size[1], size[0])
            
            # Convert to torch tensor
            return torch.from_numpy(img).float()
        else:
            i = Image.open(path)
            i = np.array(i).astype(np.float32) / 255.0
            i = torch.from_numpy(i)[None,]
            if len(i.shape) == 3:
                i = i.permute(0, 3, 1, 2)
            return i

    def load_houdini_render(self, watch_directory, file_pattern, auto_reload, wait_for_render):
        os.makedirs(watch_directory, exist_ok=True)
        
        if auto_reload:
            def on_new_render(path):
                self.latest_image = self.load_image(path)
                self.is_waiting = False
            
            if self.observer is None:
                self.setup_watcher(watch_directory, on_new_render)
        
        if wait_for_render and (self.latest_image is None or self.is_waiting):
            self.is_waiting = True
            while self.is_waiting:
                time.sleep(0.1)
        
        if self.latest_image is not None:
            return (self.latest_image,)
        
        # Return empty image if no render available
        return (torch.zeros((1, 3, 64, 64)),)

    def __del__(self):
        if self.observer is not None:
            self.observer.stop()
            self.observer.join()
