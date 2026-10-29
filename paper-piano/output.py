"""Audio and MIDI note sinks. Optional backends are imported on construction."""
from pathlib import Path
import re
from typing import Protocol


class NoteSink(Protocol):
    def note_on(self, key: int, velocity: int) -> None: ...
    def note_off(self, key: int) -> None: ...
    def all_off(self) -> None: ...
    def close(self) -> None: ...


class NullSink:
    def note_on(self, key, velocity): pass
    def note_off(self, key): pass
    def all_off(self): pass
    def close(self): pass


class CompositeSink:
    def __init__(self, *sinks): self.sinks = [s for s in sinks if s is not None]
    def note_on(self, key, velocity):
        for s in self.sinks: s.note_on(key, velocity)
    def note_off(self, key):
        for s in self.sinks: s.note_off(key)
    def all_off(self):
        for s in self.sinks: s.all_off()
    def close(self):
        for s in self.sinks: s.close()


class SampleSink:
    def __init__(self, sounds_dir, n_keys=None):
        import pygame
        self.pygame = pygame
        pygame.mixer.pre_init(44100, -16, 2, 256)
        pygame.mixer.init()
        pygame.mixer.set_num_channels(32)
        files = sorted(Path(sounds_dir).glob("*.mp3"), key=lambda p: int(re.match(r"\d+", p.name).group()))
        self.sounds = [pygame.mixer.Sound(str(p)) for p in files]
        self.n_keys = n_keys
        self.channels = {}

    def set_key_count(self, n_keys):
        self.n_keys = n_keys

    def _selected(self):
        count = min(self.n_keys or len(self.sounds), len(self.sounds))
        start = (len(self.sounds)-count)//2
        return self.sounds[start:start+count]

    def note_on(self, key, velocity):
        selected = self._selected()
        if 0 <= key < len(selected):
            ch = selected[key].play()
            if ch: ch.set_volume(max(0, min(127, velocity))/127)
            self.channels[key] = ch

    def note_off(self, key):
        ch = self.channels.pop(key, None)
        if ch and ch.get_busy(): ch.fadeout(120)

    def all_off(self):
        for key in list(self.channels): self.note_off(key)
        self.pygame.mixer.stop()

    def close(self):
        self.all_off(); self.pygame.mixer.quit()


class MidiSink:
    SCALES = {"major": (0,2,4,5,7,9,11), "chromatic": tuple(range(12)), "original_wholetone": tuple(range(0,12,2))}
    def __init__(self, port_name=None, scale="major", root=60, channel=0, virtual=False, pedal_key=None, start_index=0):
        import mido
        self.mido, self.channel, self.root, self.scale = mido, channel, root, scale
        self.pedal_key, self.start_index = pedal_key, start_index
        self.port = mido.open_output(port_name, virtual=virtual)
        self.held = {}

    @staticmethod
    def list_ports():
        import mido
        return mido.get_output_names()

    def note(self, key):
        if self.scale == "original_wholetone": return 31+2*(self.start_index+key)
        degrees = self.SCALES[self.scale]
        return self.root + (key//len(degrees))*12 + degrees[key%len(degrees)]

    def note_on(self, key, velocity):
        note = self.note(key); self.held[key] = note
        self.port.send(self.mido.Message("note_on", note=note, velocity=max(1,min(127,velocity)), channel=self.channel))

    def note_off(self, key):
        note = self.held.pop(key, self.note(key))
        self.port.send(self.mido.Message("note_off", note=note, velocity=0, channel=self.channel))

    def all_off(self):
        for key in list(self.held): self.note_off(key)
        self.port.send(self.mido.Message("control_change", control=123, value=0, channel=self.channel))
        self.port.send(self.mido.Message("control_change", control=64, value=0, channel=self.channel))

    def close(self):
        self.all_off(); self.port.close()
