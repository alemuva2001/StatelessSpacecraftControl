"""Identical episode preprocessing with optional bounded-memory loading."""
from pathlib import Path
from proportional_data import load_episode


class EpisodeStore:
    def __init__(self, folder, size, mode='memory'):
        if mode not in ('memory', 'stream'):
            raise ValueError('Expected memory or stream')
        self.folder=Path(folder)
        self.size=size
        self.mode=mode
        self._cache={}

    def __getitem__(self, name):
        if name in self._cache:
            return self._cache[name]
        episode=load_episode(self.folder/name,self.size)
        if self.mode=='memory':
            self._cache[name]=episode
        return episode
