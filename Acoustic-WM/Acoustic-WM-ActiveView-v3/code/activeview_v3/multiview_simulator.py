"""Isolated sequential renderer; the original two-view server stays unchanged."""
from .runtime import ROOT, UTILITY, output_path
from .simulator_client import Simulator
import subprocess
import os


class RouteSimulator(Simulator):
    def __init__(self, log, safe=False):
        self.log = open(output_path(log), 'a', buffering=1)
        self.process = subprocess.Popen(
            [str(UTILITY/'environments/audiolidar/bin/python'), '-B', '-u', str(ROOT/('scripts/renderer_server_posefix.py' if safe else 'scripts/multiview_renderer_server.py'))],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1,
            env=dict(os.environ, MAGNUM_LOG='quiet', HABITAT_SIM_LOG='quiet', OPENBLAS_NUM_THREADS='1'))

    def route(self, current_floor, visited_positions):
        return self.request(command='route', current_floor=current_floor, visited_positions=visited_positions)
