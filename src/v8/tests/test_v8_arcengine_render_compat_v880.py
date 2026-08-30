from __future__ import annotations

import unittest

import numpy as np

from v7.environment.arc_adapter import _install_arcengine_render_compatibility


class ArcEngineRenderCompatibilityV880Tests(unittest.TestCase):
    def test_camera_accepts_list_backed_sprite_render(self):
        from arcengine.camera import Camera

        class ListBackedSprite:
            is_visible = True
            layer = 0
            x = 0
            y = 0

            def render(self):
                return [[1, 2], [3, 4]]

        _install_arcengine_render_compatibility()
        camera = Camera(x=0, y=0, width=2, height=2, background=0)
        rendered = camera._raw_render([ListBackedSprite()])

        self.assertIsInstance(rendered, np.ndarray)
        np.testing.assert_array_equal(rendered, np.asarray([[1, 2], [3, 4]], dtype=np.int8))

    def test_compatibility_install_is_idempotent(self):
        from arcengine.camera import Camera

        _install_arcengine_render_compatibility()
        first = Camera._raw_render
        _install_arcengine_render_compatibility()
        self.assertIs(Camera._raw_render, first)


if __name__ == "__main__":
    unittest.main()
