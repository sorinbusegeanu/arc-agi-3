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

    def test_sprite_dimensions_accept_list_backed_render(self):
        from arcengine.sprites import Sprite

        class ListBackedSprite(Sprite):
            def render(self):
                return [[1, 2, 3], [4, 5, 6]]

        _install_arcengine_render_compatibility()
        sprite = object.__new__(ListBackedSprite)

        self.assertEqual(sprite.width, 3)
        self.assertEqual(sprite.height, 2)

    def test_compatibility_install_is_idempotent(self):
        from arcengine.camera import Camera
        from arcengine.sprites import Sprite

        _install_arcengine_render_compatibility()
        first_camera = Camera._raw_render
        first_width = Sprite.width.fget
        first_height = Sprite.height.fget
        _install_arcengine_render_compatibility()

        self.assertIs(Camera._raw_render, first_camera)
        self.assertIs(Sprite.width.fget, first_width)
        self.assertIs(Sprite.height.fget, first_height)


if __name__ == "__main__":
    unittest.main()
