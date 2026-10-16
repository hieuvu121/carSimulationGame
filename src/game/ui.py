"""HUD + pause/countdown/game-over overlays (Lead/Triet backup, §10).

Owner: Uy (Car game).
Specification: docs/interfaces.md §4.4.

Provides: ``draw_hud()`` and ``draw_overlay()`` — called by
``GameController.render()`` after ``draw_world()``.

The HUD is how requirements GM.9 and LI.6 are met: the player can see the score,
which gesture was recognised and how confident the classifier was, which makes a
misclassification visible instead of mysterious during the demo.

Fonts are built fresh on every call and deliberately **not** cached; see ``_font()``
for why. ``pygame.font`` has to be initialised before a ``Font`` can be built, and
these functions must work when a test calls ``GameController.render()`` on a bare
surface without ever calling ``pygame.init()`` — so they initialise the font module
themselves rather than assuming a caller did.

Note on constants: as in ``render.py``, the palette and the type sizes are
module-private because ``src/config.py`` is owned by the Lead and holds no
presentation values.
"""
import math

import pygame

from src.config import (HUD_HEIGHT, LABEL_FIST, LABEL_NONE, LABEL_OPEN_PALM,
                        LABEL_V_SIGN, LOW_FPS_WARNING, NO_HAND, WINDOW_HEIGHT,
                        WINDOW_WIDTH)
from src.game.states import GameState

# ---- Palette (RGB) ----
_COLOR_HUD_BG: tuple[int, int, int] = (18, 19, 23)
_COLOR_TEXT: tuple[int, int, int] = (235, 237, 242)
_COLOR_TEXT_DIM: tuple[int, int, int] = (150, 155, 165)
_COLOR_ALERT: tuple[int, int, int] = (235, 84, 84)
_COLOR_WARNING: tuple[int, int, int] = (245, 158, 66)
_COLOR_OVERLAY_DIM: tuple[int, int, int, int] = (0, 0, 0, 170)

# ---- Type sizes (px) ----
_FONT_SIZE_HUD: int = 24
_FONT_SIZE_TITLE: int = 46
_FONT_SIZE_SUBTITLE: int = 24
_FONT_SIZE_COUNTDOWN: int = 180

_HUD_PADDING: int = 12

# Human-readable gesture names. The raw labels are snake_case machine tokens; a
# demo audience should not have to read "v_sign" off the screen.
_LABEL_TEXT: dict[str, str] = {
    LABEL_FIST: "Fist",
    LABEL_V_SIGN: "V sign",
    LABEL_OPEN_PALM: "Open palm",
    LABEL_NONE: "None",
    NO_HAND: "NO HAND",
}


def _font(size: int) -> pygame.font.Font:
    """Return a new default font at ``size`` pixels, initialising pygame.font.

    Args:
        size: Font height in pixels.

    Returns:
        A ``pygame.font.Font``. The built-in default font is used deliberately: a
        bundled .ttf would be one more file to ship, and the overlay text in
        docs/interfaces.md §4.4 is plain ASCII precisely so no special glyph
        coverage is needed.

    Do **not** cache the returned object in a module-level dict. A ``Font`` is a thin
    wrapper around a native ``TTF_Font*``; ``pygame.quit()`` calls ``TTF_Quit()`` and
    frees that pointer while the Python object stays alive. A cache therefore survives
    into the next ``pygame.init()`` and hands back a dangling pointer, which segfaults
    (``access violation``) rather than raising — and ``pygame.font.get_init()`` cannot
    detect it, because after a quit/init cycle it is ``True`` again. This is not
    hypothetical: ``src/app/main.py`` quits pygame on exit, which
    ``tests/test_main.py`` asserts, so any test that runs ``main()`` twice reaches
    exactly that state.

    Rebuilding costs ~0.35 ms and ``PLAYING`` needs one font per frame — about 2% of a
    60 FPS budget — so the cache bought little and cost a crash.
    """
    if not pygame.font.get_init():
        pygame.font.init()
    return pygame.font.Font(None, size)


def draw_hud(surface: pygame.Surface, score: int, label: str, confidence: float,
             hand_present: bool, fps: float | None = None) -> None:
    """Draw the top status bar: score, recognised gesture, and optionally the FPS.

    Layout inside the ``HUD_HEIGHT`` (48 px) bar: score on the left, gesture in the
    middle, FPS on the right.

    Args:
        surface: Target surface.
        score: Whole points to display.
        label: Gesture label — one of ``LABELS`` or ``NO_HAND``. An unrecognised
            string is displayed verbatim rather than rejected: by the time a value
            reaches the drawing code the frame is already being painted, and
            ``GameController.set_gesture_info()`` has already validated it.
        confidence: Classifier confidence in [0, 1], shown in brackets.
        hand_present: False draws ``NO HAND`` in red, which is the player's cue that
            the game auto-paused because the camera lost their hand (LI.4).
        fps: Frames per second to display, or None to omit. Shown in orange below
            ``LOW_FPS_WARNING`` so a dim-light frame-rate drop is visible during the
            demo instead of being mistaken for a bug in the gesture pipeline.

    Returns:
        None.
    """
    pygame.draw.rect(surface, _COLOR_HUD_BG,
                     pygame.Rect(0, 0, WINDOW_WIDTH, HUD_HEIGHT))

    font = _font(_FONT_SIZE_HUD)
    baseline_y = HUD_HEIGHT // 2

    score_surface = font.render(f"Score: {score}", True, _COLOR_TEXT)
    surface.blit(score_surface,
                 score_surface.get_rect(midleft=(_HUD_PADDING, baseline_y)))

    if hand_present:
        text = f"{_LABEL_TEXT.get(label, label)} ({confidence:.2f})"
        colour = _COLOR_TEXT if label != LABEL_NONE else _COLOR_TEXT_DIM
    else:
        # No hand: the confidence number would be meaningless, so it is left out.
        text = _LABEL_TEXT[NO_HAND]
        colour = _COLOR_ALERT
    gesture_surface = font.render(text, True, colour)
    surface.blit(gesture_surface,
                 gesture_surface.get_rect(center=(WINDOW_WIDTH // 2, baseline_y)))

    if fps is not None:
        colour = _COLOR_WARNING if fps < LOW_FPS_WARNING else _COLOR_TEXT_DIM
        fps_surface = font.render(f"FPS: {fps:.0f}", True, colour)
        surface.blit(
            fps_surface,
            fps_surface.get_rect(midright=(WINDOW_WIDTH - _HUD_PADDING, baseline_y)),
        )


def draw_overlay(surface: pygame.Surface, state: GameState, countdown_left: float,
                 score: int) -> None:
    """Draw the overlay belonging to the current state, if any.

    ``PLAYING`` draws nothing — the player needs an unobstructed view of the road.

    Args:
        surface: Target surface.
        state: Current game state.
        countdown_left: Seconds left in ``COUNTDOWN``; ignored in other states.
        score: Final score, shown on the ``GAME_OVER`` screen.

    Returns:
        None.
    """
    if state is GameState.PAUSED:
        _draw_dim(surface)
        _draw_centered(surface, "PAUSED", _FONT_SIZE_TITLE, _COLOR_TEXT, offset=-30)
        _draw_centered(surface, "show FIST or V SIGN to resume",
                       _FONT_SIZE_SUBTITLE, _COLOR_TEXT_DIM, offset=20)

    elif state is GameState.COUNTDOWN:
        # Counting up from the fraction: 2.0 s left shows "2", 0.4 s left shows "1".
        # max(1, ...) stops a "0" flashing on the final frame before PLAYING.
        seconds = max(1, math.ceil(countdown_left))
        _draw_centered(surface, str(seconds), _FONT_SIZE_COUNTDOWN, _COLOR_TEXT)

    elif state is GameState.GAME_OVER:
        _draw_dim(surface)
        _draw_centered(surface, "GAME OVER", _FONT_SIZE_TITLE, _COLOR_ALERT, offset=-45)
        _draw_centered(surface, f"score {score}", _FONT_SIZE_SUBTITLE,
                       _COLOR_TEXT, offset=0)
        _draw_centered(surface, "open palm or R to restart", _FONT_SIZE_SUBTITLE,
                       _COLOR_TEXT_DIM, offset=36)


def _draw_dim(surface: pygame.Surface) -> None:
    """Darken the playfield so overlay text stays readable over the road."""
    dim = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
    dim.fill(_COLOR_OVERLAY_DIM)
    surface.blit(dim, (0, 0))


def _draw_centered(surface: pygame.Surface, text: str, size: int,
                   colour: tuple[int, int, int], offset: int = 0) -> None:
    """Draw ``text`` centred in the window, ``offset`` pixels below the middle."""
    rendered = _font(size).render(text, True, colour)
    surface.blit(
        rendered,
        rendered.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + offset)),
    )
