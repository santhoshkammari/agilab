"""
Custom Gradio theme based on Scout.new design.
"""
import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
from typing import Iterable

class ScoutNewTheme(Base):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.blue,
        secondary_hue: colors.Color | str = colors.gray,
        neutral_hue: colors.Color | str = colors.gray,
        spacing_size: sizes.Size | str = sizes.spacing_md,
        radius_size: sizes.Size | str = sizes.radius_md,
        text_size: sizes.Size | str = sizes.text_md,
        font: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            "Inter",
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            body_background_fill="rgba(255, 255, 255, 0.6)",
            body_background_fill_dark="rgba(255, 255, 255, 0.6)",
            
            # Primary button using Scout blue
            button_primary_background_fill="#57BAFF",
            button_primary_background_fill_hover="#4AABF0",
            button_primary_background_fill_dark="#57BAFF",
            button_primary_background_fill_hover_dark="#4AABF0",
            button_primary_text_color="white",
            button_primary_text_color_dark="white",
            
            # Secondary elements using light blue
            button_secondary_background_fill="#EAF6FF",
            button_secondary_background_fill_hover="#D5EFFF",
            button_secondary_background_fill_dark="#EAF6FF",
            button_secondary_text_color="#57BAFF",
            
            # Neutral/background elements - all transparent/white
            block_background_fill="rgba(255, 255, 255, 0.4)",
            block_background_fill_dark="rgba(255, 255, 255, 0.4)",
            panel_background_fill="rgba(255, 255, 255, 0.3)",
            panel_background_fill_dark="rgba(255, 255, 255, 0.3)",
            
            # Input elements
            input_background_fill="rgba(255, 255, 255, 0.4)",
            input_background_fill_dark="rgba(255, 255, 255, 0.4)",
            input_background_fill_focus="rgba(90, 180, 255, 0.15)",
            input_background_fill_focus_dark="rgba(90, 180, 255, 0.15)",
            input_border_color="#F2F4F6",
            input_border_color_focus="#57BAFF",
            
            # Text colors
            body_text_color="#1F2937",
            body_text_color_dark="#1F2937",
            block_title_text_color="#1F2937",
            block_title_text_color_dark="#1F2937",
            
            # Other elements
            block_border_color="rgba(255, 255, 255, 0.4)",
            border_color_primary="rgba(87, 186, 255, 0.8)",

            # Premium rounded styling
            block_radius="16px",
            button_large_radius="12px",
            input_radius="12px",
        )

# Create the theme instance
theme = ScoutNewTheme()
