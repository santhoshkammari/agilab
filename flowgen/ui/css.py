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
            body_background_fill="#FFFFFF",
            body_background_fill_dark="#FFFFFF",
            
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
            block_background_fill="#FFFFFF",
            block_background_fill_dark="#FFFFFF",
            panel_background_fill="#FFFFFF",
            panel_background_fill_dark="#FFFFFF",
            
            # Input elements
            input_background_fill="#FFFFFF",
            input_background_fill_dark="#FFFFFF",
            input_background_fill_focus="#EAF6FF",
            input_background_fill_focus_dark="#EAF6FF",
            input_border_color="#F2F4F6",
            input_border_color_focus="#57BAFF",
            
            # Text colors
            body_text_color="#1F2937",
            body_text_color_dark="#1F2937",
            block_title_text_color="#1F2937",
            block_title_text_color_dark="#1F2937",
            
            # Other elements
            block_border_color="#F2F4F6",
            border_color_primary="#57BAFF",
            
            # Clean, minimal styling
            block_radius="8px",
            button_large_radius="6px",
            input_radius="6px",
        )

# Create the theme instance
theme = ScoutNewTheme()