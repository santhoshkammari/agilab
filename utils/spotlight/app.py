import sys
import os
import json
import logging
import time
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QWidget, QLineEdit, QVBoxLayout, QTextEdit, QDesktopWidget, QHBoxLayout, QComboBox, QPushButton
from PyQt5.QtCore import Qt, QPropertyAnimation, QEasingCurve, QRect, pyqtSignal, QObject
from PyQt5.QtGui import QPainter, QColor, QFont, QTextCursor
import threading

# Register QTextCursor for use in signals
from PyQt5.QtCore import QMetaType

from qwen import qwen  # Import the qwen function from qwen.py

QMetaType.type("QTextCursor")

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('spotlight_llm.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class StreamHandler(QObject):
    new_token_signal = pyqtSignal(str)

    def __init__(self, response_callback):
        super().__init__()
        self.response_callback = response_callback
        self.full_response = []  # Use list for efficient append
        self.token_count = 0
        self.first_token_time = None
        self.start_time = time.time()

    def on_llm_new_token(self, token: str, **kwargs):
        if self.token_count == 0:
            self.first_token_time = time.time()
            ttft = self.first_token_time - self.start_time
            logger.debug(f"Time to first token: {ttft:.3f}s")

        self.token_count += 1
        self.new_token_signal.emit(token)
        self.full_response.append(token)

    def on_llm_end(self, response, **kwargs):
        end_time = time.time()
        total_time = end_time - self.start_time
        if self.first_token_time:
            streaming_time = end_time - self.first_token_time
            logger.debug(f"Total tokens: {self.token_count}")
            logger.debug(f"Streaming time: {streaming_time:.3f}s")
            logger.debug(f"Tokens per second: {self.token_count / streaming_time:.2f}")
        logger.debug(f"Total response time: {total_time:.3f}s")
        self.response_callback(''.join(self.full_response))

DEFAULT_MODELS: list = ["Qwen"]

class SpotlightLLM(QWidget):
    def __init__(self, models:list = None):
        super().__init__()
        self.models = models or DEFAULT_MODELS
        self.current_model = self.models[0]
        self.dragging = False
        self.offset = None
        self.initUI()
        self.response_complete = threading.Event()
        self.fireoff_enabled = False


    def initUI(self):
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)

        # Create a horizontal layout for search bar, model selection, and fireoff button
        search_layout = QHBoxLayout()

        self.search_bar = QLineEdit(self)
        self.search_bar.setPlaceholderText("Drop a prompt ")
        self.search_bar.setStyleSheet("""
            QLineEdit {
                background-color: rgba(255, 255, 255, 0);
                border: none;
                padding: 5px;
                color: #FFFFFF;
                font-family: "Segoe UI", "SF Pro Display", sans-serif;
                font-size: 24px;
                font-weight: 350;
            }
        """)
        self.search_bar.returnPressed.connect(self.on_submit)
        search_layout.addWidget(self.search_bar, 5)

        self.model_selector = QComboBox(self)
        self.model_selector.addItems(self.models)
        self.model_selector.setStyleSheet(self.get_combobox_style())
        self.model_selector.currentTextChanged.connect(self.on_model_change)
        self.model_selector.hide()

        # Fireoff button
        self.fireoff_button = QPushButton("Fireoff", self)
        self.fireoff_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 10);
                border: 1px solid rgba(255, 255, 255, 20);
                border-radius: 6px;
                padding: 4px 12px;
                color: rgba(255, 255, 255, 180);
                font-size: 13px;
                font-weight: 500;
                font-family: "Segoe UI", sans-serif;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 30);
                color: white;
            }
            QPushButton:checked {
                background-color: rgba(0, 120, 212, 180); /* Accent Color */
                border: 1px solid rgba(0, 120, 212, 255);
                color: white;
            }
        """)
        self.fireoff_button.setCheckable(True)
        self.fireoff_button.setChecked(False)
        self.fireoff_button.clicked.connect(self.toggle_fireoff)
        self.fireoff_button.hide()

        layout.addLayout(search_layout)

        self.result_area = QTextEdit(self)
        self.result_area.setReadOnly(True)
        self.result_area.setStyleSheet("""
            QTextEdit {
                background-color: transparent;
                border: none;
                border-top: 1px solid rgba(255, 255, 255, 20);
                padding-top: 15px;
                margin-top: 5px;
                color: #E0E0E0;
                font-family: "Segoe UI", "SF Pro Text", sans-serif;
                font-size: 16px;
                line-height: 1.5;
            }
            QScrollBar:vertical {
                width: 6px;
                background: transparent;
            }
            QScrollBar::handle:vertical {
                background: rgba(255, 255, 255, 40);
                border-radius: 3px;
                min-height: 20px;
            }
        """)
        self.result_area.hide()
        layout.addWidget(self.result_area)

        self.setLayout(layout)

        screen = QDesktopWidget().screenNumber(QDesktopWidget().cursor().pos())
        screen_size = QDesktopWidget().screenGeometry(screen)
        window_width = 750
        window_height = 60
        x = (screen_size.width() - window_width) // 2
        y = screen_size.height() // 4
        self.setGeometry(x, y, window_width, window_height)

    def get_combobox_style(self):
        return """
            QComboBox {
                background-color: rgba(255, 255, 255, 10);
                border: 1px solid rgba(255, 255, 255, 20);
                border-radius: 6px;
                padding: 4px 10px;
                color: rgba(255, 255, 255, 180);
                font-family: "Segoe UI", sans-serif;
                font-size: 13px;
                font-weight: 500;
            }
            QComboBox:hover {
                background-color: rgba(255, 255, 255, 20);
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
             QComboBox::down-arrow {
                image: none;
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: rgba(30, 30, 30, 250);
                border: 1px solid rgba(255, 255, 255, 30);
                border-radius: 6px;
                selection-background-color: rgba(0, 120, 212, 180);
                color: white;
                outline: none;
                padding: 4px;
            }
        """


    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Solid "Acrylic" Background
        # Dark gray/black with only slight transparency
        painter.setBrush(QColor(24, 24, 24, 245))
        
        # Subtle white border
        painter.setPen(QColor(255, 255, 255, 25))
        
        rect = self.rect()
        painter.drawRoundedRect(rect.adjusted(1,1,-1,-1), 16, 16)

    def on_model_change(self, model):
        self.current_model = model

    def toggle_fireoff(self):
        self.fireoff_enabled = self.fireoff_button.isChecked()
        if self.fireoff_enabled:
            self.fireoff_button.setText("FIRED")
            self.fireoff_button.setStyleSheet("""
                QPushButton {
                    background-color: rgba(232, 17, 35, 200); /* Fluent Red */
                    border: 1px solid rgba(232, 17, 35, 255);
                    border-radius: 6px;
                    padding: 4px 12px;
                    color: white;
                    font-size: 13px;
                    font-weight: 600;
                    font-family: "Segoe UI", sans-serif;
                }
                QPushButton:hover {
                    background-color: rgba(232, 17, 35, 240);
                }
            """)
        else:
            self.fireoff_button.setText("Fireoff")
            self.fireoff_button.setStyleSheet("""
                QPushButton {
                    background-color: rgba(255, 255, 255, 10);
                    border: 1px solid rgba(255, 255, 255, 20);
                    border-radius: 6px;
                    padding: 4px 12px;
                    color: rgba(255, 255, 255, 180);
                    font-size: 13px;
                    font-weight: 500;
                    font-family: "Segoe UI", sans-serif;
                }
                QPushButton:hover {
                    background-color: rgba(255, 255, 255, 30);
                    color: white;
                }
                QPushButton:checked {
                    background-color: rgba(0, 120, 212, 180); /* Accent Color */
                    border: 1px solid rgba(0, 120, 212, 255);
                    color: white;
                }
            """)

    def on_submit(self):
        query = self.search_bar.text()
        if query:
            logger.debug(f"Query submitted: {query}")
            submit_time = time.time()

            # Check if @fire is in the query and remove it
            if "@fire" in query:
                query = query.replace("@fire", "").strip()
                # Enable fireoff mode
                if not self.fireoff_enabled:
                    self.fireoff_button.setChecked(True)
                    self.toggle_fireoff()

            ui_start = time.time()
            self.result_area.clear()
            self.result_area.show()
            self.animate_expand()
            ui_end = time.time()
            logger.debug(f"UI setup time: {ui_end - ui_start:.3f}s")

            self.response_complete.clear()
            logger.debug(f"Starting thread for query processing")
            threading.Thread(target=self.get_response, args=(query,), daemon=True).start()

    def update_result_area(self, token):
        update_start = time.time()
        # Batch updates for better performance
        cursor = self.result_area.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.beginEditBlock()  # Start atomic operation
        cursor.insertText(token)
        cursor.endEditBlock()  # End atomic operation
        self.result_area.setTextCursor(cursor)
        # Only scroll if needed - reduces repaints
        if self.result_area.verticalScrollBar().value() >= self.result_area.verticalScrollBar().maximum() - 10:
            self.result_area.ensureCursorVisible()
        update_end = time.time()
        logger.debug(f"UI update time for token: {update_end - update_start:.4f}s")

    def get_response(self, prompt):
        thread_start = time.time()
        logger.debug(f"get_response thread started")

        stream_handler = StreamHandler(self.on_response_complete)
        stream_handler.new_token_signal.connect(self.update_result_area)

        try:
            qwen_start = time.time()
            logger.debug(f"Calling qwen function")

            chunk_count = 0
            # Call the qwen function and process the streamed response
            for chunk in qwen(['-p', prompt]):
                chunk_count += 1
                if chunk_count == 1:
                    first_chunk_time = time.time()
                    logger.debug(f"Time to first chunk from qwen: {first_chunk_time - qwen_start:.3f}s")

                chunk_start = time.time()
                chunk_type = chunk.get('type')

                if chunk_type == 'assistant':
                    # Extract text content from assistant responses
                    content = chunk.get('message', {}).get('content', [])
                    for item in content:
                        if item.get('type') == 'text':
                            text = item.get('text', '')
                            stream_handler.on_llm_new_token(f'\n{text}')
                        elif item.get('type') == 'tool_use':
                            # Handle tool use if needed
                            tool_name = item.get('name', '')
                            tool_input = item.get('input', {})
                            tool_info = f"\n[Using tool: {tool_name} with inputs: {tool_input}]\n"
                            stream_handler.on_llm_new_token(tool_info)
                elif chunk_type == 'user':
                    # Handle tool results if needed
                    message = chunk.get('message', {})
                    content_items = message.get('content', [])
                    for item in content_items:
                        if item.get('type') == 'tool_result':
                            tool_result = item.get('content', '')
                            result_info = f"\n[Tool result: {tool_result}]\n"
                            stream_handler.on_llm_new_token(result_info)
                # Skip the 'result' chunk type to avoid duplicate final result display

                chunk_end = time.time()
                logger.debug(f"Chunk {chunk_count} ({chunk_type}) processing time: {chunk_end - chunk_start:.4f}s")

            qwen_end = time.time()
            logger.debug(f"Qwen call completed. Total chunks: {chunk_count}, Total qwen time: {qwen_end - qwen_start:.3f}s")

            stream_handler.on_llm_end(None)
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            logger.error(f"Error in get_response: {str(e)}", exc_info=True)
            QApplication.instance().postEvent(self.result_area, QTextCursor(self.result_area.document()))
            self.result_area.setPlainText(error_message)
        finally:
            self.response_complete.set()
            thread_end = time.time()
            logger.debug(f"get_response thread completed. Total thread time: {thread_end - thread_start:.3f}s")

    def on_response_complete(self, response):
        # If fireoff is enabled, close the app after response is complete
        if self.fireoff_enabled:
            QApplication.quit()

    def animate_expand(self):
        self.animation = QPropertyAnimation(self, b"geometry")
        self.animation.setDuration(300)
        self.animation.setStartValue(self.geometry())
        new_height = 380
        new_geometry = QRect(self.x(), self.y(), self.width(), new_height)
        self.animation.setEndValue(new_geometry)
        self.animation.setEasingCurve(QEasingCurve.OutCubic)
        self.animation.start()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.offset = event.pos()

    def mouseMoveEvent(self, event):
        if self.dragging and self.offset is not None:
            new_pos = self.mapToGlobal(event.pos() - self.offset)
            self.move(new_pos)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False
            self.offset = None

    def closeEvent(self, event):
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SpotlightLLM()
    ex.show()
    sys.exit(app.exec_())
