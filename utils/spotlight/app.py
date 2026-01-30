import sys
import os
import json
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QWidget, QLineEdit, QVBoxLayout, QTextEdit, QDesktopWidget, QHBoxLayout, QComboBox, QPushButton
from PyQt5.QtCore import Qt, QPropertyAnimation, QEasingCurve, QRect, pyqtSignal, QObject
from PyQt5.QtGui import QPainter, QColor, QFont, QTextCursor
import threading

# Register QTextCursor for use in signals
from PyQt5.QtCore import QMetaType

from qwen import qwen  # Import the qwen function from qwen.py

QMetaType.type("QTextCursor")


class StreamHandler(QObject):
    new_token_signal = pyqtSignal(str)
    
    def __init__(self, response_callback):
        super().__init__()
        self.response_callback = response_callback
        self.full_response = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.new_token_signal.emit(token)
        self.full_response += token

    def on_llm_end(self, response, **kwargs):
        self.response_callback(self.full_response)

DEFAULT_MODELS: list = ["Qwen"]

class SpotlightLLM(QWidget):
    def __init__(self, models:list = None):
        super().__init__()
        self.models = models or DEFAULT_MODELS
        self.current_model = self.models[0]
        self.dragging = False
        self.offset = None
        self.initUI()
        self.session_interactions = []
        self.response_complete = threading.Event()
        self.fireoff_enabled = False
        self.data_dir = os.path.join(os.path.expanduser("~"), ".spotlight_data")
        os.makedirs(self.data_dir, exist_ok=True)


    def initUI(self):
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)

        # Create a horizontal layout for search bar, model selection, and fireoff button
        search_layout = QHBoxLayout()

        self.search_bar = QLineEdit(self)
        self.search_bar.setStyleSheet("""
            QLineEdit {
                background-color: rgba(60, 60, 60, 200);
                border: none;
                border-radius: 20px;
                padding: 10px;
                color: white;
                font-size: 20px;
            }
        """)
        self.search_bar.returnPressed.connect(self.on_submit)
        search_layout.addWidget(self.search_bar, 5)

        self.model_selector = QComboBox(self)
        self.model_selector.addItems(self.models)
        self.model_selector.setStyleSheet(self.get_combobox_style())
        self.model_selector.currentTextChanged.connect(self.on_model_change)
        search_layout.addWidget(self.model_selector, 2)

        # Fireoff button
        self.fireoff_button = QPushButton("Fireoff", self)
        self.fireoff_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(76, 175, 80, 150);  /* Green with transparency */
                border: none;
                border-radius: 15px;
                padding: 8px;
                color: white;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(69, 160, 73, 180);
            }
            QPushButton:pressed {
                background-color: rgba(61, 139, 64, 200);
            }
            QPushButton:checked {
                background-color: rgba(139, 195, 74, 180);  /* Lighter green when active */
            }
        """)
        self.fireoff_button.setCheckable(True)
        self.fireoff_button.setChecked(False)
        self.fireoff_button.clicked.connect(self.toggle_fireoff)
        search_layout.addWidget(self.fireoff_button, 1)

        layout.addLayout(search_layout)

        self.result_area = QTextEdit(self)
        self.result_area.setReadOnly(True)
        self.result_area.setStyleSheet("""
            QTextEdit {
                background-color: rgba(60, 60, 60, 200);
                border: none;
                border-radius: 10px;
                padding: 10px;
                color: white;
                font-size: 16px;
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
                background-color: rgba(60, 60, 60, 200);
                border: none;
                border-radius: 20px;
                padding: 10px;
                color: white;
                font-size: 16px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: url(down_arrow.png);
                width: 12px;
                height: 12px;
            }
            QComboBox QAbstractItemView {
                background-color: rgba(60, 60, 60, 200);
                border: none;
                selection-background-color: rgba(80, 80, 80, 200);
                color: white;
            }
        """


    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor(30, 30, 30, 200))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(self.rect(), 20, 20)

    def on_model_change(self, model):
        self.current_model = model

    def toggle_fireoff(self):
        self.fireoff_enabled = self.fireoff_button.isChecked()
        if self.fireoff_enabled:
            self.fireoff_button.setText("FIRED")
            self.fireoff_button.setStyleSheet("""
                QPushButton {
                    background-color: rgba(139, 195, 74, 180);  /* Lighter green with transparency when active */
                    border: none;
                    border-radius: 15px;
                    padding: 8px;
                    color: white;
                    font-size: 14px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: rgba(124, 179, 66, 200);
                }
                QPushButton:pressed {
                    background-color: rgba(104, 159, 56, 220);
                }
            """)
        else:
            self.fireoff_button.setText("Fireoff")
            self.fireoff_button.setStyleSheet("""
                QPushButton {
                    background-color: rgba(76, 175, 80, 150);  /* Green with transparency */
                    border: none;
                    border-radius: 15px;
                    padding: 8px;
                    color: white;
                    font-size: 14px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: rgba(69, 160, 73, 180);
                }
                QPushButton:pressed {
                    background-color: rgba(61, 139, 64, 200);
                }
                QPushButton:checked {
                    background-color: rgba(139, 195, 74, 180);  /* Lighter green when active */
                }
            """)

    def on_submit(self):
        query = self.search_bar.text()
        if query:
            # Check if @fire is in the query and remove it
            if "@fire" in query:
                query = query.replace("@fire", "").strip()
                # Enable fireoff mode
                if not self.fireoff_enabled:
                    self.fireoff_button.setChecked(True)
                    self.toggle_fireoff()

            self.result_area.clear()
            self.result_area.show()
            self.animate_expand()
            self.current_query = query
            self.current_timestamp = datetime.now().isoformat()
            self.response_complete.clear()
            threading.Thread(target=self.get_response, args=(query,), daemon=True).start()

    def update_result_area(self, token):
        cursor = self.result_area.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(token)
        self.result_area.setTextCursor(cursor)
        self.result_area.ensureCursorVisible()

    def get_response(self, prompt):
        stream_handler = StreamHandler(self.save_interaction)
        stream_handler.new_token_signal.connect(self.update_result_area)

        try:
            # Call the qwen function and process the streamed response
            for chunk in qwen(['-p', prompt]):
                chunk_type = chunk.get('type')

                if chunk_type == 'assistant':
                    # Extract text content from assistant responses
                    content = chunk.get('message', {}).get('content', [])
                    for item in content:
                        if item.get('type') == 'text':
                            text = item.get('text', '')
                            stream_handler.on_llm_new_token(text)
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

            stream_handler.on_llm_end(None)
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            QApplication.instance().postEvent(self.result_area, QTextCursor(self.result_area.document()))
            self.result_area.setPlainText(error_message)
            self.save_interaction(error_message)
        finally:
            self.response_complete.set()

    def save_interaction(self, response):
        interaction_data = {
            "timestamp": self.current_timestamp,
            "model": self.current_model,
            "query": self.current_query,
            "response": response
        }
        self.session_interactions.append(interaction_data)

        # If fireoff is enabled, close the app after saving the interaction
        if self.fireoff_enabled:
            self.save_session()
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
            self.save_session()
            self.close()

    def save_session(self):
        if self.session_interactions:
            session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"session_{session_timestamp}.json"
            filepath = os.path.join(self.data_dir, filename)

            with open(filepath, 'w') as f:
                json.dump(self.session_interactions, f, indent=2)

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
        self.save_session()
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SpotlightLLM()
    ex.show()
    sys.exit(app.exec_())
